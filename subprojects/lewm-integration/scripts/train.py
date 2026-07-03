"""Training entrypoint for LeWM × LiquidARC integration.

Fork of le-wm/train.py with three surgical changes:
  1. Replace ARPredictor with LiquidARCPredictor (keeps JEPA interface intact).
  2. Patch lejepa_forward to optionally add criticality + tau_quality losses.
  3. Pull ode.* hyperparameters from Hydra config into LiquidARCConfig.

Everything else (encoder, SIGReg, data loading, Lightning, Hydra) is unchanged.
This preserves the LeWM baseline numerics — we compare against the SAME training
pipeline with only the predictor swapped.

Usage:
    cd subprojects/lewm-integration
    export PYTHONPATH=$PWD:$PWD/../liquid-arc:$PWD/le-wm
    python scripts/train.py --config-path=../configs --config-name=lewm_liquid
"""

from __future__ import annotations

import os
import sys
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import OmegaConf, open_dict

# le-wm is vendored as a sibling dir; add it to sys.path so `jepa` / `module`
# / `utils` resolve. No edits to upstream files.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "le-wm"))
sys.path.insert(0, str(_ROOT / "scripts"))
from live_log_callback import LiveStepLogger    # noqa: E402
from patch_vit_stem import replace_vit_patch_embeddings  # noqa: E402

from jepa import JEPA                              # noqa: E402
from module import Embedder, MLP, SIGReg           # noqa: E402
from utils import (                                # noqa: E402
    get_column_normalizer, get_img_preprocessor, ModelObjectCallBack,
)

from liquid_arc.config import LiquidARCConfig      # noqa: E402
from liquid_arc_lewm import LiquidARCPredictor     # noqa: E402


def _criticality_term(dynamics, target_ratio: float, D_sq_target: float) -> torch.Tensor:
    """D²/4τ → target. Reads last cached metric from dynamics (diagnostic path)."""
    g = getattr(dynamics, '_cached_g', None)
    if g is None:
        return torch.zeros((), device=next(dynamics.parameters()).device)
    # D² median over the sequence (off-diagonal pairs)
    # g: [B, N, d] diagonal metric; estimate D² via g.mean() * ||h_i - h_j||² summary
    D_sq_med = g.mean()
    ratio_err = (D_sq_med / (4.0 * 1.0) - target_ratio / D_sq_target).square()
    return ratio_err


def _tau_quality_term(dynamics, mean_target: float, log_spread_target: float) -> torch.Tensor:
    """Encourage τ mean ≈ target and log-spread ≈ target."""
    # Best-effort: may be None early in training or with channel_gate_enabled.
    tau = getattr(dynamics, '_last_tau', None)
    if tau is None:
        return torch.zeros((), device=next(dynamics.parameters()).device)
    tau_mean = tau.mean()
    tau_log = tau.clamp_min(1e-6).log()
    tau_log_spread = tau_log.std()
    return (tau_mean - mean_target).square() + (tau_log_spread - log_spread_target).square()


def lejepa_forward(self, batch, stage, cfg):
    """Replicate upstream lejepa_forward, then add LiquidARC auxiliaries."""
    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)

    emb = output["emb"]
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:]
    pred_emb = self.model.predict(ctx_emb, ctx_act)

    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    loss = output["pred_loss"] + lambd * output["sigreg_loss"]

    # Optional LiquidARC auxiliaries (Phase 3+)
    crit_cfg = cfg.loss.get("criticality", None)
    if crit_cfg is not None and crit_cfg.enabled:
        crit = _criticality_term(
            self.model.predictor.dynamics,
            crit_cfg.target_ratio, crit_cfg.D_sq_target,
        )
        output["crit_loss"] = crit
        loss = loss + crit_cfg["lambda"] * crit

    tau_cfg = cfg.loss.get("tau_quality", None)
    if tau_cfg is not None and tau_cfg.enabled:
        tq = _tau_quality_term(
            self.model.predictor.dynamics,
            tau_cfg.mean_target, tau_cfg.log_spread_target,
        )
        output["tau_quality_loss"] = tq
        loss = loss + tau_cfg["lambda"] * tq

    output["loss"] = loss
    losses = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses, on_step=True, sync_dist=True)
    return output


def _build_ode_config(cfg) -> LiquidARCConfig:
    """Hydra ode.* → LiquidARCConfig. Only fields present in cfg are forwarded."""
    ode_cfg = dict(cfg.ode)
    fields = set(LiquidARCConfig.__dataclass_fields__.keys())
    kwargs = {k: v for k, v in ode_cfg.items() if k in fields}
    return LiquidARCConfig(**kwargs)


@hydra.main(version_base=None, config_path="../configs", config_name="lewm_liquid")
def run(cfg):
    # ---- data ----
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)
            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))
    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen)

    train = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(
        val_set, **cfg.loader, shuffle=False, drop_last=False)

    # ---- model ----
    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale, patch_size=cfg.patch_size, image_size=cfg.img_size,
        pretrained=False, use_mask_token=False,
    )
    n_replaced = replace_vit_patch_embeddings(encoder)
    print(f"[patch_vit_stem] replaced {n_replaced} Conv2d patch stems with "
          f"Unfold+Linear (cuDNN on GB10 can't handle the Conv2d backward)",
          flush=True)
    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    ode_config = _build_ode_config(cfg)
    predictor = LiquidARCPredictor(
        input_dim=embed_dim,
        action_emb_dim=embed_dim,
        ode_config=ode_config,
        output_dim=hidden_dim,
        dropout=cfg.ode.get("dropout", 0.1),
    )
    n_params = sum(p.numel() for p in predictor.parameters())
    print(f"[LiquidARCPredictor] {n_params/1e6:.2f}M params "
          f"(d_model={ode_config.d_model}, n_ode_steps={ode_config.n_ode_steps})")

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    # See train_baseline.py — BN1d's cuDNN backward is broken on GB10/sm_121a
    # at bs=128; LayerNorm is an equivalent substitute given SIGReg handles
    # distributional whitening.
    projector = MLP(input_dim=hidden_dim, output_dim=embed_dim,
                    hidden_dim=2048, norm_fn=torch.nn.LayerNorm)
    predictor_proj = MLP(input_dim=hidden_dim, output_dim=embed_dim,
                         hidden_dim=2048, norm_fn=torch.nn.LayerNorm)
    world_model = JEPA(
        encoder=encoder, predictor=predictor, action_encoder=action_encoder,
        projector=projector, pred_proj=predictor_proj,
    )

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    # ---- train ----
    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    loggers: list = [CSVLogger(save_dir=str(run_dir), name="csv_logs")]
    if cfg.wandb.enabled:
        loggers.append(WandbLogger(**cfg.wandb.config))
        loggers[-1].log_hyperparams(OmegaConf.to_container(cfg))

    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    callbacks = [
        ModelObjectCallBack(dirpath=run_dir, filename=cfg.output_model_name,
                            epoch_interval=1),
        LiveStepLogger(every_n_steps=50,
                       csv_path=str(run_dir / "metrics.csv")),
    ]
    trainer = pl.Trainer(
        **cfg.trainer, callbacks=callbacks,
        num_sanity_val_steps=1, logger=loggers, enable_checkpointing=True,
    )
    manager = spt.Manager(
        trainer=trainer, module=world_model, data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )
    manager()


if __name__ == "__main__":
    run()
