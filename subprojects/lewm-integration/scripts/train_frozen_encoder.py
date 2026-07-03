"""Phase 2 (frozen-encoder pivot).

Pragmatic: skip reproducing the LeWM baseline from scratch. Use the released
pretrained checkpoint (`quentinll/lewm-pusht` on HuggingFace) as the encoder,
freeze it, and train ONLY the predictor on PushT. This directly tests the
spec's hypothesis — does the curved-geometry ODE predictor beat the flat
causal-attention predictor, given identical encoder features?

Why this is faster:
  - No ViT backward → avoids GB10/sm_121a cuDNN Conv2d bf16 backward failure
    entirely (we only need ViT FORWARD in eval mode).
  - Predictor is small (~5M params); gradient graph is small.
  - Same encoder features for both arms → clean comparison.

Why this is legitimate:
  - Spec revision (LEWM_SPEC_REVISION.md) is explicit that the question is
    "curved geometry over temporal history vs flat attention over temporal
    history." Encoder is not under test.
  - LeWM's own `README.md` ships these pretrained checkpoints as the
    authoritative baseline for downstream comparison.

Usage:
    cd subprojects/lewm-integration
    STABLEWM_HOME=... python scripts/run_with_cudnn_compat.py \
        scripts/train_frozen_encoder.py \
        trainer.max_epochs=10 wandb.enabled=False \
        predictor_kind=ar        # or: predictor_kind=liquid
"""

from __future__ import annotations

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

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "le-wm"))
sys.path.insert(0, str(_ROOT / "scripts"))

from jepa import JEPA                                   # noqa: E402
from module import ARPredictor, Embedder, MLP, SIGReg   # noqa: E402
from utils import (                                     # noqa: E402
    get_column_normalizer, get_img_preprocessor, ModelObjectCallBack,
)

from live_log_callback import LiveStepLogger            # noqa: E402
from patch_vit_stem import replace_vit_patch_embeddings  # noqa: E402

from liquid_arc.config import LiquidARCConfig           # noqa: E402
from liquid_arc_lewm import LiquidARCPredictor          # noqa: E402
from liquid_arc.sustained_criticality import (          # noqa: E402
    compute_criticality_loss, compute_tau_quality_loss,
)

import torch.nn.functional as F                         # noqa: E402


def _liquid_geo_terms(predictor, h0: torch.Tensor) -> tuple:
    """Compute g and tau from the Liquid predictor's dynamics on h0.

    h0: [B, T, d_model] — the projected embedding fed into the ODE (before
    integration). We run the MetricNet + TauNet once on h0 to get the g/tau
    we regularize. This is the "initial metric" path from the original spec.
    """
    dyn = predictor.dynamics
    h_n = dyn.norm_geo(h0)
    B, N, d = h_n.shape
    # Context for MetricNet: we pass a zero context since we only need the
    # structure. Using actual action context would require replaying the
    # forward. The initial state's geometry is sufficient signal.
    ctx = torch.zeros(B, d, device=h0.device, dtype=h0.dtype)
    ctx_exp = ctx.unsqueeze(1).expand(B, N, d)
    cat_input = torch.cat([h_n, ctx_exp], dim=-1)       # [B, N, 2d]
    met_hidden = F.gelu(dyn.metric_net_linear1(cat_input))
    g = F.softplus(dyn.metric_net_linear2_diag(met_hidden))  # [B, N, d]
    # Tau (only when scalar tau is enabled)
    if not dyn.channel_gate_enabled:
        tau_hidden = F.gelu(dyn.tau_net_linear1(h_n))
        tau_raw = dyn.tau_net_linear2(tau_hidden)
        tau = dyn.tau_min + F.softplus(tau_raw)
    else:
        tau = torch.ones(B, N, 1, device=h0.device, dtype=h0.dtype)
    return g, tau


def lejepa_forward(self, batch, stage, cfg):
    """Same as upstream, but encoder is in eval/no_grad mode."""
    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    # Encoder frozen — run in no_grad to save memory + time
    with torch.no_grad():
        output = self.model.encode(batch)
        output["emb"] = output["emb"].detach()

    emb = output["emb"]
    act_emb = output["act_emb"]
    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:]

    pred_emb = self.model.predict(ctx_emb, ctx_act)

    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    loss = output["pred_loss"] + lambd * output["sigreg_loss"]

    # LiquidARC sustained-criticality scaffolding (spec Phase 3)
    predictor = self.model.predictor
    crit_cfg = cfg.loss.get("criticality", None)
    tau_cfg = cfg.loss.get("tau_quality", None)
    crit_on = crit_cfg is not None and crit_cfg.enabled
    tau_on = tau_cfg is not None and tau_cfg.enabled
    if isinstance(predictor, LiquidARCPredictor) and (crit_on or tau_on):
        # Compute g, tau on the first layer of the predictor's ODE input
        h0 = predictor.proj_in(ctx_emb)   # [B, T, d_model]
        g, tau = _liquid_geo_terms(predictor, h0)

        if crit_cfg is not None and crit_cfg.enabled:
            c, diags = compute_criticality_loss(
                h0, g, tau, predictor.dynamics.t_diffusion,
                target_ratio=crit_cfg.target_ratio,
                d_sq_target=crit_cfg.D_sq_target,
            )
            output["crit_loss"] = c
            output["D_sq_median"] = torch.tensor(diags["D_sq_median"])
            output["crit_ratio"] = torch.tensor(diags["ratio"])
            loss = loss + crit_cfg["lambda"] * c

        if tau_cfg is not None and tau_cfg.enabled:
            tq = compute_tau_quality_loss(
                tau, mean_target=tau_cfg.mean_target,
                log_spread_target=tau_cfg.log_spread_target,
            )
            output["tau_quality_loss"] = tq
            loss = loss + tau_cfg["lambda"] * tq

    output["loss"] = loss
    losses = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses, on_step=True, sync_dist=True)
    return output


def _build_ode_config(cfg) -> LiquidARCConfig:
    ode_cfg = dict(cfg.get("ode", {}))
    fields = set(LiquidARCConfig.__dataclass_fields__.keys())
    return LiquidARCConfig(**{k: v for k, v in ode_cfg.items() if k in fields})


def _load_pretrained_into(world_model: JEPA, ckpt_path: Path) -> None:
    """Load encoder, action_encoder, and projector from `weights.pt`.
    Predictor & pred_proj are deliberately skipped (we're training those).
    """
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    sd = {}
    for k, v in state.items():
        # strip any lightning module prefix
        if k.startswith("model."):
            k = k[len("model."):]
        sd[k] = v

    # Load only encoder + action_encoder + projector
    own = world_model.state_dict()
    loaded = []
    skipped = []
    for k, v in sd.items():
        if not (k.startswith("encoder.") or k.startswith("action_encoder.")
                or k.startswith("projector.")):
            continue
        if k not in own:
            skipped.append(f"{k} (not in model)")
            continue
        if own[k].shape != v.shape:
            skipped.append(f"{k} (shape {tuple(v.shape)} != {tuple(own[k].shape)})")
            continue
        own[k] = v
        loaded.append(k)
    world_model.load_state_dict(own, strict=False)
    print(f"[ckpt] loaded {len(loaded)} tensors from {ckpt_path.name}", flush=True)
    if skipped:
        print(f"[ckpt] skipped {len(skipped)} (showing 5): {skipped[:5]}", flush=True)


def _freeze(module: torch.nn.Module):
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)


@hydra.main(version_base=None, config_path="../configs",
            config_name="frozen_encoder")
def run(cfg):
    # ---- data ----
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [get_img_preprocessor(source='pixels', target='pixels',
                                       img_size=cfg.img_size)]
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
        pretrained=False, use_mask_token=False)
    n_enc = replace_vit_patch_embeddings(encoder)

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    # Predictor selection
    if cfg.predictor_kind == "ar":
        predictor = ARPredictor(
            num_frames=cfg.wm.history_size, input_dim=embed_dim,
            hidden_dim=cfg.ar.get("hidden_dim", embed_dim),
            output_dim=cfg.ar.get("output_dim", embed_dim),
            depth=cfg.ar.depth, heads=cfg.ar.heads,
            mlp_dim=cfg.ar.mlp_dim, dim_head=cfg.ar.dim_head,
            dropout=cfg.ar.dropout, emb_dropout=cfg.ar.emb_dropout)
    elif cfg.predictor_kind == "liquid":
        ode_cfg = _build_ode_config(cfg)
        predictor = LiquidARCPredictor(
            input_dim=embed_dim, action_emb_dim=embed_dim,
            ode_config=ode_cfg, output_dim=embed_dim,
            dropout=cfg.ode.get("dropout", 0.1))
    else:
        raise ValueError(f"predictor_kind={cfg.predictor_kind} must be ar|liquid")

    n_pred_params = sum(p.numel() for p in predictor.parameters())
    print(f"[predictor] kind={cfg.predictor_kind} {n_pred_params/1e6:.2f}M params",
          flush=True)

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    n_ae = replace_vit_patch_embeddings(action_encoder)
    print(f"[patch] replaced {n_enc} Conv2d + {n_ae} Conv1d with Linear equivalents",
          flush=True)
    projector = MLP(input_dim=hidden_dim, output_dim=embed_dim,
                    hidden_dim=2048, norm_fn=torch.nn.LayerNorm)
    predictor_proj = MLP(input_dim=embed_dim, output_dim=embed_dim,
                         hidden_dim=2048, norm_fn=torch.nn.LayerNorm)
    world_model = JEPA(encoder=encoder, predictor=predictor,
                       action_encoder=action_encoder, projector=projector,
                       pred_proj=predictor_proj)

    # Load pretrained encoder + action_encoder + projector; freeze them.
    ckpt_path = Path(cfg.pretrained_ckpt)
    if ckpt_path.is_file():
        _load_pretrained_into(world_model, ckpt_path)
    else:
        print(f"[ckpt] WARNING: {ckpt_path} not found; using random encoder",
              flush=True)
    _freeze(world_model.encoder)
    _freeze(world_model.action_encoder)
    _freeze(world_model.projector)

    n_train = sum(p.numel() for p in world_model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in world_model.parameters())
    print(f"[model] trainable {n_train/1e6:.2f}M / total {n_total/1e6:.2f}M",
          flush=True)

    # ---- optim — only over trainable params ----
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
        model=world_model, sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg), optim=optimizers,
    )

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    loggers: list = [CSVLogger(save_dir=str(run_dir), name="csv_logs")]
    if cfg.wandb.enabled:
        loggers.append(WandbLogger(**cfg.wandb.config))

    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    callbacks = [
        ModelObjectCallBack(dirpath=run_dir, filename=cfg.output_model_name,
                            epoch_interval=1),
        LiveStepLogger(every_n_steps=20,
                       csv_path=str(run_dir / "metrics.csv")),
    ]
    trainer = pl.Trainer(
        **cfg.trainer, callbacks=callbacks, num_sanity_val_steps=1,
        logger=loggers, enable_checkpointing=True,
    )
    manager = spt.Manager(
        trainer=trainer, module=world_model, data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )
    manager()


if __name__ == "__main__":
    run()
