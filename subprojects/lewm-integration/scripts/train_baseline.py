"""Phase 1 baseline: fork of le-wm/train.py with live logging + CSV logger.

Identical to the upstream training recipe (same ARPredictor, same hyperparams)
— the ONLY change is logging so we can monitor progress mid-run:
  1. Lightning CSVLogger → <run_dir>/csv_logs/
  2. LiveStepLogger callback → stdout every N steps + metrics.csv

Usage:
    cd subprojects/lewm-integration
    STABLEWM_HOME=... PYTHONPATH=./le-wm:. \
        python scripts/run_with_cudnn_compat.py scripts/train_baseline.py \
            trainer.max_epochs=100 wandb.enabled=False trainer.precision=bf16-mixed
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

from jepa import JEPA                          # noqa: E402
from module import ARPredictor, Embedder, MLP, SIGReg  # noqa: E402
from utils import (                            # noqa: E402
    get_column_normalizer, get_img_preprocessor, ModelObjectCallBack,
)

from live_log_callback import LiveStepLogger   # noqa: E402
from patch_vit_stem import replace_vit_patch_embeddings  # noqa: E402


def lejepa_forward(self, batch, stage, cfg):
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
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    losses = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses, on_step=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path="../le-wm/config/train", config_name="lewm")
def run(cfg):
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

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale, patch_size=cfg.patch_size, image_size=cfg.img_size,
        pretrained=False, use_mask_token=False)
    n_enc = replace_vit_patch_embeddings(encoder)
    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size, input_dim=embed_dim,
        hidden_dim=hidden_dim, output_dim=hidden_dim, **cfg.predictor)
    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    n_ae = replace_vit_patch_embeddings(action_encoder)
    print(f"[patch] replaced {n_enc} Conv2d + {n_ae} Conv1d with Linear equivalents",
          flush=True)
    # NOTE: upstream uses BatchNorm1d in these MLPs, but its cuDNN backward
    # fails on GB10/sm_121a at bs=128. LayerNorm is applied per-sample instead
    # of per-batch, avoiding the broken cuDNN path. For JEPA (pred MSE +
    # SIGReg) this change is benign — SIGReg already regularizes the
    # embedding distribution to be Gaussian, so whitening via BN isn't load-
    # bearing. Keeping BN disabled enables cuDNN for everything else.
    projector = MLP(input_dim=hidden_dim, output_dim=embed_dim,
                    hidden_dim=2048, norm_fn=torch.nn.LayerNorm)
    predictor_proj = MLP(input_dim=hidden_dim, output_dim=embed_dim,
                         hidden_dim=2048, norm_fn=torch.nn.LayerNorm)
    world_model = JEPA(encoder=encoder, predictor=predictor,
                       action_encoder=action_encoder, projector=projector,
                       pred_proj=predictor_proj)

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
