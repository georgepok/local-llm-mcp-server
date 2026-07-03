"""Phase 4 rollout evaluation: horizon-k prediction MSE for both arms.

Spec criterion (LEWM_INTEGRATION_SPEC.md):
    "The ODE should accumulate less error over long horizons (16 iterative
     steps vs 1-shot MLP) — this is the key differentiator for MPC planning"

Protocol:
  1. Load frozen pretrained encoder + action_encoder + projector from HF ckpt.
  2. For each arm (AR, Liquid), load the predictor trained in Phase 2.
  3. On held-out val split, encode T+H frames → get ground-truth emb sequence.
  4. Autoregress predictor for H steps starting from T=history_size initial
     frames, using the ground-truth action sequence for steps t=H..H+H.
  5. Report per-horizon MSE between predicted emb and ground-truth emb.

Usage:
    STABLEWM_HOME=... PYTHONPATH=./le-wm:../liquid-arc:. python scripts/run_with_cudnn_compat.py \
        scripts/eval_rollout.py \
        ar_ckpt=.../ar_frozen_weights.ckpt \
        liquid_ckpt=.../liquid_frozen_weights.ckpt \
        horizons=[1,5,10,20]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "le-wm"))
sys.path.insert(0, str(_ROOT / "scripts"))

import stable_pretraining as spt
import stable_worldmodel as swm

from jepa import JEPA                                   # noqa: E402
from module import ARPredictor, Embedder, MLP           # noqa: E402
from utils import get_column_normalizer, get_img_preprocessor  # noqa: E402
from patch_vit_stem import replace_vit_patch_embeddings  # noqa: E402

from liquid_arc.config import LiquidARCConfig           # noqa: E402
from liquid_arc_lewm import LiquidARCPredictor          # noqa: E402


HISTORY = 3
ENCODER_SCALE = "tiny"
PATCH_SIZE = 14
IMG_SIZE = 224
EMBED_DIM = 192
HIDDEN_DIM = 192
ACTION_DIM = 2       # pushT
FRAMESKIP = 5


def _make_jepa(kind: str, ar_depth: int = 6, ar_heads: int = 16,
               ar_dim_head: int = 64, ar_mlp_dim: int = 2048) -> JEPA:
    encoder = spt.backbone.utils.vit_hf(
        ENCODER_SCALE, patch_size=PATCH_SIZE, image_size=IMG_SIZE,
        pretrained=False, use_mask_token=False)
    replace_vit_patch_embeddings(encoder)

    if kind == "ar":
        predictor = ARPredictor(
            num_frames=HISTORY, input_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
            output_dim=HIDDEN_DIM, depth=ar_depth, heads=ar_heads,
            mlp_dim=ar_mlp_dim, dim_head=ar_dim_head, dropout=0.1,
            emb_dropout=0.0)
    elif kind == "liquid":
        ode_cfg = LiquidARCConfig(
            d_model=EMBED_DIM, d_metric=48, d_metric_bottleneck=96,
            metric_rank=32, d_ffn=512, n_ode_steps=16,
            ode_steps_min=16, ode_steps_max=16, integration_time=2.0,
            tau_min=0.5, tau_max=1.0, use_torch_compile=False)
        predictor = LiquidARCPredictor(
            input_dim=EMBED_DIM, action_emb_dim=EMBED_DIM,
            ode_config=ode_cfg, output_dim=HIDDEN_DIM, dropout=0.1)
    else:
        raise ValueError(kind)

    action_encoder = Embedder(input_dim=FRAMESKIP * ACTION_DIM, emb_dim=EMBED_DIM)
    projector = MLP(input_dim=HIDDEN_DIM, output_dim=EMBED_DIM,
                    hidden_dim=2048, norm_fn=torch.nn.LayerNorm)
    pred_proj = MLP(input_dim=EMBED_DIM, output_dim=EMBED_DIM,
                    hidden_dim=2048, norm_fn=torch.nn.LayerNorm)
    return JEPA(encoder=encoder, predictor=predictor,
                action_encoder=action_encoder, projector=projector,
                pred_proj=pred_proj)


def _load_ckpt_into(model: JEPA, ckpt: Path, skip_prefix: str = "") -> None:
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    own = dict(model.state_dict())
    sd = {}
    for k, v in state.items():
        k2 = k
        if k2.startswith("model."):
            k2 = k2[len("model."):]
        if skip_prefix and k2.startswith(skip_prefix):
            continue
        # Remap ViT Conv2d patch weights → UnfoldLinearPatchEmbed.proj
        if k2.endswith("patch_embeddings.projection.weight") and k2 not in own:
            k2 = k2.replace("projection.weight", "projection.proj.weight")
        if k2.endswith("patch_embeddings.projection.bias") and k2 not in own:
            k2 = k2.replace("projection.bias", "projection.proj.bias")
        if k2 in own and own[k2].shape != v.shape:
            if v.ndim == 4 and own[k2].ndim == 2 and v.shape[0] == own[k2].shape[0]:
                v = v.reshape(v.shape[0], -1)
        sd[k2] = v
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[ckpt] {ckpt.name}: missing={len(missing)}, unexpected={len(unexpected)}",
          flush=True)


@torch.no_grad()
def _rollout_mse(model: JEPA, batch: dict, horizon: int, device: str) -> torch.Tensor:
    """Returns per-step MSE [horizon] averaged over this batch."""
    # Encode all T+H frames (ground truth)
    moved = {k: (v.to(device) if torch.is_tensor(v) else v)
             for k, v in batch.items()}
    gt = model.encode(moved)
    emb = gt["emb"]           # [B, T+H, D]
    if torch.isnan(emb).any() or torch.isinf(emb).any():
        print(f"[debug] encoder produced NaN/Inf — pix range "
              f"{moved['pixels'].min().item():.3f}..{moved['pixels'].max().item():.3f} "
              f"emb nan count {torch.isnan(emb).sum().item()}",
              flush=True)
    act_emb = gt["act_emb"]   # [B, T+H, D]
    T = HISTORY
    H = horizon
    assert emb.shape[1] >= T + H, (emb.shape, T + H)

    # seed with first T frames
    cur = emb[:, :T].clone()         # [B, T, D]
    per_step = []
    for h in range(H):
        # predictor ingests last T embeddings + last T actions
        act_window = act_emb[:, h:h + T]  # shift with rollout
        pred = model.predict(cur, act_window)  # [B, T, D]
        nxt = pred[:, -1:]                     # [B, 1, D]
        target = emb[:, T + h:T + h + 1]       # [B, 1, D]
        per_step.append(F.mse_loss(nxt, target).item())
        cur = torch.cat([cur[:, 1:], nxt], dim=1)  # slide window
    return torch.tensor(per_step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar_ckpt", required=True)
    ap.add_argument("--liquid_ckpt", required=True)
    ap.add_argument("--pretrained_ckpt", required=True)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 20])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--n_batches", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ar_depth", type=int, default=6)
    ap.add_argument("--ar_heads", type=int, default=16)
    ap.add_argument("--ar_dim_head", type=int, default=64)
    ap.add_argument("--ar_mlp_dim", type=int, default=2048)
    ap.add_argument("--include_identity", action="store_true",
                    help="Also report an identity baseline (pred = last emb)")
    args = ap.parse_args()

    max_H = max(args.horizons)
    total_T = HISTORY + max_H

    # Data — reuse LeWM's PushT loader config with expanded num_steps
    from omegaconf import OmegaConf
    data_cfg = OmegaConf.create({
        "dataset": {
            "num_steps": total_T,
            "frameskip": FRAMESKIP,
            "name": "pusht_expert_train",
            "keys_to_load": ["pixels", "action", "proprio", "state"],
            "keys_to_cache": ["action", "proprio", "state"],
        }
    })
    dataset = swm.data.HDF5Dataset(**data_cfg.dataset, transform=None)
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=IMG_SIZE)]
    for col in data_cfg.dataset.keys_to_load:
        if col.startswith("pixels"):
            continue
        transforms.append(get_column_normalizer(dataset, col, col))
    dataset.transform = spt.data.transforms.Compose(*transforms)

    gen = torch.Generator().manual_seed(0)
    _, val_set = spt.data.random_split(dataset, [0.9, 0.1], generator=gen)
    loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=4, drop_last=True)

    results = {}
    for kind, ckpt in [("ar", args.ar_ckpt), ("liquid", args.liquid_ckpt)]:
        print(f"\n=== {kind.upper()} ===", flush=True)
        model = _make_jepa(
            kind, ar_depth=args.ar_depth, ar_heads=args.ar_heads,
            ar_dim_head=args.ar_dim_head, ar_mlp_dim=args.ar_mlp_dim)
        _load_ckpt_into(model, Path(args.pretrained_ckpt),
                        skip_prefix="predictor.")           # encoder+act_enc+proj only
        _load_ckpt_into(model, Path(ckpt))                   # predictor (from Phase 2)
        model = model.to(args.device)
        model.eval()

        horizon_mse = {h: [] for h in args.horizons}
        for i, batch in enumerate(loader):
            if i >= args.n_batches:
                break
            per_step = _rollout_mse(model, batch, max_H, args.device)
            for h in args.horizons:
                horizon_mse[h].append(per_step[:h].mean().item())
        results[kind] = {h: float(torch.tensor(v).mean()) for h, v in horizon_mse.items()}
        print(json.dumps(results[kind], indent=2), flush=True)
        del model
        torch.cuda.empty_cache()

    # Identity baseline: predict next_emb = last_emb (no predictor at all)
    if args.include_identity:
        print("\n=== IDENTITY BASELINE ===", flush=True)
        # Use AR-shaped model to get encoder features (predictor never called)
        model = _make_jepa("ar", args.ar_depth, args.ar_heads,
                           args.ar_dim_head, args.ar_mlp_dim)
        _load_ckpt_into(model, Path(args.pretrained_ckpt), skip_prefix="predictor.")
        model = model.to(args.device); model.eval()
        horizon_mse = {h: [] for h in args.horizons}
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= args.n_batches: break
                gt = model.encode({k: v.to(args.device) if torch.is_tensor(v) else v
                                    for k, v in batch.items()})
                emb = gt["emb"]
                T, H = HISTORY, max(args.horizons)
                last = emb[:, T - 1:T]    # [B,1,D]
                per_step = []
                for h in range(H):
                    target = emb[:, T + h:T + h + 1]
                    per_step.append(F.mse_loss(last, target).item())
                for h in args.horizons:
                    horizon_mse[h].append(torch.tensor(per_step[:h]).mean().item())
        results["identity"] = {h: float(torch.tensor(v).mean()) for h, v in horizon_mse.items()}
        print(json.dumps(results["identity"], indent=2), flush=True)
        del model; torch.cuda.empty_cache()

    print("\n=== SUMMARY (cumulative MSE up to horizon H) ===", flush=True)
    has_id = "identity" in results
    hdr = f"{'H':>4}  {'AR':>12}  {'LIQUID':>12}  {'LIQUID/AR':>10}"
    if has_id:
        hdr += f"  {'IDENTITY':>12}"
    print(hdr)
    for h in args.horizons:
        ar = results["ar"][h]
        lq = results["liquid"][h]
        row = f"{h:>4}  {ar:>12.6f}  {lq:>12.6f}  {lq/ar:>10.3f}"
        if has_id:
            row += f"  {results['identity'][h]:>12.6f}"
        print(row)

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
