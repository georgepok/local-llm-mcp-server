"""Save a trained model as _object.ckpt for stable_worldmodel's AutoCostModel.

AutoCostModel expects torch.save'd Python objects. This script builds the JEPA
with our predictor, loads weights, and saves the full object.

Usage:
    PYTHONPATH=... python scripts/save_object_ckpt.py \
        --kind liquid --weights liquid_crit_weights.ckpt \
        --pretrained weights.pt --out liquid_crit_object.ckpt
"""
import argparse
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "le-wm"))
sys.path.insert(0, str(_ROOT / "scripts"))

import stable_pretraining as spt
from module import ARPredictor, Embedder, MLP
from jepa import JEPA
from patch_vit_stem import replace_vit_patch_embeddings

from liquid_arc.config import LiquidARCConfig
from liquid_arc_lewm import LiquidARCPredictor

EMBED_DIM = 192


def _load_ckpt(model, path, skip_prefix=""):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    own = dict(model.state_dict())
    sd = {}
    for k, v in state.items():
        k2 = k[6:] if k.startswith("model.") else k
        if skip_prefix and k2.startswith(skip_prefix):
            continue
        if k2.endswith("patch_embeddings.projection.weight") and k2 not in own:
            k2 = k2.replace("projection.weight", "projection.proj.weight")
        if k2.endswith("patch_embeddings.projection.bias") and k2 not in own:
            k2 = k2.replace("projection.bias", "projection.proj.bias")
        if k2 in own and own[k2].shape != v.shape:
            if v.ndim == 4 and own[k2].ndim == 2 and v.shape[0] == own[k2].shape[0]:
                v = v.reshape(v.shape[0], -1)
        sd[k2] = v
    model.load_state_dict(sd, strict=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["ar", "liquid"], required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ar_depth", type=int, default=2)
    ap.add_argument("--ar_heads", type=int, default=4)
    ap.add_argument("--ar_dim_head", type=int, default=48)
    ap.add_argument("--ar_mlp_dim", type=int, default=512)
    args = ap.parse_args()

    encoder = spt.backbone.utils.vit_hf(
        "tiny", patch_size=14, image_size=224, pretrained=False,
        use_mask_token=False)
    replace_vit_patch_embeddings(encoder)

    if args.kind == "ar":
        predictor = ARPredictor(
            num_frames=3, input_dim=EMBED_DIM, hidden_dim=EMBED_DIM,
            output_dim=EMBED_DIM, depth=args.ar_depth, heads=args.ar_heads,
            mlp_dim=args.ar_mlp_dim, dim_head=args.ar_dim_head,
            dropout=0.1, emb_dropout=0.0)
    else:
        ode_cfg = LiquidARCConfig(
            d_model=EMBED_DIM, d_metric=48, d_metric_bottleneck=96,
            metric_rank=32, d_ffn=512, n_ode_steps=16,
            ode_steps_min=16, ode_steps_max=16, integration_time=2.0,
            tau_min=0.5, tau_max=1.0, use_torch_compile=False)
        predictor = LiquidARCPredictor(
            input_dim=EMBED_DIM, action_emb_dim=EMBED_DIM,
            ode_config=ode_cfg, output_dim=EMBED_DIM, dropout=0.1)

    action_encoder = Embedder(input_dim=5 * 2, emb_dim=EMBED_DIM)
    projector = MLP(input_dim=EMBED_DIM, output_dim=EMBED_DIM,
                    hidden_dim=2048, norm_fn=torch.nn.LayerNorm)
    pred_proj = MLP(input_dim=EMBED_DIM, output_dim=EMBED_DIM,
                    hidden_dim=2048, norm_fn=torch.nn.LayerNorm)
    model = JEPA(encoder=encoder, predictor=predictor,
                 action_encoder=action_encoder, projector=projector,
                 pred_proj=pred_proj)

    _load_ckpt(model, Path(args.pretrained), skip_prefix="predictor.")
    _load_ckpt(model, Path(args.weights))

    model.eval()
    model.interpolate_pos_encoding = True
    torch.save(model, args.out)
    print(f"Saved {args.kind} object checkpoint to {args.out}", flush=True)


if __name__ == "__main__":
    main()
