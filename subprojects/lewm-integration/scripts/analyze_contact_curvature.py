"""Criterion 6: Contact curvature analysis.

Measures whether the learned Riemannian metric produces higher curvature
at contact boundary states vs free-motion states in PushT.

Protocol:
  1. Load frozen encoder + trained Liquid predictor (with low-rank metric)
  2. Encode PushT validation trajectories → latent embeddings
  3. For each timestep, compute the metric g = D + L·Lᵀ from MetricNet
  4. Extract eigenvalues of g at each position (λ₁ ≥ λ₂ ≥ ... ≥ λ_d)
  5. Proxy for "contact" vs "free motion":
     - Use state['keypoint_contact'] if available, OR
     - Use velocity magnitude as proxy: low velocity near contact (object
       decelerating), high velocity during free motion
     - Alternatively: use prediction error as proxy — high pred error = surprise
       = likely contact/state transition
  6. Compare metric eigenvalue statistics between contact and free-motion bins

Output: per-timestep {eigenvalues, velocity, pred_error, metric_cv} → correlation
analysis + bin comparison.

Usage:
    STABLEWM_HOME=... PYTHONPATH=... python scripts/analyze_contact_curvature.py \
        --liquid_ckpt /path/to/liquid_crit_weights.ckpt \
        --pretrained_ckpt /path/to/weights.pt \
        --n_batches 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "le-wm"))
sys.path.insert(0, str(_ROOT / "scripts"))

import stable_pretraining as spt
import stable_worldmodel as swm

from jepa import JEPA
from module import ARPredictor, Embedder, MLP
from utils import get_column_normalizer, get_img_preprocessor
from patch_vit_stem import replace_vit_patch_embeddings

from liquid_arc.config import LiquidARCConfig
from liquid_arc_lewm import LiquidARCPredictor


EMBED_DIM = 192
HISTORY = 3
FRAMESKIP = 5


def _make_liquid_jepa() -> JEPA:
    encoder = spt.backbone.utils.vit_hf(
        "tiny", patch_size=14, image_size=224, pretrained=False,
        use_mask_token=False)
    replace_vit_patch_embeddings(encoder)
    ode_cfg = LiquidARCConfig(
        d_model=EMBED_DIM, d_metric=48, d_metric_bottleneck=96,
        metric_rank=32, d_ffn=512, n_ode_steps=16,
        ode_steps_min=16, ode_steps_max=16, integration_time=2.0,
        tau_min=0.5, tau_max=1.0, use_torch_compile=False)
    predictor = LiquidARCPredictor(
        input_dim=EMBED_DIM, action_emb_dim=EMBED_DIM,
        ode_config=ode_cfg, output_dim=EMBED_DIM, dropout=0.1)
    action_encoder = Embedder(input_dim=FRAMESKIP * 2, emb_dim=EMBED_DIM)
    projector = MLP(input_dim=EMBED_DIM, output_dim=EMBED_DIM,
                    hidden_dim=2048, norm_fn=torch.nn.LayerNorm)
    pred_proj = MLP(input_dim=EMBED_DIM, output_dim=EMBED_DIM,
                    hidden_dim=2048, norm_fn=torch.nn.LayerNorm)
    return JEPA(encoder=encoder, predictor=predictor,
                action_encoder=action_encoder, projector=projector,
                pred_proj=pred_proj)


def _load_ckpt(model, ckpt_path, skip_prefix=""):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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


def _compute_metric_eigenvalues(predictor: LiquidARCPredictor,
                                 emb: torch.Tensor) -> dict:
    """Compute per-position metric eigenvalues from the Liquid predictor.

    Args:
        predictor: trained LiquidARCPredictor
        emb: [B, T, D] latent embeddings

    Returns:
        dict with eigenvalues [B, T, d], metric_cv [B, T], top_eigenratio [B, T]
    """
    dyn = predictor.dynamics
    h = predictor.proj_in(emb)     # [B, T, d_model]
    h_n = dyn.norm_geo(h)
    B, N, d = h_n.shape

    ctx = torch.zeros(B, d, device=h.device, dtype=h.dtype)
    ctx_exp = ctx.unsqueeze(1).expand(B, N, d)
    cat_input = torch.cat([h_n, ctx_exp], dim=-1)

    met_hidden = F.gelu(dyn.metric_net_linear1(cat_input))
    g_diag = F.softplus(dyn.metric_net_linear2_diag(met_hidden))  # [B, N, d]

    result = {"g_diag_mean": g_diag.mean(dim=-1).cpu().numpy()}  # [B, N]

    if dyn.metric_rank > 0:
        L_flat = dyn.metric_net_linear2_lr(met_hidden)
        L = L_flat.view(B, N, d, dyn.metric_rank)  # [B, N, d, rank]

        # Full metric per position: g_full = diag(g_diag) + L @ L^T
        # Eigenvalues of symmetric PSD matrix via torch.linalg.eigvalsh
        # For large d, we compute top-k via L·Lᵀ eigenvalues + diagonal shift
        # Efficient: eigenvalues of L^T @ L (rank × rank) + diagonal contribution
        # Full approach for d=192 is tractable
        eigs_all = []
        g_diag_cpu = g_diag[:min(B, 4)].cpu()
        L_cpu = L[:min(B, 4)].cpu()
        for b in range(min(B, 4)):
            for n in range(N):
                g_full = torch.diag(g_diag_cpu[b, n]) + L_cpu[b, n] @ L_cpu[b, n].T
                eigvals = torch.linalg.eigvalsh(g_full)
                eigs_all.append(eigvals.flip(0))
        eigs = torch.stack(eigs_all).view(min(B, 4), N, d).numpy()

        result["eigenvalues"] = eigs
        result["top1_eigenvalue"] = eigs[:, :, 0]       # [B, N]
        result["eigenratio_1_d"] = eigs[:, :, 0] / (eigs[:, :, -1] + 1e-8)
        result["metric_cv"] = np.std(eigs, axis=-1) / (np.mean(eigs, axis=-1) + 1e-8)
    else:
        result["eigenvalues"] = g_diag.cpu().numpy()
        result["metric_cv"] = (g_diag.std(dim=-1) / (g_diag.mean(dim=-1) + 1e-8)).cpu().numpy()

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--liquid_ckpt", required=True)
    ap.add_argument("--pretrained_ckpt", required=True)
    ap.add_argument("--n_batches", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from omegaconf import OmegaConf
    data_cfg = OmegaConf.create({
        "dataset": {
            "num_steps": 10, "frameskip": FRAMESKIP,
            "name": "pusht_expert_train",
            "keys_to_load": ["pixels", "action", "proprio", "state"],
            "keys_to_cache": ["action", "proprio", "state"],
        }
    })
    dataset = swm.data.HDF5Dataset(**data_cfg.dataset, transform=None)
    tr = [get_img_preprocessor(source="pixels", target="pixels", img_size=224)]
    for col in data_cfg.dataset.keys_to_load:
        if col.startswith("pixels"):
            continue
        tr.append(get_column_normalizer(dataset, col, col))
    dataset.transform = spt.data.transforms.Compose(*tr)

    gen = torch.Generator().manual_seed(0)
    _, val_set = spt.data.random_split(dataset, [0.9, 0.1], generator=gen)
    loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=4, drop_last=True)

    model = _make_liquid_jepa()
    _load_ckpt(model, Path(args.pretrained_ckpt), skip_prefix="predictor.")
    _load_ckpt(model, Path(args.liquid_ckpt))
    model = model.to(args.device).eval()

    all_results = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.n_batches:
                break
            moved = {k: v.to(args.device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}

            gt = model.encode(moved)
            emb = gt["emb"]          # [B, T, D]

            # Metric eigenvalues at each position
            metric = _compute_metric_eigenvalues(model.predictor, emb)

            # Velocity proxy: ||emb[t+1] - emb[t]|| — limit to same B as eigenvalues
            B_eig = min(emb.shape[0], 4)
            velocity = (emb[:B_eig, 1:] - emb[:B_eig, :-1]).norm(dim=-1).cpu().numpy()

            # Prediction error proxy: run predictor on context, compare to target
            T = emb.shape[1]
            ctx_emb = emb[:, :HISTORY]
            ctx_act = gt["act_emb"][:, :HISTORY]
            pred = model.predict(ctx_emb, ctx_act)
            pred_last = pred[:, -1]                    # [B, D]
            target = emb[:, HISTORY]                   # [B, D]
            pred_error = (pred_last - target).pow(2).sum(dim=-1).cpu().numpy()  # [B]

            all_results.append({
                "batch": i,
                "top1_eigen": metric.get("top1_eigenvalue", metric["g_diag_mean"]).tolist(),
                "eigenratio": metric.get("eigenratio_1_d",
                    np.ones_like(metric["g_diag_mean"])).tolist(),
                "metric_cv": metric["metric_cv"].tolist(),
                "velocity": velocity.tolist(),
                "pred_error": pred_error.tolist(),
            })

            if i % 10 == 0:
                print(f"  batch {i}: top1_eigen_mean={np.mean(metric.get('top1_eigenvalue', metric['g_diag_mean'])):.4f}  "
                      f"cv_mean={np.mean(metric['metric_cv']):.4f}", flush=True)

    # Aggregate: bin by velocity (low = contact proxy, high = free motion)
    all_top1 = []
    all_cv = []
    all_vel = []
    for r in all_results:
        t1 = np.array(r["top1_eigen"])    # [B, T] or [min(B,4), T]
        cv = np.array(r["metric_cv"])
        vel = np.array(r["velocity"])     # [B, T-1]
        # Align: top1/cv have T positions, vel has T-1
        T_min = min(t1.shape[-1], vel.shape[-1])
        all_top1.extend(t1[:, :T_min].flatten().tolist())
        all_cv.extend(cv[:, :T_min].flatten().tolist())
        all_vel.extend(vel[:, :T_min].flatten().tolist())

    all_top1 = np.array(all_top1)
    all_cv = np.array(all_cv)
    all_vel = np.array(all_vel)

    # Velocity tertiles: low=contact, high=free
    v_low = np.percentile(all_vel, 33)
    v_high = np.percentile(all_vel, 67)
    contact_mask = all_vel <= v_low
    free_mask = all_vel >= v_high

    print(f"\n=== CONTACT CURVATURE ANALYSIS ===")
    print(f"Total samples: {len(all_vel)}")
    print(f"Velocity tertiles: low≤{v_low:.4f}  high≥{v_high:.4f}")
    print(f"\n{'Metric':>20}  {'Contact (low-v)':>15}  {'Free (high-v)':>15}  {'Ratio':>8}")
    for name, arr in [("top1_eigenvalue", all_top1), ("metric_CV", all_cv)]:
        c = arr[contact_mask].mean()
        f_ = arr[free_mask].mean()
        print(f"{name:>20}  {c:>15.4f}  {f_:>15.4f}  {c/f_:>8.3f}")

    # Correlation
    from scipy import stats
    corr_top1, p_top1 = stats.pearsonr(all_vel, all_top1)
    corr_cv, p_cv = stats.pearsonr(all_vel, all_cv)
    print(f"\nVelocity-top1_eigen correlation: r={corr_top1:.4f} (p={p_top1:.2e})")
    print(f"Velocity-metric_CV  correlation: r={corr_cv:.4f} (p={p_cv:.2e})")

    interpretation = (
        "CONTACT CURVATURE HYPOTHESIS:\n"
        "If ratio > 1.0: metric curvature HIGHER at contact (low velocity) — SUPPORTED\n"
        "If ratio < 1.0: metric curvature higher at free motion — NOT SUPPORTED\n"
        "If ratio ≈ 1.0: metric curvature uniform — INCONCLUSIVE"
    )
    print(f"\n{interpretation}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "contact_top1_eigen": float(all_top1[contact_mask].mean()),
            "free_top1_eigen": float(all_top1[free_mask].mean()),
            "contact_cv": float(all_cv[contact_mask].mean()),
            "free_cv": float(all_cv[free_mask].mean()),
            "velocity_top1_corr": float(corr_top1),
            "velocity_cv_corr": float(corr_cv),
        }, indent=2))


if __name__ == "__main__":
    main()
