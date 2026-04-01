"""Probe model internals at a checkpoint to understand phase shift effects.

Hooks into FluidLayer to capture per-layer:
  - Diffusion kernel entropy and effective rank
  - Per-position curvature distribution
  - Distance matrix structure (D²)
  - Scale separation (which timescale dominates)
  - Metric spectrum (per-dimension activity)
  - Representation geometry (hidden state similarity)

Usage:
  python scripts/probe_phase_shift.py \
      --config configs/criticality_starved.yaml \
      --checkpoint output_grokking/checkpoints/step_5000.pt \
      --task_kwargs '{"n_rooms_max": 5, ...}'
"""

import argparse
import json
import math
import sys
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fgn.config import FGNConfig
from fgn.model_fluid import FluidNetModel
from fgn.fluid_layer import FluidLayer
from fgn.tasks import get_task


class LayerProbe:
    """Hook into a FluidLayer to capture internal tensors."""

    def __init__(self, layer: FluidLayer, layer_idx: int):
        self.layer_idx = layer_idx
        self.layer = layer
        self.captured = {}

    @torch.no_grad()
    def probe(self, h, context, mask):
        """Run a probing forward pass capturing all internals."""
        B, N, d = h.shape

        # 1. Metric
        h_normed = self.layer.norm_geo(h)
        ctx_exp = context.unsqueeze(1).expand(B, N, d)
        cat_input = torch.cat([h_normed, ctx_exp], dim=-1)
        g = F.softplus(self.layer.metric_net_linear2(
            F.gelu(self.layer.metric_net_linear1(cat_input))
        ))

        # 2. Curvature
        kappa = self.layer.curvature_engine(g)

        # 3. Distances
        if N <= self.layer.chunk_size:
            D_sq = self.layer._direct_distance(h_normed, g)
        else:
            D_sq = self.layer._chunked_distance(h_normed, g)

        # 4. Timescales
        t = F.softplus(self.layer.time_net_linear2(
            F.gelu(self.layer.time_net_linear1(self.layer.norm_time(h)))
        ))

        # 5. Kernels
        eps = 1e-6
        kernels = []
        for s in range(self.layer.n_scales):
            t_s = t[:, :, s:s + 1]
            log_K = -D_sq / (4.0 * t_s + eps)
            if mask is not None:
                log_K = log_K.masked_fill(mask.unsqueeze(0), float('-inf'))
            kernels.append(F.softmax(log_K, dim=-1))

        # 6. Values and propagated output
        h_val = self.layer.norm_val(h)
        values = [wv(h_val) for wv in self.layer.W_v]
        propagated = torch.cat(
            [K @ V for K, V in zip(kernels, values)], dim=-1
        )
        h_out = h + self.layer.resid_drop(self.layer.W_o(propagated))
        h_out = h_out + self.layer.resid_drop(self.layer.ffn(self.layer.norm_ff(h_out)))

        self.captured = {
            "g": g,           # [B, N, d] metric
            "kappa": kappa,   # [B, N] curvature
            "D_sq": D_sq,     # [B, N, N] distances
            "t": t,           # [B, N, n_scales] timescales
            "kernels": kernels,  # list of [B, N, N]
            "h_in": h,        # [B, N, d] input
            "h_out": h_out,   # [B, N, d] output
        }
        return h_out


def compute_kernel_stats(kernels, mask=None):
    """Compute entropy and effective rank of diffusion kernels."""
    stats = []
    for s, K in enumerate(kernels):
        # K: [B, N, N], each row sums to 1
        # Entropy per position
        log_K = torch.log(K + 1e-10)
        entropy = -(K * log_K).sum(dim=-1)  # [B, N]
        # Max possible entropy = log(N) for uniform
        N = K.shape[-1]
        max_entropy = math.log(N)
        normalized_entropy = entropy / max_entropy  # 0=peaked, 1=uniform

        # Effective rank: exp(entropy)
        eff_rank = torch.exp(entropy)  # how many positions effectively attended to

        # Max attention weight per position
        max_weight = K.max(dim=-1).values  # [B, N]

        # Top-5 concentration: what fraction of weight is in top 5 positions
        top5 = K.topk(min(5, N), dim=-1).values.sum(dim=-1)  # [B, N]

        stats.append({
            "scale": s,
            "entropy_mean": entropy.mean().item(),
            "entropy_std": entropy.std().item(),
            "norm_entropy": normalized_entropy.mean().item(),
            "eff_rank_mean": eff_rank.mean().item(),
            "eff_rank_std": eff_rank.std().item(),
            "max_weight_mean": max_weight.mean().item(),
            "top5_concentration": top5.mean().item(),
        })
    return stats


def compute_distance_stats(D_sq, mask=None):
    """Analyze the structure of the distance matrix."""
    B, N, _ = D_sq.shape

    # Basic stats (upper triangle, no self-distances)
    triu_mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=D_sq.device), diagonal=1)
    dists = D_sq[:, triu_mask]  # [B, N*(N-1)/2]

    stats = {
        "mean": dists.mean().item(),
        "std": dists.std().item(),
        "min": dists.min().item(),
        "max": dists.max().item(),
        "median": dists.median().item(),
    }

    # Distance distribution: how many "close" vs "far" pairs?
    # Normalized distances
    d_norm = dists / (dists.max() + 1e-8)
    stats["frac_close"] = (d_norm < 0.1).float().mean().item()  # within 10% of max
    stats["frac_far"] = (d_norm > 0.9).float().mean().item()    # beyond 90% of max

    # Clustering coefficient: ratio of within-cluster to between-cluster distances
    # Use k-means-like: split positions into 2 groups by D², check separation
    # Simple proxy: coefficient of variation of row means
    row_means = D_sq.mean(dim=-1)  # [B, N] — average distance from each position
    row_cv = row_means.std(dim=-1) / (row_means.mean(dim=-1) + 1e-8)  # [B]
    stats["row_mean_cv"] = row_cv.mean().item()  # high = positions have different neighborhoods

    return stats


def compute_metric_spectrum(g):
    """Analyze which dimensions of the metric are active."""
    # g: [B, N, d]
    B, N, d = g.shape

    # Per-dimension stats across positions
    g_mean = g.mean(dim=(0, 1))  # [d]
    g_std = g.std(dim=(0, 1))    # [d]
    g_cv = g_std / (g_mean + 1e-8)  # [d] coefficient of variation per dimension

    # Sort dimensions by activity (CV)
    sorted_cv, sorted_idx = g_cv.sort(descending=True)

    # How many "active" dimensions (CV > 0.1)?
    n_active = (g_cv > 0.1).sum().item()
    n_dead = (g_cv < 0.01).sum().item()

    # Top/bottom dimension activity
    top5_cv = sorted_cv[:5].tolist()
    bot5_cv = sorted_cv[-5:].tolist()

    return {
        "n_dims": d,
        "n_active": int(n_active),
        "n_dead": int(n_dead),
        "mean_cv": g_cv.mean().item(),
        "max_cv": g_cv.max().item(),
        "min_cv": g_cv.min().item(),
        "top5_cv": top5_cv,
        "bot5_cv": bot5_cv,
        "g_mean_range": [g_mean.min().item(), g_mean.max().item()],
    }


def compute_curvature_stats(kappa):
    """Analyze curvature distribution across positions."""
    # kappa: [B, N]
    stats = {
        "mean": kappa.mean().item(),
        "std": kappa.std().item(),
        "abs_mean": kappa.abs().mean().item(),
        "min": kappa.min().item(),
        "max": kappa.max().item(),
        "frac_positive": (kappa > 0).float().mean().item(),
        "frac_negative": (kappa < 0).float().mean().item(),
    }

    # Is curvature concentrated or spread?
    k_abs = kappa.abs()
    k_norm = k_abs / (k_abs.max() + 1e-8)
    stats["frac_high_curvature"] = (k_norm > 0.5).float().mean().item()
    stats["gini"] = _gini_coefficient(k_abs.reshape(-1)).item()

    return stats


def _gini_coefficient(values):
    """Compute Gini coefficient (0=uniform, 1=concentrated)."""
    sorted_vals = values.sort().values
    n = len(sorted_vals)
    index = torch.arange(1, n + 1, dtype=values.dtype, device=values.device)
    return (2.0 * (index * sorted_vals).sum() / (n * sorted_vals.sum() + 1e-8) - (n + 1) / n)


def compute_representation_stats(h_in, h_out):
    """Analyze how representations change through the layer."""
    B, N, d = h_in.shape

    # Cosine similarity between input and output per position
    cos_sim = F.cosine_similarity(h_in, h_out, dim=-1)  # [B, N]

    # How much does the layer change representations?
    delta = (h_out - h_in).norm(dim=-1)  # [B, N]
    h_norm = h_in.norm(dim=-1)  # [B, N]
    relative_change = delta / (h_norm + 1e-8)

    # Position-position similarity in output space
    h_out_norm = F.normalize(h_out, dim=-1)
    sim_matrix = torch.bmm(h_out_norm, h_out_norm.transpose(1, 2))  # [B, N, N]
    # Off-diagonal mean
    off_diag = sim_matrix.masked_fill(
        torch.eye(N, device=sim_matrix.device).bool().unsqueeze(0), 0
    )
    mean_sim = off_diag.sum(dim=(1, 2)) / (N * (N - 1))

    return {
        "cos_sim_mean": cos_sim.mean().item(),
        "cos_sim_std": cos_sim.std().item(),
        "relative_change_mean": relative_change.mean().item(),
        "relative_change_std": relative_change.std().item(),
        "inter_position_sim": mean_sim.mean().item(),
    }


def compute_scale_separation(kernels):
    """Analyze how different the 3 timescale kernels are from each other."""
    if len(kernels) < 2:
        return {"n_scales": len(kernels)}

    # KL divergence between scales (averaged over positions)
    stats = {}
    for i in range(len(kernels)):
        for j in range(i + 1, len(kernels)):
            Ki = kernels[i] + 1e-10  # [B, N, N]
            Kj = kernels[j] + 1e-10
            # KL(Ki || Kj) per position
            kl = (Ki * (Ki.log() - Kj.log())).sum(dim=-1)  # [B, N]
            stats[f"kl_{i}_{j}"] = kl.mean().item()

    # Jensen-Shannon divergence between local and global
    K_local = kernels[0] + 1e-10
    K_global = kernels[-1] + 1e-10
    M = (K_local + K_global) / 2
    js = 0.5 * (K_local * (K_local.log() - M.log())).sum(-1) + \
         0.5 * (K_global * (K_global.log() - M.log())).sum(-1)
    stats["js_local_global"] = js.mean().item()

    return stats


@torch.no_grad()
def probe_checkpoint(config, checkpoint_path, task_kwargs, n_episodes=16):
    """Run full diagnostic probe on a checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = FluidNetModel(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    step = ckpt.get("step", "?")

    print(f"\n{'='*60}")
    print(f"  Phase Shift Probe — step {step}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")

    # Generate episodes
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    task = get_task("CW", tokenizer, seq_len=config.max_seq_len, **task_kwargs)

    pad_id = tokenizer.eos_token_id or 0
    all_ids, all_labels, all_ctx_masks = [], [], []

    for _ in range(n_episodes):
        for _retry in range(200):
            ep = task._generate_valid_episode()
            if ep is None:
                continue
            text, _, _, _, _, _ = ep
            ids, labels, ctx_end, _, _ = task._tokenize_episode(text)
            if len(ids) > config.max_seq_len:
                ids = ids[:config.max_seq_len]
                labels = labels[:config.max_seq_len]
            else:
                pad_len = config.max_seq_len - len(ids)
                ids += [pad_id] * pad_len
                labels += [-100] * pad_len
            if sum(1 for l in labels if l != -100) >= 5:
                break
        else:
            continue

        ctx_mask = [i < min(ctx_end, config.max_seq_len) for i in range(config.max_seq_len)]
        all_ids.append(ids)
        all_labels.append(labels)
        all_ctx_masks.append(ctx_mask)

    if not all_ids:
        print("  ERROR: No episodes generated")
        return

    input_ids = torch.tensor(all_ids, dtype=torch.long, device=device)
    labels = torch.tensor(all_labels, dtype=torch.long, device=device)
    ctx_mask = torch.tensor(all_ctx_masks, dtype=torch.bool, device=device)

    print(f"  Episodes: {len(all_ids)}")

    # Forward pass to get embeddings and context
    B = input_ids.shape[0]
    N = config.max_seq_len
    causal_mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=device), diagonal=1)

    pos = torch.arange(N, device=device).unsqueeze(0)
    h = model.embed(input_ids) + model.pos_embed(pos)
    context = model.context_pool(h, ctx_mask)

    # Probe each layer
    for layer_idx, layer in enumerate(model.layers):
        probe = LayerProbe(layer, layer_idx)
        h = probe.probe(h, context, causal_mask)
        cap = probe.captured

        print(f"\n  --- Layer {layer_idx} ---")

        # Kernel stats
        k_stats = compute_kernel_stats(cap["kernels"], causal_mask)
        for ks in k_stats:
            s = ks["scale"]
            print(f"    Scale {s}: entropy={ks['entropy_mean']:.3f} "
                  f"(norm={ks['norm_entropy']:.3f}), "
                  f"eff_rank={ks['eff_rank_mean']:.1f}, "
                  f"max_w={ks['max_weight_mean']:.3f}, "
                  f"top5={ks['top5_concentration']:.3f}")

        # Scale separation
        sep = compute_scale_separation(cap["kernels"])
        if "js_local_global" in sep:
            print(f"    Scale separation (JS local↔global): {sep['js_local_global']:.4f}")

        # Distance stats
        d_stats = compute_distance_stats(cap["D_sq"], causal_mask)
        print(f"    D²: mean={d_stats['mean']:.2f}, std={d_stats['std']:.2f}, "
              f"close={d_stats['frac_close']:.3f}, far={d_stats['frac_far']:.3f}, "
              f"row_cv={d_stats['row_mean_cv']:.3f}")

        # Curvature stats
        c_stats = compute_curvature_stats(cap["kappa"])
        print(f"    κ: mean={c_stats['mean']:.4f}, |κ|={c_stats['abs_mean']:.4f}, "
              f"std={c_stats['std']:.4f}, gini={c_stats['gini']:.3f}, "
              f"+/{c_stats['frac_positive']:.2f} -/{c_stats['frac_negative']:.2f}")

        # Metric spectrum
        m_stats = compute_metric_spectrum(cap["g"])
        print(f"    Metric: {m_stats['n_active']}/{m_stats['n_dims']} active, "
              f"{m_stats['n_dead']} dead, "
              f"cv={m_stats['mean_cv']:.4f}, "
              f"g=[{m_stats['g_mean_range'][0]:.2f}, {m_stats['g_mean_range'][1]:.2f}]")

        # Representation change
        r_stats = compute_representation_stats(cap["h_in"], cap["h_out"])
        print(f"    Repr: cos_sim={r_stats['cos_sim_mean']:.4f}, "
              f"Δ/|h|={r_stats['relative_change_mean']:.4f}, "
              f"inter_pos_sim={r_stats['inter_position_sim']:.4f}")

    # Final output
    print(f"\n  --- Output ---")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                             enabled=(device.type == "cuda")):
        result = model(input_ids, labels=labels, context_mask=ctx_mask)
    print(f"    CE: {result['ce_loss'].item():.6f}")
    print(f"    |κ|: {result['avg_kappa'].item():.4f}")
    cv = result['metric_cv']
    if isinstance(cv, torch.Tensor):
        cv = cv.item()
    print(f"    CV: {cv:.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Probe model at checkpoint")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--task_kwargs", type=str, default="{}",
                        help="JSON dict of CW task kwargs")
    parser.add_argument("--n_episodes", type=int, default=16)

    args = parser.parse_args()
    config = FGNConfig.from_yaml(args.config)
    task_kwargs = json.loads(args.task_kwargs)

    probe_checkpoint(config, args.checkpoint, task_kwargs, args.n_episodes)


if __name__ == "__main__":
    main()
