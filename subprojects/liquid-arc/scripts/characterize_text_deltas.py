#!/usr/bin/env python3
"""Characterize text delta distribution for LiquidARC criticality calibration.

Feeds diverse texts through the delta extractor, computes pairwise D² statistics
with and without the MetricNet, and compares to ARC D² ranges.

This tells us:
1. What D² range text deltas produce (raw and metric-weighted)
2. Whether there's natural clustering (heavy tail vs uniform)
3. What D²/4τ target would be appropriate for text
4. Whether the ARC-trained MetricNet helps or hurts on text

Usage:
    python scripts/characterize_text_deltas.py \
        --model_path /workspace/models/qwen3-4b \
        --checkpoint output_crit_2560/checkpoints/step_500.pt \
        --config configs/mind_qwen3_delta.yaml
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liquid_arc.delta_extractor import DeltaExtractor
from liquid_arc.config import LiquidARCConfig
from liquid_arc.model import LiquidARCModel

DIVERSE_TEXTS = [
    # Short function-word heavy
    "The cat sat on the mat.",
    "It is what it is.",
    # Content-dense
    "Quantum entanglement enables instantaneous correlation between distant particles.",
    "The Riemann hypothesis concerns the distribution of prime numbers.",
    # Narrative
    "She walked into the room, noticed the broken window, and immediately called the police.",
    "After three years of drought, the river finally began to flow again.",
    # Technical
    "The gradient descent algorithm minimizes the loss function by iteratively adjusting parameters.",
    "Docker containers provide isolated environments for running microservices at scale.",
    # Conversational
    "What do you think about the new policy changes?",
    "I'm not sure I understand your question, could you rephrase it?",
    # Topic shift (long)
    "The bridge collapsed due to structural fatigue. Meanwhile, food prices continued to rise across the region.",
    "Topology studies properties preserved under continuous deformation. In physics, topological insulators exhibit edge states.",
    # Abstract
    "The relationship between form and function defines architectural design philosophy.",
    "Consciousness remains one of the most challenging problems in neuroscience.",
    # Code-like
    "def compute_loss(predictions, targets): return F.cross_entropy(predictions, targets)",
    "SELECT users.name, orders.total FROM users JOIN orders ON users.id = orders.user_id",
]


def pairwise_D_sq(h, g=None):
    """Compute pairwise D² between all position pairs.

    Args:
        h: [N, d] position vectors
        g: [N, d] diagonal metric (None = identity/Euclidean)

    Returns:
        D_sq: [N*(N-1)/2] all pairwise distances
    """
    N = h.shape[0]
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            diff = h[i] - h[j]
            if g is not None:
                g_avg = (g[i] + g[j]) * 0.5
                d_sq = (diff * g_avg * diff).sum()
            else:
                d_sq = (diff * diff).sum()
            pairs.append(d_sq.item())
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device

    # Load delta extractor
    print("═══ Loading Delta Extractor ═══")
    extractor = DeltaExtractor(
        model_path=args.model_path, d_arc=2560, device=device)

    # Load LiquidARC model (for MetricNet)
    print("\n═══ Loading LiquidARC ═══")
    config = LiquidARCConfig.from_yaml(args.config)
    model = LiquidARCModel(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
    cleaned = {k.replace('._orig_mod.', '.'): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=False)
    model.eval()
    dynamics = model.dynamics
    print(f"  Loaded: d={config.d_model}, d_metric={config.d_metric}")

    # Process texts
    print(f"\n═══ Processing {len(DIVERSE_TEXTS)} texts ═══")

    all_raw_D_sq = []      # Euclidean D² (no metric)
    all_metric_D_sq = []   # Metric-weighted D²
    all_delta_norms = []   # Per-token delta norms
    per_text_stats = []

    for i, text in enumerate(DIVERSE_TEXTS):
        result = extractor.extract(text, max_tokens=128)
        delta_h = result['delta_h'][0].float().to(device)  # [N, d]
        N = delta_h.shape[0]

        # Per-token delta norms
        norms = delta_h.norm(dim=-1).tolist()
        all_delta_norms.extend(norms)

        # Raw D² (Euclidean)
        raw_pairs = pairwise_D_sq(delta_h)
        all_raw_D_sq.extend(raw_pairs)

        # Metric-weighted D² (through MetricNet)
        with torch.no_grad():
            h_normed = dynamics.norm_geo(delta_h.unsqueeze(0))
            context = model.context_pool(delta_h.unsqueeze(0))
            ctx_exp = context.unsqueeze(1).expand(-1, N, -1)
            metric_input = torch.cat([h_normed, ctx_exp], dim=-1)
            hidden = F.gelu(dynamics.metric_net_linear1(metric_input))
            g = F.softplus(dynamics.metric_net_linear2_diag(hidden))  # [1, N, d]

            metric_pairs = pairwise_D_sq(h_normed[0], g[0])
            all_metric_D_sq.extend(metric_pairs)

            cv = (g.std() / (g.mean() + 1e-8)).item()
            tau = dynamics.compute_tau(delta_h.unsqueeze(0))

        raw_med = sorted(raw_pairs)[len(raw_pairs) // 2] if raw_pairs else 0
        met_med = sorted(metric_pairs)[len(metric_pairs) // 2] if metric_pairs else 0

        per_text_stats.append({
            'text': text[:60],
            'n_tokens': N,
            'delta_norm_mean': sum(norms) / len(norms),
            'raw_D_sq_median': raw_med,
            'metric_D_sq_median': met_med,
            'cv': cv,
            'tau_mean': tau.mean().item(),
        })

        print(f"  [{i:2d}] N={N:3d} δ_norm={sum(norms)/len(norms):.2f} "
              f"D²_raw={raw_med:.1f} D²_met={met_med:.1f} "
              f"CV={cv:.2f} tau={tau.mean().item():.2f} "
              f"\"{text[:50]}\"")

    # Global statistics
    print(f"\n═══ Global Statistics ({len(all_raw_D_sq)} pairs) ═══")

    raw_sorted = sorted(all_raw_D_sq)
    met_sorted = sorted(all_metric_D_sq)
    norm_sorted = sorted(all_delta_norms)

    def stats(vals, name):
        n = len(vals)
        s = sorted(vals)
        print(f"  {name}:")
        print(f"    mean={sum(s)/n:.2f}, median={s[n//2]:.2f}")
        print(f"    p10={s[n//10]:.2f}, p25={s[n//4]:.2f}, "
              f"p75={s[3*n//4]:.2f}, p90={s[9*n//10]:.2f}")
        print(f"    min={s[0]:.2f}, max={s[-1]:.2f}")
        return s[n // 2]

    print()
    stats(all_delta_norms, "Delta norms (per-token ||Δh||)")
    print()
    raw_median = stats(all_raw_D_sq, "Raw D² (Euclidean, no metric)")
    print()
    met_median = stats(all_metric_D_sq, "Metric D² (MetricNet-weighted)")

    # Compute tau stats
    with torch.no_grad():
        # Compute on a representative input
        rep = extractor.extract(DIVERSE_TEXTS[2], max_tokens=128)
        rep_h = rep['delta_h'].float().to(device)
        tau_rep = dynamics.compute_tau(rep_h)
        tau_med = tau_rep.median().item()

    print(f"\n═══ Criticality Assessment ═══")
    print(f"  tau median: {tau_med:.3f}")
    print(f"  Raw D²/4τ:    {raw_median / (4 * tau_med + 1e-8):.1f}  (target for text)")
    print(f"  Metric D²/4τ: {met_median / (4 * tau_med + 1e-8):.1f}  (current with ARC MetricNet)")
    print(f"  ARC target:   60.0  (from d=2560 criticality training)")
    print()

    # Ratio comparison
    amp = met_median / (raw_median + 1e-8)
    print(f"  MetricNet amplification: {amp:.3f}×")
    if amp > 1:
        print(f"  → MetricNet AMPLIFIES distances on text (makes routing LESS structured)")
    else:
        print(f"  → MetricNet COMPRESSES distances on text (makes routing MORE structured)")

    print(f"\n  Recommended text criticality target (D²/4τ): "
          f"{met_median / (4 * tau_med + 1e-8):.1f}")
    print(f"  Or adaptive: EMA track D²_median / (4 * tau_median)")

    # Distribution shape
    print(f"\n═══ Distribution Shape ═══")
    n = len(all_metric_D_sq)
    s = sorted(all_metric_D_sq)
    # Heavy tail test: ratio of p90/p50
    ratio = s[9*n//10] / (s[n//2] + 1e-8)
    print(f"  p90/p50 ratio: {ratio:.2f}")
    if ratio > 3.0:
        print(f"  → HEAVY TAIL (good for heat kernel — natural clustering)")
    elif ratio > 1.5:
        print(f"  → MODERATE tail (some structure)")
    else:
        print(f"  → UNIFORM (poor — heat kernel sees all pairs as similar)")


if __name__ == "__main__":
    main()
