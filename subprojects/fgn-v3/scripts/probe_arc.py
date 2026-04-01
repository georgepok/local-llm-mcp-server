"""Probe ARC FluidNet internals — understand the accuracy plateau.

Diagnostics:
  1. Grid fidelity: Spearman ρ between geodesic and Euclidean distances
  2. Curvature by role: |κ| at input_demo / output_demo / test_input / test_output / separator positions
  3. Scale usage: per-scale kernel entropy and effective temperature
  4. Accuracy by grid size: does accuracy degrade with larger outputs?
  5. Diffusion kernel structure: who does each test_output cell attend to?
  6. Demo utilization: how much attention goes to demo_output vs demo_input vs test_input?
"""

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fgn.config import FGNConfig
from fgn.model_arc import FluidNetARC
from fgn.model_arc_sandwich import SandwichARC, create_arc_model
from fgn.tasks.arc import ARCTask, ROLE_INPUT_DEMO, ROLE_OUTPUT_DEMO, ROLE_TEST_INPUT, ROLE_TEST_OUTPUT


def probe(model, eval_task, device, n_batches=30, batch_size=4):
    """Run comprehensive diagnostics."""
    model.eval()
    m = model._orig_mod if hasattr(model, '_orig_mod') else model
    is_fluid = isinstance(m, (FluidNetARC, SandwichARC))
    if not is_fluid:
        print("ERROR: Model is not FluidNetARC/SandwichARC, can't probe geometry")
        return

    # Get all fluid layers (handles both pure FluidNet and sandwich)
    if isinstance(m, SandwichARC):
        fluid_layers = list(m.bottom_geo) + list(m.top_geo)
    else:
        fluid_layers = list(m.layers)

    # Accumulators
    stats = {
        # Accuracy by output grid size
        "acc_by_size": defaultdict(lambda: {"correct": 0, "total": 0}),
        # Curvature by role
        "kappa_by_role": defaultdict(list),  # role -> list of mean |kappa| values
        # Grid fidelity
        "fidelity_rhos": [],
        # Scale weights per layer
        "scale_weights": [],  # list of [n_layers, n_scales] arrays
        # Diffusion kernel analysis: what fraction of attention goes to each role
        "kernel_role_fracs": defaultdict(list),  # target role -> dict of source role fracs
        # Per-layer curvature
        "kappa_per_layer": [],
        # Metric values distribution
        "metric_values": [],
        # Per-task accuracy
        "task_accuracies": [],
    }

    with torch.no_grad():
        for batch_i in range(n_batches):
            try:
                _, _, meta = eval_task.generate_batch(batch_size, device=device)
            except RuntimeError:
                continue

            colors = meta["colors"]
            xs = meta["xs"]
            ys = meta["ys"]
            roles = meta["roles"]
            sep_mask = meta["sep_mask"]
            sep_types = meta["sep_types"]
            target_mask = meta["target_mask"]
            target_labels = meta["target_labels"]
            context_mask = meta["context_mask"]
            lengths = meta["lengths"]

            B, N = colors.shape

            # Forward pass — get hidden states and geometry at each layer
            colors_masked = colors.clone()
            target_input_colors = meta.get("target_input_colors")
            if target_input_colors is not None:
                colors_masked[target_mask] = target_input_colors[target_mask]
            else:
                colors_masked[target_mask] = 10  # PAD_COLOR

            h = m.embedding(colors_masked, xs, ys, roles, sep_mask, sep_types)
            h = m.embed_drop(h)
            context = m.context_pool(h, context_mask)

            layer_kappas = []
            layer_metrics = []

            # Run through all layers (sandwich or pure FluidNet)
            if isinstance(m, SandwichARC):
                for layer in m.bottom_geo:
                    h, kappa, m_cv, t_avg = layer(h, context, mask=None)
                    layer_kappas.append(kappa)
                    g = layer.get_current_metric(h, context)
                    layer_metrics.append(g)
                for layer in m.middle_attn:
                    h = layer(h, mask=None)
                for layer in m.top_geo:
                    h, kappa, m_cv, t_avg = layer(h, context, mask=None)
                    layer_kappas.append(kappa)
                    g = layer.get_current_metric(h, context)
                    layer_metrics.append(g)
            else:
                for layer in m.layers:
                    h, kappa, m_cv, t_avg = layer(h, context, mask=None)
                    layer_kappas.append(kappa)
                    g = layer.get_current_metric(h, context)
                    layer_metrics.append(g)

            # Output predictions
            h_normed = m.norm(h)
            logits = m.output_head(h_normed)  # [B, N, 10]
            preds = logits.argmax(dim=-1)

            # Per-batch-item analysis
            for b in range(B):
                n_b = lengths[b].item() if lengths is not None else N
                tgt = target_labels[b]
                valid = tgt != -100
                n_valid = valid.sum().item()
                if n_valid == 0:
                    continue

                # Accuracy for this item
                matches = (preds[b][valid] == tgt[valid])
                item_acc = matches.float().mean().item()
                stats["task_accuracies"].append(item_acc)

                # Accuracy by output grid size
                n_out = n_valid
                size_bucket = (n_out // 25) * 25  # bucket by 25s
                stats["acc_by_size"][size_bucket]["correct"] += matches.sum().item()
                stats["acc_by_size"][size_bucket]["total"] += n_valid

                # Curvature by role (use last layer's kappa)
                kappa_last = layer_kappas[-1][b, :n_b]  # [n_b]
                roles_b = roles[b, :n_b]
                sep_b = sep_mask[b, :n_b]

                for role_id, role_name in [(0, "demo_in"), (1, "demo_out"),
                                            (2, "test_in"), (3, "test_out")]:
                    role_mask = (roles_b == role_id) & (~sep_b)
                    if role_mask.sum() > 0:
                        stats["kappa_by_role"][role_name].append(
                            kappa_last[role_mask].abs().mean().item())

                # Separator curvature
                if sep_b.sum() > 0:
                    stats["kappa_by_role"]["separator"].append(
                        kappa_last[sep_b].abs().mean().item())

                # Per-layer kappa magnitude
                for li, kappa_l in enumerate(layer_kappas):
                    while len(stats["kappa_per_layer"]) <= li:
                        stats["kappa_per_layer"].append([])
                    stats["kappa_per_layer"][li].append(
                        kappa_l[b, :n_b].abs().mean().item())

                # Grid fidelity: geodesic vs Euclidean distances
                g_last = layer_metrics[-1]
                h_b = h[b, :n_b]
                g_b = g_last[b, :n_b]

                # Get non-separator, non-padding positions within a single grid
                valid_pos = (~sep_b).nonzero(as_tuple=True)[0]
                if valid_pos.shape[0] >= 10:
                    # Subsample
                    V = min(valid_pos.shape[0], 100)
                    perm = torch.randperm(valid_pos.shape[0], device=device)[:V]
                    vp = valid_pos[perm]

                    # Euclidean distances
                    gx = xs[b, vp].float()
                    gy = ys[b, vp].float()
                    dx = gx.unsqueeze(1) - gx.unsqueeze(0)
                    dy = gy.unsqueeze(1) - gy.unsqueeze(0)
                    D_euclid = (dx * dx + dy * dy).sqrt()

                    # Geodesic distances
                    h_v = h_b[vp]
                    g_v = g_b[vp]
                    diff = h_v.unsqueeze(1) - h_v.unsqueeze(0)
                    g_avg = (g_v.unsqueeze(1) + g_v.unsqueeze(0)) / 2
                    D_geo = (diff * diff * g_avg).sum(-1).clamp(min=0).sqrt()

                    # Spearman correlation (upper triangle only, exclude diagonal)
                    mask_ut = torch.triu(torch.ones(V, V, device=device), diagonal=1).bool()
                    d_e = D_euclid[mask_ut].cpu().numpy()
                    d_g = D_geo[mask_ut].cpu().numpy()

                    if len(d_e) > 5 and np.std(d_e) > 0 and np.std(d_g) > 0:
                        from scipy.stats import spearmanr
                        rho, _ = spearmanr(d_e, d_g)
                        if not np.isnan(rho):
                            stats["fidelity_rhos"].append(rho)

                # Metric value distribution
                g_vals = g_b[~sep_b].cpu().numpy()
                stats["metric_values"].append({
                    "mean": float(np.mean(g_vals)),
                    "std": float(np.std(g_vals)),
                    "min": float(np.min(g_vals)),
                    "max": float(np.max(g_vals)),
                })

                # Diffusion kernel role analysis:
                # For each test_output position, measure what fraction of
                # diffusion weight goes to demo_in, demo_out, test_in, test_out
                test_out_pos = ((roles_b == ROLE_TEST_OUTPUT) & (~sep_b)).nonzero(as_tuple=True)[0]
                if test_out_pos.shape[0] > 0 and test_out_pos.shape[0] <= 200:
                    # Compute pairwise geodesic "affinity" (negative distance)
                    # as a proxy for diffusion kernel weight
                    h_all = h_b[:n_b]
                    g_all = g_b[:n_b]

                    # Just use a few test_out positions to save compute
                    n_probe = min(test_out_pos.shape[0], 20)
                    probe_pos = test_out_pos[:n_probe]

                    h_probe = h_all[probe_pos]  # [n_probe, d]
                    g_probe = g_all[probe_pos]  # [n_probe, d]

                    # Distance from each probe to all positions
                    diff_all = h_probe.unsqueeze(1) - h_all.unsqueeze(0)  # [n_probe, n_b, d]
                    g_avg_all = (g_probe.unsqueeze(1) + g_all.unsqueeze(0)) / 2
                    D2 = (diff_all * diff_all * g_avg_all).sum(-1)  # [n_probe, n_b]

                    # Use smallest timescale for "attention" approximation
                    t_local = 0.5  # approximate
                    K = torch.exp(-D2 / (4 * t_local + 1e-8))  # [n_probe, n_b]
                    K = K / (K.sum(dim=-1, keepdim=True) + 1e-8)

                    # Average kernel weights by role
                    for role_id, role_name in [(0, "demo_in"), (1, "demo_out"),
                                                (2, "test_in"), (3, "test_out")]:
                        role_positions = (roles_b == role_id) & (~sep_b)
                        if role_positions.sum() > 0:
                            frac = K[:, role_positions[:n_b]].sum(dim=-1).mean().item()
                            stats["kernel_role_fracs"][role_name].append(frac)

                    sep_positions = sep_b[:n_b]
                    if sep_positions.sum() > 0:
                        frac = K[:, sep_positions].sum(dim=-1).mean().item()
                        stats["kernel_role_fracs"]["separator"].append(frac)

            if (batch_i + 1) % 10 == 0:
                print(f"  Probed {batch_i + 1}/{n_batches} batches...")

    # ── Report ──────────────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print(f"  ARC FluidNet Probe Results")
    print(f"{'='*60}")

    # 1. Overall accuracy
    accs = stats["task_accuracies"]
    print(f"\n  Overall: {len(accs)} tasks, mean acc={np.mean(accs):.4f}, "
          f"median={np.median(accs):.4f}")
    # Distribution
    bins = [0, 0.25, 0.5, 0.75, 0.9, 1.0, 1.01]
    hist, _ = np.histogram(accs, bins=bins)
    print(f"  Accuracy distribution:")
    labels = ["0-25%", "25-50%", "50-75%", "75-90%", "90-99%", "100%"]
    for label, count in zip(labels, hist):
        bar = "#" * (count * 40 // max(len(accs), 1))
        print(f"    {label:>7s}: {count:4d} ({100*count/len(accs):5.1f}%) {bar}")

    # 2. Accuracy by grid size
    print(f"\n  Accuracy by output grid size:")
    for size in sorted(stats["acc_by_size"].keys()):
        d = stats["acc_by_size"][size]
        acc = d["correct"] / max(d["total"], 1)
        print(f"    {size:3d}-{size+24:3d} cells: {acc:.4f} "
              f"({d['correct']}/{d['total']})")

    # 3. Grid fidelity
    rhos = stats["fidelity_rhos"]
    if rhos:
        print(f"\n  Grid fidelity (geodesic vs Euclidean ρ):")
        print(f"    mean={np.mean(rhos):.4f}, median={np.median(rhos):.4f}, "
              f"std={np.std(rhos):.4f}")
        print(f"    range=[{np.min(rhos):.4f}, {np.max(rhos):.4f}]")
        print(f"    ρ > 0.3: {sum(1 for r in rhos if r > 0.3)}/{len(rhos)} "
              f"({100*sum(1 for r in rhos if r > 0.3)/len(rhos):.0f}%)")
    else:
        print(f"\n  Grid fidelity: no valid measurements")

    # 4. Curvature by role
    print(f"\n  Curvature |κ| by role (last layer):")
    for role_name in ["demo_in", "demo_out", "test_in", "test_out", "separator"]:
        vals = stats["kappa_by_role"].get(role_name, [])
        if vals:
            print(f"    {role_name:>10s}: mean={np.mean(vals):10.2f}, "
                  f"std={np.std(vals):10.2f}")

    # 5. Per-layer curvature
    print(f"\n  Per-layer |κ|:")
    for li, vals in enumerate(stats["kappa_per_layer"]):
        if vals:
            print(f"    Layer {li}: mean={np.mean(vals):10.2f}, "
                  f"std={np.std(vals):10.2f}")

    # 6. Metric distribution
    if stats["metric_values"]:
        means = [v["mean"] for v in stats["metric_values"]]
        stds = [v["std"] for v in stats["metric_values"]]
        mins = [v["min"] for v in stats["metric_values"]]
        maxs = [v["max"] for v in stats["metric_values"]]
        print(f"\n  Metric g(x) distribution (last layer):")
        print(f"    mean of means: {np.mean(means):.4f}")
        print(f"    mean of stds:  {np.mean(stds):.4f}")
        print(f"    min across all: {np.min(mins):.4f}")
        print(f"    max across all: {np.max(maxs):.4f}")

    # 7. Diffusion kernel role fractions (what test_output attends to)
    print(f"\n  Test output diffusion kernel — attention by source role:")
    total_frac = 0
    for role_name in ["demo_in", "demo_out", "test_in", "test_out", "separator"]:
        vals = stats["kernel_role_fracs"].get(role_name, [])
        if vals:
            mean_frac = np.mean(vals)
            total_frac += mean_frac
            bar = "#" * int(mean_frac * 50)
            print(f"    {role_name:>10s}: {mean_frac:.4f} {bar}")
    if total_frac > 0:
        print(f"    {'total':>10s}: {total_frac:.4f}")

    print(f"\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Probe ARC FluidNet internals")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/arc")
    parser.add_argument("--n_batches", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    config = FGNConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_arc_model(config, device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    step = ckpt.get("step", "?")
    print(f"Loaded checkpoint from step {step}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    eval_task = ARCTask(
        seq_len=config.max_seq_len,
        data_dir=args.data_dir,
        split="eval",
        augment=False,
    )

    probe(model, eval_task, device,
          n_batches=args.n_batches,
          batch_size=args.batch_size)


if __name__ == "__main__":
    main()
