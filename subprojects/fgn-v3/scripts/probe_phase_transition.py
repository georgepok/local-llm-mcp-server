"""Phase Transition Probes — What changed during the geometry reorganization?

Probe 1: Metric Spectrum Analysis
  - Extract diagonal metric g at room token positions
  - Compare distribution of metric values across d dimensions before/after
  - Look for: uniform → sharply peaked (dimensional selectivity)

Probe 4: Curvature Localization
  - Per-position |κ| across the full sequence
  - Categorize tokens by type: [WORLD], [OBJECTS], [START], [GOAL], [ACT], [OBS], other
  - Look for: curvature concentrated at decision-relevant positions vs uniform

Probe 5 (bonus): Geodesic Distance Matrix Eigenspectrum
  - Effective dimensionality of learned geometry between rooms
"""

import argparse
import random
import sys
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fgn.config import FGNConfig
from fgn.model_fluid import FluidNetModel
from fgn.tasks.continuous_gridworld import (
    ContinuousWorld, generate_goal, DijkstraSolver, render_episode,
)


def load_model(config, checkpoint_path, device):
    model = FluidNetModel(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    pe_key = "pos_embed.weight"
    if pe_key in state and state[pe_key].shape[0] > config.max_seq_len:
        state[pe_key] = state[pe_key][:config.max_seq_len]
    model.load_state_dict(state, strict=False)
    return model


def generate_episode(tokenizer, seq_len):
    """Generate a valid CW episode. Returns (input_ids, episode_text, world) or None."""
    rng = random.Random()
    n_rooms = rng.randint(10, 15)
    world = ContinuousWorld(
        n_rooms=n_rooms, n_objects=4,
        space_size=150.0, connect_radius=40.0, rng=rng,
    )
    goal = generate_goal(world, rng, min_state_changes=1)
    if goal is None:
        return None
    solver = DijkstraSolver(world, goal)
    solution = solver.solve()
    if solution is None:
        return None
    plan, _ = solution
    if not (3 <= len(plan) <= 12):
        return None
    episode_text, _, _ = render_episode(world, goal, plan)
    token_ids = tokenizer.encode(episode_text, add_special_tokens=False)
    if len(token_ids) > seq_len:
        token_ids = token_ids[:seq_len]
    pad_id = tokenizer.eos_token_id or 0
    padded = token_ids + [pad_id] * (seq_len - len(token_ids))
    input_ids = torch.tensor([padded], dtype=torch.long)
    return input_ids, episode_text, world, len(token_ids)


def classify_tokens(tokenizer, episode_text, seq_len):
    """Classify each token position by episode section type."""
    labels = ["other"] * seq_len
    lines = episode_text.split("\n")
    offset = 0
    for line in lines:
        if not line:
            continue
        line_ids = tokenizer.encode(line + "\n", add_special_tokens=False)
        line_len = len(line_ids)
        if line.startswith("[WORLD]"):
            tag = "WORLD"
        elif line.startswith("[OBJECTS]"):
            tag = "OBJECTS"
        elif line.startswith("[START]"):
            tag = "START"
        elif line.startswith("[GOAL]"):
            tag = "GOAL"
        elif line.startswith("[ACT]"):
            tag = "ACT"
        elif line.startswith("[OBS]"):
            tag = "OBS"
        else:
            tag = "other"
        for k in range(offset, min(offset + line_len, seq_len)):
            labels[k] = tag
        offset += line_len
    return labels


def find_room_positions(tokenizer, episode_text, seq_len):
    """Find token position of each room's [WORLD] line start."""
    lines = episode_text.split("\n")
    positions = {}
    offset = 0
    for line in lines:
        if not line:
            continue
        line_ids = tokenizer.encode(line + "\n", add_special_tokens=False)
        if line.startswith("[WORLD] Room "):
            try:
                room_idx = int(line.split("Room ")[1].split(" ")[0])
                if offset < seq_len:
                    positions[room_idx] = offset
            except (ValueError, IndexError):
                pass
        offset += len(line_ids)
    return positions


def run_forward(model, input_ids, episode_text, tokenizer, seq_len, device):
    """Run model forward, return per-layer metrics and kappas."""
    m = model._orig_mod if hasattr(model, '_orig_mod') else model
    with torch.no_grad():
        _, N = input_ids.shape
        pos = torch.arange(N, device=device).unsqueeze(0)
        h = m.embed(input_ids.to(device)) + m.pos_embed(pos)
        context_mask_list = [False] * N
        lines = episode_text.split("\n")
        offset = 0
        for line in lines:
            if not line:
                continue
            line_ids = tokenizer.encode(line + "\n", add_special_tokens=False)
            if line.startswith(("[WORLD]", "[OBJECTS]", "[START]")):
                for k in range(offset, min(offset + len(line_ids), N)):
                    context_mask_list[k] = True
            offset += len(line_ids)
        context_mask = torch.tensor([context_mask_list], dtype=torch.bool, device=device)
        context = m.context_pool(h, context_mask)
        mask = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)

        layer_kappas = []
        layer_metrics = []
        for layer in m.layers:
            h, kappa, _, _ = layer(h, context, mask=mask)
            layer_kappas.append(kappa[0].cpu().numpy())  # [N]
            g = layer.get_current_metric(h, context)
            layer_metrics.append(g[0].cpu().numpy())  # [N, d]

    return h, layer_kappas, layer_metrics


def probe_metric_spectrum(layer_metrics, room_positions, n_top=10):
    """Probe 1: Metric spectrum at room positions.

    Returns per-layer stats about metric distribution across dimensions.
    """
    results = []
    for layer_idx, g in enumerate(layer_metrics):
        # g is [N, d] — extract room positions
        rooms = sorted(room_positions.keys())
        if len(rooms) < 3:
            results.append(None)
            continue
        room_g = np.stack([g[room_positions[r]] for r in rooms])  # [R, d]
        # Average metric across rooms
        avg_g = room_g.mean(axis=0)  # [d]

        # Sort dimensions by metric weight (descending)
        sorted_idx = np.argsort(avg_g)[::-1]
        top_vals = avg_g[sorted_idx[:n_top]]
        bottom_vals = avg_g[sorted_idx[-n_top:]]

        # Concentration: fraction of total metric in top-k dims
        total = avg_g.sum()
        top5_frac = avg_g[sorted_idx[:5]].sum() / (total + 1e-8)
        top10_frac = avg_g[sorted_idx[:10]].sum() / (total + 1e-8)

        # Effective dimensionality (entropy-based)
        p = avg_g / (total + 1e-8)
        entropy = -np.sum(p * np.log(p + 1e-10))
        eff_dim = np.exp(entropy)

        results.append({
            "layer": layer_idx,
            "mean_g": avg_g.mean(),
            "std_g": avg_g.std(),
            "max_g": avg_g.max(),
            "min_g": avg_g.min(),
            "top5_frac": top5_frac,
            "top10_frac": top10_frac,
            "eff_dim": eff_dim,
            "top5_vals": top_vals[:5].tolist(),
            "bottom5_vals": bottom_vals[-5:].tolist(),
        })
    return results


def probe_curvature_localization(layer_kappas, token_labels, actual_len):
    """Probe 4: Where is curvature concentrated?

    Returns per-layer, per-section mean |κ|.
    """
    results = []
    sections = ["WORLD", "OBJECTS", "START", "GOAL", "ACT", "OBS"]

    for layer_idx, kappa in enumerate(layer_kappas):
        kappa_abs = np.abs(kappa[:actual_len])
        section_kappa = {}
        for section in sections:
            indices = [i for i, l in enumerate(token_labels[:actual_len]) if l == section]
            if indices:
                section_kappa[section] = float(np.mean(kappa_abs[indices]))
            else:
                section_kappa[section] = 0.0

        results.append({
            "layer": layer_idx,
            "overall_mean": float(kappa_abs.mean()),
            "overall_max": float(kappa_abs.max()),
            "by_section": section_kappa,
        })
    return results


def probe_geodesic_eigenspectrum(layer_metrics, h_final, room_positions):
    """Probe 5: Eigenspectrum of geodesic distance matrix between rooms.

    Returns effective dimensionality and eigenvalue distribution.
    """
    rooms = sorted(room_positions.keys())
    R = len(rooms)
    if R < 4:
        return None

    h_np = h_final[0].cpu().numpy()  # [N, d]

    # Use last layer metric
    g = layer_metrics[-1]  # [N, d]

    # Compute geodesic distance matrix
    D = np.zeros((R, R))
    for i, ri in enumerate(rooms):
        for j, rj in enumerate(rooms):
            if i >= j:
                continue
            pi, pj = room_positions[ri], room_positions[rj]
            diff = h_np[pi] - h_np[pj]
            g_avg = (g[pi] + g[pj]) / 2.0
            D[i, j] = D[j, i] = np.sum(diff ** 2 * g_avg)

    # Eigenvalues of distance matrix (centered)
    H = np.eye(R) - np.ones((R, R)) / R
    B = -0.5 * H @ D @ H  # double-centering for MDS
    eigvals = np.linalg.eigvalsh(B)
    eigvals = np.sort(eigvals)[::-1]  # descending

    # Effective dimensionality from positive eigenvalues
    pos_eigvals = eigvals[eigvals > 0]
    if len(pos_eigvals) == 0:
        return {"eff_dim": 0, "top_eigvals": []}

    total = pos_eigvals.sum()
    p = pos_eigvals / (total + 1e-8)
    eff_dim = np.exp(-np.sum(p * np.log(p + 1e-10)))

    # How many dims capture 90% of variance?
    cumsum = np.cumsum(pos_eigvals) / (total + 1e-8)
    dims_90 = int(np.searchsorted(cumsum, 0.9)) + 1

    return {
        "eff_dim": float(eff_dim),
        "dims_for_90pct": dims_90,
        "n_positive_eigvals": len(pos_eigvals),
        "top5_eigvals": pos_eigvals[:5].tolist(),
        "eigval_ratio_1_2": float(pos_eigvals[0] / (pos_eigvals[1] + 1e-10)) if len(pos_eigvals) > 1 else float('inf'),
    }


def run_probes(model, tokenizer, config, device, n_episodes=20):
    """Run all probes across multiple episodes."""
    seq_len = config.max_seq_len

    all_spectrum = []
    all_curvature = []
    all_eigenspectrum = []

    episodes_used = 0
    for _ in range(n_episodes * 3):
        if episodes_used >= n_episodes:
            break
        result = generate_episode(tokenizer, seq_len)
        if result is None:
            continue
        input_ids, episode_text, world, actual_len = result

        room_positions = find_room_positions(tokenizer, episode_text, seq_len)
        if len(room_positions) < 4:
            continue

        token_labels = classify_tokens(tokenizer, episode_text, seq_len)
        h_final, layer_kappas, layer_metrics = run_forward(
            model, input_ids, episode_text, tokenizer, seq_len, device)

        # Probe 1: Metric spectrum
        spectrum = probe_metric_spectrum(layer_metrics, room_positions)
        all_spectrum.append(spectrum)

        # Probe 4: Curvature localization
        curvature = probe_curvature_localization(layer_kappas, token_labels, actual_len)
        all_curvature.append(curvature)

        # Probe 5: Eigenspectrum
        eigen = probe_geodesic_eigenspectrum(layer_metrics, h_final, room_positions)
        if eigen is not None:
            all_eigenspectrum.append(eigen)

        episodes_used += 1

    return all_spectrum, all_curvature, all_eigenspectrum


def print_results(name, all_spectrum, all_curvature, all_eigenspectrum, n_layers):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # Probe 1: Metric Spectrum (last layer only for brevity, all layers in detail)
    print(f"\n  --- Probe 1: Metric Spectrum (dimensional selectivity) ---")
    for layer_idx in range(n_layers):
        vals = [ep[layer_idx] for ep in all_spectrum if ep[layer_idx] is not None]
        if not vals:
            continue
        mean_eff_dim = np.mean([v["eff_dim"] for v in vals])
        mean_top5 = np.mean([v["top5_frac"] for v in vals])
        mean_top10 = np.mean([v["top10_frac"] for v in vals])
        mean_max = np.mean([v["max_g"] for v in vals])
        mean_min = np.mean([v["min_g"] for v in vals])
        mean_std = np.mean([v["std_g"] for v in vals])
        print(f"  Layer {layer_idx}: eff_dim={mean_eff_dim:.1f}, "
              f"top5={mean_top5:.3f}, top10={mean_top10:.3f}, "
              f"max/min={mean_max:.2f}/{mean_min:.2f}, std={mean_std:.3f}")

    # Probe 4: Curvature Localization (last layer)
    print(f"\n  --- Probe 4: Curvature Localization (mean |κ| by section) ---")
    sections = ["WORLD", "OBJECTS", "START", "GOAL", "ACT", "OBS"]
    for layer_idx in range(n_layers):
        vals = [ep[layer_idx] for ep in all_curvature]
        overall = np.mean([v["overall_mean"] for v in vals])
        section_means = {}
        for s in sections:
            s_vals = [v["by_section"][s] for v in vals if v["by_section"][s] > 0]
            section_means[s] = np.mean(s_vals) if s_vals else 0.0
        section_str = ", ".join(f"{s}={section_means[s]:.1f}" for s in sections)
        print(f"  Layer {layer_idx}: overall={overall:.1f}, {section_str}")

    # Probe 5: Eigenspectrum
    if all_eigenspectrum:
        print(f"\n  --- Probe 5: Geodesic Distance Eigenspectrum ---")
        mean_eff = np.mean([e["eff_dim"] for e in all_eigenspectrum])
        mean_90 = np.mean([e["dims_for_90pct"] for e in all_eigenspectrum])
        mean_ratio = np.mean([e["eigval_ratio_1_2"] for e in all_eigenspectrum])
        print(f"  Effective dim: {mean_eff:.2f}")
        print(f"  Dims for 90% variance: {mean_90:.1f}")
        print(f"  λ1/λ2 ratio: {mean_ratio:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Phase Transition Probes")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--before", type=str, required=True, help="Checkpoint before transition")
    parser.add_argument("--after", type=str, required=True, help="Checkpoint after transition")
    parser.add_argument("--n_episodes", type=int, default=20)
    args = parser.parse_args()

    config = FGNConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import os
    mem_frac = float(os.environ.get("CUDA_MEMORY_FRACTION", "0"))
    if mem_frac > 0 and device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(mem_frac)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print("=" * 60)
    print("  Phase Transition Probes")
    print("  Comparing geometry before and after reorganization")
    print("=" * 60)

    # Before
    print(f"\nLoading BEFORE: {args.before}")
    model_before = load_model(config, args.before, device)
    model_before.eval()
    spec_b, curv_b, eigen_b = run_probes(
        model_before, tokenizer, config, device, n_episodes=args.n_episodes)
    del model_before
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # After
    print(f"\nLoading AFTER: {args.after}")
    model_after = load_model(config, args.after, device)
    model_after.eval()
    spec_a, curv_a, eigen_a = run_probes(
        model_after, tokenizer, config, device, n_episodes=args.n_episodes)
    del model_after

    # Print comparison
    print_results("BEFORE (step_10000)", spec_b, curv_b, eigen_b, config.n_layers)
    print_results("AFTER (step_20000)", spec_a, curv_a, eigen_a, config.n_layers)

    # Delta summary
    print(f"\n{'='*60}")
    print(f"  DELTA SUMMARY")
    print(f"{'='*60}")

    for layer_idx in range(config.n_layers):
        b_vals = [ep[layer_idx] for ep in spec_b if ep[layer_idx] is not None]
        a_vals = [ep[layer_idx] for ep in spec_a if ep[layer_idx] is not None]
        if b_vals and a_vals:
            b_eff = np.mean([v["eff_dim"] for v in b_vals])
            a_eff = np.mean([v["eff_dim"] for v in a_vals])
            b_top5 = np.mean([v["top5_frac"] for v in b_vals])
            a_top5 = np.mean([v["top5_frac"] for v in a_vals])
            print(f"  Layer {layer_idx}: eff_dim {b_eff:.1f} → {a_eff:.1f}, "
                  f"top5_frac {b_top5:.3f} → {a_top5:.3f}")

    if eigen_b and eigen_a:
        b_ed = np.mean([e["eff_dim"] for e in eigen_b])
        a_ed = np.mean([e["eff_dim"] for e in eigen_a])
        b_r = np.mean([e["eigval_ratio_1_2"] for e in eigen_b])
        a_r = np.mean([e["eigval_ratio_1_2"] for e in eigen_a])
        print(f"\n  Eigenspectrum: eff_dim {b_ed:.2f} → {a_ed:.2f}, "
              f"λ1/λ2 {b_r:.2f} → {a_r:.2f}")

    # Curvature localization delta
    print(f"\n  Curvature concentration (last layer):")
    sections = ["WORLD", "GOAL", "ACT", "OBS"]
    b_curv = [ep[-1] for ep in curv_b]
    a_curv = [ep[-1] for ep in curv_a]
    for s in sections:
        b_v = np.mean([v["by_section"][s] for v in b_curv if v["by_section"][s] > 0]) if b_curv else 0
        a_v = np.mean([v["by_section"][s] for v in a_curv if v["by_section"][s] > 0]) if a_curv else 0
        print(f"    {s}: {b_v:.1f} → {a_v:.1f}")

    print(f"\n{'='*60}")
    print(f"  Probes Complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
