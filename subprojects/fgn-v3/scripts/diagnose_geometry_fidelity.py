"""Geometry Fidelity Diagnostic — Spearman correlation between geodesic and graph distances.

For each episode:
1. Generate a ContinuousWorld and its tokenized description
2. Find token positions for each room's [WORLD] line
3. Run model forward to get h (embeddings) and g (layer 0 metric)
4. Compute geodesic distances between room positions using learned metric
5. Compute all-pairs shortest path distances in the actual world graph
6. Spearman correlate the two distance matrices

This gates whether multi-metric Experiment B is worth building.
"""

import argparse
import heapq
import random
import sys
from pathlib import Path
from scipy import stats

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fgn.config import FGNConfig
from fgn.model_fluid import FluidNetModel
from fgn.flat_model import FlatTransformerModel
from fgn.tasks.continuous_gridworld import (
    ContinuousWorld, generate_goal, render_world_description,
    render_goal_text, DijkstraSolver, render_episode,
)


def load_model(config, checkpoint_path, device):
    if config.model_type == "flat":
        model = FlatTransformerModel(config).to(device)
    elif config.architecture_version == "fluid":
        model = FluidNetModel(config).to(device)
    else:
        model = FlatTransformerModel(config).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    pe_key = "pos_embed.weight"
    if pe_key in state and state[pe_key].shape[0] > config.max_seq_len:
        state[pe_key] = state[pe_key][:config.max_seq_len]
    model.load_state_dict(state, strict=False)
    return model


def all_pairs_shortest_paths(world):
    """Dijkstra from each room to get all-pairs shortest path distances."""
    n = world.n_rooms
    dist_matrix = [[float('inf')] * n for _ in range(n)]

    for src in range(n):
        dist_matrix[src][src] = 0.0
        heap = [(0.0, src)]
        visited = set()

        while heap:
            d, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)
            dist_matrix[src][u] = d

            for v in world.graph.get(u, []):
                if v not in visited:
                    w = world.distances.get((u, v), float('inf'))
                    if d + w < dist_matrix[src][v]:
                        dist_matrix[src][v] = d + w
                        heapq.heappush(heap, (d + w, v))

    return dist_matrix


def find_room_token_positions(tokenizer, episode_text, n_rooms, seq_len):
    """Find the token position of each room's [WORLD] line.

    Returns dict: room_index -> token_position (start of the room's line).
    """
    lines = episode_text.split("\n")
    room_positions = {}
    token_offset = 0

    for line in lines:
        if not line:
            continue
        line_ids = tokenizer.encode(line + "\n", add_special_tokens=False)
        line_len = len(line_ids)

        if line.startswith("[WORLD] Room "):
            # Extract room number
            try:
                room_str = line.split("Room ")[1].split(" ")[0]
                room_idx = int(room_str)
                # Use start of line (matches training's room_token_positions)
                if token_offset < seq_len:
                    room_positions[room_idx] = token_offset
            except (ValueError, IndexError):
                pass

        token_offset += line_len

    return room_positions


def compute_geodesic_distances(model, input_ids, context_mask, room_positions, device):
    """Run model forward and compute distances between room positions.

    Uses LAST layer's processed hidden states. Returns both:
    - metric_dists: geodesic distances using diagonal metric
    - proj_dists: Euclidean distances in projection space (if model has projection head)

    Returns: (metric_dists, proj_dists) — each is dict of (room_i, room_j) -> distance
    """
    m = model._orig_mod if hasattr(model, '_orig_mod') else model

    with torch.no_grad():
        # Get embeddings
        _, N = input_ids.shape
        pos = torch.arange(N, device=device).unsqueeze(0)
        h = m.embed(input_ids) + m.pos_embed(pos)

        # Context (computed once from initial embeddings)
        context = m.context_pool(h, context_mask)

        # Forward through all layers to get processed hidden states
        mask = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)
        for layer in m.layers:
            h, _, _, _ = layer(h, context, mask=mask)

        # Get metric from LAST layer on processed hidden states
        g = m.layers[-1].get_current_metric(h, context)  # [1, N, d]

        # Check for projection head
        has_proj = (hasattr(m, 'structural_energy') and
                    m.structural_energy.proj is not None)

    rooms = sorted(room_positions.keys())
    metric_dists = {}
    proj_dists = {}

    for i, ri in enumerate(rooms):
        for j, rj in enumerate(rooms):
            if i >= j:
                continue
            pi = room_positions[ri]
            pj = room_positions[rj]

            h_i = h[0, pi]  # [d]
            h_j = h[0, pj]  # [d]

            # Metric-based geodesic distance
            g_i = g[0, pi]
            g_j = g[0, pj]
            diff = h_i - h_j
            g_avg = (g_i + g_j) / 2.0
            metric_dists[(ri, rj)] = (diff * diff * g_avg).sum().item()

            # Projected Euclidean distance
            if has_proj:
                with torch.no_grad():
                    z_i = m.structural_energy.proj(h_i)
                    z_j = m.structural_energy.proj(h_j)
                    diff_z = z_i - z_j
                    proj_dists[(ri, rj)] = (diff_z * diff_z).sum().item()

    return metric_dists, proj_dists


def run_diagnostic(model, tokenizer, config, device, n_episodes=100):
    """Run geometry fidelity diagnostic across multiple episodes.

    Returns: (metric_correlations, proj_correlations) — lists of Spearman rho values.
    proj_correlations is empty if model has no projection head.
    """
    seq_len = config.max_seq_len
    metric_correlations = []
    proj_correlations = []
    episodes_used = 0

    for ep in range(n_episodes * 3):  # over-generate since some may fail
        if episodes_used >= n_episodes:
            break

        rng = random.Random()
        n_rooms = rng.randint(10, 15)

        world = ContinuousWorld(
            n_rooms=n_rooms, n_objects=4,
            space_size=100.0, connect_radius=30.0, rng=rng,
        )

        # Generate a goal and plan
        goal = generate_goal(world, rng, min_state_changes=1)
        if goal is None:
            continue

        solver = DijkstraSolver(world, goal)
        solution = solver.solve()
        if solution is None:
            continue

        plan, _ = solution
        if not (4 <= len(plan) <= 10):
            continue

        episode_text, _, _ = render_episode(world, goal, plan)

        # Tokenize — truncate to seq_len (room positions are always near the start)
        token_ids = tokenizer.encode(episode_text, add_special_tokens=False)
        if len(token_ids) > seq_len:
            token_ids = token_ids[:seq_len]

        # Find room positions
        room_positions = find_room_token_positions(tokenizer, episode_text, n_rooms, seq_len)
        if len(room_positions) < 3:
            continue

        # Pad and create tensors
        pad_id = tokenizer.eos_token_id or 0
        padded = token_ids + [pad_id] * (seq_len - len(token_ids))
        input_ids = torch.tensor([padded], dtype=torch.long, device=device)

        # Context mask (True for [WORLD]/[OBJECTS]/[START] lines)
        context_mask_list = [False] * seq_len
        lines = episode_text.split("\n")
        offset = 0
        for line in lines:
            if not line:
                continue
            line_ids = tokenizer.encode(line + "\n", add_special_tokens=False)
            if line.startswith(("[WORLD]", "[OBJECTS]", "[START]")):
                for k in range(offset, min(offset + len(line_ids), seq_len)):
                    context_mask_list[k] = True
            offset += len(line_ids)
        context_mask = torch.tensor([context_mask_list], dtype=torch.bool, device=device)

        # Compute distances (metric-based and projected)
        metric_dists, proj_dists = compute_geodesic_distances(
            model, input_ids, context_mask, room_positions, device,
        )

        # Compute all-pairs graph distances
        graph_dist_matrix = all_pairs_shortest_paths(world)

        # Build paired lists for correlation
        rooms = sorted(room_positions.keys())
        metric_list = []
        proj_list = []
        graph_list = []

        for i, ri in enumerate(rooms):
            for j, rj in enumerate(rooms):
                if i >= j:
                    continue
                if (ri, rj) in metric_dists:
                    gd = graph_dist_matrix[ri][rj]
                    if gd < float('inf'):
                        metric_list.append(metric_dists[(ri, rj)])
                        graph_list.append(gd)
                        if (ri, rj) in proj_dists:
                            proj_list.append(proj_dists[(ri, rj)])

        if len(metric_list) < 3:
            continue

        # Spearman correlations
        rho_metric, _ = stats.spearmanr(metric_list, graph_list)
        metric_correlations.append(rho_metric)

        if len(proj_list) == len(graph_list):
            rho_proj, _ = stats.spearmanr(proj_list, graph_list)
            proj_correlations.append(rho_proj)

        episodes_used += 1

    return metric_correlations, proj_correlations


def main():
    parser = argparse.ArgumentParser(description="Geometry Fidelity Diagnostic")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--baseline_checkpoint", type=str, default=None,
                        help="Optional: also run on baseline (lambda=0) for comparison")
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
    print("  Geometry Fidelity Diagnostic")
    print("  Spearman(distance, graph_dist) for room tokens")
    print("=" * 60)

    # Run on primary checkpoint
    print(f"\nLoading: {args.checkpoint}")
    model = load_model(config, args.checkpoint, device)
    model.eval()

    # Check for projection head
    m = model._orig_mod if hasattr(model, '_orig_mod') else model
    has_proj = (hasattr(m, 'structural_energy') and
                hasattr(m.structural_energy, 'proj') and
                m.structural_energy.proj is not None)
    if has_proj:
        proj = m.structural_energy.proj
        if isinstance(proj, nn.Sequential):
            d_proj = proj[-1].out_features
            print(f"  Projection head: MLP → d_proj={d_proj}")
        else:
            print(f"  Projection head: linear → d_proj={proj.out_features}")
    else:
        print("  Projection head: none (metric-only mode)")

    print(f"Running {args.n_episodes} episodes...")
    metric_corrs, proj_corrs = run_diagnostic(
        model, tokenizer, config, device, n_episodes=args.n_episodes)

    import numpy as np

    def print_corr_stats(name, corrs):
        if not corrs:
            print(f"  {name}: no data")
            return
        arr = np.array(corrs)
        print(f"\n  {name} ({len(corrs)} episodes):")
        print(f"    Mean:   {arr.mean():.4f}")
        print(f"    Median: {np.median(arr):.4f}")
        print(f"    Std:    {arr.std():.4f}")
        print(f"    Min:    {arr.min():.4f}")
        print(f"    Max:    {arr.max():.4f}")
        print(f"    >0:     {(arr > 0).sum()}/{len(arr)} ({(arr > 0).mean()*100:.1f}%)")
        print(f"    >0.3:   {(arr > 0.3).sum()}/{len(arr)} ({(arr > 0.3).mean()*100:.1f}%)")

    print_corr_stats("Metric rho (diagonal geodesic)", metric_corrs)
    if has_proj:
        print_corr_stats("Projected rho (learned projection)", proj_corrs)

    # Optional baseline comparison
    if args.baseline_checkpoint:
        print(f"\nLoading baseline: {args.baseline_checkpoint}")
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None
        model_base = load_model(config, args.baseline_checkpoint, device)
        model_base.eval()

        print(f"Running {args.n_episodes} episodes (baseline)...")
        base_metric, base_proj = run_diagnostic(
            model_base, tokenizer, config, device, n_episodes=args.n_episodes)
        print_corr_stats("Baseline metric rho", base_metric)
        if base_proj:
            print_corr_stats("Baseline projected rho", base_proj)

    print("\n" + "=" * 60)
    print("  Diagnostic Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
