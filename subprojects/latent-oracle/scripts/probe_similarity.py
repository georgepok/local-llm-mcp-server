"""Probe oracle similarity matrices for useful ARC structure.

Tests whether Qwen's per-position hidden state similarities encode information
beyond trivial grid membership:
  1. Within-grid vs cross-grid similarity (sanity — should pass trivially)
  2. Same-color vs different-color cell similarity
  3. Input↔output cell correspondence (do matching positions correlate?)
  4. Transform-cell vs copy-cell similarity structure
  5. Spatial distance correlation (does Qwen similarity track grid distance?)

Run on Spark:
    python3 scripts/probe_similarity.py \
        --similarity /workspace/latent-oracle/similarity_matrices.pt \
        --data_dir /workspace/fgn-v3/data/arc
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_arc_tasks(data_dir: str):
    """Load raw ARC tasks from JSON."""
    tasks = {}
    for split in ["training", "evaluation"]:
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for fn in os.listdir(split_dir):
            if not fn.endswith(".json"):
                continue
            tid = fn.replace(".json", "")
            with open(os.path.join(split_dir, fn)) as f:
                tasks[tid] = json.load(f)
    return tasks


def get_cell_colors(task: dict, test_idx: int = 0):
    """Extract color for each cell in all grids of a task.

    Returns dict: (row, col, grid_id) → color_value
    """
    cell_colors = {}
    grid_id = 0

    # Training examples: pairs of (input, output)
    for pair in task["train"]:
        for key in ["input", "output"]:
            grid = pair[key]
            for r, row in enumerate(grid):
                for c, val in enumerate(row):
                    cell_colors[(r, c, grid_id)] = val
            grid_id += 1

    # Test: input only (output may not exist in eval)
    if test_idx < len(task.get("test", [])):
        test_pair = task["test"][test_idx]
        grid = test_pair["input"]
        for r, row in enumerate(grid):
            for c, val in enumerate(row):
                cell_colors[(r, c, grid_id)] = val
        grid_id += 1

        # Test output if available
        if "output" in test_pair:
            grid = test_pair["output"]
            for r, row in enumerate(grid):
                for c, val in enumerate(row):
                    cell_colors[(r, c, grid_id)] = val
            grid_id += 1

    return cell_colors


def get_transform_mask(task: dict, test_idx: int = 0):
    """Identify which cells changed between input→output in training pairs.

    Returns dict: (row, col, grid_id_output) → bool (True if cell differs from input)
    """
    changed = {}
    grid_id = 0

    for pair in task["train"]:
        inp = pair["input"]
        out = pair["output"]
        grid_id += 1  # skip input grid
        # Only compare if same dimensions
        if len(inp) == len(out) and all(len(inp[r]) == len(out[r]) for r in range(len(inp))):
            for r in range(len(out)):
                for c in range(len(out[r])):
                    inp_val = inp[r][c] if r < len(inp) and c < len(inp[r]) else -1
                    changed[(r, c, grid_id)] = (out[r][c] != inp_val)
        grid_id += 1

    return changed


def probe_1_within_vs_cross_grid(sim_matrices, cell_coords_list):
    """Test 1: Within-grid similarity > cross-grid similarity."""
    within_sims = []
    cross_sims = []

    for idx in range(len(sim_matrices)):
        sim = sim_matrices[idx].numpy()
        coords = cell_coords_list[idx]
        n = len(coords)

        grid_ids = np.array([c[2] for c in coords])

        for i in range(n):
            for j in range(i + 1, n):
                s = sim[i, j]
                if grid_ids[i] == grid_ids[j]:
                    within_sims.append(s)
                else:
                    cross_sims.append(s)

    within_mean = np.mean(within_sims) if within_sims else 0
    cross_mean = np.mean(cross_sims) if cross_sims else 0
    return {
        "within_grid_mean": float(within_mean),
        "cross_grid_mean": float(cross_mean),
        "delta": float(within_mean - cross_mean),
        "n_within": len(within_sims),
        "n_cross": len(cross_sims),
    }


def probe_2_same_vs_diff_color(sim_matrices, cell_coords_list, tasks, task_ids, test_indices):
    """Test 2: Same-color cells have higher similarity than different-color cells."""
    same_color_sims = []
    diff_color_sims = []

    for idx in range(len(sim_matrices)):
        tid = task_ids[idx]
        if tid not in tasks:
            continue
        test_idx = int(test_indices[idx])
        cell_colors = get_cell_colors(tasks[tid], test_idx)

        sim = sim_matrices[idx].numpy()
        coords = cell_coords_list[idx]
        n = len(coords)

        grid_ids = np.array([c[2] for c in coords])

        for i in range(n):
            ci = tuple(coords[i])
            if ci not in cell_colors:
                continue
            for j in range(i + 1, n):
                # Only compare within same grid
                if grid_ids[i] != grid_ids[j]:
                    continue
                cj = tuple(coords[j])
                if cj not in cell_colors:
                    continue
                s = sim[i, j]
                if cell_colors[ci] == cell_colors[cj]:
                    same_color_sims.append(s)
                else:
                    diff_color_sims.append(s)

    same_mean = np.mean(same_color_sims) if same_color_sims else 0
    diff_mean = np.mean(diff_color_sims) if diff_color_sims else 0
    return {
        "same_color_mean": float(same_mean),
        "diff_color_mean": float(diff_mean),
        "delta": float(same_mean - diff_mean),
        "n_same": len(same_color_sims),
        "n_diff": len(diff_color_sims),
    }


def probe_3_io_correspondence(sim_matrices, cell_coords_list, tasks, task_ids, test_indices):
    """Test 3: Input cell (r,c) has higher similarity with corresponding output cell (r,c)
    than with random output cells."""
    corresponding_sims = []
    random_sims = []

    for idx in range(len(sim_matrices)):
        tid = task_ids[idx]
        if tid not in tasks:
            continue

        sim = sim_matrices[idx].numpy()
        coords = cell_coords_list[idx]
        n = len(coords)

        # Build coord→index map
        coord_to_idx = {tuple(coords[i]): i for i in range(n)}

        task = tasks[tid]
        grid_id = 0
        for pair in task["train"]:
            inp = pair["input"]
            out = pair["output"]
            in_gid = grid_id
            out_gid = grid_id + 1
            grid_id += 2

            if len(inp) != len(out):
                continue
            if any(len(inp[r]) != len(out[r]) for r in range(len(inp))):
                continue

            # Corresponding (r,c) pairs between input and output
            out_indices = []
            for r in range(len(out)):
                for c in range(len(out[r])):
                    out_key = (r, c, out_gid)
                    if out_key in coord_to_idx:
                        out_indices.append(coord_to_idx[out_key])

            for r in range(len(inp)):
                for c in range(len(inp[r])):
                    in_key = (r, c, in_gid)
                    out_key = (r, c, out_gid)
                    if in_key not in coord_to_idx or out_key not in coord_to_idx:
                        continue
                    ii = coord_to_idx[in_key]
                    oi = coord_to_idx[out_key]
                    corresponding_sims.append(sim[ii, oi])

                    # Random output cell (not at same position)
                    for oj in out_indices:
                        if oj != oi:
                            random_sims.append(sim[ii, oj])

    corr_mean = np.mean(corresponding_sims) if corresponding_sims else 0
    rand_mean = np.mean(random_sims) if random_sims else 0
    return {
        "corresponding_io_mean": float(corr_mean),
        "random_io_mean": float(rand_mean),
        "delta": float(corr_mean - rand_mean),
        "n_corresponding": len(corresponding_sims),
        "n_random": len(random_sims),
    }


def probe_4_transform_vs_copy(sim_matrices, cell_coords_list, tasks, task_ids, test_indices):
    """Test 4: Do transformed cells cluster differently than copied cells?

    Within output grids, check if transform-cell↔transform-cell similarity differs
    from copy-cell↔copy-cell similarity."""
    xform_xform = []
    copy_copy = []
    xform_copy = []

    for idx in range(len(sim_matrices)):
        tid = task_ids[idx]
        if tid not in tasks:
            continue

        sim = sim_matrices[idx].numpy()
        coords = cell_coords_list[idx]
        changed = get_transform_mask(tasks[tid])

        n = len(coords)
        coord_to_idx = {tuple(coords[i]): i for i in range(n)}

        for i in range(len(coords)):
            ci = tuple(coords[i])
            if ci not in changed:
                continue
            for j in range(i + 1, len(coords)):
                cj = tuple(coords[j])
                if cj not in changed:
                    continue
                # Same output grid only
                if ci[2] != cj[2]:
                    continue
                s = sim[i, j]
                if changed[ci] and changed[cj]:
                    xform_xform.append(s)
                elif not changed[ci] and not changed[cj]:
                    copy_copy.append(s)
                else:
                    xform_copy.append(s)

    return {
        "xform_xform_mean": float(np.mean(xform_xform)) if xform_xform else 0,
        "copy_copy_mean": float(np.mean(copy_copy)) if copy_copy else 0,
        "xform_copy_mean": float(np.mean(xform_copy)) if xform_copy else 0,
        "n_xform_xform": len(xform_xform),
        "n_copy_copy": len(copy_copy),
        "n_xform_copy": len(xform_copy),
    }


def probe_5_spatial_distance(sim_matrices, cell_coords_list):
    """Test 5: Does similarity correlate with Manhattan distance within grids?

    Negative correlation = Qwen embeds spatial locality."""
    all_dists = []
    all_sims = []

    for idx in range(len(sim_matrices)):
        sim = sim_matrices[idx].numpy()
        coords = cell_coords_list[idx]
        n = len(coords)

        grid_ids = np.array([c[2] for c in coords])

        for i in range(n):
            for j in range(i + 1, n):
                if grid_ids[i] != grid_ids[j]:
                    continue
                ri, ci_coord, _ = coords[i]
                rj, cj_coord, _ = coords[j]
                dist = abs(ri - rj) + abs(ci_coord - cj_coord)
                all_dists.append(dist)
                all_sims.append(sim[i, j])

    if len(all_dists) < 10:
        return {"correlation": 0.0, "n_pairs": 0}

    corr = np.corrcoef(all_dists, all_sims)[0, 1]
    return {
        "spatial_distance_corr": float(corr),
        "n_pairs": len(all_dists),
        "mean_dist": float(np.mean(all_dists)),
        "mean_sim": float(np.mean(all_sims)),
    }


def probe_6_background_vs_object(sim_matrices, cell_coords_list, tasks, task_ids, test_indices):
    """Test 6: Background (color=0) cells vs non-background cells.

    In ARC, color 0 is typically background. Do object cells cluster together
    more than background cells?"""
    obj_obj = []
    bg_bg = []
    obj_bg = []

    for idx in range(len(sim_matrices)):
        tid = task_ids[idx]
        if tid not in tasks:
            continue
        test_idx = int(test_indices[idx])
        cell_colors = get_cell_colors(tasks[tid], test_idx)

        sim = sim_matrices[idx].numpy()
        coords = cell_coords_list[idx]
        n = len(coords)
        grid_ids = np.array([c[2] for c in coords])

        for i in range(n):
            ci = tuple(coords[i])
            if ci not in cell_colors:
                continue
            for j in range(i + 1, n):
                if grid_ids[i] != grid_ids[j]:
                    continue
                cj = tuple(coords[j])
                if cj not in cell_colors:
                    continue
                s = sim[i, j]
                i_bg = (cell_colors[ci] == 0)
                j_bg = (cell_colors[cj] == 0)
                if not i_bg and not j_bg:
                    obj_obj.append(s)
                elif i_bg and j_bg:
                    bg_bg.append(s)
                else:
                    obj_bg.append(s)

    return {
        "object_object_mean": float(np.mean(obj_obj)) if obj_obj else 0,
        "bg_bg_mean": float(np.mean(bg_bg)) if bg_bg else 0,
        "object_bg_mean": float(np.mean(obj_bg)) if obj_bg else 0,
        "n_obj_obj": len(obj_obj),
        "n_bg_bg": len(bg_bg),
        "n_obj_bg": len(obj_bg),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--similarity", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--max_tasks", type=int, default=200,
                        help="Max tasks to probe (for speed)")
    args = parser.parse_args()

    print("Loading similarity matrices...")
    data = torch.load(args.similarity, map_location="cpu", weights_only=False)
    sim_matrices = data["similarities"]
    cell_coords = data["cell_coords"]
    task_ids = data["task_ids"]
    test_indices = data["test_indices"]
    print(f"  {len(sim_matrices)} matrices, layer={data['layer_idx']}, mode={data['mode']}")

    print("Loading ARC tasks...")
    tasks = load_arc_tasks(args.data_dir)
    print(f"  {len(tasks)} tasks loaded")

    # Subsample for speed
    n = min(args.max_tasks, len(sim_matrices))
    indices = list(range(n))
    sub_sims = [sim_matrices[i] for i in indices]
    sub_coords = [cell_coords[i] for i in indices]
    sub_tids = [task_ids[i] for i in indices]
    sub_test = [test_indices[i] for i in indices]

    print(f"\nProbing {n} tasks...")
    print("=" * 70)

    # Probe 1
    print("\n[Probe 1] Within-grid vs cross-grid similarity (sanity)")
    r1 = probe_1_within_vs_cross_grid(sub_sims, sub_coords)
    print(f"  Within-grid:  {r1['within_grid_mean']:.4f} ({r1['n_within']:,} pairs)")
    print(f"  Cross-grid:   {r1['cross_grid_mean']:.4f} ({r1['n_cross']:,} pairs)")
    print(f"  Delta:        {r1['delta']:+.4f}")
    verdict = "PASS" if r1["delta"] > 0.01 else "FAIL"
    print(f"  Verdict: {verdict}")

    # Probe 2
    print("\n[Probe 2] Same-color vs different-color similarity")
    r2 = probe_2_same_vs_diff_color(sub_sims, sub_coords, tasks, sub_tids, sub_test)
    print(f"  Same color:   {r2['same_color_mean']:.4f} ({r2['n_same']:,} pairs)")
    print(f"  Diff color:   {r2['diff_color_mean']:.4f} ({r2['n_diff']:,} pairs)")
    print(f"  Delta:        {r2['delta']:+.4f}")
    verdict = "PASS" if r2["delta"] > 0.005 else "MARGINAL" if r2["delta"] > 0 else "FAIL"
    print(f"  Verdict: {verdict}")

    # Probe 3
    print("\n[Probe 3] Input↔output positional correspondence")
    r3 = probe_3_io_correspondence(sub_sims, sub_coords, tasks, sub_tids, sub_test)
    print(f"  Corresponding (r,c): {r3['corresponding_io_mean']:.4f} ({r3['n_corresponding']:,} pairs)")
    print(f"  Random output:       {r3['random_io_mean']:.4f} ({r3['n_random']:,} pairs)")
    print(f"  Delta:               {r3['delta']:+.4f}")
    verdict = "PASS" if r3["delta"] > 0.005 else "MARGINAL" if r3["delta"] > 0 else "FAIL"
    print(f"  Verdict: {verdict}")

    # Probe 4
    print("\n[Probe 4] Transform-cell vs copy-cell clustering")
    r4 = probe_4_transform_vs_copy(sub_sims, sub_coords, tasks, sub_tids, sub_test)
    print(f"  xform↔xform:  {r4['xform_xform_mean']:.4f} ({r4['n_xform_xform']:,} pairs)")
    print(f"  copy↔copy:    {r4['copy_copy_mean']:.4f} ({r4['n_copy_copy']:,} pairs)")
    print(f"  xform↔copy:   {r4['xform_copy_mean']:.4f} ({r4['n_xform_copy']:,} pairs)")
    # Signal: xform-xform and copy-copy should both be > xform-copy
    if r4["n_xform_xform"] > 0 and r4["n_copy_copy"] > 0 and r4["n_xform_copy"] > 0:
        cluster_signal = (min(r4["xform_xform_mean"], r4["copy_copy_mean"])
                         - r4["xform_copy_mean"])
        print(f"  Cluster signal: {cluster_signal:+.4f}")
        verdict = "PASS" if cluster_signal > 0.005 else "MARGINAL" if cluster_signal > 0 else "FAIL"
    else:
        verdict = "INSUFFICIENT DATA"
    print(f"  Verdict: {verdict}")

    # Probe 5
    print("\n[Probe 5] Spatial distance correlation")
    r5 = probe_5_spatial_distance(sub_sims, sub_coords)
    print(f"  Pearson r(manhattan_dist, similarity): {r5['spatial_distance_corr']:.4f}")
    print(f"  ({r5['n_pairs']:,} pairs)")
    # Negative correlation = spatial structure preserved
    verdict = "PASS" if r5["spatial_distance_corr"] < -0.05 else "MARGINAL" if r5["spatial_distance_corr"] < 0 else "FAIL"
    print(f"  Verdict: {verdict} (negative = spatial locality preserved)")

    # Probe 6
    print("\n[Probe 6] Background vs object cell similarity")
    r6 = probe_6_background_vs_object(sub_sims, sub_coords, tasks, sub_tids, sub_test)
    print(f"  Object↔object: {r6['object_object_mean']:.4f} ({r6['n_obj_obj']:,} pairs)")
    print(f"  Bg↔bg:         {r6['bg_bg_mean']:.4f} ({r6['n_bg_bg']:,} pairs)")
    print(f"  Object↔bg:     {r6['object_bg_mean']:.4f} ({r6['n_obj_bg']:,} pairs)")
    if r6["n_obj_obj"] > 0 and r6["n_bg_bg"] > 0:
        # Object cells should cluster together (higher sim) vs cross-category
        obj_cluster = r6["object_object_mean"] - r6["object_bg_mean"]
        bg_cluster = r6["bg_bg_mean"] - r6["object_bg_mean"]
        print(f"  Object clustering: {obj_cluster:+.4f}")
        print(f"  Bg clustering:     {bg_cluster:+.4f}")
        verdict = "PASS" if obj_cluster > 0.005 or bg_cluster > 0.005 else "FAIL"
    else:
        verdict = "INSUFFICIENT DATA"
    print(f"  Verdict: {verdict}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    results = [r1, r2, r3, r4, r5, r6]
    n_pass = sum(1 for r in [r1, r2, r3, r5, r6]
                 if (r.get("delta", 0) > 0.005 or
                     r.get("spatial_distance_corr", 0) < -0.05))
    print(f"Probes with positive signal: {n_pass}/5")
    if n_pass >= 3:
        print("RECOMMENDATION: Proceed with distillation training")
    elif n_pass >= 1:
        print("RECOMMENDATION: Marginal signal — proceed with caution, short run first")
    else:
        print("RECOMMENDATION: No useful signal — reconsider approach")


if __name__ == "__main__":
    main()
