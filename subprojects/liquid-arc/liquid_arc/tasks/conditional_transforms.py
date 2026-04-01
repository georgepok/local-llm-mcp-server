"""Conditional spatial transform task generator.

Rules that require per-cell conditional logic based on spatial context.
Each rule cannot be solved by inspecting a single cell in isolation — the
correct output color depends on the cell's neighborhood, row, or the
relationship between adjacent regions across the full grid.

Four rule types:
  majority_neighbor  — each non-bg cell takes the majority color among its
                       4-connected neighbors; ties keep original color
  border_mark        — cells of color C adjacent to background become color M;
                       interior cells of color C keep their color
  color_spread       — non-bg cells spread one step into adjacent bg cells;
                       conflicts resolved by lowest color index
  row_majority       — each cell becomes the most common non-bg color in its
                       row; empty or tied rows stay as-is

Compatible with ProceduralARCTask.generate_batch() interface.
"""

import random
from typing import Dict, List, Optional, Tuple

import torch

from .procedural import (
    PAD_COLOR,
    PAD_COORD,
    N_COLORS,
    build_sequence,
    _augment_grid,
    _permute_colors,
)

# ────────────────────────────────────────────────────────────────────
# Grid utilities (local copies — keep file self-contained)
# ────────────────────────────────────────────────────────────────────

def _empty_grid(H: int, W: int, bg: int = 0) -> List[List[int]]:
    return [[bg] * W for _ in range(H)]


def _copy_grid(grid: List[List[int]]) -> List[List[int]]:
    return [row[:] for row in grid]


def _rand_dims(min_d: int = 5, max_d: int = 10) -> Tuple[int, int]:
    return random.randint(min_d, max_d), random.randint(min_d, max_d)


def _rand_bg() -> int:
    return random.randint(0, N_COLORS - 1)


def _rand_palette(n: int, exclude: int = -1) -> List[int]:
    pool = [c for c in range(N_COLORS) if c != exclude]
    random.shuffle(pool)
    return pool[:n]


def _rand_grid_cond(
    H: int,
    W: int,
    bg: int,
    palette: List[int],
    density: float,
) -> List[List[int]]:
    """Random grid: each non-bg cell takes a color from palette with probability density."""
    grid = _empty_grid(H, W, bg)
    for y in range(H):
        for x in range(W):
            if random.random() < density:
                grid[y][x] = random.choice(palette)
    return grid


# ────────────────────────────────────────────────────────────────────
# Rule implementations
# ────────────────────────────────────────────────────────────────────

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right


def _apply_majority_neighbor(
    grid: List[List[int]], bg: int
) -> List[List[int]]:
    """Each non-bg cell becomes the majority color among its 4-connected neighbors.

    Neighbors that are bg or out-of-bounds are ignored when counting.
    If no non-bg neighbors exist, or a tie occurs, the original color is kept.
    The update is computed from the *original* grid (simultaneous, not in-place).
    """
    H, W = len(grid), len(grid[0])
    out = _copy_grid(grid)
    for y in range(H):
        for x in range(W):
            if grid[y][x] == bg:
                continue
            counts: Dict[int, int] = {}
            for dy, dx in _DIRS4:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    nc = grid[ny][nx]
                    if nc != bg:
                        counts[nc] = counts.get(nc, 0) + 1
            if not counts:
                continue
            max_count = max(counts.values())
            candidates = [c for c, cnt in counts.items() if cnt == max_count]
            if len(candidates) == 1:
                out[y][x] = candidates[0]
            # else: tie — keep original color (out[y][x] already == grid[y][x])
    return out


def _apply_border_mark(
    grid: List[List[int]], bg: int, target_color: int, marker_color: int
) -> List[List[int]]:
    """Cells of target_color that are 4-adjacent to bg become marker_color.

    Interior cells of target_color (no bg neighbor) keep their color.
    All other cells are unchanged.
    """
    H, W = len(grid), len(grid[0])
    out = _copy_grid(grid)
    for y in range(H):
        for x in range(W):
            if grid[y][x] != target_color:
                continue
            # Check if any 4-neighbor is bg
            adjacent_to_bg = any(
                (0 <= y + dy < H and 0 <= x + dx < W and grid[y + dy][x + dx] == bg)
                or not (0 <= y + dy < H and 0 <= x + dx < W)  # grid edge counts as bg
                for dy, dx in _DIRS4
            )
            if adjacent_to_bg:
                out[y][x] = marker_color
    return out


def _apply_color_spread(
    grid: List[List[int]], bg: int
) -> List[List[int]]:
    """Non-bg cells spread one step into adjacent bg cells.

    If multiple non-bg colors could spread into the same bg cell, the
    lowest color index wins. Non-bg cells themselves are unchanged.
    """
    H, W = len(grid), len(grid[0])
    out = _copy_grid(grid)
    # Collect candidate fills: target cell → minimum spreading color
    candidates: Dict[Tuple[int, int], int] = {}
    for y in range(H):
        for x in range(W):
            if grid[y][x] == bg:
                continue
            src_color = grid[y][x]
            for dy, dx in _DIRS4:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and grid[ny][nx] == bg:
                    key = (ny, nx)
                    if key not in candidates or src_color < candidates[key]:
                        candidates[key] = src_color
    for (ny, nx), color in candidates.items():
        out[ny][nx] = color
    return out


def _apply_row_majority(
    grid: List[List[int]], bg: int
) -> List[List[int]]:
    """Each cell becomes the most common non-bg color in its row.

    If a row has no non-bg cells, or the non-bg counts are tied, all
    cells in that row stay as-is.
    """
    H, W = len(grid), len(grid[0])
    out = _copy_grid(grid)
    for y in range(H):
        counts: Dict[int, int] = {}
        for x in range(W):
            c = grid[y][x]
            if c != bg:
                counts[c] = counts.get(c, 0) + 1
        if not counts:
            continue
        max_count = max(counts.values())
        winners = [c for c, cnt in counts.items() if cnt == max_count]
        if len(winners) != 1:
            continue  # tie — row unchanged
        majority = winners[0]
        for x in range(W):
            out[y][x] = majority
    return out


# ────────────────────────────────────────────────────────────────────
# Per-rule grid generators
# ────────────────────────────────────────────────────────────────────

def _gen_majority_neighbor() -> Tuple[List[List[int]], List[List[int]]]:
    """Generate an (input, output) pair for the majority_neighbor rule."""
    H, W = _rand_dims(5, 10)
    bg = _rand_bg()
    n_colors = random.randint(2, 4)
    palette = _rand_palette(n_colors, exclude=bg)
    density = random.uniform(0.2, 0.5)
    grid = _rand_grid_cond(H, W, bg, palette, density)
    out = _apply_majority_neighbor(grid, bg)
    return grid, out


def _gen_border_mark() -> Tuple[List[List[int]], List[List[int]]]:
    """Generate an (input, output) pair for the border_mark rule.

    Picks a random target color C and a distinct marker color M.
    Guarantees at least one C cell exists with a bg neighbor (border cell)
    and at least one C cell without a bg neighbor (interior cell) so the
    rule is non-trivial.
    """
    for _ in range(50):  # retry until a valid grid is found
        H, W = _rand_dims(5, 10)
        bg = _rand_bg()
        # Need at least 2 non-bg colors: target_color and marker_color
        palette = _rand_palette(4, exclude=bg)
        target_color = palette[0]
        marker_color = palette[1]
        # Filler colors for variety (may be absent)
        filler = palette[2:]

        density = random.uniform(0.25, 0.5)
        full_palette = [target_color] + filler
        grid = _rand_grid_cond(H, W, bg, full_palette, density)

        # Check preconditions: at least one border C cell and one interior C cell
        has_border = False
        has_interior = False
        for y in range(H):
            for x in range(W):
                if grid[y][x] != target_color:
                    continue
                adjacent_to_bg = any(
                    not (0 <= y + dy < H and 0 <= x + dx < W)
                    or grid[y + dy][x + dx] == bg
                    for dy, dx in _DIRS4
                )
                if adjacent_to_bg:
                    has_border = True
                else:
                    has_interior = True
                if has_border and has_interior:
                    break
            if has_border and has_interior:
                break

        if has_border and has_interior:
            out = _apply_border_mark(grid, bg, target_color, marker_color)
            return grid, out

    # Fallback: construct a minimal valid grid manually
    H, W = 5, 5
    bg = 0
    target_color = 1
    marker_color = 2
    grid = _empty_grid(H, W, bg)
    # Solid 3×3 block of target_color — corners/edges are border, center is interior
    for y in range(1, 4):
        for x in range(1, 4):
            grid[y][x] = target_color
    out = _apply_border_mark(grid, bg, target_color, marker_color)
    return grid, out


def _gen_color_spread() -> Tuple[List[List[int]], List[List[int]]]:
    """Generate an (input, output) pair for the color_spread rule.

    Ensures at least one non-bg cell has a bg neighbor so that the rule
    is non-trivial (input != output).
    """
    for _ in range(50):
        H, W = _rand_dims(5, 10)
        bg = _rand_bg()
        n_colors = random.randint(2, 4)
        palette = _rand_palette(n_colors, exclude=bg)
        density = random.uniform(0.2, 0.45)
        grid = _rand_grid_cond(H, W, bg, palette, density)

        # Need at least one non-bg cell adjacent to a bg cell
        has_spread = any(
            grid[y][x] != bg and any(
                0 <= y + dy < H and 0 <= x + dx < W and grid[y + dy][x + dx] == bg
                for dy, dx in _DIRS4
            )
            for y in range(H) for x in range(W)
        )
        if has_spread:
            out = _apply_color_spread(grid, bg)
            return grid, out

    # Fallback: single non-bg cell surrounded by bg
    H, W = 5, 5
    bg = 0
    grid = _empty_grid(H, W, bg)
    grid[2][2] = 1
    out = _apply_color_spread(grid, bg)
    return grid, out


def _gen_row_majority() -> Tuple[List[List[int]], List[List[int]]]:
    """Generate an (input, output) pair for the row_majority rule.

    Ensures at least one row has a clear non-bg majority so that at
    least one cell changes (input != output).
    """
    for _ in range(50):
        H, W = _rand_dims(5, 10)
        bg = _rand_bg()
        n_colors = random.randint(2, 4)
        palette = _rand_palette(n_colors, exclude=bg)
        density = random.uniform(0.25, 0.5)
        grid = _rand_grid_cond(H, W, bg, palette, density)

        # Check that at least one row will change
        changed = False
        for y in range(H):
            counts: Dict[int, int] = {}
            for x in range(W):
                c = grid[y][x]
                if c != bg:
                    counts[c] = counts.get(c, 0) + 1
            if not counts:
                continue
            max_count = max(counts.values())
            winners = [c for c, cnt in counts.items() if cnt == max_count]
            if len(winners) == 1:
                majority = winners[0]
                # Will any cell in this row actually change?
                if any(grid[y][x] != majority for x in range(W)):
                    changed = True
                    break

        if changed:
            out = _apply_row_majority(grid, bg)
            return grid, out

    # Fallback: single row with two bg cells and one non-bg, rest non-bg dominant
    H, W = 5, 5
    bg = 0
    grid = _empty_grid(H, W, bg)
    # Row 2: mostly color 1, one bg cell → output row becomes all-1
    for x in range(W):
        grid[2][x] = 1
    grid[2][0] = bg  # one bg cell that will change to 1
    out = _apply_row_majority(grid, bg)
    return grid, out


# Maps rule name → generator function
_RULE_GENERATORS = {
    "majority_neighbor": _gen_majority_neighbor,
    "border_mark": _gen_border_mark,
    "color_spread": _gen_color_spread,
    "row_majority": _gen_row_majority,
}

_RULE_NAMES = list(_RULE_GENERATORS.keys())


# ────────────────────────────────────────────────────────────────────
# Main task class
# ────────────────────────────────────────────────────────────────────

class ConditionalTransformTask:
    """Infinite conditional spatial transform task generator.

    Drop-in replacement for ProceduralARCTask — same generate_batch() interface.

    Each task instance picks one of four conditional rules and generates
    n_demos demonstration pairs plus one test pair, all drawn from the
    same rule but different random grids. The rule stays constant within
    a task; only the grid contents vary, forcing the model to learn the
    underlying conditional logic rather than memorizing specific grids.

    Args:
        seq_len:  Maximum sequence length; sequences are padded to this.
        augment:  Apply D4 symmetry augmentation and color permutation.
        n_demos:  Number of demonstration pairs per task instance.
    """

    def __init__(
        self,
        seq_len: int = 2048,
        augment: bool = True,
        n_demos: int = 2,
        **kwargs,
    ) -> None:
        self.seq_len = seq_len
        self.augment = augment
        self.n_demos = n_demos

    def _generate_one(self) -> Dict:
        """Generate a single task instance (n_demos + 1 test) and serialize.

        Returns a dict with the same keys as build_sequence(), plus a
        truncation step that mirrors ProceduralARCTask._generate_one().
        """
        rule_name = random.choice(_RULE_NAMES)
        gen_fn = _RULE_GENERATORS[rule_name]

        # Augmentation params are shared across all pairs in this task so
        # the model cannot distinguish demos from test by transformation.
        if self.augment:
            rot = random.randint(0, 3)
            flip = random.random() < 0.5
            perm_vals = list(range(1, N_COLORS))
            random.shuffle(perm_vals)
            color_perm = {0: 0}
            for i, v in enumerate(perm_vals):
                color_perm[i + 1] = v
        else:
            rot = 0
            flip = False
            color_perm = {i: i for i in range(N_COLORS)}

        demos: List[Tuple[List[List[int]], List[List[int]]]] = []
        for _ in range(self.n_demos):
            inp, out = gen_fn()
            inp = _permute_colors(_augment_grid(inp, rot, flip), color_perm)
            out = _permute_colors(_augment_grid(out, rot, flip), color_perm)
            demos.append((inp, out))

        test_inp, test_out = gen_fn()
        test_inp = _permute_colors(_augment_grid(test_inp, rot, flip), color_perm)
        test_out = _permute_colors(_augment_grid(test_out, rot, flip), color_perm)

        seq = build_sequence(demos, test_inp, test_out)

        # Truncation strategy mirrors ProceduralARCTask._generate_one()
        if seq["length"] > self.seq_len:
            if self.n_demos > 1:
                demos = demos[:1]
                seq = build_sequence(demos, test_inp, test_out)
            if seq["length"] > self.seq_len:
                for key in [
                    "colors", "xs", "ys", "roles", "sep_mask", "sep_types",
                    "grid_ids", "target_mask", "target_input_colors",
                ]:
                    seq[key] = seq[key][:self.seq_len]
                seq["length"] = self.seq_len

        return seq

    def generate_batch(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Generate a batch of conditional transform tasks.

        Returns the same format as ProceduralARCTask.generate_batch():
            (input_ids, labels, metadata)

        input_ids and labels are zero/−100 placeholders; all task data is
        carried in the metadata dict so it can be consumed by the model's
        embedding and loss layers directly.

        Args:
            batch_size: Number of independent task instances in the batch.
            device:     Target torch device; defaults to CPU.

        Returns:
            input_ids: Tensor[batch_size, seq_len] of zeros (placeholder).
            labels:    Tensor[batch_size, seq_len] filled with −100 (placeholder).
            meta:      Dict containing:
                         colors, xs, ys, roles, sep_mask, sep_types,
                         grid_ids, target_mask, target_labels,
                         target_input_colors, context_mask, lengths.
        """
        if device is None:
            device = torch.device("cpu")

        samples = [self._generate_one() for _ in range(batch_size)]

        max_N = self.seq_len  # Fixed padding for torch.compile stability

        # Allocate padded tensors — identical layout to ProceduralARCTask
        colors = torch.full((batch_size, max_N), PAD_COLOR, dtype=torch.long, device=device)
        xs_t = torch.full((batch_size, max_N), PAD_COORD, dtype=torch.long, device=device)
        ys_t = torch.full((batch_size, max_N), PAD_COORD, dtype=torch.long, device=device)
        roles = torch.zeros(batch_size, max_N, dtype=torch.long, device=device)
        sep_mask = torch.ones(batch_size, max_N, dtype=torch.bool, device=device)  # pad = sep
        sep_types = torch.zeros(batch_size, max_N, dtype=torch.long, device=device)
        grid_ids = torch.full((batch_size, max_N), -1, dtype=torch.long, device=device)
        target_mask = torch.zeros(batch_size, max_N, dtype=torch.bool, device=device)
        target_labels = torch.full((batch_size, max_N), -100, dtype=torch.long, device=device)
        target_input_colors = torch.full((batch_size, max_N), PAD_COLOR, dtype=torch.long, device=device)
        lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        context_mask = torch.ones(batch_size, max_N, dtype=torch.bool, device=device)

        for i, s in enumerate(samples):
            N = s["length"]
            lengths[i] = N
            colors[i, :N] = torch.tensor(s["colors"], dtype=torch.long)
            xs_t[i, :N] = torch.tensor(s["xs"], dtype=torch.long)
            ys_t[i, :N] = torch.tensor(s["ys"], dtype=torch.long)
            roles[i, :N] = torch.tensor(s["roles"], dtype=torch.long)
            sep_mask[i, :N] = torch.tensor(s["sep_mask"], dtype=torch.bool)
            sep_types[i, :N] = torch.tensor(s["sep_types"], dtype=torch.long)
            grid_ids[i, :N] = torch.tensor(s["grid_ids"], dtype=torch.long)
            target_mask[i, :N] = torch.tensor(s["target_mask"], dtype=torch.bool)
            target_input_colors[i, :N] = torch.tensor(s["target_input_colors"], dtype=torch.long)

            # Align target_labels: ground-truth color at each target position
            tgt_positions = [j for j, m in enumerate(s["target_mask"]) if m]
            for j, pos in enumerate(tgt_positions):
                if j < len(s["target_colors"]):
                    target_labels[i, pos] = s["target_colors"][j]

            # context_mask = ~target_mask for actual tokens, True for padding
            context_mask[i, :N] = ~target_mask[i, :N]

        # Placeholder returns (matches ARCTask / ProceduralARCTask interface)
        input_ids = torch.zeros(batch_size, max_N, dtype=torch.long, device=device)
        labels = torch.full((batch_size, max_N), -100, dtype=torch.long, device=device)

        meta = {
            "colors": colors,
            "xs": xs_t,
            "ys": ys_t,
            "roles": roles,
            "sep_mask": sep_mask,
            "sep_types": sep_types,
            "grid_ids": grid_ids,
            "target_mask": target_mask,
            "target_labels": target_labels,
            "target_input_colors": target_input_colors,
            "context_mask": context_mask,
            "lengths": lengths,
        }

        return input_ids, labels, meta


# ────────────────────────────────────────────────────────────────────
# Smoke test
# ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing ConditionalTransformTask...")

    task = ConditionalTransformTask(seq_len=2048, augment=True, n_demos=2)
    input_ids, labels, meta = task.generate_batch(8)

    print(f"  Colors shape:  {meta['colors'].shape}")
    print(f"  Lengths:       {meta['lengths'].tolist()}")
    print(f"  Targets/sample: {[(meta['target_labels'][i] != -100).sum().item() for i in range(8)]}")
    print(f"  Grid IDs range: [{meta['grid_ids'].min().item()}, {meta['grid_ids'].max().item()}]")

    # Structural assertions (mirrors ProceduralARCTask self-test)
    for i in range(8):
        n = meta["lengths"][i].item()
        n_tgt = (meta["target_labels"][i] != -100).sum().item()
        n_tmask = meta["target_mask"][i, :n].sum().item()
        n_ctx = meta["context_mask"][i, :n].sum().item()
        assert n_tgt == n_tmask, f"[{i}] target_labels/target_mask mismatch: {n_tgt} vs {n_tmask}"
        assert n_ctx + n_tmask == n, f"[{i}] context+target != length: {n_ctx}+{n_tmask} != {n}"

    # Per-rule input != output check
    print("\n  Per-rule input!=output check:")
    for rule_name, gen_fn in _RULE_GENERATORS.items():
        changed = 0
        for _ in range(30):
            inp, out = gen_fn()
            if inp != out:
                changed += 1
        status = "OK" if changed >= 28 else f"WARN ({changed}/30)"
        print(f"    {rule_name}: {status}")

    # Verify batches are non-repeating
    _, _, m1 = task.generate_batch(2)
    _, _, m2 = task.generate_batch(2)
    same = (m1["colors"] == m2["colors"]).all().item()
    print(f"\n  Two batches identical: {same} (should be False)")
    assert not same, "Two consecutive batches must not be identical"

    print("\nConditionalTransformTask OK")
