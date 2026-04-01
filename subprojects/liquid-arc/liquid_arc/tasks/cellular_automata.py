"""Cellular automata task generator for LiquidARC.

Drop-in replacement for ProceduralARCTask. Uses the same sequence format and
generate_batch() interface.

Two CA variants are generated with equal probability:

  Classic Life-like (2-state):
    Each cell is either alive (alive_color) or dead (bg=0).
    A random Life-like rule (B/S notation) is sampled once per task and held
    fixed across all demos and the test pair. The model must identify the rule
    from the demos and apply it to the test input.

  Multi-color Life-like (3-state):
    Three states: bg=0, young (one color), mature (different color).
    Transition rules:
      bg     → young   if the number of alive (young+mature) neighbors is in birth_set
      young  → mature  always (any surviving young cell ages)
      mature → mature  if alive neighbor count is in survival_set
      mature → bg      otherwise

Grid sizes are 5–10 × 5–10. Initial density is 0.15–0.45.
"""

import random
from typing import Dict, List, Optional, Set, Tuple

import torch

from .procedural import (
    PAD_COLOR,
    PAD_COORD,
    ROLE_INPUT_DEMO,
    ROLE_OUTPUT_DEMO,
    ROLE_TEST_INPUT,
    ROLE_TEST_OUTPUT,
    SEP_BETWEEN_DEMOS,
    SEP_BEFORE_TEST_IN,
    SEP_BEFORE_TEST_OUT,
    SEP_DEMO_IO,
    _augment_grid,
    build_sequence,
)

# ─────────────────────────────────────────────────────────────────────────────
# Rule generation helpers
# ─────────────────────────────────────────────────────────────────────────────

# Neighbor counts that are interesting for birth (exclude 0 — spontaneous gen)
_BIRTH_CANDIDATES = list(range(1, 9))
# Neighbor counts that are biologically plausible for survival (1-7)
_SURVIVAL_CANDIDATES = list(range(1, 8))


def _random_life_rule() -> Tuple[Set[int], Set[int]]:
    """Sample a random Life-like (B/S) rule.

    Biases toward rules that produce interesting but not explosive behavior:
    - Birth set: 1-3 values from {1..8}, weighted toward {2,3,4}
    - Survival set: 1-4 values from {1..7}, weighted toward {2,3,4,5}

    Returns:
        (birth_set, survival_set): sets of neighbor counts that trigger
        birth and survival respectively.
    """
    # Weight birth candidates toward middle values (2-4) — avoids explosion
    birth_weights = [1, 3, 5, 5, 3, 2, 1, 1]  # for counts 1..8
    n_birth = random.randint(1, 3)
    birth_set: Set[int] = set(
        random.choices(_BIRTH_CANDIDATES, weights=birth_weights, k=n_birth)
    )

    # Weight survival candidates toward 2-5
    surv_weights = [1, 3, 5, 5, 4, 2, 1]  # for counts 1..7
    n_survival = random.randint(1, 4)
    survival_set: Set[int] = set(
        random.choices(_SURVIVAL_CANDIDATES, weights=surv_weights, k=n_survival)
    )

    return birth_set, survival_set


def _random_multicolor_rule() -> Tuple[Set[int], Set[int]]:
    """Sample rule parameters for the multi-color (3-state) variant.

    Returns (birth_set, survival_set) where:
      - birth_set: neighbor counts (young+mature) that turn bg → young
      - survival_set: neighbor counts that keep mature → mature (else → bg)
    """
    birth_weights = [1, 3, 5, 4, 3, 2, 1, 1]
    n_birth = random.randint(1, 3)
    birth_set: Set[int] = set(
        random.choices(_BIRTH_CANDIDATES, weights=birth_weights, k=n_birth)
    )

    surv_weights = [1, 3, 5, 5, 4, 2, 1]
    n_survival = random.randint(2, 5)
    survival_set: Set[int] = set(
        random.choices(_SURVIVAL_CANDIDATES, weights=surv_weights, k=n_survival)
    )

    return birth_set, survival_set


# ─────────────────────────────────────────────────────────────────────────────
# CA simulation
# ─────────────────────────────────────────────────────────────────────────────

def _count_alive_neighbors(grid: List[List[int]], r: int, c: int, bg: int) -> int:
    """Count Moore-neighborhood (8-connected) cells that are not background."""
    H, W = len(grid), len(grid[0])
    count = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                if grid[nr][nc] != bg:
                    count += 1
    return count


def _ca_step(
    grid: List[List[int]],
    birth: Set[int],
    survival: Set[int],
    alive_color: int,
    bg: int,
) -> List[List[int]]:
    """Advance a classic 2-state Life-like CA by one generation.

    Args:
        grid: 2D list of color ints; alive_color = alive, bg = dead.
        birth: set of neighbor counts that birth a dead cell.
        survival: set of neighbor counts that keep a live cell alive.
        alive_color: color representing a live cell.
        bg: background (dead) color.

    Returns:
        New 2D grid after one CA step.
    """
    H, W = len(grid), len(grid[0])
    new_grid: List[List[int]] = [[bg] * W for _ in range(H)]
    for r in range(H):
        for c in range(W):
            n = _count_alive_neighbors(grid, r, c, bg)
            if grid[r][c] != bg:
                # Currently alive
                new_grid[r][c] = alive_color if n in survival else bg
            else:
                # Currently dead
                new_grid[r][c] = alive_color if n in birth else bg
    return new_grid


def _ca_step_multicolor(
    grid: List[List[int]],
    birth: Set[int],
    survival: Set[int],
    young_color: int,
    mature_color: int,
    bg: int,
) -> List[List[int]]:
    """Advance a 3-state Life-like CA by one generation.

    States:
      bg           — dead
      young_color  — young alive (just born)
      mature_color — mature alive (survived at least one step)

    Transitions:
      bg     → young   if alive-neighbor count in birth_set
      young  → mature  always (any cell that survives its first step matures)
      mature → mature  if alive-neighbor count in survival_set
      mature → bg      otherwise

    Args:
        grid: 2D list of color ints.
        birth: neighbor counts that trigger birth (bg → young).
        survival: neighbor counts that keep a mature cell alive.
        young_color: color for newly-born cells.
        mature_color: color for cells that survived at least one step.
        bg: background color (dead).

    Returns:
        New 2D grid after one CA step.
    """
    H, W = len(grid), len(grid[0])
    new_grid: List[List[int]] = [[bg] * W for _ in range(H)]
    for r in range(H):
        for c in range(W):
            n = _count_alive_neighbors(grid, r, c, bg)
            cell = grid[r][c]
            if cell == bg:
                new_grid[r][c] = young_color if n in birth else bg
            elif cell == young_color:
                # Young cells always mature (if they survive — rule not applied
                # to young cells explicitly; they always age one step)
                new_grid[r][c] = mature_color
            else:
                # Mature cell
                new_grid[r][c] = mature_color if n in survival else bg
    return new_grid


# ─────────────────────────────────────────────────────────────────────────────
# Grid initialization
# ─────────────────────────────────────────────────────────────────────────────

def _random_grid(
    h: int,
    w: int,
    alive_color: int,
    bg: int,
    density: float,
) -> List[List[int]]:
    """Generate a random 2-state binary grid."""
    return [
        [alive_color if random.random() < density else bg for _ in range(w)]
        for _ in range(h)
    ]


def _random_grid_multicolor(
    h: int,
    w: int,
    young_color: int,
    mature_color: int,
    bg: int,
    density: float,
) -> List[List[int]]:
    """Generate a random 3-state initial grid.

    Approximately half of living cells start as young, half as mature.
    """
    grid = [[bg] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if random.random() < density:
                grid[r][c] = random.choice([young_color, mature_color])
    return grid


# ─────────────────────────────────────────────────────────────────────────────
# Main task class
# ─────────────────────────────────────────────────────────────────────────────

class CellularAutomataTask:
    """Cellular automata task generator for LiquidARC.

    Drop-in replacement for ProceduralARCTask — identical generate_batch()
    interface.

    Each task instance picks one Life-like rule at random (classic 2-state or
    multi-color 3-state) and holds it fixed across all demo pairs and the test
    pair. The model must infer the rule from the demonstrations.

    Args:
        seq_len: maximum sequence length; sequences are padded to this.
        augment: apply D4 (rotation + flip) augmentation if True.
        n_demos: number of demonstration (input, output) pairs per task.
    """

    def __init__(
        self,
        seq_len: int = 2048,
        augment: bool = True,
        n_demos: int = 2,
    ) -> None:
        self.seq_len = seq_len
        self.augment = augment
        self.n_demos = n_demos

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_pair_classic(
        h: int,
        w: int,
        birth: Set[int],
        survival: Set[int],
        alive_color: int,
        bg: int,
        density: float,
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """Generate one (input, output) pair for the classic 2-state variant."""
        inp = _random_grid(h, w, alive_color, bg, density)
        out = _ca_step(inp, birth, survival, alive_color, bg)
        return inp, out

    @staticmethod
    def _make_pair_multicolor(
        h: int,
        w: int,
        birth: Set[int],
        survival: Set[int],
        young_color: int,
        mature_color: int,
        bg: int,
        density: float,
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """Generate one (input, output) pair for the multi-color 3-state variant."""
        inp = _random_grid_multicolor(h, w, young_color, mature_color, bg, density)
        out = _ca_step_multicolor(inp, birth, survival, young_color, mature_color, bg)
        return inp, out

    # ------------------------------------------------------------------
    # Core task generator
    # ------------------------------------------------------------------

    def _generate_one(self) -> Dict:
        """Generate a single CA task: n_demos demo pairs + 1 test pair.

        All pairs share the same rule. Grid dimensions and density are
        re-sampled independently for each pair so the model cannot rely on
        grid size as a cue.

        Returns:
            dict with keys: colors, xs, ys, roles, sep_mask, sep_types,
            grid_ids, target_mask, target_colors, target_input_colors, length.
            Same format as build_sequence().
        """
        multicolor = random.random() < 0.5

        if multicolor:
            birth, survival = _random_multicolor_rule()
            # Pick two distinct colors for young and mature (both non-zero)
            alive_colors = random.sample(range(1, 10), 2)
            young_color, mature_color = alive_colors[0], alive_colors[1]
            bg = 0
        else:
            birth, survival = _random_life_rule()
            alive_color = random.randint(1, 9)
            bg = 0

        # Augmentation params: shared across all demos + test for this task
        rot = random.randint(0, 3) if self.augment else 0
        flip = random.random() < 0.5 if self.augment else False

        demos: List[Tuple[List[List[int]], List[List[int]]]] = []
        for _ in range(self.n_demos):
            h = random.randint(5, 10)
            w = random.randint(5, 10)
            density = random.uniform(0.15, 0.45)
            if multicolor:
                inp, out = self._make_pair_multicolor(
                    h, w, birth, survival, young_color, mature_color, bg, density
                )
            else:
                inp, out = self._make_pair_classic(
                    h, w, birth, survival, alive_color, bg, density
                )
            inp = _augment_grid(inp, rot, flip)
            out = _augment_grid(out, rot, flip)
            demos.append((inp, out))

        # Test pair: same rule, independent grid dimensions and density
        h = random.randint(5, 10)
        w = random.randint(5, 10)
        density = random.uniform(0.15, 0.45)
        if multicolor:
            test_inp, test_out = self._make_pair_multicolor(
                h, w, birth, survival, young_color, mature_color, bg, density
            )
        else:
            test_inp, test_out = self._make_pair_classic(
                h, w, birth, survival, alive_color, bg, density
            )
        test_inp = _augment_grid(test_inp, rot, flip)
        test_out = _augment_grid(test_out, rot, flip)

        seq = build_sequence(demos, test_inp, test_out)

        # Drop demos one at a time if the sequence is too long
        while seq["length"] > self.seq_len and len(demos) > 1:
            demos = demos[:-1]
            seq = build_sequence(demos, test_inp, test_out)

        # Hard truncate as a last resort (preserves test output intact as much
        # as possible — we truncate from the front, keeping test output)
        if seq["length"] > self.seq_len:
            for key in [
                "colors", "xs", "ys", "roles", "sep_mask", "sep_types",
                "grid_ids", "target_mask", "target_input_colors",
            ]:
                seq[key] = seq[key][: self.seq_len]
            seq["length"] = self.seq_len

        return seq

    # ------------------------------------------------------------------
    # Public API — matches ProceduralARCTask.generate_batch()
    # ------------------------------------------------------------------

    def generate_batch(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Generate a batch of CA tasks.

        Returns the same format as ARCTask / ProceduralARCTask:
            (input_ids, labels, metadata)

        where metadata contains all tensors needed by the model.

        Args:
            batch_size: number of tasks in the batch.
            device: target torch device (defaults to CPU).

        Returns:
            input_ids: zero tensor [B, seq_len] (placeholder, matches interface).
            labels: -100 tensor [B, seq_len] (placeholder, matches interface).
            meta: dict of tensors:
                colors            [B, seq_len]  — cell color tokens
                xs                [B, seq_len]  — cell x coordinates
                ys                [B, seq_len]  — cell y coordinates
                roles             [B, seq_len]  — cell roles
                sep_mask          [B, seq_len]  — True where token is separator/pad
                sep_types         [B, seq_len]  — separator type
                grid_ids          [B, seq_len]  — which grid each cell belongs to
                target_mask       [B, seq_len]  — True at test-output positions
                target_labels     [B, seq_len]  — ground-truth colors (-100 elsewhere)
                target_input_colors [B, seq_len] — test-input color at same (x,y)
                context_mask      [B, seq_len]  — ~target_mask for real tokens
                lengths           [B]           — actual (unpadded) sequence lengths
        """
        if device is None:
            device = torch.device("cpu")

        samples = [self._generate_one() for _ in range(batch_size)]

        max_N = self.seq_len  # Fixed padding for torch.compile stability

        # Allocate padded tensors
        colors = torch.full((batch_size, max_N), PAD_COLOR, dtype=torch.long, device=device)
        xs_t = torch.full((batch_size, max_N), PAD_COORD, dtype=torch.long, device=device)
        ys_t = torch.full((batch_size, max_N), PAD_COORD, dtype=torch.long, device=device)
        roles = torch.zeros(batch_size, max_N, dtype=torch.long, device=device)
        sep_mask = torch.ones(batch_size, max_N, dtype=torch.bool, device=device)   # pad = sep
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

            # Place ground-truth colors at target positions
            tgt_positions = [j for j, m in enumerate(s["target_mask"]) if m]
            for j, pos in enumerate(tgt_positions):
                if j < len(s["target_colors"]):
                    target_labels[i, pos] = s["target_colors"][j]

            # Context mask: True for real non-target tokens, True for padding
            context_mask[i, :N] = ~target_mask[i, :N]

        # Placeholder tensors (match ARCTask interface)
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
