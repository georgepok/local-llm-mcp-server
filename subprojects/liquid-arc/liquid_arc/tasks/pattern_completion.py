"""Pattern completion task — sequential/temporal dependencies.

Input: Grid where rows encode a repeating color pattern, with one row blanked.
Output: Same grid with the blank row filled following the pattern.

The pattern is in the ORDER of elements, not spatial arrangement. The metric
must learn that cells in the same COLUMN across rows should communicate
(to detect repetition), regardless of row distance.
"""

import random
from typing import Dict, List, Optional, Tuple

import torch

from liquid_arc.tasks.procedural import (
    PAD_COLOR, PAD_COORD, N_COLORS,
    build_sequence,
)

BG_COLOR = 0


class PatternCompletionTask:
    """Infinite stream of pattern completion tasks.

    Each task:
    - Picks a repeating color pattern with period 2-4
    - Grid width = 1-2x the period
    - 3-6 rows, each row is the same repeating pattern
    - One row is blanked (all BG)
    - Output: blank row filled with the pattern
    - Different blank position and palette across demos
    """

    def __init__(self, seq_len: int = 2048, augment: bool = True, n_demos: int = 2, **kwargs):
        self.seq_len = seq_len
        self.augment = augment
        self.n_demos = n_demos
        self._seed_counter = random.randint(0, 2**31)

    def _next_seed(self) -> int:
        self._seed_counter += 1
        return self._seed_counter

    def _generate_one(self) -> Dict:
        seed = self._next_seed()
        random.seed(seed)

        # Pattern period
        period = random.randint(2, 4)
        # Grid width: 1-2x period
        W = period * random.randint(1, 2)
        # Number of rows (repetitions)
        n_rows = random.randint(3, 6)

        demos = []
        for _ in range(self.n_demos):
            inp_grid, out_grid = self._make_pair(period, W, n_rows)
            demos.append((inp_grid, out_grid))

        test_inp, test_out = self._make_pair(period, W, n_rows)

        seq = build_sequence(demos, test_inp, test_out)

        if seq["length"] > self.seq_len:
            if self.n_demos > 1:
                demos = demos[:1]
                seq = build_sequence(demos, test_inp, test_out)
            if seq["length"] > self.seq_len:
                for key in ["colors", "xs", "ys", "roles", "sep_mask", "sep_types",
                            "grid_ids", "target_mask", "target_input_colors"]:
                    seq[key] = seq[key][:self.seq_len]
                seq["length"] = self.seq_len

        return seq

    def _make_pair(self, period: int, W: int, n_rows: int) -> Tuple[List[List[int]], List[List[int]]]:
        """Generate one (input, output) pair for pattern completion."""
        # Pick distinct colors for the pattern (avoid BG)
        available = [c for c in range(1, N_COLORS)]
        random.shuffle(available)
        pattern = available[:period]

        # Build the full pattern row (tile to width W)
        pattern_row = [pattern[x % period] for x in range(W)]

        # Build complete grid (all rows same pattern)
        full_grid = [pattern_row[:] for _ in range(n_rows)]

        # Pick which row to blank
        blank_row = random.randint(0, n_rows - 1)

        # Input: blank one row
        inp_grid = [row[:] for row in full_grid]
        inp_grid[blank_row] = [BG_COLOR] * W

        # Output: complete grid
        out_grid = [row[:] for row in full_grid]

        return inp_grid, out_grid

    def generate_batch(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        if device is None:
            device = torch.device("cpu")

        samples = [self._generate_one() for _ in range(batch_size)]
        max_N = self.seq_len

        colors = torch.full((batch_size, max_N), PAD_COLOR, dtype=torch.long, device=device)
        xs_t = torch.full((batch_size, max_N), PAD_COORD, dtype=torch.long, device=device)
        ys_t = torch.full((batch_size, max_N), PAD_COORD, dtype=torch.long, device=device)
        roles = torch.zeros(batch_size, max_N, dtype=torch.long, device=device)
        sep_mask = torch.ones(batch_size, max_N, dtype=torch.bool, device=device)
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

            tgt_positions = [j for j, m in enumerate(s["target_mask"]) if m]
            for j, pos in enumerate(tgt_positions):
                if j < len(s["target_colors"]):
                    target_labels[i, pos] = s["target_colors"][j]

            context_mask[i, :N] = ~target_mask[i, :N]

        input_ids = torch.zeros(batch_size, max_N, dtype=torch.long, device=device)
        labels = torch.full((batch_size, max_N), -100, dtype=torch.long, device=device)

        meta = {
            "colors": colors, "xs": xs_t, "ys": ys_t, "roles": roles,
            "sep_mask": sep_mask, "sep_types": sep_types, "grid_ids": grid_ids,
            "target_mask": target_mask, "target_labels": target_labels,
            "target_input_colors": target_input_colors, "context_mask": context_mask,
            "lengths": lengths,
        }
        return input_ids, labels, meta
