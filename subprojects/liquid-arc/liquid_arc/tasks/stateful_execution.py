"""Stateful execution task — cumulative state tracking.

Variables are columns, steps are rows. Each operation either sets a variable
to a direct value, copies from another variable, or conditionally copies.
The model must track accumulated state across all rows and predict the final
values in the result row.

Encoding:
  Colors 1-7  : variable values
  Color 8     : COPY_MARKER  (copy from source column)
  Color 9     : COND_MARKER  (conditional copy if source > 2)
  Color 0     : BG / blank

Grid layout:
  Row 0          : initial state (each column = initial value, 0 = unset)
  Rows 1..n_ops  : one operation per row
  Last row       : result — input is blank, output is final state

For copy/conditional rows the source and target column cells are both marked
with the appropriate marker; all other cells in that row are 0.

Different operation sequences per demo (same n_vars / n_ops structure,
different operations). Operations are NOT shuffled — temporal order matters.
"""

import random
from typing import Dict, List, Optional, Tuple

import torch

from liquid_arc.tasks.procedural import (
    PAD_COLOR, PAD_COORD, N_COLORS,
    build_sequence,
)

COPY_MARKER = 8
COND_MARKER = 9
BG = 0

# Usable value colors (1-7)
VALUE_COLORS = list(range(1, 8))


def _empty_grid(H: int, W: int, bg: int = 0) -> List[List[int]]:
    return [[bg] * W for _ in range(H)]


class StatefulExecutionTask:
    """Infinite stream of stateful instruction execution tasks.

    Each task instance:
      - Picks n_vars (3-5) and n_ops (3-6).
      - Generates n_demos demo pairs plus a test pair.
      - Each pair uses a fresh random sequence of operations on the same
        structural template (same n_vars / n_ops).
      - The result row input is all-zero; output is the final variable state.
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

        n_vars = random.randint(3, 5)
        n_ops = random.randint(3, 6)

        demos = []
        for _ in range(self.n_demos):
            inp_grid, out_grid = self._generate_pair(n_vars, n_ops)
            demos.append((inp_grid, out_grid))

        test_inp, test_out = self._generate_pair(n_vars, n_ops)

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

    def _generate_pair(
        self, n_vars: int, n_ops: int
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """Build one (input, output) grid pair.

        Grid height: n_ops + 2  (initial-state row + op rows + result row)
        Grid width:  n_vars
        """
        H = n_ops + 2
        W = n_vars

        # Initial state: some variables set to values 1-7, rest 0
        state: List[int] = [0] * n_vars
        for v in range(n_vars):
            if random.random() < 0.5:
                state[v] = random.choice(VALUE_COLORS)

        # Build operation sequence and execute it incrementally
        op_rows_inp: List[List[int]] = []   # what appears in the INPUT grid
        op_rows_out: List[List[int]] = []   # what appears in the OUTPUT grid (same)

        current: List[int] = state[:]

        for _ in range(n_ops):
            op_type = random.choice(["set", "copy", "cond"])
            row_inp = [BG] * W
            row_out = [BG] * W  # operations rows are identical in input and output

            if op_type == "set":
                # Set a target variable to a direct value
                tgt = random.randint(0, n_vars - 1)
                val = random.choice(VALUE_COLORS)
                row_inp[tgt] = val
                row_out[tgt] = val
                current[tgt] = val

            elif op_type == "copy":
                # Copy source variable value into target variable
                tgt = random.randint(0, n_vars - 1)
                src = random.randint(0, n_vars - 1)
                while src == tgt and n_vars > 1:
                    src = random.randint(0, n_vars - 1)
                row_inp[tgt] = COPY_MARKER
                row_inp[src] = COPY_MARKER
                row_out[tgt] = COPY_MARKER
                row_out[src] = COPY_MARKER
                current[tgt] = current[src]

            else:  # cond
                # Conditional copy: copy src->tgt only if current[src] > 2
                tgt = random.randint(0, n_vars - 1)
                src = random.randint(0, n_vars - 1)
                while src == tgt and n_vars > 1:
                    src = random.randint(0, n_vars - 1)
                row_inp[tgt] = COND_MARKER
                row_inp[src] = COND_MARKER
                row_out[tgt] = COND_MARKER
                row_out[src] = COND_MARKER
                if current[src] > 2:
                    current[tgt] = current[src]

            op_rows_inp.append(row_inp)
            op_rows_out.append(row_out)

        # Input grid: initial-state row | op rows | blank result row
        inp_grid = _empty_grid(H, W)
        inp_grid[0] = state[:]
        for i, row in enumerate(op_rows_inp):
            inp_grid[i + 1] = row
        # Last row stays blank (all BG) in the input

        # Output: ONLY the result row (final state).
        # Context (initial state + operations) is in the input; copying it
        # to output drowns the transform signal with 83% copy cells.
        out_grid = [current[:]]

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
