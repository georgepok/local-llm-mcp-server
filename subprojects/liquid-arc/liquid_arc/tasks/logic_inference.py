"""Logical inference task — non-spatial chain following.

Input grid encodes implications (A→B, B→C, ...) and a starting fact.
Output grid adds a RESULT row showing all reachable propositions.

The implications are in SHUFFLED row order — the model can't rely on row
position to determine the inference chain. The metric must learn that
propositions connected by implication chains should communicate regardless
of spatial row position.

Different proposition colors between demos and test force learning the
STRUCTURE, not specific colors.
"""

import random
from typing import Dict, List, Optional, Tuple

import torch

from liquid_arc.tasks.procedural import (
    PAD_COLOR, PAD_COORD, N_COLORS,
    build_sequence,
)

# Reserve colors for markers
ARROW_COLOR = 8   # "→" marker
STAR_COLOR = 9    # "START" / "RESULT" marker
BG_COLOR = 0      # background


class LogicInferenceTask:
    """Infinite stream of logical inference tasks.

    Each task:
    - Generates a chain of implications: P0→P1→P2→...→Pn (chain_len 2-5)
    - Each proposition is a distinct color (1-7)
    - Input grid: shuffled implication rows + START row
    - Output grid: same + RESULT row with all reachable propositions
    - Different color assignment per demo/test (same structure, different colors)
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

        # Chain length: 2-5 propositions in the chain
        chain_len = random.randint(2, 5)
        # Grid width: enough for "color ARROW color BG ..." pattern
        W = 5  # fixed width for simplicity

        demos = []
        for _ in range(self.n_demos):
            inp_grid, out_grid = self._make_pair(chain_len, W)
            demos.append((inp_grid, out_grid))

        test_inp, test_out = self._make_pair(chain_len, W)

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

    def _make_pair(self, chain_len: int, W: int) -> Tuple[List[List[int]], List[List[int]]]:
        """Generate one (input, output) pair for logical inference.

        Uses fresh color assignment each time (same chain structure).
        """
        # Pick distinct proposition colors (avoid ARROW_COLOR, STAR_COLOR, BG)
        available = [c for c in range(1, N_COLORS) if c not in (ARROW_COLOR, STAR_COLOR)]
        random.shuffle(available)
        # chain_len+1 propositions: P0→P1→...→P_{chain_len}
        n_props = chain_len + 1
        props = available[:n_props]

        # Build implication rows: P_i → P_{i+1}
        impl_rows = []
        for i in range(chain_len):
            row = [BG_COLOR] * W
            row[0] = props[i]
            row[1] = ARROW_COLOR
            row[2] = props[i + 1]
            impl_rows.append(row)

        # Shuffle implication rows (key: prevents positional shortcutting)
        random.shuffle(impl_rows)

        # START row: shows starting proposition
        start_row = [BG_COLOR] * W
        start_row[0] = props[0]
        start_row[1] = STAR_COLOR

        # Input grid: shuffled implications + start row
        inp_grid = impl_rows + [start_row]

        # Output grid: same as input + RESULT row with all reachable
        result_row = [BG_COLOR] * W
        result_row[0] = STAR_COLOR  # marker for result
        # Fill reachable propositions (all of them, in chain order)
        for j, p in enumerate(props):
            if j + 1 < W:
                result_row[j + 1] = p

        out_grid = [row[:] for row in inp_grid] + [result_row]

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
