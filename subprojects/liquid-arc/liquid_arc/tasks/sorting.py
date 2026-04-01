"""Sorting task — non-spatial ordinal reasoning.

Input: 1×N grid with colors in random order.
Output: Same colors sorted by value (ascending).

The spatial positions (x coordinates) are IRRELEVANT — the rule depends on
COLOR VALUES. The metric must learn that tokens communicate based on relative
value, not spatial adjacency.

Color permutation across demos ensures the model learns "sort by whatever
ordering the demos show" rather than memorizing absolute color indices.
"""

import random
from typing import Dict, List, Optional, Tuple

import torch

from liquid_arc.tasks.procedural import (
    PAD_COLOR, PAD_COORD, N_COLORS,
    build_sequence,
)


class SortingTask:
    """Infinite stream of sequence sorting tasks.

    Each task instance:
    - Picks a random permutation of colors 1-9 as the "ordering"
    - Generates 2 demo pairs showing shuffled → sorted (by that ordering)
    - Test pair uses same ordering, different random permutation
    - N in [4, 12] (grid width)
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

        # Pick how many distinct values to use (4-9)
        n_values = random.randint(4, 9)
        # Pick which colors represent these values
        all_colors = list(range(1, N_COLORS))  # 1-9
        random.shuffle(all_colors)
        palette = all_colors[:n_values]  # These colors in this order define the sorting

        # Grid width: use n_values (each color appears exactly once)
        # Or allow repeats for harder tasks
        N = random.randint(max(4, n_values), 12)

        demos = []
        for _ in range(self.n_demos):
            inp_grid, out_grid = self._make_pair(palette, N)
            demos.append((inp_grid, out_grid))

        test_inp, test_out = self._make_pair(palette, N)

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

    def _make_pair(self, palette: List[int], N: int) -> Tuple[List[List[int]], List[List[int]]]:
        """Generate one (input, output) pair for sorting.

        Input: 1×N grid with colors from palette in random order.
        Output: 1×N grid with same colors sorted by palette ordering.
        """
        # Generate N values by sampling from palette (with replacement if N > len(palette))
        values = [random.choice(palette) for _ in range(N)]

        # Input: shuffled
        inp_row = values[:]
        random.shuffle(inp_row)

        # Output: sorted by position in palette (the "ordering")
        rank = {c: i for i, c in enumerate(palette)}
        out_row = sorted(inp_row, key=lambda c: rank[c])

        # Encode as 1×N grids (single row)
        inp_grid = [inp_row]
        out_grid = [out_row]
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
