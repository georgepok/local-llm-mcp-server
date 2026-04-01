"""Context relevance filtering — selective attention.

Input grid has context rows (category + value pairs), a query row identifying
a target category, and a blank answer row. The output fills the answer row
with the values from all context items whose category matches the query.

The metric must learn to selectively route information from only those context
rows that share a category with the query — ignoring irrelevant rows — and
aggregate the matched values into the fixed answer position.

Encoding:
  Colors 1-7  : category and value identifiers
  Color 9     : QUERY_MARKER (marks the query row)
  Color 0     : BG / blank

Grid layout (H × W):
  Rows 0..n_items-1 : context rows, each = [category, value, 0, ...]
  Row n_items       : query row   = [QUERY_MARKER, target_category, 0, ...]
  Row n_items+1     : answer row  (input: all 0; output: matched values)

Answer row: matched values written left-to-right in sorted order (for
determinism). Remaining cells stay 0.

Context rows are in shuffled order per demo to prevent positional shortcuts.
Category and value colors are randomized across demos and the test instance.
"""

import random
from typing import Dict, List, Optional, Tuple

import torch

from liquid_arc.tasks.procedural import (
    PAD_COLOR, PAD_COORD, N_COLORS,
    build_sequence,
)

QUERY_MARKER = 9
BG = 0

# Usable colors for categories and values (avoid BG and QUERY_MARKER)
ITEM_COLORS = list(range(1, 8))   # 1-7


def _empty_grid(H: int, W: int, bg: int = 0) -> List[List[int]]:
    return [[bg] * W for _ in range(H)]


class ContextRelevanceTask:
    """Infinite stream of context relevance filtering tasks.

    Each task instance:
      - Picks n_items (4-7) context items.
      - Assigns 1-3 of them the same target category (relevant items).
      - Generates n_demos demo pairs and a test pair.
      - Each pair uses a fresh color assignment and shuffled row order.
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

        n_items = random.randint(4, 7)
        n_relevant = random.randint(1, 3)
        n_relevant = min(n_relevant, n_items)

        demos = []
        for _ in range(self.n_demos):
            inp_grid, out_grid = self._generate_pair(n_items, n_relevant)
            demos.append((inp_grid, out_grid))

        test_inp, test_out = self._generate_pair(n_items, n_relevant)

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
        self, n_items: int, n_relevant: int
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """Build one (input, output) grid pair.

        Fresh color assignment on every call.
        Grid height: n_items + 2  (context rows + query row + answer row)
        Grid width:  max(6, n_items)
        """
        W = max(6, n_items)
        H = n_items + 2

        # Pick fresh colors for categories and values
        available = ITEM_COLORS[:]
        random.shuffle(available)

        # Need at least n_items distinct category slots + 1 query category.
        # We use a small set of categories (2-4 distinct categories),
        # one of which is the target.
        n_categories = random.randint(2, min(4, len(available)))
        categories = available[:n_categories]
        available_values = available[n_categories:]
        if not available_values:
            # Reuse category colors as values if we run short
            available_values = available[:]

        target_cat = random.choice(categories)

        # Assign categories to items
        item_categories: List[int] = []
        relevant_indices = random.sample(range(n_items), n_relevant)
        relevant_set = set(relevant_indices)

        for idx in range(n_items):
            if idx in relevant_set:
                item_categories.append(target_cat)
            else:
                # Pick a non-target category
                non_target = [c for c in categories if c != target_cat]
                if non_target:
                    item_categories.append(random.choice(non_target))
                else:
                    # Fallback: only one category exists, add a dummy value
                    item_categories.append(target_cat)

        # Assign values to items
        item_values: List[int] = []
        for _ in range(n_items):
            item_values.append(random.choice(available_values) if available_values
                               else random.choice(ITEM_COLORS))

        # Shuffle row order (prevents positional shortcutting)
        row_order = list(range(n_items))
        random.shuffle(row_order)

        # Build input grid
        inp_grid = _empty_grid(H, W)

        for row_pos, item_idx in enumerate(row_order):
            inp_grid[row_pos][0] = item_categories[item_idx]
            inp_grid[row_pos][1] = item_values[item_idx]

        # Query row
        inp_grid[n_items][0] = QUERY_MARKER
        inp_grid[n_items][1] = target_cat
        # Answer row stays blank in input (all BG)

        # Output: ONLY the answer row (not the context/query which never change)
        # Forces the model to focus on the filtering task, not copying.
        matched_values = sorted([
            item_values[idx] for idx in range(n_items)
            if item_categories[idx] == target_cat
        ])

        answer_row = [BG] * W
        for col, val in enumerate(matched_values):
            if col < W:
                answer_row[col] = val

        out_grid = [answer_row]

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
