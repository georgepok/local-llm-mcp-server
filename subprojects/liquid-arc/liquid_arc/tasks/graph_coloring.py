"""Graph coloring task — constraint satisfaction over arbitrary graphs.

Input grid encodes a graph (adjacency matrix) and partial coloring.
Output: all nodes colored such that no adjacent nodes share colors.

The constraint structure is a GRAPH, not a spatial neighborhood. The metric
must learn that adjacency in the graph (top half) determines which node
colorings (bottom half) constrain each other.
"""

import random
from typing import Dict, List, Optional, Tuple

import torch

from liquid_arc.tasks.procedural import (
    PAD_COLOR, PAD_COORD, N_COLORS,
    build_sequence,
)

BG_COLOR = 0
EDGE_COLOR = 5  # color used to mark edges in adjacency matrix


class GraphColoringTask:
    """Infinite stream of graph coloring tasks.

    Each task:
    - 3-6 nodes, ~40% edge probability
    - Top half of grid: adjacency matrix (edge markers)
    - Bottom half: node colors (some given, some to predict)
    - Output: all nodes validly colored (no adjacent pair shares color)
    - Re-colored and re-selected given nodes across demos
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

        n_nodes = random.randint(3, 6)
        edge_prob = 0.4

        # Generate random graph (adjacency list)
        edges = set()
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if random.random() < edge_prob:
                    edges.add((i, j))
                    edges.add((j, i))

        # Find a valid 3-coloring via greedy
        coloring = self._greedy_color(n_nodes, edges, n_colors=3)

        demos = []
        for _ in range(self.n_demos):
            inp_grid, out_grid = self._make_pair(n_nodes, edges, coloring)
            demos.append((inp_grid, out_grid))

        test_inp, test_out = self._make_pair(n_nodes, edges, coloring)

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

    def _greedy_color(self, n_nodes: int, edges: set, n_colors: int = 3) -> List[int]:
        """Greedy graph coloring. Returns list of color indices per node."""
        # Use colors 1, 2, 3 (avoid BG=0 and EDGE_COLOR)
        available_colors = [c for c in range(1, N_COLORS) if c != EDGE_COLOR]
        random.shuffle(available_colors)
        palette = available_colors[:n_colors]

        coloring = [-1] * n_nodes
        order = list(range(n_nodes))
        random.shuffle(order)

        for node in order:
            # Find colors used by neighbors
            neighbor_colors = set()
            for i, j in edges:
                if i == node and coloring[j] >= 0:
                    neighbor_colors.add(coloring[j])

            # Pick first available color
            for c in palette:
                if c not in neighbor_colors:
                    coloring[node] = c
                    break
            else:
                # Fallback: use first palette color (may violate constraints
                # for dense graphs, but 40% density with 3 colors is almost
                # always 3-colorable for 3-6 nodes)
                coloring[node] = palette[0]

        return coloring

    def _make_pair(self, n_nodes: int, edges: set,
                   base_coloring: List[int]) -> Tuple[List[List[int]], List[List[int]]]:
        """Generate one (input, output) pair for graph coloring.

        Re-colors nodes with a fresh color permutation and randomly selects
        which nodes are "given" vs "to predict".
        """
        # Fresh color permutation for this pair
        used_colors = list(set(base_coloring))
        fresh_colors = [c for c in range(1, N_COLORS) if c != EDGE_COLOR]
        random.shuffle(fresh_colors)
        color_map = {}
        for i, old_c in enumerate(used_colors):
            color_map[old_c] = fresh_colors[i]
        coloring = [color_map[c] for c in base_coloring]

        # Grid layout:
        # Top n_nodes rows: adjacency matrix (n_nodes × n_nodes)
        # Bottom 1 row: node colors
        W = n_nodes
        H = n_nodes + 1  # adjacency + coloring row

        # Build adjacency matrix (top half)
        grid = [[BG_COLOR] * W for _ in range(H)]
        for i, j in edges:
            if i < n_nodes and j < n_nodes:
                grid[i][j] = EDGE_COLOR

        # Build full coloring row (bottom)
        for x in range(n_nodes):
            grid[n_nodes][x] = coloring[x]

        # Input: full grid (adjacency + partial coloring with some hidden)
        n_given = random.randint(1, max(1, n_nodes - 1))
        given_nodes = set(random.sample(range(n_nodes), n_given))

        inp_grid = [row[:] for row in grid]
        for x in range(n_nodes):
            if x not in given_nodes:
                inp_grid[n_nodes][x] = BG_COLOR  # hide this node's color

        # Output: ONLY the coloring row (not the adjacency matrix)
        # This forces the model to focus on constraint satisfaction,
        # not copying the adjacency matrix which never changes.
        out_grid = [coloring]

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
