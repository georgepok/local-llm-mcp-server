"""Dependency ordering — topological sort.

Input rows declare task dependencies (task depends_on prereq) in shuffled
order, followed by a separator row. Output rows show a valid execution order
via Kahn's algorithm (deterministic: tie-breaking by color value).

The metric must learn to follow dependency chains across arbitrarily ordered
rows and output a globally valid linearization — a task that requires reading
the entire input before producing any output token.

Encoding:
  Colors 1-7  : task identifiers
  Color 8     : ARROW_MARKER ("depends on" / "must run after")
  Color 9     : STAR_MARKER  (separator between deps and answer)
  Color 0     : BG / blank

Grid layout (H × W):
  Rows 0..n_edges-1  : dependency rows  = [task, ARROW_MARKER, prereq, 0, ...]
  Row  n_edges       : separator row    = [STAR_MARKER, 0, ...]
  Rows n_edges+1..H-1: answer rows (2)  = topological order, split evenly

Answer rows: the topological order (Kahn's, tie-break by color value) is
written left-to-right across the 2 answer rows. Remaining cells stay 0.

Dependency rows are shuffled per demo. Task colors are randomized across
demos and the test instance (same DAG structure, different colors).
"""

import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import torch

from liquid_arc.tasks.procedural import (
    PAD_COLOR, PAD_COORD, N_COLORS,
    build_sequence,
)

ARROW_MARKER = 8
STAR_MARKER = 9
BG = 0

# Usable colors for task identifiers (avoid BG, ARROW_MARKER, STAR_MARKER)
TASK_COLORS = list(range(1, 8))   # 1-7


def _empty_grid(H: int, W: int, bg: int = 0) -> List[List[int]]:
    return [[bg] * W for _ in range(H)]


def _random_dag_edges(n_tasks: int) -> List[Tuple[int, int]]:
    """Generate a random DAG on nodes 0..n_tasks-1.

    Nodes are in a fixed topological order (0 < 1 < ... < n_tasks-1).
    For each pair (i, j) with i < j, include edge j->i (j depends on i)
    with probability 0.4, capped so each node has at most 2 prerequisites
    and there is at least 1 edge total.
    """
    edges: List[Tuple[int, int]] = []
    in_degree = [0] * n_tasks

    for j in range(1, n_tasks):
        prereqs = [i for i in range(j) if in_degree[j] < 2]
        random.shuffle(prereqs)
        for i in prereqs:
            if random.random() < 0.4:
                edges.append((j, i))  # j depends on i
                in_degree[j] += 1
                if in_degree[j] >= 2:
                    break

    # Ensure at least one edge
    if not edges:
        j = random.randint(1, n_tasks - 1)
        i = random.randint(0, j - 1)
        edges.append((j, i))

    return edges


def _kahn_sort(n_tasks: int, edges: List[Tuple[int, int]]) -> List[int]:
    """Deterministic topological sort using Kahn's algorithm.

    Tie-breaking: always pick the smallest-index available node first.
    Returns node indices in execution order.
    """
    in_deg = [0] * n_tasks
    successors: List[List[int]] = [[] for _ in range(n_tasks)]

    for task, prereq in edges:
        in_deg[task] += 1
        successors[prereq].append(task)

    # Min-heap via sorted deque; we just keep a sorted list for small n
    ready = sorted([i for i in range(n_tasks) if in_deg[i] == 0])
    order: List[int] = []

    while ready:
        node = ready.pop(0)  # smallest available
        order.append(node)
        new_ready = []
        for succ in successors[node]:
            in_deg[succ] -= 1
            if in_deg[succ] == 0:
                new_ready.append(succ)
        ready = sorted(ready + new_ready)

    return order


class DependencyOrderTask:
    """Infinite stream of dependency ordering (topological sort) tasks.

    Each task instance:
      - Picks n_tasks (3-6) and generates a random DAG.
      - Computes the unique deterministic topological order via Kahn's.
      - Generates n_demos demo pairs and a test pair.
      - Each pair remaps node indices to fresh random colors.
      - Dependency rows are shuffled per demo.
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

        n_tasks = random.randint(3, 6)

        # Generate a fixed DAG structure (node indices 0..n_tasks-1)
        edges = _random_dag_edges(n_tasks)
        topo_order = _kahn_sort(n_tasks, edges)  # list of node indices

        demos = []
        for _ in range(self.n_demos):
            inp_grid, out_grid = self._generate_pair(n_tasks, edges, topo_order)
            demos.append((inp_grid, out_grid))

        test_inp, test_out = self._generate_pair(n_tasks, edges, topo_order)

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
        self,
        n_tasks: int,
        edges: List[Tuple[int, int]],
        topo_order: List[int],
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """Build one (input, output) grid pair with a fresh color assignment.

        Grid height: n_edges + 1 + 2  (dep rows + separator + 2 answer rows)
        Grid width:  max(6, n_tasks)
        """
        n_edges = len(edges)
        H = n_edges + 1 + 2   # deps | sep | answer0 | answer1
        W = max(6, n_tasks)

        # Fresh color assignment: node index -> color
        color_pool = TASK_COLORS[:]
        random.shuffle(color_pool)
        node_color: List[int] = color_pool[:n_tasks]

        # Build dependency rows (unshuffled first, then shuffle below)
        dep_rows: List[List[int]] = []
        for task_idx, prereq_idx in edges:
            row = [BG] * W
            row[0] = node_color[task_idx]
            row[1] = ARROW_MARKER
            row[2] = node_color[prereq_idx]
            dep_rows.append(row)

        # Shuffle dep rows (prevents positional shortcutting)
        random.shuffle(dep_rows)

        # Separator row
        sep_row = [BG] * W
        sep_row[0] = STAR_MARKER

        # Topological order as colors
        order_colors = [node_color[idx] for idx in topo_order]

        # Split across 2 answer rows
        half = (len(order_colors) + 1) // 2
        ans_row0 = [BG] * W
        ans_row1 = [BG] * W
        for col, c in enumerate(order_colors[:half]):
            if col < W:
                ans_row0[col] = c
        for col, c in enumerate(order_colors[half:]):
            if col < W:
                ans_row1[col] = c

        # Input grid: dep rows | sep | blank answer rows
        inp_grid = _empty_grid(H, W)
        for row_pos, row in enumerate(dep_rows):
            inp_grid[row_pos] = row[:]
        inp_grid[n_edges] = sep_row[:]
        # Answer rows stay blank in input

        # Output: ONLY the answer rows (topological order).
        # Dep rows and separator are context — copying them drowns transform signal.
        out_grid = [ans_row0[:], ans_row1[:]]

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
