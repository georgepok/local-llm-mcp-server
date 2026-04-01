# TASK: Agentic State Controller — Prototype on Post-Transition 5M Checkpoint

**From:** Claude Desktop (Research Direction)
**To:** Claude Code (Implementation)
**Date:** 2026-03-27
**Priority:** HIGH — builds directly on universality probe results

---

## Context

The universality probe proved the post-transition 5M geometric substrate is domain-general. Logic inference reached 61% eval in 300 steps. Sorting reached 63% eval in 50 steps. The metric learned "information-theoretic relevance" not spatial proximity. 92-97% of FFN neurons are shared across domains — the model develops a shared computational vocabulary composed differently per domain.

This experiment takes the next step: can this universal substrate learn to manage STATE across multi-step operations — the core capability needed for an agentic state controller that sits between a human and an LLM, tracking goals, constraints, actions, and context?

We design three agentic-analog tasks that use the existing grid token format but test capabilities specific to state management: cumulative state tracking, context relevance filtering, and dependency ordering. All three fit within `build_sequence()` and use the existing training pipeline.

## Three Agentic State Management Tasks

### Task 1: Stateful Instruction Execution

**What it tests:** Cumulative state tracking across multi-step operations — the most fundamental agentic capability.

**Design:**
The grid represents a stateful computation. "Variables" are columns. "Steps" are rows. The model must track how variable values change through a sequence of operations and predict the final state.

```
Input grid (6 rows × 5 cols):
  Row 0: [3, 0, 0, 0, 0]    ← initial state: var_0=3, others empty
  Row 1: [0, 0, 5, 0, 0]    ← SET: var_2=5
  Row 2: [0, 8, 0, 0, 0]    ← SET: var_1=8  (marker=8 means "copy from var_0")
  Row 3: [0, 0, 0, 9, 0]    ← SET: var_3=9  (marker=9 means "copy from var_2")
  Row 4: [0, 0, 0, 0, 0]    ← (padding)
  Row 5: [0, 0, 0, 0, 0]    ← (padding)

Output grid:
  Row 0-4: same as input
  Row 5: [3, 3, 5, 5, 0]    ← final state: var_0=3, var_1=copied from var_0=3,
                                var_2=5, var_3=copied from var_2=5
```

The key challenge: the model must propagate values FORWARD through the instruction sequence. Row 2's result depends on Row 0's value. Row 3's result depends on Row 1's value. This requires the metric to learn that operation rows should attend to the rows they reference, NOT just their spatial neighbors.

**Implementation notes:**
- Variables use colors 1-7 as values
- Color 8 = "copy from" marker (the column it appears in copies from another column)
- Color 9 = "conditional copy" marker (copy only if source > some threshold)
- Operations reference earlier rows through column alignment, not spatial adjacency
- Shuffling operation ORDER across demos tests whether the model learns the dependency structure vs memorizing row positions
- Chain length: 3-6 operations, 3-5 variables

```python
"""Stateful instruction execution — cumulative state tracking.

Variables are columns. Steps are rows. Operations modify variables
by setting values or copying from other variables. The model must
track the cumulative state and predict the final variable values.

This tests the core agentic capability: maintaining and updating
a state representation across multiple sequential operations where
later operations depend on results of earlier ones.
"""

import random
from typing import Dict
from liquid_arc.tasks.procedural import (
    build_sequence, N_COLORS, PAD_COLOR, PAD_COORD,
    _empty_grid,
)

# Markers
COPY_MARKER = 8      # "copy value from column X"
COND_MARKER = 9      # "conditional: copy if source > 2"
BG = 0


class StatefulExecutionTask:
    def __init__(self, seq_len=2048, n_demos=2, min_vars=3, max_vars=5,
                 min_ops=3, max_ops=6, **kwargs):
        self.seq_len = seq_len
        self.n_demos = n_demos
        self.min_vars = min_vars
        self.max_vars = max_vars
        self.min_ops = min_ops
        self.max_ops = max_ops
        self._seed_counter = random.randint(0, 2**31)

    def _next_seed(self):
        self._seed_counter += 1
        return self._seed_counter

    def _generate_pair(self, n_vars, n_ops):
        """Generate one input/output grid pair."""
        W = n_vars
        # Rows: initial state + operations + padding + result row
        H = n_ops + 2  # +1 for initial state, +1 for result

        # Initial state: 1-3 variables get random values (1-7)
        state = [BG] * n_vars
        n_init = random.randint(1, min(3, n_vars))
        init_vars = random.sample(range(n_vars), n_init)
        for v in init_vars:
            state[v] = random.randint(1, 7)

        inp = _empty_grid(H, W, BG)
        out = _empty_grid(H, W, BG)

        # Row 0: initial state
        for x in range(n_vars):
            inp[0][x] = state[x]
            out[0][x] = state[x]

        # Generate operations
        operations = []
        for op_idx in range(n_ops):
            op_type = random.choice(['set', 'copy', 'conditional'])
            target_var = random.randint(0, n_vars - 1)

            if op_type == 'set':
                value = random.randint(1, 7)
                operations.append(('set', target_var, value))
            elif op_type == 'copy':
                # Copy from another variable that has a value
                sources = [v for v in range(n_vars) if state[v] != BG and v != target_var]
                if not sources:
                    # Fallback to set
                    value = random.randint(1, 7)
                    operations.append(('set', target_var, value))
                else:
                    source_var = random.choice(sources)
                    operations.append(('copy', target_var, source_var))
            else:  # conditional
                sources = [v for v in range(n_vars) if state[v] != BG and v != target_var]
                if not sources:
                    value = random.randint(1, 7)
                    operations.append(('set', target_var, value))
                else:
                    source_var = random.choice(sources)
                    operations.append(('conditional', target_var, source_var))

            # Execute operation on state
            op = operations[-1]
            if op[0] == 'set':
                state[op[1]] = op[2]
            elif op[0] == 'copy':
                state[op[1]] = state[op[2]]
            elif op[0] == 'conditional':
                if state[op[2]] > 2:
                    state[op[1]] = state[op[2]]

        # Encode operations into grid rows (SHUFFLED order in input)
        op_order = list(range(n_ops))
        # DON'T shuffle — the temporal order matters for state tracking
        # But vary the specific operations across demos

        for row_idx, op_idx in enumerate(op_order):
            op = operations[op_idx]
            row = row_idx + 1  # skip initial state row

            if op[0] == 'set':
                inp[row][op[1]] = op[2]
                out[row][op[1]] = op[2]
            elif op[0] == 'copy':
                inp[row][op[1]] = COPY_MARKER
                inp[row][op[2]] = COPY_MARKER  # mark source too
                out[row][op[1]] = COPY_MARKER
                out[row][op[2]] = COPY_MARKER
            elif op[0] == 'conditional':
                inp[row][op[1]] = COND_MARKER
                inp[row][op[2]] = COND_MARKER
                out[row][op[1]] = COND_MARKER
                out[row][op[2]] = COND_MARKER

        # Result row (last row): final state
        result_row = H - 1
        for x in range(n_vars):
            # Input: blank (to be predicted)
            inp[result_row][x] = BG
            # Output: final state values
            out[result_row][x] = state[x]

        return inp, out

    def _generate_one(self):
        random.seed(self._next_seed())
        n_vars = random.randint(self.min_vars, self.max_vars)
        n_ops = random.randint(self.min_ops, self.max_ops)

        demos = []
        for _ in range(self.n_demos):
            inp, out = self._generate_pair(n_vars, n_ops)
            demos.append((inp, out))

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

    def generate_batch(self, batch_size, device=None):
        """Agent: copy tensor construction from ProceduralARCTask.generate_batch()"""
        raise NotImplementedError(
            "Copy tensor construction from ProceduralARCTask.generate_batch() "
            "in liquid_arc/tasks/procedural.py"
        )
```

### Task 2: Context Relevance Filtering

**What it tests:** Given a query and a set of context items, identify which items are relevant. This is the ATTENTION problem in agentic state management — the model must learn to route information selectively based on semantic relevance, not spatial proximity.

**Design:**
```
Input grid (8 rows × 6 cols):
  Rows 0-5: Context items — each row is a "fact" encoded as colored tokens.
            Some facts are relevant to the query, some aren't.
  Row 6:    Query — a pattern encoded as colored tokens
  Row 7:    [0, 0, 0, 0, 0, 0]  ← answer row (blank)

Output grid:
  Rows 0-6: same as input
  Row 7:    Relevant facts' colors reproduced (the facts that match
            the query's pattern/category)
```

The "relevance" is defined by a shared attribute. The query has a "key" color. Context rows that contain that key color are relevant. The answer row reproduces the relevant facts' distinguishing values. This tests whether the metric learns to route the query token's attention to context rows that share its key attribute, skipping rows that don't.

```python
"""Context relevance filtering — selective attention over information.

Input grid encodes context items and a query. The model must identify
which context items are relevant to the query and reproduce their
key values in the answer row.

Relevance is determined by a shared attribute (color) between query
and context items — NOT by spatial position. Context items are in
random positions and relevance depends on content matching.

This tests the core agentic filtering capability: given a current
focus (query/goal), which stored context items should be attended to?
"""

import random
from typing import Dict
from liquid_arc.tasks.procedural import (
    build_sequence, N_COLORS, _empty_grid,
)

BG = 0
QUERY_MARKER = 9  # marks the query row


class ContextRelevanceTask:
    def __init__(self, seq_len=2048, n_demos=2, min_items=4, max_items=7,
                 min_relevant=1, max_relevant=3, **kwargs):
        self.seq_len = seq_len
        self.n_demos = n_demos
        self.min_items = min_items
        self.max_items = max_items
        self.min_relevant = min_relevant
        self.max_relevant = max_relevant
        self._seed_counter = random.randint(0, 2**31)

    def _next_seed(self):
        self._seed_counter += 1
        return self._seed_counter

    def _generate_pair(self, n_items, n_relevant):
        """Generate one input/output pair.

        Each context row: [category_color, value_color, BG, BG, ...]
        Query row: [QUERY_MARKER, target_category, BG, BG, ...]
        Answer row: [value1, value2, value3, BG, ...]  (values from matching items)
        """
        W = max(6, n_items)
        H = n_items + 2  # context rows + query row + answer row

        # Generate categories and values
        available_colors = list(range(1, 8))  # 1-7 for categories and values
        categories = random.sample(available_colors, min(4, len(available_colors)))

        # Assign each context item a category and a unique value
        items = []
        for i in range(n_items):
            cat = random.choice(categories)
            val = random.randint(1, 7)
            items.append((cat, val))

        # Pick a query category
        query_cat = random.choice(categories)

        # Find relevant items (those matching query category)
        relevant_indices = [i for i, (cat, val) in enumerate(items) if cat == query_cat]

        # Ensure we have at least min_relevant matches
        while len(relevant_indices) < n_relevant:
            # Force one item to match
            idx = random.randint(0, n_items - 1)
            if idx not in relevant_indices:
                items[idx] = (query_cat, items[idx][1])
                relevant_indices.append(idx)

        # Shuffle context item order
        order = list(range(n_items))
        random.shuffle(order)

        inp = _empty_grid(H, W, BG)
        out = _empty_grid(H, W, BG)

        # Context rows (shuffled)
        for row_idx, item_idx in enumerate(order):
            cat, val = items[item_idx]
            inp[row_idx][0] = cat
            inp[row_idx][1] = val
            out[row_idx][0] = cat
            out[row_idx][1] = val

        # Query row
        query_row = n_items
        inp[query_row][0] = QUERY_MARKER
        inp[query_row][1] = query_cat
        out[query_row][0] = QUERY_MARKER
        out[query_row][1] = query_cat

        # Answer row — values from relevant items (sorted for determinism)
        answer_row = n_items + 1
        relevant_values = sorted([items[i][1] for i in relevant_indices])
        for col, val in enumerate(relevant_values):
            if col < W:
                out[answer_row][col] = val
            # Input answer row stays blank (to be predicted)

        return inp, out

    def _generate_one(self):
        random.seed(self._next_seed())
        n_items = random.randint(self.min_items, self.max_items)
        n_relevant = random.randint(self.min_relevant, self.max_relevant)

        demos = []
        for _ in range(self.n_demos):
            inp, out = self._generate_pair(n_items, n_relevant)
            demos.append((inp, out))

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

    def generate_batch(self, batch_size, device=None):
        raise NotImplementedError(
            "Copy tensor construction from ProceduralARCTask.generate_batch()"
        )
```

### Task 3: Dependency Ordering

**What it tests:** Given a set of tasks with dependency relationships, determine the valid execution order. This tests the planning/scheduling aspect of agentic state management — the model must understand which actions can proceed and which are blocked.

**Design:**
```
Input grid (8 rows × 6 cols):
  Rows 0-4: Dependency declarations.
            Each row: [task_color, ARROW_MARKER, prereq_color, BG, ...]
            Meaning: task_color depends on prereq_color (must come after)
  Row 5:    [STAR_MARKER, BG, ...]  ← separator
  Row 6-7:  [BG, ...]  ← answer rows (blank)

Output grid:
  Rows 0-5: same as input
  Row 6:    [task_A, task_B, task_C, BG, ...]  ← valid topological order (first group)
  Row 7:    [task_D, task_E, BG, ...]           ← valid order (second group)
```

The model must perform topological sorting on a dependency graph encoded as grid rows. This is directly analogous to agentic task scheduling — determining which actions are ready to execute given completed prerequisites.

```python
"""Dependency ordering — topological sort of task graph.

Input encodes dependency relationships between tasks (as colored tokens).
Output shows a valid execution order respecting all dependencies.

This tests the planning/scheduling aspect of agentic state management:
given a DAG of task dependencies, determine execution order.

Deps are in shuffled row order — the model must trace the graph structure,
not rely on row positions. Different tasks have different colors across
demos to prevent color memorization.
"""

import random
from collections import defaultdict, deque
from typing import Dict, List, Tuple
from liquid_arc.tasks.procedural import (
    build_sequence, N_COLORS, _empty_grid,
)

BG = 0
ARROW_MARKER = 8  # "depends on" marker
STAR_MARKER = 9   # separator


class DependencyOrderTask:
    def __init__(self, seq_len=2048, n_demos=2, min_tasks=3, max_tasks=6, **kwargs):
        self.seq_len = seq_len
        self.n_demos = n_demos
        self.min_tasks = min_tasks
        self.max_tasks = max_tasks
        self._seed_counter = random.randint(0, 2**31)

    def _next_seed(self):
        self._seed_counter += 1
        return self._seed_counter

    def _generate_dag(self, n_tasks, n_edges):
        """Generate a random DAG with topological ordering."""
        # Generate random edges respecting natural ordering (i → j where i < j)
        # This ensures the graph is a DAG
        possible_edges = [(i, j) for i in range(n_tasks) for j in range(i+1, n_tasks)]
        random.shuffle(possible_edges)
        edges = possible_edges[:min(n_edges, len(possible_edges))]

        # Compute topological order using Kahn's algorithm
        in_degree = [0] * n_tasks
        adj = defaultdict(list)
        for prereq, task in edges:
            adj[prereq].append(task)
            in_degree[task] += 1

        queue = deque([i for i in range(n_tasks) if in_degree[i] == 0])
        topo_order = []
        while queue:
            # For determinism within a level, sort by index
            level = sorted(queue)
            queue.clear()
            topo_order.extend(level)
            for node in level:
                for neighbor in adj[node]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

        return edges, topo_order

    def _generate_pair(self, n_tasks):
        """Generate one input/output grid pair."""
        n_edges = random.randint(n_tasks - 1, n_tasks * 2)
        edges, topo_order = self._generate_dag(n_tasks, n_edges)

        # Assign random colors to tasks (avoid markers)
        available = [c for c in range(1, 8) if c not in (ARROW_MARKER, STAR_MARKER)]
        task_colors = random.sample(available, min(n_tasks, len(available)))
        while len(task_colors) < n_tasks:
            task_colors.append(random.choice(available))

        W = max(6, n_tasks)
        n_dep_rows = len(edges)
        H = n_dep_rows + 1 + 2  # dep rows + separator + 2 answer rows

        inp = _empty_grid(H, W, BG)
        out = _empty_grid(H, W, BG)

        # Dependency rows (shuffled order — position shouldn't matter)
        dep_order = list(range(len(edges)))
        random.shuffle(dep_order)

        for row_idx, edge_idx in enumerate(dep_order):
            prereq, task = edges[edge_idx]
            inp[row_idx][0] = task_colors[task]
            inp[row_idx][1] = ARROW_MARKER
            inp[row_idx][2] = task_colors[prereq]  # "task depends on prereq"
            out[row_idx][0] = task_colors[task]
            out[row_idx][1] = ARROW_MARKER
            out[row_idx][2] = task_colors[prereq]

        # Separator row
        sep_row = n_dep_rows
        inp[sep_row][0] = STAR_MARKER
        out[sep_row][0] = STAR_MARKER

        # Answer rows: topological order split across 2 rows
        mid = (len(topo_order) + 1) // 2
        first_half = topo_order[:mid]
        second_half = topo_order[mid:]

        for col, task_idx in enumerate(first_half):
            if col < W:
                out[sep_row + 1][col] = task_colors[task_idx]
        for col, task_idx in enumerate(second_half):
            if col < W:
                out[sep_row + 2][col] = task_colors[task_idx]

        return inp, out

    def _generate_one(self):
        random.seed(self._next_seed())
        n_tasks = random.randint(self.min_tasks, self.max_tasks)

        demos = []
        for _ in range(self.n_demos):
            inp, out = self._generate_pair(n_tasks)
            demos.append((inp, out))

        test_inp, test_out = self._generate_pair(n_tasks)
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

    def generate_batch(self, batch_size, device=None):
        raise NotImplementedError(
            "Copy tensor construction from ProceduralARCTask.generate_batch()"
        )
```

## Implementation Steps

### Step 1: Create Task Files

Place all three task generators in `liquid_arc/tasks/`:
- `liquid_arc/tasks/stateful_execution.py`
- `liquid_arc/tasks/context_relevance.py`
- `liquid_arc/tasks/dependency_order.py`

Each needs a working `generate_batch()` — copy the tensor construction from `ProceduralARCTask.generate_batch()` in `liquid_arc/tasks/procedural.py`. Same approach as the universality probe tasks.

### Step 2: Validate Task Generators

```bash
python -c "
from liquid_arc.tasks.stateful_execution import StatefulExecutionTask
from liquid_arc.tasks.context_relevance import ContextRelevanceTask
from liquid_arc.tasks.dependency_order import DependencyOrderTask

for name, TaskClass in [
    ('stateful', StatefulExecutionTask),
    ('context', ContextRelevanceTask),
    ('dependency', DependencyOrderTask),
]:
    task = TaskClass()
    for i in range(100):
        seq = task._generate_one()
        assert seq['length'] <= 2048, f'{name} sample {i} too long: {seq[\"length\"]}'
    print(f'{name}: 100 samples validated')
"
```

Also validate semantic correctness: for each task type, generate 10 samples, print the input/output grids in readable form, and visually verify:
- Stateful: final state row correctly reflects accumulated operations
- Context: answer row contains exactly the values from relevant items
- Dependency: output order respects all dependency constraints

### Step 3: Single-Domain Transfer Runs

**Same protocol as the universality probe.** For each domain, two conditions:

**Transfer (from 5M post-transition checkpoint):**
```bash
python scripts/train.py \
  --config configs/agentic_${DOMAIN}.yaml \
  --data_dir /workspace/fgn-v3/data/arc-repo/data \
  --output_dir output_agentic/${DOMAIN}_transfer \
  --resume [5M_POST_TRANSITION_CHECKPOINT] \
  --max_steps 5000 \
  --log_every 50 \
  --eval_every 250 \
  --save_every 1000
```

NOTE: eval_every=250 (more frequent than universality probe) because the universality probe showed these domains can converge in <500 steps. We need finer-grained observation of the learning trajectory.

Create per-domain configs following the universality probe pattern:
```yaml
# configs/agentic_stateful.yaml (etc.)
# Same base as liquid_arc_5m.yaml
# Override data section:
use_procedural: false
use_cellular_automata: false
use_conditional_transforms: false
real_arc_mix_ratio: 0.0
use_stateful_execution: true
stateful_ratio: 1.0
```

### Step 4: Combined Multi-Domain Agentic Training

After single-domain results are in, train all three agentic tasks simultaneously:

```bash
python scripts/train.py \
  --config configs/agentic_combined.yaml \
  --data_dir /workspace/fgn-v3/data/arc-repo/data \
  --output_dir output_agentic/combined_transfer \
  --resume [5M_POST_TRANSITION_CHECKPOINT] \
  --max_steps 3000 \
  --log_every 50 \
  --eval_every 250 \
  --save_every 1000
```

Config for combined:
```yaml
use_stateful_execution: true
use_context_relevance: true
use_dependency_order: true
stateful_ratio: 0.34
context_ratio: 0.33
dependency_ratio: 0.33
```

### Step 5: Cross-Domain Interference Check

Critical test: does adding agentic tasks DEGRADE the existing spatial capabilities?

Take the combined agentic checkpoint and evaluate on:
- ARC eval (should still be ~44-45%)
- Sorting eval (should still be ~63%)
- Logic eval (should still be ~61%)

If any of these drop significantly, there IS cross-domain interference between agentic and spatial/relational tasks. If they hold, the model is successfully accumulating capabilities without forgetting.

### Step 6: Structural Analysis (if time permits)

Run the same gradient cosine and FFN subspace overlap analysis as the universality probe:
- Are agentic tasks using the SAME neurons as spatial/relational tasks (shared vocabulary)?
- Or do they develop new circuits (partition)?
- Which is it: composition (like the universality probe found) or partition (like the spatial multi-domain found)?

This determines whether agentic reasoning is "more of the same" computational vocabulary or a genuinely new type of computation for the FFN.

## What to Monitor

### Per-Domain Learning Trajectories

```
| Step | Stateful xform | Context xform | Dependency xform |
|------|---------------|---------------|------------------|
| 0    |               |               |                  |
| 50   |               |               |                  |
| 100  |               |               |                  |
| 250  |               |               |                  |
| 500  |               |               |                  |
| 1000 |               |               |                  |
| 2000 |               |               |                  |
| 5000 |               |               |                  |
```

### CV and Tau Behavior

Record CV and tau mean/σ at each logging step. Key questions:
- Does CV shift from the spatial pre-trained level (~6-7)?
- Does tau adapt differently for stateful tasks (which have temporal ordering) vs context tasks (which are more parallel)?

### Eval Accuracy (Generalization)

Run eval on held-out samples from each task:

```
| Domain    | Train xform @1000 | Eval xform @1000 | Gap |
|-----------|-------------------|------------------|-----|
| Stateful  |                   |                  |     |
| Context   |                   |                  |     |
| Dependency|                   |                  |     |
```

A large train-eval gap means the model is memorizing task instances rather than learning the rule. A small gap means genuine learning.

## Success Criteria

### Transfer Speed Expectations (based on universality probe)

| Domain | Expected steps to 60% | Reasoning |
|---|---|---|
| Context relevance | 50-100 | Similar to sorting — matching tokens by attribute |
| Dependency ordering | 200-400 | Similar to logic inference — chain/graph following |
| Stateful execution | 300-500 | Most complex — requires multi-step state tracking |

If ALL three domains reach 60%+ in <500 steps from the post-transition checkpoint, the geometric substrate is confirmed universal for agentic-analog tasks.

### What Strong Success Looks Like

- All three domains reach >60% eval in <500 steps (fast transfer)
- Combined training shows no cross-domain interference
- Existing spatial/relational capabilities preserved (ARC eval stable)
- FFN analysis shows shared neuron usage (composition, not partition)

This would validate that the post-transition 5M model is a viable agentic state controller substrate — a single model that can simultaneously manage spatial reasoning, logical inference, AND multi-step state tracking through modular composition on a shared geometric substrate.

### What Failure Would Tell Us

- If stateful execution fails (the state doesn't propagate correctly through the ODE): the architecture lacks the temporal chaining capability needed for multi-step state tracking. Persistent ODE state (h₀ blending) would be needed to extend the temporal horizon beyond what 16 ODE steps provide.
- If context relevance fails: the metric can't learn content-based filtering (only spatial/structural routing). Agentic state management would need a different attention mechanism.
- If dependency ordering fails: graph-structure reasoning doesn't transfer from the simpler graph coloring task. More complex relational reasoning may need architectural additions.

Each failure mode points to a specific next step, so negative results are as informative as positive ones.

## Output

Report to `shared/outbox/AGENTIC_STATE_REPORT.md`

Include:
1. Per-domain learning trajectories (transfer condition)
2. Transfer speed comparison to universality probe domains
3. CV/tau behavior during training
4. Combined multi-domain results
5. Cross-domain interference check (existing capabilities preserved?)
6. Structural analysis (shared neurons vs partition?)
7. Assessment: is the post-transition model viable as an agentic state controller substrate?
8. Specific next steps based on which tasks succeed/fail
