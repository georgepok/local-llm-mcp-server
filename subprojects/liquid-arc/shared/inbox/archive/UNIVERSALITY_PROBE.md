# TASK: Geometric Substrate Universality Probe — Non-Spatial Domain Acquisition Test

**From:** Claude Desktop (Research Direction)
**To:** Claude Code (Implementation)
**Date:** 2026-03-27

---

## The Core Question

The post-transition LiquidARC model developed a universal geometric routing substrate that rapidly acquires new spatial skills (CA reached 80% in ~2500 steps on the 572K checkpoint; the 5M model acquires skills even faster). But ALL tested domains so far are spatial grid tasks — procedural, CA, conditional, ARC. The metric learned spatial proximity. New spatial tasks benefited from spatial routing.

**Does the geometric substrate generalize BEYOND spatial tasks?** If the metric learned a general principle of "information relevance between entities" rather than specifically "spatial proximity between grid cells," it should transfer to non-spatial relational tasks. If it's spatial-specific, non-spatial tasks will show no transfer benefit — or the pre-trained geometry might actively interfere.

This experiment answers that question definitively.

## Experimental Design

**Two conditions for each new domain:**

1. **Transfer:** Load post-transition 5M checkpoint → train on new domain → measure steps to threshold performance
2. **Baseline:** Initialize 5M model from scratch (random weights) → train on same domain through phase transition → measure steps to same threshold

The ratio `steps_baseline / steps_transfer` is the **transfer coefficient**. Values >1 mean the geometric substrate helps. Values ~1 mean no benefit. Values <1 mean the substrate interferes.

**Use the 5M model** (d=768). Use the post-transition 5M multi-domain checkpoint. If that specific checkpoint isn't available, use any post-transition 5M checkpoint. For the baseline condition: initialize a fresh 5M model with the same architecture and train from random initialization.

## Four Test Domains

Each domain tests a different type of relational structure while using the same token-sequence format LiquidARC already processes. The input/output format is identical to ARC — sequences of (color, x, y, role, grid_id) tokens. Only the TASK SEMANTICS differ.

### Domain 1: Sequence Sorting

**What it tests:** Ordinal relations. No spatial structure — the "correct" output depends on value ordering, not position.

Input: A 1×N grid (single row) with colored cells in random order. Colors represent values (color 1 < color 2 < ... < color 9). Output: Same cells sorted by color value (ascending). Demo pairs show the sorting rule. Test input is a new random permutation.

The spatial positions (x coordinates) are IRRELEVANT to the rule — the rule depends on COLOR VALUES. The metric must learn that tokens should communicate based on relative value, not spatial adjacency.

### Domain 2: Logical Inference

**What it tests:** Inference chain following. Tokens represent propositions; the rule is logical implication, not spatial transformation.

Input grid encodes a set of implications and a starting fact:
- Row 0: "A → B" encoded as [color_A, arrow_marker, color_B, bg, ...]
- Row 1: "B → C" encoded as [color_B, arrow_marker, color_C, bg, ...]
- Last input row: "START: A" encoded as [color_A, star_marker, bg, ...]

Output grid: same as input but with a "RESULT" row showing all reachable propositions.

The implications are presented in SHUFFLED row order across demos — the model can't rely on row position to determine the inference chain. The metric must learn that propositions connected by implication chains should communicate regardless of spatial row position.

### Domain 3: Sequence Pattern Completion

**What it tests:** Sequential/temporal dependencies. The pattern is in the ORDER of elements, not their spatial arrangement.

Input: A grid where rows encode a repeating pattern with one row blanked out. Output: Same grid with blank row filled following the pattern. The metric must learn that cells in the same COLUMN across rows should communicate (to detect repetition), regardless of row distance. Tests columnar rather than spatial-neighborhood routing.

### Domain 4: Graph Coloring (Constraint Satisfaction)

**What it tests:** Arbitrary graph relationships encoded on a grid.

Input grid encodes a graph and partial coloring:
- Top half: adjacency matrix (edge markers show which nodes connect)
- Bottom half: node coloring (some pre-colored, others blank)

Output: all nodes colored such that no adjacent nodes share colors. The constraint structure is a GRAPH, not a spatial neighborhood. The metric must learn that adjacency in the graph (top half) determines which node colorings (bottom half) constrain each other.

## Implementation

### Step 1: Create Task Files

Create four task generator files in `liquid_arc/tasks/`:
- `liquid_arc/tasks/sorting.py`
- `liquid_arc/tasks/logic_inference.py`
- `liquid_arc/tasks/pattern_completion.py`
- `liquid_arc/tasks/graph_coloring.py`

Each must implement:
- A task class with `_generate_one()` returning a sequence dict (same format as ProceduralARCTask)
- A working `generate_batch(batch_size, device)` method — copy the tensor construction logic from `ProceduralARCTask.generate_batch()` in `liquid_arc/tasks/procedural.py`
- Use `build_sequence()` from procedural.py to construct the demo+test token sequences
- Color/position encoding is identical to ARC tasks — the same embedding layer processes these tokens

**Key design principle:** All tasks use the SAME token format (color, x, y, role, grid_id). The tasks are encoded as grids and use existing grid infrastructure. What differs is the SEMANTIC STRUCTURE — which tokens are relevant to which, and why. The metric's job is to discover this relevance from the task structure.

**Sorting implementation details:**
- 1×N grids, N in [4, 12]
- Colors 1-9 represent ordinal values
- Input: shuffled sequence. Output: sorted sequence
- Augment with color permutation (so the model learns "sort by whatever ordering the demos show" not "sort by absolute color index")

**Logic inference implementation details:**
- Use two reserved colors as markers: one for "→" (implication arrow) and one for "START"
- Chain lengths 2-5 propositions
- Shuffle row order of implications in each demo (prevents positional shortcutting)
- Different proposition colors between demos and test (model must learn the STRUCTURE, not the specific colors)

**Pattern completion implementation details:**
- Repeating color patterns with period 2-4
- 3-6 rows (repetitions), one blanked
- Different blank position and different palette across demos
- Width is 1-2x the period

**Graph coloring implementation details:**
- 3-6 nodes, random edges with ~40% connection probability
- 3-colorable (use greedy coloring to generate solutions)
- Top half of grid: adjacency matrix with edge marker color
- Bottom half: node colors (some given, some to predict)
- Re-color and re-select given nodes across demos (model must learn the constraint structure, not memorize specific colorings)

### Step 2: Validate Task Generators

Before any training, generate 100 samples from each task:
```bash
python -c "
from liquid_arc.tasks.sorting import SortingTask
from liquid_arc.tasks.logic_inference import LogicInferenceTask
from liquid_arc.tasks.pattern_completion import PatternCompletionTask
from liquid_arc.tasks.graph_coloring import GraphColoringTask
import torch

for name, TaskClass in [
    ('sorting', SortingTask),
    ('logic', LogicInferenceTask),
    ('pattern', PatternCompletionTask),
    ('graph', GraphColoringTask),
]:
    task = TaskClass()
    # Test _generate_one
    for i in range(100):
        seq = task._generate_one()
        assert seq['length'] <= 2048, f'{name} sample {i} too long: {seq[\"length\"]}'
    # Test generate_batch
    batch = task.generate_batch(4, device=torch.device('cpu'))
    print(f'{name}: 100 samples + batch validated')
"
```

### Step 3: Modify Training Loop for Single-Domain Mode

The training script needs to support running on a SINGLE non-standard domain. The simplest approach: add config flags for each new domain.

```yaml
# Example: configs/universality_sorting.yaml
# Copy from liquid_arc_5m.yaml, then override data section:
use_procedural: false
use_cellular_automata: false
use_conditional_transforms: false
real_arc_mix_ratio: 0.0
use_sorting: true
sorting_ratio: 1.0
```

Create one config per domain. If the training script doesn't support these flags, modify the data sampling section to accept a `--domain` argument:

```bash
python scripts/train.py --domain sorting ...
```

### Step 4: Run Experiments

**For EACH of the 4 domains, run 2 conditions:**

**A) Transfer (from 5M post-transition checkpoint):**
```bash
python scripts/train.py \
  --config configs/universality_${DOMAIN}.yaml \
  --data_dir /workspace/fgn-v3/data/arc-repo/data \
  --output_dir output_universality/${DOMAIN}_transfer \
  --resume [PATH_TO_5M_POST_TRANSITION_CHECKPOINT] \
  --max_steps 5000 \
  --log_every 50 \
  --eval_every 500 \
  --save_every 2500
```

**B) Baseline (from scratch):**
```bash
python scripts/train.py \
  --config configs/universality_${DOMAIN}.yaml \
  --data_dir /workspace/fgn-v3/data/arc-repo/data \
  --output_dir output_universality/${DOMAIN}_baseline \
  --max_steps 15000 \
  --log_every 50 \
  --eval_every 500 \
  --save_every 2500
```

Baseline needs 15000 steps to allow for phase transition (~5000 steps) plus post-transition learning.

Run all 8 experiments sequentially:
```bash
for domain in sorting logic pattern graph; do
    echo "=== TRANSFER: $domain ==="
    # [transfer run command]
    echo "=== BASELINE: $domain ==="
    # [baseline run command]
done
```

## What to Monitor

### Primary: Training Transform Accuracy Trajectory

For each domain, record xform_acc at regular intervals for both conditions:

```
Domain: [name]
| Step  | Transfer | Baseline |
|-------|----------|----------|
| 500   |          |          |
| 1000  |          |          |
| 2000  |          |          |
| 3000  |          |          |
| 5000  |          |          |
| 7500  |          |          |
| 10000 |          |          |
| 15000 |          |          |
```

### Secondary: CV Behavior

**Transfer condition:** Does CV remain at ~6-7 (pre-trained level) or shift? A shift means the metric is ADAPTING its geometry for the new domain. Stable CV means the pre-trained geometry is being reused as-is.

**Baseline condition:** Does the CV-driven phase transition fire for non-spatial tasks? If yes: at what step and CV threshold? If the phase transition fires for sorting (pure ordinal, no spatial structure), that's strong evidence the mechanism is domain-general.

### Tertiary: Tau and Curvature

- Does tau distribution change differently for non-spatial tasks?
- Does curvature (|κ|) develop different patterns than for spatial tasks?
- Record tau mean, σ, and |κ| at each logging step for both conditions

### Per-Domain Logging

Use the existing domain logging format. Each log line should show:
```
[domains] (last 500 steps)
  sorting     : xform= XX.X%  cv=X.XX  n=NNN
```

## Success Criteria

### Transfer Coefficients

For each domain, compute: `steps_to_60%_baseline / steps_to_60%_transfer`

(Use whatever threshold is reachable by both conditions. 60% is a rough target; adjust based on task difficulty.)

| Domain   | Transfer Coeff | Interpretation |
|----------|---------------|----------------|
| >3×      | Strong transfer: geometry aids this domain type |
| 1-3×     | Modest: geometry doesn't hurt, slight help |
| ~1×      | No transfer: geometry is irrelevant |
| <1×      | Negative transfer: spatial geometry interferes |

### Overall Assessment

- **Universal substrate (all domains >3×):** The geometric routing principle generalizes beyond spatial tasks. Validates using the post-transition model for robotics, agentic state management, and other non-spatial applications.

- **Spatial-specific (spatial-like domains help, others don't):** Pattern completion might benefit (columnar spatial structure) while sorting and logic don't. Non-spatial applications need different geometric substrate.

- **Interference (any domain <1×):** Pre-trained spatial geometry actively hurts some domains. The model ISN'T a universal platform. Domain-specific geometric substrates are needed.

### What If Phase Transitions Don't Fire for Non-Spatial Baselines?

If the baseline (from-scratch) runs for sorting/logic DON'T produce a phase transition, that itself is a major finding: the CV-driven phase transition may be specific to tasks with spatial structure that the heat kernel can exploit. Report this prominently — it constrains the architecture's applicability.

## Output

Report to `shared/outbox/UNIVERSALITY_PROBE_REPORT.md`

Include:
1. Per-domain training accuracy trajectories (transfer vs baseline), with plots if possible
2. Transfer coefficients table (the key result)
3. CV trajectories for both conditions across all domains
4. Phase transition analysis: did it fire for non-spatial baselines?
5. Tau and curvature adaptation analysis
6. Overall assessment: universal, spatial-specific, or interference?
7. Recommendations for next direction based on results

**This experiment has maximum diagnostic value for minimum infrastructure investment. The results determine whether to pursue robotics (Isaac Sim), agentic state management, or both — and whether the post-transition model is the right starting point.**
