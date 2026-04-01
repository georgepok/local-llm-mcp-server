# CURRENT TASK: Multi-Domain Ratcheting — Computational Generalization via Build-Disrupt Dynamics

## The Core Insight

The phase transition was caused by a specific dynamical regime: 70% procedural batches BUILD coherent metric structure, 30% ARC batches DISRUPT task-specific structure. Only task-INVARIANT structure survives the disruption cycles. Over 5000 steps, this ratcheting accumulated enough general routing structure to cross the critical CV threshold.

**The evidence:** Post-transition, the CV oscillation between procedural and ARC batches dropped from 1.5 units to 0.5 units. The model didn't just develop routing contrast — it developed routing that GENERALIZES across task types. The transition was a generalization event, not just a capability event.

**The hypothesis:** The same build-disrupt ratcheting principle can push COMPUTATIONAL generalization post-transition. If the "building" phase uses diverse spatial tasks (procedural ARC + cellular automata + new task types), the FFN must develop computation that works across ALL of them. The 30% real ARC "disruption" then selects for computation that transfers to novel patterns. Over time, the comprehended fraction of ARC tasks should grow from 36% toward higher percentages.

## Why This Is Different From Simply Adding CA

Adding CA to a 50/50 mix would be like adding kindergarten math — below the model's capability, no productive tension. The key is the RHYTHM:

```
Pre-transition recipe (produced routing generalization):
  70% procedural (coherent building) + 30% ARC (disruption/testing)
  
Post-transition recipe (targeting computational generalization):
  40% procedural ARC + 15% cellular automata + 15% NEW spatial tasks (diverse building)
  + 30% real ARC (disruption/testing)
```

The building phase must be DIVERSE enough that consecutive batches exercise different computational primitives (global transforms from procedural, local counting from CA, conditional logic from new tasks). But each individual batch must be SOLVABLE so the gradient signal is useful. The ARC disruption tests whether the accumulated computation transfers.

## Implementation

### 1. New Task Type: Conditional Grid Transforms

Add a task generator that requires conditional logic — something between CA's local rules and ARC's global patterns:

**Conditional Recolor:** Given a grid with colored regions, recolor each cell based on a condition involving its neighbors:
- "If a cell has more red neighbors than blue, make it green"
- "If a cell is on the border of a region, change to marker color"
- "If a cell's color matches the majority color in its row, keep it; else flip to background"

These require the model to: (1) identify spatial relationships (routing — already good), (2) evaluate conditions per cell (computation — the bottleneck), (3) apply the conditional output.

Create `liquid_arc/tasks/conditional_transforms.py`:

```python
"""Conditional grid transform tasks.

Each task presents demo pairs showing a conditional recolor rule:
  - Demo shows input grid → output grid under some condition
  - Model must identify the condition and apply to test grid

Conditions involve local neighborhoods (like CA) but with COLOR-DEPENDENT
logic (unlike CA's pure counting). This exercises conditional computation
that neither procedural ARC nor CA alone requires.
"""

import random
from typing import List, Tuple, Dict
from .procedural import (
    build_sequence, N_COLORS, PAD_COLOR, PAD_COORD,
    _empty_grid, _rand_dims, _rand_bg, _rand_palette,
    _augment_grid, _permute_colors,
)


def rule_majority_neighbor(rng_seed):
    """Each cell becomes the majority color among its 4-neighbors.
    Ties keep original color. Background cells don't participate."""
    random.seed(rng_seed)
    H, W = _rand_dims(5, 8)
    bg = _rand_bg()
    palette = _rand_palette(3, exclude=bg)
    
    inp = _empty_grid(H, W, bg)
    for y in range(H):
        for x in range(W):
            if random.random() < 0.4:
                inp[y][x] = random.choice(palette)
    
    out = [row[:] for row in inp]
    for y in range(H):
        for x in range(W):
            if inp[y][x] == bg:
                continue
            # Count neighbor colors (4-connected)
            counts = {}
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = y+dy, x+dx
                if 0 <= ny < H and 0 <= nx < W and inp[ny][nx] != bg:
                    c = inp[ny][nx]
                    counts[c] = counts.get(c, 0) + 1
            if counts:
                max_count = max(counts.values())
                majority = [c for c, n in counts.items() if n == max_count]
                if len(majority) == 1:
                    out[y][x] = majority[0]
                # ties: keep original
    
    return inp, out


def rule_border_mark(rng_seed):
    """Cells adjacent to background become a marker color.
    Interior cells (all 4 neighbors non-bg) keep their color."""
    random.seed(rng_seed)
    H, W = _rand_dims(5, 9)
    bg = _rand_bg()
    fill_color, border_color = _rand_palette(2, exclude=bg)
    
    inp = _empty_grid(H, W, bg)
    # Place a filled region
    rh = random.randint(3, min(6, H))
    rw = random.randint(3, min(6, W))
    ry = random.randint(0, H - rh)
    rx = random.randint(0, W - rw)
    for y in range(ry, ry + rh):
        for x in range(rx, rx + rw):
            inp[y][x] = fill_color
    
    out = [row[:] for row in inp]
    for y in range(H):
        for x in range(W):
            if inp[y][x] == bg:
                continue
            # Check if any neighbor is bg (= border cell)
            is_border = False
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = y+dy, x+dx
                if ny < 0 or ny >= H or nx < 0 or nx >= W:
                    is_border = True  # grid edge counts as border
                elif inp[ny][nx] == bg:
                    is_border = True
            if is_border:
                out[y][x] = border_color
    
    return inp, out


def rule_color_spread(rng_seed):
    """Each non-bg cell spreads its color to adjacent bg cells.
    If multiple colors compete for a bg cell, lowest color index wins."""
    random.seed(rng_seed)
    H, W = _rand_dims(5, 9)
    bg = _rand_bg()
    palette = _rand_palette(3, exclude=bg)
    
    inp = _empty_grid(H, W, bg)
    # Scatter some colored seeds
    n_seeds = random.randint(3, 8)
    for _ in range(n_seeds):
        y, x = random.randint(0, H-1), random.randint(0, W-1)
        inp[y][x] = random.choice(palette)
    
    out = [row[:] for row in inp]
    for y in range(H):
        for x in range(W):
            if inp[y][x] != bg:
                continue
            # Check neighbors for colors
            neighbor_colors = []
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = y+dy, x+dx
                if 0 <= ny < H and 0 <= nx < W and inp[ny][nx] != bg:
                    neighbor_colors.append(inp[ny][nx])
            if neighbor_colors:
                out[y][x] = min(neighbor_colors)  # deterministic: lowest color wins
    
    return inp, out


def rule_row_majority(rng_seed):
    """Each cell becomes the most common non-bg color in its row.
    Requires counting across the full row — more global than neighbor rules."""
    random.seed(rng_seed)
    H, W = _rand_dims(4, 8)
    bg = _rand_bg()
    palette = _rand_palette(3, exclude=bg)
    
    inp = _empty_grid(H, W, bg)
    for y in range(H):
        for x in range(W):
            if random.random() < 0.5:
                inp[y][x] = random.choice(palette)
    
    out = [row[:] for row in inp]
    for y in range(H):
        counts = {}
        for x in range(W):
            if inp[y][x] != bg:
                c = inp[y][x]
                counts[c] = counts.get(c, 0) + 1
        if counts:
            majority = max(counts, key=counts.get)
            for x in range(W):
                if inp[y][x] != bg:
                    out[y][x] = majority
    
    return inp, out


CONDITIONAL_RULES = [
    rule_majority_neighbor,
    rule_border_mark,
    rule_color_spread,
    rule_row_majority,
]


class ConditionalTransformTask:
    """Conditional grid transform task generator.
    Same generate_batch() interface as ProceduralARCTask."""
    
    def __init__(self, seq_len=2048, augment=True, n_demos=2, **kwargs):
        self.seq_len = seq_len
        self.augment = augment
        self.n_demos = n_demos
        self._seed_counter = random.randint(0, 2**31)
        self.rules = CONDITIONAL_RULES
    
    def _next_seed(self):
        self._seed_counter += 1
        return self._seed_counter
    
    def _generate_one(self):
        rule = random.choice(self.rules)
        rot = random.randint(0, 3) if self.augment else 0
        flip = random.random() < 0.5 if self.augment else False
        
        if self.augment:
            perm_vals = list(range(1, N_COLORS))
            random.shuffle(perm_vals)
            color_perm = {0: 0}
            for i, v in enumerate(perm_vals):
                color_perm[i + 1] = v
        else:
            color_perm = {i: i for i in range(N_COLORS)}
        
        demos = []
        for _ in range(self.n_demos):
            seed = self._next_seed()
            inp, out = rule(seed)
            inp = _permute_colors(_augment_grid(inp, rot, flip), color_perm)
            out = _permute_colors(_augment_grid(out, rot, flip), color_perm)
            demos.append((inp, out))
        
        test_seed = self._next_seed()
        test_inp, test_out = rule(test_seed)
        test_inp = _permute_colors(_augment_grid(test_inp, rot, flip), color_perm)
        test_out = _permute_colors(_augment_grid(test_out, rot, flip), color_perm)
        
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
        """Same interface as ProceduralARCTask. Copy the tensor construction
        from ProceduralARCTask.generate_batch() — identical logic."""
        # [Agent: copy the generate_batch tensor construction from 
        #  ProceduralARCTask in procedural.py — it's identical except
        #  _generate_one() produces conditional tasks]
        import torch
        if device is None:
            device = torch.device("cpu")
        
        samples = [self._generate_one() for _ in range(batch_size)]
        max_N = self.seq_len
        
        # [Same tensor construction as ProceduralARCTask.generate_batch()]
        # Copy from procedural.py — colors, xs, ys, roles, sep_mask, etc.
        # Return (input_ids, labels, meta) in the same format.
        raise NotImplementedError("Copy tensor construction from ProceduralARCTask.generate_batch()")
```

The agent should complete `generate_batch()` by copying the tensor construction from `ProceduralARCTask.generate_batch()` in `procedural.py`.

### 2. Multi-Domain Data Sampler

Modify the training loop to sample from THREE task sources with configurable ratios:

```python
# In the training loop:
data_sources = {
    'procedural': (procedural_task, 0.40),    # 40% procedural ARC (coherent spatial building)
    'ca': (ca_task, 0.15),                     # 15% cellular automata (local counting)
    'conditional': (conditional_task, 0.15),   # 15% conditional transforms (conditional logic)
    'real_arc': (real_arc_task, 0.30),          # 30% real ARC (disruption/testing)
}

# Each step: sample a source according to weights
import random
source_name = random.choices(
    list(data_sources.keys()),
    weights=[v[1] for v in data_sources.values()],
    k=1
)[0]
task_source, _ = data_sources[source_name]
_, _, meta = task_source.generate_batch(batch_size, device=device)
```

### 3. Config

Create `configs/multi_domain_ratchet.yaml`:

```yaml
# Multi-Domain Ratcheting — Post-Transition Computational Generalization
#
# Same architecture, same loss, same optimizer as the transition recipe.
# ONLY DIFFERENCE: training data is a mix of 4 spatial task types.
#
# The build-disrupt rhythm:
#   Building (70%): procedural + CA + conditional transforms
#     - Each exercises different computational primitives
#     - FFN must develop general spatial computation
#   Disruption (30%): real ARC
#     - Tests whether computation generalizes to novel patterns
#     - Strips task-specific computation, preserves general
#
# Resume from post-transition checkpoint (step 7500 or best).

d_model: 256
d_metric: 64
d_ffn: 512
max_seq_len: 2048
n_ode_steps: 16
ode_steps_min: 12
ode_steps_max: 20
tau_min: 0.5
tau_max: 1.0
t_diffusion_init: 1.0
chunk_size: 256
dropout: 0.1
curvature_lambda: 0.05
tau_var_lambda: 0.001
use_torch_compile: true
ode_chunk_size: 4
invertible_solver: false
n_fp_iters: 5
deq_solver: false
deq_ift_iters: 10
n_colors: 10
n_roles: 8
n_sep_types: 4
max_grid_size: 30
max_grids: 16
model_type: liquid
transform_weight: 5.0
copy_weight: 0.05
alpha_logit_init: 2.2

# Zero-scaffold (unchanged)
tau_freeze_steps: 0
geo_loss_enabled: false
cv_floor_target: 3.0
cv_ceiling_target: 8.0
cv_floor_lambda: 0.1

# Multi-domain data mixing
use_procedural: true
use_cellular_automata: true
use_conditional_transforms: true
procedural_ratio: 0.40
ca_ratio: 0.15
conditional_ratio: 0.15
real_arc_mix_ratio: 0.30

# Curriculum (procedural only)
curriculum_stage1_end: 20000
curriculum_stage2_end: 100000

# TTT for eval
ttt_enabled: true
ttt_steps: 100
ttt_lr: 0.001
ttt_curvature_lambda: 0.01
ttt_early_stop_threshold: 0.01
```

### 4. Run Command

**CRITICAL: Resume from post-transition checkpoint. Do NOT train from scratch.**

```bash
cd /workspace/liquid-arc
python scripts/train.py \
  --config configs/multi_domain_ratchet.yaml \
  --data_dir /workspace/fgn-v3/data/arc-repo/data \
  --output_dir output_multi_domain \
  --resume output_reproduce/checkpoints/step_7500.pt \
  --max_steps 20000 \
  --save_every 2500 \
  --eval_every 500 \
  --log_every 50
```

If `output_reproduce/checkpoints/step_7500.pt` doesn't exist, use any post-transition checkpoint.

Also run a CONTROL with the standard 30%/70% recipe from the same checkpoint:

```bash
python scripts/train.py \
  --config configs/liquid_arc_zero_scaffold.yaml \
  --data_dir /workspace/fgn-v3/data/arc-repo/data \
  --output_dir output_control_30pct \
  --resume output_reproduce/checkpoints/step_7500.pt \
  --max_steps 20000 \
  --save_every 2500 \
  --eval_every 500 \
  --log_every 50
```

Both from the SAME checkpoint, same total steps. The only difference is the data mix.

### 5. What to Monitor

**Primary: ARC eval xform over time.**

Both runs evaluate on the SAME ARC eval set. The multi-domain run trains on diverse spatial tasks; the control trains on procedural + ARC only. If multi-domain training pushes ARC eval xform higher, the diverse computation transfers.

```
| Step   | Control (30%/70%) | Multi-Domain | Delta |
|--------|------------------|--------------|-------|
| 7500   | (shared start)   | (shared start)|   0  |
| 8000   |                  |              |       |
| 9000   |                  |              |       |
| 10000  |                  |              |       |
| 12500  |                  |              |       |
| 15000  |                  |              |       |
| 20000  |                  |              |       |
```

**Secondary: Per-domain train xform.**

Track training transform accuracy separately for each task type:

```
| Step | Procedural | CA    | Conditional | Real ARC |
|------|-----------|-------|-------------|----------|
| 8000 |           |       |             |          |
| ...  |           |       |             |          |
```

If CA and conditional train xform are high (>70%) while ARC stays at ~40-50%, the model is learning the building tasks without them transferring. If all train xforms improve together, there's cross-domain transfer.

**Tertiary: Verified TTT comparison.**

Run verified TTT on both the multi-domain and control checkpoints at step 15000 and 20000:

```
| Metric                    | Control TTT | Multi-Domain TTT |
|---------------------------|-------------|------------------|
| Tasks passing gate        | 101 (36%)   |                  |
| TTT accuracy on verified  | 77.1%       |                  |
| Overall verified xform    | 48.5%       |                  |
```

**The key question: does the "comprehended" fraction grow?** Currently 36% of ARC tasks are comprehended (TTT helps). If multi-domain training expands this to 42-50%, the model learned computational primitives from CA/conditional tasks that help it understand more ARC patterns. This would be the strongest evidence that the ratcheting produced computational generalization.

**Also monitor: CV and tau stability.**

If CV remains stable at ~5.0-5.3 throughout training, the routing structure is preserved (the building tasks aren't disrupting the phase-transitioned geometry). If CV drops or oscillates, the new tasks are destabilizing the routing — reduce their proportion.

### 6. What Success Looks Like

**Minimum success:** No degradation. Multi-domain matches control on ARC eval. CV stays stable. This means the diverse tasks don't hurt even if they don't help.

**Good success:** Multi-domain exceeds control by 2-5 pp on ARC eval xform (50%+ vs 47%). The diverse computation is transferring. The model learned something from CA/conditional that helps with ARC.

**Strong success:** Verified TTT "comprehended" fraction grows from 36% to 45%+. The model now UNDERSTANDS more ARC tasks — not just slightly better average accuracy, but genuinely handling tasks it couldn't before. This would validate the ratcheting hypothesis: diverse building + ARC disruption selects for general computation.

**Breakthrough:** ARC eval xform exceeds 55% sustained (not just a spike). This would demonstrate that multi-domain ratcheting produces a SECOND capability jump — not a sharp transition like the first, but a steady ratcheting of computational generalization that breaks the single-domain ceiling.

### 7. Why This Could Work Where Memory Failed

Memory modules (v1-v4) tried to ADD computation on top of frozen base. They failed because 40K external params can't improve on 572K co-adapted internal params.

Multi-domain training modifies the BASE MODEL'S OWN PARAMETERS — specifically the FFN (263K params, 46% of the model) and W_v/W_o (131K params, 23%). These are the computation components. The diverse tasks force these parameters to develop general spatial computation through the same ratcheting mechanism that forced MetricNet to develop general routing.

The model's own parameters are where the capacity lives. Training those parameters on diverse tasks is the proven improvement strategy (sequential curriculum: 47.8%, adaptive controller: 55.6% peak). Multi-domain training is the most principled extension of this strategy — diversifying the "building" phase while keeping the "disruption" phase unchanged.

### 8. If It Doesn't Work

If multi-domain training doesn't improve ARC eval:
- The CA and conditional tasks are too easy (model already handles them, no new computation needed)
- The computational primitives from these tasks don't transfer to ARC
- ARC's reasoning requirements (counting, composition, symbolic interpretation) are fundamentally different from any procedural spatial task

In that case, the ceiling is truly in the architecture's computational capacity. The FFN + ODE dynamics can only represent certain classes of spatial computation, and ARC requires classes it can't. The path forward would be scaling (larger d_model → more FFN capacity) or architectural changes (adding discrete computation modules).

But the ratcheting hypothesis deserves testing because it follows directly from the mechanism that produced the only genuine capability improvement we've seen (the phase transition itself).

### Output

Report to shared outbox: `MULTI_DOMAIN_RATCHET_REPORT.md`

Include:
1. Comparison table (multi-domain vs control) at each eval step
2. Per-domain training accuracy breakdown  
3. CV/tau stability throughout training
4. Verified TTT comparison at step 15000/20000 (if TTT eval is available)
5. Assessment: did ratcheting produce computational generalization?
