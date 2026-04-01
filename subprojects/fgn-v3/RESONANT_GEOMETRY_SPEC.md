# Resonant Geometry — Self-Organizing Metric Fields

## Implementation Specification for Dev

**Version:** v0.1 — Seed Experiment  
**Date:** 2026-02-20  
**Base:** FluidNet v1 codebase in `/workspace/fgn-v3/`  
**Goal:** Test whether a metric field, driven by structural energy rather than task loss alone, spontaneously organizes to mirror world geometry — and retains that structure rather than flattening.

---

## 1. Motivation

### 1.1 The Flattening Problem

FluidNet v1 showed that a pure geometric diffusion architecture develops rich curvature (CV=0.33, |κ|=0.60) during early training, then **deflates** as loss converges. By step 3700, curvature is declining because the CE loss no longer rewards geometric complexity. The model voluntarily flattens toward attention-like behavior because the task doesn't require persistent geometric structure.

Meanwhile, flat transformers match geometric models on all metrics — accuracy, generalization, sample efficiency — across every gridworld variant tested (discrete, continuous, v4 through v7).

### 1.2 The Hypothesis

The flattening occurs because CE loss is the **only** training signal. Once the model can predict the next token, geometry becomes overhead. The metric has no reason to maintain structure that the loss doesn't reward.

**Proposed solution:** Add a structural energy term that rewards the metric for aligning its geometry with the actual structure present in the data. This creates a second drive — *resonance* — that maintains geometric structure independently of task performance. The metric can't flatten because flattening would increase structural energy even as it decreases CE loss.

### 1.3 What We're Testing

**Minimal claim:** A metric trained with structural energy will maintain higher CV/|κ| after convergence than one trained with CE alone.

**Stronger claim:** That maintained structure will improve OOD generalization, particularly on tasks where world geometry changes between episodes (multi-metric worlds).

---

## 2. Architecture Changes

### 2.1 Overview

We extend FluidNet v1 with:
1. **Structural Energy** — a self-supervised loss that rewards metric-data alignment
2. **Multi-Metric Worlds** — a new task where connectivity rules change per episode
3. **Resonance monitoring** — new metrics to track whether geometry resonates or flattens

The FluidLayer, FluidNetModel, and training loop are modified. No new layer types.

### 2.2 Structural Energy Function

The core idea: the metric should place tokens that are structurally related (co-occur in relational context) at small geodesic distance, and tokens that are structurally unrelated at large geodesic distance.

**E_structural** measures the alignment between:
- `d_geo(i,j)` — geodesic distance according to the learned metric
- `d_struct(i,j)` — structural distance derived from the data itself

#### Computing d_structural from data (no labels needed):

The [WORLD] prefix already encodes relationships. We extract structural distance from token co-occurrence within relational phrases:

```
"room_0 connects room_1" → d_struct(room_0_pos, room_1_pos) = 1.0
"cup in room_2"          → d_struct(cup_pos, room_2_pos) = 0.5
"room_0" ... "room_5" (no connection stated) → d_struct = large (default)
```

**Implementation approach — Context Window Proximity:**

Rather than parsing semantic relationships (fragile), use a simpler proxy: token distance within the [WORLD] section, weighted by the context mask.

For positions i, j both within the context-masked region:
```
d_struct(i,j) = |i - j| / context_length        (normalized positional distance)
```

For positions where one or both are outside context (action tokens):
```
d_struct(i,j) = 1.0                              (maximum — no structural info)
```

This is crude but grounded: in the world description, related entities (connected rooms, objects in rooms) are described near each other in the text. The structural distance is a proxy for relational proximity.

#### Energy Formulation:

```
E_structural = (1/M) Σ_{i,j ∈ context} (d_geo(i,j) / d_geo_max - d_struct(i,j))²
```

Where:
- Sum is only over context-masked positions (world description)
- `d_geo_max` normalizes geodesic distances to [0,1] range
- M = number of context position pairs

This is a simple MSE between normalized geodesic distances and structural distances. The metric minimizes energy by making its geometry isometric to the data's relational structure.

### 2.3 Combined Loss

```
L_total = L_CE + λ_struct * E_structural
```

Where `λ_struct` is a hyperparameter controlling the resonance drive strength.

**Key design decision:** λ_struct is NOT annealed. It stays constant throughout training. This ensures the structural drive persists after CE converges — preventing the flattening we observed in FluidNet v1.

**Hyperparameter sweep:** Test `λ_struct ∈ {0.0, 0.01, 0.1, 1.0}`:
- `λ_struct = 0.0` — baseline, equivalent to current FluidNet (expect flattening)
- `λ_struct = 0.01` — weak resonance (does it prevent flattening?)
- `λ_struct = 0.1` — moderate resonance
- `λ_struct = 1.0` — strong resonance (does it hurt CE convergence?)

---

## 3. New Task: Multi-Metric Worlds

### 3.1 Motivation

Even with structural energy, a single connectivity rule (Euclidean distance < R) produces one static geometry per episode. The metric learns one mapping and stabilizes. To force the geometry to be truly fluid — dynamically reshaping per episode — we need episodes where the **type** of connectivity changes.

### 3.2 Design

Each episode randomly selects one of several connectivity rules:

**Rule 1 — Proximity (current CW behavior):**
```
Rooms connect if Euclidean distance < R
[WORLD] metric:proximity room_0 pos(23.4,67.1) ...
connected(room_0,room_1,dist=15.2) ...
```

**Rule 2 — Color clusters:**
```
Each room has a color attribute (red, blue, green, yellow)
Rooms connect if same color OR if adjacent in color wheel
[WORLD] metric:color room_0 pos(23.4,67.1) color:red ...
connected(room_0,room_3) ...    (room_3 is also red)
```

**Rule 3 — Grid (Manhattan distance):**
```
Rooms placed on an integer grid
Rooms connect if Manhattan distance ≤ 2
[WORLD] metric:grid room_0 grid(2,3) ...
connected(room_0,room_1) ...
```

**Rule 4 — Hub-spoke:**
```
One hub room connects to all others. Other rooms only connect through hub.
[WORLD] metric:hub hub=room_0 room_0 pos(50,50) ...
connected(room_0,room_1) connected(room_0,room_2) ...
```

The `metric:TYPE` token at the start of [WORLD] tells the model which connectivity rule is in effect. The model must read this, reshape its internal geometry accordingly, and navigate using the appropriate distance concept.

**Critical:** The shortest path changes depending on the metric type. In proximity, nearby rooms are preferred. In color, same-color rooms are preferred regardless of position. In hub-spoke, everything goes through the hub. A flat transformer can solve each by reading the connections. A geometric model can potentially encode the metric type in its geometry — making path planning implicit rather than explicit.

### 3.3 Task Parameters

```
n_rooms_min: 10
n_rooms_max: 15
space_size: 100.0
connect_radius: 30.0        (proximity rule)
n_objects: 4
min_steps: 4
max_steps: 10
min_state_changes: 1
metric_types: ["proximity", "color", "grid", "hub"]
```

For training, all metric types are sampled uniformly. For evaluation, we can test per-metric-type accuracy to see if the model handles each geometry.

### 3.4 Evaluation Conditions

```
ID:         10-15 rooms, all 4 metric types
Near-OOD:   15-20 rooms, all 4 metric types
Near-OOD:   10-15 rooms, new metric type not seen in training (e.g., "ring" — rooms connect to their 2 nearest neighbors in a circular arrangement)
Far-OOD:    20-25 rooms, mixed metric types
```

The **new metric type** condition is the key discriminator. If the geometric model can generalize to an unseen connectivity rule better than the flat model, that demonstrates genuine geometric adaptation — the metric reshaping in response to a novel world structure it hasn't been trained on.

---

## 4. File Changes

### 4.1 New Files

```
fgn/structural_energy.py       — StructuralEnergy module (E_structural computation)
fgn/tasks/multi_metric_world.py — MultiMetricWorld task (extends ContinuousGridWorld)
configs/resonant_6l.yaml        — FluidNet + structural energy config
configs/resonant_flat.yaml      — Flat baseline for multi-metric task
scripts/train_resonant.py       — Training script with structural energy
scripts/eval_resonant.py        — Evaluation with per-metric-type breakdown
scripts/run_resonant_experiment.sh — Full experiment runner
```

### 4.2 Modified Files

```
fgn/model_fluid.py    — Add structural energy to loss computation
fgn/config.py         — Add resonance hyperparameters
```

### 4.3 Unchanged Files (reuse from FluidNet v1)

```
fgn/fluid_layer.py         — FluidLayer (no changes)
fgn/metric.py              — MetricNetwork (no changes)
fgn/context_pool.py        — ContextPool (no changes)  
fgn/curvature.py           — CurvatureEngine (no changes)
fgn/losses.py              — CurvatureRegularization (no changes)
fgn/standard_attention.py  — StandardAttention (no changes, used by flat baseline)
fgn/flat_model.py          — FlatModel (no changes, used by flat baseline)
```

---

## 5. Component Specifications

### 5.1 StructuralEnergy (new file: `fgn/structural_energy.py`)

```
class StructuralEnergy(nn.Module):
    """Computes alignment energy between metric geodesic distances and
    structural distances derived from data.
    
    The structural distance is a proxy for relational proximity:
    positions close together within the [WORLD] section are 
    considered structurally related.
    """
    
    def __init__(self, max_context_pairs: int = 2048):
        """
        Args:
            max_context_pairs: Maximum number of position pairs to sample
                              for energy computation (memory bound).
                              If context has more pairs, subsample randomly.
        """
    
    def forward(
        self, 
        h: torch.Tensor,           # [B, N, d] — hidden states (after embedding)
        g: torch.Tensor,           # [B, N, d] — metric field (per-position)
        context_mask: torch.Tensor, # [B, N]   — True for [WORLD] positions
    ) -> torch.Tensor:
        """Compute structural energy.
        
        Steps:
        1. Extract context positions from h and g using context_mask
        2. Compute pairwise geodesic distances D_geo using metric g
        3. Compute pairwise structural distances D_struct from position indices
        4. Return MSE between normalized D_geo and D_struct
        
        Returns:
            energy: scalar tensor, the structural alignment energy
        """
```

**Implementation details:**

1. **Context extraction:** Use `context_mask` to identify positions within the [WORLD] section. For each batch element, gather those positions.

2. **Geodesic distance:** Same computation as FluidLayer._direct_distance, but only over context positions:
   ```
   diff = h_ctx.unsqueeze(2) - h_ctx.unsqueeze(1)    # [B, C, C, d]
   g_avg = (g_ctx.unsqueeze(2) + g_ctx.unsqueeze(1)) / 2  # [B, C, C, d]
   D_geo = (diff * diff * g_avg).sum(-1)               # [B, C, C]
   ```

3. **Structural distance:** Normalized positional distance within context:
   ```
   pos_indices = context positions (integers)
   D_struct[i,j] = |pos_i - pos_j| / max_context_len   # [B, C, C]
   ```

4. **Normalization:** Normalize D_geo to [0,1] per batch element:
   ```
   D_geo_norm = D_geo / (D_geo.max() + ε)
   ```

5. **Energy:** MSE over all pairs:
   ```
   energy = ((D_geo_norm - D_struct) ** 2).mean()
   ```

6. **Subsampling:** If context has > sqrt(max_context_pairs) positions, randomly sample positions to keep computation bounded. Use `torch.randperm` for random selection.

**Unit test:**
- Create synthetic h, g, context_mask
- Verify energy is 0 when D_geo ∝ D_struct (perfect alignment)
- Verify energy > 0 when D_geo is random
- Verify gradient flows to both h and g
- Verify subsampling activates for large context

### 5.2 Config Additions (`fgn/config.py`)

Add these fields to FGNConfig:

```
# Resonance parameters
structural_energy_lambda: float = 0.0     # λ_struct — weight of structural energy
structural_energy_max_pairs: int = 2048   # max context pairs for energy computation
multi_metric_types: list = None           # metric types for MultiMetricWorld
                                          # None = use standard CW task
```

### 5.3 Model Modifications (`fgn/model_fluid.py`)

Add StructuralEnergy to FluidNetModel. Compute it once from the initial embeddings and first layer's metric (before any diffusion), since we want the energy to shape how the metric organizes raw input.

```
# In __init__:
self.structural_energy = StructuralEnergy(
    max_context_pairs=config.structural_energy_max_pairs
)
self.lambda_struct = config.structural_energy_lambda

# In forward(), after first layer:
h_post_layer0, kappa0, mcv0, tavg0 = self.layers[0](h, context, mask=mask)

# Compute structural energy using layer 0's metric
# Need to extract g from layer 0 — add a method to FluidLayer
g_layer0 = self.layers[0].get_current_metric(h, context)
e_struct = self.structural_energy(h, g_layer0, context_mask)

# Continue forward through remaining layers...
h = h_post_layer0
for layer in self.layers[1:]:
    h, kappa, mcv, tavg = layer(h, context, mask=mask)
    ...

# In loss computation:
result["structural_energy"] = e_struct
result["loss"] = ce_loss + curv_loss + self.lambda_struct * e_struct
```

**FluidLayer addition** — add method to extract metric without full forward pass:

```
# In FluidLayer:
def get_current_metric(self, h: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    """Compute metric field without running diffusion. For structural energy."""
    B, N, d = h.shape
    h_normed = self.norm_geo(h)
    ctx_exp = context.unsqueeze(1).expand(B, N, d)
    cat_input = torch.cat([h_normed, ctx_exp], dim=-1)
    g = F.softplus(self.metric_net_linear2(
        F.gelu(self.metric_net_linear1(cat_input))
    ))
    return g
```

### 5.4 MultiMetricWorld Task (`fgn/tasks/multi_metric_world.py`)

Extends ContinuousGridWorld with multiple connectivity rules.

```
class MultiMetricWorld(ContinuousWorld):
    """World with variable connectivity rules.
    
    Each episode randomly selects a metric type that determines
    how rooms are connected. The model must infer the metric type
    from the [WORLD] description and plan accordingly.
    """
    
    METRIC_TYPES = ["proximity", "color", "grid", "hub"]
    COLORS = ["red", "blue", "green", "yellow"]
    COLOR_WHEEL = {"red": ["yellow", "blue"], 
                   "blue": ["red", "green"],
                   "green": ["blue", "yellow"],
                   "yellow": ["green", "red"]}
    
    def __init__(self, n_rooms, n_objects, space_size, connect_radius,
                 metric_types=None, rng=None):
        self.metric_type = rng.choice(metric_types or self.METRIC_TYPES)
        # ... rest of init, calling different _build_graph depending on metric_type
```

**Connectivity rules:**

1. **proximity** — current ContinuousWorld behavior (Euclidean < R)
2. **color** — each room assigned a color; connect if same color or adjacent on color wheel. Path planning ignores position, uses color adjacency.
3. **grid** — rooms assigned to integer grid positions; connect if Manhattan distance ≤ 2. Override random coordinates with grid coordinates.
4. **hub** — one room designated hub; hub connects to all; other rooms connect only to hub and their 2 nearest neighbors.

**World description format:**

```
# Proximity:
[WORLD] metric:proximity room_0 pos(23.4,67.1) kitchen ...
connected(room_0,room_1,dist=15.2) ...

# Color:  
[WORLD] metric:color room_0 pos(23.4,67.1) color:red kitchen ...
connected(room_0,room_3) ...

# Grid:
[WORLD] metric:grid room_0 grid(2,3) kitchen ...
connected(room_0,room_1) ...

# Hub:
[WORLD] metric:hub hub=room_0 room_0 pos(50.0,50.0) kitchen ...
connected(room_0,room_1) connected(room_0,room_2) ...
```

**Critical implementation detail:** The `metric:TYPE` token must be added to the vocabulary. Add tokens: `metric:proximity`, `metric:color`, `metric:grid`, `metric:hub`, `color:red`, `color:blue`, `color:green`, `color:yellow`, `hub=room_X`, `grid(X,Y)`.

**Shortest path:** Use Dijkstra with metric-appropriate edge weights:
- proximity: Euclidean distance
- color: 1 for same-color, 2 for color-wheel-adjacent
- grid: Manhattan distance between grid positions
- hub: 1 for all edges (hop count, since hub is always ≤2 hops)

**MultiMetricGridWorldTask** wraps MultiMetricWorld and ContinuousGridWorldTask:

```
class MultiMetricGridWorldTask:
    """Task wrapper that generates episodes using MultiMetricWorld."""
    
    def __init__(self, vocab_size, max_seq_len, metric_types=None, **kwargs):
        # kwargs passed to MultiMetricWorld
        # metric_types defaults to all 4
    
    def generate_batch(self, batch_size, device, rng=None):
        # Returns same format as ContinuousGridWorldTask:
        # input_ids, labels, context_mask, action_spans, metadata
        # metadata includes "metric_type" for eval stratification
```

**Unit tests:**
- Generate 100 episodes, verify each metric type appears ~25 times
- For each metric type, verify shortest path is correct per that metric's distance
- Verify world description includes `metric:TYPE` token
- Verify connectivity matches the selected rule (no proximity connections in color world, etc.)

### 5.5 Evaluation Script (`scripts/eval_resonant.py`)

Extends eval_fluid_gridworld.py with:

1. **Per-metric-type breakdown:** Report SeqAcc, NavAcc, ManipAcc separately for each metric type within each condition.

2. **Geometry tracking per metric type:** Report CV and |κ| separately for proximity/color/grid/hub episodes. If the metric adapts per episode, different metric types should show different geometric signatures.

3. **New metric — Metric Differentiation Score:**
   ```
   MDS = variance(mean_CV_per_metric_type) / mean(CV)
   ```
   High MDS = the metric produces different geometry for different world types.
   Low MDS = the metric uses the same geometry regardless of world type.

Output table format:

```
=================================================================
Condition                     Type      SeqAcc  NavAcc  ManipAcc     CV    |κ|
--------------------------- --------- -------- ------- --------- ------ ------
ID: 10-15rm all              ALL        0.xxx   0.xxx    0.xxx   0.xxx  0.xxx
ID: 10-15rm all              proximity  0.xxx   0.xxx    0.xxx   0.xxx  0.xxx
ID: 10-15rm all              color      0.xxx   0.xxx    0.xxx   0.xxx  0.xxx
ID: 10-15rm all              grid       0.xxx   0.xxx    0.xxx   0.xxx  0.xxx
ID: 10-15rm all              hub        0.xxx   0.xxx    0.xxx   0.xxx  0.xxx
...
Metric Differentiation Score:  0.xxx
=================================================================
```

---

## 6. Training Protocol

### 6.1 Experiment Structure

**Experiment A — Resonance Test (Continuous GridWorld, single metric type):**
Tests whether structural energy prevents flattening on the EXISTING task.

```
Models to train (4 models):
  1. FluidNet-6L  λ_struct=0.0   (baseline — expect flattening)
  2. FluidNet-6L  λ_struct=0.01  (weak resonance)
  3. FluidNet-6L  λ_struct=0.1   (moderate resonance)
  4. FluidNet-6L  λ_struct=1.0   (strong resonance)

Task: ContinuousGridWorld (existing, proximity only)
Steps: 10,000
Other hyperparameters: same as current FluidNet v1 experiment
```

**Primary metric:** CV and |κ| at step 10K. Does the flattening stop?

**Secondary metric:** CE loss at step 10K. Does structural energy hurt task learning?

**Experiment B — Multi-Metric Generalization:**
Tests whether dynamic geometry improves generalization on multi-metric worlds.

```
Models to train (3 models):
  1. FluidNet-6L  λ_struct=BEST  (best λ from Experiment A)
  2. FluidNet-6L  λ_struct=0.0   (FluidNet without resonance)
  3. Flat-6L                      (flat transformer baseline)

Task: MultiMetricGridWorld (all 4 metric types)
Steps: 10,000
```

**Primary metric:** SeqAcc on Novel-Metric condition (metric type not seen in training).

**Secondary metric:** Metric Differentiation Score (MDS) — does the geometry adapt per metric type?

### 6.2 Hyperparameters

```
# Shared across all models:
d_model: 256
n_layers: 6
n_scales: 3          # (FluidNet only)
d_metric: 64         # (FluidNet only)  
d_ffn_fluid: 512     # (FluidNet only)
d_ff: 512            # (flat only)
n_heads: 8           # (flat only)
vocab_size: 50304    # may need increase for new tokens
max_seq_len: 1024
batch_size: 4
lr: 3e-4
weight_decay: 0.1
warmup_steps: 1000
grad_clip: 1.0
dropout: 0.1
use_torch_compile: true
bf16: true (if available)

# Multi-metric task:
n_rooms_min: 10
n_rooms_max: 15
space_size: 100.0
connect_radius: 30.0
n_objects: 4
min_steps: 4
max_steps: 10
min_state_changes: 1
metric_types: ["proximity", "color", "grid", "hub"]  # training
novel_metric_type: "ring"                              # eval-only
```

### 6.3 Training Script (`scripts/train_resonant.py`)

Adapt from train_fluid.py. Key changes:

1. Accept `--lambda_struct` argument
2. Accept `--task` argument that can be `CW` (ContinuousGridWorld) or `MM` (MultiMetricWorld)
3. Log structural energy alongside CE loss
4. Log CV and |κ| trajectory for flattening analysis

Log format:
```
[step=N] loss=X.XXXX, ce=X.XXXX, e_struct=X.XXXX, cv=X.XXXX, |k|=X.XXXX, tok/s=XXXX, t=[X.XX,X.XX,X.XX]
```

### 6.4 Config Files

**configs/resonant_6l.yaml:**
```yaml
d_model: 256
n_heads: 8
n_layers: 6
d_ff: 512
d_ffn_fluid: 512
d_metric: 64
n_scales: 3
vocab_size: 50304
max_seq_len: 1024
model_type: fgn
architecture_version: "fluid"
geo_metric_type: learned
curvature_lambda: 0.0
curvature_reward_mu: 0.0
structural_energy_lambda: 0.1
structural_energy_max_pairs: 2048
dropout: 0.1
use_torch_compile: true
```

**configs/resonant_flat.yaml:**
```yaml
d_model: 256
n_heads: 8
n_layers: 6
d_ff: 512
vocab_size: 50304
max_seq_len: 1024
model_type: flat
architecture_version: "flat"
dropout: 0.1
use_torch_compile: true
```

---

## 7. Run Script (`scripts/run_resonant_experiment.sh`)

```bash
#!/bin/bash
cd /workspace/fgn-v3
export PYTHONUNBUFFERED=1
export CUDA_MEMORY_FRACTION=0.85
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CW_KWARGS='{"n_rooms_min": 10, "n_rooms_max": 15, "space_size": 100.0, "connect_radius": 30.0, "n_objects": 4, "min_steps": 4, "max_steps": 10, "min_state_changes": 1}'
MM_KWARGS='{"n_rooms_min": 10, "n_rooms_max": 15, "space_size": 100.0, "connect_radius": 30.0, "n_objects": 4, "min_steps": 4, "max_steps": 10, "min_state_changes": 1, "metric_types": ["proximity", "color", "grid", "hub"]}'
EVAL_ARGS="--n_batches 50 --batch_size 8"
STEPS=10000

echo "============================================"
echo "  Experiment A — Resonance Test"
echo "  Does structural energy prevent flattening?"
echo "============================================"

for LAMBDA in 0.0 0.01 0.1 1.0; do
    echo ""
    echo ">>> Training FluidNet λ_struct=${LAMBDA}..."
    python scripts/train_resonant.py \
        --config configs/resonant_6l.yaml \
        --task CW --batch_size 4 --lr 3e-4 --weight_decay 0.1 \
        --warmup_steps 1000 --grad_clip 1.0 --log_every 100 \
        --save_every 5000 --max_steps $STEPS \
        --lambda_struct $LAMBDA \
        --task_kwargs "$CW_KWARGS" \
        --output_dir output_resonant_lambda${LAMBDA}
done

echo ""
echo ">>> Evaluating Experiment A..."
for LAMBDA in 0.0 0.01 0.1 1.0; do
    echo ""
    echo "--- λ_struct=${LAMBDA} ---"
    python scripts/eval_resonant.py \
        --config configs/resonant_6l.yaml \
        --checkpoint output_resonant_lambda${LAMBDA}/checkpoints/final.pt \
        --task CW --task_kwargs "$CW_KWARGS" \
        $EVAL_ARGS
done

echo ""
echo "============================================"
echo "  Experiment B — Multi-Metric Generalization"
echo "  (Run after determining best λ from Exp A)"
echo "============================================"

# Set BEST_LAMBDA manually after reviewing Experiment A results
BEST_LAMBDA=0.1

echo ""
echo ">>> Training FluidNet + resonance on MultiMetric..."
python scripts/train_resonant.py \
    --config configs/resonant_6l.yaml \
    --task MM --batch_size 4 --lr 3e-4 --weight_decay 0.1 \
    --warmup_steps 1000 --grad_clip 1.0 --log_every 100 \
    --save_every 5000 --max_steps $STEPS \
    --lambda_struct $BEST_LAMBDA \
    --task_kwargs "$MM_KWARGS" \
    --output_dir output_resonant_mm

echo ""
echo ">>> Training FluidNet (no resonance) on MultiMetric..."
python scripts/train_resonant.py \
    --config configs/resonant_6l.yaml \
    --task MM --batch_size 4 --lr 3e-4 --weight_decay 0.1 \
    --warmup_steps 1000 --grad_clip 1.0 --log_every 100 \
    --save_every 5000 --max_steps $STEPS \
    --lambda_struct 0.0 \
    --task_kwargs "$MM_KWARGS" \
    --output_dir output_fluid_mm

echo ""
echo ">>> Training Flat-6L on MultiMetric..."
python scripts/train_resonant.py \
    --config configs/resonant_flat.yaml \
    --task MM --batch_size 4 --lr 3e-4 --weight_decay 0.1 \
    --warmup_steps 1000 --grad_clip 1.0 --log_every 100 \
    --save_every 5000 --max_steps $STEPS \
    --task_kwargs "$MM_KWARGS" \
    --output_dir output_flat_mm

echo ""
echo ">>> Evaluating Experiment B (ID + OOD + Novel Metric)..."
for MODEL_DIR in output_resonant_mm output_fluid_mm output_flat_mm; do
    CFG="configs/resonant_6l.yaml"
    if [ "$MODEL_DIR" = "output_flat_mm" ]; then
        CFG="configs/resonant_flat.yaml"
    fi
    echo ""
    echo "--- $MODEL_DIR ---"
    python scripts/eval_resonant.py \
        --config $CFG \
        --checkpoint ${MODEL_DIR}/checkpoints/final.pt \
        --task MM --task_kwargs "$MM_KWARGS" \
        --eval_novel_metric ring \
        $EVAL_ARGS
done

echo ""
echo "============================================"
echo "  All experiments complete."
echo "============================================"
```

---

## 8. Success Criteria

### Experiment A — Resonance Test

**Primary (must achieve):**
- λ_struct > 0 models maintain CV > 0.25 at step 10K (vs baseline's expected decline toward 0.20)
- λ_struct > 0 models do NOT degrade CE loss by more than 2× vs λ=0 baseline

**Target:**
- Clear monotonic relationship: higher λ → higher CV at convergence
- Best λ achieves CV > 0.30 at step 10K with CE ≤ 1.5× baseline

**Failure indicators:**
- CV drops regardless of λ → structural energy doesn't create sufficient gradient
- CE explodes for all λ > 0 → structural energy interferes with task learning
- CV increases but OOD accuracy doesn't → structure is maintained but not useful

### Experiment B — Multi-Metric Generalization

**Primary (must achieve):**
- Resonant FluidNet matches flat-6L on ID accuracy (all metric types)
- Per-metric-type breakdown shows model handles all 4 metric types

**Target:**
- Resonant FluidNet outperforms flat-6L on Novel-Metric condition by > 5pp SeqAcc
- MDS > 0.1 for resonant FluidNet (geometry differentiates between metric types)
- MDS ≈ 0 for flat-6L (no geometric differentiation, as expected)

**Stretch:**
- Resonant FluidNet outperforms non-resonant FluidNet on Novel-Metric
- This would confirm structural energy improves geometric adaptation

**Failure indicators:**
- All three models identical on Novel-Metric → geometry doesn't help with novel connectivity
- MDS ≈ 0 for all models → metric doesn't differentiate between world types
- Resonant FluidNet underperforms flat on ID → structural energy hurts basic learning

---

## 9. Implementation Order

```
Step 1:  Config additions (structural_energy_lambda, etc.)
         Test: config loads without error

Step 2:  StructuralEnergy module
         Test: unit tests (alignment=0 for proportional distances, gradient flow)

Step 3:  FluidLayer.get_current_metric() method
         Test: returns [B, N, d] tensor, matches what forward() computes

Step 4:  Model integration (structural energy in forward pass)
         Test: loss includes e_struct term, gradient flows through it

Step 5:  Training script (train_resonant.py)
         Test: runs 100 steps with λ_struct=0.1, logs e_struct

Step 6:  Experiment A — λ sweep on ContinuousGridWorld
         Deliverable: CV/|κ| trajectory comparison for all λ values

Step 7:  MultiMetricWorld task
         Test: generates episodes for all 4 metric types, shortest paths correct

Step 8:  Evaluation script (eval_resonant.py)
         Test: per-metric-type breakdown, MDS computation

Step 9:  Experiment B — Multi-metric comparison
         Deliverable: full comparison table including Novel-Metric condition

Step 10: Results analysis and reporting
```

**IMPORTANT: Run Experiment A before building Experiment B infrastructure.** If structural energy doesn't prevent flattening (Step 6 fails), the multi-metric experiment needs a different approach. Don't build the multi-metric task until Experiment A validates the resonance mechanism.

---

## 10. What's NOT in This Spec

- **Multi-field splitting:** The emergence of multiple metric fields from one is a future direction. This spec tests the prerequisite: can ONE field maintain structure via resonance?
- **Variable depth iteration:** Resonant models use fixed 6 layers. Adaptive depth is a separate research question.
- **Reconstruction loss:** The structural energy is simpler than full autoencoding. If it works, reconstruction can be added later.
- **Phase 0 unsupervised pretraining:** Both losses (CE + structural) run simultaneously. No separate unsupervised phase.
- **Curriculum learning:** Fixed task distribution throughout. Curriculum is a future direction if resonance + multi-metric shows promise.
- **Ring metric implementation:** The "ring" connectivity rule for Novel-Metric eval needs implementation but is simple: connect each room to its 2 nearest neighbors forming a ring topology. Only needed for eval, not training.

---

## 11. Parameter Budget

Same as FluidNet v1 (~28.9M). StructuralEnergy has no learned parameters — it computes distances from existing metric and hidden states.

The only additional compute cost: one extra geodesic distance computation per step (context positions only, typically ~100-200 positions vs full sequence of ~500-1000). Estimated overhead: <5% of training time.

---

## 12. Predicted Results

### Experiment A

| λ_struct | CV @ 10K | |κ| @ 10K | CE @ 10K | Flattening? |
|----------|----------|----------|----------|-------------|
| 0.0 | ~0.25 | ~0.45 | ~0.0002 | Yes (from 0.33 peak) |
| 0.01 | ~0.28 | ~0.50 | ~0.0003 | Reduced |
| 0.1 | ~0.33 | ~0.60 | ~0.0005 | No (holds at peak) |
| 1.0 | ~0.40 | ~0.70 | ~0.0020 | No (but CE degraded) |

**Rationale:** Structural energy provides continuous gradient to maintain metric structure even after CE converges. Higher λ → stronger geometric drive → more curvature retained, but at the cost of less CE optimization.

### Experiment B

| Model | ID SeqAcc | Novel-Metric SeqAcc | MDS |
|-------|-----------|---------------------|-----|
| Resonant FluidNet | 100% | ~30% | >0.1 |
| FluidNet (no res.) | 100% | ~20% | ~0.05 |
| Flat-6L | 100% | ~20% | N/A |

**Rationale:** Novel metric type is hard for everyone. But resonant FluidNet's maintained geometry gives it a structural prior that partially transfers to the unseen connectivity rule. The MDS confirms the metric actually differentiates between world types.

---

## 13. Key Risks and Mitigations

**Risk 1: Structural distance proxy is too crude.**
Positional distance within [WORLD] is a rough approximation of relational proximity. If the world description format doesn't place related entities near each other, the structural energy pushes the metric toward the wrong geometry.

*Mitigation:* The world description IS generated by our task code, so we control the format. Ensure connected rooms and their objects are described adjacently. Verify by inspecting generated episodes before training.

**Risk 2: Structural energy and CE loss conflict.**
The metric that minimizes structural energy (geometry ∝ data structure) might not be the metric that minimizes CE loss (geometry ∝ prediction utility). The two objectives could fight.

*Mitigation:* The λ sweep in Experiment A specifically tests this. If CE degrades too much at all λ > 0, the losses are incompatible and a different structural energy formulation is needed.

**Risk 3: Multi-metric task is too hard at 30M parameters.**
Four different connectivity rules with different planning strategies might exceed the model's capacity, producing random performance on all metric types.

*Mitigation:* Monitor per-metric-type ID accuracy. If any single metric type has < 90% ID accuracy, the model is struggling with basic task learning, not just generalization. In that case, reduce to 2 metric types (proximity + one other) and retry.

**Risk 4: Novel-metric eval is meaningless if model memorizes metric:TYPE token.**
If the model just maps `metric:proximity → one behavior, metric:color → another behavior`, the metric:ring token in Novel-Metric eval is just an unknown token, not a geometric challenge.

*Mitigation:* Run an additional eval: remove the `metric:TYPE` token entirely from all eval episodes. If performance drops dramatically, the model is relying on the type label rather than inferring geometry from the connection structure. This would mean the model learned metric-type classification, not geometric adaptation.

---

*End of specification. Experiment A is the minimum viable test. Build and run it before investing in Experiment B infrastructure.*
