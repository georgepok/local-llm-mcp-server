# FGN v7 Implementation Spec — Sandwich Architecture

## Executive Summary

v7 restructures the proven v6 components into a **biological sandwich architecture**: geometric layers handle all world-facing computation (input/output), transformer attention handles all internal reasoning. The division is **structural, not learned** — no gates, no budgets, no escalation, no thresholds.

```
Tokens → [Embed + Pos] → Bottom Geo (L1-L2) → Middle Attn (L3-L6) → Top Geo (L7-L8) → [LM Head]
                               ↑                                          ↑
                           ContextPool ────────────────────────────── same ctx
                         (reads [WORLD])
```

**What changes:** Layer architecture (sandwich vs uniform mixed layers), removal of all gating/budget mechanisms.
**What stays:** ContextPool, context-conditioned MetricNetwork, multi-head GeoRoute with Q/K projections, StandardAttention, CurvatureEngine, GridWorldTask, training/eval infrastructure.

---

## 1. Motivation — Why Sandwich

### v6 Result Recap

| Condition | v6-metric | flat-6 |
|-----------|-----------|--------|
| ID (8-12rm, 8-15st) | **100%** | 100% |
| Near (13-15rm, 8-15st) | **84%** | 39.75% |
| Far (15-18rm, 12-17st) | **78%** | 6.25% |
| ManipAcc (all conditions) | **1.0** | variable |
| NavAcc (Far-OOD) | **~78%** | ~6% |

v6 works. The question is whether separating responsibilities architecturally produces cleaner training dynamics and better OOD, especially on navigation.

### Architectural Problem with v6

In v6, every layer contains **both** GeoRoute and Attention, with hand-tuned budgets `[0.00, 0.05, 0.10, 0.10, 0.20, 0.30]` controlling how much attention each layer uses. This creates:

1. **Budget tuning problem** — budgets are hand-set with no principled method
2. **Effective attention is thin** — 6 layers × ~15% average budget ≈ 0.9 effective attention layers
3. **Interpretability gap** — each layer does "some geo + some attn" at learned ratios

### Sandwich Solution

- Bottom geo layers: 2 layers × 100% geometric = 2.0 effective geo layers
- Middle attn layers: 4 layers × 100% attention = 4.0 effective attention layers
- Top geo layers: 2 layers × 100% geometric = 2.0 effective geo layers

Result: **4× more effective attention** than v6, plus **clear division of labor** without any tuning knobs.

---

## 2. Architecture — Detailed Specification

### 2.1 New Config Parameters

**File:** `fgn/config.py`

Add to `FGNConfig`:

```python
# v7 sandwich architecture
sandwich_mode: bool = False                    # True enables sandwich routing
sandwich_bottom_geo_layers: int = 2            # pure GeoRoute layers at bottom
sandwich_middle_attn_layers: int = 4           # pure StandardAttention layers in middle
sandwich_top_geo_layers: int = 2               # pure GeoRoute layers at top
sandwich_separate_top_metric: bool = True       # top geo uses separate MetricNetwork
```

**Derived property:**

```python
@property
def n_layers(self) -> int:
    if self.sandwich_mode:
        return (self.sandwich_bottom_geo_layers +
                self.sandwich_middle_attn_layers +
                self.sandwich_top_geo_layers)
    return self._n_layers  # fallback for non-sandwich configs
```

**IMPORTANT:** The existing `n_layers`, `n_heads`, `geo_heads`, `d_model`, `d_ff`, `vocab_size`, `max_seq_len`, `dropout`, `geo_metric_type`, `use_torch_compile` fields are unchanged. All v6 config fields remain for backward compatibility.

### 2.2 New Layer Types

Create **two** new layer classes. Do NOT modify existing `layer_v6.py`.

#### 2.2.1 `GeoOnlyLayer` — Pure Geometric Routing

**File:** `fgn/layer_v7_geo.py` (new file)

This is a stripped-down version of `FGNv6Layer` with budget permanently set to 0.0 — no attention pathway at all.

```
Forward pass:
  h_norm = LayerNorm(h)
  g = MetricNetwork(h_norm, context)           # context-conditioned
  kappa = CurvatureEngine(g)
  h_geo, geo_weights = GeoRoute(h_norm, g)     # multi-head with Q/K projections
  h = h + Dropout(h_geo)
  h = h + Dropout(FFN(LayerNorm(h)))
  return h, kappa, metric_cv, avg_entropy
```

**Constructor signature:**

```python
class GeoOnlyLayer(nn.Module):
    def __init__(self, config: FGNConfig, layer_idx: int):
```

**Components to instantiate:**
- `nn.LayerNorm(d_model)` — `norm_metric`, `norm_geo`, `norm_ff` (3 layer norms)
- `MetricNetwork(config)` — context-conditioned metric (from `fgn/metric.py`, unchanged)
- `CurvatureEngine()` — (from `fgn/curvature.py`, unchanged)
- `GeoRoute(config)` — multi-head with Q/K projections (from `fgn/geo_route.py`, unchanged, using `architecture_version="v6"` path)
- `FFN` — `nn.Sequential(Linear(d_model, d_ff), GELU(), Dropout(dropout), Linear(d_ff, d_model))`
- `nn.Dropout(dropout)` — residual dropout

**Components NOT instantiated (this is critical):**
- NO `StandardAttention`
- NO `norm_attn`
- NO budget, escalation, or gating logic

**Forward signature:**

```python
def forward(self, h: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            context: Optional[torch.Tensor] = None
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        h: [B, N, d_model] updated hidden states
        kappa: [B, N] scalar curvature
        metric_cv: scalar coefficient of variation
        avg_entropy: scalar average GeoRoute entropy (for diagnostics)
    """
```

**Entropy computation for diagnostics only** (no budget selection):

```python
# Compute entropy from geo_weights for monitoring (same formula as v6)
eps = 1e-8
entropy = -(geo_weights * torch.log(geo_weights + eps)).sum(dim=-1)
positions = torch.arange(1, N + 1, device=h.device, dtype=h.dtype)
max_entropy = torch.log(positions + eps).unsqueeze(0)
normalized_entropy = entropy / (max_entropy + eps)
avg_entropy = normalized_entropy.mean()
```

#### 2.2.2 `AttnOnlyLayer` — Pure Transformer Attention

**File:** `fgn/layer_v7_attn.py` (new file)

Standard transformer layer. Identical to what `FlatTransformerModel` uses internally, but extracted as a standalone module.

```
Forward pass:
  h = h + Dropout(StandardAttention(LayerNorm(h), mask))
  h = h + Dropout(FFN(LayerNorm(h)))
  return h
```

**Constructor signature:**

```python
class AttnOnlyLayer(nn.Module):
    def __init__(self, config: FGNConfig, layer_idx: int):
```

**Components to instantiate:**
- `nn.LayerNorm(d_model)` — `norm_attn`, `norm_ff` (2 layer norms)
- `StandardAttention(config)` — (from `fgn/standard_attention.py`, unchanged)
- `FFN` — same as GeoOnlyLayer
- `nn.Dropout(dropout)` — residual dropout

**Components NOT instantiated:**
- NO `MetricNetwork`
- NO `CurvatureEngine`
- NO `GeoRoute`
- NO LayerNorm for metric or geo

**Forward signature:**

```python
def forward(self, h: torch.Tensor,
            mask: Optional[torch.Tensor] = None
            ) -> torch.Tensor:
    """
    Returns:
        h: [B, N, d_model] updated hidden states
    """
```

NOTE: No context parameter. Attention layers don't use the metric or context.

### 2.3 Sandwich Model

**File:** `fgn/model_v7.py` (new file)

```python
class FGNv7Model(nn.Module):
```

#### Constructor

```python
def __init__(self, config: FGNConfig):
    super().__init__()
    self.config = config

    # Embeddings (identical to v6)
    self.embed = nn.Embedding(config.vocab_size, config.d_model)
    self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)

    # Shared ContextPool (identical to v6, computed once from embeddings)
    if config.geo_metric_type == "learned":
        self.context_pool = ContextPool(config)

    # === SANDWICH LAYERS ===

    # Bottom geometric layers
    self.bottom_geo = nn.ModuleList([
        GeoOnlyLayer(config, layer_idx=i)
        for i in range(config.sandwich_bottom_geo_layers)
    ])

    # Middle attention layers
    mid_start = config.sandwich_bottom_geo_layers
    self.middle_attn = nn.ModuleList([
        AttnOnlyLayer(config, layer_idx=mid_start + i)
        for i in range(config.sandwich_middle_attn_layers)
    ])

    # Top geometric layers (separate MetricNetwork per layer)
    top_start = mid_start + config.sandwich_middle_attn_layers
    self.top_geo = nn.ModuleList([
        GeoOnlyLayer(config, layer_idx=top_start + i)
        for i in range(config.sandwich_top_geo_layers)
    ])

    # Output (identical to v6)
    self.norm = nn.LayerNorm(config.d_model)
    self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    # Curvature regularization (covers bottom + top geo layers)
    n_geo_layers = config.sandwich_bottom_geo_layers + config.sandwich_top_geo_layers
    self.curv_reg = CurvatureRegularization(
        n_layers=n_geo_layers,
        curvature_lambda=config.curvature_lambda,
        correlation_length_init=config.correlation_length_init,
        curvature_reward_mu=config.curvature_reward_mu,
    )

    # Weight initialization (identical to v6)
    self.apply(self._init_weights)
```

#### Forward Pass

```python
def forward(self, input_ids, labels=None, context_mask=None):
    B, N = input_ids.shape
    device = input_ids.device

    # 1. Embeddings
    pos = torch.arange(N, device=device).unsqueeze(0)
    h = self.embed(input_ids) + self.pos_embed(pos)

    # 2. Shared context (once, from embeddings)
    context = None
    if hasattr(self, 'context_pool'):
        context = self.context_pool(h, context_mask)

    # 3. Causal mask
    mask = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)

    # 4. Bottom geometric layers
    curvatures = []
    metric_cvs = []
    geo_entropies = []

    for layer in self.bottom_geo:
        h, kappa, m_cv, avg_ent = layer(h, mask=mask, context=context)
        curvatures.append(kappa)
        metric_cvs.append(m_cv)
        geo_entropies.append(avg_ent)

    # 5. Middle attention layers
    for layer in self.middle_attn:
        h = layer(h, mask=mask)

    # 6. Top geometric layers (same context, separate metrics)
    for layer in self.top_geo:
        h, kappa, m_cv, avg_ent = layer(h, mask=mask, context=context)
        curvatures.append(kappa)
        metric_cvs.append(m_cv)
        geo_entropies.append(avg_ent)

    # 7. LM head
    h = self.norm(h)
    logits = self.lm_head(h)

    # 8. Build result dict
    result = {"logits": logits}

    # Monitoring stats
    result["metric_cv"] = sum(metric_cvs) / len(metric_cvs)
    result["avg_kappa"] = sum(k.abs().mean() for k in curvatures) / len(curvatures)
    result["avg_entropy"] = sum(geo_entropies) / len(geo_entropies)

    # Per-stage stats (new in v7)
    n_bot = self.config.sandwich_bottom_geo_layers
    result["bottom_metric_cv"] = sum(metric_cvs[:n_bot]) / max(n_bot, 1)
    result["top_metric_cv"] = sum(metric_cvs[n_bot:]) / max(len(metric_cvs) - n_bot, 1)
    result["bottom_avg_kappa"] = sum(
        k.abs().mean() for k in curvatures[:n_bot]) / max(n_bot, 1)
    result["top_avg_kappa"] = sum(
        k.abs().mean() for k in curvatures[n_bot:]) / max(len(curvatures) - n_bot, 1)

    # v6 compatibility fields
    result["escalation_rate"] = torch.tensor(0.0, device=device)
    result["esc_rates_per_layer"] = []
    result["entropies_per_layer"] = [e for e in geo_entropies]
    result["scale_loss"] = torch.tensor(0.0, device=device)
    result["avg_gate"] = torch.tensor(0.0, device=device)

    if labels is not None:
        ce_loss = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            labels.reshape(-1),
            ignore_index=-100,
        )
        result["ce_loss"] = ce_loss

        curv_loss = self.curv_reg(curvatures)
        result["curv_loss"] = curv_loss

        result["esc_penalty"] = torch.tensor(0.0, device=device)
        result["loss"] = ce_loss + curv_loss

    return result
```

#### Parameter Groups

```python
def geo_parameters(self) -> List[nn.Parameter]:
    """All geometric pathway params (bottom MetricNet/GeoRoute + top MetricNet/GeoRoute + ContextPool)."""
    params = []
    if hasattr(self, 'context_pool'):
        params.extend(self.context_pool.parameters())
    for layer in self.bottom_geo:
        params.extend(layer.metric.parameters())
        params.extend(layer.geo_route.parameters())
    for layer in self.top_geo:
        params.extend(layer.metric.parameters())
        params.extend(layer.geo_route.parameters())
    params.extend(self.curv_reg.parameters())
    return params

def attn_parameters(self) -> List[nn.Parameter]:
    """All attention pathway params (middle attention layers)."""
    params = []
    for layer in self.middle_attn:
        params.extend(layer.attention.parameters())
    return params

def other_parameters(self) -> List[nn.Parameter]:
    """Embeddings, all FFNs, all LayerNorms, lm_head."""
    geo_ids = {id(p) for p in self.geo_parameters()}
    attn_ids = {id(p) for p in self.attn_parameters()}
    return [p for p in self.parameters()
            if id(p) not in geo_ids and id(p) not in attn_ids]
```

### 2.4 Separate Top MetricNetwork

When `sandwich_separate_top_metric=True` (default), bottom and top `GeoOnlyLayer` instances each create their own `MetricNetwork`. This happens automatically — each `GeoOnlyLayer.__init__` creates `self.metric = MetricNetwork(config)`. No special handling needed; separate layers = separate weights.

**Shared context:** Both bottom and top layers receive the same `context` vector from `ContextPool`. The metric interpretation differs because the weights are different:
- Bottom metric: learns perceptual geometry (how environment is structured for reading)
- Top metric: learns action geometry (how environment is structured for acting)

### 2.5 Flat-8 Baseline

For controlled comparison, create an 8-layer flat baseline matching v7's total depth.

**Config:** `configs/v7_flat8.yaml`
```yaml
d_model: 256
n_heads: 8
n_layers: 8
d_ff: 1024
vocab_size: 50304
max_seq_len: 1024
model_type: flat
architecture_version: "v6"
dropout: 0.1
use_torch_compile: true
```

This uses the existing `FlatTransformerModel` with `n_layers=8`. No code changes needed.

---

## 3. File Inventory

### 3.1 New Files

| File | Description | Based On |
|------|-------------|----------|
| `fgn/layer_v7_geo.py` | `GeoOnlyLayer` — pure geometric routing layer | `layer_v6.py` with budget=0.0 path only, no attention |
| `fgn/layer_v7_attn.py` | `AttnOnlyLayer` — pure transformer attention layer | `flat_model.py` layer logic extracted |
| `fgn/model_v7.py` | `FGNv7Model` — sandwich model orchestrator | `model_v6.py` restructured |
| `scripts/train_v7.py` | v7 training script | `train_v6.py` with v7 model creation |
| `scripts/eval_v7_gridworld.py` | v7 eval script | `eval_v6_gridworld.py` with v7 model loading + new metrics |
| `scripts/run_v7_gridworld.sh` | Run script for full v7 experiment | `run_v6_gridworld.sh` adapted |
| `configs/v7_sandwich.yaml` | v7 sandwich config | New |
| `configs/v7_flat8.yaml` | 8-layer flat baseline config | New |

### 3.2 Modified Files

| File | Change |
|------|--------|
| `fgn/config.py` | Add `sandwich_mode`, `sandwich_bottom_geo_layers`, `sandwich_middle_attn_layers`, `sandwich_top_geo_layers`, `sandwich_separate_top_metric` fields |
| `fgn/geo_route.py` | Add `"v7"` to the architecture version check for Q/K path (single line) |

### 3.3 Unchanged Files (reuse as-is)

| File | Used By |
|------|---------|
| `fgn/metric.py` | `GeoOnlyLayer` (both bottom and top) |
| `fgn/context_pool.py` | `FGNv7Model` |
| `fgn/standard_attention.py` | `AttnOnlyLayer` |
| `fgn/curvature.py` | `GeoOnlyLayer` |
| `fgn/losses.py` | `FGNv7Model` |
| `fgn/flat_model.py` | Flat-8 baseline |
| `fgn/tasks/gridworld.py` | All models (no task changes) |

---

## 4. Config Files

### 4.1 `configs/v7_sandwich.yaml`

```yaml
# FGN v7 — Sandwich Architecture
d_model: 256
n_heads: 8
n_layers: 8               # informational (2+4+2), overridden by sandwich params
d_ff: 1024
vocab_size: 50304
max_seq_len: 1024
model_type: fgn
architecture_version: "v7"

# Sandwich structure
sandwich_mode: true
sandwich_bottom_geo_layers: 2
sandwich_middle_attn_layers: 4
sandwich_top_geo_layers: 2
sandwich_separate_top_metric: true

# Geometric components (same as v6)
geo_heads: 4
geo_metric_type: learned

# Curvature (disabled, pure CE drives geometry — v3 lesson)
curvature_lambda: 0.0
curvature_reward_mu: 0.0

# Standard
dropout: 0.1
use_torch_compile: true
```

### 4.2 `configs/v7_flat8.yaml`

```yaml
# Flat-8 Baseline — matches v7 layer count
d_model: 256
n_heads: 8
n_layers: 8
d_ff: 1024
vocab_size: 50304
max_seq_len: 1024
model_type: flat
architecture_version: "v7"
dropout: 0.1
use_torch_compile: true
```

---

## 5. Training

### 5.1 Training Script

**File:** `scripts/train_v7.py` — based on `train_v6.py`

Only change to `create_model()`:

```python
def create_model(config: FGNConfig, device: torch.device):
    if config.model_type == "flat":
        return FlatTransformerModel(config).to(device)
    elif config.architecture_version == "v7" and config.sandwich_mode:
        return FGNv7Model(config).to(device)
    elif config.architecture_version == "v6":
        return FGNv6Model(config).to(device)
    else:
        raise ValueError(f"Unknown architecture: {config.architecture_version}")
```

And add v7-specific status logging in `print_v7_status()`:

```python
def print_v7_status(model, result, step, tok_per_sec):
    m = model._orig_mod if hasattr(model, '_orig_mod') else model
    is_v7 = isinstance(m, FGNv7Model)

    base = (f"  [step={step}] loss={result['loss'].item():.4f}, "
            f"ce={result['ce_loss'].item():.4f}")

    if is_v7:
        base += (f", bot_cv={result['bottom_metric_cv'].item():.4f}"
                 f", top_cv={result['top_metric_cv'].item():.4f}"
                 f", bot_|k|={result['bottom_avg_kappa'].item():.4f}"
                 f", top_|k|={result['top_avg_kappa'].item():.4f}"
                 f", tok/s={tok_per_sec:.0f}")
    else:
        base += (f", cv={result['metric_cv'].item():.4f}"
                 f", |k|={result['avg_kappa'].item():.4f}"
                 f", tok/s={tok_per_sec:.0f}")

    print(base)
```

**TensorBoard logging additions** (in training loop):

```python
if is_v7:
    writer.add_scalar("metric/bottom_cv", result["bottom_metric_cv"].item(), step)
    writer.add_scalar("metric/top_cv", result["top_metric_cv"].item(), step)
    writer.add_scalar("metric/bottom_kappa", result["bottom_avg_kappa"].item(), step)
    writer.add_scalar("metric/top_kappa", result["top_avg_kappa"].item(), step)
```

**Everything else in the training loop is identical to `train_v6.py`** — same optimizer, same scheduler, same context_mask passing, same gradient clipping.

### 5.2 Training Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `lr` | 3e-4 | Same as v6 |
| `weight_decay` | 0.1 | Same as v6 |
| `warmup_steps` | 1000 | Same as v6 |
| `max_steps` | 15000 | Extended from v6's 10K for extra layers |
| `batch_size` | 4 | Same as v6 |
| `grad_clip` | 1.0 | Same as v6 |
| `log_every` | 100 | Same as v6 |
| `save_every` | 5000 | Same as v6 |
| `max_seq_len` | 1024 | Same as v6 |
| Precision | BF16 | Same as v6 |

**Single optimizer, uniform LR for all parameters.** No differential learning rates. The sandwich structure eliminates competition between geo and attention for representational control — they have separate territory.

### 5.3 Task Parameters (unchanged from v6)

```json
{
    "n_rooms_min": 8,
    "n_rooms_max": 12,
    "n_objects": 6,
    "min_steps": 8,
    "max_steps": 15,
    "min_state_changes": 2,
    "randomize_topology": true
}
```

### 5.4 Three-Way Comparison

Train three models sequentially:

| Model | Config | Architecture | Total Layers |
|-------|--------|-------------|--------------|
| v7-sandwich | `v7_sandwich.yaml` | 2 geo + 4 attn + 2 geo | 8 |
| v6-metric | `v6_hier_metric.yaml` | 6 mixed (geo+attn per layer) | 6 |
| flat-8 | `v7_flat8.yaml` | 8 standard attention | 8 |

**Why flat-8 and not flat-6:** v7 has 8 layers. We need to confirm any improvement isn't simply from extra depth. flat-8 controls for this.

### 5.5 Run Script

**File:** `scripts/run_v7_gridworld.sh`

```bash
#!/bin/bash
cd /workspace/fgn-v3
export PYTHONUNBUFFERED=1
export CUDA_MEMORY_FRACTION=0.85
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TASK_KWARGS='{"n_rooms_min": 8, "n_rooms_max": 12, "n_objects": 6, "min_steps": 8, "max_steps": 15, "min_state_changes": 2, "randomize_topology": true}'
TRAIN_ARGS="--task W --batch_size 4 --lr 3e-4 --weight_decay 0.1 --warmup_steps 1000 --grad_clip 1.0 --log_every 100 --save_every 5000 --max_steps 15000"

echo "============================================"
echo "  FGN v7 Sandwich Experiment"
echo "============================================"

echo ""
echo ">>> [1/3] Training v7-sandwich..."
rm -rf output_v7_sandwich
python scripts/train_v7.py \
    --config configs/v7_sandwich.yaml \
    $TRAIN_ARGS \
    --task_kwargs "$TASK_KWARGS" \
    --output_dir output_v7_sandwich

echo ""
echo ">>> [2/3] Training v6-metric (baseline)..."
rm -rf output_v7_v6baseline
python scripts/train_v7.py \
    --config configs/v6_hier_metric.yaml \
    $TRAIN_ARGS \
    --max_steps 10000 \
    --task_kwargs "$TASK_KWARGS" \
    --output_dir output_v7_v6baseline

echo ""
echo ">>> [3/3] Training flat-8 (baseline)..."
rm -rf output_v7_flat8
python scripts/train_v7.py \
    --config configs/v7_flat8.yaml \
    $TRAIN_ARGS \
    --task_kwargs "$TASK_KWARGS" \
    --output_dir output_v7_flat8

echo ""
echo "============================================"
echo "  Evaluation"
echo "============================================"

EVAL_ARGS="--n_batches 50 --batch_size 8"

echo ""
echo ">>> Evaluating v7-sandwich..."
python scripts/eval_v7_gridworld.py \
    --config configs/v7_sandwich.yaml \
    --checkpoint output_v7_sandwich/checkpoints/final.pt \
    $EVAL_ARGS

echo ""
echo ">>> Evaluating v6-metric..."
python scripts/eval_v7_gridworld.py \
    --config configs/v6_hier_metric.yaml \
    --checkpoint output_v7_v6baseline/checkpoints/final.pt \
    $EVAL_ARGS

echo ""
echo ">>> Evaluating flat-8..."
python scripts/eval_v7_gridworld.py \
    --config configs/v7_flat8.yaml \
    --checkpoint output_v7_flat8/checkpoints/final.pt \
    $EVAL_ARGS

echo ""
echo "============================================"
echo "  v7 Experiment Complete"
echo "============================================"
```

---

## 6. Evaluation

### 6.1 Eval Script

**File:** `scripts/eval_v7_gridworld.py` — based on `eval_v6_gridworld.py`

Changes from v6 eval:

**1. Model loading supports v7:**

```python
def load_model(config, checkpoint_path, device):
    if config.model_type == "flat":
        model = FlatTransformerModel(config).to(device)
    elif config.architecture_version == "v7" and config.sandwich_mode:
        model = FGNv7Model(config).to(device)
    elif config.architecture_version == "v6":
        model = FGNv6Model(config).to(device)
    else:
        raise ValueError(f"Unknown architecture: {config.architecture_version}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}

    pe_key = "pos_embed.weight"
    if pe_key in state and state[pe_key].shape[0] > config.max_seq_len:
        state[pe_key] = state[pe_key][:config.max_seq_len]

    model.load_state_dict(state)
    return model
```

**2. Forward pass routing:**

```python
# In evaluate_gridworld():
with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                         enabled=(device.type == "cuda")):
    if is_v7 or is_v6:
        result = model(input_ids, labels=labels, context_mask=context_mask)
    else:
        result = model(input_ids, labels=labels)
```

**3. Additional v7 metrics in output:**

```python
# After aggregation:
if is_v7:
    results["bottom_metric_cv"] = bot_cv_sum / n_batches
    results["top_metric_cv"] = top_cv_sum / n_batches
    results["bottom_avg_kappa"] = bot_kappa_sum / n_batches
    results["top_avg_kappa"] = top_kappa_sum / n_batches
```

**4. Print header adds v7 columns:**

```python
# Conditional column for bottom/top CV when v7
if is_v7:
    print(f"{'Condition':<26} {'SeqAcc':>8} {'TokAcc':>8} {'NavAcc':>8} "
          f"{'ManipAcc':>8} {'CE':>8} {'BotCV':>7} {'TopCV':>7} "
          f"{'Bot|κ|':>7} {'Top|κ|':>7}")
```

### 6.2 Eval Conditions (unchanged from v6)

```python
EVAL_CONDITIONS = [
    ("ID: 8-12rm 8-15st",
     dict(n_rooms_min=8, n_rooms_max=12, min_steps=8, max_steps=15,
          min_state_changes=2)),
    ("Near: 8-12rm 8-15st 4sc",
     dict(n_rooms_min=8, n_rooms_max=12, min_steps=8, max_steps=15,
          min_state_changes=4)),
    ("Near: 8-12rm 12-17st",
     dict(n_rooms_min=8, n_rooms_max=12, min_steps=12, max_steps=17,
          min_state_changes=2)),
    ("Near: 13-15rm 8-15st",
     dict(n_rooms_min=13, n_rooms_max=15, min_steps=8, max_steps=15,
          min_state_changes=2)),
    ("Far: 15-18rm 12-17st",
     dict(n_rooms_min=15, n_rooms_max=18, min_steps=12, max_steps=17,
          min_state_changes=2)),
    ("Far: 15-18rm 15-20st",
     dict(n_rooms_min=15, n_rooms_max=18, min_steps=15, max_steps=20,
          min_state_changes=2)),
]
```

### 6.3 Metrics Per Condition

| Metric | Source | Present in v6 |
|--------|--------|---------------|
| `seq_acc` | % of episodes where all supervised tokens correct | Yes |
| `token_acc` | % of individual supervised tokens correct | Yes |
| `nav_acc` | % of navigation action tokens correct | Yes |
| `manip_acc` | % of manipulation action tokens correct | Yes |
| `ce_loss` | Cross-entropy loss | Yes |
| `metric_cv` | Average metric coefficient of variation (all geo layers) | Yes |
| `avg_kappa` | Average absolute curvature (all geo layers) | Yes |
| `bottom_metric_cv` | CV for bottom geo layers only | **New** |
| `top_metric_cv` | CV for top geo layers only | **New** |
| `bottom_avg_kappa` | |κ| for bottom geo layers only | **New** |
| `top_avg_kappa` | |κ| for top geo layers only | **New** |

---

## 7. Unit Tests

Implement `__main__` self-tests in each new file (following v6 convention).

### 7.1 `layer_v7_geo.py` Tests

```python
if __name__ == "__main__":
    # Test 1: Basic forward pass shape
    # cfg with geo_metric_type="learned", geo_heads=4, architecture_version="v6"
    # Input: [B=2, N=16, d=64]
    # Verify: output shape [B, N, d], kappa shape [B, N], metric_cv is scalar, avg_entropy is scalar

    # Test 2: No attention components exist
    # Verify: not hasattr(layer, 'attention'), not hasattr(layer, 'norm_attn')

    # Test 3: Context conditioning
    # Pass context [B=2, d=64], verify output shape unchanged

    # Test 4: Gradient flow
    # Backward through output sum + kappa sum
    # Verify: metric params have non-zero grad
    # Verify: geo_route params have non-zero grad
    # Verify: ffn params have non-zero grad

    # Test 5: Flat metric mode
    # cfg with geo_metric_type="flat"
    # Verify: metric_cv == 0.0, kappa all zeros
```

### 7.2 `layer_v7_attn.py` Tests

```python
if __name__ == "__main__":
    # Test 1: Basic forward pass shape
    # Input: [B=2, N=16, d=64]
    # Verify: output shape [B, N, d]

    # Test 2: No geometric components exist
    # Verify: not hasattr(layer, 'metric'), not hasattr(layer, 'geo_route')

    # Test 3: Gradient flow
    # Backward, verify all params have grad

    # Test 4: Causal masking
    # Verify: output at position i doesn't depend on position j > i
```

### 7.3 `model_v7.py` Tests

```python
if __name__ == "__main__":
    # Test 1: Full forward/backward with labels
    # cfg: sandwich_mode=True, bottom=2, middle=4, top=2
    # Verify: result has all expected keys (matching v6 result dict keys for compatibility)
    # Verify: loss is scalar, gradient flows

    # Test 2: Layer count verification
    # Verify: len(model.bottom_geo) == 2
    # Verify: len(model.middle_attn) == 4
    # Verify: len(model.top_geo) == 2

    # Test 3: Separate metric weights
    # Verify: model.bottom_geo[0].metric is not model.top_geo[0].metric
    # Verify: bottom and top metric parameters are different objects

    # Test 4: Parameter groups are disjoint
    # geo_params ∩ attn_params = ∅
    # geo_params ∩ other_params = ∅
    # attn_params ∩ other_params = ∅
    # geo_params ∪ attn_params ∪ other_params = all params

    # Test 5: Context flows to all geo layers
    # Pass context_mask, verify no errors
    # Verify: bottom_metric_cv and top_metric_cv are both populated

    # Test 6: Parameter count comparison
    # Print total params, geo params, attn params, other params
    # Verify: total roughly comparable to v6 (~26M for d_model=256)

    # Test 7: v6 compatibility fields
    # Verify: result["escalation_rate"] == 0.0
    # Verify: result["esc_penalty"] == 0.0
    # Verify: result["scale_loss"] == 0.0
    # Verify: result["avg_gate"] == 0.0
```

---

## 8. Gradient Flow Analysis

This is the critical design advantage. Verify in tests:

### 8.1 Bottom Geo Layers

Every supervised token's gradient flows: `lm_head → norm → top_geo → middle_attn → bottom_geo → embed`. The bottom geo layers are in the **direct gradient path** for every token at every step. No gate can suppress gradient. No budget can skip positions.

**Test:** After `loss.backward()`, verify `bottom_geo[0].metric.h_proj.weight.grad.abs().sum() > 0`.

### 8.2 Top Geo Layers

Every supervised token's gradient flows: `lm_head → norm → top_geo`. The top geo layers are **immediately before the loss**. Strongest gradient signal of any geo layer in any FGN version.

**Test:** After `loss.backward()`, verify `top_geo[0].metric.h_proj.weight.grad.abs().sum() > 0` and that it's larger in magnitude than `bottom_geo[0]` equivalent.

### 8.3 Metric Gradient Path

In v6: `loss → lm_head → norm → layer → (gate or budget mask) → geo_route → metric`. The gate/budget can attenuate.

In v7: `loss → lm_head → norm → top_geo → metric` (no intermediary). And independently: `loss → ... → bottom_geo → metric` (through all subsequent layers, but no gate).

---

## 9. Success Criteria

### Primary (must achieve)

| Criterion | Threshold | What It Validates |
|-----------|-----------|-------------------|
| v7-sandwich ID seq_acc | ≥ 100% | Bottom geo layers aren't a bottleneck |
| v7-sandwich Far-OOD seq_acc | ≥ 78% (v6 baseline) | No regression from sandwich restructuring |
| ManipAcc all conditions | = 1.0 | Manipulation reasoning preserved |

### Target (expected improvement)

| Criterion | Threshold | What It Validates |
|-----------|-----------|-------------------|
| v7 Far-OOD NavAcc | > v6 Far-OOD NavAcc + 5pp | Full attention on pre-organized reps helps multi-hop nav |
| v7 Far-OOD seq_acc | > 80% | Compositional generalization improved |
| Both bottom_cv and top_cv | > 0.10 | Both metrics develop non-trivial structure |
| bottom_cv ≠ top_cv | Difference > 0.05 | Perceptual vs action geometry diverged |

### Failure Indicators (stop and diagnose)

| Signal | Diagnosis | Action |
|--------|-----------|--------|
| ID seq_acc < 100% | Bottom geo layers are a bottleneck | Compare against flat-8; if flat-8 gets 100%, problem is geo layers |
| ManipAcc < 1.0 | Sandwich broke manipulation reasoning | Check top_metric_cv — if zero, context isn't reaching top layers |
| v7 ≈ flat-8 on all conditions | Geometric preprocessing provides no value | Check representations: do connected rooms cluster after bottom geo? |
| top_metric_cv ≈ 0 | Context stale for top layers | Try recomputing context after middle attn (v7.1) |
| v7 < v6 on Far-OOD | Sandwich worse than mixed layers | Attention layers may need some geometric signal; try 1 mixed layer |

---

## 10. What's NOT in v7

Explicitly listing removed/excluded components and why:

| Component | Why Excluded |
|-----------|-------------|
| Gate mechanisms | Freeze at init (v4 lesson). Architecture enforces division. |
| Attention budgets | Unnecessary — layers are 100% one type or the other. |
| Escalation mechanism | No boundary to learn. Fixed roles. |
| Sharpness annealing | Nothing to anneal. No soft decisions. |
| Curvature reward/penalty | Disabled (`curvature_lambda=0.0`). Pure CE drives geometry (v3 lesson). |
| Geo auxiliary loss | Not needed — geo layers in direct gradient path, always receive signal. |
| Escalation penalty | Removed. No escalation exists. |
| Phased training | Not needed — no competition between pathways. Single phase. |
| Differential LR | Not needed — separate territories, no dominance risk. |
| Pretrained transformer injection | Future work (v8). v7 validates sandwich principle with trained-from-scratch components. |

---

## 11. Implementation Order

Implement in this exact order. Each step has a clear test before proceeding.

| Step | File | Test Gate |
|------|------|-----------|
| 1 | `fgn/config.py` | YAML round-trip with new fields |
| 2 | `fgn/geo_route.py` | Add "v7" to version check, existing tests still pass |
| 3 | `fgn/layer_v7_geo.py` | All 5 `__main__` tests pass |
| 4 | `fgn/layer_v7_attn.py` | All 4 `__main__` tests pass |
| 5 | `fgn/model_v7.py` | All 7 `__main__` tests pass |
| 6 | `configs/v7_sandwich.yaml` | Loads without error, creates model |
| 7 | `configs/v7_flat8.yaml` | Loads without error, creates flat model |
| 8 | `scripts/train_v7.py` | Runs 10 steps without error, loss decreases |
| 9 | `scripts/eval_v7_gridworld.py` | Runs eval on random checkpoint, prints table |
| 10 | `scripts/run_v7_gridworld.sh` | Full experiment end-to-end |

**Critical verification between steps 5 and 6:**

```python
# Quick smoke test — run in Python REPL
from fgn.config import FGNConfig
from fgn.model_v7 import FGNv7Model
import torch

cfg = FGNConfig.from_yaml("configs/v7_sandwich.yaml")
model = FGNv7Model(cfg)
n = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n:,}")  # Should be ~26-30M

x = torch.randint(0, 100, (2, 32))
labels = torch.randint(0, 100, (2, 32))
ctx_mask = torch.zeros(2, 32, dtype=torch.bool)
ctx_mask[:, :8] = True

result = model(x, labels=labels, context_mask=ctx_mask)
print(f"Loss: {result['loss'].item():.4f}")
print(f"Bottom CV: {result['bottom_metric_cv'].item():.4f}")
print(f"Top CV: {result['top_metric_cv'].item():.4f}")

result['loss'].backward()
for name, p in model.named_parameters():
    if p.grad is None:
        print(f"WARNING: No gradient for {name}")
```

---

## 12. Parameter Budget

**Estimated parameter counts (d_model=256, d_ff=1024, geo_heads=4, n_heads=8):**

| Component | Params | Count |
|-----------|--------|-------|
| Embeddings (token + position) | 50304×256 + 1024×256 | 13.1M |
| ContextPool | 256→64→1 + 256→256 | ~0.1M |
| GeoOnlyLayer (×4 total) | 4 × [MetricNet + GeoRoute + FFN + 3×LN] | 4 × 1.5M = 6.0M |
| AttnOnlyLayer (×4) | 4 × [Attention + FFN + 2×LN] | 4 × 1.8M = 7.2M |
| Final LN + LM head (tied) | 256 + 256 | ~0.001M |
| **Total** | | **~26.4M** |

For comparison: v6 was ~24M (6 mixed layers). flat-8 baseline should be set to match v7's parameter count by adjusting d_ff if needed.

---

## 13. Key Implementation Notes

### 13.1 GeoRoute Architecture Version

`GeoOnlyLayer` must create `GeoRoute` with the Q/K projection path. The config has `architecture_version="v7"` but GeoRoute currently only checks for `"v6"`. Either:
- Set `config.architecture_version = "v6"` when passing to GeoRoute, OR
- Modify GeoRoute to treat "v7" same as "v6" for Q/K path selection

**Recommended approach:** In `geo_route.py`, change:

```python
# Before:
if self.arch_version == "v6":

# After:
if self.arch_version in ("v6", "v7"):
```

This is a one-line change. No other modifications to geo_route.py.

### 13.2 CurvatureRegularization Layer Count

v6 passes `n_layers=6` to `CurvatureRegularization`. v7 should pass `n_layers = n_bottom_geo + n_top_geo = 4` (only geometric layers have curvature). The attention layers produce no curvature tensors.

### 13.3 Context Mask Handling

The `context_mask` computation in `GridWorldTask` is unchanged. It marks [WORLD] prefix positions. v7 passes it to `ContextPool` exactly like v6.

### 13.4 torch.compile Compatibility

v7's model structure (3 ModuleLists) should work with `torch.compile(model, mode="default")`. If compilation fails on the sandwich structure, compile each stage separately:

```python
# Fallback if full model compile fails:
model.bottom_geo = torch.compile(model.bottom_geo)
model.middle_attn = torch.compile(model.middle_attn)
model.top_geo = torch.compile(model.top_geo)
```

---

## 14. Predicted Results

| Condition | v7-sandwich | v6-metric | flat-8 |
|-----------|-------------|-----------|--------|
| ID (8-12rm, 8-15st) | 100% | 100% | 100% |
| Near (8-12rm, 12-17st) | 100% | 100% | 100% |
| Near (13-15rm, 8-15st) | ≥85% | 84% | ~40% |
| Far (15-18rm, 12-17st) | ≥80% | 78% | ~10% |
| Far (15-18rm, 15-20st) | ≥80% | 78% | ~6% |
| ManipAcc (all) | 1.0 | 1.0 | variable |

**Rationale for v7 > v6 prediction:** v7's 4 full attention layers provide 4× the effective attention compute of v6's budgeted attention (~0.9 effective layers). This should specifically help multi-hop navigation planning through unseen graph sizes, which is v6's primary failure mode.
