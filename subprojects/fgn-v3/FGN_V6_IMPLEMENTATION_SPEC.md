# FGN v6 Implementation Spec

## Context: Why v6

The v5 experiment produced the first genuine metric structure in the FGN program (CV=1.06, |κ|=1.22, organic layer-differentiated escalation). However, **the metric model catastrophically failed at eval**: 0% sequence accuracy on in-distribution data, while flat scored 100%.

**Root cause:** `MetricNetwork` computes `g_i = f(h_i)` — per-position, independently. It has no cross-positional context. On a task with randomized topology per episode, the metric learned training-episode-specific distance patterns that don't transfer to new episodes. On unseen episodes, the metric produces nonsensical distances, GeoRoute produces near-uniform weights, and the model fails.

**Secondary cause:** GeoRoute computes distances on raw representations with no Q/K projections. By layer 3+, the model needs content-based routing ("which past observation mentions a sink?"), but GeoRoute can only answer "which positions have similar representations?" This drove the v5 escalation drift where L2-L6 all converged to majority-attention usage.

**v6 goal:** Fix the metric to be episode-context-conditioned, give GeoRoute Q/K projections for content-aware geometric routing, use multi-head GeoRoute for multiple routing strategies, and constrain attention via hard budgets to maintain the hierarchical design under gradient pressure.

---

## Architecture Changes

### Overview

```
v5 layer:                          v6 layer:
  g = Metric(h_i)                    ctx = ContextPool(h, world_mask)
  h_geo = GeoRoute(h, g)             g = Metric(h_i, ctx)
  entropy → soft threshold            h_geo = MultiHeadGeoRoute(h, g)    ← with Q/K projections
  h_attn = Attention(h)              entropy → top-k budget
  h += esc_weight * h_attn            h_attn = Attention(h, budget_mask) ← sparse, budget-limited
  h += FFN(h)                         h += h_attn
                                      h += FFN(h)
```

### Change 1: Context-Conditioned Metric

**File:** `fgn/metric.py` — replace `MetricNetwork`

**Current (broken):**
```python
class MetricNetwork(nn.Module):
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """[B, N, d] -> [B, N, d], all components > 0."""
        return F.softplus(self.net(h))
```

**New:**
```python
class MetricNetwork(nn.Module):
    def __init__(self, config: FGNConfig):
        super().__init__()
        d = config.d_model
        bottleneck = d // 4
        
        # Context compression: d_model -> bottleneck
        self.context_proj = nn.Linear(d, bottleneck)
        
        # Main network: takes h_i concatenated with context
        # Input: bottleneck (from h_i) + bottleneck (from context) = 2 * bottleneck
        self.net = nn.Sequential(
            nn.Linear(bottleneck, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, d),
        )
        
        # Project h_i to bottleneck
        self.h_proj = nn.Linear(d, bottleneck)
        
        # Initialize for identity metric at init
        with torch.no_grad():
            self.net[-1].bias.fill_(math.log(math.e - 1))  # softplus^-1(1.0)
            nn.init.normal_(self.net[-1].weight, std=0.05)
    
    def forward(self, h: torch.Tensor, 
                context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B, N, d] layer-normed hidden states
            context: [B, d] episode context vector
            
        Returns:
            g: [B, N, d] positive diagonal metric
        """
        h_compressed = self.h_proj(h)                          # [B, N, bottleneck]
        ctx_expanded = self.context_proj(context).unsqueeze(1)  # [B, 1, bottleneck]
        
        # Additive conditioning (not concat — keeps parameter count low)
        combined = h_compressed + ctx_expanded                  # [B, N, bottleneck]
        
        return F.softplus(self.net(combined))
```

**Rationale:** Additive conditioning rather than concatenation keeps the parameter count comparable to v5. The context vector summarizes episode structure, so the metric can adapt its geometry to *this* episode's room graph.

**Parameter impact:** Adds `context_proj` (d × bottleneck = 256×64 = 16K) and `h_proj` (same). Total ~32K additional params per layer, ~192K total. Negligible vs 30M model.

### Change 2: Context Pooling Module

**File:** new file `fgn/context_pool.py`

```python
class ContextPool(nn.Module):
    """Extract episode context from the [WORLD] prefix.
    
    Pools over positions corresponding to the world description 
    to produce a single context vector per sequence.
    """
    
    def __init__(self, config: FGNConfig):
        super().__init__()
        d = config.d_model
        self.attn_pool = nn.Sequential(
            nn.Linear(d, d // 4),
            nn.Tanh(),
            nn.Linear(d // 4, 1),
        )
        self.out_proj = nn.Linear(d, d)
    
    def forward(self, h: torch.Tensor, 
                context_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Args:
            h: [B, N, d] hidden states
            context_mask: [B, N] bool, True for world-description positions.
                          If None, uses first 25% of sequence as heuristic.
        
        Returns:
            context: [B, d] pooled episode context
        """
        B, N, d = h.shape
        
        if context_mask is None:
            # Heuristic: first 25% of sequence is world description
            context_mask = torch.zeros(B, N, dtype=torch.bool, device=h.device)
            prefix_len = N // 4
            context_mask[:, :prefix_len] = True
        
        # Attention-weighted pooling over context positions
        scores = self.attn_pool(h).squeeze(-1)           # [B, N]
        scores = scores.masked_fill(~context_mask, -1e9)
        weights = F.softmax(scores, dim=-1)               # [B, N]
        pooled = (weights.unsqueeze(-1) * h).sum(dim=1)   # [B, d]
        
        return self.out_proj(pooled)
```

**Gridworld integration:** The `GridWorldTask` must output a `context_mask` tensor marking which token positions correspond to the `[WORLD]` prefix (room graph description). See Training Changes below.

**Design note:** Attention-weighted pooling (not mean pooling) lets the model learn which parts of the world description are most relevant to metric computation — room connectivity vs object placements vs appliance locations.

### Change 3: Multi-Head GeoRoute with Q/K Projections

**File:** `fgn/geo_route.py` — rewrite `GeoRoute`

**Current (broken):**
```python
# Distance on raw representations
diff = h_normed.unsqueeze(2) - h_normed.unsqueeze(1)  # [B, N, N, d]
g_avg = (g.unsqueeze(2) + g.unsqueeze(1)) / 2.0
d_sq = (diff * diff * g_avg).sum(-1)
```

**New:**
```python
class GeoRoute(nn.Module):
    def __init__(self, config: FGNConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.geo_heads       # Now 4 (was 1)
        self.d_head = config.d_model // config.geo_heads
        
        assert config.d_model % config.geo_heads == 0
        
        # Q/K projections — one per head
        self.W_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_o = nn.Linear(config.d_model, config.d_model, bias=False)
        
        # Per-head learned temperature (log-space)
        # Initialize with spread: heads cover different receptive fields
        init_log_ts = torch.linspace(-1.0, 2.0, self.n_heads)  # t from ~0.37 to ~7.4
        self.log_t = nn.Parameter(init_log_ts)  # [n_heads]
        
        self.attn_drop = nn.Dropout(config.dropout)
        self.chunk_size = 256
    
    def forward(self, h_normed, g, mask=None, return_weights=False):
        """
        Args:
            h_normed: [B, N, d_model]
            g: [B, N, d_model] diagonal metric (positive)
            mask: [N, N] causal mask
            return_weights: if True, return [B, N, N] averaged weights
            
        Returns:
            (h_geo [B, N, d_model], geo_weights or None)
        """
        B, N, _ = h_normed.shape
        H = self.n_heads
        d_h = self.d_head
        
        # Project to Q/K/V per head
        Q = self.W_q(h_normed).view(B, N, H, d_h).permute(0, 2, 1, 3)  # [B, H, N, d_h]
        K = self.W_k(h_normed).view(B, N, H, d_h).permute(0, 2, 1, 3)
        V = self.W_v(h_normed).view(B, N, H, d_h).permute(0, 2, 1, 3)
        
        # Reshape metric for per-head slicing
        # g is [B, N, d_model] -> [B, N, H, d_h] -> [B, H, N, d_h]
        g_heads = g.view(B, N, H, d_h).permute(0, 2, 1, 3)  # [B, H, N, d_h]
        
        # Per-head temperatures
        t = self.log_t.exp()  # [H]
        
        # Metric-weighted distance in projected space
        # diff_ij = Q_i - K_j
        # d²_ij = sum_k g_avg_ijk * (Q_ik - K_jk)²
        diff = Q.unsqueeze(3) - K.unsqueeze(2)                    # [B, H, N, N, d_h]
        g_avg = (g_heads.unsqueeze(3) + g_heads.unsqueeze(2)) / 2  # [B, H, N, N, d_h]
        d_sq = (diff * diff * g_avg).sum(-1)                       # [B, H, N, N]
        
        # Scale by per-head temperature
        log_w = -d_sq / (4.0 * t.view(1, H, 1, 1))               # [B, H, N, N]
        
        if mask is not None:
            log_w = log_w.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        w_geo = F.softmax(log_w, dim=-1)                           # [B, H, N, N]
        
        # Apply to values
        out = self.attn_drop(w_geo) @ V                            # [B, H, N, d_h]
        out = out.permute(0, 2, 1, 3).reshape(B, N, self.d_model)  # [B, N, d_model]
        
        weights_out = None
        if return_weights:
            weights_out = w_geo.mean(dim=1)  # [B, N, N] averaged across heads
        
        return self.W_o(out), weights_out
```

**Key differences from v5 GeoRoute:**
1. **Q/K projections** — distances computed in learned relevance space, not raw representation space. Now GeoRoute can answer "which position has the sink?" not just "which position looks similar."
2. **4 heads** (was 1) — each with independent Q/K slices and independent temperature. Enables simultaneous local and global routing.
3. **Per-head temperature** initialized with spread — head 0 tight (local), head 3 broad (global). Each head "sees" a different d_head-sized slice of the metric.
4. **Memory note:** The `[B, H, N, N, d_h]` diff tensor is large. For N=1024, H=4, d_h=64, B=8: ~134GB in fp32. **Must use chunked computation** for N > 256. Implement chunked path same as v5 but with head dimension. Alternatively, use bf16 throughout (67GB) and reduce batch size. See Training Changes.

**IMPORTANT — Chunked implementation:** The `_direct` method shown above is the reference. For actual training with N=1024, you MUST implement the chunked variant that processes query blocks of size C=128 to keep memory bounded. The chunked path structure is identical to v5's `_chunked` method but operates on `[B, H, C, N, d_h]` blocks instead of `[B, C, N, d]`.

**Parameter impact:** Adds W_q and W_k (each d_model² = 65K). n_heads-1 additional log_t values (3 params). Total ~130K additional per layer, ~780K total.

### Change 4: Attention Budget (Top-k) Replacing Threshold

**File:** `fgn/layer_v5.py` → rename to `fgn/layer_v6.py`

Replace the soft threshold escalation with a hard per-layer budget:

```python
class FGNv6Layer(nn.Module):
    def __init__(self, config: FGNConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_learned_metric = (config.geo_metric_type == "learned")
        
        # Attention budget: fraction of tokens allowed to use attention
        self.budget = config.attention_budgets[layer_idx]
        
        # ... norms, metric, geo_route, attention, ffn same as v5 ...
        # REMOVE: threshold_raw, sharpness, escalation_mode
        # ADD: context_pool (shared reference set at model level)
        
    def forward(self, h, mask=None, context=None):
        """
        Args:
            h: [B, N, d_model]
            mask: [N, N] causal mask
            context: [B, d] episode context (from ContextPool)
            
        Returns:
            (h, kappa, metric_cv, avg_entropy, escalation_rate)
        """
        B, N, _ = h.shape
        
        # Step 1: Context-conditioned metric
        if self.use_learned_metric:
            g = self.metric(self.norm_metric(h), context)  # NEW: context arg
            kappa = self.curvature(g)
            metric_cv = g.std() / g.mean()
        else:
            g = torch.ones_like(h)
            kappa = torch.zeros(B, N, device=h.device)
            metric_cv = torch.tensor(0.0, device=h.device)
        
        # Step 2: Multi-head GeoRoute with Q/K projections (ALWAYS runs)
        h_geo, geo_weights = self.geo_route(
            self.norm_geo(h), g, mask=mask, return_weights=True)
        h = h + self.resid_drop(h_geo)
        
        # Step 3: Compute entropy from GeoRoute weights
        eps = 1e-8
        entropy = -(geo_weights * torch.log(geo_weights + eps)).sum(dim=-1)
        positions = torch.arange(1, N + 1, device=h.device, dtype=h.dtype)
        max_entropy = torch.log(positions + eps).unsqueeze(0)
        normalized_entropy = entropy / (max_entropy + eps)  # [B, N]
        
        # Step 4: Budget-constrained attention (top-k most uncertain tokens)
        k = max(1, int(self.budget * N))
        
        if self.budget == 0.0:
            # Pure geometric layer — skip attention entirely
            escalation_rate = torch.tensor(0.0, device=h.device)
        else:
            # Select top-k highest entropy positions
            _, top_indices = normalized_entropy.topk(k, dim=-1)  # [B, k]
            
            # Create escalation mask
            escalate_mask = torch.zeros(B, N, device=h.device)
            escalate_mask.scatter_(1, top_indices, 1.0)  # [B, N]
            
            # Run attention on ALL tokens (dense), but only apply to budgeted ones
            h_attn = self.attention(self.norm_attn(h), mask=mask)
            h = h + escalate_mask.unsqueeze(-1) * self.resid_drop(h_attn)
            
            escalation_rate = escalate_mask.mean()
        
        # Step 5: FFN
        h = h + self.resid_drop(self.ffn(self.norm_ff(h)))
        
        avg_entropy = normalized_entropy.mean()
        return h, kappa, metric_cv, avg_entropy, escalation_rate
```

**Gradient note on top-k:** The `topk` operation itself is not differentiable (selection is discrete). However, gradients flow through the *values* at selected positions normally — `h_attn` receives gradients at the budgeted positions, and the attention parameters learn to produce useful outputs there. The metric receives gradient through the GeoRoute forward pass (step 2), not through the budget selection. This is the correct design: the metric is trained by the task loss flowing through GeoRoute, not by the escalation mechanism.

### Change 5: Model-Level Context Pool

**File:** `fgn/model_v5.py` → rename to `fgn/model_v6.py`

```python
class FGNv6Model(nn.Module):
    def __init__(self, config: FGNConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)
        
        # Context pooling (shared across layers)
        if config.geo_metric_type == "learned":
            self.context_pool = ContextPool(config)
        
        # Layers
        self.layers = nn.ModuleList([
            FGNv6Layer(config, i) for i in range(config.n_layers)
        ])
        
        # Output
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # ... rest same as v5 ...
    
    def forward(self, input_ids, labels=None, context_mask=None):
        """
        Args:
            input_ids: [B, N]
            labels: [B, N] optional
            context_mask: [B, N] bool, True for world-description tokens.
                          If None, ContextPool uses heuristic.
        """
        B, N = input_ids.shape
        pos = torch.arange(N, device=input_ids.device).unsqueeze(0)
        h = self.embed(input_ids) + self.pos_embed(pos)
        mask = torch.triu(torch.ones(N, N, device=input_ids.device, dtype=torch.bool), diagonal=1)
        
        # Compute episode context ONCE from initial embeddings
        if self.config.geo_metric_type == "learned":
            context = self.context_pool(h, context_mask)  # [B, d]
        else:
            context = None
        
        # Forward through layers (context shared)
        curvatures, metric_cvs, escalation_rates, entropies = [], [], [], []
        
        for layer in self.layers:
            h, kappa, m_cv, avg_ent, esc_rate = layer(h, mask=mask, context=context)
            curvatures.append(kappa)
            metric_cvs.append(m_cv)
            entropies.append(avg_ent)
            escalation_rates.append(esc_rate)
        
        h = self.norm(h)
        logits = self.lm_head(h)
        
        # ... loss computation same as v5, but REMOVE escalation_penalty ...
        # ... (budget replaces penalty) ...
```

**Design decision — context from initial embeddings vs intermediate representations:** Context is pooled from the embedding layer output (before any transformer layers). This means context captures the raw token content of the world description, not processed representations. This is intentional:

1. It prevents a circular dependency (layers need context to compute metric, but context would need layers to compute).
2. The world description is explicitly stated in text — the embedding layer already captures "room 3 connects to room 7" at the token level.
3. An alternative is to pool context from an intermediate layer (e.g., after layer 2), then use it for layers 3-6 only. This is more complex and can be explored as v6.1.

### Change 6: Config Updates

**File:** `fgn/config.py`

Add to `FGNConfig`:

```python
    # v6 architecture
    attention_budgets: Tuple[float, ...] = (0.00, 0.05, 0.10, 0.10, 0.20, 0.30)
    # Fraction of tokens per layer that can use attention.
    # L1=0% (pure geometric), L2=5%, L3-L4=10%, L5=20%, L6=30%
    # Must have len == n_layers
    
    geo_heads: int = 4    # Changed default from 1 to 4
```

Remove or deprecate:
```python
    # REMOVE these v5 params (replaced by budget):
    # escalation_mode, escalation_threshold, escalation_sharpness_init,
    # escalation_sharpness_final, escalation_sharpness_steps,
    # escalation_penalty_alpha
```

**Budget tuning guidance:** Start with `(0.00, 0.05, 0.10, 0.10, 0.20, 0.30)`. If metric develops well (CV > 0.5 by step 2000), try tighter: `(0.00, 0.00, 0.05, 0.05, 0.10, 0.20)`. If metric struggles (CV < 0.2), loosen: `(0.00, 0.10, 0.20, 0.20, 0.40, 0.50)`.

---

## Training Changes

### Task: Context Mask Output

**File:** `fgn/tasks/gridworld.py`

The `GridWorldTask.generate_batch()` must return a `context_mask` tensor identifying which positions correspond to the `[WORLD]` prefix.

**Current return:** `(input_ids, labels, metadata)`

**New return:** `(input_ids, labels, metadata)` where `metadata["context_mask"]` is a `[B, N]` bool tensor.

**Implementation:** After tokenizing the episode, find the token position of the `[PLAN]` or `[ACT]` marker. Everything before it is world description. Set `context_mask[:, :plan_start] = True`.

```python
# In GridWorldTask.generate_batch():
# After tokenizing the full episode text:
for b in range(batch_size):
    # Find where [WORLD] section ends
    # The world section ends at the first [PLAN] or [OBS] token
    plan_tokens = tokenizer.encode("[PLAN]", add_special_tokens=False)
    # Search for plan_tokens in input_ids[b]
    for pos in range(len(input_ids[b]) - len(plan_tokens)):
        if input_ids[b, pos:pos+len(plan_tokens)].tolist() == plan_tokens:
            context_mask[b, :pos] = True
            break
```

### Training Script

**File:** `scripts/train_v6.py` (new, based on `train_v5.py`)

Key differences from v5:

1. **Remove sharpness annealing.** Budget is fixed — no annealing schedule needed.
2. **Remove escalation penalty.** Budget makes it unnecessary.
3. **Pass context_mask through forward.** Training loop extracts `metadata["context_mask"]` from task and passes to model.
4. **Reduce seq_len to 512.** The 4-head GeoRoute with Q/K projections has 4× the memory of v5's 1-head no-projection GeoRoute. At N=1024, the `[B, H, N, N, d_h]` diff tensor is too large. N=512 is manageable. Compensate by increasing batch_size if GPU memory allows.
5. **Log per-layer budget utilization.** Report what fraction of the budget each layer actually uses (some layers may have all tokens below the entropy floor, using less than the budget allows).

```python
# Core training loop change:
for step in range(total_steps):
    input_ids, labels, meta = task.generate_batch(batch_size, device=device)
    context_mask = meta.get("context_mask", None)
    
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        result = compiled_model(input_ids, labels=labels, 
                                context_mask=context_mask)
        loss = result["loss"]
    
    # ... rest same as v5 ...
```

### Training Hyperparameters

| Parameter | v5 Value | v6 Value | Reason |
|-----------|----------|----------|--------|
| `max_steps` | 50,000 | 50,000 | Same |
| `batch_size` | 8 | 4 | Memory (4-head GeoRoute) |
| `seq_len` | 1024 | 512 | Memory (4-head GeoRoute) |
| `lr` | 3e-4 | 3e-4 | Same |
| `geo_heads` | 1 | 4 | Multi-head GeoRoute |
| `n_heads` (attn) | 8 | 8 | Same |
| `d_model` | 256 | 256 | Same |
| `n_layers` | 6 | 6 | Same |
| `warmup_steps` | 1000 | 1000 | Same |
| `weight_decay` | 0.1 | 0.1 | Same |
| `grad_clip` | 1.0 | 1.0 | Same |
| `curvature_lambda` | 0.0 | 0.0 | Metric learns from CE only |
| `escalation_penalty_alpha` | 0.0 | N/A | Removed, replaced by budget |
| `attention_budgets` | N/A | (0,0.05,0.1,0.1,0.2,0.3) | Per-layer hard budget |

### Three-Way Comparison

Train three models in sequence (same hardware, same task):

**Model A — Flat baseline:**
```yaml
model_type: flat
architecture_version: v6
d_model: 256
n_heads: 8
n_layers: 6
d_ff: 1024
max_seq_len: 512
```

**Model B — v6-flat (context-conditioned GeoRoute + budget, but g=1 constant metric):**
```yaml
model_type: fgn
architecture_version: v6
geo_metric_type: flat
d_model: 256
n_heads: 8
n_layers: 6
d_ff: 1024
max_seq_len: 512
geo_heads: 4
attention_budgets: [0.00, 0.05, 0.10, 0.10, 0.20, 0.30]
```

**Model C — v6-metric (full architecture with learned context-conditioned metric):**
```yaml
model_type: fgn
architecture_version: v6
geo_metric_type: learned
d_model: 256
n_heads: 8
n_layers: 6
d_ff: 1024
max_seq_len: 512
geo_heads: 4
attention_budgets: [0.00, 0.05, 0.10, 0.10, 0.20, 0.30]
```

This isolates the metric's contribution: B vs C differ only in whether the metric is learned or constant. A vs B shows whether the constrained GeoRoute mechanism (even without metric) helps.

---

## Eval Changes

### Fix 1: Nav/Manip Token Classification Bug

**File:** `scripts/eval_v5_gridworld.py` → `scripts/eval_v6_gridworld.py`

**Bug:** v5 eval reported `nav_acc = 0.0000` everywhere with note "all tokens classified as manip." The heuristic checking if the first supervised token is a "go" token is fragile because tokenization splits "go" differently in context.

**Fix:** Classify at the action level, not token level. The task generates actions as complete strings. Tag each action during generation:

```python
# In GridWorldTask: tag actions during episode generation
# Each action is either "go <room>" (navigation) or "take/put/clean/heat/cool ..." (manipulation)
# Store action boundaries and types in metadata

# In eval:
for action_start, action_end, action_type in meta["action_spans"][b]:
    action_preds = preds[b][action_start:action_end]
    action_labels = labels[b][action_start:action_end]
    action_mask = labels[b][action_start:action_end] != -100
    
    if not action_mask.any():
        continue
    
    matches = (action_preds[action_mask] == action_labels[action_mask])
    
    if action_type == "nav":
        total_nav_tokens += matches.numel()
        correct_nav_tokens += matches.sum().item()
    else:
        total_manip_tokens += matches.numel()
        correct_manip_tokens += matches.sum().item()
```

**Task-side change:** `GridWorldTask.generate_batch()` must include `metadata["action_spans"]`: a list (per batch element) of `(start_pos, end_pos, action_type)` tuples marking where each action appears in the token sequence and whether it's "nav" or "manip".

### Fix 2: Add Per-Layer Eval Stats

Report escalation rate and entropy per layer at eval time (v5 only reported averages):

```python
# In eval loop:
if "esc_rates_per_layer" in result:
    for i, r in enumerate(result["esc_rates_per_layer"]):
        per_layer_esc[i].append(r.item())
    for i, e in enumerate(result["entropies_per_layer"]):
        per_layer_ent[i].append(e.item())

# In output:
print(f"\nPer-layer escalation rates at eval:")
for i in range(n_layers):
    mean_esc = sum(per_layer_esc[i]) / len(per_layer_esc[i])
    mean_ent = sum(per_layer_ent[i]) / len(per_layer_ent[i])
    print(f"  L{i+1}: esc={mean_esc:.4f}, entropy={mean_ent:.4f}")
```

### Fix 3: Expanded OOD Conditions

Add a condition that tests topology generalization specifically (same room count, same plan length, but novel room types not in training):

```python
EVAL_CONDITIONS = [
    # In-distribution
    ("ID: 8-12rm 8-15st",
     dict(n_rooms_min=8, n_rooms_max=12, min_steps=8, max_steps=15,
          min_state_changes=2)),
    
    # Near-OOD: longer plans
    ("Near: 8-12rm 12-17st",
     dict(n_rooms_min=8, n_rooms_max=12, min_steps=12, max_steps=17,
          min_state_changes=2)),
    
    # Near-OOD: bigger world
    ("Near: 13-15rm 8-15st",
     dict(n_rooms_min=13, n_rooms_max=15, min_steps=8, max_steps=15,
          min_state_changes=2)),
    
    # Near-OOD: more state changes (tests manipulation depth)
    ("Near: 8-12rm 8-15st 4sc",
     dict(n_rooms_min=8, n_rooms_max=12, min_steps=8, max_steps=15,
          min_state_changes=4)),
    
    # Far-OOD: bigger + longer
    ("Far: 15-18rm 12-17st",
     dict(n_rooms_min=15, n_rooms_max=18, min_steps=12, max_steps=17,
          min_state_changes=2)),
    
    # Far-OOD: biggest + longest
    ("Far: 15-18rm 15-20st",
     dict(n_rooms_min=15, n_rooms_max=18, min_steps=15, max_steps=20,
          min_state_changes=2)),
]
```

### Eval Metric: Context Consistency

New eval metric that specifically tests whether the metric adapts to episode context:

```python
# After running eval, for v6-metric model only:
# Generate two batches from the SAME world (same room graph) but different plans.
# Compare metric outputs g between the two batches.
# If context-conditioning works, g should be similar for same-world episodes
# and different for different-world episodes.

def eval_metric_consistency(model, task, device, n_pairs=50):
    """Test if metric produces similar geometry for same-world episodes."""
    same_world_cosines = []
    diff_world_cosines = []
    
    for _ in range(n_pairs):
        # Same world, two different plans
        world = task.generate_world()
        ep1 = task.generate_episode(world)
        ep2 = task.generate_episode(world)
        
        # Different worlds
        world2 = task.generate_world()
        ep3 = task.generate_episode(world2)
        
        # Get metrics for each
        g1 = extract_metric(model, ep1, device)  # [N, d]
        g2 = extract_metric(model, ep2, device)  # [N, d]
        g3 = extract_metric(model, ep3, device)  # [N, d]
        
        # Compare (cosine similarity of metric field means)
        same_world_cosines.append(cosine(g1.mean(0), g2.mean(0)))
        diff_world_cosines.append(cosine(g1.mean(0), g3.mean(0)))
    
    print(f"Metric consistency: same_world={mean(same_world_cosines):.4f}, "
          f"diff_world={mean(diff_world_cosines):.4f}")
    # Success: same_world > diff_world significantly
```

This requires a small refactor of `GridWorldTask` to allow generating multiple episodes from the same world state. Add a `generate_world()` method that returns a world object, and a `generate_episode(world)` method that generates a plan+execution for a given world.

---

## Success Criteria

### Primary (must achieve to claim positive result):

1. **v6-metric ID seq_acc ≥ 0.95.** The v5 metric model got 0% here. If v6-metric can't match flat on in-distribution, the architecture is still broken.

2. **v6-metric manip_acc ≥ flat manip_acc + 3pp on Far-OOD conditions.** This is the core hypothesis: metric-informed geometric routing helps specifically at state-change decision points.

3. **Metric health: CV > 0.10, escalation stays within budgets.** If CV collapses to 0 or budgets are saturated everywhere (all budgeted tokens have max entropy), the metric isn't doing useful work.

### Secondary (support the finding):

4. **v6-metric nav_acc ≈ flat nav_acc (within ±2pp).** Metric shouldn't hurt navigation — it should primarily help manipulation.

5. **Per-layer entropy gradient maintained at eval.** L1 entropy ≈ 0 (GeoRoute confident), L6 entropy > L1 entropy. If entropy is uniform across layers at eval, the hierarchy collapsed on unseen data (same failure as v5).

6. **Metric consistency test passes.** Same-world metric cosine > different-world metric cosine by ≥ 0.1. This confirms context conditioning works.

7. **v6-flat > flat on any OOD condition.** This would show the constrained GeoRoute mechanism (Q/K projections + budget) helps even without a learned metric.

### Negative Result Criteria (when to stop):

- If v6-metric ID seq_acc = 0%, same as v5: the context-conditioning fix didn't help. Check whether context_mask is being passed correctly, whether ContextPool gradients flow.
- If v6-metric ID seq_acc > 0% but ≤ flat on all conditions: the metric is no longer harmful but provides no benefit. Publishable as a null result.
- If CV → 0 by step 5000: metric isn't developing structure. Try loosening budgets or increasing metric_lr_mult.

---

## File Inventory

### New Files
| File | Description |
|------|-------------|
| `fgn/context_pool.py` | ContextPool module (attention-weighted pooling over world prefix) |
| `fgn/layer_v6.py` | v6 layer with budget-constrained escalation and context-conditioned metric |
| `fgn/model_v6.py` | v6 model with shared ContextPool |
| `scripts/train_v6.py` | v6 training script (no sharpness annealing, passes context_mask) |
| `scripts/eval_v6_gridworld.py` | v6 eval with fixed nav/manip classification and metric consistency test |
| `configs/v6_flat.yaml` | Flat baseline config |
| `configs/v6_hier_flat.yaml` | v6-flat (GeoRoute with g=1) config |
| `configs/v6_hier_metric.yaml` | v6-metric (full architecture) config |

### Modified Files
| File | Changes |
|------|---------|
| `fgn/metric.py` | MetricNetwork now takes (h, context) instead of (h). Add context_proj, h_proj. Change forward signature. |
| `fgn/geo_route.py` | Add W_q, W_k projections. Change n_heads default to 4. Per-head log_t. Distance on projected space. Chunked path must handle [B,H,C,N,d_h]. |
| `fgn/config.py` | Add attention_budgets tuple. Change geo_heads default to 4. Remove escalation_* params. |
| `fgn/tasks/gridworld.py` | Add context_mask to metadata. Add action_spans to metadata. Add generate_world() and generate_episode(world) methods. |

### Unchanged Files
| File | Notes |
|------|-------|
| `fgn/curvature.py` | Same CurvatureEngine |
| `fgn/standard_attention.py` | Same StandardAttention |
| `fgn/flat_model.py` | Same flat baseline |
| `fgn/losses.py` | Same (curvature reg disabled anyway) |

---

## Implementation Order

1. **`fgn/config.py`** — Add new params, keep old ones for backward compat.
2. **`fgn/context_pool.py`** — New module, self-contained, easy to unit test.
3. **`fgn/metric.py`** — Add context parameter. Unit test: verify forward works with context vector, verify init still produces g ≈ 1.0.
4. **`fgn/geo_route.py`** — Add Q/K projections, multi-head temperatures, chunked path. Unit test: verify output shapes, gradient flow through metric, different heads produce different weights.
5. **`fgn/layer_v6.py`** — Combine components with budget mechanism. Unit test: verify budget=0.0 produces no attention, budget=1.0 applies attention everywhere, budget=0.1 selects exactly top 10% tokens.
6. **`fgn/model_v6.py`** — Wire ContextPool, pass context to layers. Unit test: full forward/backward pass, parameter group separation.
7. **`fgn/tasks/gridworld.py`** — Add context_mask and action_spans to metadata. Unit test: verify context_mask marks correct positions, action_spans cover all supervised tokens.
8. **`scripts/train_v6.py`** — Adapt from train_v5.py. Remove sharpness annealing, add context_mask passing.
9. **`scripts/eval_v6_gridworld.py`** — Fix nav/manip classification, add per-layer stats, add metric consistency test.
10. **configs/** — Create three YAML configs.

### Unit Test Checklist

Each component should pass before integration:

- [ ] `ContextPool`: `(B=2, N=32, d=64)` → `(B=2, d=64)`, gradient flows
- [ ] `MetricNetwork` with context: `(B=2, N=32, d=64)` + `(B=2, d=64)` → `(B=2, N=32, d=64)`, all positive, mean ≈ 1.0 at init
- [ ] `GeoRoute` 4-head with Q/K: `(B=2, N=32, d=64)` → `(B=2, N=32, d=64)`, 4 different weight patterns, gradient flows to metric
- [ ] `GeoRoute` chunked path (chunk_size=16, N=32): output matches direct path within fp tolerance
- [ ] `FGNv6Layer` budget=0.0: attention never fires, output identical to GeoRoute-only
- [ ] `FGNv6Layer` budget=0.3, N=32: exactly 10 tokens (32×0.3=9.6→10) get attention
- [ ] `FGNv6Model` full forward/backward: loss decreases over 10 steps on random data
- [ ] `GridWorldTask` context_mask: marks only [WORLD] prefix positions
- [ ] `GridWorldTask` action_spans: covers all supervised positions, types match action verbs

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Context pooling learns nothing (mean/uniform weights) | Medium | High | Monitor pool attention weights during training. If uniform, try cross-attention pooling instead of additive. |
| 4-head GeoRoute OOM on GPU | Medium | High | Chunked implementation mandatory. Reduce seq_len to 512. Reduce batch_size to 4. bf16 throughout. |
| Budget too tight → model can't learn | Low | Medium | Start with generous budget, tighten if metric develops. Monitor training loss — if it plateaus above 0.01 with tight budgets, loosen. |
| Budget too loose → same drift as v5 | Low | Medium | Budgets are hard caps, not soft thresholds. Drift is structurally impossible — L1 budget=0 means zero attention forever. |
| Q/K projections make GeoRoute = attention | Medium | Low | Not equivalent: GeoRoute uses RBF kernel (distance-based, monotonically decreasing) not dot-product. Metric modulates dimension-specific distances. Verify by comparing GeoRoute weight patterns vs attention weight patterns. |
| Context from embeddings too shallow | Medium | Medium | If metric consistency test fails (same-world ≈ diff-world), try pooling from after layer 2 instead of embeddings (v6.1). |
