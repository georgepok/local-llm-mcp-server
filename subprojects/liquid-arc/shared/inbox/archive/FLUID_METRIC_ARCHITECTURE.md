# TASK: Fluid Metric Architecture — Low-Rank Metric + Wider Bottleneck via Geometry Distillation

**From:** Claude Desktop (Research Direction)
**To:** Claude Code (Implementation)
**Date:** 2026-04-03
**Priority:** HIGH — addresses the fundamental geometric fluidity limitation

**Prerequisites:**
- Post-transition 5M checkpoint at `/workspace/liquid-arc/PRECIOUS_CHECKPOINTS/5m_post_transition.pt` (step 10000)
- Geometry distillation results in `shared/outbox/GEOMETRY_DISTILLATION_REPORT.md` (student reached 71.1% eval — PROVEN approach)
- ARC data at `/workspace/fgn-v3/data/arc-repo/data`
- Existing training infrastructure: `scripts/train.py`, `liquid_arc/` module

**Supersedes:** `GEOMETRY_DISTILLATION.md` (moved to archive — this spec incorporates and extends it)

---

## Motivation

### The Geometric Fluidity Problem

The current LiquidARC MetricNet uses a **diagonal metric**: g is a d-dimensional vector per position, applied as per-dimension scaling. The geodesic distance is:

```
D²(i,j) = Σ_k g_k · (h_i^k - h_j^k)²
```

This can only make dimensions MORE or LESS important. It CANNOT rotate the representation space. This is why:

1. **The Resonant Geometry fidelity gap** (ρ never exceeded 0.3) — diagonal metric can't embed arbitrary graph distances
2. **ARC geometry is domain-locked** — the post-transition metric organizes axis-aligned spatial structure that doesn't transfer to other representational formats
3. **The Mind can't process language** — linguistic structure is distributed across dimensions in rotated relationships a diagonal metric can't capture

### The MetricNet Capacity Problem

The current MetricNet bottleneck is 64 dimensions: `[h_normed ∥ context] → Linear(512→64) → GELU → Linear(64→256)`. All geometric variation passes through 64 dimensions. For ARC (spatial routing needs ~15 geometric modes), 64 is overprovisioned. For language (30-50+ independent relationship types) or multi-domain fluidity (manifold of manifolds), 64 is insufficient.

The WHERE/WHAT parameter ratio is 1:5 (MetricNet ~49K vs FFN ~263K), even though the metric is theoretically the primary object from which all computation derives (per FGN v3 framework).

### Why This Works: Geometry Distillation is Proven

The previous geometry distillation experiment demonstrated:
- Student reached **71.1% eval** at step 1000, surpassing teacher's 54.2% peak at step 21000
- **21× faster** convergence by bypassing the phase transition
- 100× slower geometric LR preserved the distilled regime throughout training
- The phase transition needs to happen **ONCE** — distillation propagates its product

This spec applies the same proven distillation mechanism to a redesigned architecture with the geometric capacity to be truly fluid.

---

## The Architectural Changes

### Change 1: Low-Rank-Plus-Diagonal Metric

Replace the diagonal metric output with `g = D + L·L^T`:

```python
# CURRENT MetricNet output (diagonal only):
# Linear(bottleneck → d_model) → Softplus → g  [B, N, d]
# Geodesic: D²(i,j) = Σ_k g_k · (h_i^k - h_j^k)²

# NEW MetricNet output (diagonal + low-rank):
# Linear(bottleneck → d_model) → Softplus → D  [B, N, d]       (diagonal)
# Linear(bottleneck → d_model * rank) → L      [B, N, d, rank]  (low-rank factors)
# Effective metric: g = diag(D) + L·L^T
# Geodesic: D²(i,j) = Σ_k D_k·(h_i^k - h_j^k)² + ||L^T(h_i - h_j)||²
```

**The SDPA factorization is PRESERVED.** This is the critical implementation detail:

```
D²(i,j) = [diagonal term] + [low-rank term]

Diagonal term: Σ_k D_k · (h_i^k - h_j^k)²
  = Σ_k D_k·(h_i^k)² - 2·D_k·h_i^k·h_j^k + D_k·(h_j^k)²
  → factors into SDPA with q_i = sqrt(D_i) ⊙ h_i (current implementation)

Low-rank term: ||L_i^T·h_i - L_j^T·h_j||²
  = (L^T·h_i)·(L^T·h_i) - 2·(L^T·h_i)·(L^T·h_j) + (L^T·h_j)·(L^T·h_j)
  → factors into SDPA with q_proj = L^T·h (rank-dimensional)

Combined: attention_logits = diagonal_SDPA_logits + projected_SDPA_logits
```

Both terms factor into separate SDPA calls. FlashAttention applies to BOTH. The combined logits go through a single softmax. No N×N distance matrix materialized.

### Change 2: Wider MetricNet Bottleneck

Widen from 64 to 256 dimensions:

```python
# CURRENT:
# metric_net_linear1: Linear(d_model*2, 64)   — 512→64
# metric_net_linear2: Linear(64, d_model)      — 64→256 (diagonal only)

# NEW:
# metric_net_linear1: Linear(d_model*2, 256)   — 1536→256 (at d=768)
# metric_net_linear2_diag: Linear(256, d_model) — 256→768 (diagonal D)
# metric_net_linear2_lr: Linear(256, d_model*rank) — 256→6144 (low-rank L, rank=8)
```

256 geometric modes instead of 64. The wider bottleneck allows the MetricNet to learn qualitatively different geometric patterns for different input types — spatial patterns in some modes, linguistic patterns in others, temporal patterns in yet others.

### Change 3: Rebalanced Parameter Budget

At d=768, rank=8:

| Component | Current Params | New Params | Change |
|-----------|---------------|------------|--------|
| MetricNet linear1 | 512×64+64 = 32,832 | 1536×256+256 = 393,472 | +360K |
| MetricNet linear2 (diag) | 64×768+768 = 49,920 | 256×768+768 = 197,376 | +147K |
| MetricNet linear2 (LR) | — | 256×6144+6144 = 1,579,008 | +1.58M |
| MetricNet total | ~49K | ~2.17M | +2.12M |
| FFN (reduce to compensate) | ~263K at d=256 / ~1.2M at d=768 | Reduce d_ffn | Adjust |
| TauNet | ~17K | ~17K (unchanged) | 0 |

**Note on total parameter budget:** At 5M total, the enlarged MetricNet (~2.2M) takes ~44% of the budget. This INVERTS the WHERE/WHAT ratio to ~2:1, making the metric the dominant component — matching its theoretical centrality. The FFN can be reduced (e.g., d_ffn from 1536 to 768 or even 512) to stay within budget. Alternatively, if the total model can grow to ~7M, keep FFN at current size.

**Agent decision:** Choose whichever approach keeps total params at 5-7M. The MetricNet size is fixed (it's the point of the experiment). The FFN is the flex component. Log the actual parameter counts.

---

## Implementation

### Phase 1: Modify ContinuousDynamics

In `liquid_arc/dynamics.py`, modify the `ContinuousDynamics` class:

```python
class ContinuousDynamics(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        d = config.d_model
        # Wider bottleneck
        d_metric_bottleneck = getattr(config, 'd_metric_bottleneck', 256)
        # Low-rank parameters
        metric_rank = getattr(config, 'metric_rank', 8)
        self.metric_rank = metric_rank
        
        # MetricNet — wider bottleneck
        self.metric_net_linear1 = nn.Linear(d * 2, d_metric_bottleneck)
        
        # Diagonal output (same as before but from wider bottleneck)
        self.metric_net_linear2_diag = nn.Linear(d_metric_bottleneck, d)
        
        # Low-rank output: produces L factors [B, N, d * rank]
        # Initialize near zero so model starts in diagonal regime
        self.metric_net_linear2_lr = nn.Linear(d_metric_bottleneck, d * metric_rank)
        # CRITICAL: Initialize low-rank output near zero
        nn.init.zeros_(self.metric_net_linear2_lr.weight)
        nn.init.zeros_(self.metric_net_linear2_lr.bias)
        
        # ... rest of __init__ unchanged (TauNet, W_v, W_o, FFN, norms) ...
        # Optionally reduce FFN:
        d_ffn = getattr(config, 'd_ffn', d * 2)  # allow config to reduce
        self.ffn_linear1 = nn.Linear(d, d_ffn)
        self.ffn_linear2 = nn.Linear(d_ffn, d)
    
    def compute_metric(self, h, context=None):
        """Compute diagonal + low-rank metric from hidden state.
        
        Returns:
            D: [B, N, d] diagonal metric weights (positive via Softplus)
            L: [B, N, d, rank] low-rank factors (raw, no activation)
        """
        B, N, d = h.shape
        h_normed = self.norm_geo(h)
        
        if context is not None:
            ctx_exp = context.unsqueeze(1).expand(B, N, -1)
            cat_input = torch.cat([h_normed, ctx_exp], dim=-1)
        else:
            cat_input = torch.cat([h_normed, h_normed], dim=-1)  # fallback
        
        bottleneck = F.gelu(self.metric_net_linear1(cat_input))  # [B, N, 256]
        
        # Diagonal component
        D = F.softplus(self.metric_net_linear2_diag(bottleneck))  # [B, N, d]
        
        # Low-rank component
        L_flat = self.metric_net_linear2_lr(bottleneck)  # [B, N, d*rank]
        L = L_flat.view(B, N, d, self.metric_rank)  # [B, N, d, rank]
        
        return D, L
    
    def compute_heat_kernel_sdpa(self, h, D, L, t_diffusion):
        """Compute heat kernel attention via DUAL SDPA — diagonal + projected.
        
        The key insight: both components factor into SDPA independently.
        
        Diagonal SDPA (existing):
            q_diag = sqrt(D / (4t)) ⊙ h    [B, N, d]
            k_diag = sqrt(D / (4t)) ⊙ h    [B, N, d]
            logits_diag = q_diag @ k_diag^T  [B, N, N]
        
        Low-rank SDPA (new):
            h_proj = L^T @ h                [B, N, rank]  (L is [B,N,d,rank])
            q_proj = h_proj / sqrt(4t)      [B, N, rank]
            k_proj = h_proj / sqrt(4t)      [B, N, rank]
            logits_lr = q_proj @ k_proj^T   [B, N, N]
        
        Combined:
            logits = logits_diag + logits_lr
            K = softmax(logits)
        """
        B, N, d = h.shape
        
        # Diagonal SDPA (existing mechanism)
        scale_diag = torch.sqrt(D / (4.0 * t_diffusion + 1e-8))  # [B, N, d]
        q_diag = scale_diag * h  # [B, N, d]
        k_diag = scale_diag * h  # [B, N, d]
        
        # Low-rank SDPA
        # h_proj = einsum('bnd,bndr->bnr', h, L)
        h_proj = torch.einsum('bnd,bndr->bnr', h, L)  # [B, N, rank]
        scale_lr = 1.0 / torch.sqrt(4.0 * t_diffusion + 1e-8)
        q_proj = h_proj * scale_lr  # [B, N, rank]
        k_proj = h_proj * scale_lr  # [B, N, rank]
        
        # Value computation (unchanged)
        V = self.w_v(self.norm_v(h))  # [B, N, d]
        
        # Option A: Combined logits then single softmax
        # This is mathematically correct but requires materializing logits
        logits_diag = torch.bmm(q_diag, k_diag.transpose(1, 2))  # [B, N, N]
        logits_lr = torch.bmm(q_proj, k_proj.transpose(1, 2))    # [B, N, N]
        logits = logits_diag + logits_lr
        
        # Apply mask if needed
        if self._mask is not None:
            logits = logits.masked_fill(~self._mask.unsqueeze(0), float('-inf'))
        
        attn = F.softmax(logits, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.bmm(attn, V)  # [B, N, d]
        
        # Option B: Use FlashAttention with concatenated Q,K
        # If N is large enough for FlashAttention to matter, concatenate:
        #   Q = cat([q_diag, q_proj], dim=-1)  [B, N, d+rank]
        #   K = cat([k_diag, k_proj], dim=-1)  [B, N, d+rank]
        #   V stays [B, N, d]
        #   F.scaled_dot_product_attention(Q, K, V) 
        # This computes exactly logits_diag + logits_lr via the concatenation identity.
        # PREFER THIS if N > 256 for FlashAttention acceleration.
        
        return out
    
    def forward(self, t, h):
        """One ODE step with low-rank metric."""
        D, L = self.compute_metric(h, self._context)
        
        t_diff = F.softplus(self.t_diffusion)
        out = self.compute_heat_kernel_sdpa(h, D, L, t_diff)
        
        target = h + self.w_o(out)
        
        # FFN
        ffn_out = self.ffn_linear2(F.gelu(self.ffn_linear1(self.norm_ffn(h))))
        target = target + ffn_out / self._n_steps
        
        # Tau (unchanged)
        h_normed = self.norm_tau(h)
        tau_raw = self.tau_net_linear2(F.gelu(self.tau_net_linear1(h_normed)))
        tau = self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(tau_raw)
        
        # LTC dynamics (unchanged)
        dhdt = -(1.0 / tau) * (h - target)
        
        return dhdt
```

### CRITICAL: FlashAttention via Concatenation

For production use with FlashAttention (avoids materializing N×N logits):

```python
def compute_heat_kernel_flash(self, h, D, L, t_diffusion):
    """FlashAttention-compatible dual SDPA via Q/K concatenation.
    
    Mathematical identity:
        (q_diag @ k_diag^T) + (q_proj @ k_proj^T) 
        = [q_diag, q_proj] @ [k_diag, k_proj]^T
    
    So concatenating the diagonal and projected queries/keys into single
    Q, K tensors of dimension (d + rank) gives the combined logits in
    one SDPA call. FlashAttention handles this natively.
    """
    B, N, d = h.shape
    
    scale_diag = torch.sqrt(D / (4.0 * t_diffusion + 1e-8))
    q_diag = scale_diag * h
    k_diag = scale_diag * h
    
    h_proj = torch.einsum('bnd,bndr->bnr', h, L)
    scale_lr = 1.0 / torch.sqrt(4.0 * t_diffusion + 1e-8)
    q_proj = h_proj * scale_lr
    k_proj = h_proj * scale_lr
    
    # Concatenate: [B, N, d+rank]
    Q = torch.cat([q_diag, q_proj], dim=-1)
    K = torch.cat([k_diag, k_proj], dim=-1)
    V = self.w_v(self.norm_v(h))
    
    # Single SDPA call — FlashAttention compatible
    # Note: V is d-dimensional, Q/K are (d+rank)-dimensional
    # F.scaled_dot_product_attention handles this (logits are N×N regardless of head dim)
    # BUT: PyTorch SDPA requires Q, K, V to have matching last dim OR use multi-head format
    
    # Solution: add rank zero-columns to V (or use the manual logit path for now)
    # ACTUALLY: SDPA computes softmax(QK^T / sqrt(d_k)) @ V
    # d_k = Q.shape[-1] = d + rank
    # The /sqrt(d_k) normalization changes! We need to compensate.
    
    # Correct approach: don't use the built-in scaling.
    # Use F.scaled_dot_product_attention with scale=1.0 (we pre-scaled Q and K)
    # V padding: pad V with rank zeros so shapes match
    V_padded = F.pad(V, (0, self.metric_rank))  # [B, N, d+rank]
    
    # Reshape for multi-head SDPA (1 head)
    Q = Q.unsqueeze(1)  # [B, 1, N, d+rank]
    K = K.unsqueeze(1)
    V_padded = V_padded.unsqueeze(1)
    
    attn_mask = self._mask.unsqueeze(0).unsqueeze(0) if self._mask is not None else None
    
    out_padded = F.scaled_dot_product_attention(
        Q, K, V_padded,
        attn_mask=attn_mask,
        scale=1.0,  # We pre-scaled Q and K
    )  # [B, 1, N, d+rank]
    
    # Strip the padding
    out = out_padded[:, 0, :, :d]  # [B, N, d]
    
    return out
```

**Agent note:** Test BOTH the explicit logit path (Option A) and the FlashAttention path. Use Option A for correctness verification, then switch to Flash if N > 256. The V-padding approach adds negligible compute (rank=8 zeros) but enables FlashAttention's memory-efficient backward pass.

---

## Phase 2: Geometry Distillation into New Architecture

### Step 1: Transfer diagonal weights from teacher

The teacher's MetricNet has shape `Linear(512→64)` → `Linear(64→256)`. The new architecture has `Linear(1536→256)` → `Linear(256→768)` + `Linear(256→6144)`.

The shapes DON'T match directly. Use **Approach A from the previous spec** — statistical target matching — adapted for the new architecture:

```python
def initialize_from_teacher(student, teacher_checkpoint, device='cuda'):
    """Initialize the new architecture's MetricNet to reproduce teacher's geometry.
    
    Strategy:
    1. Load teacher model
    2. Run teacher on batch of tasks, record g statistics per ODE step
    3. Run short optimization: adjust student's MetricNet so its DIAGONAL output
       matches teacher's g statistics. Low-rank output stays near zero.
    4. After init, student produces same CV/tau profile as teacher.
    """
    # Load teacher
    teacher = load_teacher_model(teacher_checkpoint, device)
    teacher.eval()
    
    # Record teacher geometry targets (reuse from geometry_targets.pt if available)
    # OR record fresh:
    teacher_stats = record_teacher_geometry(teacher, n_tasks=500, device=device)
    target_cv = teacher_stats['global']['cv_mean']  # ~6.75
    target_tau = teacher_stats['global']['tau_mean']  # ~0.65
    
    # Also transfer what CAN be transferred directly:
    # - TauNet weights (same shape if d_model matches)
    # - t_diffusion, alpha_logit
    # - context_pool
    # - W_v, W_o (same shape)
    teacher_sd = get_teacher_state_dict(teacher_checkpoint)
    student_sd = dict(student.named_parameters())
    
    direct_transfer_keys = [
        'dynamics.tau_net_linear1.weight', 'dynamics.tau_net_linear1.bias',
        'dynamics.tau_net_linear2.weight', 'dynamics.tau_net_linear2.bias',
        'dynamics.t_diffusion', 'dynamics.alpha_logit',
        'dynamics.w_v.weight', 'dynamics.w_o.weight',
        # Norms
        'dynamics.norm_geo.weight', 'dynamics.norm_geo.bias',
        'dynamics.norm_v.weight', 'dynamics.norm_v.bias',
        'dynamics.norm_tau.weight', 'dynamics.norm_tau.bias',
        'dynamics.norm_ffn.weight', 'dynamics.norm_ffn.bias',
    ]
    # Also context_pool and embedding if shapes match
    
    transferred = 0
    for key in direct_transfer_keys:
        if key in teacher_sd:
            p = student_sd.get(key)
            if p is not None and p.shape == teacher_sd[key].shape:
                p.data.copy_(teacher_sd[key])
                transferred += 1
    
    print(f"Direct transfer: {transferred} parameters")
    
    # Now optimize MetricNet to match teacher's geometry
    # Only the diagonal output — low-rank stays at zero
    init_params = list(student.dynamics.metric_net_linear1.parameters()) + \
                  list(student.dynamics.metric_net_linear2_diag.parameters())
    
    init_opt = torch.optim.Adam(init_params, lr=1e-3)
    
    procedural = ProceduralARCTask(seq_len=2048)
    
    for step in range(500):
        batch = procedural.generate_batch(batch_size=4, device=device)
        h0 = student.embed(batch)  # get initial hidden state
        
        D, L = student.dynamics.compute_metric(h0)
        
        current_cv = D.std() / (D.mean() + 1e-8)
        
        cv_loss = (current_cv - target_cv) ** 2
        # Also match per-step profile if available
        
        init_opt.zero_grad()
        cv_loss.backward()
        init_opt.step()
        
        if (step + 1) % 100 == 0:
            print(f"  Init {step+1}: CV={current_cv.item():.3f} (target {target_cv:.3f})")
    
    # Verify: L should still be near zero
    L_norm = student.dynamics.metric_net_linear2_lr.weight.norm().item()
    print(f"  Low-rank weight norm: {L_norm:.6f} (should be ~0)")
    print(f"  Final CV: {current_cv.item():.3f}")
```

### Step 2: Train with 100× geometric LR ratio

Same proven mechanism as the first distillation experiment:

```python
# Three parameter groups:
# Group 1: ALL MetricNet parameters (diagonal + low-rank + bottleneck)
#           Learn at base_lr * 0.01 (100× slower)
# Group 2: TauNet, t_diffusion, alpha_logit, context_pool
#           Learn at base_lr * 0.01 (geometric infrastructure)
# Group 3: FFN, W_v, W_o, embedding, output head
#           Learn at base_lr (content parameters)

geometric_params = (
    list(model.dynamics.metric_net_linear1.parameters()) +
    list(model.dynamics.metric_net_linear2_diag.parameters()) +
    list(model.dynamics.metric_net_linear2_lr.parameters()) +
    list(model.dynamics.tau_net_linear1.parameters()) +
    list(model.dynamics.tau_net_linear2.parameters()) +
    [model.dynamics.t_diffusion, model.dynamics.alpha_logit] +
    list(model.context_pool.parameters())
)

geo_param_ids = {id(p) for p in geometric_params}
content_params = [p for p in model.parameters() if id(p) not in geo_param_ids]

optimizer = torch.optim.AdamW([
    {'params': geometric_params, 'lr': 3e-4 * 0.01},   # 3e-6
    {'params': content_params, 'lr': 3e-4},              # 3e-4
], weight_decay=0.01)
```

---

## Phase 3: Training Protocol

### Stage A: ARC-only (validate distillation works with new architecture)

```bash
python scripts/train_fluid_metric.py \
  --config configs/liquid_arc_fluid_metric.yaml \
  --teacher_checkpoint /workspace/liquid-arc/PRECIOUS_CHECKPOINTS/5m_post_transition.pt \
  --data_dir /workspace/fgn-v3/data/arc-repo/data \
  --stage arc_only \
  --max_steps 5000 \
  --output_dir output/fluid_metric/stage_a
```

**Success criteria:**
- CV stays in 6-8 range throughout (distilled geometry preserved)
- ARC eval xform ≥ 30% by step 1000 (bypassed transition)
- Low-rank L_norm stays near zero (no rotational pressure from ARC alone)
- Total params logged and within 5-7M budget

### Stage B: Multi-domain (test if low-rank terms develop under diverse pressure)

Add text data alongside ARC. Use a small text corpus — WikiText-2 or similar that fits on the Spark. The model processes text tokens through the same architecture, with next-token prediction as the text loss:

```bash
python scripts/train_fluid_metric.py \
  --config configs/liquid_arc_fluid_metric.yaml \
  --resume output/fluid_metric/stage_a/step_5000.pt \
  --data_dir /workspace/fgn-v3/data/arc-repo/data \
  --text_data /workspace/liquid-arc/data/wikitext-2/ \
  --stage multi_domain \
  --text_mix_ratio 0.3 \
  --max_steps 10000 \
  --output_dir output/fluid_metric/stage_b
```

**Text data handling:**
- Tokenize with a simple BPE tokenizer (tiktoken or similar available on Spark)
- Map token IDs to the same embedding space (add a text_embed layer if needed, separate from ARC color/pos embeddings)
- Text loss: cross-entropy on next-token prediction
- Combined loss: `loss = arc_transform_loss + text_weight * text_ce_loss`
- `text_weight`: start at 0.1, can tune

**What to monitor:**
- Does L_norm grow? (Low-rank terms activating = rotational geometry developing)
- Does CV change when processing text vs ARC? (Different metric profiles per domain)
- Does ARC performance degrade? (Text training shouldn't destroy spatial geometry if low-rank terms handle text independently)
- Text perplexity: any meaningful language modeling? (5M params won't be GPT-4, but should beat random)

### Stage C: Universality probe (same as previous experiments)

From the Stage B checkpoint, test rapid adaptation to sorting, logic, graph coloring, pattern completion:

```bash
for DOMAIN in sorting logic_inference graph_coloring pattern_completion; do
    python scripts/train_fluid_metric.py \
      --resume output/fluid_metric/stage_b/step_10000.pt \
      --domain $DOMAIN \
      --max_steps 500 \
      --output_dir output/fluid_metric/universality/$DOMAIN
done
```

**Key comparison:** Are transfer speeds BETTER than the original diagonal model? If the low-rank metric developed rotational structure that helps non-ARC domains, transfer should be faster.

---

## Config File

Create `configs/liquid_arc_fluid_metric.yaml`:

```yaml
# LiquidARC — Fluid Metric Architecture
# Low-rank metric + wider bottleneck + geometry distillation

d_model: 768
d_metric_bottleneck: 256    # wider: was 64
metric_rank: 8               # low-rank factors
d_ffn: 768                   # reduced from 1536 to stay within 5-7M budget
max_seq_len: 2048
n_ode_steps: 16
tau_min: 0.1
tau_max: 3.0
t_diffusion_init: 1.0
dropout: 0.1
n_colors: 10
n_roles: 8
n_sep_types: 4
max_grid_size: 30
max_grids: 16
model_type: liquid
transform_weight: 5.0
copy_weight: 0.05
alpha_logit_init: 2.2
use_torch_compile: true

# CV floor/ceiling
cv_floor_target: 3.0
cv_ceiling_target: 8.0
cv_floor_lambda: 0.1

# Optimizer
base_lr: 0.0003
geometric_lr_ratio: 0.01     # 100× slower for MetricNet+TauNet
warmup_steps: 500
weight_decay: 0.01

# Training
use_procedural: true
real_arc_mix_ratio: 0.3
batch_size: 4

# Multi-domain (Stage B)
text_mix_ratio: 0.0          # 0.0 for Stage A, 0.3 for Stage B
text_loss_weight: 0.1
```

---

## Diagnostics to Log

Every 50 steps, log:

```python
diagnostics = {
    'step': step,
    'loss_total': loss.item(),
    'loss_arc': arc_loss.item(),
    'loss_text': text_loss.item() if text_enabled else 0,
    
    # Metric diagnostics
    'D_cv': D.std().item() / (D.mean().item() + 1e-8),  # diagonal CV
    'D_mean': D.mean().item(),
    'L_norm': L.norm().item(),                            # low-rank magnitude
    'L_rank_usage': (L.norm(dim=2).mean(dim=[0,1]) > 0.01).sum().item(),  # how many ranks active
    
    # Per-domain metrics (if multi-domain)
    'cv_on_arc': cv_arc,      # CV when processing ARC batches
    'cv_on_text': cv_text,    # CV when processing text batches
    'L_norm_arc': L_norm_arc,
    'L_norm_text': L_norm_text,
    
    # Standard
    'tau_mean': tau.mean().item(),
    'tau_std': tau.std().item(),
    'h_norm': h.norm().item(),
    'eval_xform': eval_accuracy,
    'text_ppl': text_perplexity if text_enabled else 0,
    
    # Parameter counts
    'metricnet_params': count_metricnet_params,
    'ffn_params': count_ffn_params,
    'total_params': total_params,
}
```

The CRITICAL new diagnostics are `L_norm` and `L_norm_arc` vs `L_norm_text`. If L_norm grows only during text processing and stays flat during ARC processing, the low-rank metric is learning domain-specific rotational geometry while the diagonal component handles spatial routing. That's fluidity.

---

## Success Criteria

### Stage A (ARC-only)
- **Minimum:** Model trains without errors. CV preserved in 6-8 range. ARC eval xform ≥ 25% by step 2000.
- **Good:** ARC eval matches or exceeds original distilled model's 71.1%. L_norm stays near zero (no rotational pressure from ARC alone).
- **Note:** If ARC performance is significantly WORSE than 71.1%, the wider bottleneck or reduced FFN may be hurting. Report this — it constrains the parameter budget decision.

### Stage B (Multi-domain)
- **Minimum:** Both ARC and text losses decrease. Model doesn't collapse.
- **Good:** L_norm grows during text training. `L_norm_text > L_norm_arc`. ARC performance doesn't degrade from Stage A.
- **Strong:** Different CV profiles on ARC vs text batches — the metric produces genuinely different geometries for different domains. Text perplexity beats random baseline (< 100 on WikiText-2).
- **Headline:** L_rank_usage > 1 — multiple low-rank dimensions activate, each potentially encoding different rotational structures. This is the empirical signature of fluid geometry.

### Stage C (Universality)
- **Success:** Transfer speeds match or beat the original diagonal model.
- **Strong:** Transfer speeds are FASTER — the rotational geometry provides richer domain-adaptive substrate.

---

## Output

Report to `shared/outbox/FLUID_METRIC_REPORT.md`

Include:
1. Parameter count breakdown (MetricNet, FFN, total)
2. SDPA verification: does concatenated Q/K produce same results as explicit logits?
3. Distillation initialization results: CV match to teacher
4. Stage A training curves (CV, eval xform, L_norm)
5. Stage B training curves — ESPECIALLY L_norm_arc vs L_norm_text, and per-domain CV
6. Stage C universality probe transfer speeds
7. Comparison table: diagonal model vs fluid metric model on all metrics
8. Assessment: does the low-rank metric develop domain-specific rotational geometry?
9. Assessment: is ARC performance preserved despite the architectural rebalancing?
10. Recommendations: should metric_rank increase? Should bottleneck widen further? Is FFN reduction acceptable?

**This experiment tests whether the fundamental geometric fluidity limitation — diagonal metric can't rotate — is the root cause of domain-specific geometry. If low-rank terms develop under multi-domain pressure while diagonal terms preserve the distilled ARC geometry, we've demonstrated that a single architecture can produce genuinely different geometric structures for different domains. That's the Fluid Geometry Networks vision made concrete.**
