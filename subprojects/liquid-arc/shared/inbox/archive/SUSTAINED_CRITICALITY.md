# SUSTAINED_CRITICALITY: Self-Organized Phase Transition as Architectural Invariant

## Goal

Make the phase transition a PERMANENT, CONTROLLABLE feature of LiquidARC's dynamics — not a one-time training event. The system should maintain itself at the critical boundary through internal dynamics, capable of repeated reorganization.

## Background

The distillation experiment showed the phase transition cascade:
```
Step 120: D²=427 amp=0.7× ent=0.01 diag=0.999 tau=2.94 ← sub-critical
Step 123: D²= 97 amp=0.2× ent=3.03 diag=0.779 tau=2.84 ← BIFURCATION
Step 134: D²= 37 amp=0.1× ent=5.02 diag=0.056 tau=0.10 ← post-critical
```

The trigger: D²/4τ crosses ~18, softmax breaks degeneracy, MetricNet flips from amplifying to compressing. Currently this happens once and the system settles into a fixed post-transition state. We need the system to LIVE at the bifurcation point.

Reference: Droste et al. (2013) "Analytical investigation of self-organized criticality in neural networks" — proves that activity-dependent rewiring with asymmetric time scales creates an attractor at the critical point.

---

## Critical Design Constraint: Tau Management

### The problems with naive tau losses

**Problem 1: tau_var_loss maximizes variance blindly.** Pushing tau to extremes (some positions at tau_min, others at tau_max) doesn't produce useful computation — it produces saturated dynamics where half the positions freeze and half oscillate wildly.

**Problem 2: Criticality loss is scale-invariant.** D²/4τ = 18 is satisfied by (D²=36, τ=0.5) OR (D²=1440, τ=20). Without anchoring the absolute scale, tau drifts wherever the variance loss pushes it.

**Problem 3: No feedback from dynamics quality.** Nothing tells tau that τ=6.5 produces sluggish ODE integration. The task CE loss eventually signals this, but slowly and indirectly.

### What tau SHOULD do

Per-position adaptive integration speed:
- Fast τ (small) at positions needing rapid routing updates
- Slow τ (large) at positions that should preserve state
- The **RATIO** between positions matters, not absolute scale

### Design principles for tau

**1. Anchor the absolute scale through dynamics quality.**

The ODE integrates `dh/dt = -(1/τ)(h - target)` over 16 Euler steps. The convergence rate of this integration depends on τ relative to the step size dt = T/n_steps. If τ >> T, the position barely moves in 16 steps (sluggish). If τ << dt, the position overshoots and oscillates. The productive regime is τ ≈ 0.3-2.0 for T=1.0, n_steps=16.

Instead of a variance loss, anchor tau's MEAN to the productive range and encourage SPREAD around that mean:

```python
def tau_quality_loss(tau):
    """Encourage tau to be in the productive integration range
    with meaningful per-position differentiation.
    
    Args:
        tau: [B, N, 1] per-position time constants
    """
    tau_flat = tau.squeeze(-1)  # [B, N]
    
    # 1. Anchor mean to productive range (0.5-1.5)
    #    NOT a hard target — soft preference via smooth_l1
    tau_mean = tau_flat.mean(dim=-1)  # [B]
    mean_anchor = F.smooth_l1_loss(
        tau_mean, 
        torch.ones_like(tau_mean) * 1.0  # target mean ≈ 1.0
    )
    
    # 2. Encourage RATIO spread, not absolute spread
    #    Use log-space variance: var(log(τ)) measures multiplicative spread
    #    log(τ)_std = 0.5 means positions differ by ~1.6× (e^0.5)
    #    log(τ)_std = 1.0 means positions differ by ~2.7× (e^1.0)
    log_tau = torch.log(tau_flat + 1e-8)
    log_tau_std = log_tau.std(dim=-1)  # [B]
    
    # Target: meaningful differentiation without extremes
    # log_std ≈ 0.5-0.8 → positions differ by 1.6-2.2× (healthy)
    spread_target = 0.6
    spread_loss = (log_tau_std - spread_target) ** 2
    
    return mean_anchor + 0.5 * spread_loss
```

This replaces tau_var_loss entirely. It says: "keep tau centered in the productive range, with positions differing by ~2× in integration speed." No saturation at extremes.

**2. Couple tau to local dynamics quality, not just metric variance.**

The LTC residual `||h - target||` measures how far each position is from its diffusion target. Large residual = position hasn't converged = needs more integration time (lower τ) or the target is wrong. Small residual = position converged = can afford higher τ.

This is a DIRECT measure of dynamics quality per position:

```python
# Inside forward(), after computing dh_dt:
# self._last_residual already stores ||h - target|| per position

# Tau modulation from convergence quality (NOT a loss — structural coupling)
if self.tau_convergence_coupling_enabled:
    with torch.no_grad():
        residual = (h - target).norm(dim=-1, keepdim=True)  # [B, N, 1]
        residual_norm = residual / (residual.mean() + 1e-8)  # normalize
        
        # High residual → position is struggling → LOWER tau (integrate faster)
        # Low residual → position converged → HIGHER tau (preserve state)
        # This is the OPPOSITE of naive variance maximization
        convergence_factor = 1.0 / (1.0 + beta * residual_norm)  # [B, N, 1]
        # convergence_factor ∈ (0, 1]: high residual → small factor → tau decreases
        
        tau = tau * (0.5 + 0.5 * convergence_factor)  # modulate within [0.5τ, τ]
```

This creates an INTERNAL feedback loop: positions that are far from their target get faster integration (lower τ) to help them converge. Positions that have converged get slower integration (higher τ) to preserve their state. The feedback is local (per-position), continuous (every forward step), and based on dynamics quality, not arbitrary variance targets.

**3. Fix the criticality loss scale ambiguity.**

The criticality loss targets D²/4τ ≈ 18. To prevent scale drift, anchor it through the MetricNet's amplification factor `amp`, not through tau:

```python
def compute_criticality_loss(self, h, g, tau):
    """Target the critical regime via MetricNet compression,
    with tau anchored independently.
    """
    B, N, d = h.shape
    
    # Sample pairwise metric-weighted distances
    n_pairs = min(N * 4, N * (N-1) // 2)
    idx_i = torch.randint(0, N, (n_pairs,), device=h.device)
    idx_j = (idx_i + torch.randint(1, N, (n_pairs,), device=h.device)) % N
    
    delta = h[:, idx_i, :] - h[:, idx_j, :]
    g_avg = (g[:, idx_i, :] + g[:, idx_j, :]) / 2
    D_sq = (delta * g_avg * delta).sum(dim=-1)  # [B, n_pairs]
    
    D_sq_median = D_sq.median(dim=-1).values  # [B]
    tau_median = tau.squeeze(-1).median(dim=-1).values  # [B]
    
    ratio = D_sq_median / (4 * tau_median + 1e-8)
    
    # Target the ratio, but ALSO constrain absolute D² and tau separately:
    # This prevents the degenerate solutions
    
    # Ratio loss (primary)
    target_ratio = 18.0
    ratio_loss = F.smooth_l1_loss(ratio, torch.full_like(ratio, target_ratio))
    
    # D² scale anchor (secondary): keep D² in productive range
    # Too small → positions collapsed, no routing diversity
    # Too large → softmax too degenerate for any reasonable tau
    D_sq_log = torch.log(D_sq_median + 1e-8)
    D_sq_target_log = math.log(60.0)  # target D² ≈ 60 (with tau≈0.8, ratio≈18.7)
    D_sq_anchor = 0.1 * (D_sq_log - D_sq_target_log) ** 2
    
    # tau is anchored by tau_quality_loss separately — don't duplicate here
    
    return ratio_loss + D_sq_anchor
```

By anchoring D² independently (log-scale, soft target ≈ 60), the system can't satisfy D²/4τ = 18 by inflating both D² and τ to arbitrarily large values. The MetricNet must compress D² to ~60 via its amplification factor. With τ anchored at ~0.8-1.0 by tau_quality_loss, the ratio naturally falls near 60/(4×0.8) ≈ 18.75.

---

## Architecture Changes

### 1. Replace tau_var_loss with tau_quality_loss

Remove any existing loss that maximizes tau variance or tau standard deviation. Replace with `tau_quality_loss` defined above: mean anchor + log-space spread.

### 2. Add tau-convergence coupling

In `forward()`, after computing `target` and before computing `dh_dt`, modulate tau based on local convergence residual. This is a STRUCTURAL coupling in the forward pass, not a loss term. Positions that struggle get faster integration. Positions that converge get stability.

Config:
```python
tau_convergence_coupling_enabled: bool = False
tau_convergence_beta: float = 1.0  # sensitivity of coupling
```

### 3. τ-CV coupling (revised)

The earlier spec had τ-CV coupling as a structural modification. Revise to work WITH the convergence coupling, not against it:

```python
# The two couplings combine multiplicatively:
# tau_effective = tau_base * convergence_factor * cv_factor

if self.tau_cv_coupling_enabled:
    with torch.no_grad():
        g_mean = g.mean(dim=-1, keepdim=True)
        g_std = g.std(dim=-1, keepdim=True)
        local_cv = g_std / (g_mean + 1e-8)
        
        # CV coupling: high local metric variance → slow down (stabilize)
        cv_target = self.cv_coupling_target
        cv_factor = 1.0 + self.cv_coupling_strength * (local_cv - cv_target)
        cv_factor = cv_factor.clamp(0.5, 2.0)
        
        tau = tau * cv_factor

if self.tau_convergence_coupling_enabled:
    with torch.no_grad():
        residual = (h - target).norm(dim=-1, keepdim=True)
        residual_norm = residual / (residual.mean() + 1e-8)
        conv_factor = 1.0 / (1.0 + self.tau_convergence_beta * residual_norm)
        tau = tau * (0.5 + 0.5 * conv_factor)

# Final clamp to physical range
tau = tau.clamp(min=self.tau_min, max=self.tau_max)
```

The CV coupling is the Droste mechanism (metric complexity → integration rate). The convergence coupling is dynamics quality feedback (struggling → go faster). They compose: a position with high local CV AND high residual gets competing signals — slow down for stability vs speed up for convergence. The net effect depends on which signal is stronger, which is determined by the relative strengths (cv_coupling_strength vs tau_convergence_beta).

### 4. Criticality loss with D² anchoring

As defined above. Two components:
- `ratio_loss`: D²/4τ near critical value (the bifurcation target)
- `D_sq_anchor`: D² near productive range (prevents scale drift)

### 5. Curvature diversity loss (unchanged from original spec)

CV floor/ceiling + metric entropy. This prevents metric collapse without pushing tau to extremes.

---

## Loss Budget

```python
total_loss = (
    ce_loss                                              # primary task signal
    + lambda_crit * criticality_loss                     # D²/4τ near bifurcation (0.01-0.1)
    + lambda_curv * curvature_diversity_loss              # metric stays diverse (0.01)
    + lambda_tau * tau_quality_loss                       # tau in productive range (0.05)
    # NOTE: cv_homeostatic_loss (existing) stays as-is
    # NOTE: NO tau_var_loss — removed entirely
)
```

The structural couplings (convergence, CV) are NOT losses — they're forward-pass modifications that don't add to the loss landscape. They provide instant per-position feedback without optimizer interference.

---

## The Tau Design Philosophy

The key shift: tau is NOT a knob to be tuned by a variance loss. It's an EMERGENT property of each position's computational needs:

- **Where the ODE struggles to converge** (high residual) → tau drops → faster integration
- **Where the ODE has converged** (low residual) → tau rises → state preservation
- **Where metric is complex** (high local CV) → tau rises → careful processing
- **Where metric is flat** (low local CV) → tau drops → rapid exploration

The tau_quality_loss only ensures the MEAN stays in the productive range and the SPREAD is meaningful (positions differ by ~2× in integration speed). Everything else comes from the structural couplings responding to the actual dynamics quality.

This is the biological analog: neurons don't have their time constants set by a global variance maximizer. Each neuron's integration rate adapts to its local circuit dynamics — how much input it's receiving, how far it is from firing threshold, what its neighbors are doing. The time constant is LOCALLY ADAPTED, not globally optimized.

---

## Experiment Design

### Phase 1: Verify on ARC (where transition works)

**Experiment 1a: Baseline with full logging**
- Standard ARC training, NO new losses/couplings
- Log at every step: CV, tau_mean, tau_std, log_tau_std, D²_median, D²/4τ, entropy, cv_tau_product, amp, convergence_residual_mean, convergence_residual_std
- Capture through natural transition
- Verify: does log_tau_std provide better spread signal than tau_std?
- Verify: does cv_tau_product stay constant through transition?
- Verify: what is D² at the moment of bifurcation? (confirms target for anchoring)

**Experiment 1b: tau_quality_loss only (replace tau_var_loss)**
- Remove tau_var_loss, add tau_quality_loss (lambda=0.05)
- Does tau develop meaningful per-position differentiation?
- Does tau MEAN stay in [0.5, 1.5] range?
- Does log_tau_std reach ~0.6 (2× position ratio)?
- Does the transition still occur? (tau_quality_loss shouldn't block it)

**Experiment 1c: Add convergence coupling**
- tau_quality_loss + tau_convergence_coupling (beta=1.0)
- Does tau respond to local convergence quality?
- Do positions with high residual get lower tau?
- Does the overall dynamics quality improve (lower mean residual)?

**Experiment 1d: Add criticality loss with D² anchoring**
- Everything from 1c + criticality_loss (lambda=0.01) + D²_anchor
- Does D² converge to ~60? Does D²/4τ stay near 18?
- Does the system maintain criticality POST-transition?
- KEY: perturb MetricNet weights at step 8000, does it recover?

**Experiment 1e: Add CV coupling**
- Everything from 1d + τ-CV coupling (alpha=0.5, target=3.5)
- Does the full Droste-type mechanism produce sustained criticality?
- Does CV oscillate around the target or converge and stay?
- FINAL TEST: change task distribution at step 8000 (30% → 70% real ARC). Does the system undergo a SECOND transition?

### Phase 2: Controllable transitions

(Same as original spec — only proceed after Phase 1 validates the tau management)

---

## Config Additions

```python
# In LiquidARCConfig:

# Criticality
criticality_loss_enabled: bool = False
criticality_loss_lambda: float = 0.01
criticality_target_ratio: float = 18.0
criticality_D_sq_target: float = 60.0    # D² anchor (log-scale)

# Curvature diversity
curvature_diversity_loss_enabled: bool = False
curvature_diversity_lambda: float = 0.01
curvature_cv_floor: float = 2.0
curvature_cv_ceiling: float = 10.0

# Tau quality (REPLACES tau_var_loss)
tau_quality_loss_enabled: bool = False
tau_quality_lambda: float = 0.05
tau_mean_target: float = 1.0             # productive integration range center
tau_log_spread_target: float = 0.6       # log-space std target (~2× ratio)

# Tau-convergence coupling (structural, not loss)
tau_convergence_coupling_enabled: bool = False
tau_convergence_beta: float = 1.0

# Tau-CV coupling (structural, not loss)
tau_cv_coupling_enabled: bool = False
cv_coupling_target: float = 3.5
cv_coupling_strength: float = 0.5
```

## Logging Requirements

Every training step:
```
step | ce | xform% | CV | tau_mean | tau_std | log_tau_std | D²_med | D²/4τ | ent | ent/max | amp | ∇geo | ∇cont | cv_tau | residual_mean | residual_std | L_crit | L_curv | L_tau
```

The additions vs original spec: `log_tau_std`, `residual_mean`, `residual_std`, `L_tau`. These are essential for diagnosing whether the tau management is working.

## Success Criteria

1. **tau_mean stays in [0.3, 2.0]** throughout training (not drifting to extremes)
2. **log_tau_std reaches 0.4-0.8** (positions differ by 1.5-2.2× in integration speed)
3. **No tau saturation**: fewer than 10% of positions at tau_min or tau_max
4. **CV stays in [2.5, 7.0]** post-transition
5. **D²/4τ stays in [12, 25]** post-transition (near critical regime)
6. **System recovers from perturbation** within 200 steps
7. **Second transition occurs** when task distribution changes

## Dependencies

- Post-transition d=768 checkpoint
- ARC training pipeline
- Remove existing tau_var_loss from training code

## Priority

HIGH — this must be validated before any Mamba state capture or text integration work.

## References

- Droste, Do, Gross (2013) "Analytical investigation of self-organized criticality in neural networks" — PMC3565782
- Hesse, Gross (2014) "Self-organized criticality as a fundamental property of neural systems" — Frontiers
- Gross (2021) "Not One, but Many Critical States" — Frontiers Neural Circuits
- LiquidARC distillation transition data (steps 120-134 cascade)
