# LAYER-WISE PHASE 1 RESULTS — Analysis & Fix

## What the Data Shows

D² runaway through depth is the dominant problem:

```
Layer depth:    Early (0-11)    Mid (12-23)    Late (24-35)
D²:             268             1,533          20,708
Max individual: 1.9 → 56,832   (30,000× increase through depth)
```

Each ODE Euler step amplifies pairwise distances. Over 36 layers, this compounds into runaway. By layer 35, heat kernel produces exp(-56832/4τ) ≈ 0 for all pairs. The routing signal is destroyed by the ODE's own amplification.

The B_within/B_across pattern is structurally correct (geometry separates events) but magnitudes are near-zero (overwhelmed by D² saturation).

The causal chain test (5/6 = 5/6) shows no improvement because the pre-transition checkpoint (CV=0.4) amplifies rather than compresses, and 36 layers of amplification = catastrophic D² growth.

## Root Cause

The MetricNet at CV=0.4 is in AMPLIFICATION mode (amp > 1.0). In the distillation transition data, the MetricNet was at amp=0.7× before transition. Post-transition it switched to amp=0.1× (compression). The current checkpoint is pre-transition: it amplifies distances at every step.

With single-shot ODE (16 steps, no inter-layer), the amplification is bounded — 16 steps of slight amplification. With layer-wise co-processing (36 steps, fresh residual input at each), the amplification COMPOUNDS: each layer's residual stream is amplified, then the NEXT layer's (already evolved) residual is amplified further.

## Immediate Fix: Norm Anchoring + Stronger Sensory Forcing

### 1. Norm anchoring (prevents D² runaway)

After each ODE step, re-normalize the ODE state to match the residual stream's scale:

```python
def process_layer(self, layer_idx, h_residual):
    # ... existing ODE step ...
    
    # Anchor ODE state norms to residual stream
    h_ode_norm = self.h_ode.norm(dim=-1, keepdim=True)  # [B, N, 1]
    h_res_norm = h_residual.norm(dim=-1, keepdim=True)   # [B, N, 1]
    scale = h_res_norm / (h_ode_norm + 1e-8)
    self.h_ode = self.h_ode * scale.clamp(0.5, 2.0)  # bounded rescaling
```

This prevents multiplicative D² growth. The ODE state can't diverge beyond 2× the residual stream's scale at any layer. The MetricNet still computes routing (which DIMENSIONS matter), but the absolute DISTANCES stay bounded.

### 2. Stronger sensory forcing (α = 0.6, up from 0.2)

```python
# Before:
alpha = 0.2  # ODE retains 80%, gets 20% new info
self.h_ode[:, K:, :] = (1 - alpha) * self.h_ode[:, K:, :] + alpha * h_residual

# After:
alpha = 0.6  # ODE retains 40%, gets 60% new info
self.h_ode[:, K:, :] = (1 - alpha) * self.h_ode[:, K:, :] + alpha * h_residual
```

With α=0.6, the ODE state tracks the residual stream closely. The 40% it retains is the geometric CORRECTION — what the MetricNet and heat kernel routing added. The 60% re-anchoring prevents divergence.

This changes the ODE's role: from "parallel processor building independent geometric state" to "geometric perturbation engine nudging the residual stream." The ODE still discovers routing structure, but it operates as a CORRECTION to the LLM's computation, not an independent trajectory.

### 3. Expected impact on diagnostics

```
Current:     D² = 268 → 56,832  (runaway)
With fixes:  D² = 268 → ~300-500 (bounded, slight growth from geometric correction)

Current:     B_within ≈ +0.1, B_across ≈ -0.3  (weak, near-zero)
With fixes:  B_within ≈ +0.3-0.5, B_across ≈ -0.1-+0.2  (structured, meaningful)

Current:     Entropy 0.35 → 0.64  (approaching uniform)
With fixes:  Entropy 0.35 → 0.40-0.50  (structured throughout)
```

## Why This Should Help the 5-Hop Chain

The 5-hop earthquake chain (earthquake → building collapse → ... → evacuation) fails because flat attention loses the connection over 5 hops. The geometric bias needs to create TRANSITIVE routing: if hop 1→2 is connected and 2→3 is connected, the heat kernel's diffusion should produce 1→3 connectivity after a few ODE steps.

With bounded D², the heat kernel produces non-degenerate attention at every layer. Early layers connect adjacent hops (1→2, 2→3). Middle layers connect 2-hop paths (1→3, 2→4) through diffusion. Late layers connect the full chain (1→5). This progressive deepening of connectivity IS what 36 ODE steps through depth should produce — but only if D² stays in the structured regime at every layer.

## Phase 2 (After Fix Validated)

Once norm anchoring + α=0.6 produces bounded D² and structured B values:

1. Rerun the causal chain test — does the 5-hop chain now succeed?
2. Run the parallel chains test (pesticide/wheat vs drought/rice) — does within/across separation prevent cross-contamination?
3. Measure per-layer B_across trajectory — does cross-event connectivity increase through depth (the progressive deepening prediction)?

## Longer Term: Train with Layer-Wise Criticality

The runtime fixes (norm anchoring, forcing) are engineering patches. The proper solution is training the MetricNet to produce depth-appropriate compression:

- Train on Qwen3-4B with layer-wise hooks active
- D²/4τ loss per layer (target ≈ 18 at every depth, or depth-varying schedule)
- The MetricNet learns: "at layer 5 with these residual stream features, compress by 0.3×; at layer 25 with these features, compress by 0.1×"
- After training, the MetricNet ITSELF prevents D² runaway without runtime patches

This requires a training loop where Qwen3-4B's forward pass includes the ODE hooks. Start with the runtime fixes to prove the architecture works, then move to end-to-end training.

## What We Learned

The layer-wise architecture IS the right design — the per-layer diagnostics show depth-dependent geometric evolution that single-hook can't capture. The B_within/B_across separation emerging through depth is structurally correct. The problem is purely quantitative: the amplification compounds too aggressively without criticality regulation.

The 30,000× D² growth is actually informative — it shows the MetricNet IS doing geometric work (not identity). It's just doing TOO MUCH. Damping the amplitude while preserving the direction is the right fix.
