# LAYER-WISE FIX: Pure Perturbation Architecture

## Why Norm Anchoring Failed

D² runaway got WORSE (1.6→164,864 vs 1.9→56,832) because:
- Norm anchoring operates in raw space: clips ‖h‖
- D² operates in metric space: D² = Σ(h_i - h_j)² · g
- MetricNet's g values amplify D² independently of h norms
- Clipping h norms does nothing to g-weighted distances

No fix in raw space can solve a metric-space problem.

## The Fix: ODE as Perturbation Engine

The ODE state should NOT be an independent trajectory. It should be:

```
h_ode = h_residual + ε · correction
```

Where:
- h_residual: the LLM's actual hidden state at this layer (well-behaved, bounded)
- correction: what the ODE's geometric routing wants to CHANGE
- ε: coupling strength (0.05-0.2)

### Why this prevents D² runaway

```
D²(h_ode_i, h_ode_j) = D²(h_res_i + ε·δ_i, h_res_j + ε·δ_j)
                      ≈ D²(h_res_i, h_res_j) + O(ε) terms
```

The base D² comes from the residual stream — which the LLM controls and is well-bounded. The MetricNet amplifies the O(ε) terms, not the base. Even with amp=10× over 36 layers, the correction terms stay small relative to the residual base.

### Implementation

```python
class PerturbationODE:
    def __init__(self, dynamics, n_layers, epsilon=0.1):
        self.dynamics = dynamics
        self.n_layers = n_layers
        self.epsilon = epsilon
        self.correction = None  # accumulated geometric correction
    
    def start_forward(self):
        self.correction = None
    
    def process_layer(self, layer_idx, h_residual):
        B, N, d = h_residual.shape
        
        if self.correction is None:
            self.correction = torch.zeros_like(h_residual)
        
        # Build ODE input: residual + scaled correction
        h_ode = h_residual + self.epsilon * self.correction
        
        # One ODE step (MetricNet + heat kernel + LTC + FFN)
        self.dynamics.set_step_index(layer_idx, self.n_layers)
        self.dynamics.set_context(h_residual.mean(dim=1))
        dh = self.dynamics(t=layer_idx, h=h_ode)
        
        # Update CORRECTION only (not full state)
        dt = 1.0 / self.n_layers
        self.correction = self.correction + dh * dt
        
        # Compute bias from the corrected representation
        h_corrected = h_residual + self.epsilon * self.correction
        B = compute_state_cosine_bias(h_corrected)
        
        return B
```

### What ε controls

- ε = 0.0: ODE has zero effect. Pure LLM. Baseline.
- ε = 0.05: Subtle geometric nudge. Safe starting point.
- ε = 0.1: Moderate correction. Expected working range.
- ε = 0.2: Strong correction. May need per-layer tuning.
- ε = 1.0: Full ODE state. D² runaway. What we had.

Start at ε=0.1 and tune. The causal chain test (5-hop earthquake) is the benchmark — does geometric correction help where plain LLM fails?

### Key property: correction accumulates but D² doesn't blow up

```
Layer 1:  correction₁ = dh₁ · dt
Layer 2:  correction₂ = correction₁ + dh₂ · dt  (carries layer 1's geometric info)
Layer 3:  correction₃ = correction₂ + dh₃ · dt  (carries layers 1+2)
...
Layer 36: correction₃₆ = Σ all geometric corrections through depth
```

The correction grows additively (O(36 · dt · ‖dh‖)). But D² is computed from h_residual + ε·correction, where h_residual dominates. The MetricNet sees the COMBINED representation and computes routing — but the distances are anchored by the residual, not by the (potentially amplified) correction.

### What the bias captures

At each layer, the bias B reflects:
- Base structure: how the LLM already relates tokens (from h_residual)
- Geometric correction: what the ODE's routing adds (from ε·correction)

The bias is a BLEND of the LLM's native similarity and the ODE's learned geometry. At ε=0.1, it's 90% LLM + 10% ODE. The 10% is what carries causal chain structure, event boundaries, and cross-event connections that flat attention misses.

### Diagnostics to track

Per layer, report:
```
D²_residual:    D² from h_residual alone (should be stable through depth)
D²_corrected:   D² from h_residual + ε·correction (should grow SLOWLY)
correction_norm: ‖correction‖ / ‖h_residual‖ (should stay < 0.5)
B_range:         max(B) - min(B) (structured routing signal)
```

If D²_corrected grows faster than 10× through depth → ε is too large.
If correction_norm > 0.5 → ODE is trying to dominate the residual. Reduce ε.
If B_range < 0.1 → ODE isn't contributing. Increase ε.

### Connection to sustained criticality

The criticality system (D²/4τ target, tau_quality, convergence coupling) still applies — but now operating on the CORRECTED state, not on an independent ODE trajectory. The D²/4τ that matters is from the corrected representation. With the residual anchoring, this value should be much more stable through depth.

A post-transition checkpoint (CV≈7) would produce corrections that COMPRESS distances where the LLM leaves tokens too far apart (causal chain endpoints). The perturbation approach makes the MetricNet's compression/amplification operate on the RIGHT scale — adjusting the LLM's own distances rather than computing independent ones.
