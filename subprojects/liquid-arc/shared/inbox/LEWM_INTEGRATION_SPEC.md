# LIQUIDARC × LeWorldModel — Integration Spec

## Motivation

LeWorldModel (LeWM) is a JEPA world model: encoder maps pixels to latent embeddings, predictor predicts next embedding given current state + action. 15M params, single GPU, two loss terms (prediction MSE + SIGReg Gaussian regularizer). Prevents representation collapse without hacks (no EMA, no stop-gradients, no pretrained encoders). Released March 2026 by Maes, Le Lidec, Scieur, LeCun, Balestriero.

LiquidARC provides what LeWM lacks: learned Riemannian geometry in latent space. The integration: replace LeWM's flat MLP predictor with LiquidARC's ODE on a curved manifold. The encoder stays unchanged. The geometry IS the dynamics.

Paper: arxiv.org/abs/2603.19312  
Code: github.com/lucas-maes/le-wm

## Architecture

```
BASELINE LeWM:
  pixels → Encoder → z_t → MLP(z_t, a_t) → ẑ_{t+1}
                            flat Euclidean, 1 function eval, ~5M params

INTEGRATED:
  pixels → Encoder → z_t → MetricNet(z_t) → g_t
                          → ActionEmbed(a_t) → context
                          → ODE(z_t, context, g_t, 16 steps) → ẑ_{t+1}
                            curved Riemannian, 16 iterative steps, ~1M params
```

The ODE predictor is a DROP-IN replacement. Same interface: `(z_t, a_t) → ẑ_{t+1}`. Different internals.

### LiquidARCPredictor module

```python
class LiquidARCPredictor(nn.Module):
    """Drop-in replacement for LeWM MLP predictor."""
    
    def __init__(self, latent_dim, action_dim, ode_config):
        super().__init__()
        self.action_embed = nn.Linear(action_dim, latent_dim)
        self.dynamics = ContinuousDynamics(ode_config)  # reuse existing
        self.proj_in = nn.Linear(latent_dim, ode_config.d_model)
        self.proj_out = nn.Linear(ode_config.d_model, latent_dim)
        self.n_steps = ode_config.n_steps  # 16
        self.dropout = nn.Dropout(0.1)  # match LeWM predictor
    
    def forward(self, z_t, a_t):
        # Action as context for MetricNet
        self.dynamics.set_context(self.action_embed(a_t))
        
        # Project to ODE space, integrate
        h = self.proj_in(z_t).unsqueeze(1)  # [B, 1, d_model]
        dt = 1.0 / self.n_steps
        for step in range(self.n_steps):
            self.dynamics.set_step_index(step, self.n_steps)
            dh = self.dynamics(t=step * dt, h=h)
            h = h + dh * dt
        
        return self.dropout(self.proj_out(h.squeeze(1)))
```

### Action conditioning

Action enters via `set_context()` — the MetricNet and TauNet already accept context. Different actions produce different metrics: "push left" creates geometry that weights x-position dimensions; "rotate" weights angular dimensions. The geometry ADAPTS to the action.

### Why ODE replaces MLP

The MLP makes ONE prediction (single function evaluation). The ODE makes the same prediction through 16 iterative steps with geometry-aware routing at each step. More computational depth for fewer parameters. The metric tells the predictor which latent dimensions matter for THIS state and THIS action.

## Training

### Loss function

```python
total_loss = pred_loss                      # ||ẑ_{t+1} - z_{t+1}||²
           + lambda_sig * sigreg_loss       # Gaussian regularizer (from LeWM)
           + mu_crit * criticality_loss     # D²/4τ → target (from LiquidARC)
           + gamma_tau * tau_quality_loss    # τ diversity (from LiquidARC)
```

SIGReg prevents distributional collapse (embeddings collapsing to a point).
Criticality prevents geometric deflation (metric collapsing to flat).
These are complementary — dual protection against the two modes of collapse.

### Variant: Metric-weighted prediction loss

```python
# Standard: Euclidean MSE
pred_loss = ((z_hat - z_target) ** 2).mean()

# Metric-weighted: errors in metrically important dimensions matter more
g = dynamics.get_current_metric()  # [d] diagonal metric
pred_loss = (g * (z_hat - z_target) ** 2).mean()
```

Start with Euclidean (matches baseline). Ablate metric-weighted later.

### Optimizer

```yaml
groups:
  - params: encoder          lr: 3e-4   # match LeWM paper
  - params: ode.dynamics     lr: 1e-4   # MetricNet/TauNet slower
  - params: action_embed     lr: 3e-4
scheduler: cosine decay
batch_size: 256
gradient_clip: 1.0
```

### MetricNet initialization

Initialize to produce g ≈ 1 (identity / flat metric). This ensures the system starts equivalent to flat Euclidean dynamics. Curvature develops during training. No risk of early instability from random metric.

## Environments

### Tier 1 (start here)
- **PushT**: push T-block to target. Contact dynamics = curvature singularity.
- **BlockPush**: push block through maze. Path planning = geodesic finding.

### Tier 2 (if Tier 1 succeeds)
- 3D tasks from LeWM paper

PushT is ideal because contact boundaries are physical discontinuities that should produce high metric curvature. The MetricNet should learn: latent dimensions near contact = high g (important, handle carefully). Latent dimensions during free motion = low g (simple, predict easily).

## Evaluation

### Primary: Control performance
- PushT success rate: LeWM + LiquidARC vs LeWM baseline
- Must match or exceed baseline to validate the integration

### Secondary: Prediction quality
- MSE at horizon 1, 5, 10, 20 steps
- The ODE should accumulate less error over long horizons (16 iterative steps vs 1-shot MLP)
- This is the key differentiator — multi-step rollouts for MPC planning

### Tertiary: Geometric structure
- CV of learned metric (does non-trivial geometry emerge?)
- D²/4τ trajectory during training
- τ distribution per latent dimension
- Metric values at contact vs free-motion states (curvature at boundaries?)

### Quaternary: Physical understanding  
- Violation of expectation tests (from LeWM protocol)
- Surprise = prediction error in metric space
- The curved model should assign HIGHER surprise to implausible events (crossing geometric barriers)

## Phased Experiments

### Phase 1: Reproduce LeWM baseline
Train original LeWM on PushT. Verify we match their numbers on Spark.

### Phase 2: ODE predictor without criticality
Replace MLP with LiquidARCPredictor. Train with pred_loss + SIGReg only.
Monitor: does it train stably? Does prediction match baseline?

### Phase 3: Add criticality scaffolding
Add criticality_loss + tau_quality_loss.
Monitor: does CV emerge? Does criticality improve prediction?

### Phase 4: Full evaluation + ablations
```
A: LeWM baseline (MLP predictor)
B: LeWM + ODE, no criticality
C: LeWM + ODE + criticality
D: LeWM + ODE + criticality + metric-weighted loss

A vs B: ODE integration value
B vs C: criticality value
C vs D: metric-weighted loss value
```

## Success Criteria

1. Training stability — no collapse with SIGReg + criticality together
2. 1-step prediction — ODE matches or beats MLP on MSE
3. Long-horizon prediction — ODE shows slower error growth at horizon 10-20
4. Control performance — matches or exceeds LeWM on PushT
5. Geometric structure — CV > 2.0 (non-trivial metric learned)
6. Contact curvature — metric values higher at contact boundaries than free motion

## Why This Should Work

| Previous integration (LLM) | This integration (LeWM) |
|---|---|
| Perturbation to 4B pretrained model | Replacement of 5M MLP predictor |
| Fought pretrained attention routing | No pretrained predictor to fight |
| Text embedding distribution mismatch | Native latent space designed for prediction |
| Task: improve NTP (indirect) | Task: predict dynamics (exactly what ODE does) |
| One forward pass per prediction | 16 ODE steps per prediction |
| LLM already handled the task well | MLP predictor is a weak baseline to beat |

The MLP predictor is a WEAK component. It's a 2-3 layer network making a single-shot prediction. The ODE with 16 iterative steps, geometry-aware routing, and adaptive τ has strictly more computational power. On a prediction task — not attention routing, not NTP, but explicit dynamics prediction — the ODE should be at home.

## Compute

LeWM trains on 1 GPU in hours. ODE adds ~16× predictor FLOPS (16 steps vs 1) but predictor is small vs encoder. Estimated: 2-3× total training time. Well within Spark capacity. Memory: ~3GB total.

## Timeline

- Phase 1 (baseline): 1-2 days
- Module creation: 2-3 days  
- Phase 2 (ODE train): 2-3 days
- Phase 3 (criticality): 2-3 days
- Phase 4 (ablations): 2-3 days
- Total: ~2-3 weeks

## What Transfers

From LiquidARC: ContinuousDynamics, MetricNet, TauNet, heat kernel SDPA, criticality scaffolding, tau quality loss, convergence coupling, step embeddings, all diagnostic infrastructure.

New: ActionConditionedODE, LiquidARCPredictor wrapper, metric-weighted prediction loss, LeWM codebase integration, visual environment setup.

## The Hypothesis

Physics is not isotropic. It has preferred directions, conservation laws, contact discontinuities, symmetries. A flat Euclidean predictor treats all latent dimensions equally. A Riemannian predictor treats dimensions according to their geometric importance — learned from the dynamics. If any prediction task benefits from geometric structure, it's PHYSICAL DYNAMICS PREDICTION where the geometry has physical meaning.

This is the first integration where LiquidARC's geometry serves the EXACT PURPOSE it was designed for: defining the metric structure of a space where dynamical processes evolve. Not token routing. Not attention perturbation. Actual dynamics on a curved manifold. The mathematics of FGN v3 applied to the domain it describes.
