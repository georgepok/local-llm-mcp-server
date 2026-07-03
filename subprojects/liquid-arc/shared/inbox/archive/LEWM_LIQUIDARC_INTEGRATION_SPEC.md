# LIQUIDARC × LeWorldModel INTEGRATION — Experiment Spec

## Motivation

LeWorldModel (LeWM) is a JEPA-based world model that learns latent dynamics from pixels using two loss terms: prediction loss + Gaussian regularizer (SIGReg). 15M parameters, single GPU, trains in hours. Released March 2026 by Maes, Le Lidec, Scieur, LeCun, Balestriero.

LiquidARC provides what LeWM's flat latent space lacks: learned Riemannian geometry. The integration hypothesis: replacing LeWM's Euclidean predictor with LiquidARC's ODE on a curved manifold will improve prediction quality, enable multi-scale temporal reasoning, and provide phase-transition-based novelty detection — while maintaining LeWM's training stability through the complementary collapse prevention of SIGReg + geometric criticality.

This is the first integration where LiquidARC DEFINES the geometry of a space designed for prediction, rather than perturbing an existing system's routing. LeWM's encoder produces the embeddings. LiquidARC's MetricNet produces the metric. The ODE evolves states on the curved manifold. No fighting with pretrained attention. No distributional mismatch. The geometry IS the computational medium.

## Background: How LeWM Works

```
Architecture:
  Encoder:    z_t = enc(o_t)           — CNN/ViT maps pixels to z ∈ ℝ^d (d=256 typical)
  Predictor:  ẑ_{t+1} = pred(z_t, a_t) — MLP predicts next embedding

Training:
  L = ||ẑ_{t+1} - z_{t+1}||²          — prediction loss (MSE in Euclidean space)
    + λ · SIGReg(z)                     — Gaussian regularizer (prevents collapse)

Planning (inference):
  MPC: optimize action sequence a_{1:T} to minimize cost in latent space
  Cost evaluated by rolling out predictor: z → ẑ₁ → ẑ₂ → ... → ẑ_T

Key properties:
  - No decoder (never reconstructs pixels)
  - No generative model (predicts embeddings, doesn't sample)
  - SIGReg uses Cramér-Wold theorem: test Gaussianity via 1D projections
  - Dropout 0.1 in predictor + BN projection after encoder = stability
  - ~15M params total (encoder ~10M, predictor ~5M)
```

Paper: arxiv.org/abs/2603.19312
Code: github.com/lucas-maes/le-wm
Checkpoints: available on Google Drive (linked from repo)

## Integration Architecture

### Design: LiquidARC as Curved Predictor

Replace LeWM's flat MLP predictor with LiquidARC's ODE on a learned Riemannian manifold. The encoder stays as-is — it maps pixels to latent embeddings. LiquidARC provides the dynamics of HOW those embeddings evolve.

```
BASELINE LeWM:
  o_t → Encoder → z_t → MLP_predictor(z_t, a_t) → ẑ_{t+1}
                         └── flat Euclidean dynamics ──┘

INTEGRATED LeWM + LiquidARC:
  o_t → Encoder → z_t → MetricNet(z_t) → g_t (Riemannian metric)
                       → ActionEmbed(a_t) → a_emb
                       → ODE(z_t, a_emb, g_t) → ẑ_{t+1}
                         └── curved Riemannian dynamics ──┘
```

### Component mapping

```
LeWM component          → Integrated system
────────────────────────────────────────────────────
Encoder (CNN/ViT)       → UNCHANGED — maps pixels to z_t ∈ ℝ^d
Predictor (MLP)         → REPLACED by LiquidARC ODE:
                            - MetricNet computes g from z_t
                            - TauNet computes τ from z_t
                            - Heat kernel routes between latent positions
                            - LTC contraction: dz/dt = -(1/τ)(z - target) + FFN
                            - 16 Euler steps (or fewer, tunable)
                            - Action conditioning via context embedding
SIGReg                  → KEPT — prevents distributional collapse
(new) Criticality loss  → ADDED — prevents geometric deflation
(new) Tau quality loss   → ADDED — maintains τ diversity
```

### Why the ODE replaces the MLP predictor

The MLP predictor computes `ẑ_{t+1} = MLP(z_t, a_t)` in one shot — a single function evaluation. This is a first-order approximation to dynamics.

The ODE computes `ẑ_{t+1}` through 16 iterative steps of heat-kernel-guided integration. Each step:
1. MetricNet evaluates the local geometry at the current state
2. Heat kernel determines how information flows between latent dimensions
3. LTC contraction moves the state toward the dynamically computed target
4. Action embedding biases the dynamics (which direction to evolve)

This provides:
- **Iterative refinement**: 16 chances to correct the prediction (vs 1 for MLP)
- **Geometry-aware dynamics**: the metric tells the predictor which dimensions matter for THIS state
- **Adaptive integration speed**: τ varies per latent dimension — some evolve fast, some slow
- **Phase transition capability**: the MetricNet can reorganize routing when the state enters a new dynamical regime

### Action conditioning

LeWM's predictor takes `(z_t, a_t)` as input. For the ODE, action enters as context:

```python
class ActionConditionedODE(ContinuousDynamics):
    def __init__(self, config, action_dim):
        super().__init__(config)
        self.action_embed = nn.Linear(action_dim, config.d_model)
    
    def forward(self, t, h):
        # h includes latent state z_t
        # self._context includes embedded action
        # MetricNet sees both z_t and action context
        # Heat kernel routes z_t dimensions conditioned on action
        
        # Standard ContinuousDynamics forward, but context includes action
        return super().forward(t, h)
    
    def set_action(self, a_t):
        a_emb = self.action_embed(a_t)
        self.set_context(a_emb)
```

The action embedding enters through `set_context()`, which the MetricNet and TauNet already use for context-dependent metric computation. The MetricNet produces DIFFERENT geometry for different actions — "pushing" creates a metric that weights spatial dimensions highly, "rotating" creates a metric that weights angular dimensions.

### Latent space dimensionality

LeWM uses d=256 for the latent space (typical for their environments). LiquidARC's MetricNet operates per-dimension with diagonal approximation. At d=256:

```
MetricNet: Linear(d + context_d, hidden) → Linear(hidden, d)  — ~130K params
TauNet:    Linear(d + context_d, hidden) → Linear(hidden, 1)  — ~33K params  
FFN:       Linear(d, 4d) → Linear(4d, d)                      — ~525K params
W_v, W_o:  Linear(d, d) each                                   — ~131K params
Total ODE dynamics: ~820K params

vs LeWM MLP predictor: ~5M params (2-3 layer MLP, d=256)
```

The ODE dynamics are actually SMALLER than the MLP predictor they replace. The capacity comes from the 16 iterative steps, not from parameter count.

## Training

### Loss function

```python
# Combined loss: LeWM prediction + SIGReg + LiquidARC criticality
def compute_loss(encoder, ode, o_t, o_t1, a_t, lambda_sig, mu_crit, gamma_tau):
    # Encode observations
    z_t = encoder(o_t)
    z_t1 = encoder(o_t1)  # target (stop gradient for predictor learning)
    
    # Predict next state via ODE on curved manifold
    ode.set_action(a_t)
    z_hat_t1 = ode_integrate(ode, z_t.unsqueeze(1), n_steps=16).squeeze(1)
    
    # 1. Prediction loss (MSE — or metric-weighted MSE, see variant below)
    pred_loss = F.mse_loss(z_hat_t1, z_t1.detach())
    
    # 2. SIGReg (from LeWM — prevents distributional collapse)
    sigreg_loss = compute_sigreg(z_t)
    
    # 3. Criticality loss (from LiquidARC — prevents geometric deflation)
    crit_loss = compute_criticality_loss(ode.dynamics)  # D²/4τ → target
    
    # 4. Tau quality loss (from LiquidARC — maintains τ diversity)
    tau_loss = compute_tau_quality_loss(ode.dynamics)
    
    total = pred_loss + lambda_sig * sigreg_loss + mu_crit * crit_loss + gamma_tau * tau_loss
    return total, {
        'pred': pred_loss.item(),
        'sigreg': sigreg_loss.item(), 
        'crit': crit_loss.item(),
        'tau': tau_loss.item()
    }
```

### Variant: Metric-weighted prediction loss

Instead of Euclidean MSE, compute prediction error in the learned metric:

```python
# Standard: ||ẑ - z||² = Σ (ẑ_i - z_i)²
pred_loss_euclidean = F.mse_loss(z_hat_t1, z_t1.detach())

# Metric-weighted: ||ẑ - z||²_g = Σ g_i · (ẑ_i - z_i)²
g = ode.dynamics.get_current_metric()  # diagonal metric [d]
diff = z_hat_t1 - z_t1.detach()
pred_loss_metric = (g * diff * diff).mean()
```

The metric-weighted loss tells the encoder: "errors in metrically important dimensions matter more." The MetricNet learns which dimensions carry dynamically important information, and the prediction loss concentrates on those dimensions. This creates a feedback loop:

```
MetricNet weights dimensions → prediction loss focuses on those dimensions
→ encoder learns to encode dynamically important features in those dimensions
→ MetricNet discovers these are indeed the important dimensions → reinforces
```

Start with Euclidean MSE (simpler, matches LeWM baseline). Switch to metric-weighted as an ablation once the baseline is established.

### Training protocol

```yaml
# Phase 1: Reproduce LeWM baseline (verify our setup works)
phase1:
  model: original LeWM (MLP predictor)
  environments: [pusht, blockpush]  # start with 2D tasks
  epochs: as per LeWM paper
  purpose: establish baseline scores on our hardware

# Phase 2: Replace predictor with ODE (no criticality yet)
phase2:
  model: LeWM encoder + LiquidARC ODE predictor
  loss: pred_loss + SIGReg only (no criticality)
  ode_steps: 16
  action_conditioning: via set_context
  purpose: verify ODE predictor trains stably with SIGReg

# Phase 3: Add criticality scaffolding
phase3:
  model: same as phase 2
  loss: pred_loss + SIGReg + criticality + tau_quality
  criticality_target: start with 18 (from ARC), tune if needed
  purpose: determine if geometric criticality helps prediction

# Phase 4: Full evaluation
phase4:
  model: best from phase 2/3
  evaluation: 
    - control performance (success rate on tasks)
    - prediction quality (MSE at different horizons)
    - physical understanding (violation of expectation tests)
    - latent structure (probing of physical quantities)
    - planning speed (MPC iterations to convergence)
  compare against: LeWM baseline, other world models from paper
```

### Optimizer configuration

```yaml
optimizer: Adam
groups:
  - params: encoder        lr: 3e-4   # match LeWM paper
  - params: ode.dynamics   lr: 1e-4   # MetricNet, TauNet — slower (from ARC experience)
  - params: action_embed   lr: 3e-4   # match encoder
  
scheduler: cosine decay (match LeWM paper)
batch_size: 256 (match LeWM paper)
gradient_clip: 1.0
```

### MetricNet initialization

Option A: Random init (recommended for clean experiment)
- The MetricNet has never seen visual latent embeddings
- ARC checkpoint won't help (different domain, different d)
- Let it learn from the dynamics loss

Option B: Identity init
- Initialize MetricNet to produce g = 1 (flat metric)
- The ODE starts by reproducing flat Euclidean dynamics
- Gradually develops curvature as training progresses
- Smoother training curve, less risk of early instability

Start with Option B — it ensures the integrated system is AT LEAST as good as flat-space prediction at initialization, then can only improve as it develops curvature.

## Environments

Use LeWM's evaluation environments (available in their codebase):

### Tier 1: 2D control (simplest)
- **PushT**: Push a T-shaped block to a target pose
- **BlockPush**: Push a block through a maze
- These have simple physics, low-dimensional state, and established baselines

### Tier 2: 3D control (harder)
- LeWM evaluates on additional 3D tasks (specifics in their paper)
- Attempt only after Tier 1 succeeds

### Why these environments matter for LiquidARC

PushT involves:
- Contact dynamics (discontinuous — object is or isn't being pushed)
- Spatial reasoning (where is the block relative to the target)
- Planning horizon (multiple steps to reach goal)

The contact discontinuity is a CURVATURE SINGULARITY on the manifold — the point where "not touching" transitions to "pushing" should have high metric curvature. The MetricNet should learn this: high g values at the contact boundary, creating a geometric barrier that the predictor must explicitly "cross" when predicting contact events.

This is directly analogous to the phase transition: the contact event IS a structural transition in the dynamics. The MetricNet at criticality would have maximum sensitivity to this transition.

## Evaluation Metrics

### Primary: Control performance
- Success rate on PushT (percentage of episodes reaching target)
- Score on each LeWM benchmark task
- Compare: LeWM baseline vs LeWM + LiquidARC

### Secondary: Prediction quality
- MSE at horizon 1, 5, 10, 20 steps
- Does the ODE predictor degrade more slowly with horizon than MLP?
- The iterative ODE should accumulate less error per step than single-shot MLP

### Tertiary: Geometric structure
- CV of the learned metric across latent dimensions
- D²/4τ trajectory during training (does criticality emerge or need to be imposed?)
- τ distribution per latent dimension (does TauNet differentiate dimensions?)
- Metric curvature at contact boundaries (does high κ emerge at discontinuities?)

### Quaternary: Physical understanding
- Violation of expectation tests (LeWM's protocol)
- Does the curved-space model assign HIGHER surprise to physically implausible events?
- The geometric interpretation: implausible events require crossing high-curvature barriers, producing high prediction error in metric space

### Planning efficiency
- MPC iterations to convergence
- On a curved manifold, geodesic planning should find shorter paths
- Does the metric accelerate planning convergence?

## Success Criteria

1. **Training stability**: LeWM + LiquidARC trains end-to-end without collapse (SIGReg + criticality together prevent both distributional and geometric collapse)

2. **Prediction quality**: ODE predictor matches or exceeds MLP predictor on 1-step MSE (it has 16 iterative steps vs 1 function evaluation — should be at least equal)

3. **Long-horizon prediction**: ODE predictor shows slower error accumulation at horizon 10-20 than MLP predictor (the geometric structure should improve multi-step rollouts)

4. **Control performance**: LeWM + LiquidARC matches or exceeds LeWM baseline on PushT success rate

5. **Geometric structure emerges**: CV > 2.0 on the latent space metric (the MetricNet learns non-trivial geometry from the dynamics)

6. **Phase transition at contact boundaries**: Metric curvature is measurably higher at contact transitions than at smooth dynamics (the geometry captures physics discontinuities)

## Implementation Plan

### Step 1: Set up LeWM baseline (1-2 days)

```bash
# Clone LeWM repo
git clone https://github.com/lucas-maes/le-wm
# Install dependencies
pip install -r requirements.txt
# Train baseline on PushT
python train.py --config-name=pusht.yaml
# Evaluate
python eval.py --config-name=pusht.yaml policy=pusht/lewm
```

Verify we can reproduce their reported numbers on Spark.

### Step 2: Create LiquidARC predictor module (2-3 days)

```python
class LiquidARCPredictor(nn.Module):
    """Drop-in replacement for LeWM's MLP predictor.
    
    Interface matches LeWM: takes (z_t, a_t), returns ẑ_{t+1}.
    Internally uses ODE integration on curved manifold.
    """
    def __init__(self, latent_dim, action_dim, config):
        super().__init__()
        self.action_embed = nn.Linear(action_dim, latent_dim)
        
        # LiquidARC dynamics (reuse existing ContinuousDynamics)
        self.dynamics = ContinuousDynamics(config)
        
        # Projection: LeWM latent → ODE space (if dimensions differ)
        self.proj_in = nn.Linear(latent_dim, config.d_model) if latent_dim != config.d_model else nn.Identity()
        self.proj_out = nn.Linear(config.d_model, latent_dim) if latent_dim != config.d_model else nn.Identity()
        
        self.n_steps = config.n_steps  # 16
        self.dropout = nn.Dropout(0.1)  # match LeWM predictor dropout
        
    def forward(self, z_t, a_t):
        """Predict next latent state.
        
        Args:
            z_t: [B, d] current latent state
            a_t: [B, a] action
        Returns:
            z_hat: [B, d] predicted next latent state
        """
        # Embed action as context for MetricNet
        a_emb = self.action_embed(a_t)
        self.dynamics.set_context(a_emb)
        
        # Project to ODE space
        h = self.proj_in(z_t).unsqueeze(1)  # [B, 1, d_model]
        
        # ODE integration (16 Euler steps on curved manifold)
        dt = 1.0 / self.n_steps
        for step in range(self.n_steps):
            self.dynamics.set_step_index(step, self.n_steps)
            dh = self.dynamics(t=step * dt, h=h)
            h = h + dh * dt
        
        # Project back to latent space
        z_hat = self.proj_out(h.squeeze(1))
        z_hat = self.dropout(z_hat)
        
        return z_hat
    
    def get_metrics(self):
        """Return geometric diagnostics for logging."""
        return {
            'cv': self.dynamics.last_cv,
            'd_sq_over_4tau': self.dynamics.last_criticality_ratio,
            'tau_mean': self.dynamics.last_tau_mean,
        }
```

Key design: the module is a DROP-IN replacement for LeWM's predictor. Same input/output interface. The training loop doesn't change — just swap the predictor and add criticality terms to the loss.

### Step 3: Integrate with LeWM training loop (1-2 days)

Modify LeWM's training script to:
1. Accept `predictor_type: [mlp, liquidarc]` config option
2. When `liquidarc`: instantiate LiquidARCPredictor instead of MLP
3. Add criticality_loss and tau_quality_loss to the total loss
4. Log geometric diagnostics (CV, D²/4τ, τ distribution) alongside LeWM metrics

Minimal changes to LeWM codebase — the integration should be a CONFIG OPTION, not a fork.

### Step 4: Train and evaluate Phase 2 (2-3 days)

Train LeWM + LiquidARC on PushT without criticality loss.
Monitor: prediction MSE, SIGReg, CV, training stability.
Compare 1-step and multi-step prediction against MLP baseline.

If stable → proceed to Phase 3.
If unstable → diagnose (likely MetricNet amplification, addressed by identity init).

### Step 5: Train and evaluate Phase 3 (2-3 days)

Add criticality scaffolding. Sweep criticality target if needed.
Full evaluation suite: control, prediction, physical understanding, geometry.

### Step 6: Ablation studies (2-3 days)

```
A: LeWM baseline (MLP predictor)
B: LeWM + ODE predictor, no criticality (just SIGReg)
C: LeWM + ODE predictor + criticality (SIGReg + geometric)
D: LeWM + ODE predictor + metric-weighted loss + criticality

Compare A vs B: does ODE integration help?
Compare B vs C: does criticality help?
Compare C vs D: does metric-weighted prediction help?
```

## What Transfers From Current Work

Everything from the ODE/geometry side:
- ContinuousDynamics (MetricNet, TauNet, heat kernel SDPA, FFN, LTC)
- Sustained criticality system (D²/4τ loss, tau_quality, convergence coupling)
- Euler integration with norm homeostasis
- Step embeddings for depth-dependent routing
- All diagnostic infrastructure (CV, D², entropy, tau distribution)

What's new:
- ActionConditionedODE (action embedding into context)
- LiquidARCPredictor wrapper (drop-in for LeWM MLP)
- Metric-weighted prediction loss (optional variant)
- Integration with LeWM's encoder, SIGReg, training loop, evaluation suite
- Visual environment setup (PushT, BlockPush)

## Why This Should Succeed Where Transformer Integration Didn't

| Transformer integration | LeWM integration |
|---|---|
| Perturbation: 60M modifying 4B | Replacement: ODE IS the predictor |
| Pretrained routing competed | No pretrained predictor to fight |
| Distribution mismatch (text embeds) | Native latent space (encoder designed for prediction) |
| Collapse prevention absent | Dual prevention: SIGReg + criticality |
| No pixel grounding | Encoder provides pixel → latent mapping |
| Single forward pass, one chance | 16 ODE steps, iterative refinement |
| Task: improve NTP (indirect) | Task: predict dynamics (direct, exactly what ODE does) |
| The LLM already handled the task | The MLP predictor is a WEAK baseline to beat |

The last point is crucial. LeWM's MLP predictor is a 2-3 layer network making ONE prediction. The ODE makes the same prediction through 16 iterative steps with geometry-aware routing. On prediction quality alone, the ODE should match or exceed the MLP because it has more computational depth for the same parameter budget.

## Timeline

- Step 1 (baseline): 1-2 days
- Step 2 (module): 2-3 days
- Step 3 (integration): 1-2 days
- Step 4 (Phase 2 train): 2-3 days
- Step 5 (Phase 3 train): 2-3 days
- Step 6 (ablations): 2-3 days

Total: ~2-3 weeks

## Compute Requirements

LeWM trains on single GPU in hours. LiquidARC ODE adds ~16× the predictor FLOPS (16 steps vs 1), but the predictor is small relative to the encoder. Estimated overhead: 2-3× total training time vs LeWM baseline. Well within Spark's capacity.

Memory: LeWM ~2GB + LiquidARC ODE ~0.5GB = ~2.5GB. Spark has 128GB. No constraints.

## The Deeper Hypothesis

If the ODE predictor with learned Riemannian metric outperforms the MLP predictor at WORLD MODELING — predicting physical dynamics from latent states — this validates the core FGN claim: curved geometry is the correct computational substrate for modeling processes that have intrinsic structure (physics, causality, temporal dynamics).

The flat MLP predictor treats all latent dimensions equally. The curved ODE predictor treats dimensions according to their geometric importance — which dimensions carry dynamic information, which are static, where the discontinuities are. Physics is NOT isotropic — it has preferred directions, conservation laws, contact boundaries. The Riemannian metric is the mathematical object that captures this anisotropy.

This would be the first demonstration that learned Riemannian geometry improves prediction on a task where the geometry has PHYSICAL MEANING — not abstract token routing (which transformers already handle), but actual physical dynamics (which is what world models need and where flat prediction is provably suboptimal).
