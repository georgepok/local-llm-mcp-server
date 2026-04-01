# LiquidARC: Continuous-Time Geometric Computation via Riemannian ODE Dynamics

## Abstract

We present LiquidARC, a neural architecture that replaces discrete transformer layers with a single weight-tied dynamics module integrated as a continuous-time ordinary differential equation (ODE). The model learns a Riemannian metric field over token positions and routes information through heat kernel diffusion on the resulting manifold. A Liquid Time-Constant (LTC) contraction mechanism guarantees stability and enables adaptive per-position computation rates. The entire forward pass consists of embedding, context pooling, and 16 Euler integration steps of the same 356K-parameter dynamics module — yielding a 572K-parameter model whose computation depth emerges from iteration rather than parameter replication. We evaluate on ARC-AGI, a benchmark requiring few-shot visual reasoning over discrete grid transformations. LiquidARC achieves 61.1% transform accuracy on held-out ARC tasks and demonstrates a clean decomposition of geometric routing (WHERE information flows) from content transformation (WHAT operation is applied) — enabling test-time adaptation via 100 gradient steps on a single task to reach 44.4% on novel transforms. We present detailed analysis of training dynamics including a CV-driven phase transition, scaling behavior from 572K to 5M parameters, and evidence that the generalization bottleneck lies in training data distribution rather than model capacity.

---

## 1. Introduction

Standard transformer architectures process sequences through a fixed stack of discrete layers, each with independent parameters. This design has two fundamental properties: (1) the number of "reasoning steps" equals the number of layers, and (2) the model's capacity scales with depth times width. For tasks requiring variable-depth reasoning — such as ARC-AGI, where some transformations are simple color swaps and others involve multi-step spatial compositions — fixed depth is a structural mismatch.

We propose an alternative: a single dynamics module applied continuously via ODE integration. Instead of N layers with N×P parameters, we have one module with P parameters applied for a continuous duration. The model's effective depth is determined by the integration time and step count, which can be varied without changing the parameter count. The same weights are reused at every step — emergence comes from iteration, not from stacking.

The core innovation is how information routes between positions. Rather than learned query-key attention matrices, routing is determined by a Riemannian metric field that the model learns over token positions. The metric defines distances on a curved manifold; a heat kernel computed from these distances determines how much information flows between any pair of positions. Positions that the metric places "close together" exchange information readily; positions that are "far apart" on the manifold are informationally isolated.

This paper presents the architecture, its mathematical foundations, training methodology, and experimental results on ARC-AGI. We focus on three contributions:

1. **SDPA-factored heat kernel**: We show that the Riemannian heat kernel `K = softmax(-D²/(4t))` can be algebraically reformulated as scaled dot-product attention (SDPA), enabling FlashAttention acceleration without ever materializing the N×N distance matrix to HBM.

2. **WHERE/WHAT decomposition**: The architecture naturally separates geometric routing (controlled by the metric network) from content transformation (controlled by the output projection W_o). We demonstrate this decomposition quantitatively through test-time training ablations.

3. **CV-driven phase transition**: Training exhibits a sharp phase transition when the metric's coefficient of variation reaches a critical threshold (~6-7), breaking a copy-everything equilibrium. We characterize this transition and show it is scale-invariant.

---

## 2. Architecture

### 2.1 Overview

LiquidARC processes ARC tasks as flat token sequences. Each cell in each grid becomes a token with additive embeddings for color, (x,y) position, grid role, and grid identity. The model architecture is:

```
Input → ARCEmbedding → ContextPool → euler_solve(ContinuousDynamics, h₀, 16 steps) → OutputHead
```

The entire model has four components:

| Component | Parameters | Function |
|-----------|-----------|----------|
| ARCEmbedding | ~213K | Token → hidden state via additive embeddings |
| ContextPool | ~5.8K | Attention-weighted pooling → episode context vector |
| ContinuousDynamics | ~356K | Shared ODE dynamics (MetricNet + HeatKernel + LTC + FFN) |
| OutputHead | ~2.6K | Hidden state → color logits |

**Total: 572,238 parameters** at d_model=256.

### 2.2 Embedding

Each ARC grid cell is embedded as:

$$h = \text{ColorEmbed}(c) + \text{PosX}(x) + \text{PosY}(y) + \text{RoleEmbed}(r) + \text{GridIdEmbed}(g) + \text{SepEmbed}(s) \cdot \mathbf{1}_{sep}$$

where:
- `ColorEmbed`: 11 colors × d (10 ARC colors + 1 padding token)
- `PosX`, `PosY`: 31 positions × d (grids up to 30×30)
- `RoleEmbed`: 4 roles × d (input_demo, output_demo, test_input, test_output), with test_output sharing test_input's embedding to enable spatial transfer
- `GridIdEmbed`: 16 grids × d (identifies which demonstration pair a token belongs to)
- `SepEmbed`: 4 separator types × d (grid boundary markers)

The embedding is followed by LayerNorm and dropout. Notably, we do **not** use sequential positional embeddings — ARC grids are inherently 2D, and a 1D sequential position would impose harmful ordering bias. The (x,y) embeddings and grid identity provide all necessary structural information.

### 2.3 Context Pooling

Before ODE integration, a context vector is computed from the input tokens via attention-weighted pooling:

$$\alpha = \text{softmax}(\text{MLP}_{d \to d/4 \to 1}(h_0))$$
$$\text{context} = W_{out} \sum_{i \in \text{context}} \alpha_i \cdot h_{0,i}$$

The context vector conditions the metric network at every ODE step, allowing the geometry to depend on the full input episode. The attention scoring network uses a tanh bottleneck (`d → d/4 → Tanh → d/4 → 1`) to prevent attention collapse.

### 2.4 ContinuousDynamics: The Core Module

The dynamics module computes dh/dt at each ODE step. It is called 16 times with the same weights. The computation within each step follows this sequence:

#### Step 1: Metric Computation

The Riemannian metric field g(h) assigns a positive-definite diagonal metric tensor to each position:

$$g = \text{Softplus}(W_2 \cdot \text{GELU}(W_1 \cdot [\text{LN}(h) \| \text{ctx}]))$$

where `[· ‖ ·]` denotes concatenation, LN is LayerNorm, and ctx is the episode context vector broadcast to all positions. The metric network maps from 2d to d_metric (bottleneck) to d, with Softplus ensuring positivity.

**Initialization**: The bias of the final linear layer is initialized to `log(e-1) ≈ 1.3133` so that `Softplus(bias) ≈ 1.0`, yielding an identity metric at initialization. Weights are initialized with `N(0, 0.05)` to keep early metric variation small. This ensures the ODE starts as a near-identity system.

#### Step 2: SDPA Heat Kernel Diffusion

The heat kernel on a Riemannian manifold with diagonal metric g is:

$$K_{ij} = \text{softmax}\left(-\frac{D^2_{ij}}{4t}\right), \quad D^2_{ij} = \sum_k g_k \cdot (h_i^k - h_j^k)^2$$

where t is a learnable diffusion timescale. Direct computation requires materializing the N×N distance matrix D², which is prohibitive for long sequences.

**Key insight**: The softmax operation is invariant to adding a row-constant term. We exploit this to factor the heat kernel as standard scaled dot-product attention:

$$K_{ij} = \text{softmax}\left(\frac{q_i \cdot k_j}{2t} - \frac{\|k_j\|^2}{4t}\right)$$

where $q = k = h_{\text{normed}} \odot \sqrt{g}$ (element-wise product with the square root of the metric). The term $\|h_i\|^2_g / (4t)$ is constant across columns j and drops out under softmax. The remaining expression is a standard SDPA operation with:

- **Query**: $q_{\text{scaled}} = q \cdot \frac{\sqrt{d}}{2t}$ (pre-scaled so SDPA's internal $1/\sqrt{d}$ normalization yields $1/(2t)$)
- **Key**: $k = h_{\text{normed}} \odot \sqrt{g}$
- **Value**: $V = W_v(\text{LN}(h))$
- **Attention bias**: $-\|k_j\|^2 / (4t)$ as a column-wise additive term

This factorization enables PyTorch's `F.scaled_dot_product_attention` to dispatch through FlashAttention. The N×N attention matrix is computed in SRAM tile-by-tile and never materialized to HBM, achieving **18,000 tokens/second** on NVIDIA DGX Spark — a 47× speedup over naive chunked distance-matrix computation.

The diffusion timescale t is parameterized as `t = Softplus(t_raw)` with a learnable scalar `t_raw`, initialized to produce t ≈ 1.0.

#### Step 3: Output Projection and Residual Target

The routed values are projected through a linear layer to produce the dynamics target:

$$\text{update} = W_o \cdot \text{SDPA}(q, k, V)$$
$$\text{target} = h + \text{update}$$

W_o is initialized with `N(0, 0.02)`. At initialization, the update is small but non-zero, which breaks copy symmetry — different positions receive different perturbations from the routed values, seeding geometric differentiation.

The target represents where the ODE wants to drive h. The contraction mechanism (next step) determines how fast h approaches this target.

#### Step 4: Liquid Time-Constant Contraction

The time derivative follows a Liquid Time-Constant (LTC) form:

$$\frac{dh}{dt} = -\frac{1}{\tau(h)} \cdot (h - \text{target})$$

where τ(h) is a per-position adaptive time constant:

$$\tau = \sigma(\text{TauNet}(h_{\text{normed}})) \cdot (\tau_{\max} - \tau_{\min}) + \tau_{\min}$$

TauNet is a small MLP (`d → d_metric → 1`) with sigmoid output scaled to [τ_min, τ_max] = [0.1, 3.0]. High τ means slow convergence (the position retains memory); low τ means fast convergence (the position rapidly adopts routed information). This gives each position an independent "viscosity" — some grid cells are stable references while others are actively transformed.

The LTC form guarantees contraction: for any target, `‖dh/dt‖ → 0` as `h → target`. This is essential for the Deep Equilibrium solver variant (Section 2.6) and for training stability.

**Alternative: Channel-wise gating.** We also implement a d-dimensional gate variant:

$$\frac{dh}{dt} = -\text{gate}(h) \odot (h - \text{target}), \quad \text{gate} = \sigma(\text{GateNet}(h_{\text{normed}}))$$

where each of the d dimensions has an independent gate in (0,1). This provides finer-grained control: specific representation dimensions can be frozen (gate ≈ 0) while others evolve (gate ≈ 1), effectively making the model's capacity per-position and per-dimension adaptive.

#### Step 5: FFN Residual

A feedforward network contributes an additive residual, amortized across integration steps:

$$\frac{dh}{dt} += \frac{\text{FFN}(\text{LN}(h))}{N_{\text{steps}}}$$

The FFN is a standard two-layer network with GELU activation (`d → d_ffn → GELU → dropout → d`). The division by N_steps ensures the total FFN contribution is independent of the integration resolution.

### 2.5 ODE Integration

The dynamics are integrated with forward Euler:

$$h_{n+1} = h_n + \Delta t \cdot f(t_n, h_n), \quad \Delta t = 1/N_{\text{steps}}$$

where f is the ContinuousDynamics module. We use N_steps = 16 by default, integrating over [0, 1]. During training, N_steps is optionally randomized in [12, 20] for temporal invariance — the model must produce correct outputs regardless of integration resolution.

The fixed-loop Euler solver is fully compatible with `torch.compile`, which unrolls the 16 iterations at trace time to produce a single optimized computation graph.

### 2.6 Solver Variants

We implement four ODE solvers with different memory/compute trade-offs:

| Solver | Memory | Compute | torch.compile | Use case |
|--------|--------|---------|--------------|----------|
| **Euler** (standard) | O(N_steps) | 1× | Yes | Default — fastest forward |
| **Chunked Euler** | O(N_steps/chunk) | ~3× | Yes | Memory-constrained training |
| **Deep Equilibrium (DEQ)** | O(1) | ~2× | Yes | Longest sequences |
| **Invertible Euler** | O(1) | ~7× | No | Deprecated |

The DEQ solver exploits the LTC dynamics' guaranteed contraction. The forward pass runs Euler with `torch.no_grad()` (zero autograd tape). The backward pass uses the Implicit Function Theorem at the equilibrium point h*:

$$(I - J_f^T) z = \nabla_{\text{output}} L$$

solved via fixed-point iteration: $z_{k+1} = \nabla L + J_f^T z_k$, converging in ~30 iterations when the spectral radius of J_f < 1 (guaranteed by LTC contraction). A single vector-Jacobian product then gives parameter gradients. Total backward cost: ~31 dynamics evaluations with O(1) memory, versus 16 evaluations with O(16) memory for standard Euler.

### 2.7 Curvature Engine

We compute a discrete approximation to the Ricci scalar curvature from the metric field:

$$\kappa_i = \frac{1}{d} \sum_k \frac{1}{(g_i^k)^2} \left( g_i^k \cdot D^2 g_i^k - \frac{1}{2} (Dg_i^k)^2 \right)$$

where Dg and D²g are central finite differences with mirror padding. This is a fully differentiable diagnostic used for regularization (`L_curv = λ · |κ|.mean()`) and monitoring. High curvature indicates sharp metric transitions; low curvature indicates smooth manifold geometry.

### 2.8 Output Head

After ODE integration, a LayerNorm + Linear layer maps h_final to 10-class color logits per position. Only positions marked as test output targets contribute to the loss.

---

## 3. Training

### 3.1 Loss Function

The total loss combines four terms:

$$L = \lambda_{CE} \cdot L_{CE} + \lambda_{curv} \cdot L_{curv} + \lambda_{\tau} \cdot L_{\tau} + \lambda_{CV} \cdot L_{CV}$$

**Cross-entropy with transform weighting:**

$$L_{CE} = \frac{1}{\sum w_i} \sum_i w_i \cdot \text{CE}(\hat{y}_i, y_i)$$

where $w_i = 5.0$ for transform cells (cells that changed between input and output) and $w_i = 0.05$ for copy cells. This 100:1 weighting ratio focuses the model on learning transformations rather than learning to copy. Additionally, per-grid losses are weighted by `3.0 - 2.0 × accuracy_b`, upweighting grids the model performs poorly on.

**Curvature penalty:** $L_{curv} = \lambda_{curv} \cdot |\kappa|_{\text{mean}}$ — penalizes non-smooth manifold geometry, preventing the metric from developing sharp discontinuities.

**Tau variance maximization:** $L_{\tau} = -\lambda_{\tau} \cdot \text{Var}(\tau)$ — encourages differentiation between high-tau (memory) and low-tau (active reasoning) positions.

**CV hinge loss:**

$$L_{CV} = \lambda_{CV} \cdot [\max(0, \text{floor} - \text{CV})^2 + \max(0, \text{CV} - \text{ceiling})^2]$$

A two-sided hinge that keeps the metric's coefficient of variation within a band [floor=3.0, ceiling=8.0]. The floor prevents metric rigidity collapse (where all metric components converge to the same value, destroying routing diversity). The ceiling prevents metric runaway (observed at wider scales).

### 3.2 Geometric Auxiliary Loss (Optional)

An optional geometric supervision phase directly trains the metric to encode spatial structure:

- **Phase 1** (steps 0-5K): Target distances are squared Manhattan distances between grid positions. CE weight is zero — the model only learns geometry.
- **Phase 2** (steps 5K+): Target distances incorporate object boundaries detected via BFS on same-color connected components. CE activates.

$$L_{geo} = \text{MSE}(D^2_{\text{model}}, D^2_{\text{target}})$$

where $D^2_{\text{model}}$ is computed from the SDPA-factored heat kernel keys. This scaffold helps the metric develop spatial awareness before task learning begins.

### 3.3 Training Procedure

Training proceeds in two phases with a sharp transition at step 5000:

**Phase 1 (Steps 0-5000): Geometric Scaffold**
- CE weight = 0, Geo loss weight = 1.0
- Tau frozen at 1.0 (uniform dynamics)
- Only MetricNet receives gradient
- Goal: learn spatial manifold structure

**Phase 2 (Steps 5000+): Task Learning**
- CE weight = 1.0, Geo loss weight = 0
- Tau unfreezes (position-adaptive dynamics)
- Curvature penalty activates (λ = 0.05)
- CV floor/ceiling penalty activates
- Full model receives gradients

**Data**: 70% procedural tasks from an infinite stream of 13 transformation rules (gravity, rotation, reflection, color remapping, etc.) and 30% real ARC training tasks (400 examples with D4 augmentation). The real data mixing is critical — without it, the model memorizes procedural-specific patterns that do not transfer to real ARC evaluation.

**Optimizer**: AdamW with two parameter groups:
- Geometric parameters (MetricNet, TauNet, t_diffusion, context pool): potentially different LR
- Other parameters (embedding, W_v, W_o, FFN, output head): base LR

Warmup: 500 steps linear, followed by cosine decay. Base LR: 3×10⁻⁴ at 572K params, scaled as `3×10⁻⁴ × sqrt(256/d)` for wider models.

### 3.4 Curriculum

The procedural task generator operates in three curriculum stages:

| Stage | Steps | Task Type | Description |
|-------|-------|-----------|-------------|
| 1 (GLOBAL) | 0-20K | Whole-grid spatial transforms | Rotation, reflection, gravity, color remap |
| 2 (RELATIONAL) | 20K-100K | Object-level reasoning | Connected component operations, pattern matching |
| 3 (COMPOSITION) | 100K+ | Multi-step patterns | Compound transformations |

Each stage introduces progressively more complex transformation types. However, as we discuss in Section 5, the curriculum signal from procedural tasks is less beneficial than real ARC data mixing.

---

## 4. The Phase Transition

Training exhibits a distinctive phase transition that we characterize in detail, as it reveals the fundamental mechanism by which the architecture bootstraps from random initialization.

### 4.1 The Copy Equilibrium

At initialization, with an identity metric (g ≈ 1 everywhere), the heat kernel assigns approximately uniform attention across all positions. The output projection W_o produces small random perturbations. The model's optimal strategy is to output the input unchanged — a copy equilibrium where every cell predicts its input color.

This equilibrium is stable because the copy loss gradient (from 95%+ cells that should be copied) is far stronger than the transform loss gradient (from the few cells that should change). The metric receives no gradient signal to differentiate positions.

### 4.2 CV as the Transition Variable

The CV floor penalty provides a persistent force pushing the metric toward higher variance. As training progresses, the metric's coefficient of variation gradually increases:

```
Step 0:      CV ≈ 0.13  (near-identity metric)
Step 500:    CV ≈ 3.0   (floor penalty engages)
Step 2500:   CV ≈ 3.3   (gradual climb)
Step 5000:   CV ≈ 6.6   (approaching threshold)
Step 5350:   CV ≈ 7.0   → PHASE TRANSITION
```

When CV reaches approximately 6-7, the metric has developed enough position-to-position variation for the SDPA heat kernel to produce non-uniform routing. Some positions become "closer" than others on the manifold, breaking the copy equilibrium. Once a few transform cells receive correct predictions via non-trivial routing, the gradient signal amplifies, and the model rapidly transitions:

```
Step 5350: Train xform  6.8% → Step 7000: 67.3% → Step 7500: 87.6%
```

### 4.3 Scale Invariance

The transition is observed identically at 572K (d=256) and 5M (d=768) parameters:

| Scale | Transition Step | CV at Transition | Post-Transition Train Xform |
|-------|----------------|------------------|-----------------------------|
| 572K (d=256) | ~5,350 | ~6.0 | 80-95% |
| 5M (d=768) | ~5,500 | ~7.0 | 80-95% |

The 5M model requires slightly higher CV because wider dimensions with Kaiming initialization produce smaller per-parameter metric deviations — more total variance is needed before individual position pairs differentiate. The transition mechanism is identical; only the threshold shifts.

### 4.4 Curvature as Consequence, Not Cause

Curvature (|κ|) is near zero before the transition (0.0002-0.001) and develops rapidly after (0.005 → 0.05 within 2000 steps). The metric first develops variance (CV), which enables non-trivial routing, which enables task learning, which produces curvature as a structural consequence. Curvature is diagnostic, not causal.

---

## 5. Experimental Results

### 5.1 Setup

**Task**: ARC-AGI — 400 training tasks, 400 evaluation tasks. Each task presents 2-5 input/output demonstration pairs and one test input; the model must predict the test output.

**Evaluation metric**: Transform accuracy — fraction of correctly predicted cells among cells that change between input and output. This excludes copy cells, isolating the model's ability to learn the underlying transformation rule.

**Hardware**: NVIDIA DGX Spark (GB10, 128GB unified memory), `torch.compile` enabled.

### 5.2 Baseline Results (No Test-Time Training)

| Configuration | Params | Peak Eval Xform | Best Step | Throughput |
|--------------|--------|-----------------|-----------|------------|
| LiquidARC (d=256) | 572K | **61.1%** | 42K | 9,500 tok/s |
| LiquidARC (d=768) | 5.0M | 54.2% | 21K | 3,200 tok/s |
| Flat Transformer (d=256, 2 layers) | ~830K | — | — | — |

The 572K model achieves the best eval transform accuracy, peaking at 61.1% at step 42K. The 5M model, with 8.5× more parameters, peaks 7 points lower at 54.2%. This counterintuitive result reveals that the bottleneck is generalization, not capacity: the larger model achieves 90%+ training accuracy but memorizes procedural patterns rather than learning transferable structure.

### 5.3 Training Data Distribution is the Dominant Factor

The single most impactful change across all experiments was adding 30% real ARC data to the training mix:

| Data Mix | Peak Eval Xform |
|----------|-----------------|
| 100% procedural (V1) | ~19% |
| 70% procedural + 30% real ARC (V2) | **~61%** |

This 3.2× improvement dwarfs all architectural modifications. The 13-rule procedural generator creates tasks that share statistical regularities absent from real ARC — the model learns to exploit these regularities rather than learning general transformation logic.

### 5.4 Test-Time Training: WHERE/WHAT Decomposition

Test-time training (TTT) adapts the model's parameters to a single task using 100 gradient steps on the demonstration pairs. The key architectural finding is the separation of two adaptation channels:

**WHERE** — MetricNet + TauNet (53K params, 9.3%): Controls the manifold geometry that routes information between positions. Adapting these parameters teaches the model which grid cells are related for this specific task.

**WHAT** — W_o (66K params, 11.5%): Controls the content transformation applied to routed values. Adapting this teaches the model what operation to perform (recolor, reflect, transpose, etc.).

Ablation on step 15K checkpoint:

| TTT Configuration | Xform Acc | Delta |
|-------------------|-----------|-------|
| No TTT (baseline) | 13.7% | — |
| WHERE only (MetricNet + TauNet) | 15.9% | +2.2pp |
| **WHERE + WHAT (+ W_o)** | **43.7%** | **+30.0pp** |

W_o adaptation alone accounts for 93% of the TTT improvement (+27.8pp out of +30.0pp). This makes physical sense: the manifold geometry provides the routing substrate (which positions communicate), but the output projection determines the content of that communication. For a novel transformation rule, the model needs to express a new content operation, not just re-route existing operations.

### 5.5 TTT Loss: Transform Cells Only

An important implementation detail: the TTT loss must be computed only on transform cells (`xform_loss`), not on all cells (`ce_loss`). Full CE includes copy cells, which constitute 95%+ of the sequence. Adapting on copy cells teaches the model to preserve information unchanged — the exact opposite of what's needed for novel transforms. Switching from `ce_loss` to `xform_loss` turned TTT from destructive (-30pp) to constructive (+10pp) on post-transition checkpoints.

### 5.6 TTT Trajectory Across Training

| Step | Baseline Xform | TTT Xform | TTT Delta |
|------|---------------|-----------|-----------|
| 20K | 32.5% | 42.3% | +9.8pp |
| 25K | 37.5% | **44.4%** | +7.0pp |
| 30K | 39.1% | 43.5% | +4.4pp |
| 35K | 38.9% | 39.4% | +0.5pp |
| 40K | 34.8% | 41.7% | +7.0pp |
| 45K | 40.0% | 41.6% | +1.6pp |
| 50K | 40.0% | 41.4% | +1.4pp |

TTT remains consistently positive across all checkpoints (20K-50K) — no degradation, unlike V1 which peaked at step 15K then declined. The combination of CV floor penalty (maintaining metric plasticity), real ARC data (preventing procedural overfitting), and correct loss formulation (xform_loss) resolved the instability.

### 5.7 Scaling Behavior

Width scaling from 572K to 5M parameters reveals several properties:

**1. Capacity is not the bottleneck.** The 40-point train/eval gap at 5M (90% train, 50% eval) shows the model can compute arbitrary transformations but cannot generalize from training distribution to evaluation distribution. More parameters make memorization easier, not generalization better.

**2. CV ceiling is necessary at wider scales.** The 572K model's metric self-regulates via task gradients. At 5M, smaller per-parameter deviations from Kaiming initialization mean the CV floor penalty dominates the metric gradient before task learning begins, causing runaway (CV 3 → 14 → 19 → NaN). A ceiling at 8.0 bounds CV during the dangerous pre-transition phase.

**3. TTT becomes destructive post-transition at large scale.** Pre-transition: +28.5pp lift (baseline 16.1% → TTT 44.6%). Post-transition: -25.6pp at step 27.5K. Larger models develop richer universal geometric structure that TTT's 100 gradient steps overwhelm. TTT compensates for underdeveloped geometry; it cannot improve already-trained universal structure at this learning rate.

**4. Extended training is harmful.** Beyond step 10-15K, eval CE degrades monotonically (1.50 → 1.89 by step 50K) while train accuracy continues improving. The model overfits to the procedural training distribution. The optimal stopping point is 10-15K steps regardless of model size.

---

## 6. Geometric Dynamics

### 6.1 Metric Coefficient of Variation

CV tracks the diversity of the metric field across positions. It serves as the primary indicator of geometric health:

| CV Range | Interpretation |
|----------|---------------|
| < 3.0 | Metric rigidity — near-identity metric, routing is uniform |
| 3.0-6.0 | Healthy regime — differential routing emerging |
| 6.0-7.0 | Critical threshold — phase transition triggers |
| 5.0-6.0 | Post-transition equilibrium — task-driven metric structure |
| > 8.0 | Runaway (requires ceiling at wider scales) |

The CV floor/ceiling hinge loss maintains metric plasticity throughout training. Without the floor, CV decays from 7.7 to 3.5 over 50K steps (V1 observation), collapsing metric diversity and destroying test-time adaptation capability.

### 6.2 Tau Dynamics

The per-position time constant τ converges quickly regardless of model scale:

```
Step 0:    τ_avg ≈ 0.81 (near-max, freshly initialized)
Step 500:  τ_avg ≈ 0.60 (rapid drop)
Step 2000+: τ_avg ≈ 0.58-0.65 (stable)
           τ_std ≈ 0.16 (moderate position diversity)
           τ range: [0.50, 1.00]
```

The model learns to use lower τ (faster convergence) for most positions, with a spread of 0.5-1.0 across positions. Higher τ positions serve as stable reference points; lower τ positions are actively transformed.

### 6.3 Curvature

Curvature |κ| follows a characteristic trajectory:

```
Pre-transition:  |κ| ≈ 0.0002  (flat manifold)
During transition: |κ| jumps to 0.005-0.05
Post-transition: |κ| ≈ 0.006-0.015 (structured manifold)
Extended training: slow linear growth 0.007 → 0.017
```

The curvature penalty (λ = 0.05) prevents sharp metric discontinuities while allowing meaningful manifold structure. Without regularization, curvature can grow unboundedly, producing numerically unstable heat kernels.

---

## 7. Connection to Physics

### 7.1 Heat Equation on Riemannian Manifolds

The architecture draws directly from the heat equation on a Riemannian manifold (M, g):

$$\frac{\partial u}{\partial t} = \Delta_g u$$

where Δ_g is the Laplace-Beltrami operator. The heat kernel K(x, y, t) gives the fundamental solution — the amount of "heat" (information) that flows from y to x in time t. On a flat manifold, this is a Gaussian; on a curved manifold, the kernel is shaped by the geometry.

In LiquidARC, the token hidden states are "heat" distributed on a learned manifold. The MetricNet defines the manifold geometry; the SDPA heat kernel implements one step of diffusion; and the 16 Euler steps integrate the diffusion process over a finite time horizon.

### 7.2 Liquid Time Constants as Viscosity

The LTC form `dh/dt = -(1/τ) · (h - target)` is analogous to a viscous fluid with position-dependent viscosity. Low τ (low viscosity) positions rapidly adopt information from their neighbors via the heat kernel. High τ (high viscosity) positions resist change, maintaining their initial state as reference anchors. The model learns a spatially varying viscosity field that controls the flow of information across the geometric manifold.

### 7.3 Gauge Invariance of SDPA Factorization

The SDPA factorization exploits a gauge invariance of the softmax: adding a position-independent constant to logits leaves the distribution unchanged. The term $\|h_i\|^2_g / (4t)$ depends only on position i (the query) and is constant across all keys j, so it drops out under softmax. This is not an approximation — the SDPA formulation computes the exact same heat kernel as the direct N×N distance matrix computation.

---

## 8. Discussion

### 8.1 Emergence from Iteration

The fundamental design principle is that complex behavior emerges from iterated application of simple dynamics, rather than from stacking specialized layers. A single 356K-parameter dynamics module, applied 16 times, produces behavior equivalent to a much deeper network — but with several advantages:

1. **Memory efficiency**: The autograd tape stores 16 applications of one module, not 16 independent modules. Combined with the DEQ solver, memory can be reduced to O(1).

2. **Continuous depth**: The number of integration steps can be varied at test time. Simple tasks may converge in 4 steps; complex tasks may need 24. The architecture naturally supports adaptive computation.

3. **Physical inductive bias**: The ODE dynamics have guaranteed stability (LTC contraction), smooth interpolation between steps, and time-reversal structure that discrete layers lack.

### 8.2 The Generalization Bottleneck

Across all experiments — 572K vs 5M parameters, with and without geometric scaffold, various curriculum strategies — the eval transform accuracy plateau at ~50-61% is determined by the training data distribution, not by model capacity or architectural choices. The 30% real ARC mixing produced more improvement than any other single intervention.

This suggests that for few-shot visual reasoning, the critical resource is task diversity during training, not model expressiveness. 400 ARC training tasks, even with D4 augmentation and procedural supplements, may be insufficient to learn the meta-skill of "figuring out transformation rules from examples."

### 8.3 WHERE/WHAT as a General Principle

The WHERE/WHAT decomposition is not specific to ARC or to ODE-based architectures. In any system that routes information between positions and then transforms it, the routing mechanism (WHERE) and the transformation mechanism (WHAT) serve fundamentally different roles.

For task adaptation, WHAT is the dominant lever (+28pp vs +2pp). This makes sense: most ARC tasks use similar spatial reasoning (nearby cells influence each other) but require different content operations (color swap vs rotation vs pattern completion). The geometry provides a universal routing substrate; the content transformation must be task-specific.

### 8.4 Limitations

1. **Sequence length**: Maximum context of 2048 tokens covers ~64% of ARC tasks. Larger grids or more demonstration pairs require longer sequences.

2. **TTT latency**: 100 gradient steps take ~2.8 seconds per task. For competition settings requiring < 1s/task, an amortized approach (predicting W_o deltas in a single forward pass) is needed.

3. **Diagonal metric**: The current metric is diagonal (g is a d-dimensional vector per position, not a d×d matrix). A full metric tensor would allow correlation between representation dimensions but at O(d²) cost per position.

4. **Fixed integration schedule**: All 16 steps use the same dynamics. Step-evolving dynamics (where the metric network is modulated per step) would allow different ODE steps to serve different computational roles — early steps for analysis, late steps for output generation.

---

## 9. Conclusion

LiquidARC demonstrates that continuous-time ODE dynamics with learned Riemannian geometry can serve as a viable alternative to discrete transformer layers for few-shot visual reasoning. The SDPA-factored heat kernel enables practical throughput (18K tok/s) while providing a principled information routing mechanism grounded in differential geometry. The architecture exhibits a clean phase transition from random initialization, a natural WHERE/WHAT decomposition that enables targeted test-time adaptation, and scale-invariant training dynamics.

The central finding is that the bottleneck for this class of models is not architectural — it is the training data distribution. With 572K parameters, a single shared dynamics module, and 30% real ARC training data, the model achieves 61.1% transform accuracy on held-out tasks. This suggests that future progress lies in task diversity and training methodology, not in scaling parameters or adding architectural complexity.

---

## Appendix A: Hyperparameters

### Model Configuration (d=256)

| Parameter | Value |
|-----------|-------|
| d_model | 256 |
| d_metric | 64 |
| d_ffn | 512 |
| n_ode_steps | 16 |
| tau_min | 0.1 |
| tau_max | 3.0 |
| t_diffusion_init | 1.0 |
| alpha_logit_init | 2.2 |
| dropout | 0.1 |
| n_colors | 10 |
| max_grid_size | 30 |
| max_grids | 16 |
| max_seq_len | 2048 |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| batch_size | 8 |
| learning_rate | 3e-4 |
| warmup_steps | 500 |
| weight_decay | 0.01 |
| max_steps | 50,000 |
| curvature_lambda | 0.05 |
| tau_var_lambda | 0.001 |
| cv_floor_target | 3.0 |
| cv_ceiling_target | 8.0 |
| cv_floor_lambda | 0.1 |
| transform_weight | 5.0 |
| copy_weight | 0.05 |
| real_arc_mix_ratio | 0.3 |
| tau_freeze_steps | 5000 |

### TTT Configuration

| Parameter | Value |
|-----------|-------|
| ttt_steps | 100 |
| ttt_lr | 1e-3 |
| ttt_loss | xform_loss |
| ttt_early_stop_threshold | 0.01 |
| ttt_curvature_lambda | 0.01 |
| ttt_unfreeze_modules | MetricNet + TauNet + W_o |

## Appendix B: Parameter Count Breakdown (d=256)

| Component | Parameters |
|-----------|-----------|
| **ARCEmbedding** | **~213,248** |
| - color_embed (11 × 256) | 2,816 |
| - pos_x_embed (31 × 256) | 7,936 |
| - pos_y_embed (31 × 256) | 7,936 |
| - role_embed (4 × 256) | 1,024 |
| - sep_embed (4 × 256) | 1,024 |
| - grid_id_embed (16 × 256) | 4,096 |
| - LayerNorm | 512 |
| **ContextPool** | **~5,826** |
| - attn_pool (256→64→1) | 16,577 |
| - out_proj (256→256) | 65,792 |
| **ContinuousDynamics** | **~356,000** |
| - MetricNet (512→64→256 + biases) | ~49,472 |
| - TauNet (256→64→1 + biases) | ~16,513 |
| - W_v (256→256, no bias) | 65,536 |
| - W_o (256→256, no bias) | 65,536 |
| - FFN (256→512→256 + biases) | ~263,168 |
| - LayerNorms (×3) | 1,536 |
| - t_diffusion | 1 |
| - alpha_logit | 1 |
| **OutputHead** | **~2,570** |
| - LayerNorm + Linear (256→10) | 2,570 |
| **Total** | **572,238** |
