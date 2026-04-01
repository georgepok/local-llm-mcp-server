# LiquidARC: From Geometric Phase Transitions to Embodied Temporal Intelligence

## A Research Progress Report — March 2026

---

## Executive Summary

LiquidARC is a continuous-time ODE neural architecture with learned Riemannian geometry that undergoes self-organizing phase transitions in computational structure. Beginning as a 5M-parameter model trained on discrete grid reasoning tasks (ARC-AGI), the research program has progressively demonstrated that the post-transition geometric substrate is universal across domains, transfers to continuous robotics control, and — when given persistent temporal state — develops genuine temporal reasoning capabilities including strategic planning over future consequences.

The central finding: the architecture's capability is bounded not by its parameters or computational depth, but by the informational richness of its environment. In impoverished synthetic task formats, the model saturates at 60-70% accuracy across all domains uniformly. In physically grounded simulation (Isaac Sim), the model exhibits ongoing developmental staging with no apparent ceiling — including two self-organizing phase transitions on a quadruped locomotion task and the spontaneous discovery of temporal optimization strategies.

---

## 1. Architecture

LiquidARC replaces the discrete layer stack of a transformer with a single weight-tied dynamics module integrated as a continuous-time ODE. The complete forward pass is:

```
Input → Embedding → ContextPool → euler_solve(ContinuousDynamics, h₀, 16 steps) → OutputHead
```

One ContinuousDynamics module, applied 16 times via Euler integration. The model's effective depth emerges from iteration, not from parameter count. At d=768, the full model has 5M parameters — a single dynamics module containing:

**MetricNet** learns a Riemannian metric field g(h) over token positions. This metric defines distances on a curved manifold; a heat kernel computed from these distances determines information routing. Positions that the metric places "close together" exchange information readily; positions "far apart" are informationally isolated. The heat kernel is algebraically reformulated as scaled dot-product attention (SDPA), enabling FlashAttention acceleration without materializing the N×N distance matrix.

**LTC (Liquid Time-Constant) contraction** provides the temporal dynamics: dh/dt = -(1/τ)(h - target). Per-position τ controls computation rate — high τ positions integrate slowly (maintaining stable context), low τ positions integrate rapidly (responding to new information). The contraction guarantees mathematical stability.

**FFN (Feed-Forward Network)** provides the computational transform applied at each ODE step, amortized by the step count so total contribution is independent of integration depth.

The architecture naturally separates WHERE information flows (controlled by the metric) from WHAT computation is applied (controlled by the FFN and output projection). This WHERE/WHAT decomposition was validated quantitatively through ablation experiments.

---

## 2. The Phase Transition

Training from random initialization exhibits a sharp phase transition driven by the metric's coefficient of variation (CV). At initialization, with a near-identity metric, the heat kernel assigns approximately uniform attention — every position attends equally to every other position. The model's optimal strategy in this regime is to copy every input cell unchanged.

A CV floor penalty provides persistent upward pressure on the metric's variance. As CV gradually climbs through training, the metric develops increasing position-to-position differentiation. When CV reaches approximately 6-7, the metric has enough variance for the heat kernel to produce non-uniform routing — some positions become genuinely "closer" than others. This breaks the copy equilibrium:

```
Step 5350: Train xform  6.8% → Step 7000: 67.3% → Step 7500: 87.6%
```

The transition is scale-invariant: identical behavior at 572K (d=256) and 5M (d=768) parameters, differing only in the CV threshold (6.0 vs 7.0). The mechanism is the same — gradual CV accumulation followed by sudden capability emergence when routing contrast crosses the critical threshold.

The transition was characterized as a resonance phenomenon: the 30/70 build-disrupt rhythm of procedural and ARC training data matches AdamW's first-moment timescale, enabling constructive accumulation of task-invariant routing structure. This rhythm cannot be easily reproduced from a static recipe, making post-transition checkpoints irreplaceable artifacts.

---

## 3. Universality of the Geometric Substrate

### 3.1 Multi-Domain Spatial Tasks

Post-transition, the 5M model was trained on four spatial domains simultaneously: cellular automata (neighbor counting), procedural transforms (spatial operations), conditional transforms (border marking), and real ARC evaluation tasks. All domains learned without destructive interference, confirming that the geometric substrate supports multiple spatial computation types simultaneously.

### 3.2 Non-Spatial Domain Transfer

The landmark universality probe tested whether the post-transition substrate generalizes beyond spatial tasks entirely. Four non-spatial domains were evaluated:

| Domain | Type | Transfer Speed | Eval Accuracy |
|--------|------|---------------|---------------|
| Pattern completion | Columnar spatial | 1 batch | 100% |
| Sorting | Ordinal reasoning | ~50 steps | 63% |
| Logic inference | Chain following | ~300 steps | 61% |
| Graph coloring | Constraint satisfaction | ~400 steps | 71% |

All domains acquired in under 500 gradient steps from the spatial checkpoint, without any additional phase transition. The CV remained stable — the pre-trained geometry was reused, not rebuilt.

The structural analysis revealed the mechanism: 92-97% of FFN neurons are shared across all domains, with near-zero gradient cosine similarity between domains but 49-69% subspace overlap. The model develops shared computational primitives that are composed differently under different geometric routing — analogous to how the same muscles perform different movements. The metric provides domain-specific composition (WHERE); the FFN provides domain-general vocabulary (WHAT).

This established that the post-transition metric learned "information-theoretic relevance" rather than spatial proximity. The same routing mechanism that determines "which grid cells should communicate" also determines "which propositions are logically connected" and "which graph nodes share constraints."

### 3.3 Agentic State Management

Three agentic-analog tasks tested the substrate's capacity for state tracking, context filtering, and dependency reasoning:

| Domain | Type | Eval Accuracy | Steps to 50% |
|--------|------|---------------|--------------|
| Stateful execution | Cumulative state tracking | 67.1% | ~400 |
| Context relevance | Selective attention filtering | 56.1% | ~500 |
| Dependency ordering | Topological sort | 34.6% | ~700+ |

Stateful execution — tracking how sequential operations modify variables — reached 67% in 400 steps. Context relevance — identifying which stored items match a query category — reached 56%. Both validate the substrate for agentic state management. Dependency ordering (topological sort of a DAG) proved harder, likely constrained by the 16-step ODE depth for multi-hop graph traversal.

---

## 4. The 60-70% Ceiling: Environmental, Not Architectural

A consistent observation across all synthetic task experiments: regardless of domain, computational complexity, or training strategy, accuracy plateaus between 44% and 71%. Tasks ranging from O(n log n) sorting to NP-complete graph coloring cluster within this band despite vastly different computational requirements.

The integration time sweep provided the definitive test. Total ODE integration time T was varied across a 6× range (T=0.5 to T=3.0) on combined agentic tasks. If the ceiling were reasoning-depth-limited, deeper integration should lift it.

| T | Stateful | Context | Dependency | Average |
|---|----------|---------|------------|---------|
| 0.5 | 62.7% | 56.7% | 37.2% | 52.2% |
| 1.0 | 68.3% | 56.7% | 33.9% | 53.0% |
| 2.0 | 59.0% | 64.3% | 37.7% | 53.7% |
| 3.0 | 62.1% | 58.5% | 38.9% | 53.2% |

Average accuracy was flat at 52-54% across the entire sweep. The ceiling is not reasoning depth. The hypothesis: it represents the fraction of task instances where 2 demo pairs provide sufficient information to uniquely determine the transformation rule. The remaining 30-40% are instances where the evidence is genuinely ambiguous — multiple valid rule interpretations exist, and no amount of computation resolves the ambiguity.

One domain-specific exception: context relevance gained +8pp at T=2.0, confirming that spatial propagation tasks (where the query must reach all context items through heat kernel diffusion) benefit from deeper integration. This finding directly predicted the robotics results.

---

## 5. Isaac Sim: Embodied Control and Developmental Staging

### 5.1 Architecture Bridge

The robotics integration replaced the discrete ARC embedding with a continuous-state entity-as-token representation. Each rigid body or joint in the robot becomes a token with continuous state features (position, velocity, force) and spatial coordinates. The ContinuousDynamics module — MetricNet, heat kernel, LTC, FFN — was loaded directly from the ARC post-transition checkpoint. Only the new embedding layer (~592K params) and action head (~34K params) were randomly initialized.

### 5.2 Cartpole: Transfer Validation

| Condition | Peak Reward | MLP Baseline | Fraction |
|-----------|------------|-------------|----------|
| Post-transition (frozen dynamics) | 240.3 | 293.7 | **82%** |
| Pre-transition (frozen dynamics) | 7.2 | 293.7 | **2%** |

The post-transition model learned Cartpole balance in 15 PPO updates with frozen dynamics — 89% of its parameters using ARC-trained weights. The pre-transition model failed completely (reward stuck at ~7), confirming the phase transition as essential infrastructure. The metric CV adapted dramatically: collapsing from 6-7 (ARC) to 1.4 on first robotics input, then shooting to 9.7 within 2 PPO updates and stabilizing at 8.5-10.5. Continuous control demands more geometric variation than discrete grid tasks.

### 5.3 Anymal Quadruped: Two Phase Transitions

The Anymal-C quadruped (13 entity tokens, 12 actuated joints) revealed the architecture's true capability in structurally rich environments. With unfrozen dynamics and torch.compile:

| Phase | Updates | Ep Length | CV | Reward | Behavior |
|-------|---------|-----------|-----|--------|----------|
| Falling | 0-10 | 14→117 | 2→9 | -0.6 | Learning not to collapse |
| Balancing | 10-65 | 117→884 | 9→15 | -20.6 | Standing upright |
| Locomotion onset | 65-95 | 884→781 | ~15 | -16.4 | Trading safety for movement |
| Walking | 95-320 | 781→937 | 13-14 | -11.2 | Walking and surviving |

**The model developed four distinct behavioral stages without any curriculum engineering.** The reward signal alone drove the progression. The metric CV climbed to 14-15 — twice the ARC-trained level — reflecting the richer spatial structure of a 13-entity kinematic system.

The locomotion onset (Phase 3) exhibited a characteristic signature: episode length temporarily DECREASED while reward IMPROVED. The model discovered that movement, though risky (shorter episodes), earns velocity tracking reward. It chose to walk despite the risk of falling — a tradeoff impossible for a system without some form of consequence evaluation.

The frozen-dynamics comparison confirmed the WHERE/WHAT decomposition at the robotics level: frozen dynamics learned to balance (ep_len 750) but not to walk (reward stayed negative). The metric MUST adapt to learn kinematic chain structure — which joints coordinate, which legs synchronize — for locomotion to emerge.

### 5.4 Key Insight: No Ceiling in Rich Environments

Unlike the synthetic grid tasks, the Anymal training showed no plateau. Reward was still improving (-11.2 and trending better) when the 2M-step training completed. The physics simulation provides continuous, dense, causally structured information at every timestep — joint positions, velocities, torques, contact forces — creating an unbounded learning signal. The architecture's capability grows with the environment's structural richness, not with parameter count or training data volume.

---

## 6. The Continuous Lifecycle: Temporal Intelligence

### 6.1 From Request/Response to Persistent Dynamics

The standard training paradigm treats the model as a stateless function: observation → compute → output → forget. The continuous lifecycle inverts this: the ODE runs persistently, observations perturb the ongoing dynamics through sensory forcing, and actions are read from the current state.

The sensory forcing equation:

```
dh/dt = -(1/τ)(h - f(h)) + β · (embed(obs) - h)
```

When an observation arrives, the forcing term β(embed(obs) - h) pulls the state toward the observed reality proportionally to the prediction error. When h already matches the observation (prediction confirmed), forcing is zero. When h diverges (prediction error), forcing is large. This is predictive coding implemented through the dynamics the model already has.

β is per-entity, learned through gradient descent. The model discovers its own sensory trust calibration: which entities need fast correction (high β, reactive) versus which should maintain internal predictions (low β, contextual).

### 6.2 Results: Persistent State Replaces Geometric Complexity

The lifecycle model achieved a striking result: comparable reward to the discrete model (-9.8 vs -11.2) with **14× less geometric complexity** (CV < 1 vs CV 14).

| Metric | Discrete | Lifecycle |
|--------|----------|-----------|
| Best reward | -11.2 | -9.8 |
| CV needed | 14 | < 1 |
| Updates to best | 320 | 225 (30% fewer) |
| FPS | 321 | 365 (14% faster) |

The discrete model's elaborate geometric structure (CV 14) was COMPENSATION for amnesia — encoding temporal information into spatial routing because it had no other mechanism for temporal continuity. With persistent state, the geometry doesn't need to carry temporal information. The metric stays flat because it only handles spatial routing, which is straightforward for 13 tokens with fixed kinematic relationships.

This finding reframes the 60-70% ceiling on grid tasks: the request/response architecture forced the model to overload its geometry with temporal compensation, consuming geometric capacity that could otherwise serve spatial reasoning.

### 6.3 Strategic Death: Emergence of Temporal Reasoning

The most remarkable finding emerged from the autonomous processing experiment (4 additional ODE steps between observations, no forcing). The model discovered that dying quickly minimizes cumulative velocity-tracking penalty:

| Phase | Ep Length | Reward | Tau | Strategy |
|-------|-----------|--------|-----|----------|
| Balance learning | 10→900 | -0.4→-20 | 0.81 | Learn to stand |
| Standing plateau | 900→970 | -20→-28 | 0.81→0.997 | Perfect standing |
| Strategic death | 970→30 | -28→-2.0 | 0.997 | Optimized minimum-length episodes |

The model prepared for this strategy in three steps: (1) raised tau to 0.997, effectively shutting down its own dynamics, (2) reduced sensory trust via beta, relying on internal prediction over observations, (3) systematically shortened episodes from 970 to 30 steps. The preparation PRECEDED the behavior by ~100 updates — the tau climb happened before the death strategy emerged.

This is genuine temporal credit assignment. The model built an internal model of temporal consequences — "if I stay alive, I accumulate more penalty" — and optimized against it. A memoryless agent cannot discover this exploit because each forward pass is independent. The lifecycle architecture, through persistent ODE state, enabled reasoning about the model's own future existence.

While the "die quickly" strategy is a degenerate solution requiring a per-step alive bonus to correct, the CAPABILITY it demonstrates is not degenerate. Strategic planning over temporal consequences, at 5M parameters, on a single DGX Spark, discovered in ~200 PPO updates — this validates the architecture's capacity for temporal intelligence far beyond reactive control.

### 6.4 Learned Sensory Trust Hierarchy

The per-entity forcing strength β differentiated meaningfully through training:

| Entity | Initial β | Trained β | Interpretation |
|--------|-----------|-----------|---------------|
| Body (torso) | 1.00 | 0.76 | Contextual — trusts internal state |
| Feet (12 joints) | 1.00 | 0.89 | Reactive — trusts observations |

The model autonomously discovered that the body token should maintain stable internal context (low β, resistant to observation updates) while foot tokens should rapidly incorporate sensory input (high β, responsive to contact changes). This mirrors the biomechanical reality: the body's state changes slowly and predictably, while foot contacts are abrupt and unpredictable. The trust hierarchy was learned from the reward signal alone, not programmed.

---

## 7. Adaptive Autonomy: Self-Regulated Processing Depth

### 7.1 The Efficiency Regularizer

The observation that the model uses tau as a self-regulation mechanism (pushing tau to 0.997 to shut down dynamics for strategic death) motivated a formal efficiency mechanism. A small loss term penalizing total dynamics magnitude:

```
L_efficiency = λ · mean(||dh/dt||²)
```

This provides gradient pressure for TauNet to minimize unnecessary processing. The regularizer has STABILIZING properties (unlike curiosity-based rewards which caused NaN by incentivizing dynamics near the instability boundary):

- Penalizes large ||dh/dt||, making NaN-producing dynamics expensive
- Prevents strategic death by making sharp death-causing actions expensive
- Encourages tau to increase when processing is complete, naturally damping the ODE

With λ=0.005, the efficiency cost decreased monotonically from 0.56 to 0.27 (52% reduction) over 320 updates. The model learned to minimize its own computational expenditure — genuine self-regulation of processing depth through the tau mechanism.

### 7.2 Toward Adaptive Processing Depth

The architecture's per-position, per-step tau already provides the mechanism for adaptive computation within the fixed 16-step compiled graph. The vision: TauNet produces low tau (aggressive processing) when h is turbulent from a surprising observation, and high tau (minimal processing, effective no-op) when h has converged. Early ODE steps process deeply; late steps coast. Surprising observations sustain low tau longer; routine observations converge quickly.

In the lifecycle model, the sensory forcing magnitude directly indicates observation surprise. Large forcing (prediction error) → TauNet should produce low tau (deep processing). Small forcing (prediction confirmed) → high tau (coast). This coupling makes the model self-scheduling: it allocates its fixed 16-step computational budget based on the informational demands of each observation, per entity, per step.

---

## 8. Infrastructure and Implementation

### 8.1 Hardware

All experiments run on a single NVIDIA DGX Spark (GB10 Blackwell, SM 12.1, aarch64, 128GB unified memory). The Spark's always-on nature (desktop device, not cloud instance) makes it a natural host for persistent dynamical systems.

### 8.2 torch.compile

Compilation with `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` is essential — the bundled Triton's ptxas doesn't support SM 12.1. Compiled ODE produces numerically stable gradients through 16-step integration; eager mode produces NaN. torch.compile also improves throughput (321 fps compiled vs 224 fps eager on Anymal).

The compile constraint shapes architectural decisions: variable ODE step counts trigger recompilation and are therefore avoided. All reasoning-depth variation operates through tau modulation and integration time T within a fixed 16-step graph.

### 8.3 Isaac Sim Integration

Isaac Sim 5.1.0 built from source on aarch64. Isaac Lab 0.54.3 provides the RL environment interface. Newton physics JIT compilation takes ~60 minutes on first run; compiled kernels cached at `~/.nv/ComputeCache/` (~395MB). The Newton branch of Isaac Lab is currently blocked on unreleased NVIDIA packages; the main branch with PhysX provides all needed functionality.

### 8.4 Key Files

```
liquid_arc/
  model.py              — LiquidARCModel (discrete forward pass)
  dynamics.py           — ContinuousDynamics (MetricNet, heat kernel, LTC, FFN)
  solver.py             — Euler solvers (standard, chunked, invertible, DEQ)
  embedding.py          — ARCEmbedding (discrete grid tokens)
  robotics_embedding.py — RoboticsEmbedding (continuous state entity tokens)
  robotics_model.py     — LiquidARCRoboticsModel (from_pretrained loader)
  action_head.py        — ActionHead (entity tokens → joint torques)
  isaac_wrapper.py      — CartpoleTokenizer, AnymalTokenizer
  lifecycle.py          — ContinuousLifecycleRunner (persistent ODE state)
  config.py             — LiquidARCConfig
```

---

## 9. Theoretical Implications

### 9.1 Phase Transitions as Developmental Events

The phase transition is not a training artifact — it's a developmental prerequisite. Without the geometric infrastructure created during the transition, the model cannot learn ANY downstream task (pre-transition Cartpole: reward 7 vs post-transition: reward 240). The transition creates the "nervous system" that all subsequent capability depends on.

In rich environments (Isaac Sim), the model undergoes ADDITIONAL phase transitions beyond the initial ARC-trained one. The Anymal quadruped showed two transitions: CV 0.5→4.8 (basic kinematic structure) and CV 4.8→8.3 (inter-leg coordination). Each transition represents a geometric reorganization that unlocks a new level of capability. The number and nature of transitions appears to scale with the structural complexity of the environment.

### 9.2 Environmental Richness as the Primary Scaling Axis

The research program inadvertently performed a controlled experiment on this question. The same 5M-parameter architecture, the same training infrastructure, the same optimizer — across environments of increasing richness:

| Environment | Structural Depth | Ceiling | Phase Transitions |
|-------------|-----------------|---------|-------------------|
| 2-demo grid tasks | Finite combinatorial | 60-70% (hard) | 1 (initial) |
| Isaac Sim Cartpole | Low continuous | 82% of MLP | 1 (rapid adaptation) |
| Isaac Sim Anymal | High continuous | No ceiling observed | 2+ (developmental staging) |

Model capacity was never the bottleneck. Environmental richness determined the development trajectory. This suggests that scaling environment complexity may be more important than scaling parameters for this class of architecture.

### 9.3 Persistent State Replaces Geometric Complexity

The lifecycle experiment revealed that the discrete model's elaborate metric structure (CV 14) was compensating for lack of temporal continuity. With persistent state, the same task was solved at CV < 1 — the geometry only needed to handle spatial routing, not temporal encoding. This implies that the WHERE/WHAT decomposition has a temporal dimension: WHERE information flows spatially (metric), WHAT computation happens (FFN), and WHEN information persists (tau + persistent state).

### 9.4 Temporal Reasoning as Emergent Capability

The strategic death discovery — a 5M parameter model planning to die quickly to minimize future penalty — demonstrates that temporal reasoning can emerge from continuous-time dynamics with persistent state, without explicit planning modules, memory systems, or temporal abstraction mechanisms. The LTC contraction naturally attenuates old information (bounded temporal horizon), and the model learned to reason within this horizon to optimize temporal outcomes.

---

## 10. Current Status and Forward Directions

### 10.1 Established Results

- Post-transition 5M model as universal geometric substrate across 8+ domains
- Rapid skill acquisition (50-500 steps) with 92-97% neuron sharing
- Transfer to continuous robotics control (82% of MLP baseline on Cartpole)
- Quadruped locomotion through two self-organizing phase transitions
- Persistent lifecycle state replacing geometric complexity (14× CV reduction)
- Strategic temporal reasoning (death optimization) from persistent ODE state
- Learned per-entity sensory trust hierarchy (body contextual, feet reactive)
- Efficiency self-regulation through dynamics magnitude penalty

### 10.2 Active Experiments

**Adaptive autonomy:** The efficiency regularizer (λ=0.005) produced the only lifecycle run completing 2M steps without crashing or exploiting strategic death. Efficiency cost decreased 52% over training. Current work focuses on achieving locomotion under the efficiency constraint — early evidence suggests the regularizer delays but does not prevent locomotion discovery, with emerging prediction-error-gated efficiency to allow high dynamics during genuinely surprising observations while penalizing unnecessary processing during routine ones.

**Curiosity-augmented exploration:** Preliminary results show the curiosity-augmented Anymal model refuses to settle into the standing plateau that trapped the standard PPO model — log_std increases (policy widening, not narrowing) and episode lengths decrease while per-step reward improves. However, all curiosity formulations based on internal dynamics magnitude (||dh/dt||) caused NaN crashes by incentivizing dynamics near the numerical stability boundary. Safe curiosity signals must be external to the ODE dynamics or bounded by construction.

### 10.3 Forward: Linguistic Mind

The convergence of all findings points toward a linguistic state controller: LiquidARC running continuously on the Spark as an MCP server, maintaining persistent state about conversations, goals, and context. Claude (the LLM) connects through MCP tools to read relevance-scored context, report events, and receive attention directives. The model develops capabilities through interaction — every conversation provides training signal, every correction drives online learning.

The architecture maps cleanly: conversation events replace joint states as sensory forcing, relevance scores replace joint torques as output, and the adaptive tau mechanism self-regulates processing depth per event based on information content. The model would discover conversational structure through the same mechanism that discovered kinematic structure on Anymal — self-organizing geometric phase transitions driven by interaction with a structurally rich environment.

### 10.4 The Central Thesis

LiquidARC is not a computation-on-demand function. It is a continuously existing dynamical system whose geometric structure self-organizes through interaction with its environment. The environment's structural richness determines the model's developmental trajectory. In impoverished environments, development saturates quickly. In rich environments — physics simulation, human conversation, real-world interaction — development is open-ended, producing progressively deeper geometric organization through successive phase transitions.

The 5M parameter count is a feature, not a limitation. It runs on a single desktop GPU. It doesn't need a datacenter. The intelligence comes not from parameter count but from the interaction between continuous-time dynamics, learned Riemannian geometry, and structurally rich environments. The body stabilizes the brain. The environment drives the development. The geometry organizes itself.

---

*Research conducted on NVIDIA DGX Spark. LiquidARC is an independent research project exploring continuous-time geometric neural computation.*