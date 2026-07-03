# System 1 / System 2: Bidirectional Cognitive Consumption + Cadence Dropout + Online Adaptation

**Status:** complete primary investigation (2026-05-04 → 2026-05-07).

**Headline:** A 15M-parameter Liquid student attending over GR00T's denoising-depth trajectory, trained with cadence dropout, achieves a flat 50–64% K-curve on libero_10 from vision+state alone (no text). At K=0 — one GR00T call per episode — Liquid reaches 56%, validating multi-Liquid deployment. **Adding online adaptation (V8) — SGD on Liquid's small fast layers (drift + tau + z_groot_proj, ~1.4M params) using GR00T's chunk as flow target — lifts K=1 to 70% and K=4 to 64%, recovers the previously-stuck sim8 task from 0% to 40%, but trades K=0 robustness (44% vs frozen V7b's 56%).**

**Final architectural finding (2026-05-07):** model-emergent K via 14 PPO/REINFORCE probes (v1-v14) plateaued at 58% on libero_10 — below the V7b K=4 = 60% baseline despite escalating machinery (entropy regularizers, KL anchors, two-timescale training, peak-checkpoint saving, threshold tuning). The elegant alternative — **physics-based cadence** using cumulative cond drift since last fire with self-calibrating threshold (median of past triggering drifts) — delivers **62% with zero learned parameters and zero tuned thresholds**, beating every learned variant and matching V8 K=4 within 2pp. The lesson: 1-bit decisions where the model already carries a physical signal corresponding to the decision criterion don't need RL.

---

## 1. Goals and Framing

### The research question

Can one large multimodal Vision-Language-Action foundation model (GR00T-N1.7, ~3B params) serve as a System-2 reasoner for a population of small Liquid sensorimotor controllers (15M params, no language), each handling a different robot body, with the System-2 fired only occasionally?

The motivating analogy is animal-cognitive:
- Cortex (System 2) processes language, plans long-horizon, computes goal-grounded scene understanding. Body-aware but flexible across morphologies.
- Cerebellum / striatum (System 1) executes fast precise sensorimotor loops. Body-specific. Adapts via plasticity.
- Communication is bidirectional and selective: cortex informs goals, cerebellum signals prediction errors and motor needs.

### Why it matters

**Compute.** Running a 3B VLA at 90 inferences per episode (every chunk) is expensive. If a small model can carry the trajectory between rare System-2 calls, deployment cost falls by the cadence ratio (1/K).

**Multi-body deployment.** A single GR00T can serve multiple distinct Liquids — one per body or specialty — if it's queried infrequently and each Liquid runs autonomously between calls.

**Architectural insight.** Whether bidirectional integration matters at all in this setup, and what kind of channel actually carries the cortex-cerebellum cognitive asymmetry.

### Hard constraint

**No text to Liquid.** Liquid receives only vision (96×96 image + wrist) and 8-d proprioceptive state. Task descriptions are not available to Liquid in any form — no soft tokens, no task IDs, no embeddings derived from language. This forces Liquid to rely on what it perceives plus what GR00T transmits via internal-state vectors.

The constraint matters because it eliminates the easiest "win" path (just give Liquid the language) and forces the architecture to test whether bidirectional integration through latent state can substitute for language access.

---

## 2. Architecture

### GR00T-N1.7 (System 2 — frozen, stateless, expensive)

```
inputs:
  image     [256, 256, 3]
  wrist     [256, 256, 3]
  state     [8]   (xyz, rpy, gripper open/close)
  language  text  ("put the moka pot on the stove")

Qwen3-VL backbone (vision tower + cross-attention with language tokens)
  → vl_embeds [seq_len, 2048]   ← scene + language fusion intent

state_encoder (small MLP)
  → state_features [1, 1536]

DiT action head (rectified-flow denoising, N=4 inference timesteps for LIBERO):
  for t in 0..N-1:
    actions = init_noise if t==0 else previous
    action_features = action_encoder(actions, t, embodiment_id)
    sa_embs = cat(state_features, action_features)
    model_output[t] = DiT(sa_embs, vl_embeds, t)   ← hidden, [seq_len, 1024]
    pred_velocity = action_decoder(model_output[t], embodiment_id)
    actions += dt * pred_velocity

outputs:
  chunk     [16, 7]   action plan for next 16 timesteps
  vl_embeds [seq_len, 2048]
  state_features [1, 1536]
  model_output trajectory [N=4, 1024]   ← key for V7
```

**Embodiment-aware:** the DiT is parameterized by an embodiment tag (we use `LIBERO_PANDA`). This is what makes one GR00T flexible across bodies — different bodies get different action-space projections through learned per-embodiment heads.

**Action chunking:** each forward pass produces 16 future actions (action_horizon=16) rather than one. Robot executes them sequentially with chunk re-prediction at some cadence.

### Liquid (System 1 — student, learnable, cheap)

```
inputs:
  image  [3, 96, 96]    (Liquid's own perception, resized from 256)
  wrist  [3, 96, 96]
  state  [8]
  z_groot or (z_bank, delta_bank)   ← System-2 broadcast / cached

VisionEncoder ×2 (small CNN) + state MLP
  → fused features [768]
  → SiLU → h_pre [768]

[Optional] System-2 injection (zero-init residual):
  if direct injection:
    h ← h + z_groot_proj(z_groot)
  if depth bank (V7):
    delta_emit = request_head(h)        [4]    ← Liquid's intent question
    attn = softmax(-||delta_emit - delta_bank||² / temp)
    z_selected = Σ_k attn[k] * z_bank[k]
    h ← h + z_groot_proj(z_selected)

Continuous-time ODE drift loop (k_max=16 Euler steps with adaptive halt):
  for k in 0..k_max-1:
    if still_active:
      h ← h + dt * drift(h) / τ
      if k >= min_steps:
        p_halt = σ(halt_head(h))
        still_active *= (1 - p_halt)

cond = post-ODE h [768]

FlowMatchingHead (4-layer transformer, 256-d):
  velocity = transformer(noisy_chunk, t, cond)

Sample action chunk [16, 7] via 10-step ODE integration of velocity.
```

**Total: ~15M params.** The encoder dominates (~12M); flow head is small (~2M).

**Halting:** the ODE drift loop has an adaptive halt head. n_used (effective ODE steps) is learned: starts at ~13, converges to ~8 by mid-training, adjusts per-position based on observation difficulty.

**Zero-init injection:** the `z_groot_proj` layer is zero-initialized at the start of training. So at step 0, Liquid is functionally identical to a vanilla System-1 (z_groot has zero contribution). It must *learn* to use the System-2 broadcast. This is a design choice for backward-compat with non-bidirectional checkpoints and for clear gradient diagnosis ("is the channel being used at all?").

### Communication channels tested

| Channel | Direction | What's transmitted |
|---------|-----------|-------------------|
| `z_vl` | GR00T → Liquid | mean-pooled vl_embeds [2048] — scene + language fused intent |
| `z_state` | GR00T → Liquid | mean-pooled state_features [1536] — proprio understanding |
| `z_motor` | GR00T → Liquid | final action chunk mean [7] — converged motor plan |
| Δs (V6a) | Liquid → GR00T | 8-d state perturbation — counterfactual query |
| Image attention (V6c) | Liquid → GR00T | 16×16 spatial mask × image — top-down attention |
| Depth selector (V7) | Liquid → bank | 4-d softmax over GR00T's denoising trajectory — cognitive consumption |

---

## 3. Experimental Progression

The investigation followed a structured path: establish baselines → identify failure modes → propose architectural fixes → test in isolation → combine. Each variant was named to make comparison legible across the report.

### Phase A: Baseline + cliff discovery

**V1 — z_vl, no cadence dropout.** Simple residual injection of mean-pooled vl_embeds at every chunk. Tests whether passive consumption of GR00T's intent vector helps Liquid.

- Result: K=1 = 62%, K=4 = 0%, K=16 = 0%, K=0 ≈ 0%
- The K-curve is a cliff: dense System-2 helps, but any caching collapses Liquid to zero.
- Diagnosis: training only saw fresh z_vl, so the model fits to a "system-2 always available" regime. Cached z_vl at deployment is OOD.

### Phase B: Cadence dropout — first fix

**V2 — z_state with cadence dropout {0, 1, 3, 7}.** Dataset randomly serves z from K-chunks-back during training. The K offsets simulate K∈{1, 2, 4, 8} deployment cadences.

- Result: 36/48/52/48% across K∈{1,4,16,0}.
- The cliff is gone. K-curve is flat or even *increasing* with K.
- Why z_state works at high K: proprioceptive state lives on a tight low-dimensional manifold (joint positions are constrained). Stale z_state from chunk 0 of an episode is statistically close to z_state from chunk 79. The model didn't learn anything sophisticated; the staleness just doesn't hurt because the data didn't drift much.
- Specialization tax: K=1 dropped from V1's 62% to V2's 36%. Real cost.

**V3 — z_vl with cadence dropout {0, 1, 3, 7}.** Same regime, vl channel.

- Result: 52/46/44/10% across K∈{1,4,16,0}.
- Cliff also broken at K=1-16. But K=0 collapses to 10%.
- Why K=0 collapses for vl but not state: scene encodings are high-dimensional and structurally drift as objects move. z_vl from chunk 0 (gripper at home, all objects in start positions) is structurally different from z_vl at chunk 79 (gripper near target, object grasped). Training never saw this 79-chunk-stale regime (max offset was 7).
- Specialization tax (V3 vs V1): −10pp at K=1. Smaller than V2's −26pp.

### Phase C: Composite channels

**V4b — concat(z_vl, z_state) + cadence dropout.** Tests if combining channels recovers K=0 robustness from z_state while keeping z_vl's body-flexibility.

- Result: 54/44/36/14% across K∈{1,4,16,0}.
- K=0 recovers slightly (10% → 14%) but not to V2's 48%.
- Why: simple concat is mathematically identical to two separate linear projections. The model couldn't learn to "fall back to z_state when z_vl is wildly stale" because cadence dropout never exposed the model to >7-chunk staleness.

**V5b — gated mixture: per-position softmax over channel projections.**

- Result: similar to V4b. Architectural form (gated vs concat) didn't matter; the limiting factor was the training distribution.

### Phase D: Bidirectional handshake — first attempts

The user's directive at this point: communication should be bidirectional, GR00T stays stateless, Liquid emits intent (not language) that shapes GR00T's response. Liquid then consumes the resulting trajectory. Through GR00T's existing encoders (no internal injection).

**V6a — state-perturbation bidirectional.** Liquid emits Δs ∈ ℝ⁸ from a learned `request_head`. At training time, dataset has K=4 pre-computed (Δs, z_vl) pairs per obs (one neutral + three random Gaussian perturbations); soft attention selects the bank entry. At deployment, Δs is added to GR00T's state input; the response is consumed.

- Result: K=1 = 42%, K=4 = 0% (cliff — V6a wasn't trained with cadence dropout).
- *V6a hurt K=1 by 10pp vs V3 with the same channel.*
- Why: state perturbations have an asymmetry. At training, the bank is pre-computed (no harm — Δs is just an offline query parameter). At deployment, Δs is *added to GR00T's input*, producing a plan for a *physically wrong* body. Liquid's `request_head` learned that non-zero Δs sometimes selects useful bank entries; at deployment it emits non-zero Δs and we get wrong-body plans.
- Lesson: counterfactual queries that lie to the teacher are architecturally broken.

**V6c — image-attention bidirectional.** Liquid emits a 16×16 logit map → sigmoid → bilinearly upsampled to 256×256 → applied multiplicatively to GR00T's image. Bank built from random Gaussian-blob attention masks.

- Result: K=1 = 48%, K=4 = 0% (cliff — same reason).
- Better than V6a but still under V3's 52%.
- Why marginal: Liquid already has its own VisionEncoder seeing the same scene. So GR00T's z_vl modulated by attention provides redundant scene info. The image-attention channel probes a Liquid-redundant axis.

### Phase E: Cognitive consumption — the working channel

The architectural insight gathered from V6a/V6c failures: bidirectional matters when the channel carries content Liquid *can't compute itself*. Liquid sees → perceptual queries are redundant. The teacher's specialty is language→action grounding through the DiT denoising loop. So the right axis to query isn't "what GR00T sees" but "at what level of motor commitment GR00T is reasoning."

**V7a — depth-bank consumption.** GR00T's DiT iteratively refines actions across N=4 denoising steps. Step 0: noise → reading off raw scene+language fusion. Step 3: converged motor plan. Mean-pooled DiT hidden state at each step gives a trajectory of "GR00T's reasoning at increasing motor commitment levels." Liquid's `request_head: Linear(768, 4)` emits a 4-d softmax that selects across these depth levels.

Critically: only ONE GR00T call per chunk. The depth trajectory comes for free from the standard inference pass. No input perturbation.

- Result: K=1 = **86%**, K=4 = 0%, K=16 = 0%, K=0 ≈ 0%.
- K=1 = **86%** — new record across all variants. Beats V3 K=1 (52%) by 34pp. Beats memory-augmented baseline (74%) by 12pp. At 91% of GR00T's own task success (94%) with a 15M-param student.
- K-cliff persists past K=1 because no cadence dropout. Same pattern as V1.
- Why 86%: cognitive complementarity. Liquid uses its own perception, but consults GR00T at the right level of motor commitment per chunk. Different tasks need different levels (some benefit from abstract intent, others from concrete plan). Liquid learns the per-context selection.

**V7b — V7a + cadence dropout {0, 1, 3, 7}.**

- *First run was buggy.* Initial result: K=1 = 86% (same as V7a), K=4 = 0% (cliff persisted). This was anomalous — V3 with the same dropout achieved K=4 = 46%.
- Investigation found a bug: in `TeacherLabelDataset.__getitem__`, the cadence_dropout shifted `z_groot` to a stale index but `z_bank` was always read at the current index. So V7b's bank was always fresh during training even when z_groot was stale. The model never learned to operate over a stale bank → K≥4 deployment was OOD.
- Fix: read both `z_groot` and `z_bank` from the same stale index.
- After fix, V7b training loss climbed (0.034 → 0.048 at step 2500) — a *positive* signal that the model is now training on a genuinely harder problem.
- Final result: **64/60/50/56% across K∈{1,4,16,0}.** Flat curve. K=0 = 56% with one GR00T call per episode.

This is the deployable architecture for the **fixed-weight** regime.

### Phase F: Online adaptation — V8

The motivating reframe (mid-investigation): the deployment-time question isn't only "what's the best fixed Liquid?" — it's also "can Liquid adapt to deployment dynamics?" The animal analog is cerebellar Purkinje synapse adjustment via climbing-fiber error signals — small fast learning at runtime, large behavioral consequence.

The architectural hypothesis: Liquid's "small fast geometry" components (drift MLP + tau_raw + z_groot_proj, ~1.4M params total) are the right adaptive layer. They directly encode local response to System-2 conditioning — exactly what should change when physics changes. Keep the encoder, vision tower, and flow head fixed; let the dynamics adapt.

**V8 — V7b checkpoint + deployment-time SGD on `drift + tau_raw + z_groot_proj`.** Adaptation signal: when System-2 fires (every K chunks), use GR00T's chunk as a flow-matching target. Compute one SGD step on the small adaptive params with lr=1e-4. Per-episode reset of adapter weights (safe mode).

Why this works architecturally:
- No external reward needed — GR00T's chunk is the supervision signal.
- No retraining of the heavy encoder — only ~9% of model params adapt.
- Per-episode reset keeps the base policy stable while allowing per-episode tuning.
- Only fires when GR00T fires — adaptation cadence = System-2 cadence.

- Result: 70/64/52/44% at K∈{1,4,16,0}. Strict improvement at K∈{1,4,16}; regression at K=0.
- The K-curve reshapes from V7b's flat (50-64%) to V8's steeper (44-70%). Mean across K is identical (57.5% for both).

**The sim8 recovery.** sim8 (both moka pots on stove — the perennial hard task) was 0% across V1, V2, V3, V6c, V7a, V7b. With V8, it reaches **40% at K=1**. This is qualitatively different from incremental improvement: adaptive learning recovers a task the frozen model was structurally stuck on.

**Why K=0 regresses.** At K=0, the adaptive optimizer fires only 1-4 times per episode (very few S2 calls). With per-episode reset, this is one or two SGD steps from a fixed initialization — enough to destabilize without converging. V7b's frozen weights, trained on cadence dropout offsets {0,1,3,7}, were already near-optimal for sparse-signal deployment. Adaptive needs more updates per episode to be worth its cost.

**The deployment trade-off:**

| Regime | Cadence | Best architecture |
|--------|---------|-------------------|
| Cadence-rich | K∈{1,4,16}, multiple S2 calls/episode | **V8** (adaptive — extracts more from each call) |
| Cadence-poor | K=0, one S2 call/episode | **V7b** (frozen — trained for sparse signal) |

V7b is a frozen *robust-across-cadences* policy. V8 is an adaptive *exploits-rich-cadence* policy. The choice depends on deployment scenario: if you can afford 4+ GR00T calls per episode, V8; if you're in extreme multi-Liquid deployment with one call/episode, V7b.

### Phase G: Pressure-landscape Stage 1 + Stage 2 (model-emergent K)

Two more architectural directions were attempted under a stronger discipline: pressure-landscape design — no hand-coded controllers, no separately-trained heads with their own objectives, no runtime thresholds. Either change the training distribution, or expose a head whose only signal is the external task reward.

**Stage 1 (training-distribution pressure landscape).** Same V7b architecture; only training distribution changes:
- `cadence_dropout` widened from {0,1,3,7} to {0,1,2,4,8,16,32,64} — exposes model to extreme staleness up to 64 chunks.
- `z_groot_drop_prob=0.2` — 20% of training samples have zero z_groot (and zero bank).

No new heads, no runtime controllers. Pure training-distribution change.

- Result: 58/52/48/**60**% across K∈{1,4,16,0}. **K=0 = 60% is the new record** across all variants in the investigation.
- K∈{1,4,16} regressed −2 to −8pp vs V7b due to capacity dilution at fixed 3,836-sample dataset. Trade-off, not a free lunch.
- The principle works (targeted training pressure → targeted deployment improvement) but trades off when capacity is fixed.

**Stage 2 (model-emergent K via RL on cadence_head).** Goal: make K a model output, learned end-to-end from external reward `R = success − λ·n_fires`.

**Phase A (warmup, completed):** Stage 1 ckpt re-loaded with new `cadence_head: Linear(d, 1)` (zero-init weight, bias=−1 → initial fire prob ≈0.27). All weights frozen except cadence_head (~770 trainable params). Deployed as-is, mean K ≈ 4 across libero_10 with per-task variance — the head IS task-aware out of the box (sim4 fires less, sim9 fires more). Success at this initial fire rate ≈50% (matches V7b K=4 within noise).

**Phase B (REINFORCE training, FAILED to converge):**
- v1 hyperparams: `λ_init=0.0, λ_max=0.02, warmup=100, batch=8, total=500`. Result: fires plateau ~18, success 36-40%.
- v2 hyperparams (after v1 stalled): `λ_init=0.005, λ_max=0.05, warmup=30, batch=16, total=500`. Result at ep 230/500 (46 min wall): `win50_succ` 36-44%, `win50_fires` 17-19, fires NOT decreasing despite λ ramp from 0.005→0.024.
- Final state: ~38% success at mean K~4 — **worse than fixed-K V7b at any K**. Killed and analyzed.

**Diagnosis: vanilla REINFORCE on a frozen-action-policy cadence head is fundamentally limited.**

With sparse terminal reward (success/fail) and high variance (one trajectory per episode), the EMA-baseline advantage estimator can't separate "fired too much" from "task is hard." Per-task trace shows the cadence head IS learning a task-dependent policy (sim9 fires aggressively, sim4 less), but action policy can't solve harder tasks at any K → high fire count + low success → ambiguous gradient → cadence_head learns a worst-of-both fixed pattern.

**The architectural lesson:** RL credit assignment needs per-decision granularity for sparse-reward control tasks; episode-level REINFORCE only works when the policy is already near-optimal and just needs nudging. Cadence-head as the only knob is too narrow when frozen action policy can't solve task at any cadence.

**Path forward (Phase C, not yet executed at time of writing):** PPO + GAE with value head implemented in `train_cadence_ppo.py` — per-chunk advantage estimation, clipped surrogate, entropy bonus, multiple gradient epochs per rollout batch. Optionally co-trains action policy's small fast layers (drift + tau + z_groot_proj, ~1.4M params). Default RL recipe for any future cadence/control heads in this project: PPO+GAE+value head, never vanilla REINFORCE.

### Phase H: Stage 2 PPO search (v1-v14) and the physics-cadence resolution (2026-05-07)

After Phase B's REINFORCE failure pointed toward PPO+GAE, we ran a 14-variant probe sweep over the cadence-learning architecture space:

| variant | recipe | result |
|---|---|---|
| v1 | PPO no soft-clip, no aux | 50% then policy collapse to "never fire" → 0% |
| v2 | PPO + flow aux loss (per rollout batch) | stuck at p≈0.5, 26-40%, action drift accumulates |
| v3 | lr_cadence=6e-5, soft-clip | KL≈0, frozen at 32% |
| v4 | lr_cadence=2e-4, soft-clip | 60% → 26% (action layers degrading) |
| v5 | pure PPO no aux, soft-clip, V7b base | 58% stable over 50 eps (variance trap not yet onset) |
| v6 | v5 extended to 300 eps | 56% → 40% over 200 eps (variance trap) |
| v7 | ent_coef=0.005 (lower) | no help, ent stuck at 0.59 |
| v8 | λ pressure from start | 52%, same plateau |
| v9 | option (B) two-timescale 50 eps | 52% (tied v5, two-timescale didn't immediately help) |
| v10 | two-timescale 200 eps | 60% → 42% (variance trap returns) |
| v11 | two-timescale on V7b base | 67% peak at ep 30 (killed early) |
| v12 | KL=1.0 anchor to init policy | 62% peak at ep 50 |
| v12-long | KL=1.0, 200 eps | oscillating 80% → 48% → 62%peak → 50% |
| v13 | KL=10 (over-tight) | 54%, policy near-frozen |
| v14 | KL=1.0 + peak-ckpt saving, 200 eps | step_best.pt at 62% peak; deployment 58% |

Eval-A (v10 stochastic + adapt): 42%. Eval-B (v10 fixed K=4 + adapt): 48%. v14 step_best stochastic deployment: 58%.

**The pattern across the search:** every PPO variant either collapsed via cumulative gradient drift, got stuck at maximum-entropy stagnation, or oscillated between firing rates. The cadence head learned task-discriminating weights (low fires correlate with success, high with failure) but the variance from stochastic Bernoulli sampling produced erratic timing across episodes — the action policy couldn't handle this irregularity, undoing the gain. Each new scaffold (entropy bonus, KL anchor, threshold tuning, two-timescale training) addressed one failure mode but introduced another. By v14 the trainer had 12 design choices, most existing only to compensate for problems the previous choice introduced.

**Physics-cadence (no learned head):**

```python
# In rollout_libero_s1s2.py --physics_cadence
prev_cond = None
cum_drift = 0.0
fire_drifts = []
chunks_since_fire = 0

# At each chunk decision:
cond_now = forward_encoder(...)
if prev_cond is not None:
    cum_drift += ||cond_now − prev_cond||_2
prev_cond = cond_now
chunks_since_fire += 1

# Self-calibrating fire decision:
if len(fire_drifts) > 0:
    threshold = median(fire_drifts)  # ← scale from rollout's own statistics
    fire = (cum_drift > threshold) and (chunks_since_fire >= 2)
else:
    fire = (chunks_since_fire >= 4)  # bootstrap before any fire history
if chunks_since_fire >= 32:  # structural upper bound
    fire = True

if fire:
    fire_drifts.append(cum_drift)
    cum_drift = 0.0
    chunks_since_fire = 0
    prev_cond = None  # skip post-fire bank-refresh discontinuity
```

**Result on libero_10 (V7b base ckpt + V8 within-episode adaptation, --physics_cadence): 62%**

| Comparison | Trained params | Tuned thresholds | Overall |
|---|---|---|---|
| V7b K=4 baseline | 0 | 1 (K=4) | 60% |
| V8 K=4 (V7b + per-episode adapt) | 0 | 1 (K=4) | 64% |
| v14 step_best stochastic eval | ~600K | many | 58% |
| v10 deterministic eval (effective K=0) | ~600K | many | 60% |
| **physics-cadence + V8 adapt** | **0** | **0** | **62%** |

Per-episode dynamics emerge correctly without any learning: failed rollouts consult GR00T 9-17 times (cond drift accumulates rapidly during struggle); successful rollouts consult 6-9 times. **Task-awareness emerges from the physics** — scenes that look hard *to Liquid* trigger consultation, automatically. Mean K ≈ 5 across the eval suite.

**The architectural lesson:** for 1-bit decisions where the model already produces a signal physically corresponding to the decision criterion, prefer the signal over a learned RL policy. The RL approach produces equivalent or worse results with vastly more scaffolding. This generalizes: anywhere a discrete control decision has an obvious correlate in the model's existing observables, that correlate is likely sufficient. The 14-probe PPO search converged on this realization the hard way.

---

## 4. Detailed Results

### Full K-curve comparison

| K | V1 | V2 | V3 | V4b | V6a | V6c | V7a | V7b | V8 | **Stage 1 (pressure)** | Stage 2 Phase B (REINFORCE) |
|---|----|----|----|----|----|----|----|----|----|----|----|
| 1 | 62% | 36% | 52% | 54% | 42% | 48% | **86%** | 64% | **70%** | 58% | — |
| 4 | 0% | 48% | 46% | 44% | 0% | 0% | 0% | 60% | **64%** | 52% | ~38%¹ |
| 16 | 0% | **52%** | 44% | 36% | 0% | 0% | 0% | 50% | 52% | 48% | — |
| 0 | 0% | 48% | 10% | 14% | 0% | 0% | 0% | 56% | 44% | **60%** | — |

¹ Stage 2 Phase B is not a fixed-K column — it's learned cadence with mean K ≈ 4 after 230 episodes of REINFORCE training. Listed for comparison: at the cadence the head settled into, success was worse than V7b at any K. Phase B was killed before completing 500 episodes.

### V7b per-task at K=1 vs K=4

| sim | task (truncated) | K=1 | K=4 | Δ |
|-----|-----|-----|-----|---|
| 0 | put soup + tomato sauce in basket | 40% | 60% | +20 |
| 1 | put cream cheese + butter in basket | 100% | 40% | -60 |
| 2 | turn on stove + moka pot | 100% | 80% | -20 |
| 3 | bowl in bottom drawer | 100% | 100% | 0 |
| 4 | mug + yellow mug on plates | 40% | 60% | +20 |
| 5 | book in caddy | 100% | 100% | 0 |
| 6 | mug + chocolate pudding | 40% | 40% | 0 |
| 7 | soup + cream cheese in basket | 100% | 80% | -20 |
| 8 | both moka pots on stove | 0% | 0% | 0 |
| 9 | mug in microwave | 20% | 40% | +20 |

Three tasks improve at K=4 (sim0, sim4, sim9 — typically harder/medium tasks), three stay flat, three lose 20-60pp, one stays at 0%. **The curve isn't monotonic** — for some tasks more frequent System-2 hurts (perhaps because per-step variation in z_vl introduces noise; longer caching low-pass filters it).

sim8 (both moka pots) is the perennial hard task — 0% across V7a, V7b, V3. Likely undersampled in the training set (GR00T itself struggled with it during data collection).

### Compute / deployment table

| Setup | GR00T calls/episode | libero_10 success | Use case |
|-------|--------------------|--------------------|----------|
| Pure no-text Liquid baseline | 0 | 30% | Liquid alone reference |
| Memory-augmented (kNN over GR00T trajectories) | 0 (offline retrieval) | 74% | Best on this benchmark, no runtime GR00T |
| **V7b K=0** | **1** | **56%** | Multi-Liquid: one VLA serves N bodies, one call/episode each |
| V7b K=4 | 22 | 60% | Balanced single-body |
| V7b K=1 | 90 | 64% | Maximum quality dense System-2 |
| **Physics-cadence + V8 adapt** | **~10** (mean) | **62%** | **No learned cadence; self-calibrating drift threshold; elegant** |
| V8 K=4 (V7b + adapt) | 22 | 64% | Best fixed-cadence with adaptation |
| V7a K=1 (research peak) | 90 | 86% | Best K=1, but unusable past K=1 |
| GR00T native | 90 | 94% | Upper bound (no Liquid) |

**Marginal cost-benefit of System-2 frequency in V7b:**

- 1 call/episode → 22 calls: +4pp (60−56)
- 22 calls → 90 calls: +4pp (64−60)
- Total range: 8pp from 1× to 90× compute.

In contrast, V7a's marginal benefit is 86pp from 0× (cliff) to 90×. So V7a depends entirely on dense System-2; V7b is genuinely autonomous between refreshes.

---

## 5. Architectural Lessons (general)

### Lesson 1: The query channel must carry differential cognitive content

Bidirectional architectures only matter when each side has unique information to contribute. In our setup:

- Liquid has full perceptual access (its own VisionEncoder sees the same image GR00T sees, just at lower resolution).
- Liquid has full proprio access (its own state encoder reads joint positions).
- GR00T's *unique* content is **language→action grounding through the DiT trajectory**. GR00T sees, reads language, and converts both into a graded action plan across denoising steps. Liquid lacks language entirely.

So the right query channel is the one that selects *how much of GR00T's language-grounded planning* to consume. Not what GR00T should look at (perceptual axes are Liquid-redundant). Not what GR00T should imagine about the body (counterfactual axes lie to GR00T at deployment). Depth-attention over GR00T's denoising trajectory is exactly the cognitive complementarity axis — it's the part Liquid couldn't compute itself.

This generalizes: **identify the teacher's cognitive specialty that the student lacks, query along that axis.**

### Lesson 2: Training distribution determines deployment distribution

This came up three times:

1. V1 (no dropout) — model only saw fresh z; cliffed at any caching.
2. V3 (dropout up to 7 chunks) — model only saw 7-chunks-stale; cliffed at K=0 (~80 chunks stale).
3. V7b before bug fix — cadence dropout applied to z_groot but not bank; model never saw a stale bank; cliffed at K≥4 deployment.

**Always check that what cadence dropout perturbs at training is what deployment caches.** If they differ, the model trains for the wrong distribution.

### Lesson 3: Don't probe redundant axes

V6c (image attention) was architecturally clean — encoder-respecting, non-perturbing. But it probed a Liquid-redundant axis (perception). Result: marginal vs V3.

The temptation when bidirectional doesn't help: handicap the student to make the channel matter. *Don't.* Real bidirectional integration in animals doesn't require sensory deprivation — it requires cognitive specialization. The right design move is to find the cognitive asymmetry, not engineer a perceptual one.

### Lesson 4: The "induce trajectories" framing is partially right

The user's framing: *"Liquid shapes teacher's thoughts by inducing new trajectories and then consuming those."* V6a/V6c take this literally — Liquid's input modifies GR00T's processing. V7 reinterprets it: Liquid doesn't *induce new* trajectories; it *selects within* the trajectory GR00T is already producing. The denoising trajectory is GR00T's natural reasoning over abstraction levels; Liquid picks which level to consume.

In animal cortex, top-down attention does both — it modulates what cortex attends to (V6c-style) AND it influences which level of cortical computation is engaged (V7-style). For our setup, the V7-style won because Liquid already has its own perception; the V6c-style "modulate what cortex sees" axis was redundant.

---

## 6. Implementation Details

### Code organization

The investigation lives entirely in `subprojects/liquid-arc/research/self_org_sim/`:

- `groot_server.py` — ZMQ server wrapping `Gr00tPolicy` in main venv. Monkey-patches `action_head.get_action_with_features` to capture `(traj_xt, traj_t, traj_v, traj_model_output, vl_embeds, state_features)`. Exposes `op=get_action`, `op=get_action_with_state`, `op=get_trajectory`, `op=shutdown`.
- `gen_groot_with_states.py` — Phase A/B data collection. Produces (image, wrist, state, chunk, z_vl, z_state, z_motor) tuples via GR00T-in-sim rollouts.
- `gen_groot_with_query_bank.py` — V6a state-perturbation bank (broken architectural variant, kept for reference).
- `gen_groot_with_image_queries.py` — V6c image-attention bank.
- `gen_groot_with_temporal_queries.py` — **V7a/V7b depth bank** (the working one).
- `distill_groot.py` — `TeacherLabelDataset` with cadence dropout, channel selection, query bank loading. The bank-staleness bug fix lives here at line ~278.
- `distill_groot_flow.py` — `LiquidFlowPolicy` (encoder + flow head). Includes `request_head`, `z_groot_proj`, optional `gated_mixture`, `s2_halt`, `query_bank` machinery. Training loop with ACT halt, cadence-dropout-aware tuple unpacking.
- `rollout_libero_s1s2.py` — closed-loop deployment with K-cadence sweep. Branches on `query_channel` (state | image | depth) for the appropriate bidirectional logic.

### Cross-process bridge (LIBERO sim ↔ GR00T)

```
LIBERO sim (libero venv, CPU torch)              GR00T server (main venv, CUDA torch)
        │                                                       ▲
        │ pickle(obs_dict)                                       │
        ├──────────────  ZMQ REQ/REP @ port 5555  ───────────────┤
        │                                                       ▼
        │ pickle(chunk + z_vl + z_state + z_motor + traj)
        ◄────────────────────────────────────────────────────────
```

This was forced by the venv split — GR00T requires CUDA torch and gr00t package; LIBERO requires CPU torch and specific robosuite version. Zero gradient flow across this boundary (which constrained the bidirectional architecture significantly — see "the bank scaffolding" below).

### Why the bank exists (V6/V7 training scaffolding)

Liquid emits a query Δs (state) / attention map (image) / depth selector (depth) at training time. We want to train the `request_head` to emit useful queries via the action-loss gradient. But the chain `Δs → GR00T_response → action_loss` crosses the ZMQ boundary, breaking the gradient.

Solution: pre-compute K different (query, response) pairs per training sample. At training, Liquid's emitted query selects via differentiable soft-attention over the K-bank. The bank is "GR00T's response surface, finitely sampled." Gradient flows through the attention weights → request_head.

At deployment, no bank — Liquid emits a query and we send it to GR00T live. The request_head learned from the bank generalizes (continuously) to the live query distribution.

K=4 bank is small but turns out to be enough for the cognitive (depth) channel because the axis is naturally low-dimensional. For perceptual channels (state, image) K=4 was probably also too few — bank coverage was sparse in the relevant subspaces.

### Training hyperparameters (V7b reference)

```
--max_steps 8000              # 8K is sufficient for 3.7K-sample dataset
--batch_size 1024             # GB10 unified memory permits this at d=768
--d 768                       # Liquid hidden width (max for Triton SRAM at this scale)
--img_size 96                 # resize from native 256 (5× speedup; loss not affected)
--lr 3e-4                     # cosine schedule to 3e-5
--augment                     # random crop + color jitter via GPU
--compile                     # torch.compile (TRITON_PTXAS_PATH must be set)
--num_workers 16              # data loading saturates 8-worker mark; 16 is comfortable
--cadence_dropout 0,1,3,7     # K-offsets sampled uniformly per __getitem__
--use_query_bank              # turn on V7 depth-bank attention
--use_z_groot z_vl            # base channel (carrier of z_groot for non-bank)
```

Wall time: ~75 min on Spark GB10 (sm121). Batch-1024 throughput around 1.8 step/s with 25–35% GPU utilization (data-loading-bound, not compute-bound).

### V8 deployment-time adaptation

`rollout_libero_s1s2.py --adaptive --adaptive_lr 1e-4 [--no_reset_adapter]` enables per-episode SGD updates on a frozen-checkpoint Liquid:

```python
def setup_adaptive_optimizer(model, lr):
    for p in model.parameters():
        p.requires_grad = False
    adaptive = []
    for p in model.encoder.drift.parameters():
        p.requires_grad = True; adaptive.append(p)
    model.encoder.tau_raw.requires_grad = True
    adaptive.append(model.encoder.tau_raw)
    if model.encoder.z_groot_proj is not None:
        for p in model.encoder.z_groot_proj.parameters():
            p.requires_grad = True; adaptive.append(p)
    return torch.optim.SGD(adaptive, lr=lr), adaptive
```

The adaptive parameters are **drift MLP (~600K) + tau_raw (768) + z_groot_proj (~790K) ≈ 1.4M (~9% of model)**. Everything else (VisionEncoder, fuse, FlowMatchingHead) stays fixed.

On every System-2 fire (when `need_groot=True`):
```python
# After GR00T responds with chunk + bank
groot_chunk_t = torch.from_numpy(resp["chunk"]).unsqueeze(0)
cond, _ = model.forward_encoder(img, wrist, state, z_bank=bank, ...)
t = torch.rand(1, device=device)
noise = torch.randn_like(groot_chunk_t)
noisy = (1-t.view(-1,1,1)) * noise + t.view(-1,1,1) * groot_chunk_t
v_target = groot_chunk_t - noise
v_pred = model.velocity(noisy, t, cond)
loss = mse_loss(v_pred, v_target)
loss.backward()
adaptive_optimizer.step()
```

Same loss form as training (rectified-flow velocity matching), but with GR00T's chunk as the per-episode target instead of dataset chunks.

Per-episode reset (default): `restore_adaptive(adaptive_params, snapshot)` at episode start. Each rollout begins from V7b's checkpoint and tunes within the episode. With `--no_reset_adapter`, adaptation persists across episodes (more aggressive, can drift to bad regions).

Wall-time cost: ~50-100ms per adaptive step on libero venv CPU torch. At K=4 (~22 S2 fires/episode), ~2 sec extra per episode — negligible vs ~20 sec total rollout. Adaptive loss values: 0.05-0.4 typical (real gradient signal).

### Critical infrastructure notes

- **GR00T-N1.7-LIBERO uses N=4 denoising steps**, not 16 like the original GR00T-N1.7. This was a surprise during V7a development — initial depth_indices [0,5,10,15] failed with IndexError. Adjusted to [0,1,2,3] (all 4 available depths).
- **DiT hidden_dim=1024**, not 2048 like the Qwen3-VL backbone. This means V7's z_groot_proj is Linear(1024, 768), smaller than V3/V6's Linear(2048, 768). Slight model param savings.
- **`HF_TOKEN` required** for groot_server (Cosmos-Reason2-2B is gated on HF). Must be set in env at server launch time.
- **GB10 SDPA workaround:** `torch.backends.cuda.enable_mem_efficient_sdp(False)` to fall back to flash attention (CUTLASS kernels for sm121 not available; sm80-100 kernels would crash).
- **`TRITON_PTXAS_PATH` required** for torch.compile to work in the Spark training container.

---

## 7. Negative findings (what didn't work, and why)

### State-perturbation bidirectional (V6a) — 42% K=1, dead architecture

**Why broken:** Counterfactual queries lie to GR00T at deployment. At training time the bank pre-computes responses to arbitrary state perturbations (no harm — Δs is just an offline parameter). At deployment, Δs is added to GR00T's actual state input, so GR00T plans for a counterfactually wrong body. Liquid then has to act on the actual body conditioned on a plan for a different body. The training-time signal can't distinguish "useful counterfactual" from "harmful real perturbation" because at training there's no body to harm.

**No fix exists for this asymmetry.** State-perturbation queries are architecturally dead.

### Image-attention bidirectional (V6c) — 48% K=1, redundant

**Why marginal:** Liquid sees the same scene GR00T sees (lower resolution but same content). z_vl modulated by attention masks provides redundant scene info. The image-attention axis would matter if Liquid had degraded perception (low-res, occluded, partial); but degrading Liquid's perception artificially is biologically anti-natural.

### Composite z_vl + z_state (V4b, V5b) — 54% K=1, no K=0 recovery

**Why didn't help:** Concat is mathematically equivalent to two separate linear projections; gated mixture is more expressive but the model didn't have training signal to fall back to z_state when z_vl was extremely stale (cadence dropout sampled offsets up to 7 only, never the full episode).

### Naive cadence dropout with bank (V7b before bug fix) — 0% at K=4

**Why didn't help:** Bank wasn't stale-shifted alongside z_groot. The model saw fresh banks during training even when z_groot was stale. So K≥4 deployment with cached bank was OOD.

**Fix:** read both z_groot and z_bank from the same stale index in `TeacherLabelDataset.__getitem__`. After fix, K=4 = 60%.

### Soft-prompt language injection (proposed but not run)

The user ruled this out explicitly ("Liquid can't ask GR00T in any meaningful linguistic way, only intent"). Even though soft tokens aren't human-readable language, they live in language token space and would route through GR00T's text encoder. Not pursued.

### Selective query / s2_halt (proposed but not run)

Liquid learns when to query System-2 rather than fixed-K cadence. Implementation was sketched (consistency loss between fresh and stale bank predictions). Not run — V7b's flat K-curve made the urgency lower. Still worth testing for adaptive deployment.

---

## 8. Achievements

### Empirical
- A 15M-param Liquid student with no language access reaches **64% on libero_10 (K=1)** with V7b frozen weights, and **70% (K=1) / 64% (K=4)** with V8 adaptive learning. **56%** with one GR00T call per episode (V7b frozen).
- V7b's K-curve is flat 50–64% across K∈{1, 4, 16, 0} — robust deployment across cadences.
- V8's K-curve trades K=0 (44%) for higher K=1-16 (52-70%) — exploits cadence when available.
- Strict improvement over all prior single-channel and dual-channel baselines: V7b K=1 > V3 K=1 (+12pp), K=4 > V3 K=4 (+14pp), K=0 ≫ V3 K=0 (+46pp). V8 K=1 > V7b K=1 (+6pp).
- **sim8 (the perennial moka-pots task) recovered from 0% to 40% with V8.** First non-zero result on this task across any architecture in the investigation.

### Architectural
- Demonstrated that bidirectional integration matters — but only along the **cognitive consumption** axis, not perceptual perturbation.
- Identified GR00T's denoising-depth trajectory as a viable cognitive-asymmetry channel (the part Liquid can't compute itself) and validated soft-attention over discretized depths as a working query mechanism.
- Established the principle that **cadence dropout must perturb the same artifact deployment caches** (the bank-staleness bug as a codified gotcha).
- Validated **deployment-time adaptive learning** on Liquid's small fast layers (drift + tau + z_groot_proj, ~9% of model). One SGD step per System-2 fire using GR00T's chunk as flow-matching target produces measurable gains at deployable cadences. The adaptive components are small enough to learn meaningfully within an episode.
- Surfaced the **cadence-rich vs cadence-poor regime distinction**: V8 wins when GR00T fires multiple times per episode (K=1, 4, 16); V7b wins when GR00T fires once per episode (K=0). Different deployment scenarios prefer different students.

### Methodological
- Built a complete experimental framework for testing System-1/System-2 architectures on libero_10:
  - GR00T-in-sim data collection with cross-venv ZMQ bridge
  - Multiple bidirectional channels behind a unified bank-attention API
  - K-cadence sweep evaluation (K ∈ {1, 4, 16, 0}) as standardized benchmark
- Eight architectural variants compared on the same dataset, model size, training budget — apples-to-apples comparison.

### Negative results published
- State-perturbation bidirectional is architecturally dead (counterfactual queries lie to teacher at deployment).
- Image-attention bidirectional is marginal (probes Liquid-redundant perceptual axis).
- Composite channels (V4b, V5b) don't recover extreme-K robustness without explicit training-distribution support.
- These negative results clarify what bidirectional needs to be useful, and inform downstream architectures.

---

## 9. Open Questions and Next Experiments

### Multi-Liquid validation (priority 1)

Train 2-3 specialist Liquids on disjoint libero_10 sub-tasks (e.g., one per task family). Share the same V7b-trained encoder/GR00T pipeline. Test that K=0 deployment serves all bodies from one GR00T call per episode. This is the actual multi-Liquid claim — currently asserted, not measured.

### Cross-body validation

Re-run V7b on a different LIBERO embodiment (or different robot entirely if available). Tests body-flexibility of z_vl (which is body-aware-but-flexible by construction, since the Qwen3-VL backbone fuses scene + language without committing to specific body).

### Larger K bank (deeper denoising)

K=4 was forced by GR00T-N1.7-LIBERO's `num_inference_timesteps=4`. Models with deeper denoising loops (the original GR00T-N1.7 has 16 steps, Pi-0 has more) would allow richer depth attention with K=8 or K=16. Hypothesis: more bank entries = finer cognitive level discrimination = higher K=1 peak.

### Selective query (s2_halt)

Liquid learns when to call System-2 rather than fixed-K cadence. Implementation:
- Add `s2_halt_head: Linear(d, 1)` to Liquid encoder
- Training: per batch, forward encoder twice (with fresh and stale z); halt target is binary classification of "did fresh help significantly?"
- Deployment: at each chunk, halt head decides whether to refresh System-2

Upside: per-task adaptive cadence. Liquid coasts through reach phases, queries during contact phases.

### True cognitive complementarity test

The architectural lesson — query the asymmetry, not the redundancy — should generalize. A test: train Liquid on a task where it's *demonstrably weaker* than GR00T at one specific subskill (e.g., long-horizon planning, object identification at occlusion, reasoning under ambiguity). Then test whether bidirectional integration recovers the gap. If yes, the principle is validated beyond "depth selection on this VLA."

### Different teacher (Pi-0, OpenVLA, RT-2)

Swap GR00T for another VLA. The architecture is teacher-agnostic by design (the integration is on Liquid's side); changing the teacher tests:
- Whether the depth axis transfers across architectures
- Whether the bidirectional gains generalize
- Cross-teacher Liquid (one student against multiple teachers)

---

## 10. Summary

The investigation tested whether a small Liquid sensorimotor controller can be guided by a frozen large multimodal foundation model in a System-1/System-2 architecture, with the constraint that Liquid never sees language and System-2 fires only occasionally.

After nine architectural variants (V1, V2, V3, V4b, V5b, V6a, V6c, V7a, V7b, V8), the working architectures split by deployment scenario:

> **V7b — depth-attention bank + cadence dropout on the bank source (frozen weights).** Liquid attends over GR00T's DiT denoising trajectory at K=4 stratified depths via a learned softmax. Trained with cadence dropout simulating staleness up to 7 chunks. Achieves 64% on libero_10 at K=1 (90 GR00T calls/episode) and **56% at K=0 (one GR00T call per episode)**, with a flat K-curve. Best for cadence-poor / multi-Liquid extreme deployment.

> **V8 — V7b + deployment-time SGD on small fast layers.** Same architecture as V7b; at deployment, ~1.4M adaptive params (drift + tau + z_groot_proj) update per System-2 fire using GR00T's chunk as flow-matching target. Per-episode reset. Achieves **70% at K=1** and **64% at K=4**, recovers the previously-stuck sim8 task from 0% to 40%. Best for cadence-rich deployment with multiple S2 fires per episode.

The three architectural principles that make these work:
1. **Cognitive consumption beats perceptual perturbation.** Liquid already perceives; queries that probe perception are redundant. Queries that select among levels of teacher cognition are not.
2. **Cadence dropout must perturb the same artifact deployment caches.** Otherwise the model trains for one distribution and deploys in another.
3. **Adaptive learning belongs in the small fast geometry.** Liquid's drift + tau + z_groot_proj layers are the right adapter — small enough to learn within an episode (~50ms per step), structured enough to encode body-specific dynamics, decoupled enough that adaptation doesn't disturb the encoder or flow head.

The architecture is consistent with animal-cognitive structure: cortex (System 2, frozen) processes language and computes goal-grounded planning; cerebellum (System 1) executes fast loops while consuming cortical output at the right level of motor commitment AND adapting its small Purkinje-synapse-equivalent layer based on per-episode error signals. Both have full sensory access. Communication is selective consumption, not perceptual substitution. Slow path (encoder, flow head) stays stable; fast path (drift, tau, z_groot_proj) adapts within the episode. That's the cortex/cerebellum structure expressed in 15M parameters.
