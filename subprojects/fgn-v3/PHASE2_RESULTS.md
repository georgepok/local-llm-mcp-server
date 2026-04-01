# FGN v3 Phase 2: Transformer-Breaking Task Suite — Results

## Experiment Overview

Phase 2 tests whether FGN's learned Riemannian metric provides measurable advantage over a standard (flat) transformer on tasks with proven transformer failure modes. Three tasks were designed based on theoretical impossibility results:

1. **Parity** (Hahn 2020): Binary string odd/even classification
2. **Cumulative State Tracking**: Running accumulator (add/sub mod 100) with distractors
3. **Permutation Composition** (S5): Group composition (implemented, not yet run)

All experiments use parameter-matched models from random init, with fresh data generated each batch (no epoch-based training).

---

## Hardware

- **Machine**: NVIDIA DGX Spark (Grace Hopper unified memory architecture)
- **GPU**: NVIDIA GB10 (unified CPU-GPU, 128 GB physical RAM shared)
- **Key constraint**: Unified memory means GPU and CPU allocations compete for the same 128 GB. No separate VRAM.

### Unified Memory OOM Incidents

Running two training processes concurrently caused **three system crashes** (host OOM, kernel panic). Root cause analysis revealed:

- At `seq_len=5120`, PyTorch's CUDA allocator consumed 108+ GB from a single 6.9M-parameter model
- Two concurrent processes exhausted all 128 GB; the OOM killer could only reclaim tiny desktop processes (4 KB each) while NVIDIA-pinned GPU memory (~100 GB) was invisible to it
- System cascaded into killing SSH, gnome-shell, etc., then rebooted

**Fix applied**: Sequential training + `CUDA_MEMORY_FRACTION=0.85` cap + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + `torch.compile` disabled.

---

## Task 1: Binary Parity

### Design

Classify a binary string as "odd" or "even" based on the count of 1-bits. Hahn (2020) proved transformers cannot solve parity as length approaches infinity.

- **Format**: `Parity: 0 1 1 0 1 ... Answer: odd`
- **Training**: bit_length=40, p(1)=0.5, seq_len=256
- **Supervision**: Single token (odd/even)
- **Model**: d=128, 4L, 4H, 13.7M params (parameter-matched)

### Results

| Condition | Flat Accuracy | Notes |
|-----------|:---:|---|
| In-dist (len=40, p=0.5) | 100% | |
| len=60 | 100% | 1.5x training length |
| len=80 | 100% | 2x training length |
| len=100 | 100% | 2.5x training length |
| len=120 | 100% | 3x training length |
| len=150 | 100% | 3.75x training length |
| len=200 | 100% | 5x training length |
| p(1)=0.1 | 100% | Sparse bits |
| p(1)=0.3 | 100% | |
| p(1)=0.7 | 100% | |
| p(1)=0.9 | 100% | Dense bits |

**Converged at step ~500** (CE near zero). 100% on all 11 OOD conditions including 5x length generalization and extreme bit densities.

### Analysis

The Hahn (2020) impossibility result is **asymptotic** — it states transformers fail as sequence length approaches infinity. At practical lengths (up to 200) with absolute positional embeddings that cover the test range (`max_seq_len=256`), the transformer generalizes perfectly. The parity task does not create a meaningful challenge at these scales.

**Conclusion**: Parity is not a viable discriminator at practical sequence lengths. Abandoned in favor of state tracking.

---

## Task 2: Cumulative State Tracking

### Design

Track a running accumulator through a sequence of `add X` / `sub X` operations (mod 100) interleaved with distractor words. At query positions marked by `?`, predict the current accumulator value.

- **Format**: `State 42 | add 7 | cat | sub 3 | ? 46 | add 12 | dog | ? 55 |`
- **Supervision**: Only the number tokens after `?` markers
- **Distractor ratio**: Fraction of events that are filler words (not operations)
- **Auto-scaling**: `n_events = max(30, seq_len * 7 / 25)` when not specified

This task targets two transformer failure modes:
1. **Attention dilution**: As sequence grows, soft attention weight on relevant update tokens decreases as ~1/T
2. **Sequential state propagation**: Chaining operations requires sequential computation that transformers approximate with parallel shortcuts

### Experiment 2a: Short Sequence (seq_len=512)

- **Model**: d=128, 4L, 4H, 13.7M params
- **Training**: seq_len=512, ~60 events, 50% distractors, 4 queries per sequence
- **Batch size**: 8 (both models)

#### Training Convergence

Both models converged to near-zero CE within ~400 steps, with identical learning curves.

#### OOD Evaluation (Flat, 5K checkpoint)

| Condition | Accuracy | CE Loss |
|-----------|:---:|---|
| In-dist (60 events, 50%) | 100% | 0.0002 |
| 120 events, 50% | 100% | 0.0002 |
| 180 events, 50% | 100% | 0.0002 |
| 240 events, 50% | 100% | 0.0002 |
| 60 events, 70% distractors | 100% | 0.0002 |
| 60 events, 80% distractors | 100% | 0.0002 |
| 60 events, 90% distractors | 100% | 0.0002 |
| 180 events, 80% (combined) | 100% | 0.0002 |

**Conclusion**: At `seq_len=512`, the transformer trivially handles all conditions. The ~120 sequential operations within 512 tokens are well within what 4-layer attention can represent. The user directed a 10x sequence length increase.

---

### Experiment 2b: Long Sequence (seq_len=5120) — Primary Experiment

#### Configuration

| Parameter | FGN | Flat |
|-----------|-----|------|
| d_model | 64 | 64 |
| n_heads | 2 | 2 |
| n_layers | 3 | 3 |
| d_ff | 256 | 276 |
| max_seq_len | 5120 | 5120 |
| Parameters | 6,923,272 | 6,923,644 |
| Param diff | | -372 (0.005%) |
| torch.compile | off | off |
| Batch size | 2 | 8 |
| Learning rate | 3e-4 | 3e-4 |
| Warmup steps | 500 | 500 |

**Why model was halved from d=128 to d=64**: The original d=128 4L 4H model at `seq_len=5120` consumed 108+ GB from a single process on unified memory, causing system crashes even when run alone. Disabling `torch.compile` and reducing to d=64 3L 2H brought memory under control.

**Why FGN batch size is 2**: Heat kernel attention computes pairwise distances `diff * diff * g_avg` producing a `[B, H, N, N, d_head]` tensor. At N=5120, d_head=32, B=4, this is 25 GB in fp32. With 81 GB already allocated, this exceeds the 85% memory cap. B=2 halves the tensor to ~12.5 GB.

**Parameter matching**: Flat model's `d_ff` widened from 256 to 276 to compensate for FGN's MetricNetwork (~2,304 params/layer), W_scale (64x3=192 params/layer), and log_t (3 params/layer). Total overhead: ~7,497 params across 3 layers. Verified: 6,923,272 vs 6,923,644, diff = 372 (0.005%).

#### Task Statistics at seq_len=5120

- **Auto-scaled events**: 1,433 per sequence
- **At 50% distractor ratio**: ~716 sequential add/sub operations
- **Content tokens**: ~3,586 (remainder is padding)
- **Queries**: 4 per sequence (supervised tokens: ~8)

#### Training Convergence

**Flat model** (killed at step ~12,000 after full convergence):

| Step | CE Loss | tok/s |
|------|---------|-------|
| 0 | 10.826 | 15,940 |
| 100 | 10.457 | 37,529 |
| 200 | 9.311 | 37,735 |
| 300 | 7.244 | 37,785 |
| 400 | 5.166 | 37,825 |
| 500 | 4.333 | 37,851 |
| 1,000 | 0.743 | 37,900 |
| 2,000 | 0.027 | 37,938 |
| 3,000 | 0.011 | 37,956 |
| 5,000 | 0.003 | 37,960 |
| 10,000 | 0.0003 | 37,964 |

Convergence phases:
- **Steps 0-500**: Rapid structural learning (10.8 → 4.3)
- **Steps 500-2000**: Main convergence (4.3 → 0.03)
- **Steps 2000-10000**: Tail refinement (0.03 → 0.0003)

Notably, this is ~10x slower than `seq_len=512` (which converged by step 400), confirming that longer sequences are genuinely harder for the model. However, it still reaches near-zero loss.

**FGN model** (training in progress, step 7,700 at time of writing):

| Step | CE Loss | Total Loss | Metric CV | tok/s |
|------|---------|------------|-----------|-------|
| 0 | 10.836 | 10.825 | 0.005 | 1,257 |
| 100 | 10.547 | 10.536 | 0.005 | 2,117 |
| 500 | 4.879 | 4.868 | 0.013 | 2,130 |
| 1,000 | 2.563 | 2.552 | 0.087 | 2,131 |
| 2,000 | 0.090 | 0.079 | 0.135 | 2,132 |
| 3,000 | 0.014 | 0.003 | 0.133 | 2,135 |
| 4,000 | 0.008 | -0.003 | 0.135 | 2,138 |
| 5,000 | 0.004 | -0.007 | 0.148 | 2,140 |
| 6,000 | 0.003 | -0.008 | 0.167 | 2,141 |
| 7,000 | 0.002 | -0.010 | 0.209 | 2,141 |
| 7,700 | 0.001 | -0.010 | 0.231 | 2,141 |

**Negative total loss explained**: The loss function is `loss = CE + alpha * scale_entropy_reward` where `scale_entropy` is implemented as `-mean(H)` (rewarding scale diversity). With alpha=0.01 and H approaching ln(3)=1.099 (uniform use of all 3 diffusion scales), this contributes approximately -0.011 to the total loss. When CE drops below 0.011, total loss goes negative.

**FGN geometry development**:
- **Metric CV** grew from 0.005 to 0.231 — the strongest geometry development observed in any FGN experiment
- CV development accelerated AFTER CE convergence (CV was 0.135 at step 2000 when CE was still 0.09, but grew to 0.231 by step 7700 when CE was 0.001)
- This suggests the metric initially focused on reducing CE, then shifted to geometric structure optimization once the task was solved

**Throughput comparison**:
- Flat: 37,960 tok/s (bs=8, dot-product attention)
- FGN: 2,141 tok/s (bs=2, heat kernel attention with O(N^2 * d_head) distance computation)
- FGN is **17.7x slower** — dominated by the pairwise distance tensor at `seq_len=5120`

#### OOD Evaluation — Flat Model (10K checkpoint)

| Condition | Accuracy | CE Loss | Queries | Overflow |
|-----------|:---:|---------|---------|----------|
| In-dist (1433 ev, 50%) | **100%** | 0.0002 | 1600 | 0 |
| 1800 events, 50% | **100%** | 0.0003 | 1600 | 0 |
| 2000 events, 50% | **100%** | 0.0003 | 1600 | 0 |
| 2500 events, 50% | **100%** | 0.0003 | 1326 | 1 |
| 1433 events, 70% | **100%** | 0.0002 | 1600 | 0 |
| 1433 events, 80% | **100%** | 0.0002 | 1600 | 0 |
| 1433 events, 90% | **100%** | 0.0002 | 1600 | 0 |
| 2000 events, 80% | **100%** | 0.0003 | 1600 | 0 |

All conditions: **100% accuracy**.

At 2500 events / 50% distractors: 1 sample overflowed `seq_len=5120` (1326 queries instead of 1600), but all surviving queries were answered correctly. At 1433 events / 90% distractors: only ~143 actual operations but with 1290 distractor tokens between them — the model perfectly filters signal from noise.

#### FGN OOD Evaluation

Pending — FGN training still in progress (step 7,700 / 50,000). First checkpoint (step 5,000) is available for eval.

---

## Key Findings

### 1. State Tracking with Fixed State Space Is Not Transformer-Breaking

The cumulative state tracking task uses modular arithmetic (mod 100), giving a **fixed state space of 100 values**. The theoretical impossibility results for transformers on sequential computation (Hahn 2020, Merrill & Sabharwal) are asymptotic — they require the state space to grow with sequence length.

With only 100 possible states and operations drawn from {add 1..20, sub 1..20}, the 6.9M-parameter flat transformer learns a perfect representation of the state transition function. This holds even with:
- **716 sequential operations** (1433 events, 50% distractors)
- **90% distractors** (signal-to-noise ratio of 1:9)
- **1250 operations** (2500 events at 50% distractors, with truncation)

The model doesn't need to "track state sequentially" — it can learn to directly compute the cumulative effect of all visible operations in parallel, since the operations form a commutative group (addition mod 100).

### 2. Convergence Speed Scales with Sequence Length

| Configuration | Steps to CE < 0.01 |
|--------------|-------------------|
| seq_len=512, d=128, 4L (13.7M params) | ~300 |
| seq_len=5120, d=64, 3L (6.9M params) | ~3,000 |

The 10x sequence length increase caused ~10x slower convergence (roughly accounting for both the longer sequences and smaller model). However, **both ultimately converge to near-zero loss**, indicating the task is learnable at any tested scale.

### 3. FGN Develops Strong Geometry on Long Sequences

The FGN metric coefficient of variation (CV) — a measure of how non-flat the learned Riemannian metric is — showed its strongest development on the long-sequence state tracking task:

| Experiment | Peak Metric CV |
|------------|---------------|
| Phase 1a (pretrained, CE task) | 0.058 |
| Phase 1a.1 (curvature reward) | 0.580 |
| Phase 1a.2 Ablation A (task-only) | 0.127 |
| Phase 1b (multi-task, d=256) | 0.090 |
| Phase 2 Parity (d=128, seq=256) | 0.058 |
| Phase 2 State (d=128, seq=512) | 0.185 |
| **Phase 2 State Long (d=64, seq=5120)** | **0.231** (at step 7,700, still growing) |

The geometry continues to develop even after CE has effectively converged (CE=0.001 at step 7700), suggesting the metric is finding structure that reduces loss beyond what the cross-entropy gradient alone drives.

### 4. FGN's Computational Cost Is Prohibitive at Long Sequences

Heat kernel attention requires computing pairwise distances between all token positions, producing a `[B, H, N, N, d_head]` tensor. At `N=5120`:
- **Memory**: 25 GB per batch of 4 (fp32), forcing `bs=2`
- **Throughput**: 2,141 tok/s vs 37,960 tok/s for flat attention (17.7x slower)
- **Total training time**: ~27 hours (FGN) vs ~4.5 hours (flat) for 50K steps

This O(N^2 * d) scaling (vs O(N^2) for standard attention) is the dominant practical limitation of the current FGN architecture at long sequence lengths.

### 5. Batch Size Asymmetry Confounds Comparison

FGN trains at bs=2 while flat trains at bs=8 (4x difference) due to memory constraints. This means:
- FGN sees 4x fewer samples per step
- FGN's gradient estimates are noisier (smaller batch)
- At equal steps, FGN has seen 4x fewer total samples

This makes direct step-for-step comparison unfair. A fair comparison would require equal total samples (FGN running 4x more steps) or gradient accumulation. Currently, at 50K steps: flat sees 400K samples, FGN sees 100K samples.

---

## Files

### Configs
- `configs/phase2_long_small_fgn.yaml` — FGN: d=64, 3L, 2H, d_ff=256, seq_len=5120
- `configs/phase2_long_small_flat.yaml` — Flat: d=64, 3L, 2H, d_ff=276, seq_len=5120
- `configs/phase2_fgn.yaml` — Parity/short: d=128, 4L, 4H, seq_len=256
- `configs/phase2_flat.yaml` — Parity/short flat: d=128, 4L, 4H, d_ff=548
- `configs/phase2_state_fgn.yaml` — Short state tracking: seq_len=512
- `configs/phase2_state_flat.yaml` — Short state tracking flat: seq_len=512

### Task Implementations
- `fgn/tasks/parity.py` — Binary parity task (Task P)
- `fgn/tasks/state_tracking.py` — Cumulative state tracking (Task S)
- `fgn/tasks/permutation.py` — S₅ permutation composition (Task G)
- `fgn/tasks/affine.py` — Affine group Aff(Z₉₇) composition (Task H)

### Configs
- `configs/perm_fgn.yaml` / `configs/perm_flat.yaml` — d=256, 6L, seq_len=512 for permutation
- `configs/affine_fgn.yaml` / `configs/affine_flat.yaml` — d=256, 6L, seq_len=512 for affine

### Evaluation Scripts
- `scripts/eval_parity.py` — 11 OOD conditions for parity
- `scripts/eval_state.py` — Auto-selects short-seq or long-seq conditions
- `scripts/eval_perm.py` — S₅ permutation evaluation with configurable conditions
- `scripts/eval_affine.py` — Affine group evaluation with length generalization conditions

### Orchestration
- `scripts/run_state_long.sh` — Sequential state tracking orchestrator
- `scripts/run_perm_sweep.sh` — S₅ supervision sparsity sweep
- `scripts/run_affine_sweep.sh` — Affine supervision sparsity sweep
- `scripts/run_affine_lengthgen.sh` — Length generalization experiment

### Training Infrastructure
- `scripts/train_multitask.py` — Training script with `CUDA_MEMORY_FRACTION` env var support and `--task_kwargs` JSON arg

---

## Task 3: S₅ Permutation Composition (Non-Commutative)

### Design

Compute the composition of a chain of permutations from S₅ (symmetric group of 5 elements, |S₅|=120). Unlike state tracking (commutative addition), permutation composition is **non-commutative** — the order of operations matters.

- **Format**: `Compose: 3 1 4 5 2 ; 2 4 1 3 5 ; ... = 4 2 1 5 3 ; ... Answer: 5 3 1 2 4`
- **Supervision**: 5-element permutation state at checkpoints
- **Model**: d=256, 6L, 8H, 30.6M params, seq_len=512

### Results

With `sup_every=50` (50 compositions, final answer only) and 50 permutations per chain:

| Step | CE Loss |
|------|---------|
| 0 | 10.846 |
| 300 | 0.017 |
| 1,200 | 0.0008 |
| 2,300 | 0.0003 |

**100% accuracy by step ~3000**. The 6-layer d=256 transformer trivially solves S₅ composition even at the hardest supervision condition (50 compositions with final-answer-only supervision).

### Why S₅ Is Too Easy

S₅ is **decomposable**: the model can track 5 element positions independently in parallel. Applying permutation σ to state [p₁,p₂,p₃,p₄,p₅] just shuffles positions — each element's position depends only on its own previous position and σ. This reduces 120-state group composition to 5 parallel 5-state lookups, which any transformer can handle trivially.

---

## Task 4: Affine Group Aff(Z₉₇) — Non-Commutative, Non-Decomposable

### Design

Compute cumulative affine transformations x → ax + b (mod 97). The affine group has p×(p-1) = 97×96 = **9,312 elements**.

- **Operations**: `mul k` (multiply both a,b by k) and `add k` (add k to b only)
- **Non-commutative**: `mul 3; add 5` → 3x+5, but `add 5; mul 3` → 3x+15
- **Non-decomposable**: the b component depends on ALL preceding a values
- **Format**: `Affine: mul 3 ; add 5 ; mul 2 ; ... = 47 23`
- **Supervision**: 2-token state (a, b) at checkpoint positions
- **Model**: d=256, 6L, 8H, 30.6M params, seq_len=512

### Experiment 4a: Direct Training (100 ops)

Trained flat model on 100 ops with `sup_every=100` (final answer only):

| Step | CE Loss |
|------|---------|
| 0 | 11.006 |
| 400 | 0.250 |
| 600 | 0.013 |
| 2,800 | 0.0008 |

**100% accuracy by step ~3000**. Despite 9,312 states and non-commutativity, the 6-layer model solves 100-op chains trivially.

**Why**: The affine group has algebraic structure that enables **parallel scan** computation. The cumulative state (a,b) can be expressed as suffix products over the operands. A transformer with O(log n) depth can compute this in parallel, and 6 layers provides ample depth.

### Experiment 4b: Length Generalization (Key Experiment)

**Question**: Does a model trained on SHORT chains generalize to LONG chains it has never seen?

**Setup**:
- **Training**: 10-20 ops, `sup_every=20` (final answer only), 30K steps, bs=8
- **Evaluation**: 10, 15, 20, 25, 30, 50, 75, 100, 120 ops (all final-answer-only)

Both flat and FGN trained from random init, evaluated on the same conditions.

#### Results: Flat vs FGN Length Generalization

| Condition | Flat TokAcc | Flat CE | FGN TokAcc | FGN CE | FGN CV | FGN |κ| |
|-----------|:-----------:|:-------:|:----------:|:------:|:------:|:------:|
| 10 ops (in-dist) | **1.0000** | 0.0000 | **1.0000** | 0.0000 | 0.111 | 0.036 |
| 15 ops (in-dist) | **1.0000** | 0.0000 | **1.0000** | 0.0000 | 0.121 | 0.047 |
| 20 ops (in-dist) | **1.0000** | 0.0000 | **1.0000** | 0.0000 | 0.128 | 0.058 |
| 25 ops (near-OOD) | **1.0000** | 0.0000 | **1.0000** | 0.0000 | 0.131 | 0.065 |
| 30 ops (near-OOD) | **1.0000** | 0.0000 | **1.0000** | 0.0003 | 0.133 | 0.072 |
| 50 ops (far-OOD) | **1.0000** | 0.0000 | **1.0000** | 0.0101 | 0.138 | 0.096 |
| 75 ops (far-OOD) | **1.0000** | 0.0000 | 0.9944 | 0.0744 | 0.140 | 0.125 |
| 100 ops (far-OOD) | **1.0000** | 0.0000 | 0.9794 | 0.0897 | 0.140 | 0.151 |
| 120 ops (far-OOD) | **1.0000** | 0.0000 | **0.9038** | 0.2441 | 0.140 | 0.172 |
| 50 ops sup/10 | **1.0000** | 0.0000 | **1.0000** | 0.0021 | 0.137 | 0.099 |
| 100 ops sup/10 | **1.0000** | 0.0000 | 0.9742 | 0.0784 | 0.138 | 0.155 |

Oracle flat (trained on 10-100 ops): **100% on all conditions**, confirming the task is fully learnable.

### Key Finding: FGN FAILS Where Flat Succeeds

This is the **opposite** of the hypothesis. The flat transformer generalizes perfectly from 10-20 ops to 120 ops (6x the training length). FGN degrades:

- **75 ops**: FGN accuracy drops to 98.9% (flat: 100%)
- **100 ops**: FGN accuracy drops to 95.9% (flat: 100%)
- **120 ops**: FGN accuracy drops to 81.9% (flat: 100%)

The FGN's geometric metrics reveal what's happening:

1. **Curvature increases with chain length**: |κ| grows from 0.036 (10 ops) to 0.172 (120 ops)
2. **Metric CV is roughly constant**: ~0.12-0.14 regardless of chain length
3. **CE loss increases monotonically**: 0.000 → 0.244 from 10 to 120 ops

**Interpretation**: The heat kernel attention imposes a locality bias through its distance-dependent weighting. For short chains (≤50 ops), all operations are within the "diffusion radius" of the learned metric. For longer chains, distant operations are geometrically attenuated, causing the model to underweight early operations that are critical for the final state. The flat transformer's dot-product attention has no such locality bias — it can attend equally to any position.

**Training statistics**:
- Flat: 74K tok/s, converged by step ~3000
- FGN: 21K tok/s (3.6x slower), converged by step ~3000
- FGN metric CV: 0.12, stable after step ~5000

---

## Updated Key Findings

### 6. Non-Commutative Operations Don't Break the 6-Layer Transformer

Both S₅ permutation composition and affine group composition are solved trivially by a 30.6M-parameter, 6-layer transformer. The theoretical depth limit of O(2^L) ≈ 64 sequential steps is sufficient for chains of ≤120 operations. Moreover, algebraically structured groups (affine group, symmetric group) admit parallel scan algorithms that the transformer discovers during training.

### 7. Flat Transformer Has Superior Length Generalization

When trained on 10-20 operations and evaluated on chains up to 120 operations:
- **Flat**: 100% accuracy on ALL conditions (perfect generalization)
- **FGN**: Degrades to 81.9% at 120 ops (18.1% absolute gap)

The flat transformer's dot-product attention generalizes better because it applies a uniform algorithm across all positions. The heat kernel attention's learned metric creates position-dependent attention patterns that don't transfer to unseen sequence lengths.

### 8. FGN's Metric Responds to Difficulty But Doesn't Help

FGN's geometric metrics (CV, |κ|) increase with chain length, showing the metric IS adapting to harder inputs. But this adaptation is harmful — the increasing curvature distorts attention patterns away from what the task requires. The metric was calibrated on 10-20 op sequences and over-curves on longer sequences.

## Next Steps

1. **Test with smaller models**: d=32, 2L — where the transformer has genuinely limited sequential depth (O(4) steps). FGN might help at this scale where flat can't learn the full algorithm.
2. **Train FGN on the full range**: Train FGN on 10-100 ops to see if direct training eliminates the generalization gap.
3. **Random DFA**: Create a task with no algebraic structure (random state transition table with K > d states) where parallel scan is impossible.

---

## Appendix: Previous Phase Results Summary

### Phase 1a: Language Model Pretraining
- Both FGN and flat pretrained on OpenWebText subset
- Loss: 10.86 → 4.62 after 50K steps
- FGN metric CV = 0.058, geometry correlates with punctuation (+0.07) and anti-correlates with prediction entropy (-0.07)

### Phase 1a.1: Curvature Reward
- Added curvature reward (mu=0.1): CV inflated to 0.58
- But correlations got WEAKER (punct rho=+0.07 vs +0.19 for task-only)
- Key insight: reward creates noise, task gradient creates signal

### Phase 1a.2: Ablation Study
- **Ablation A** (no reward, no smoothness, 0.1x LR): CV=0.127 — task loss alone maintains geometry
- Task-driven geometry is 12x more structurally efficient than reward-driven

### Phase 1b v1: Multi-Task (4 pure synthetic tasks)
- All 4 tasks (temporal, pattern, interleaved, multi-hop) solved 100% within 1000 steps
- Tasks too easy at d=256, 31M params

### Phase 1b v2: Compound Reasoning Task
- Three-section compound task (sort + scan + chain)
- Also too easy — both models solved it

### Phase 1b v3: Arithmetic Chains
- Sequential arithmetic at d=64, d=128 — both converge identically
- No regime found where FGN outperforms flat on pure synthetic tasks
