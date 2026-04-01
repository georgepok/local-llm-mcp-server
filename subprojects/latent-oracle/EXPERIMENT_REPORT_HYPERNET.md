# Oracle HyperNet Experiment Report

**Date**: 2026-03-05
**Status**: Running (step 38K/50K), outcome determined — negative result.

## Motivation

The 5M LiquidARC model plateaued at **54.2% eval xform accuracy** on ARC-AGI. Prior work established that:

1. **W_o is the dominant lever for task adaptation.** TTT (test-time training) showed that unfreezing W_o alone gave +28pp (13.7% → 43.7%), while MetricNet-only gave +2pp. This established a clean **WHERE/WHAT decomposition**: MetricNet controls WHERE to route information (geometric routing), W_o controls WHAT content transformation to apply.

2. **Similarity distillation failed.** Probes on Qwen3.5-9B's internal cosine similarities found no useful ARC structure — no color signal, no spatial locality, no transform/copy distinction. Only 2/5 probes passed (trivially). The oracle's internal geometry does not encode ARC-relevant patterns in a form that can supervise the heat kernel.

3. **The bottleneck is generalization, not capacity.** The 5M width scaling experiment showed a 40pt train/eval gap (90% train, 50% eval). Procedural training tasks don't transfer to real ARC evaluation tasks.

**Hypothesis**: Instead of matching the oracle's internal representations (which contain no useful geometric signal), use the oracle embedding as a **task descriptor** to predict task-specific W_o weight deltas via a lightweight HyperNet. This amortizes what TTT does in 100 gradient steps into a single forward pass. No auxiliary loss — CE backpropagates through the delta directly.

## Architecture

```
Oracle (Qwen3.5-9B, precomputed)
  │ [B, 4096] mean-pooled hidden states per task
  ▼
OracleProjectionHead (6M params, from prior work)
  ├→ z_context [B, 768]  ──→ dynamics.set_context()  (WHERE routing)
  ├→ kappa_target [B, 1]  ──→ curvature supervision
  │
  └→ z_context.mean(0) [768]  ──→ OracleHyperNet (NEW, 226K params)
                                    ├→ adapter: Linear(768,256) + GELU
                                    └→ LowRankHead(rank=8): U[768,8] @ diag(c) @ V[8,768]
                                        └→ delta_W_o [768, 768]

ODE Loop (16 Euler steps, torch.compile'd):
  update = F.linear(routed_v, W_o.weight + delta_W_o, None)
```

**Key design choices:**
- **Batch mean-pooling**: z_context averaged across batch before adapter, producing one delta per batch (not per sample). This was chosen because each batch samples different tasks, but at B=16 the gradient to each sample is diluted 16x.
- **Low-rank factorization**: rank-8 constrains the delta subspace to ~29K params, preventing arbitrary weight perturbation.
- **Zero buffer**: `_delta_W_o` initialized to zeros, making `W_o.weight + 0` = original behavior. No conditional branches in forward — torch.compile sees identical graph regardless of hypernet state.
- **Direct assignment for gradient flow**: `set_delta_W_o(delta)` uses attribute assignment (not `copy_()`) so the hypernet output stays in the autograd graph. `zeros_like()` for None reset avoids in-place-on-leaf errors.

**Total new parameters**: 226,121 (4.6% of 5M base model).

## Training Protocol

| Phase | Steps | Trainable | HyperNet | Loss |
|-------|-------|-----------|----------|------|
| 0 (warmup) | 0–2K | Projection head only | Idle (delta=0) | CE only |
| 1 (distill) | 2K–20K | Proj + ODE + HyperNet | Active | CE + 0.1·κ_distill |
| 2 (finetune) | 20K–50K | Same | Active | CE + 0.001·κ_distill |

- **Base model**: 5M LiquidARC checkpoint (step 21K, 54.2% eval xform)
- **Data**: 400 ARC train tasks, 400 ARC eval tasks, 6680 precomputed oracle embeddings with D4 augmentation
- **Optimizer**: AdamW, projection+hypernet LR=3e-4, ODE LR=1e-4
- **Infrastructure**: DGX Spark (GB10), `oracle-train` container, torch.compile enabled

## Results

### Full Training Trajectory

| Step | Phase | Train CE | Train xform | **Eval xform** | **Eval CE** | Δ_norm | |κ| | CV |
|------|-------|----------|-------------|----------------|-------------|--------|-----|-----|
| 0 | P0 | 3.03 | 27.1% | — | — | — | 0.131 | 5.7 |
| 50 | P0 | 1.38 | 74.5% | — | — | — | 0.182 | 6.6 |
| 500 | P0 | 1.38 | 56.4% | — | — | — | 0.084 | 6.5 |
| 1000 | P0 | 0.89 | 63.2% | — | — | — | 0.085 | 6.4 |
| 2000 | P1 | 0.90 | 63.6% | — | — | 0.0008 | 0.057 | 8.1 |
| 2500 | P1 | 0.76 | 65.1% | **54.1%** | 1.77 | 0.002 | 0.038 | 6.8 |
| 5000 | P1 | 0.53 | 82.9% | **46.5%** | 2.64 | 0.015 | 0.037 | 6.8 |
| 7500 | P1 | 0.28 | 91.4% | **47.2%** | 2.72 | 0.010 | 0.035 | 7.4 |
| 10000 | P1 | 0.29 | 90.8% | **42.1%** | 3.46 | 0.011 | 0.038 | 7.2 |
| 12500 | P1 | 0.21 | 91.7% | **36.9%** | 3.92 | 0.006 | 0.034 | 6.6 |
| 15000 | P1 | 0.07 | 99.7% | **38.9%** | 4.37 | 0.005 | 0.038 | 6.7 |
| 17500 | P1 | 0.17 | 94.1% | **38.6%** | 4.72 | 0.005 | 0.040 | 6.6 |
| 20000 | P2 | 0.11 | 98.5% | **34.8%** | 5.71 | 0.005 | 0.034 | 7.3 |
| 25000 | P2 | 0.06 | 98.7% | **34.5%** | 5.89 | 0.005 | 0.031 | 6.5 |
| 30000 | P2 | 0.09 | 98.5% | **34.0%** | 5.90 | 0.008 | 0.025 | 7.2 |
| 35000 | P2 | 0.06 | 99.4% | **27.3%** | 7.49 | 0.015 | 0.027 | 6.6 |
| 37500 | P2 | 0.04 | 100% | **30.1%** | 7.63 | 0.009 | 0.024 | 6.6 |

**Best eval xform: 54.1%** at step 2500 (first eval after Phase 1 start). Baseline without oracle: 54.2%.

### Dynamics Analysis

**Phase 0 (steps 0–2K): Projection warmup — working as intended.**
- Projection head learned to map oracle [4096] → context [768]. CE dropped 3.03 → 0.89.
- ODE frozen, hypernet idle. No delta produced.
- κ converged from 0.131 to 0.040 (projection learning curvature targets).
- CV climbed from 5.7 to 6.5 (metric already adjusting from frozen ODE's perspective).

**Phase 1 transition (step 2000): ODE unfrozen — immediate CV spike.**
- CV jumped from 6.5 → 8.1 at step 2000 as the ODE began responding to oracle context.
- Settled back to 6.5–7.4 by step 3000.
- First eval at step 2500: **54.1% xform** — essentially matching the 54.2% baseline. The oracle context injection through MetricNet (WHERE path) preserved the pretrained model's eval performance.

**Phase 1 divergence (steps 3K–20K): Train/eval gap opens catastrophically.**
- Train xform: 65% → 99%. The model memorized 400 training tasks perfectly.
- Eval xform: 54% → 35%. Every step of training made generalization worse.
- Eval CE: 1.77 → 5.71. The model became increasingly confident in wrong predictions on unseen tasks.
- Δ_norm peaked at 0.015 (step 5K), then shrank to 0.005 (step 15K). The hypernet initially explored large deltas, then learned to produce smaller ones — but this didn't help eval.

**Phase 2 (steps 20K–38K): Continued degradation.**
- κ weight reduction (0.1 → 0.001) had no visible effect on the overfitting trajectory.
- Eval xform continued declining: 35% → 27%.
- Δ_norm rebounded from 0.005 to 0.015 in Phase 2 — the hypernet became more active, but this correlated with faster eval degradation.
- |κ| drifted down from 0.034 to 0.024, suggesting the geometric manifold was flattening as the model relied more on memorized W_o patterns.

### HyperNet Behavior

**The hypernet learned to produce non-zero, task-varying deltas.** Δ_norm > 0 throughout Phase 1+, with variation across batches (range 0.005–0.018), confirming the oracle embeddings carry task-distinguishing information.

**But the deltas encode training-task memorization, not generalizable transforms.**

The Δ_norm trajectory reveals an interesting dynamic:
1. Steps 2K–5K: Δ_norm grows rapidly (0.001 → 0.015) — hypernet exploring delta space
2. Steps 5K–17.5K: Δ_norm shrinks (0.015 → 0.005) — implicit regularization, gradient signal weakens as train CE approaches 0
3. Steps 20K–38K: Δ_norm rebounds (0.005 → 0.015) — with κ weight reduced, the hypernet has more freedom to produce larger deltas

None of these phases improved eval. The deltas were consistently useful for training tasks and consistently harmful for eval tasks.

### Comparison with Prior Approaches

| Approach | Eval xform | Effect on generalization |
|----------|-----------|------------------------|
| 5M baseline (no oracle) | 54.2% | Reference |
| Oracle HyperNet (best) | 54.1% | Neutral (step 2500, hypernet barely active) |
| Oracle HyperNet (step 10K) | 42.1% | Harmful |
| Oracle HyperNet (step 35K) | 27.3% | Catastrophic |
| TTT V2 (gradient-based) | +28pp over pre-transition | Helpful pre-transition, destructive post |
| Similarity distillation | N/A (probes showed no signal) | Abandoned |

## Interpretation

### Why Did It Fail?

**1. Overfitting on 400 tasks is the dominant failure mode.**

The 5M model without any oracle already had a 40pt train/eval gap. Adding 226K trainable parameters (hypernet) + 6M (projection head) on top of 5M model parameters, all training on 400 tasks, made the overfitting strictly worse. The model has ~11M effective parameters fitting 400 examples.

**2. The oracle embeddings distinguish tasks but don't encode generalizable task structure.**

The probes were right: Qwen's representations don't encode ARC-relevant structure (rules, spatial transforms, color mappings). They encode something — the non-zero Δ_norm proves this — but that "something" is more like a task fingerprint than a task description. The hypernet learned "when I see fingerprint X, apply memorized delta Y" rather than "this is a rotation task, apply the rotation delta."

**3. Amortized TTT inherits TTT's fundamental limitation.**

Gradient-based TTT was destructive post-transition in the 5M experiment because task-specific adaptation overwrites universal geometric structure. The hypernet is a differentiable approximation of the same mechanism — it learns to produce the same destructive deltas, just faster. The failure mode is architectural, not a matter of optimization.

**4. Batch mean-pooling may have diluted the signal.**

Averaging z_context across B=16 different tasks before the adapter means the hypernet sees a blurred "average task" rather than individual task descriptors. The per-task gradient is diluted 16x. However, given that even the early steps (when Δ_norm was tiny) didn't improve eval, this is likely a minor factor.

### What This Rules Out

- **Task-conditional W_o modulation from frozen oracle embeddings** does not improve generalization on ARC.
- The oracle embeddings are not useful as task descriptors for predicting weight deltas, even though they carry task-distinguishing information.
- Adding capacity for task-specific adaptation makes the overfitting problem worse, regardless of whether that capacity comes from gradient-based TTT (100 steps) or amortized forward-pass prediction (1 step).

### What Remains

The core problem is unchanged: **400 ARC training tasks are insufficient to learn generalizable geometric transforms at 5M parameters.** All approaches that add task-specific adaptation capacity (TTT, HyperNet, extended training) make eval worse.

Potential directions that this experiment does NOT rule out:
- **Architecture changes** that constrain the model to learn only generalizable transforms (e.g., equivariant layers, harder inductive biases)
- **More training data** (synthetic tasks that actually transfer to ARC, unlike procedural curriculum)
- **Smaller models** with fewer parameters per training example
- **Test-time methods that don't require training** (e.g., ensembling, prompt engineering at the task encoding level)

## Technical Notes

### Implementation Details
- `latent_oracle/oracle_hypernet.py`: OracleHyperNet class, reuses LowRankHead from liquid_arc/hypernet.py
- `liquid_arc/dynamics.py`: `_delta_W_o` zero buffer (persistent=False), `F.linear(routed_v, W_o.weight + delta, None)`, `set_delta_W_o()` with direct assignment for gradient flow
- `latent_oracle/train_utils.py`: `delta_W_o` parameter in `forward_with_oracle()`
- `configs/latent_oracle_hypernet.yaml`: `oracle_hypernet_enabled: true`, similarity distillation disabled

### Bugs Fixed During Implementation
- **`set_delta_W_o(None)` after grad-tracking tensor**: `zero_()` on a buffer that was replaced by direct assignment with a requires_grad tensor causes "leaf Variable in-place operation" error. Fix: use `torch.zeros_like(self.W_o.weight)` instead of `self._delta_W_o.zero_()`.
- **torch.compile failure on DGX Spark**: `oracle-train` container (vllm-openai:latest) missing `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`. The `fgn-train` container (nvcr.io/nvidia/vllm:26.01-py3) has this baked in. Fix: export env var before training.

### Throughput
- torch.compile enabled: 7K–15K tok/s (varies with sequence length)
- No recompilation stalls observed after initial compile
- Total training time estimate: ~5 hours for 50K steps
