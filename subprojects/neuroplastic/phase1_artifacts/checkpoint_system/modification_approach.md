# Recommended Modification Approach for Phase 2

## Three-Stage Strategy

### Stage 1: Behavioral Modification via Logits Processor (Approach C)

**Duration:** First experiments
**Risk:** Zero (no weight changes)
**Cycle time:** Seconds (parameter change) to minutes (file update + restart)

The FluidGeometryLogitsProcessor is already deployed. Extend it to accept Nemotron-proposed parameter modifications:
- Temperature curves, think-token thresholds, custom token biases
- New structural laws proposed by the model itself
- Nemotron reasons about the geometric engine using its blueprint, proposes changes, we apply them

This tests self-modification reasoning without touching weights.

### Stage 2: Targeted Weight Modification (Approach A — Offline)

**Duration:** After Stage 1 validates reasoning quality
**Risk:** Low (selective backup, full rollback)
**Cycle time:** ~6 minutes per modification cycle

Workflow:
1. `save_state.py --layers 40 --output /workspace/checkpoints/pre_exp001`
2. Modification script: load tensor, apply change, write back to shard
3. `docker restart vllm-nemotron-serve` (6 min)
4. Run evaluation harness
5. If degraded: `restore_state.py --checkpoint /workspace/checkpoints/pre_exp001` + restart

**Safe first targets (from Phase 0 analysis):**
- MoE expert weights (128 experts, CV 1-4% — highly redundant)
- Router/gate weights (small tensors, [128, 2688])
- Mamba dt_bias (scalar per head, affects update rate)

**Avoid initially:**
- Attention Q/K/V/O in layer 42 (largest norms, highest impact)
- Embedding and lm_head (global effect)
- A_log (controls state decay — fundamental to SSM dynamics)

### Stage 3: Hot-Swap Iteration (Approach B)

**Duration:** After Stage 2 patterns are proven safe
**Risk:** Medium (unclear torch.compile interaction)
**Cycle time:** ~4 minutes (reload only, no recompile)

Requires adding a trigger mechanism for `reload_weights()` — either a custom API endpoint or a docker exec side-channel. Investigate only after Stage 2 validates the offline workflow.

## Key Safety Principle

Every modification follows: **backup → modify → restart → evaluate → accept/rollback**

The Markov chain walk from the requirements: each step is a proposed modification. Accept if evaluation improves or holds. Rollback if it degrades. The checkpoint system makes every step reversible.
