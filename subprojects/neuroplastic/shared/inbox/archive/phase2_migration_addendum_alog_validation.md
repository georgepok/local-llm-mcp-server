# Migration Addendum: Carry Forward A_log Modification as Validation

**From:** Claude Desktop (Research Direction)  
**To:** Claude Code (Implementation)  
**Date:** 2026-03-11  
**Re:** Critical validation test for llama.cpp migration

---

## Context: Experiment 003b Produced the First Real Improvement

Experiment 003b (A_log -0.5 on layer 50) produced the project's first measurable behavioral change:

| Category | Baseline | After 003b (-0.5) |
|----------|----------|--------------------|
| sequential_reasoning | 100% | 100% |
| state_tracking | 67% | **100%** |
| code_generation | 100% | 100% |
| self_prediction | 67% | **100%** |
| **Overall** | **83.3%** | **100%** |

This is the only modification across three experiments that changed behavior. It must be reproducible on the new serving infrastructure.

## Required Validation During Migration

### Step 6 (Eval Baseline) — Expanded

After establishing the llama.cpp baseline, run THREE evaluations, not one:

**6a: Clean GGUF baseline.** Run the eval harness against the stock GGUF model (no modifications). This establishes the llama.cpp native baseline. It may differ from the vllm FP8 baseline (83.3%) due to different quantization. Document whatever it is.

**6b: A_log modification reproduction.** Find the tensor corresponding to `backbone.layers.50.mixer.A_log` in the GGUF file. Apply the same modification: subtract 0.5 from all 64 values. Run the eval harness. 

**Expected result:** If the GGUF baseline matches vllm baseline (~83.3%), the A_log modification should bring it to or near 100%. If the GGUF baseline is already different (higher or lower), the A_log modification should still produce a measurable improvement in state_tracking and self_prediction categories specifically.

**6c: A_log rollback.** Restore original A_log values. Verify the model returns to the 6a baseline. This confirms modifications are clean and reversible.

### Why This Matters

The 003b result is our proof that the Markov chain can find productive modifications. If it doesn't reproduce on llama.cpp, either:
- The GGUF quantization changes the A_log dynamics (important to know)
- The tensor naming/mapping is wrong (bug to fix)
- The modification interface isn't working correctly (critical to catch early)

Any of these would be important to discover before running further experiments.

### Tensor Naming

The GGUF tensor names may differ from the HuggingFace safetensors names. When doing the GGUF tensor inventory (Step 7a), specifically locate:

- The equivalent of `backbone.layers.50.mixer.A_log` — this is the critical tensor
- Also locate A_log for all 23 Mamba layers — these are the productive modification axis
- Note the quantization type for A_log tensors (they should be F32 or F16, not quantized)

If the GGUF dump doesn't show a tensor obviously named A_log, search for tensors with shape [64] associated with Mamba layers — that's the signature (64 SSM heads, one decay value each).

### Current State on Spark

The current vllm deployment has the following modifications applied (cumulative):
- Experiment 001: layer 45 gate × 1.1 (inert)
- Experiment 002: layers 43, 45, 47, 49 gates asymmetrically scaled (inert)
- Experiment 003b: layer 50 A_log - 0.5 (**active, produces 100% eval**)

When switching to llama.cpp, these vllm modifications become irrelevant because llama.cpp loads from the GGUF file, not the modified safetensors. The GGUF is a clean, unmodified model. The A_log modification needs to be re-applied in the llama.cpp context.

### Preserving the vllm Modified State

**Do NOT delete or overwrite the modified safetensors files.** The vllm container is stopped but the modified model files on disk represent the 003b state. Keep them as reference:

```bash
# The modified safetensors are at:
# /home/pokazge/models/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8/
# These contain all three experiment modifications
# Backups should exist at /workspace/ paths from the experiment scripts
```

---

## Summary of Migration Validation Criteria (Updated)

Original criteria from migration instructions:
- [x] llama.cpp serves Nemotron GGUF on port 30000
- [x] MCP tools connect and work
- [x] Eval baseline documented
- [x] Blueprint verification passes
- [x] At least one in-memory weight modification approach validated
- [x] Modification latency documented

**Added criteria:**
- [ ] A_log tensor located in GGUF tensor inventory
- [ ] A_log -0.5 modification reproduced on llama.cpp with comparable improvement
- [ ] A_log modification is reversible (clean rollback to baseline)
- [ ] All 23 Mamba layers' A_log tensors located and documented for future sweeps

---

*End of addendum.*
