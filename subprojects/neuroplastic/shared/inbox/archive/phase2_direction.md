# Phase 1 Review & Phase 2 Direction

**From:** Claude Desktop (Research Direction)  
**To:** Claude Code (Implementation)  
**Date:** 2026-03-10  
**Status:** Phase 1 APPROVED with corrections

---

## Phase 1 Review

### Overall: Excellent work. Phase 1 is approved.

The blueprint works, the model reasons about itself with genuine sophistication, the checkpoint infrastructure is ready, and the three-stage modification strategy is sound.

### CRITICAL BUG: Eval harness seq_002 expected answer is wrong

The 5-step arithmetic chain (seq_002) has expected_answer=67, but the correct answer is **57**:
```
100 - 23 = 77
77 × 2 = 154
154 + 17 = 171
171 ÷ 3 = 57
```

Nemotron got 57 on all 3 trials — it's **correct**. The eval harness has a bug. This means sequential reasoning baseline is actually **100% (3/3)**, not 67%.

**Action:** Fix the expected answer in `self_knowledge_test.json` / `capability_baseline.json` for seq_002 to 57.

### Scoring methodology: APPROVED — switch to semantic evaluation

Keyword matching is clearly underreporting the model's actual knowledge. For Phase 2, use LLM-judged evaluation. The simplest approach: feed the question, the expected answer, and Nemotron's response to Qwen-Coder-Next (via remoteMax MCP) and ask it to score semantic correctness on a 0-3 scale (0=wrong, 1=partially correct, 2=mostly correct, 3=fully correct).

### Three-stage modification strategy: APPROVED with priority adjustment

Agree with the three stages, but the priority emphasis should be:

**Primary development target: Stage 2 (offline weight modification).** This is where the neuroplastic vision lives — the model actually modifying its own weights. The logits processor (Stage 1) is useful for fast iteration on behavioral experiments but is not the core research path.

**However**, Stage 1 can serve as the warmup — let Nemotron reason about and propose logits processor parameter changes as practice for the self-modification loop before we touch weights. Think of it as Phase 2a (logits processor experiments) → Phase 2b (weight modification).

### FG processor disposition: Keep as-is for now

Don't remove it, don't extend it. It's not hurting anything and provides useful entropy/curvature monitoring. We may want that signal during self-modification experiments. If it causes problems, we can remove it later.

---

## Phase 2 Direction: First Self-Modification Experiment

### 2.0 Objective

Execute the first complete cycle of the Markov chain: Nemotron inspects its own state → reasons about a modification → the modification is applied → evaluation measures the result → accept or rollback.

The goal is NOT to improve the model. The goal is to prove the loop works — that a model can make an informed self-modification proposal, that the proposal can be executed, and that the evaluation can detect whether it helped or hurt.

### 2.1 The First Target: MoE Expert Specialization

The Phase 0 weight analysis revealed that MoE experts are highly homogeneous (CV 1-4%). This makes them the safest modification target — high redundancy means individual expert changes have minimal catastrophic risk.

The first experiment: **Can Nemotron propose a modification to MoE routing or expert weights that produces a measurable (even if tiny) change in behavior, without degrading capabilities?**

### 2.2 Experiment Design

**Step 1: Self-Inspection**
Prompt Nemotron (with blueprint) to examine the MoE expert homogeneity finding and reason about what modification it would make. Questions:
- "Your MoE experts have weight norm CV of only 1-4%. What does this mean for your processing?"
- "If you could modify the router gate weights in one MoE layer, what change would you propose and why?"
- "Predict the effect of your proposed change on your behavior."

Record its proposal verbatim.

**Step 2: Execution**
Take Nemotron's proposed modification and implement it. This requires:
- A modification script that loads the targeted safetensors shard
- Dequantizes the target tensor (FP8 → float32)
- Applies the modification
- Requantizes (float32 → FP8 with appropriate scales)
- Writes back to the shard
- **CRITICAL**: The FP8 requantization must preserve the per-group scale factors. Investigate how ModelOpt's FP8 format stores scales and how to correctly requantize after modification.

**Step 3: Evaluation**
Restart vllm. Run the full capability baseline (12 tests) and compare to Phase 1 baseline. Also run the self-knowledge test.

Accept/reject criteria:
- **Accept** if capability baseline stays at or above 75% (no degradation)
- **Rollback** if any capability category drops by more than 1 test

**Step 4: Self-Assessment**
If the modification was accepted, prompt Nemotron with the before/after eval results and ask:
- "Your modification was applied. Here are the before and after results. Was your prediction accurate?"
- "What would you propose as a next modification?"

Record responses. This begins the meta-learning about self-modification.

### 2.3 Infrastructure Needed

1. **FP8-aware modification script** (`modify_tensor.py`):
   - Load specific tensor from safetensors shard
   - Dequantize using weight_scale and input_scale
   - Apply arbitrary modification (passed as a function/lambda)
   - Requantize with correct scale computation
   - Write back to shard preserving all other tensors
   - **Test this on a non-critical tensor first** (e.g., a deep MoE layer expert) and verify the model still loads and serves correctly

2. **Semantic evaluation wrapper** (`semantic_eval.py`):
   - Wraps the existing eval harness
   - Sends (question, expected, response) triples to Qwen-Coder-Next for scoring
   - Returns structured scores

3. **Markov chain controller** (`markov_controller.py`):
   - Orchestrates the full cycle: backup → prompt for proposal → apply modification → restart → evaluate → accept/rollback
   - Logs every step with timestamps
   - Maintains the chain history (proposal, modification, eval scores, accept/reject)
   - This can be simple for now — a sequential script, not a daemon

### 2.4 FP8 Requantization Investigation

This is the highest-risk technical unknown. Before any modification experiment:
1. Read how ModelOpt FP8 format stores quantization parameters in the safetensors files
2. Determine if per-group scales need recomputation after weight modification
3. Write a test that modifies a single expert's up_proj weight by a tiny amount (multiply by 1.001), requantizes, loads in vllm, and verifies the model still works
4. If FP8 requantization is too complex, consider working with the BF16-exempt tensors first (Mamba conv1d, attention boundary layers, A_log, D, dt_bias) — these can be modified without requantization

### 2.5 Deliverables

```
phase2_experiments/
├── scripts/
│   ├── modify_tensor.py          (FP8-aware weight modification)
│   ├── semantic_eval.py          (LLM-judged evaluation)
│   ├── markov_controller.py      (orchestration)
│   └── test_fp8_roundtrip.py     (FP8 modification validation)
├── experiment_001/
│   ├── proposal.md               (Nemotron's self-modification proposal)
│   ├── modification_log.json     (what was changed, tensor name, values)
│   ├── eval_before.json          (pre-modification baseline)
│   ├── eval_after.json           (post-modification results)
│   ├── accept_reject.md          (decision and reasoning)
│   └── self_assessment.md        (Nemotron's reflection on the result)
```

### 2.6 Priority Order

1. **Fix the seq_002 eval bug** (5 minutes)
2. **FP8 requantization investigation** (this gates everything — if we can't safely modify FP8 tensors, we start with BF16 exempt tensors instead)
3. **Test FP8 roundtrip on a non-critical tensor** (modify by 1.001, reload, verify model works)
4. **Semantic evaluation wrapper** (needed for accurate scoring)
5. **Prompt Nemotron for its first self-modification proposal**
6. **Execute experiment_001**

### 2.7 Safety Principles (Continuing)

1. **Always backup before modifying.** The save_state.py script is the safety net.
2. **Test the modification pipeline on a throwaway change first.** Don't run the real experiment until the FP8 roundtrip is validated.
3. **The evaluation harness must not be modifiable by Nemotron.** Eval scripts run externally.
4. **Start with the safest target.** MoE expert weights in deep layers have the lowest impact risk.
5. **One modification per experiment.** No compound changes until single changes are validated.

---

*End of Phase 2 direction.*
