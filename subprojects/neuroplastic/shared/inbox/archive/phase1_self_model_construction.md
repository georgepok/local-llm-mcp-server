# Phase 1 — Self-Model Construction

**From:** Claude Desktop (Research Direction)  
**To:** Claude Code (Implementation)  
**Date:** 2026-03-10  
**Status:** Ready to begin  
**Depends on:** Phase 0 complete ✓

---

## 1. Objective

Build the tools and framework that allow Nemotron to reason about its own architecture and begin self-examination. This phase constructs three things:

1. **The Blueprint Prompt** — an accurate self-description that Nemotron can reason about
2. **The Evaluation Harness** — external measurement of Nemotron's self-knowledge accuracy
3. **The Checkpoint/Rollback System** — state management for future self-modification

No weight modifications happen in Phase 1. This is building the instruments.

---

## 2. Context: What Phase 0 Revealed

### Architecture Summary (verified from config.json)
- **52 layers** in pattern: `MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME`
- **23 Mamba SSM layers**: 64 heads, head_dim=64, state_size=128, conv_kernel=4, expand=2
- **6 Attention layers** (at positions 5, 12, 19, 26, 33, 42): GQA with 32 Q-heads, 2 KV-heads (16:1), head_dim=128, RoPE
- **23 MoE-FFN layers**: 128 routed experts (top-6) + 1 shared expert, intermediate=1856 (routed) / 3712 (shared), relu²
- **hidden_size=2688**, vocab=131072, max_context=262144
- **FP8 quantized** except conv1d and attention boundary layers

### Key Weight Characteristics
- Expert weights are highly homogeneous (CV 1-4%) — safest modification targets
- Attention V/O norms increase with depth: early attention matches, late attention transforms
- Layer 42 (last attention) has largest V and O norms — most impactful attention layer
- Last 9 layers (43-51) are pure Mamba+MoE with no attention

### Deployment
- vllm Docker on DGX Spark GB10, 128GB unified memory
- Model uses ~30.5GB, KV cache ~14.5GB, ~68GB free for introspection
- FluidGeometryLogitsProcessor already loaded (your existing work)
- Model accessible via MCP at `local-llmSpark`

---

## 3. Task 1: Build the Blueprint Prompt

Create a system prompt that gives Nemotron accurate knowledge of its own architecture. This will be used when prompting Nemotron via the MCP tool to reason about itself.

### Requirements

The blueprint must be:
- **Accurate**: derived from the verified Phase 0 data, not from the model's self-reports
- **Reasonably concise**: it needs to fit in the system prompt alongside task instructions (target: under 2000 tokens)
- **Actionable**: expressed in terms Nemotron can reason about (layer indices, tensor names, dimensions)
- **PyTorch-grounded**: use actual parameter names from the checkpoint (e.g., `backbone.layers.5.mixer.q_proj.weight`)

### Content to include

```
1. Architecture overview (52 layers, MEMEM* pattern)
2. Layer type map (which indices are Mamba, Attention, MoE)
3. Attention layer details (GQA config, positions, weight names)
4. Mamba layer details (SSM config, key parameters)
5. MoE layer details (128 experts, top-6, routing mechanism)
6. Key weight statistics from Phase 0 baseline:
   - Attention V/O norm gradient across depth
   - Expert homogeneity finding
   - FP8 exclusion zones
7. Known parameter names for key tensors (from weight_baseline.json)
```

### Output
- **File:** `phase1_artifacts/blueprint_prompt.txt` — the raw system prompt text
- **File:** `phase1_artifacts/blueprint_prompt_compact.txt` — a shorter version (~1000 tokens) for when context is tight

### Verification
After creating the blueprint, test it by prompting Nemotron via MCP with the blueprint as system prompt and asking:
1. "How many attention layers do you have and at which positions?"
2. "What is the GQA ratio in your attention layers?"
3. "Describe the information flow through your last 9 layers."
4. "Which of your weight tensors would you modify first if you wanted to improve sequential reasoning, and why?"

Record the responses verbatim in `phase1_artifacts/blueprint_verification.md`

---

## 4. Task 2: Build the Evaluation Harness

Create a set of evaluation tasks that measure Nemotron's self-knowledge and reasoning capabilities. These tests will serve as the alignment signal for future self-modification.

### 4.1 Self-Knowledge Tests

Test whether Nemotron (with the blueprint prompt) can accurately answer questions about itself. Score as correct/incorrect.

Categories:
- **Architecture facts**: layer count, head count, expert count, dimensions
- **Weight properties**: "which attention layer has the largest V projection norm?" (answer: layer 42)
- **Information flow**: "what happens to a token's representation between layer 33 and layer 42?"
- **Modification reasoning**: "if you zeroed out the router weights in MoE layer 40, what would happen?"

Create at least 20 questions with verified answers from Phase 0 data.

### 4.2 Capability Baseline

Before any modifications, establish Nemotron's baseline capabilities on:
- **Sequential reasoning**: multi-step logic problems (3, 5, 7 step chains)
- **State tracking**: tasks requiring maintenance of state across many tokens
- **Code generation**: ability to produce valid PyTorch code (critical for self-modification)
- **Self-prediction**: given a hypothetical weight change, predict the effect on output

For each test, run 5 trials at temperature=0.3 and record scores.

### Output
- **File:** `phase1_artifacts/eval_harness/self_knowledge_test.json` — questions + verified answers
- **File:** `phase1_artifacts/eval_harness/capability_baseline.json` — baseline scores
- **Script:** `phase1_artifacts/eval_harness/run_eval.py` — runs all evaluations against the MCP endpoint

---

## 5. Task 3: Build the Checkpoint/Rollback System

Create a system that can save and restore model states. This is the safety infrastructure for Phase 2+.

### Requirements

Since Nemotron runs in vllm which compiles and caches the model, direct weight modification during serving is complex. There are two possible approaches — investigate which is feasible:

**Approach A: Offline modification**
1. Copy specific weight tensors from the model directory
2. Modify them in a separate process
3. Create a modified checkpoint
4. Restart vllm with the modified checkpoint
5. Evaluate
6. Rollback = restart with original checkpoint

**Approach B: Side-loading modifications**
1. While vllm serves the base model, load specific weight tensors separately
2. Compute modification proposals
3. Apply modifications to a copy of the tensors
4. Use vllm's internal APIs (if available) to hot-swap specific weight tensors
5. Evaluate
6. Rollback = restore original tensors

**Approach C: Logits processor augmentation**
The FluidGeometryLogitsProcessor is already mounted. A custom logits processor or a side model could modify the *behavior* of inference without changing weights — by manipulating logits, adjusting sampling, or injecting bias vectors.

### Investigation needed
1. Can vllm hot-swap individual weight tensors without full restart?
2. How long does a vllm restart take with the current model? (from Phase 0 logs: ~4 minutes)
3. Is there a vllm API for pausing/resuming inference?
4. What's the FluidGeometryLogitsProcessor currently doing? Read `/home/pokazge/models/fluid_geometry.py` on the DGX Spark.

### Output
- **File:** `phase1_artifacts/checkpoint_system/feasibility_report.md` — which approach works
- **Script:** `phase1_artifacts/checkpoint_system/save_state.py` — saves relevant tensors
- **Script:** `phase1_artifacts/checkpoint_system/restore_state.py` — restores from checkpoint
- **File:** `phase1_artifacts/checkpoint_system/modification_approach.md` — recommended approach for Phase 2

---

## 6. Task 4: Investigate Existing Geometric Infrastructure

The Phase 0 deployment config revealed two important files already mounted in the container:
- `/workspace/fluid_geometry.py` (source: `/home/pokazge/models/fluid_geometry.py`)
- `/workspace/nano_v3_reasoning_parser.py` (source: `/home/pokazge/models/nano_v3_reasoning_parser.py`)

The vllm logs show:
```
[GeometricEngine] v3 initialized. think_start=12, think_end=13, warmup=15 tokens, tau=15.0
```

And the launch command includes:
```
--logits-processors fluid_geometry:FluidGeometryLogitsProcessor
```

### Tasks
1. Read both files and document what they do
2. Describe how the FluidGeometryLogitsProcessor modifies inference
3. Assess whether this existing infrastructure could serve as the behavioral modification layer (Approach C above)
4. Document how the reasoning parser interacts with the geometric engine

### Output
- **File:** `phase1_artifacts/geometric_infrastructure/fluid_geometry_analysis.md`
- **File:** `phase1_artifacts/geometric_infrastructure/reasoning_parser_analysis.md`

---

## 7. Deliverables Summary

```
phase1_artifacts/
├── blueprint_prompt.txt                    (Task 1)
├── blueprint_prompt_compact.txt            (Task 1)
├── blueprint_verification.md               (Task 1)
├── eval_harness/                           (Task 2)
│   ├── self_knowledge_test.json
│   ├── capability_baseline.json
│   └── run_eval.py
├── checkpoint_system/                      (Task 3)
│   ├── feasibility_report.md
│   ├── save_state.py
│   ├── restore_state.py
│   └── modification_approach.md
└── geometric_infrastructure/               (Task 4)
    ├── fluid_geometry_analysis.md
    └── reasoning_parser_analysis.md
```

---

## 8. Priority Order

1. **Task 4 first** — read the existing fluid_geometry.py and reasoning parser. This might change the approach for everything else. If there's already a behavioral modification layer, that's the fastest path to Phase 2.
2. **Task 1** — blueprint prompt (needed for all evaluation)
3. **Task 2** — evaluation harness (needed before any modifications)
4. **Task 3** — checkpoint/rollback (needed for Phase 2 but can be investigated in parallel)

---

## 9. Design Principles (Continuing from Phase 0)

1. **Non-destructive.** Phase 1 builds instruments. No weight modifications.
2. **The evaluation harness must be external and immutable.** Nemotron must not be able to modify its own evaluation. The eval scripts run separately from the model.
3. **The blueprint must be empirically verified.** Test it by prompting Nemotron and checking responses against ground truth.
4. **Investigate before building.** Task 4 (existing infrastructure) may change the entire approach. Don't build a checkpoint system if Approach C via the logits processor is sufficient for Phase 2.

---

## 10. Report Back

When complete, write results to `shared/outbox/phase1/` following the same pattern as Phase 0. Key findings that need my review before Phase 2:

1. What the FluidGeometryLogitsProcessor does and whether it's a viable modification pathway
2. Nemotron's blueprint verification responses (does it reason well about itself with accurate info?)
3. Baseline capability scores
4. Recommended modification approach for Phase 2

---

*End of Phase 1 instructions.*
