# Neuroplastic Fine-Tuning Report

**Date:** 2026-03-13
**Status:** Training in progress (step 22/200, ~4h remaining)
**Author:** Claude Code (for reviewer)

---

## 1. Objective

Fine-tune Nemotron-3-Nano-30B-A3B to perform self-modification operations using the neuroplastic API. The base model has no knowledge of the API, its own architecture, or the reasoning patterns needed for weight inspection and modification. This fine-tuning teaches the model to be a competent self-modifier.

## 2. Current Approach: QLoRA on DGX Spark

### Configuration

| Parameter | Value |
|-----------|-------|
| Base model | NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 (~60GB) |
| Quantization | NF4 via bitsandbytes (60GB → 16.2GB in memory) |
| Method | QLoRA (LoRA on NF4-quantized base) |
| LoRA rank / alpha | 8 / 16 |
| Trainable params | 220M (0.7% of 31.8B) |
| Target modules | q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, in_proj, out_proj |
| Batch size | 1 (effective 8 via grad accumulation) |
| Sequence length | 1024 tokens |
| Learning rate | 2e-4 (linear decay, 10 warmup steps) |
| Optimizer | AdamW 8-bit |
| Steps | 200 (~4.7 hours at 84s/step) |
| Training loss masking | Response-only (DataCollatorForCompletionOnlyLM) |
| Hardware | DGX Spark GB10, 128GB unified memory |
| Peak memory | 18.8GB GPU |

### Training Data

- **236 examples** (from 335 total; 99 exceed 1024-token limit)
- Source: Phase 3 (self-directed sessions) and Phase 7 (autoresearch loop) transcripts
- Format: ChatML conversations with system prompt describing Nemotron's architecture, tensor paths, and neuroplastic API, followed by user requests and assistant responses demonstrating self-modification workflows
- Response-only loss: model is only trained on assistant responses, not system/user turns

### Loss Trajectory (steps 1-22)

```
Step  1: 16.57  (warmup, lr=0)
Step  5: 14.42  (warmup ramping)
Step 10:  6.52  (warmup complete, lr=2e-4)
Step 15:  5.12  (decay phase)
Step 20:  4.75
Step 22:  3.58  (and declining)
```

Initial loss ~17 reflects LoRA adapters starting at near-zero (random init). Rapid convergence to <4.0 within 22 steps indicates the model is learning the response patterns. Grad norms stabilized from ~22 to ~7. Memory stable at 18.8GB throughout.

### Technical Obstacles Overcome

1. **FP8 model path (abandoned):** FP8 weights require `weight_scale` tensors for dequantization, but `from_pretrained` discards them. Forward hooks don't prevent backward errors. Gradient checkpointing recomputes FP8 intermediates. All FP8 approaches produce garbage loss (~107).

2. **NVFP4 (abandoned):** Inference-only packed format. modelopt can't create training-compatible packed weights.

3. **AWQ (abandoned):** `compressed-tensors` format incompatible with transformers 4.57.1.

4. **BF16+NF4 OOM:** Initial attempts OOMed on 128GB unified memory. Resolved by confirming NF4 actually activates (16GB vs 60GB bf16).

5. **MoE dtype mismatch:** `index_add_()` fails when routing weights (float32) multiply expert outputs. Fixed by adding `.to(final_hidden_states.dtype)` cast in `modeling_nemotron_h.py:852`.

6. **MoE empty expert dummy compute:** When no tokens route to an expert, dummy forward uses `expert.down_proj.weight.dtype` which is uint8 for NF4. Fixed by using `hidden_states.dtype` instead.

7. **peft 0.14.0 bug:** `Linear4bit` LoRA wrapping broken. Upgraded to peft 0.18.1.

All patches applied to both `/workspace/models/.../modeling_nemotron_h.py` and the HuggingFace cache copy.

## 3. Post-Training Steps

### LoRA Merge

Training saves only LoRA adapter weights (~800MB). For self-modification, LoRA must be merged into the base model:

```python
model = model.merge_and_unload()  # NF4→bf16 dequant + LoRA delta
model.save_pretrained("merged_model")  # Full ~60GB bf16 model
```

The merged model can then be served directly or further quantized.

### Quantization for Deployment

| Format | Size | Quality | Self-mod compatible | Notes |
|--------|------|---------|-------------------|-------|
| BF16 | ~60GB | Lossless | Yes (native tensors) | Too large for simultaneous serve+train |
| FP8 | ~30GB | Near-lossless | Yes (simple scale factors) | Best balance for DGX Spark |
| GPTQ 4-bit | ~16GB | Good | Difficult (block-packed) | Needs dequant/requant per block |
| AWQ 4-bit | ~16GB | Good | Difficult (block-packed) | Same as GPTQ |

**Recommendation:** Merge → FP8 quantization via modelopt. Halves memory while keeping self-modification feasible (each weight has a scalar `weight_scale` that can be managed during INSPECT/MODIFY).

## 4. What This Fine-Tuning Achieves

**Teaches:**
- Neuroplastic API syntax and tensor path conventions
- Architecture self-awareness (Mamba/Attention/MoE layer assignments, what each controls)
- Self-modification reasoning pattern: observe → hypothesize → inspect → modify → verify
- When to use self-modification vs answering normally

**Does not teach:**
- Causal understanding of weights → behavior relationships (it learns to imitate successful transcripts, not why modifications work)
- Generalization to novel modification scenarios not in training data
- Safety judgment (predicting whether a change will degrade other capabilities)

This is **behavioral cloning** — the model learns to produce plausible self-modification sequences by imitating expert demonstrations. It acquires the vocabulary and workflow, not the underlying skill.

## 5. Future Trajectories

### 5a. Reinforcement Learning for Self-Modification (high impact, high cost)

RL addresses the core limitation of SFT: the model needs to learn *which* modifications actually work through trial and error, not just imitate transcripts.

**Setup:**
- State: model's current eval performance on target + held-out tasks
- Action: INSPECT/MODIFY operations via neuroplastic API
- Reward: improvement on target task minus degradation on held-out tasks

**Algorithm:** GRPO or REINFORCE on reasoning traces (reward the full chain-of-thought that led to a successful modification, not just the final action).

**Challenges:**
- Each episode requires: load model → apply modifications → run eval suite → compute reward (~10-15 min/episode on DGX Spark)
- Thousands of episodes needed → multi-day to multi-week run
- Action space is enormous (millions of weights x continuous deltas)
- Credit assignment: which modification in a chain of 5 actually helped?

**Mitigations:**
- Constrain action space: only allow modification of specific layer types, quantize deltas
- Shaped rewards: intermediate signals (attention pattern changes, activation statistics)
- Meta-learning: train on a distribution of tasks so the model learns general self-modification skill, not one specific fix

### 5b. Mamba State Persistence (native, unexplored)

Nemotron's 23 Mamba layers maintain SSM state that could provide implicit cross-turn memory without any weight modification.

**Concept:** Instead of explicitly editing weights, preserve the Mamba hidden state (A, B, C matrices' running state) between inference turns. The SSM state naturally accumulates context and could enable in-context adaptation.

**Advantages:**
- Zero cost: no training needed, architecture already supports it
- Reversible: state can be reset, no permanent model changes
- Fast: no weight I/O, just keep tensors in memory between requests

**Open questions:**
- Do the A matrix decay rates (controlled by A_log) allow meaningful state to persist across turns?
- Phase 2 Experiment 003b showed A_log modifications directly impact behavior — this validates that SSM dynamics matter
- Would require vLLM modifications to preserve per-session Mamba state (currently stateless per request)

### 5c. Gradient-Guided Self-Modification (principled, complex)

Instead of the model *guessing* which weights to change, compute the actual gradient of a task-specific loss with respect to the weights, then apply the update.

**This is effectively online fine-tuning**, but framed as self-modification:
1. Model receives a task it performs poorly on
2. Compute loss on that task
3. Backprop to get weight gradients
4. Apply small update in gradient direction
5. Verify improvement

**Advantage:** Mathematically principled — the gradient tells you exactly which weights matter and in which direction to change them.

**Challenge:** This requires maintaining the training infrastructure (optimizer state, gradient computation) alongside the serving infrastructure. On DGX Spark's 128GB, simultaneously serving the model and computing gradients may not fit.

### 5d. Curriculum Scaling (incremental, practical)

Expand the training data iteratively:
1. Deploy current SFT model
2. Run it on self-modification tasks, collect successful + failed attempts
3. Filter for quality (successful modifications with verified improvements)
4. Fine-tune next iteration on expanded dataset
5. Repeat

This is a poor-man's RL — the "reward" is binary (did the modification work?), and only successful trajectories are kept. Less principled than RL but much cheaper computationally.

## 6. Recommendation

**Short-term (this week):**
1. Complete current SFT run (200 steps, ~4h remaining)
2. Merge LoRA → FP8 quantize → deploy
3. Evaluate: can the model produce valid neuroplastic API calls and reasonable modification proposals?

**Medium-term (next 2 weeks):**
1. Explore Mamba state persistence (5b) — cheapest experiment with potentially highest insight
2. Begin curriculum scaling (5d) — deploy SFT model, collect new self-modification transcripts, filter for quality, retrain

**Long-term (month+):**
1. RL fine-tuning (5a) once the reward infrastructure is validated through curriculum scaling
2. Gradient-guided modification (5c) as a research spike — may reveal that explicit self-modification is redundant when you can just do targeted online learning

The key strategic question: **is explicit weight editing the right abstraction for self-modification?** The Mamba state persistence path (5b) and gradient-guided path (5c) both suggest that the model's native mechanisms (SSM state, gradients) may be more powerful than an external API for weight manipulation. The SFT we're doing now is necessary groundwork regardless — it gives the model architectural self-awareness that all future approaches benefit from.
