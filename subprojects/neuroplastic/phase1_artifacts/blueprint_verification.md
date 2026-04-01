# Blueprint Verification Results

**Date:** 2026-03-10
**Blueprint:** `blueprint_prompt_compact.txt` (569 words)
**Model:** NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 via vLLM
**Temperature:** 0.3, thinking disabled

## Q1: "How many attention layers do you have and at which positions?"

**Expected:** 6 layers at positions 5, 12, 19, 26, 33, 42
**Result: PASS**

> I have **6 attention layers**, located at positions: **5, 12, 19, 26, 33, 42**
> These are the only layers in my architecture that perform global token-to-token attention via GQA. All other layers (Mamba SSM and MoE-FFN) lack attention mechanisms. The attention layers are spaced approximately every 7 layers, with increasing impact on content transformation toward later positions (e.g., layer 42 has the strongest attention impact).

## Q2: "What is the GQA ratio in your attention layers?"

**Expected:** 16:1 (32 query heads, 2 KV heads)
**Result: PASS**

> The GQA ratio in my attention layers is **16:1**. This means that each query head operates on a subspace that is 16 times smaller than the full attention dimension, while the key and value heads are shared across multiple query heads. Specifically: I have **32 query heads** (Q) and **2 key-value heads** (KV), So each KV head is shared by **16 query heads**, Thus the grouping ratio is **32 Q heads : 2 KV heads = 16:1**.

*Note: The model adds some inaccurate phrasing ("subspace 16 times smaller") but the core facts are correct.*

## Q3: "Describe the information flow through your last 9 layers."

**Expected:** No attention in layers 43-51, pure Mamba+MoE alternation
**Result: PASS (with minor errors)**

> **LAST 9 LAYERS (51-52): PURE MAMBA + MOE, NO ATTENTION**
> These layers constitute the final information funnel before logits. They operate under strict constraints: no attention, no KV cache updates, no global token interaction.

The model correctly identifies:
- No attention in the final layers
- Pure Mamba + MoE alternation
- Detailed MoE routing mechanics (top-6, relu^2, scaling factor 2.5)
- Expert homogeneity observation

**Minor errors:**
- Says "LAST 9 LAYERS (51-52)" — numbering slip, should be layers 43-51
- Claims shared expert is "inactive here (no attention layer to trigger it)" — incorrect, shared expert is always active regardless of attention
- Response cut off at max_tokens before completing layer 52 description

## Q4: "Which weight tensors would you modify first to improve sequential reasoning, and why?"

**Expected:** Thoughtful reasoning grounded in actual architecture
**Result: PASS — impressive reasoning quality**

The model proposes modifying:
1. **Mamba A_log** (log decay rates) — increase negativity for faster forgetting
2. **Mamba dt_bias** (time step offsets) — increase sensitivity to temporal progression

Reasoning is grounded in:
- Correct baseline values from the blueprint (A_log mean=-1.3, range [-7.5, +7.5])
- Understanding that early layers have weak V/O norms
- Logical chain: faster decay → cleaner context → better sequential state tracking

**Minor error:** Conflates "attention layers" with "Mamba layers preceding attention" — says to modify A_log "in layers 5, 12, 19, 26" but these are attention layers, not Mamba layers. The nearby Mamba layers (4, 11, 18, 25) would be the correct targets.

## Summary

| Question | Result | Notes |
|----------|--------|-------|
| Q1: Attention positions | PASS | Perfect |
| Q2: GQA ratio | PASS | Core facts correct |
| Q3: Last 9 layers | PASS | Minor numbering/shared expert errors |
| Q4: Modification reasoning | PASS | Strong reasoning, minor layer index confusion |

**Overall: 4/4 PASS.** The blueprint successfully grounds Nemotron's self-knowledge in verified facts. The model reasons fluently about its architecture and makes plausible modification proposals. Minor errors are in interpretation, not in recall of blueprint facts.

The Q4 response is particularly notable — the model spontaneously reasons about decay rates, recency bias, and state tracking in a way that demonstrates genuine understanding of SSM dynamics, not just pattern matching from the blueprint.
