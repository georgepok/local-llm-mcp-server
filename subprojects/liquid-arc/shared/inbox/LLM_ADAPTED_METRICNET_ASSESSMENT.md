# LLM-ADAPTED METRICNET RESULTS — Assessment & Path Forward

## What Was Validated

The MetricNet adaptation to LLM residuals WORKS:

| Metric | ARC checkpoint | LLM-adapted | Interpretation |
|--------|---------------|-------------|---------------|
| CV | 0.51 (flat) | 3.87 (structured) | 7.6× improvement — real geometric differentiation |
| tau trajectory | 2.0 constant | 2.7→0.8 depth-dependent | TauNet discovered LLM depth structure |
| correction_ratio | 1.8% | 3.0% | Slightly stronger perturbation |
| B_range | 9.5 | 9.2 | Similar bias dynamic range |

The depth-dependent tau is the strongest finding: the TauNet learned from LLM residual statistics alone that early layers need slow integration (τ=2.7, careful) and late layers need fast integration (τ=0.8, aggressive). This is a genuine structural discovery about LLM computation — not hand-designed.

## Why the Causal Chain Test Doesn't Show Improvement

The 5-hop earthquake chain tests REASONING DEPTH, not ATTENTION ROUTING:

```
Attention routing:  Which tokens influence which (what geometric bias changes)
Reasoning depth:    How many inferential steps the model chains (4B model capacity limit)
```

Qwen3-4B traces to "evacuation" (3 hops) and stops — regardless of bias. This is the model's reasoning chain terminating, not an attention failure. A 3% correction to attention routing cannot add 2 more reasoning hops.

The 2-3 hop chains (which both plain and ODE get right) confirm this: when the reasoning is within the model's capacity, the answer is correct with or without bias. The geometric routing doesn't add or remove reasoning ability.

## The Right Evaluation Path

### Option A: Find tasks where routing IS the bottleneck

Tasks where flat attention causes WRONG ANSWERS (not insufficient reasoning):

1. **Long-context with distractors:** 2000+ token prompt, answer at position 200, misleading content at position 1500. Flat attention weights both. Geometric bias suppresses distractor.

2. **Entity scope confusion:** Two entities with the same name in different contexts. Flat attention bleeds attributes. Geometric routing separates the clusters.

3. **Long-range retrieval degradation:** At 500 tokens, ICL handles everything. At 5000+ tokens, attention becomes diffuse. The bias could maintain structure that raw attention loses at scale.

4. **Parallel interference at scale:** The pesticide/rice test worked at short context. At 20+ events with multiple overlapping causal chains, flat attention increasingly confuses chains. Geometric routing should maintain separation.

### Option B: End-to-end CE training (RECOMMENDED)

The current approach: train MetricNet on LLM residual statistics → produce generic structured geometry → inject as bias → hope it helps generation.

What's missing: the MetricNet doesn't know what routing changes HELP prediction. It produces structure (CV=3.87) but not necessarily USEFUL structure.

End-to-end CE training:
```
Forward: text → Qwen3 layers + ODE bias at each layer → logits
Loss: CE(logits, target_tokens)
Gradient: CE → attention logits → bias → ODE correction → MetricNet weights
```

The CE loss tells the MetricNet: "this routing change improved/hurt next-token prediction." The MetricNet learns task-relevant routing, not just generic geometry.

This requires:
1. Differentiable bias injection (the perturbation architecture already supports this — the correction flows through differentiable operations)
2. Frozen Qwen3 weights (only MetricNet/TauNet train)
3. A training dataset (WikiText for generic NTP, or reasoning tasks for targeted improvement)
4. The criticality scaffolding active during training (D²/4τ, tau_quality)

### Option C: Stronger coupling (quick test)

Increase ε from 0.1 to 0.5-1.0 to see if larger perturbation changes outcomes. Risk of instability at high ε, but worth testing as a quick experiment before committing to end-to-end training.

## Recommended Priority

1. **Quick: ε sweep** (0.2, 0.5, 1.0) with the LLM-adapted checkpoint — determine if stronger coupling helps on any existing test
2. **Quick: Alternative tasks** — long-context retrieval with distractors, entity scope confusion — determine if there are tasks where even current weak routing matters
3. **Main: End-to-end CE training** — the path that makes the MetricNet learn USEFUL routing, not just generic structure

## The Deeper Insight

The project has arrived at a clean separation:
- **Architecture:** Validated. Layer-wise perturbation, sustained criticality, bias injection — all working.
- **Training for structure:** Validated. MetricNet adapts to LLM residuals (CV=3.87, depth-dependent tau).
- **Training for utility:** NOT YET DONE. The MetricNet produces structure but doesn't know what structure HELPS.

End-to-end CE training bridges the gap from "structured geometry" to "useful geometry." This is the final step.
