# Experiment 001: Nemotron's First Self-Modification Proposal

**Date:** 2026-03-10

## Self-Inspection (Q1): What does low expert CV mean?

Nemotron provided a sophisticated analysis:
- Expert differentiation is **directional, not magnitude-based** — experts differ in weight direction, not scale
- Routing is **entropy-driven and fragile** — small input changes flip expert assignments
- Functional specialization is **shallow** — no coherent "skills" per expert
- The shared expert and attention layers carry the real reasoning load, not routed experts
- Compared to high-CV MoE systems where experts develop distinct functional roles

## Proposed Modification (Q2)

**Target tensor:** `backbone.layers.30.mixer.gate.weight` (shape [128, 2688], float32)

**Proposed operation:** Structured sparsity via positional bucket masking — partition 128 experts into 8 groups of 16, mask gate logits so each token only routes to its assigned group.

**However:** This proposal requires runtime modification (position-dependent masking), not a static weight change. It cannot be implemented as a simple tensor modification.

## Simplified Version for Experiment 001

Given the constraints (static weight modification only, must be expressible as a tensor operation), we'll simplify Nemotron's proposal to its core insight: **increase gate weight differentiation to force more diverse expert routing.**

**Actual modification:** Scale the gate weight matrix to amplify existing directional differences between expert rows. Each row of the gate weight matrix is the routing vector for one expert. Amplifying the differences between rows encourages the router to make more decisive, differentiated routing decisions.

Concretely: normalize each gate row to unit norm, then scale by the original mean norm. This preserves the overall scale but maximizes directional differentiation.

Alternative (simpler): multiply gate weights by a small amplification factor (e.g., 1.1) to sharpen the softmax routing distribution, making expert selection more decisive.

## Prediction (Q3)

Nemotron's Q3 response switched to a different proposal (splitting shared expert), which is inconsistent with Q2. This reveals a limitation: the model doesn't maintain proposal coherence across multi-turn self-modification reasoning without explicit chain-of-thought.

## Decision

Proceed with the simpler gate weight amplification (multiply by 1.1) as the first experiment. This:
- Is a single static tensor operation
- Targets a float32 tensor (no FP8 requantization needed)
- Sharpens routing softmax without changing which experts exist
- Is easily reversible
- Tests the full Markov chain loop end-to-end
