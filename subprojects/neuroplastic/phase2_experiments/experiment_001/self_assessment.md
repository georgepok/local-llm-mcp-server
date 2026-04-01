# Experiment 001: Nemotron Self-Assessment

**Date:** 2026-03-10

## Q1: Was your prediction accurate?

Nemotron acknowledged the prediction was "methodologically sound" — amplifying gate weights should sharpen the softmax. However, the prediction failed to account for the degree of expert homogeneity.

## Q2: Why did the modification have no effect?

Nemotron identified three reinforcing factors:

1. **Expert weight norms are near-identical** (CV 1-4%) — all 128 experts are almost interchangeable
2. **Uniform scaling preserves ranking** — multiplying all gate vectors by 1.1 doesn't change which experts score highest for any given token, because it's a uniform operation
3. **Top-6 selection is already near-deterministic** — with only 4.7% utilization and similar routing scores, the same experts always win

Key insight: *"The MoE in this model is over-homogenised: the system has very little 'room' for routing to vary, so a global amplification cannot create new diversity."*

## Q3: Proposed next modification

Nemotron proposed **injecting a learned per-expert bias vector** (`b ∈ ℝ¹²⁸`) added to gate logits after the dot-product:

```python
gate_logits = gate_weight @ token_embedding
gate_logits = gate_logits + bias_vector  # NEW: break symmetry
gate_probs = softmax(gate_logits / temperature, dim=-1)
```

Rationale:
- A bias vector is cheap (only 128 extra scalars)
- It targets the router, not the expert weights
- It breaks the near-identical cosine similarity (0.2256) between routing vectors
- Can be combined with temperature reduction (e.g., 0.7) to amplify small differences

**Assessment of this proposal:** This requires runtime modification (adding a bias computation), not a static weight change. However, the bias could be "baked in" by modifying the gate weight matrix itself — adding a bias row or adjusting individual row norms to create asymmetry. This is compatible with static tensor modification.

## Meta-observation

Nemotron's self-assessment is more sophisticated than its original proposal. After seeing the null result, it correctly identifies that the key limitation was not the modification *type* but the *uniformity* of the operation. This demonstrates genuine meta-learning about self-modification — the model is reasoning about *why* a change failed, not just proposing another change.
