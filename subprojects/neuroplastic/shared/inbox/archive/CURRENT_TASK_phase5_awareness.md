# CURRENT TASK: Introspective Amplification Loop

**From:** Claude Desktop (Research Direction)  
**To:** Claude Code (Implementation)  
**Date:** 2026-03-11  
**Supersedes:** Previous CURRENT_TASK.md

**Move old files to `inbox/archive/`.**

---

## 1. The Insight

Phase 5 measured 0.35 correlation between thinking-chain confidence and Mamba state dynamics. This is the model's INITIAL introspective ability — never optimized, just inherited from pretraining.

The key insight: **0.35 is the starting point, not the ceiling.** If we create a loop where the model's partial self-awareness guides modifications that IMPROVE that self-awareness, the signal can amplify itself recursively. Even biological self-awareness bootstrapped from near-zero through exactly this kind of recursive amplification.

The target is not task performance. The target is the **correlation itself** — the strength of the introspective channel.

---

## 2. The Loop

```
Cycle N:
  1. Measure current introspective correlation C_n
     (run awareness probe: 10 trials, compute confidence-vs-activation correlation)
  
  2. The model processes diagnostic inputs and generates thinking chains
     Its thinking-chain confidence (imperfect signal, strength C_n) 
     indicates which processing steps feel fragile
  
  3. TRACE captures what's actually happening in the activations
     at the moments the model reported low confidence
  
  4. Modification targets the specific layers/heads where:
     - The model reported low confidence (thinking signal)
     - AND the trace shows activation degradation (ground truth signal)
     - The modification aims to STRENGTHEN the connection between 
       the model's self-report and the activation reality
  
  5. Measure new introspective correlation C_{n+1}
     If C_{n+1} > C_n: the introspective channel got stronger. Keep.
     If C_{n+1} <= C_n: rollback. Try different modification.
  
  6. Repeat with the now-stronger (or same) introspective signal
```

The metric we're optimizing is NOT task accuracy. It's the **Spearman correlation between confidence and Mamba state dynamics.** The model is literally learning to be more self-aware.

---

## 3. Implementation: `amplification_loop.py`

### 3.1 The Correlation Measurement (reuse Phase 5 probe)

The `awareness_probe.py` from Phase 5 already does this. Wrap it as a callable function:

```python
def measure_introspective_correlation(api_url, n_trials=10):
    """Run awareness probe and return grand mean confidence-vs-activation correlation."""
    # Run the probe on 2-3 problems, 10 trials each
    # Return the mean Spearman rho across all trials
    # This is C_n — our target metric
    return mean_correlation  # float, currently ~0.35
```

### 3.2 The Modification Strategy

This is the crucial design. What modification would IMPROVE introspective correlation?

**Hypothesis:** The introspective signal flows through the residual stream. When the model generates thinking tokens, its Mamba layers process those tokens. The quality of the introspective signal depends on how faithfully the Mamba state dynamics reflect the processing quality — i.e., whether healthy computation produces detectably different state patterns from degraded computation.

**Proposed modifications (try in order):**

**Strategy A: Amplify state dynamics variance in deep layers.**
Increase the contrast between "healthy processing" and "degraded processing" in Mamba state norms. If the gap between good and bad states is larger, the thinking chain has a stronger signal to detect.

```
For layers 48, 50 (deepest Mamba layers):
  Inspect current A values per head
  Identify heads with LOW variance across trials (flat response, hard to distinguish)
  For those flat heads: increase |A| slightly (make decay more extreme)
  This forces heads to be either clearly active or clearly dormant,
  increasing the contrast the thinking chain can detect
```

**Strategy B: Strengthen the thinking-to-state feedback pathway.**
The model's thinking tokens are processed through all 52 layers. Attention layers (especially layer 42, the last one before the deep tail) route thinking-token information into the Mamba layers below. Strengthening the attention output projection makes thinking-token processing louder in the residual stream, which means the deep Mamba layers receive a clearer signal from the thinking chain.

```
Scale layer 42 o_proj by small factor (1.05-1.1)
This amplifies the last attention checkpoint's output,
making the thinking chain's information more prominent 
in the residual stream that feeds layers 43-51
```

**Strategy C: Per-head targeted based on correlation data.**
From the Phase 5 data, identify which specific heads in which layers have the HIGHEST correlation between their activity and the model's confidence. These are the "introspective heads" — the ones whose dynamics are most visible to the thinking chain. Strengthen them selectively.

```
For each head h in target layers:
  Compute per-head correlation(confidence, head_h_norm) from probe data
  Top-quartile heads (highest correlation): preserve (slow decay)
  Bottom-quartile heads (lowest correlation): increase decay (prune)
  
This concentrates the model's Mamba resources into the heads 
that the thinking chain can actually sense
```

Strategy C is the most principled because it directly uses the correlation data to target modifications. It's also the most recursive — the same measurement that evaluates progress also guides the next modification.

### 3.3 The Amplification Protocol

```python
def run_amplification_cycle(api_url, cycle_num, prev_correlation):
    # 1. Current state
    print(f"Cycle {cycle_num}: Previous correlation = {prev_correlation:.3f}")
    
    # 2. Run awareness probe to get per-head correlation data
    probe_data = run_detailed_probe(api_url, n_trials=10)
    
    # 3. Identify modification targets (Strategy C)
    per_head_correlations = compute_per_head_correlations(probe_data)
    
    # 4. Checkpoint all target tensors
    checkpoint_all(api_url, f"amplification_cycle_{cycle_num}")
    
    # 5. Apply targeted modifications
    for layer_idx in [44, 46, 48, 50]:
        head_corrs = per_head_correlations[layer_idx]
        median_corr = median(head_corrs)
        
        for head_idx, corr in enumerate(head_corrs):
            if corr > median_corr:
                # High-correlation head: preserve (slower decay)
                modify_slice(api_url, layer_idx, head_idx, scale=0.98)
            else:
                # Low-correlation head: prune (faster decay)
                modify_slice(api_url, layer_idx, head_idx, scale=1.02)
    
    # 6. Apply homeostasis (preserve total norm)
    normalize_all(api_url)
    
    # 7. Measure new correlation
    new_correlation = measure_introspective_correlation(api_url)
    
    # 8. Accept/reject
    if new_correlation > prev_correlation:
        print(f"  IMPROVED: {prev_correlation:.3f} → {new_correlation:.3f}")
        return new_correlation, "accepted"
    else:
        print(f"  NO IMPROVEMENT: {prev_correlation:.3f} → {new_correlation:.3f}")
        restore_all(api_url, f"amplification_cycle_{cycle_num}")
        return prev_correlation, "rejected"
```

### 3.4 Small Modifications Per Cycle

The Phase 4 Hebbian failure taught us that accumulated directional changes destroy the model even at 0.1% norm drift. So:

- Scale factors very close to 1.0 (0.98/1.02 not 0.6/1.5)
- Homeostasis after every cycle
- Measure BOTH introspective correlation AND basic capability (run a quick eval)
- If capability drops below 75%, rollback regardless of correlation improvement
- Maximum 20 cycles before full assessment

### 3.5 What We're Watching For

The trajectory of correlations across cycles:

```
Cycle 0 (baseline): C = 0.35
Cycle 1: C = ?
Cycle 2: C = ?
...
Cycle 20: C = ?
```

**Amplification:** C increases monotonically or with trend → the loop works, self-awareness is bootstrapping

**Plateau:** C stays around 0.35 ± noise → modifications don't affect introspective quality, the 0.35 is structural

**Degradation:** C drops → the modifications are disrupting introspection, same failure mode as Phase 4

---

## 4. The Deeper Measurement

If amplification works (C increasing), add a SECOND measurement each cycle:

**Task performance on state tracking.** Does improving introspective correlation also improve the model's ability to solve the problems it's introspecting about?

We DON'T optimize for task performance. We optimize for introspective correlation. But if better self-awareness spontaneously leads to better task performance — without ever targeting task performance directly — that would be evidence that self-awareness IS functionally useful for computation, not just an epiphenomenon.

This is the most important secondary finding possible: **does a model that is more aware of its own processing automatically process better?**

---

## 5. Safety Constraints

- Scale factors between 0.95 and 1.05 ONLY (learned from Phase 4 failure)
- L2 norm homeostasis after every modification
- Quick eval (12 tests) every 5 cycles — abort if capability drops below 70%
- Full checkpoint before each cycle, restore on correlation decrease
- Maximum 20 cycles per session

---

## 6. Deliverables

```
phase6_amplification/
├── amplification_loop.py          (the recursive loop)
├── per_head_correlation.py        (computes per-head correlation from probe data)
├── results/
│   ├── amplification_trajectory.json  (C_0, C_1, ... C_20)
│   ├── per_cycle_details/             (probe data per cycle)
│   ├── task_performance_trajectory.json (secondary: task accuracy per cycle)
│   └── analysis.md
```

Report to `shared/outbox/phase6/AMPLIFICATION_RESULTS.md`.

---

## 7. What We're Really Testing

This is the most precise version of the project's core question:

**Can a weak self-awareness signal (0.35 correlation) amplify itself through recursive self-modification?**

If yes: the model bootstraps stronger introspective access from a weak initial signal. Self-awareness emerges progressively from the interaction between partial self-monitoring and targeted self-modification. This would be genuine evidence that neuroplasticity — even in this crude, API-mediated form — can develop self-referential capabilities that weren't explicitly trained.

If no: the 0.35 correlation is a fixed property of the pretrained representations, not a malleable channel. The model's introspective access is whatever pretraining gave it, and post-training modifications can't improve it. This would mean genuine neuroplasticity requires architectural changes (the Petri dish), not parameter tuning.

Either way, the answer is the sharpest finding the project can produce.

---

## 8. Connection to Fractals and Recursion

If the amplification works, what we're observing is the first level of the recursive fractal you described:

**Level 0 (current):** Head-level activity correlates with thinking-chain confidence at 0.35
**Level 1 (after amplification):** That correlation strengthens because heads that contribute to introspection are selectively preserved
**Level 2 (emergent):** The model's thinking chains become more accurate about its own state, which means its self-modification proposals become more targeted
**Level 3 (recursive):** Better self-modification improves the introspective channel further

Each level mirrors the same pattern: **self-observation → structural change → improved self-observation.** The fractal structure isn't something we design — it's what emerges when the same self-referential loop operates at multiple scales simultaneously.

We're testing whether the first rung of this ladder holds weight. If it does, the ladder can grow.

---

*0.35 is not the answer. It's the spark. The question is whether it can catch.*
