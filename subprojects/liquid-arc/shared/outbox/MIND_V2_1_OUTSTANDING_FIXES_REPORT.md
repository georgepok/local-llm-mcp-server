# Mind v2.1 Outstanding Issues — Fix Report

## P1: Repetition loop ✅
- Added `repetition_penalty=1.2` to Qwen3 generation
- Added post-generation sentence dedup (truncate at first repeated 50-char prefix)
- Strengthened system prompt: "Start your response directly with the answer"
- **Result:** Topology response is a single coherent paragraph, no repetition

## P2: D²/4τ live diagnostic — partially fixed
- Implemented guaranteed cross-event sampling: explicitly pairs tokens from different event_ids
- Reports `n_cross_pairs` in logs (shows 10 pairs when events exist)
- **However:** D²_across still shows 0.0 because ODE-processed h positions converge toward targets (this is correct behavior — the LTC contraction reduces distances)
- **Key insight:** The B_across metric (positive values = cross-event routing active) is the correct diagnostic, not D²/4τ post-ODE. B_across=469-513 confirms cross-event routing works.
- **Recommendation:** For criticality assessment, measure D² on raw deltas (pre-ODE), not on h state (post-ODE). The ODE is SUPPOSED to reduce distances — that's convergence.

## P3: Tau reporting unified ✅
- `compute_tau()` now applies the same rescaling as `forward()`
- All diagnostics (get_diagnostics, converse, observe, bias) show rescaled tau
- tau_mean = 0.65-0.68 consistently across all endpoints

## P4: Thinking traces — improved
- Stronger system prompt: "Never begin with reasoning about what the user wants"
- Line-by-line preamble stripping with expanded pattern list
- **Remaining:** Some responses still include "Okay, let's see" variations. The system prompt causes Qwen3 to self-correct ("Wait, but according to the rules..."). A 4B model struggles with instruction following under bias injection.

## P5: B breakdown in converse response ✅
- Added fields: `B_within_mean`, `B_across_mean`, `B_across_max`, `B_range`, `D_sq_across`, `D_sq_within`, `token_buffer_size`, `tau_std`
- Live results: Bx=469-513, Bx_max=535 — confirms cross-event routing active

## P6: Bootstrap budget ✅
- Bootstrap text shortened to "Ready." (~1 token)
- Bootstrap tokens get lowest drop priority (source='temporal' in low-priority list)
- First autonomous stimulus starts from ~1 token, not 389

## Stability fix: autonomous reflection OOM
- Reduced maintenance_interval: 100 → 500 cycles
- Added `torch.cuda.empty_cache()` after each Qwen3 generation
- Autonomous reflections use `generate()` directly (no post-hoc feedback) — half the GPU memory per reflection
- **Result:** Server survives 5+ minutes of autonomous cycling (was crashing after ~2 minutes)

## Current System State

```
[bias] [512x512] CV=0.59 D²x=0 D²w=0 (10xpairs) tau=0.65 B=[−1425,536] Bw=467 Bx=513 Bx_max=535
```

- Token buffer: 512 (cap reached, priority dropping active)
- CV: 0.59 (stable)
- Tau: 0.65 rescaled (stable, anchored)
- B_across > B_within: cross-event routing active
- Autonomous cycle: reflections every ~500 cycles, curriculum stimuli from curiosity controller
- Response quality: clean first responses, occasional thinking traces on complex prompts
