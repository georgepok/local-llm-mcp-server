# CURRENT TASK: Seed Sweep — Test Resonance Hypothesis and Find New Transitions

## Why This Is the Highest Priority

We have ONE post-transition checkpoint that we can never recreate. Every experiment we run consumes it (modifies optimizer state, potentially degrades). A seed sweep could produce ADDITIONAL post-transition checkpoints — each one is priceless.

Simultaneously, it tests the resonance hypothesis: if 5-20% of seeds produce transitions and the rest don't, the transition depends on the numerical trajectory (supporting resonance). If 0% produce transitions, something environmental changed since the original runs. If 80%+ produce transitions, the recipe actually works and the clean reproduction failure was a fluke.

## What to Do

Run 20 identical training runs with different random seeds. Each run uses the EXACT same config (the documented 30% recipe). The ONLY difference is the seed. Each run goes for 8000 steps (enough to see if the transition fires at ~5400).

### Script: `scripts/seed_sweep.py`

```python
"""Seed sweep: run N identical training runs with different seeds.

Tests whether the phase transition is seed-dependent (resonance hypothesis)
or environment-dependent (something else changed).

Each run: 8000 steps, 30% ARC, standard zero-scaffold config.
Transition expected at step ~5400 if it fires.
"""

import subprocess
import sys
import os
import json
import time

N_SEEDS = 20
MAX_STEPS = 8000
CONFIG = "configs/liquid_arc_zero_scaffold.yaml"
BASE_OUTPUT = "output_seed_sweep"

results = []

for seed in range(N_SEEDS):
    output_dir = f"{BASE_OUTPUT}/seed_{seed:03d}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"SEED {seed}/{N_SEEDS-1} — output: {output_dir}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    cmd = [
        sys.executable, "scripts/train.py",
        "--config", CONFIG,
        "--data_dir", "/workspace/fgn-v3/data/arc-repo/data",
        "--output_dir", output_dir,
        "--max_steps", str(MAX_STEPS),
        "--seed", str(seed),
        "--log_every", "100",
        "--eval_every", "1000",
        "--save_every", "2500",  # save checkpoints at 2500, 5000, 7500
    ]
    
    # Run training, capture output
    log_file = os.path.join(output_dir, "train.log")
    with open(log_file, 'w') as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    
    elapsed = time.time() - start_time
    
    # Parse results from log
    cv_at_5000 = None
    eval_xform_at_5000 = None
    eval_xform_at_7500 = None
    max_cv = 0.0
    transition_fired = False
    
    with open(log_file, 'r') as f:
        for line in f:
            # Extract CV values
            if 'cv=' in line:
                try:
                    cv_str = line.split('cv=')[1].split(',')[0].split('|')[0].strip()
                    cv = float(cv_str)
                    max_cv = max(max_cv, cv)
                    
                    # Check step
                    if 'step=' in line:
                        step_str = line.split('step=')[1].split(']')[0]
                        step = int(step_str)
                        if step == 5000:
                            cv_at_5000 = cv
                except (ValueError, IndexError):
                    pass
            
            # Extract eval xform
            if 'EVAL' in line and 'xform_acc' in line:
                try:
                    xform_str = line.split('xform_acc=')[1].split(',')[0]
                    xform = float(xform_str)
                    if 'step=5000' in line:
                        eval_xform_at_5000 = xform
                    if 'step=7500' in line or 'step=8000' in line:
                        eval_xform_at_7500 = xform
                except (ValueError, IndexError):
                    pass
    
    # Did the transition fire?
    # Criteria: max CV > 5.0 AND eval xform at 7500 > 35%
    if max_cv > 5.0 and eval_xform_at_7500 is not None and eval_xform_at_7500 > 0.35:
        transition_fired = True
    
    result = {
        "seed": seed,
        "max_cv": max_cv,
        "cv_at_5000": cv_at_5000,
        "eval_xform_5000": eval_xform_at_5000,
        "eval_xform_7500": eval_xform_at_7500,
        "transition_fired": transition_fired,
        "elapsed_seconds": elapsed,
    }
    results.append(result)
    
    status = "TRANSITION ✓" if transition_fired else "no transition"
    print(f"  Seed {seed}: max_cv={max_cv:.2f}, xform@7500={eval_xform_at_7500}, {status} ({elapsed:.0f}s)")

# Summary
print(f"\n{'='*60}")
print(f"SEED SWEEP SUMMARY")
print(f"{'='*60}")

n_transitioned = sum(1 for r in results if r["transition_fired"])
print(f"Transitions: {n_transitioned}/{N_SEEDS} ({n_transitioned/N_SEEDS*100:.0f}%)")
print(f"\nPer-seed results:")
print(f"{'Seed':>5} {'MaxCV':>7} {'CV@5K':>7} {'Xform@5K':>10} {'Xform@7.5K':>12} {'Transitioned':>13}")
for r in results:
    print(f"{r['seed']:5d} {r['max_cv']:7.2f} {r['cv_at_5000'] or 0:7.2f} "
          f"{r['eval_xform_5000'] or 0:10.4f} {r['eval_xform_7500'] or 0:12.4f} "
          f"{'YES ✓' if r['transition_fired'] else 'no':>13}")

# Save results
with open(os.path.join(BASE_OUTPUT, "sweep_results.json"), 'w') as f:
    json.dump(results, f, indent=2)

# If any transitioned, highlight them
if n_transitioned > 0:
    print(f"\n*** TRANSITIONS FOUND! Seeds: {[r['seed'] for r in results if r['transition_fired']]}")
    print(f"*** Checkpoints saved at: {[f'{BASE_OUTPUT}/seed_{r[\"seed\"]:03d}/checkpoints/' for r in results if r['transition_fired']]}")
    print(f"*** BACK THESE UP IMMEDIATELY — they are irreplaceable")
else:
    print(f"\n*** No transitions in {N_SEEDS} seeds.")
    print(f"*** This suggests environmental factors (not seed) determine transition.")
```

**IMPORTANT implementation note:** The training script needs to accept a `--seed` argument and use it to seed:
- `random.seed(seed)`
- `torch.manual_seed(seed)`
- `torch.cuda.manual_seed_all(seed)`
- `np.random.seed(seed)` (if numpy is used)

Check `scripts/train.py` for existing seed handling. If there's no `--seed` argument, add one. If there is, make sure it seeds ALL random sources.

### Alternative: Parallel Execution

If the Spark has enough memory for 2-3 concurrent runs (the model is tiny — 572K params, ~50MB each):

```bash
# Run 4 at a time in background
for seed in 0 1 2 3; do
    python scripts/train.py --config configs/liquid_arc_zero_scaffold.yaml \
      --data_dir /workspace/fgn-v3/data/arc-repo/data \
      --output_dir output_seed_sweep/seed_$(printf "%03d" $seed) \
      --max_steps 8000 --seed $seed --log_every 100 --eval_every 1000 --save_every 2500 \
      > output_seed_sweep/seed_$(printf "%03d" $seed)/train.log 2>&1 &
done
wait

# Then next 4
for seed in 4 5 6 7; do
    # ...same pattern
done
```

But torch.compile might not work well with multiple processes sharing the GPU. Serial execution is safer. Each run takes ~25 minutes (8K steps), so 20 runs = ~8 hours serial.

**If time is limited:** Run 10 seeds instead of 20. Or run only 8000 steps instead of 15000 (the transition fires at ~5400, so 8000 is enough to detect it).

### Config

Use `configs/liquid_arc_zero_scaffold.yaml` UNCHANGED. The whole point is to test the exact recipe that produced the original transitions. No modifications.

### What to Look For

**1. How many seeds produce transitions?**
```
0/20:    Environmental factors changed. The recipe worked on the original PyTorch/CUDA 
         but doesn't work on the current environment. Resonance hypothesis uncertain.

1-4/20:  Resonance hypothesis strongly supported. The transition is seed-dependent —
         some numerical trajectories create resonance, most don't. Each transitioning 
         seed is a PRECIOUS new checkpoint.

5-10/20: The recipe is moderately robust but sensitive to initialization.
         The resonance window is broader than expected.

15+/20:  The recipe actually works! The clean reproduction failure was a fluke
         (maybe a code issue we missed). The transition is reproducible.
```

**2. CV trajectory patterns in non-transitioning seeds**

For seeds that DON'T transition, examine the CV trajectory:
- Does CV get stuck at exactly 3.0? (floor penalty holding it)
- Does CV oscillate around 3.0-3.5? (trying to build but failing)
- Does CV reach 4.0-5.0 but not cross the threshold? (close but not resonant)

If most non-transitioning seeds reach CV 4.0-5.0 but stall, the resonance window is narrow — the system ALMOST transitions but the forcing frequency is slightly off.

If most non-transitioning seeds stay at CV 3.0, the build signal is too weak in most numerical trajectories.

**3. Transition timing in successful seeds**

If multiple seeds transition:
- Do they all transition at ~step 5400? (same mechanism)
- Do they transition at different steps? (different natural frequencies → different resonance buildup times)
- Is there correlation between CV-at-step-2500 and whether the transition fires? (early indicator)

### After the Sweep

**If transitions are found:**
1. IMMEDIATELY back up the transitioning checkpoints
2. Compare the CV trajectories of transitioning vs non-transitioning seeds
3. Look for early predictors (any metric at step 1000-2000 that predicts transition at step 5400)
4. Run the multi-domain ratcheting experiment on a NEW transitioning checkpoint (preserves the original)

**If NO transitions are found:**
1. The environment changed since the original runs
2. Try different PyTorch version, CUDA version, or disable torch.compile
3. Or: the original transitions required a specific seed range (try seeds 42, 123, 0, 1 — common defaults)
4. The multi-domain experiment on the existing checkpoint is still the right next step

### Output

Save all logs to: `output_seed_sweep/seed_XXX/train.log`
Save summary to: `output_seed_sweep/sweep_results.json`
Report to shared outbox: `SEED_SWEEP_REPORT.md`

Include:
- Transition count and percentage
- Per-seed CV trajectory summary
- Any early predictors of transition
- List of transitioning seeds and their checkpoint locations
- Assessment: is the transition seed-dependent (resonance) or environment-dependent?
