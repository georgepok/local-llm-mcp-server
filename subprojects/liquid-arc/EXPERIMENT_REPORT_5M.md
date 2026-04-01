# LiquidARC 5M — Width Scaling Experiment Report

## Objective

Test whether the ~50% eval xform plateau observed in the 572K zero-scaffold model is a capacity bottleneck or an architectural/data limitation. Scale from d=256 (572K params) to d=768 (4.96M params) via pure width, keeping all other architecture identical.

## Setup

### Architecture

| Component | 572K (baseline) | 5M (this run) |
|-----------|----------------|---------------|
| d_model | 256 | 768 |
| d_metric | 64 | 192 |
| d_ffn | 512 | 1536 |
| Total params | 572K | 4,960,718 |
| Geo params | — | 1,329,604 (26.8%) |
| Other params | — | 3,631,114 (73.2%) |

Same architecture: single shared ContinuousDynamics applied 16x via Euler ODE integration with SDPA heat kernel routing. No structural changes — pure width scaling.

### Training Configuration

| Setting | Value | Notes |
|---------|-------|-------|
| batch_size | 16 | Same as 572K |
| lr | 1.7e-4 | Scaled: 3e-4 × sqrt(256/768) |
| warmup_steps | 1000 | 2x longer than 572K |
| max_steps | 30,000 | Same as 572K |
| n_ode_steps | 16 (fixed) | Randomization disabled to avoid torch.compile recompilation |
| torch.compile | enabled | d=768 is the max that fits Triton shared memory (101KB limit) |
| geo_loss_enabled | false | Zero-scaffold: CE from step 0 |
| tau_freeze_steps | 0 | TauNet learns immediately |
| cv_floor_target | 3.0 | Metric plasticity floor |
| cv_ceiling_target | 8.0 | **New**: prevents CV runaway (see Issues) |
| cv_floor_lambda | 0.1 | Floor/ceiling penalty weight |
| real_arc_mix_ratio | 0.3 | 30% real ARC, 70% procedural |
| curvature_lambda | 0.05 | |
| curriculum | Stage 1 (GLOBAL) 0-20K, Stage 2 (RELATIONAL) 20K+ | |

### Code Changes

1. **CV ceiling** added to `model.py`: penalizes metric CV above `cv_ceiling_target` with same quadratic hinge as floor. Config field `cv_ceiling_target` added to `config.py`. Required because the 5M model's metric diverged (CV 3→14→19→NaN) in the first run without a ceiling.

2. **Fixed ODE steps**: `ode_steps_min=ode_steps_max=16`. The randomized step count [12,20] caused torch.compile recompilation stalls of 30-60 minutes each at d=768 (9 possible step counts × long compile time). Trade-off: loses temporal invariance training.

### Infrastructure

- Hardware: DGX Spark (128GB unified memory, GH200)
- Container: `liquid-arc-ttt-v2`
- Output: `/workspace/liquid-arc/output_30m/`
- Memory usage: ~1.6 GB estimated (very comfortable)
- Throughput: ~3,200 tok/s steady state (vs ~9,500 tok/s at 572K)

## Issues Encountered

### 1. Triton Shared Memory OOM (d=1920)

Original plan targeted d=1920 (~30M params). torch.compile failed:
```
OutOfMemoryError: out of resource: Required: 213072 Hardware limit: 101376
```
Triton fused layer norm kernel exceeds shared memory at d=1920. Reduced to d=768 (largest that fits). d=1920 without torch.compile would be ~380 tok/s — impractical.

### 2. CV Runaway → NaN (first run)

First 5M run hit NaN at step 6300. Root cause: CV floor penalty (push CV above 3.0) had no ceiling. At d=768 with near-zero curvature (|k|≈0.0004), the CV floor penalty was the dominant gradient on the metric. Once CV passed 3.0, the metric weights had been pushed into a high-variance regime with no restoring force.

```
step=6050: CV=5.37
step=6100: CV=6.20
step=6150: CV=8.62
step=6200: CV=14.56
step=6250: CV=19.60
step=6300: NaN
```

Fix: added CV ceiling at 8.0 (quadratic hinge penalty symmetric with floor).

### 3. torch.compile Recompilation Stalls

ODE step randomization [12,20] caused torch.compile to recompile for each new step count encountered at runtime. At d=768, each recompilation took 30-60+ minutes. Training appeared hung at step 4300 for over an hour.

Fix: fixed ODE steps to 16 (ode_steps_min = ode_steps_max = 16).

## Training Dynamics

### Phase 1: Plateau (Steps 0-5300)

| Step | Loss | Eval xform | CV | |k| | tau |
|------|------|-----------|-----|------|-----|
| 0 | 3.37 | — | 0.13 | 0.005 | 0.81 |
| 500 | 2.31 | 14.4% | 3.12 | 0.001 | 0.60 |
| 1000 | 2.29 | 19.2% | 3.05 | 0.0003 | 0.59 |
| 2500 | 2.22 | 18.7% | 3.27 | 0.0002 | 0.69 |
| 5000 | 2.32 | 16.1% | 6.62 | 0.0007 | 0.58 |

Loss plateaued at ~2.3 for 5000+ steps. Curvature near zero (|k| ≈ 0.0002). CV climbing steadily from floor (3.0) toward 7.0, driven by CV floor penalty. Model learning to copy cells but unable to distinguish transform cells.

Comparison: 572K model showed identical plateau at loss ~2.3, same duration.

### Phase 2: Phase Transition (Steps 5300-7700)

| Step | Loss | Train xform | CV | |k| |
|------|------|-----------|-----|------|
| 5350 | 2.24 | 6.8% | 7.65 | 0.001 |
| 5500 | 2.15 | 29.0% | 7.00 | 0.005 |
| 6000 | 1.69 | 31.0% | 7.12 | 0.029 |
| 7000 | 1.36 | 67.3% | 7.45 | 0.049 |
| 7500 | 0.45 | 87.6% | 6.19 | 0.007 |
| 7700 | 0.45 | 89.6% | 6.39 | 0.008 |

**Trigger**: CV reaching ~7.0. The metric developed enough variance for the SDPA heat kernel to produce non-uniform routing, breaking the copy-everything equilibrium. Curvature (|k|) only developed as a consequence, not a cause.

Comparison: 572K transition happened at step 5350 with CV ≈ 6.0. The 5M model needed CV ≈ 7.0 — higher threshold due to wider dimensions with Kaiming init producing smaller per-parameter metric deviations.

### Phase 3: Rapid Learning (Steps 7700-10000)

| Step | Loss | Train xform | Eval xform | CV | |k| |
|------|------|-----------|-----------|-----|------|
| 8000 | 0.42 | 86.0% | 41.1% | 6.36 | 0.007 |
| 8500 | 0.37 | 95.1% | 36.5% | 5.84 | 0.006 |
| 9000 | 0.53 | 87.3% | **48.2%** | 5.67 | 0.008 |
| 10000 | 0.48 | 85.6% | **48.5%** | 5.81 | 0.007 |

Train xform rapidly reached 85-95%. Eval xform climbed to ~48% — already matching the 572K model's peak. CV settled at 5.5-6.0, |k| stable at 0.006-0.008.

### Phase 4: Plateau & Curriculum Shift (Steps 10000-30000)

| Step | Eval xform | Eval cell_acc | Eval CE | CV | |k| |
|------|-----------|--------------|---------|-----|------|
| 10000 | 48.5% | 21.3% | 1.58 | 5.8 | 0.007 |
| 11500 | **49.7%** | 26.3% | 1.39 | — | — |
| 13000 | **52.9%** | 22.4% | 1.52 | — | — |
| 17000 | **53.3%** | 22.6% | 1.45 | — | — |
| 20000 | 46.8% | 24.9% | 1.53 | 5.3 | 0.009 |
| 21000 | **54.2%** | 27.7% | 1.60 | — | — |
| 25000 | 45.9% | 26.0% | 1.63 | 5.3 | 0.011 |
| 27500 | 49.8% | 24.6% | 1.50 | — | — |
| 28000 | 44.9% | 25.1% | 1.71 | — | — |

Eval xform oscillated between 42-54% with peak at **54.2%** (step 21K). Curriculum switched from Stage 1 (GLOBAL) to Stage 2 (RELATIONAL) at step 20K — no visible impact on eval.

**Peak eval xform: 54.2%** (step 21K) — modestly above 572K model's peak (~50%).

## TTT (Test-Time Training) Dynamics

| Step | Baseline xform | TTT xform | TTT lift | Phase |
|------|---------------|-----------|----------|-------|
| 2,500 | 18.7% | **43.6%** | **+24.9%** | Pre-transition |
| 5,000 | 16.1% | **44.6%** | **+28.5%** | Pre-transition |
| 7,500 | 37.2% | **38.5%** | +1.3% | During transition |
| 10,000 | 48.5% | **37.6%** | -10.9% | Post-transition |
| 12,500 | 46.6% | **33.9%** | -12.7% | Plateau |
| 15,000 | 45.6% | **33.9%** | -11.7% | Plateau |
| 17,500 | 48.2% | **33.5%** | -14.7% | Plateau |
| 20,000 | 46.8% | **28.2%** | -18.6% | Curriculum stage 2 |
| 22,500 | 47.6% | **27.3%** | -20.3% | Plateau |
| 25,000 | 45.9% | **26.4%** | -19.5% | Plateau |
| 27,500 | 49.8% | **24.2%** | -25.6% | Plateau |

### TTT Analysis

**Pre-transition (steps 0-5K)**: TTT provides dramatic 2.3-2.8x lift. With just 100 gradient steps on a single task, TTT achieves what training takes 7000+ steps to reach. This works because TTT specializes the metric to one task (coherent gradient signal) while training must find universal structure across all tasks (conflicting gradients).

**Post-transition (steps 7.5K+)**: TTT becomes increasingly destructive, eventually reducing xform by 25 percentage points. Once training discovers universal routing structure (CV ≈ 5.5-6.0), TTT's 100 per-task gradient steps overwrite this structure with task-specific noise. The TTT learning rate (0.001) and step count (100) are too aggressive for fine-tuning already-good geometry.

**Monotonic degradation**: TTT lift worsens continuously from +28.5% to -25.6% over training. This is not a phase — it's the model progressively learning better universal structure that TTT progressively destroys more of.

## Geometric Dynamics

### Metric CV Trajectory

```
Step 0-150:     0.13 → 3.0   (rapid rise from CV floor penalty)
Step 150-5000:  3.0 → 6.6    (gradual climb, penalty-driven + task-driven)
Step 5000-5500: 6.6 → 7.6    (acceleration before phase transition)
Step 5500-8000: 7.0 → 8.3    (peak during transition, ceiling engages)
Step 8000+:     settles 5.0-6.0 (task-driven equilibrium)
```

The CV ceiling (8.0) engaged briefly during the phase transition (step 7950: CV=8.26) and successfully prevented the runaway that killed the first run. Post-transition, CV naturally settled at 5.0-6.0 without needing the ceiling.

### Curvature (|k|) Trajectory

```
Step 0-5000:    0.005 → 0.0002  (collapsed to near-zero)
Step 5000-7000: 0.0007 → 0.049  (rapid development during transition)
Step 7000-10K:  0.006-0.008     (stabilized)
Step 10K-28K:   0.007 → 0.015   (slow continued growth)
```

Curvature was not a driver of the phase transition — it developed as a consequence of learning. The 5M model's pre-transition |k| was 3x lower than the 572K model's at the same step (0.0002 vs 0.0006), likely due to Kaiming init scaling: wider layers produce smaller initial metric deviations.

### Tau Trajectory

```
Step 0:    0.81 (near-max, just initialized)
Step 500:  0.60 (rapid drop)
Step 2500: 0.57-0.69 (settled)
Step 8K+:  0.57-0.65 (stable, slightly lower post-transition)
```

Tau dropped to ~0.58 and remained stable, matching the 572K model. TauNet converged quickly regardless of model scale.

## Comparison: 5M vs 572K Zero-Scaffold

| Metric | 572K | 5M |
|--------|------|-----|
| Phase transition step | ~5,350 | ~5,500 |
| CV at transition | ~6.0 | ~7.0 |
| Post-transition train xform | 80-95% | 80-95% |
| **Peak eval xform** | **~50%** | **~54%** |
| Peak eval cell_acc | ~29% | ~28% |
| Post-transition CV | 4.0-5.0 | 5.0-6.0 |
| Post-transition |k| | 0.01-0.02 | 0.006-0.015 |
| Post-transition tau | 0.55-0.58 | 0.57-0.65 |
| TTT peak lift | +16% (step 5K) | +28.5% (step 5K) |
| TTT post-transition | destructive | destructive |
| Throughput | ~9,500 tok/s | ~3,200 tok/s |

## Key Findings

### 1. The ~50% eval xform ceiling is NOT a capacity bottleneck

8.5x more parameters produced only a marginal improvement (50% → 54%). Both models hit the same ceiling. The 40-point train/eval gap (90%+ train vs 50% eval) confirms the model can compute transformations but can't generalize from procedural training to real ARC eval tasks.

### 2. Phase transition is CV-driven, not curvature-driven

Both models transitioned when metric CV reached a threshold (~6-7). Curvature (|k|) was near zero before the transition and developed after. The transition represents the metric developing enough variance for SDPA heat kernel routing to differentiate positions — breaking the copy-everything equilibrium.

### 3. TTT is a pre-transition phenomenon

TTT provides dramatic lift only before the model has learned universal routing structure. Post-transition, TTT is destructive and gets progressively worse. This suggests TTT at current hyperparameters (lr=0.001, 100 steps) is too aggressive to fine-tune already-trained geometry.

### 4. CV ceiling is necessary at wider scales

The 572K model's CV self-regulated via task gradients. The 5M model's metric, with 3x smaller initial deviations from Kaiming init, was dominated by the CV floor penalty gradient before the phase transition, causing runaway. The ceiling at 8.0 successfully bounded CV during the dangerous pre-transition phase without constraining post-transition dynamics.

### 5. Fixed ODE steps don't hurt at this scale

Disabling ODE step randomization (fixed 16 steps) showed no visible impact on training dynamics compared to the 572K model's randomized [12,20] range. The practical benefit (eliminating 30-60 minute recompilation stalls) far outweighed the theoretical temporal invariance benefit.

## Extended Training (Steps 30K-50K)

Training was resumed from the step 30K checkpoint (`final.pt`) with `--resume` support added to `train.py`. The `_orig_mod.` key prefix from torch.compile'd checkpoints is stripped automatically on load.

### Configuration

Same as initial run. Curriculum at Stage 2 (RELATIONAL, 11 rules) for the entire extension. Target: 100K steps. Killed at step ~50K due to eval stagnation.

### Eval Trajectory (Extended)

| Step | Eval xform | Eval cell_acc | Eval CE |
|------|-----------|--------------|---------|
| 30,000 | 43.4% | 27.4% | 1.67 |
| 32,500 | **55.6%** | 22.0% | 1.48 |
| 35,000 | 46.3% | 23.6% | 1.54 |
| 37,500 | 45.2% | 23.3% | 1.71 |
| 40,000 | 44.1% | 25.5% | 1.71 |
| 42,500 | 50.3% | 26.1% | 1.68 |
| 45,000 | 50.4% | 26.6% | 1.62 |
| 47,500 | 46.6% | 23.5% | 1.75 |
| 50,000 | 47.8% | 31.9% | 1.89 |

Eval xform remained flat at ~48% (range 43-56%). No improvement over 20K additional steps.

### Eval CE Degradation

The most concerning signal: eval CE steadily worsened from 1.50 (step 10K) to 1.89 (step 50K). The model became more confident on wrong predictions — a clear sign of overfitting to the RELATIONAL procedural distribution at the expense of real ARC generalization.

| Window | Avg Eval CE |
|--------|------------|
| Steps 10-15K | 1.50 |
| Steps 20-25K | 1.60 |
| Steps 30-35K | 1.60 |
| Steps 40-45K | 1.70 |
| Steps 45-50K | 1.85 |

### TTT Trajectory (Extended)

| Step | Baseline xform | TTT xform | TTT lift |
|------|---------------|-----------|----------|
| 30,000 | 43.4% | 29.3% | -14.1% |
| 35,000 | 46.3% | 34.2% | -12.1% |
| 40,000 | 44.1% | 29.8% | -14.3% |
| 45,000 | 50.4% | 31.2% | -19.2% |
| 50,000 | 47.8% | 31.7% | -16.1% |

TTT remained consistently destructive throughout the extension, reducing xform by 12-19 percentage points.

### Geometric Dynamics (Extended)

| Metric | Step 30K | Step 50K | Trend |
|--------|----------|----------|-------|
| CV | 5.2 | 5.0-5.5 | Stable |
| \|k\| | 0.012 | 0.017 | Slow linear climb |
| tau | 0.58 | 0.58-0.66 | Stable |

No geometric reorganization observed. |k| increased linearly but without acceleration — no precursor to a second phase transition. The geometry settled into a stable configuration that was not evolving toward better generalization.

### Training Dynamics (Extended)

Train xform on RELATIONAL tasks: 60-90% (improving). The model continued learning harder procedural tasks, but this training signal was counterproductive for eval. The widening train/eval gap (90% train vs 48% eval) confirmed that procedural RELATIONAL tasks do not transfer to real ARC.

### Decision to Kill

Training was terminated at step ~50K (of planned 100K) because:
1. Eval xform flat for 40K steps (no improvement since step 10K)
2. Eval CE actively degrading (1.50 → 1.89) — model moving away from eval distribution
3. No geometric precursors to a phase transition (stable CV, linear |k|, stable tau)
4. 50K more steps of the same trajectory would further degrade generalization

### Artifacts Preserved

All artifacts on Spark at `/workspace/liquid-arc/output_30m/`:
- **18 checkpoints**: steps 2.5K, 5K, 7.5K, 10K, 12.5K, 15K, 17.5K, 20K, 22.5K, 25K, 27.5K, 30K, 35K, 40K, 45K, 50K, best.pt, final.pt (1GB total)
- **train.log**: steps 0-30K (initial run)
- **train_resume.log**: steps 30K-50K (extended run)
- **logs/**: TensorBoard event files

## Key Findings

### 1. The ~50% eval xform ceiling is NOT a capacity bottleneck

8.5x more parameters produced only a marginal improvement (50% → 54%). Both models hit the same ceiling. The 40-point train/eval gap (90%+ train vs 50% eval) confirms the model can compute transformations but can't generalize from procedural training to real ARC eval tasks.

### 2. Phase transition is CV-driven, not curvature-driven

Both models transitioned when metric CV reached a threshold (~6-7). Curvature (|k|) was near zero before the transition and developed after. The transition represents the metric developing enough variance for SDPA heat kernel routing to differentiate positions — breaking the copy-everything equilibrium.

### 3. TTT is a pre-transition phenomenon

TTT provides dramatic lift only before the model has learned universal routing structure. Post-transition, TTT is destructive and gets progressively worse. This suggests TTT at current hyperparameters (lr=0.001, 100 steps) is too aggressive to fine-tune already-trained geometry.

### 4. CV ceiling is necessary at wider scales

The 572K model's CV self-regulated via task gradients. The 5M model's metric, with 3x smaller initial deviations from Kaiming init, was dominated by the CV floor penalty gradient before the phase transition, causing runaway. The ceiling at 8.0 successfully bounded CV during the dangerous pre-transition phase without constraining post-transition dynamics.

### 5. Fixed ODE steps don't hurt at this scale

Disabling ODE step randomization (fixed 16 steps) showed no visible impact on training dynamics compared to the 572K model's randomized [12,20] range. The practical benefit (eliminating 30-60 minute recompilation stalls) far outweighed the theoretical temporal invariance benefit.

### 6. Extended training degrades generalization

Continued training beyond step 10K did not improve eval xform but actively degraded eval CE (1.50 → 1.89). The RELATIONAL procedural curriculum provides training signal that is counterproductive for real ARC transfer. The model overfits to procedural-specific features at the expense of universal structure.

## Next Steps

The bottleneck is generalization, not capacity or training duration. Promising directions:

1. **Increase real ARC mix ratio** (0.3 → 0.5+): the most direct way to close the train/eval distribution gap
2. **Expand procedural rule set**: current 13 rules may not cover the transformation types needed for real ARC eval
3. **Adaptive TTT**: reduce lr/steps post-transition so TTT makes small refinements rather than overwriting universal structure
4. **Depth scaling**: try unshared dynamics (different weights per ODE step) instead of width — this would let different steps learn different transformation types
5. **Early stopping**: best eval performance was at step 10-15K; extended training is harmful. Future runs should stop when eval CE starts rising
