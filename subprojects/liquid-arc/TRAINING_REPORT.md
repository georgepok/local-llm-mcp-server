# LiquidARC Geometric Training Report

**Run**: `output_geo` on DGX Spark
**Config**: `configs/liquid_arc_geo.yaml` (572K params, d=256, d_metric=64)
**Status**: Step 17,800 / 100,000 — still running
**Wall time**: ~72 minutes (14:20–15:32), ~17K tok/s average

---

## 1. Training Phases

| Phase | Steps | λ_ce | λ_geo | What Trains |
|-------|-------|------|-------|-------------|
| **1: Pure Geometry** | 0–5K | 0.0 | 1.0 | MetricNet only (all other grads zeroed) |
| **2: Object Islands** | 5K–15K | 0→1.0 ramp | 1.0 | All parameters |
| **3: The Release** | 15K–20K | 1.0 | 1.0→0.1 decay | All parameters |
| **4: Steady State** | 20K+ | 1.0 | 0.1 | All parameters |

---

## 2. Phase-by-Phase Dynamics

### Phase 1: Pure Geometry (steps 0–5K)

**Goal**: Force MetricNet to build a 2D grid manifold via MSE(model_D², squared_Manhattan).

**geo_MSE** — Rapid initial collapse, then plateau:
```
Step     0:  41,994   (random metric vs grid distances)
Step   200:   7,283   (10× reduction in 200 steps)
Step   300:     806
Step   500:     674
Step 1,000:     324
Step 2,000:     221
Step 3,000:     278   (plateaus around 150–300)
Step 5,000:     193
```

**Metric CV** — Grew from flat to highly structured:
```
Step     0:  0.05   (flat, uniform metric)
Step   200:  0.75
Step   500:  2.31
Step 1,000:  3.30
Step 2,000:  4.72
Step 5,000:  5.34   (rich metric variation)
```

**|κ| (curvature)** — Exponential growth, unconstrained:
```
Step     0:  0.002
Step   500:  0.096
Step 1,000:  3.8
Step 2,000:  900
Step 3,000:  3,012
Step 5,000:  5,397
```

**τ (time constant)**: Frozen at 1.0 (by design).

**Eval**: Random — cell_acc ~0.08, xform_acc ~0.11. Expected: output head is completely untrained.

**Assessment**: Phase 1 achieved its primary objective. geo_MSE dropped 200× from 42K to ~200, proving MetricNet can learn squared Manhattan distances. Metric CV reached 5.3, indicating strong spatial structure. However, curvature grew unchecked to 5,400 — the metric is becoming increasingly curved while matching distances, suggesting the learned manifold is non-trivially warped rather than flat Euclidean.

---

### Phase 2: Object Islands (steps 5K–15K)

**Goal**: Introduce object boundaries (same-object→0, cross-object→50.0). Ramp CE from 0→1.

**Critical Event — Tau Unfreeze at step 5,000**:

The transition from τ=1.0 (frozen) to learned τ produced a dramatic restructuring:

```
Step 5,000:  τ=1.00, |κ|=5,397, CV=5.34
Step 5,050:  τ=1.00, |κ|=0.26,  CV=3.26   ← curvature COLLAPSES 20,000×
Step 5,100:  τ=1.00, |κ|=0.14,  CV=2.92
Step 5,350:  τ=1.00, |κ|=2.18,  CV=2.70   ← tau starts moving
Step 5,450:  τ=0.55, |κ|=3.82,  CV=2.15   ← tau rapidly drops
Step 5,500:  τ=0.34, |κ|=2.94,  CV=2.47
Step 5,600:  τ=0.31, |κ|=6.78,  CV=2.28
Step 6,300:  τ=0.14, |κ|=32.8,  CV=1.57   ← tau settles near floor
```

**Interpretation**: When τ unfreezes, the optimizer immediately drives it toward τ_min (0.10). This concentrates the heat kernel to very local neighborhoods, which effectively resets the curvature landscape (|κ| drops from 5,397 to 0.14). The Phase 1 metric structure (CV=5.3) partially collapses to CV~2.0. The optimizer traded the broad spatial manifold for tight local kernels — the model chose *locality* over *global geometry*.

**CE Ramp and Transform Learning** (steps 5K–15K):

Train xform_acc emergence:
```
Step 5,000:  0.12  (random baseline, CE just turning on)
Step 5,500:  0.32
Step 6,000:  0.00  (tau restructuring disrupts learning)
Step 6,700:  0.49
Step 7,500:  0.52
Step 8,100:  0.80  (first strong transform accuracy)
Step 9,350:  0.76
Step 10,200: 0.82  (peak observed)
Step 12,000-15,000: 0.30–0.75 (volatile but averaging ~0.55)
```

**geo_MSE** after Phase 2 target switch:
```
Step 5,000:  2,009  (spike: target switched to object boundaries!)
Step 5,050:    145  (rapid adaptation to new targets)
Step 6,000:     98
Step 8,000:     83
Step 10,000:   134
Step 12,000:   144
Step 15,000:    91
```
The geo_MSE never dropped below ~80 in Phase 2. The model cannot perfectly match the binary 0/50 object-boundary targets.

**|κ| regrowth**:
```
Step 5,050:    0.26  (post-collapse)
Step 6,000:   15.5
Step 7,000:   44.3
Step 8,000:   83.7
Step 10,000:  175
Step 12,000:  666
Step 14,000:  2,067
Step 15,000:  955
```
Curvature regrows steadily throughout Phase 2, reaching 2,000+ by the end. This is with `curvature_lambda=0.0` (no curvature penalty).

---

### Phase 3: The Release (steps 15K–17.8K, ongoing)

**Goal**: Full CE + decaying geo scaffold (λ_geo: 1.0→0.1 over steps 15K–20K).

**λ_geo schedule** (at sampled steps):
```
Step 15,000:  λ_geo=1.00  (decay starts)
Step 16,000:  λ_geo=0.82
Step 17,000:  λ_geo=0.64
Step 17,800:  λ_geo=0.50  ← current
Step 18,000:  λ_geo=0.46  (projected)
Step 20,000:  λ_geo=0.10  (final)
```

**geo_MSE** (raw, not weighted) is INCREASING as scaffold weakens:
```
Step 15,000:   91
Step 15,600:  147
Step 16,000:  147
Step 16,700:  117
Step 17,000:  119
Step 17,500:  143
Step 17,800:   96
```
Raw geo_MSE fluctuates 70–160, trending upward. The metric is drifting from the spatial targets as the geo loss becomes less influential.

**|κ| acceleration**:
```
Step 15,000:  955
Step 16,000:  2,681
Step 16,700:  5,141  ← new all-time high
Step 17,000:  4,108
Step 17,750:  4,536
Step 17,800:  3,760
```
Curvature is now routinely 2,000–5,000 and spiking above 5,000. With no curvature penalty and weakening geo scaffold, there is nothing constraining curvature growth.

**Train CE** — Improving:
```
Step 15,000:  1.41
Step 16,000:  1.03
Step 16,800:  0.90  ← best observed
Step 17,000:  0.98
Step 17,550:  0.89
Step 17,800:  1.03
```

**Train xform_acc** — Strong:
```
Step 15,800:  0.81
Step 16,350:  0.78
Step 16,800:  0.79
Step 17,300:  0.76
Step 17,550:  0.77
```
Training on procedural tasks is working well.

---

## 3. Eval Trajectory (The Critical Metric)

All evals are on 400 real ARC tasks (not procedural).

| Step | cell_acc | xform_acc | copy_bl | CE | Phase |
|------|----------|-----------|---------|-----|-------|
| 1000 | 0.0788 | 0.1137 | 0.616 | 2.319 | P1 |
| 2000 | 0.0846 | 0.1068 | 0.595 | 2.314 | P1 |
| 3000 | 0.0826 | 0.1138 | 0.632 | 2.302 | P1 |
| 4000 | 0.0800 | 0.1100 | 0.612 | 2.306 | P1 |
| 5000 | 0.0722 | 0.1122 | 0.596 | 2.307 | P1→P2 |
| 6000 | **0.5983** | 0.0001 | 0.601 | 2.345 | P2 |
| 7000 | 0.5486 | 0.1198 | 0.596 | 2.656 | P2 |
| 8000 | 0.5435 | 0.0941 | 0.646 | 2.615 | P2 |
| 9000 | 0.4922 | 0.0940 | 0.592 | 2.667 | P2 |
| 10000 | 0.5770 | 0.1275 | 0.652 | 3.017 | P2 |
| 11000 | 0.5462 | 0.0951 | 0.620 | 2.675 | P2 |
| 12000 | 0.5015 | 0.1039 | 0.641 | 2.533 | P2 |
| 13000 | 0.5358 | **0.1587** | 0.607 | 2.733 | P2 |
| 14000 | 0.4747 | 0.0992 | 0.635 | 2.581 | P2 |
| 15000 | 0.5749 | 0.1113 | 0.618 | 2.635 | P2→P3 |
| 16000 | 0.5543 | 0.0929 | 0.620 | 2.846 | P3 |
| 17000 | 0.5502 | **0.0734** | 0.604 | 2.728 | P3 |

### Key Findings:

1. **cell_acc ≈ copy_bl** throughout. The model learned to copy the input grid (cell_acc ~0.55 vs copy baseline ~0.61). It is not learning transforms on real ARC.

2. **xform_acc peaked at 0.1587 (step 13K) and is now DECLINING to 0.0734**. Phase 3 is making eval WORSE, not better.

3. **Eval CE is INCREASING** (2.3 → 2.7–2.8). The model is becoming less calibrated on real ARC as it overfits to procedural tasks.

4. **Step 6K anomaly**: cell_acc jumped from 0.08 to 0.60 while xform_acc dropped to 0.0001. This is the tau unfreeze teaching the model to copy the input grid — the cheapest way to reduce CE on simple tasks.

---

## 4. Metric Health Dashboard

| Metric | Phase 1 End | Phase 2 Mid | Phase 3 Now | Trend |
|--------|-------------|-------------|-------------|-------|
| CV | 5.34 | 1.7–2.5 | 2.5–3.1 | Stable |
| \|κ\| | 5,397 | 100–2,000 | 2,000–5,000 | **Exploding** |
| τ_mean | 1.00 (frozen) | 0.14–0.16 | 0.15–0.17 | Stable |
| τ_sigma | 0.00 | 0.17–0.20 | 0.19–0.22 | Slight growth |
| geo_MSE | 193 | 80–145 | 70–160 | Drifting up |
| Train CE | 2.3 (frozen) | 1.2–1.8 | 0.9–1.7 | Improving |
| Train xform | ~0.10 (random) | 0.30–0.80 | 0.30–0.81 | Strong |
| Eval xform | ~0.11 | 0.09–0.16 | **0.07–0.11** | **Declining** |

---

## 5. Diagnosis

### 5.1 The Generalization Gap is the Core Problem

Train xform_acc (procedural) = 0.55–0.80. Eval xform_acc (real ARC) = 0.07–0.16. This is a **5–10× gap** that is not closing. The model is solving procedural tasks through pattern matching that does not transfer to real ARC.

### 5.2 The Copy Shortcut Persists

Despite transform-weighted loss (5× on changed cells, 0.05× on unchanged), the model's eval behavior is dominated by copying. Eval cell_acc tracks the copy baseline, not improving beyond it.

### 5.3 Curvature Explosion

|κ| has grown from 0 to 5,000+ with no penalty to constrain it. This means the metric is developing extreme curvature, which:
- Makes the heat kernel highly concentrated on tiny neighborhoods
- Reduces the effective reach of information routing
- Could eventually cause numerical instability (exp(-D²/4t) with large D²)

With τ settled near 0.14 and |κ| at 4,000+, the heat kernel is effectively degenerate — attending almost exclusively to same-position or nearest neighbors.

### 5.4 Geo Scaffold Erosion

As λ_geo decays (currently 0.50, heading to 0.10), the raw geo_MSE is increasing. The metric is drifting away from the spatial targets. By step 20K when λ_geo=0.1, the scaffold will have minimal influence.

### 5.5 Phase 2 Object Boundaries Were Too Abrupt

At step 5,000, the target switched from squared-Manhattan to binary 0/50 object boundaries. The geo_MSE spiked from 193 to 2,009, then recovered to ~100–150 but never approached the Phase 1 levels. The 0/50 binary targets may be too extreme for the model to match precisely.

---

## 6. What the Geometry Did and Didn't Achieve

### Achieved:
- **Phase 1 succeeded**: MetricNet learned to approximate squared Manhattan distances (42K → 200 MSE)
- **Metric structure emerged**: CV grew from 0.05 to 5.3, proving the metric isn't flat
- **Tau self-organized**: Found its operating point (0.14–0.16) within 1,000 steps of unfreezing
- **Train task learning works**: Model reaches 0.80+ xform_acc on procedural tasks
- **No NaN/divergence**: Training is numerically stable despite |κ|=5,000+

### Did Not Achieve:
- **No eval improvement**: Real ARC xform_acc never exceeded 0.16 and is now declining
- **Copy bias not broken**: Eval cell_acc ≈ copy baseline throughout
- **Curvature unconstrained**: |κ| growing exponentially with no damping
- **Geo scaffold did not persist**: As λ_geo decays, metric drifts from spatial structure
- **Procedural → ARC transfer**: Zero evidence of generalization from procedural to real tasks

---

## 7. Recommendations

### Immediate (this run):
1. **Enable curvature penalty**: Set `curvature_lambda > 0` (e.g., 0.01) to prevent |κ| from growing unbounded
2. **Monitor for instability**: |κ| at 5,000+ with τ=0.14 means very peaked kernels — watch for NaN

### Next Run Experiments:
3. **Increase geo_lambda_final** from 0.1 to 0.5 — keep stronger geometric scaffold
4. **Add curvature_lambda=0.01** throughout training
5. **Smooth Phase 2 transition**: Instead of binary 0/50 targets, interpolate: `target = (1-α)*manhattan² + α*boundary_target` with α ramping over 2K steps
6. **Evaluate procedural-to-ARC transfer separately**: The procedural generator may need fundamental changes if its patterns don't overlap with real ARC
7. **Consider eval on procedural held-out set**: To distinguish "doesn't generalize to ARC" from "doesn't generalize at all"
8. **Slower tau unfreeze**: The dramatic τ=1.0→0.14 collapse destroyed Phase 1 metric structure. Consider clamping τ_min higher initially (e.g., 0.5) and lowering it gradually

---

## 8. Timeline

```
14:20:26  Training started — Phase 1 (Pure Geometry)
14:20:38  Step 0:     geo_MSE=41,994, CV=0.05, |κ|=0.002
14:22:17  Step 400:   geo_MSE=484, CV=2.70 — metric crystallizing
14:24:18  Step 1000:  geo_MSE=324, CV=3.30 — first eval (cell=0.08)
14:31:48  Step 3000:  geo_MSE=278, CV=5.17, |κ|=3,012
14:39:13  Step 5000:  TAU UNFROZEN → Phase 2 starts
14:39:21  Step 5000:  geo_MSE=2,009 (target change spike)
14:39:40  Step 5050:  |κ| collapses 5,397 → 0.26
14:41:32  Step 5500:  τ drops to 0.34 — optimizer found locality
14:43:39  Step 6000:  EVAL cell_acc jumps to 0.60 (copy learning)
14:52:24  Step 8100:  Train xform_acc hits 0.80 for first time
15:12:50  Step 13000: Best eval xform_acc = 0.1587
15:21:09  Step 15000: Phase 2→3 transition
15:29:28  Step 17000: Eval xform_acc drops to 0.0734
15:32:47  Step 17800: |κ|=3,760, CV=2.46, geo_MSE=96 ← current
```
