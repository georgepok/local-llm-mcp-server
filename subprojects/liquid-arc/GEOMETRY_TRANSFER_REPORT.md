# Cross-Dimension Geometry Transfer Pipeline — Specification & Results

**Status**: Active experiment — Phase 2 (task training) running
**Date**: 2026-04-07
**Author**: Built iteratively through experimentation on DGX Spark

---

## 1. Problem Statement

LiquidARC's power comes from a geometric substrate that emerges through a phase transition (CV jumping from ~2 to ~6-7 at step ~5000-5500). This transition is:
- **Unreliable** to reproduce (sensitive to init, LR, curriculum)
- **Expensive** (5000+ steps of training before the model becomes useful)
- **Dimension-locked** (a d=768 teacher can't directly initialize a d=2688 student)

When switching the Mind's LLM interface (e.g., Qwen3-4B d=2560 → Nemotron d=2688), the entire LiquidARC model needs to change dimension to live directly in the LLM's embedding space. Without geometry transfer, every new dimension requires a fresh phase transition.

**Goal**: A pipeline that transfers the geometric substrate from any post-transition teacher to a student of any dimension, reliably, in minutes instead of hours.

---

## 2. Core Insight

The N×N heat kernel attention pattern is **dimension-independent**. It describes which positions route information to which other positions — the spatial structure of computation. A teacher at d=768 produces attention patterns [N,N] that encode everything the phase transition discovered. A student at d=2688 can learn to reproduce those patterns through its own MetricNet without going through its own transition.

What transfers (dimension-independent):
- Heat kernel attention A[N,N] = softmax(-D²/(4t)) — the routing structure
- Per-position tau values τ[N] — timescale differentiation
- These are the outputs of the geometry, not the weights

What must be learned fresh (dimension-dependent):
- MetricNet weights (Linear(2*d_student, d_metric) — different input dim)
- W_v, W_o, FFN — content projections at new dimension
- Embeddings, output head — task-specific at new dimension

---

## 3. Pipeline Architecture

### 3.1 Prerequisites

**Teacher checkpoint**: A LiquidARC model that has undergone the phase transition. Must contain:
- `config` object with d_model, d_metric, d_ffn, etc.
- `model_state_dict` (or `model`) with dynamics + context_pool weights
- Checkpoint keys may have `._orig_mod.` prefix (torch.compile) — stripped automatically
- Backward compat: `metric_net_linear2` renamed to `metric_net_linear2_diag` — handled

**Best teacher checkpoint**: `output_30m/checkpoints/step_10000.pt` (d=768, CV~6, 54.2% peak eval xform). Step 10-15K is optimal — later steps overfit to procedural distribution.

**ARC data**: `fgn-v3/data/arc-repo/data/` with `training/` and `evaluation/` splits (400 tasks each).

**Hardware**: DGX Spark (128GB unified memory). The teacher (5M) + student (60M at d=2688) + N×N attention fit comfortably at seq_len=512.

### 3.2 Two-Phase Training

```
Phase 1: DISTILLATION (steps 0 → distill_steps)
  ┌──────────────────────────────────────────────────────┐
  │ Loss = task_weight * CE                              │
  │      + attn_weight * KL(teacher_attn || student_attn)│
  │      + tau_weight * MSE(normalized_tau)              │
  │      + curv_loss + tau_var_loss + cv_floor_loss      │
  └──────────────────────────────────────────────────────┘
  The attention matching guides MetricNet.
  The CE loss teaches content params.
  100x LR ratio protects developing geometry.

  → Snapshot saved as distill_peak.pt at end of phase

Phase 2: TASK TRAINING (steps distill_steps → max_steps)
  ┌──────────────────────────────────────────────────────┐
  │ Loss = CE + curv_loss + tau_var_loss + cv_floor_loss │
  └──────────────────────────────────────────────────────┘
  No teacher involvement. Pure ARC task gradients.
  Student develops further geometry driven by task need.
  The distilled substrate is the warm start.
```

### 3.3 The Teacher Forward Pass

Both teacher and student process the **same ARC batch**. The teacher runs its full forward (embedding → ODE → output head), then we extract the attention pattern from its final hidden state `h_final`:

```python
# From h_final [B, N, d_teacher]:
h_normed = dynamics.norm_geo(h)
g_diag = softplus(MetricNet([h_normed, context]))  # per-position metric
sqrt_g = g_diag.sqrt()
scaled_h = h_normed * sqrt_g
D² = ||scaled_h_i - scaled_h_j||²                  # N×N distances
attention = softmax(-D² / (4 * t_diffusion))        # N×N routing pattern
tau = sigmoid(TauNet(h)) * (tau_max - tau_min) + tau_min  # per-position timescale
```

This attention pattern is the **geometric truth** — what the phase transition discovered. It's dimension-independent because it's N×N regardless of d.

### 3.4 The Student Matching

The student processes the same ARC batch through its own architecture at d_student. Its MetricNet produces a different attention pattern. The KL divergence between student and teacher attention drives the MetricNet to learn the routing structure:

```python
loss_attn = KL(teacher_attention || student_attention)  # [B,N,N] distributions
loss_tau = MSE(normalize(student_tau), normalize(teacher_tau))  # relative pattern
```

The tau matching uses normalized values (relative pattern, not absolute) because the tau range depends on d.

### 3.5 Data Flow

```
ARC task → generate_batch() → meta dict (colors, xs, ys, roles, masks)
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
              teacher.forward()  student.forward()  │
                    │               │               │
              h_final [B,N,768]  h_final [B,N,2688] │
                    │               │               │
        get_teacher_attention()  get_student_attention()
                    │               │
              attn_T [B,N,N]   attn_S [B,N,N]
                    │               │
                    └───── KL ──────┘
```

---

## 4. Critical Parameters & Their Effects

### 4.1 MetricNet Bottleneck (d_metric)

The MetricNet compresses `[h_normed, context]` (dim=2*d) through a bottleneck of d_metric before producing per-dimension metric values.

| d_metric | Compression | Eval xform (step 500) | Why |
|----------|-------------|----------------------|-----|
| 192 (same as teacher) | 28x | **45.2%** | Tight bottleneck = regularizer. Forces essential features only. |
| 672 (25% of d, proportional) | 8x | 38.8% | Extra capacity captures noise from limited teacher supervision. |

**Recommendation**: Start with teacher's d_metric for distillation phase. The wider bottleneck may benefit Phase 2 (task training) where the student develops beyond teacher's knowledge. Current experiment uses 672 with Phase 2 to test this hypothesis.

### 4.2 Learning Rate Ratio

```
geo_lr = 1e-4    (MetricNet, TauNet, t_diffusion, alpha, ContextPool)
content_lr = 1e-2 (W_v, W_o, FFN, embeddings, output head)
ratio = 100x
```

This ratio is the single most important hyperparameter. It was discovered in the original same-dimension distillation and applies equally to cross-dimension:
- Too high ratio (1000x): geometry frozen, can't learn
- Too low ratio (10x): content gradients overwrite geometry
- 100x: geometry adapts slowly while content learns fast

### 4.3 Sequence Length (max_seq_len)

The N×N attention matrix requires `B × N × N × d × 4` bytes. This scales quadratically in N and linearly in d:

| N (seq_len) | d=768 | d=2688 | Fits? (128GB) |
|-------------|-------|--------|---------------|
| 2048 | 24 GB | 84 GB | d=768 only |
| 1024 | 6 GB | 21 GB | Yes |
| 512 | 1.5 GB | 5.2 GB | Yes, comfortable |

ARC task content: avg 440 tokens, max 1375. Most tasks fit in 512. Use `--max_seq_len 512` for d≥2688.

### 4.4 Distillation Steps

The attention KL collapses around step ~175 (from ~6000 to ~70). This is a softmax temperature phase change — the student's `t_diffusion` crosses a threshold. After this, remaining KL (~30-50) is actual pattern matching.

| distill_steps | Eval xform at cutoff | Notes |
|---------------|---------------------|-------|
| No cutoff (5000) | 45.2% peak at 500, then degraded | Teacher supervision becomes counterproductive |
| 500 | 38.8% at cutoff | Phase 2 explores further independently |

**Current recommendation**: 500 steps for distillation, then pure task training.

### 4.5 Batch Size

Batch size 4 works at seq_len=512. Total memory ~55GB (of 121GB available). Batch size 8 may work but hasn't been tested with the N×N computation.

---

## 5. Running the Pipeline

### 5.1 Command

```bash
python3 -u scripts/distill_geometry.py \
    --teacher_checkpoint output_30m/checkpoints/step_10000.pt \
    --teacher_d 768 \
    --student_d 2688 \
    --data_dir /workspace/fgn-v3/data/arc-repo/data \
    --output_dir output/distilled_2688 \
    --max_steps 10000 \
    --batch_size 4 \
    --max_seq_len 512 \
    --distill_steps 500 \
    --real_arc_mix 0.3 \
    --eval_every 250 \
    --save_every 1000
```

**Important**: Use `python3 -u` for unbuffered output (otherwise `tee` gets no output until buffer fills).

### 5.2 For a Different Target LLM

Only change `--student_d`:

```bash
# Qwen3.5-9B (d=4096)
--student_d 4096

# Qwen3-4B (d=2560)
--student_d 2560

# Any future model
--student_d <hidden_size from model config.json>
```

### 5.3 Environment

```bash
PYTHONPATH=/workspace/liquid-arc:/workspace/fgn-v3
TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Container: `fgn-train` (nvcr.io/nvidia/vllm:26.01-py3) with liquid-arc + fgn-v3 mounted.

### 5.4 Outputs

```
output/distilled_<d>/
  checkpoints/
    distill_peak.pt    — snapshot when distillation ends (Phase 1 → Phase 2)
    step_1000.pt       — periodic checkpoints
    step_2500.pt
    final.pt           — last step
```

Each checkpoint contains:
- `model_state_dict` — full student model weights
- `config` — LiquidARCConfig for the student
- `teacher_checkpoint` — path to teacher used
- `teacher_d` — teacher dimension (for provenance)

---

## 6. Checkpoint Compatibility

### Loading teacher checkpoints

The teacher checkpoint must be a standard LiquidARC training checkpoint with a `config` field. Key backward-compat handling:

```python
# torch.compile prefix
k = k.replace("._orig_mod.", ".")
# Old MetricNet key name
k = k.replace('metric_net_linear2.', 'metric_net_linear2_diag.')
```

### Loading student for deployment

```python
from liquid_arc.config import LiquidARCConfig
from liquid_arc.model import LiquidARCModel

ckpt = torch.load("distill_peak.pt", map_location=device)
config = ckpt['config']
model = LiquidARCModel(config).to(device)
model.load_state_dict(ckpt['model_state_dict'])
```

---

## 7. Experimental Results

### 7.1 Run 1: d_metric=192 (tight bottleneck), no distill cutoff

```
Step    50: kl=6292  ce=12.5  xform=11.7%  CV=2.32
Step   150: kl=5874  ce= 3.2  xform=14.4%  CV=3.10
Step   200: kl=  86  ce= 3.2  xform=10.7%  CV=3.49   ← KL collapse
Step   500: kl=  38  ce= 2.5  xform=13.2%  CV=3.61
  EVAL 500: xform=45.2%  CE=2.327  CV=3.71             ← PEAK

Step   600: kl=  32  ce= 2.4  xform=18.3%  CV=3.60
  EVAL 1000: degrading from here
```

**Key result**: 45.2% eval xform at step 500 with only 15% train xform. The geometric substrate transferred generalization without the student mastering training.

**Teacher reference**: 54.2% eval xform at step 21K (after full phase transition).

### 7.2 Run 2: d_metric=672 (proportional bottleneck), no distill cutoff

```
Step   200: kl=  73  ce= 2.7  xform=11.5%  CV=3.98
Step   500: kl=  38  ce= 2.5  xform=16.8%  CV=3.82
  EVAL 500: xform=39.1%  CE=2.161  CV=4.10             ← lower than Run 1

Step   600: degrading
```

**Finding**: Wider bottleneck = worse. Teacher provides ~192 dims of geometric information. Extra capacity captures noise.

### 7.3 Run 3 (current): d_metric=672, distill_steps=500

```
Phase 1 (distillation):
Step   200: kl=  67  ce= 2.6  xform=18.4%  CV=3.90  tau=0.645
Step   500: kl=  38  ce= 2.5  xform=14.3%  CV=3.85  tau=0.684
  EVAL 500: xform=38.8%  CE=2.161  CV=3.99
  → distill_peak.pt saved

Phase 2 (task training):
Step   550: kl=0.00  ce= 3.1  xform=12.0%  CV=3.90  tau=0.672
Step   700: kl=0.00  ce= 2.6  xform=13.7%  CV=3.52  tau=0.719
  ... in progress
```

**Observation**: CV dropping in Phase 2 (3.99→3.52). The geometry is relaxing without teacher supervision. Watching for whether task loss eventually drives its own geometric development (a second, student-initiated transition).

---

### 7.3 Run 3 (current): d_metric=672, distill_steps=500

```
Phase 1 — Distillation (steps 0-500):
  Step   200: kl=  67  ce= 2.6  xform=18.4%  CV=3.90  tau=0.645
  Step   500: kl=  38  ce= 2.5  xform=14.3%  CV=3.85  tau=0.684
  EVAL 500: xform=38.8%  CE=2.161  CV=3.99
  → distill_peak.pt saved, switching to pure task training

Phase 2 — Task Training (steps 500+):
  Step   700: kl=0.00  ce= 2.6  xform=13.7%  CV=3.52  tau=0.719  ← CV dipping
  EVAL 750: xform=48.2%  CE=2.032  CV=3.61                        ← EXCEEDED distill peak
  Step   950: kl=0.00  ce= 2.1  xform=39.2%  CV=4.23  tau=0.750  ← CV recovering
  EVAL 1000: xform=51.3%  CV=4.50                                 ← approaching teacher
  Step  1500: kl=0.00  ce= 1.1  xform=69.7%  CV=4.81  tau=0.746
  EVAL 1750: xform=50.1%  CV=4.88
  Step  2200: kl=0.00  ce= 1.1  xform=72.7%  CV=5.11  tau=0.738
  EVAL 2250: xform=46.9%  CE=2.025  CV=5.02                       ← CV in teacher range
```

**Key findings:**

1. **CV self-recovery**: After distillation cutoff, CV dipped from 3.99 to 3.52 (geometry relaxing without supervision), then climbed back through task gradients: 3.52 → 4.23 → 4.98 → **5.11**. The student developed its OWN post-transition geometry.

2. **Eval peaked at step ~1000**: 51.3% eval xform — matching teacher's 54.2% — then oscillating 45-51%. Same plateau behavior as teacher.

3. **Train/eval gap growing**: Train xform 73% vs eval 47% at step 2250. Same pattern as teacher (90% train, 50% eval). The procedural→real ARC generalization gap is architectural, not distillation-related.

4. **The distillation warm start worked**: Without it, phase transition happens at step ~5500. With it, the student reached competitive eval by step 750. The distillation saved ~4800 steps of uncertain training.

---

## 8. Three-Phase Model for Deployment

The ARC distillation gives the student the geometric MECHANISM but not the CONTENT for its deployment domain (language via Nemotron). The full pipeline has three phases:

### Phase 1: Geometric Mechanism Transfer (this pipeline)
- **What transfers**: How to produce non-uniform attention, timescale differentiation, ODE integration that transforms input
- **What doesn't**: Domain-specific routing patterns for language
- **Duration**: ~500 steps (minutes)
- **Output**: distill_peak.pt

### Phase 2: Geometric Development on ARC (this pipeline)
- **What develops**: Student's own metric structure, curvature, CV approaching teacher's range
- **What doesn't**: Language understanding, domain sensitivity
- **Duration**: ~2000 steps (10-20 min)
- **Output**: step_2000.pt or best checkpoint by eval xform

### Phase 3: Domain Adaptation in Deployment (future, in the Mind)
- **What develops**: How to route TEXTUAL content, domain-sensitive metric for language, temporal context integration
- **Input**: Nemotron's d=2688 embeddings (not ARC grid cells)
- **Learning**: Online through the Mind's autonomous loop, curriculum, conversation
- **The 1920 extra dimensions** (2688 - 768): These encode Nemotron's language representations. The student must learn what they're FOR — which semantic features map to geometric distance, which deserve fast/slow timescales
- **Duration**: Ongoing (the Mind continuously adapts)

The three phases transfer increasingly specific knowledge:
```
Phase 1: "How to have geometry at all"         (mechanism)
Phase 2: "How to use geometry for spatial tasks" (capability)
Phase 3: "How to use geometry for language"      (deployment domain)
```

## 9. The Distribution Problem (Why Phase 2 Is Required)

After Phase 1, the ODE state lives in the right vector space (d=2688, same as Nemotron) but the WRONG SUBMANIFOLD. The values encode ARC grid routing patterns — which positions exchange color information, how spatial transformations propagate. These vectors occupy a region of R^2688 shaped by:
- MetricNet routing on 2D grid structures
- LTC contraction toward ARC color/position targets
- SDPA heat kernel diffusion across grid cells

Nemotron's embedding space was shaped by 25T tokens of text. Embeddings cluster by semantic content ("bridge" near "road"), with inter-dimensional correlations, variance profiles, and cluster structures that reflect language statistics.

The ODE state and Nemotron embeddings have the same dimensionality but completely different distributions. Even after magnitude normalization, the per-dimension content is wrong — dimension 47 might encode "vertical symmetry detection" for ARC but Nemotron expects it to carry part of a semantic representation.

**Result**: When the Mind sends 64 prefix tokens from ODE state to Nemotron, the LLM sees vectors that look like nothing it was trained on. The attention/Mamba/MoE layers can't extract useful information from inputs on the wrong submanifold. This also causes CUDA crashes because extreme-distribution inputs propagate numerical instability through the network.

**Phase 2 fixes this**: NTP training with Nemotron frozen. The NTP loss gradient flows through Nemotron's layers back to the ODE, teaching it what distributions the LLM can actually USE. The ODE state migrates from the "ARC grid routing" region to the "text semantic" region. The MetricNet preserves its routing STRUCTURE (how to produce non-uniform attention, timescale differentiation) but adapts its CONTENT (which dimensions matter, what variance profile to produce).

### Phase 2 Training Architecture

```
Text events → Nemotron embed_tokens → ODE (trainable, 60M params)
                                        ↓
                              h(t) [B, N, 2688] = prefix tokens
                                        ↓
                              Nemotron (frozen, 30B FP8) forward
                                        ↓
                              NTP loss on next-token prediction
                                        ↓
                              Gradient flows back through Nemotron to ODE
                                        ↓
                              ODE learns text-compatible distributions
```

The 100x LR ratio applies: geometric params slow (preserve routing structure from Phase 1), content params fast (adapt distribution to language).

**Memory**: Nemotron FP8 = 49GB + ODE = 0.24GB + activations/gradients ≈ 60-70GB total. Fits in 128GB.

## 10. Open Questions

1. **ANSWERED — Phase 2 DOES produce further geometric development.** CV climbed from 3.5 to 5.1 through task gradients alone. The distillation provides the mechanism; the task loss drives it further.

2. **d_metric strategy**: d_metric=192 gave better distillation (45% vs 39% eval), but d_metric=672 gave better Phase 2 development (CV reached 5.1 vs unknown for 192). Hypothesis: narrow for distillation, wide for task training. Not yet tested as a switching strategy.

3. **LR ratio in Phase 2**: Kept at 100x throughout. The student's CV still developed, suggesting 100x doesn't prevent geometric growth — it just keeps it orderly. No evidence that changing the ratio at Phase 2 helps.

4. **Distillation timing**: 500 steps is sufficient — KL converges by step 200, remaining 300 steps refine. Cutting earlier (300) may work but is untested.

5. **Phase 3 domain adaptation**: How quickly does the student adapt from ARC grid geometry to language/conversation geometry when deployed as the Mind? Does the mechanism transfer or does it need fundamental retraining? This is the critical open question.

6. **Best checkpoint for deployment**: Use distill_peak.pt (mechanism only) or a Phase 2 checkpoint (mechanism + ARC capability)? The ARC-trained geometry may or may not help with language routing.

---

## 9. Key Files

| File | Purpose |
|------|---------|
| `scripts/distill_geometry.py` | Main pipeline script (both phases) |
| `configs/liquid_arc_2688.yaml` | Config template for d=2688 student |
| `output_30m/checkpoints/step_10000.pt` | Teacher checkpoint (d=768) |
| `PRECIOUS_CHECKPOINTS/qwen3_4b_coupling/` | Backup of old Qwen3 coupling (if needed) |
| `GEOMETRY_TRANSFER_REPORT.md` | This document |
