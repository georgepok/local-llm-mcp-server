# LiquidARC 572K Phase Transition — 30% ARC Mix Reproduction

Complete reconstruction document. Everything needed to reproduce the experiment from scratch.

---

## 1. Neural Network Architecture

### Overview

```
Input (ARC grid cells as tokens)
  → ARCEmbedding       [~51K params]    Additive: color + pos_x + pos_y + role + sep + grid_id → LN → Dropout
  → ContextPool        [~5.8K params]   Attention-weighted pooling → single [B, d] context vector
  → euler_solve(ContinuousDynamics, h₀, 16 steps)  [~356K params, shared across all 16 steps]
  → OutputHead         [~2.6K params]   LN → Linear → 10 color logits
```

**Total: 572,238 parameters** (d_model=256, d_metric=64, d_ffn=512)

### ARCEmbedding (~51K params)

Additive embedding — no sequential position encoding (ARC grids are 2D):

```
h = ColorEmbed(color)     [11 × 256]    — 10 ARC colors + 1 padding
  + PosX(x)               [31 × 256]    — x coordinate (0-30)
  + PosY(y)               [31 × 256]    — y coordinate (0-30)
  + RoleEmbed(role)        [4 × 256]     — input_demo / output_demo / test_input / test_output
  + SepEmbed(sep_type)     [4 × 256]     — grid boundary markers (only at separator positions)
  + GridIdEmbed(grid_id)   [16 × 256]    — which demonstration pair a token belongs to
  → LayerNorm(256)
  → Dropout(0.1)
```

**Key design choice**: test_output shares test_input's role embedding (role 3 → 2) so spatial reasoning transfers from input to output.

### ContextPool (~5.8K params)

Attention-weighted pooling over context positions → fixed-size episode vector:

```
scores = Linear(256→64) → Tanh → Linear(64→1)     [B, N] attention logits
alpha  = softmax(scores, masked to context_mask)     [B, N] weights
context = Linear(256→256) · Σ(alpha_i · h_i)        [B, 256] output
```

This context vector is broadcast to every position at every ODE step, conditioning the metric on the full episode.

### ContinuousDynamics (~356K params) — Applied 16× with Shared Weights

Each ODE step computes dh/dt through 7 sub-operations:

```
1. METRIC COMPUTATION (MetricNet: 49K params)
   h_normed = LayerNorm(h)                                     [B, N, 256]
   cat = concat(h_normed, context.expand(B,N,256))             [B, N, 512]
   met_hidden = GELU(Linear(512→64)(cat))                      [B, N, 64]
   g = Softplus(Linear(64→256)(met_hidden))                    [B, N, 256]  (positive diagonal metric)

   Init: final bias = log(e-1) ≈ 1.313 → Softplus ≈ 1.0 (identity metric at start)
   Init: final weights ~ N(0, 0.05) (small initial metric variation)

2. SDPA HEAT KERNEL (0 extra params — reuses metric output)
   t = Softplus(t_diffusion)                   learnable scalar, init 1.0
   sqrt_g = √g                                [B, N, 256]
   q = k = h_normed · sqrt_g                  metric-weighted queries/keys

   # SDPA factorization of heat kernel:
   # K_ij = softmax(-D²_ij/(4t)) where D²_ij = Σ_k g_k·(h_i^k - h_j^k)²
   # Factored as: softmax(q·k/(2t) - ||k_j||²/(4t))
   # The ||h_i||²_g term is row-constant → drops out under softmax

   q_scaled = q · √d/(2t)                     pre-scale for SDPA's internal 1/√d
   attn_bias = -||k_j||²/(4t)                 [B, 1, N] column-wise bias

   routed_v = F.scaled_dot_product_attention(q_scaled, k, V, attn_mask=attn_bias)
   # N×N matrix never materializes to HBM — stays in SRAM via FlashAttention

3. VALUE PROJECTION (W_v: 65.5K params)
   V = Linear(256→256, no bias)(LayerNorm(h))

4. OUTPUT PROJECTION + RESIDUAL (W_o: 65.5K params)
   update = Linear(256→256, no bias)(routed_v)     W_o zero-initialized
   target = h + update                              identity ODE at initialization

5. TAU (TauNet: 16.4K params)
   tau_logits = Linear(64→1)(GELU(Linear(256→64)(h_normed)))
   tau = sigmoid(tau_logits) · (τ_max - τ_min) + τ_min     [B, N, 1]
   # τ_min=0.5, τ_max=1.0 — bounded per-position time constant
   # Init: bias = log(e-1) → tau ≈ 1.0 + 0.5 = ~0.82 initially

6. LTC CONTRACTION
   dh/dt = -(1/τ) · (h - target)
   # Guaranteed stable: h exponentially decays toward diffusion target
   # Per-position τ controls computation speed

7. FFN RESIDUAL (FFN: 263K params)
   dh/dt += FFN(LayerNorm(h)) / n_ode_steps
   # FFN: Linear(256→512) → GELU → Dropout(0.1) → Linear(512→256)
   # Amortized by step count so total FFN contribution is constant
```

### Euler Solver

```python
dt = 1.0 / 16  # t ∈ [0, 1], 16 steps
for i in range(16):
    dh = dynamics(t, h)     # same weights every step
    h = h + dt * dh         # forward Euler
    t = t + dt
```

Temporal invariance: step count randomized [12, 20] per batch during training. Fixed at 16 for eval.

### Parameter Breakdown

| Component | Params | % of Total |
|-----------|--------|------------|
| MetricNet (W₁ 512→64, W₂ 64→256) | 49,408 | 8.6% |
| TauNet (W₁ 256→64, W₂ 64→1) | 16,449 | 2.9% |
| W_v (256→256, no bias) | 65,536 | 11.5% |
| W_o (256→256, no bias) | 65,536 | 11.5% |
| FFN (256→512→256 + biases) | 263,168 | 46.0% |
| 3× LayerNorm (norm_geo, norm_val, norm_ff) | 1,536 | 0.3% |
| t_diffusion (scalar) | 1 | 0.0% |
| alpha_logit (scalar, unused in current version) | 1 | 0.0% |
| Embedding (color+pos+role+sep+grid_id) | ~51,200 | 8.9% |
| ContextPool (attn scoring + out_proj) | ~5,800 | 1.0% |
| OutputHead (LN + Linear 256→10) | ~2,582 | 0.5% |
| **Total** | **572,238** | **100%** |

---

## 2. Training Configuration

### Config File: `configs/liquid_arc_zero_scaffold.yaml`

```yaml
# Architecture
d_model: 256
d_metric: 64          # MetricNet bottleneck dimension (d/4)
d_ffn: 512            # FFN hidden dimension (2d)
max_seq_len: 2048
n_ode_steps: 16
ode_steps_min: 12     # temporal invariance: randomize [12, 20]
ode_steps_max: 20
tau_min: 0.5
tau_max: 1.0
t_diffusion_init: 1.0
model_type: liquid

# Zero-scaffold: no geometric pre-training phase
tau_freeze_steps: 0       # TauNet learns from step 0
geo_loss_enabled: false   # no geometric supervision at all

# Metric plasticity via CV floor penalty
cv_floor_target: 3.0      # push CV above 3.0
cv_floor_lambda: 0.1      # penalty weight

# Data mixing
real_arc_mix_ratio: 0.3   # 30% real ARC, 70% procedural
use_procedural: true      # procedural infinite stream

# Procedural curriculum
curriculum_stage1_end: 20000    # GLOBAL rules 0-20K
curriculum_stage2_end: 100000   # RELATIONAL rules 20K-100K

# Loss weights
transform_weight: 5.0    # 5x weight on changed cells
copy_weight: 0.05        # 0.05x weight on unchanged cells
curvature_lambda: 0.05   # |κ| penalty
tau_var_lambda: 0.001     # -Var(τ) encourages tau diversity
alpha_logit_init: 2.2     # identity residual (unused in current W_o zero-init)
dropout: 0.1

# Solver
use_torch_compile: true
ode_chunk_size: 4
chunk_size: 256
invertible_solver: false
deq_solver: false

# TTT (test-time training) — eval only, does not affect training
ttt_enabled: true
ttt_steps: 100
ttt_lr: 0.001
ttt_curvature_lambda: 0.01
ttt_early_stop_threshold: 0.01
```

### Command Line

```bash
python scripts/train.py \
    --config configs/liquid_arc_zero_scaffold.yaml \
    --data_dir /workspace/fgn-v3/data/arc-repo/data \
    --output_dir /workspace/liquid-arc/output_reproduce \
    --max_steps 15000 \
    --batch_size 16 \
    --lr 3e-4 \
    --weight_decay 0.1 \
    --warmup_steps 500 \
    --grad_clip 1.0 \
    --log_every 50 \
    --eval_every 500 \
    --save_every 2500
```

### Loss Function

Assembled in `train.py` (model returns components, script combines):

```
total_loss = CE_loss + curvature_loss + tau_var_loss + cv_floor_loss

Where:
  CE_loss = CrossEntropy(logits, target_labels)
    - Weighted: 5.0× on transform cells, 0.05× on copy cells
    - Only computed on target_mask positions (test output cells)

  curvature_loss = 0.05 × mean(|κ|)
    - κ = Ricci scalar curvature from metric field
    - Keeps curvature bounded

  tau_var_loss = -0.001 × Var(τ)
    - Negative sign: encourages τ diversity across positions
    - Positions should have different computation rates

  cv_floor_loss = 0.1 × max(0, 3.0 - CV)²
    - Quadratic hinge: penalizes CV below 3.0
    - Keeps metric from collapsing to identity
```

### Optimizer & Schedule

```
Optimizer: AdamW (lr=3e-4, weight_decay=0.1)
Schedule: linear warmup (500 steps) → cosine decay (to step 15000)
Gradient clipping: max_norm=1.0
Gradient accumulation: 1 (no accumulation)
Batch size: 16
```

### Data Pipeline

**Procedural (70% of batches):**
- `ProceduralARCTask` — infinite stream, never repeats
- 13 rules: gravity, translate, reflect, draw_line, raycast, connect_same_color, copy_object, enclosed_fill, recolor, extend_pattern, pattern_repeat, scale_up, rule_none
- Curriculum: GLOBAL (7 rules) → RELATIONAL (11 rules) at step 20K
- D4 augmentation (8 symmetries: 4 rotations × 2 reflections)

**Real ARC (30% of batches):**
- 400 training tasks from ARC-AGI dataset
- Loaded from `data_dir/training/*.json`
- Augmented with 10 color permutations

**Eval (always real ARC):**
- 400 evaluation tasks from ARC-AGI dataset
- No augmentation, loaded from `data_dir/evaluation/*.json`
- Evaluated every 500 steps (20 batches × 8 per batch)

### Infrastructure

- **Hardware**: NVIDIA DGX Spark (GB10, 128GB unified memory)
- **Container**: `fgn-train` (nvcr.io/nvidia/vllm:26.01-py3)
- **Mount**: `/home/pokazge/liquid-arc` → `/workspace/liquid-arc`
- **torch.compile**: dynamics module compiled with `mode="default", dynamic=True`
- **TRITON_PTXAS_PATH**: baked into fgn-train container
- **Throughput**: ~9,000-10,000 tok/s steady state (when GPU not shared)

---

## 3. Training Dynamics — Raw Data

### Phase 1: Plateau (Steps 0-4800)

| Step | Loss | CE | Eval Xform | CV | |κ| | τ mean | τ σ | τ range |
|------|------|-----|-----------|-----|------|--------|------|---------|
| 0 | 3.25 | 2.39 | — | 0.057 | 0.0024 | 0.82 | 0.003 | 0.80-0.82 |
| 50 | 3.10 | 2.26 | — | 0.112 | 0.0027 | 0.82 | 0.003 | 0.80-0.83 |
| 100 | 3.00 | 2.31 | — | 0.359 | 0.0064 | 0.81 | 0.004 | 0.80-0.82 |
| 150 | 2.50 | 2.31 | — | 1.612 | 0.0241 | 0.81 | 0.003 | 0.79-0.82 |
| 250 | 2.34 | 2.33 | — | 2.825 | 0.0469 | 0.81 | 0.002 | 0.80-0.82 |
| 300 | 2.30 | 2.29 | — | 2.954 | 0.0493 | 0.81 | 0.004 | 0.80-0.84 |
| 500 | 2.33 | 2.33 | 15.1% | 3.119 | 0.0079 | 0.85 | 0.035 | 0.78-1.00 |
| 1000 | 2.35 | 2.35 | 24.2% | 3.182 | 0.0015 | 0.77 | 0.053 | 0.65-1.00 |
| 2000 | 2.29 | 2.29 | 18.8% | 3.243 | 0.0005 | 0.62 | 0.143 | 0.55-1.00 |
| 2500 | 2.34 | 2.34 | 22.4% | 3.314 | 0.0004 | 0.59 | 0.138 | 0.53-1.00 |
| 3000 | 2.31 | 2.31 | 14.2% | 3.212 | 0.0004 | 0.58 | 0.176 | 0.50-1.00 |
| 3500 | 2.34 | 2.34 | 17.2% | 3.156 | 0.0003 | 0.59 | 0.184 | 0.50-1.00 |
| 4000 | 2.30 | 2.30 | 20.4% | 3.315 | 0.0003 | 0.59 | 0.189 | 0.50-1.00 |
| 4500 | 2.30 | 2.30 | 15.2% | 3.706 | 0.0005 | 0.58 | 0.177 | 0.50-1.00 |

**Characteristics:**
- Loss stuck at ~2.3 for 4800+ steps
- Eval xform oscillating 14-24% (no trend)
- CV climbing slowly: 0.06 → 3.0 (step 250) → 3.7 (step 4500)
- Curvature collapsed to near-zero (0.0003) by step 3000
- Tau diversifying steadily: σ grew 0.003 → 0.189, range expanded to [0.50, 1.00]

### Phase 2: CV Acceleration & Transition (Steps 4800-5800)

| Step | Loss | Eval Xform | CV | |κ| | τ mean | τ σ | Cell Acc |
|------|------|-----------|-----|------|--------|------|----------|
| 4800 | 2.26 | — | 3.862 | 0.0007 | 0.66 | 0.205 | 0.073 |
| 4850 | 2.30 | — | 4.179 | 0.0007 | 0.59 | 0.181 | 0.063 |
| 4900 | 2.29 | — | 4.619 | 0.0012 | 0.58 | 0.175 | 0.102 |
| 4950 | 2.33 | — | **5.595** | 0.0020 | 0.60 | 0.167 | 0.016 |
| 5000 | 2.11 | 16.1% | **5.864** | 0.0032 | 0.68 | 0.189 | 0.094 |
| 5300 | 2.31 | — | 4.908 | 0.0029 | 0.68 | 0.190 | 0.195 |
| 5350 | 2.22 | — | **5.179** | 0.0012 | 0.59 | 0.160 | 0.115 |
| 5400 | 2.17 | — | **5.704** | 0.0018 | 0.60 | 0.166 | **0.246** |
| 5500 | 2.21 | 15.8% | 5.325 | 0.0013 | 0.72 | 0.113 | **0.308** |
| 5600 | 2.15 | — | 5.659 | 0.0013 | 0.60 | 0.150 | **0.470** |
| 5650 | 2.13 | — | 5.742 | 0.0017 | 0.62 | 0.146 | **0.630** |
| 5700 | 1.99 | — | 5.134 | 0.0020 | 0.60 | 0.132 | **0.678** |
| 5750 | 2.17 | — | 5.370 | 0.0021 | 0.64 | 0.128 | **0.703** |

**The transition:**
- CV accelerated 3.86 → 5.86 in 200 steps (4800-5000)
- Cell accuracy jumped 0.07 → 0.70 in 350 steps (5400-5750) — copy equilibrium broke
- Loss started dropping below 2.0
- Curvature developed AFTER CV crossed threshold (consequence, not cause)

### Phase 3: Rapid Learning (Steps 5800-7500)

| Step | Loss | Train Xform | Eval Xform | CV | |κ| |
|------|------|------------|-----------|-----|------|
| 6000 | 1.94 | 0.348 | 26.7% | 5.187 | 0.0016 |
| 6250 | 1.71 | 0.562 | — | 5.238 | 0.0013 |
| 6450 | 1.36 | 0.649 | — | 5.022 | 0.0017 |
| 6500 | 1.74 | 0.548 | **36.3%** | 5.014 | 0.0016 |
| 6800 | 1.27 | 0.737 | — | 5.123 | 0.0020 |
| 6950 | 1.20 | 0.764 | — | 5.163 | 0.0021 |
| 7000 | 1.29 | 0.704 | **42.4%** | 5.181 | 0.0021 |

### Phase 4: Plateau & Second Peak (Steps 7500-12500)

| Step | Eval Xform | Eval Cell | Eval CE | Notes |
|------|-----------|----------|---------|-------|
| 7000 | **42.4%** | 26.1% | 1.750 | First peak |
| 7500 | 40.0% | 27.6% | 1.751 | |
| 8000 | 32.5% | 25.6% | 1.816 | Dip |
| 8500 | 35.9% | 23.9% | 1.736 | Recovery |
| 9000 | 38.1% | 22.0% | 1.744 | |
| 9500 | 35.6% | 27.0% | 1.668 | |
| 10000 | 40.1% | 21.9% | 1.664 | |
| 10500 | 38.3% | 23.1% | 1.627 | |
| 11000 | **45.2%** | 25.1% | **1.616** | **Second peak** |
| 11500 | 41.8% | 25.2% | 1.596 | Best CE |
| 12000 | 40.9% | 22.8% | 1.611 | |
| 12500 | 39.6% | 24.0% | 1.636 | |

**Key observations:**
- Second peak at step 11000 (45.2%) exceeds first peak (42.4%)
- Eval CE continues improving: 1.75 → 1.60 through step 11500
- No catastrophic degradation like the original's 1.50 → 1.89 over extended training

### TTT Results

| Step | Baseline Xform | TTT Time | Phase |
|------|---------------|----------|-------|
| 2500 | 22.4% | 108.3s | Pre-transition |
| 5000 | 16.1% | 98.7s | At transition |
| 7500 | 40.0% | 85.9s | Post-transition |
| 10000 | 40.1% | 82.2s | Plateau |
| 12500 | 39.6% | 78.6s | Plateau |

(TTT xform values not logged separately in this run — only time and task counts)

---

## 4. Key Findings

### 4.1 Phase Transition Confirmed Reproducible

The CV-driven phase transition reproduced with different random initialization:

| Metric | Original Run | Reproduction |
|--------|-------------|--------------|
| Transition step | ~5350 | ~5350-5400 |
| CV at transition | ~6.0 | ~5.2-5.7 |
| Pre-transition eval xform | 14-22% | 14-24% |
| Post-transition peak xform | ~50% (at 10K) | 45.2% (at 11K) |
| Loss plateau | ~2.3 | ~2.3 |
| Loss post-transition | ~1.26 | ~1.20 |

### 4.2 Two-Mode Oscillation (New Finding)

Post-transition, the model alternates between two geometric regimes visible in per-step throughput:

**Procedural batches** (tok/s ~5,000-7,000):
- CV ~5.0-5.3, |κ| ~0.001-0.003
- Train xform 60-90%
- Geometry stable, well-adapted

**Real ARC batches** (tok/s ~11,000-22,000):
- CV drops to 3.5-4.2, |κ| spikes to 0.02-0.05
- Train xform 15-40%
- Geometry partially collapses — metric reorganizing for unfamiliar structure

The throughput difference is because real ARC tasks have shorter sequences than procedural tasks.

### 4.3 Tau as Co-Requisite

The system needed BOTH metric diversity (CV) and temporal diversity (tau σ) to reach the critical point:
- tau σ: 0.003 → 0.189 during plateau (positions becoming increasingly differentiated)
- tau range: [0.80-0.82] → [0.50-1.00] (some positions computing at half the rate of others)
- CV and tau σ both needed to cross thresholds before transition could fire

### 4.4 ARC Mix Ratio Comparison

| Mix Ratio | Transition Step | Peak Eval Xform | Notes |
|-----------|----------------|-----------------|-------|
| 15% | TBD (running) | TBD | |
| 30% | ~5350 | 45.2% | Baseline, this run |
| 70% | ~7500 | 42.4% | Delayed transition, same ceiling |

Higher ARC mixing delays the transition (less procedural signal to push CV) but doesn't improve the eval ceiling. The 42-45% appears to be an architectural limit at d=256.

---

## 5. How to Reproduce

### Prerequisites

1. NVIDIA GPU with CUDA support (4GB+ VRAM for 572K model)
2. PyTorch 2.0+ with CUDA
3. ARC-AGI dataset: `git clone https://github.com/fchollet/ARC-AGI`
4. Place ARC data at `data/arc/` with `training/` and `evaluation/` subdirectories

### Repository Structure

```
subprojects/liquid-arc/
├── liquid_arc/
│   ├── config.py          # LiquidARCConfig dataclass (all hyperparameters)
│   ├── model.py           # LiquidARCModel + FlatBaselineARC
│   ├── dynamics.py        # ContinuousDynamics (the shared ODE module)
│   ├── solver.py          # euler_solve, euler_solve_chunked, deq_solve
│   ├── embedding.py       # ARCEmbedding (additive cell-as-token)
│   ├── context_pool.py    # ContextPool (attention-weighted pooling)
│   ├── curvature.py       # CurvatureEngine (Ricci scalar from metric)
│   ├── geo_loss.py        # GeometricLoss (unused in zero-scaffold)
│   ├── ttt.py             # Test-time training (eval only)
│   ├── reptile.py         # Reptile meta-learning (unused in zero-scaffold)
│   └── tasks/
│       └── procedural.py  # ProceduralARCTask (infinite generator, 13 rules)
├── configs/
│   └── liquid_arc_zero_scaffold.yaml
├── scripts/
│   ├── train.py           # Main training script
│   ├── train_standalone.py # Standalone version (no fgn-v3 dependency)
│   └── eval.py            # Evaluation script
└── data/
    └── arc/               # ARC-AGI dataset (not in repo, must download)
```

**External dependency**: `subprojects/fgn-v3/fgn/tasks/arc.py` provides `ARCTask` class for loading real ARC data. Must be a sibling directory or at `/workspace/fgn-v3/`.

### Run Command

```bash
# On DGX Spark in fgn-train container:
cd /workspace/liquid-arc
python -u scripts/train.py \
    --config configs/liquid_arc_zero_scaffold.yaml \
    --data_dir /workspace/fgn-v3/data/arc-repo/data \
    --output_dir /workspace/liquid-arc/output_reproduce \
    --max_steps 15000 \
    --log_every 50 --eval_every 500 --save_every 2500

# For standalone (procedural-only, no ARC data needed):
python -u scripts/train_standalone.py \
    --config configs/reproduce_phase_transition.yaml \
    --output_dir output_reproduce \
    --max_steps 15000 --seed 42
```

### Expected Timeline

At ~10,000 tok/s on DGX Spark (full GPU):
- Steps 0-300: CV rises 0.06 → 3.0 (~1 min)
- Steps 300-4800: Plateau, CV at 3.0-3.7 (~15 min)
- Steps 4800-5400: CV acceleration 3.7 → 5.7, transition fires (~2 min)
- Steps 5400-7000: Rapid learning, loss 2.3 → 1.2 (~5 min)
- Steps 7000-15000: Plateau/consolidation (~25 min)
- Total: ~45-50 minutes

### What to Watch

1. **CV trajectory**: Should climb from 0.06 → 3.0 (step 250) → plateau → accelerate past 5.0 (step 4800-5000)
2. **Loss**: Stuck at 2.3 for 5000 steps, then collapses to 1.2-1.5
3. **Cell accuracy**: Jumps from ~0.07 to ~0.70 at transition (copy learning)
4. **Train xform**: Jumps from ~0.10 to ~0.65-0.77 post-transition
5. **Eval xform**: 15-22% pre-transition → 36-45% post-transition
6. **Tau σ**: Should grow from 0.003 to ~0.19 during plateau (co-requisite)

---

## 6. Artifacts

### On DGX Spark

```
/workspace/liquid-arc/output_reproduce/          # This reproduction run
/workspace/liquid-arc/output_zero_scaffold/      # Original 572K zero-scaffold run
/workspace/liquid-arc/output_30m/                # 5M width scaling
/workspace/liquid-arc/output_ttt_v1/             # TTT V1 (geo scaffold)
/workspace/liquid-arc/output_ttt_v2/             # TTT V2 (CV floor + real ARC)
/workspace/liquid-arc/output_high_arc_mix/       # 70% ARC mix experiment
/workspace/liquid-arc/output_low_arc_mix/        # 15% ARC mix experiment
```

### Checkpoints

Saved every 2500 steps + `best.pt` + `final.pt` in `checkpoints/` subdirectory.
Each checkpoint contains: `model` state_dict, `optimizer` state_dict, `config`, `step`.

For torch.compile'd models, state_dict keys have `_orig_mod.` prefix which is stripped automatically on load via `--resume`.
