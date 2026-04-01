# Nemotron-3-Nano-30B — Weight Baseline Analysis

Source: `weight_baseline.json` — 18,179 tensors, 30.4GB FP8

## MoE Expert Diversity

All 23 MoE layers have 128 routed experts + 1 shared expert. Expert weight norms show:

| Layer | up_proj CV | down_proj CV | Gate norm | Interpretation |
|-------|-----------|-------------|-----------|----------------|
| 1     | 0.019     | 0.011       | 40.21     | Very homogeneous |
| 3     | 0.040     | 0.035       | 37.98     | Slightly more differentiated |
| 6     | 0.033     | 0.025       | 31.85     | Moderate |
| 24    | 0.030     | 0.029       | 14.74     | Gate norm decreasing |
| 40    | 0.015     | 0.016       | 9.79      | Very homogeneous, weak gate |
| 51    | 0.028     | 0.027       | 11.23     | Moderate |

**Key insight:** Expert weight coefficient of variation (CV) is extremely low (1-4%). The 128 experts are nearly identical by weight norm — differentiation is subtle. Gate weight norms decrease from 40 (layer 1) to ~10 (layer 40+), suggesting routing decisions become weaker in deeper layers.

This is important for self-modification: expert weights are highly redundant. Small perturbations to individual experts should have minimal impact, making experts relatively safe targets for experimentation.

## Attention Layers — Depth Progression

| Layer | Q norm | K norm | V norm | O norm |
|-------|--------|--------|--------|--------|
| 5     | 98.07  | 39.81  | 6.32   | 58.30  |
| 12    | 98.66  | 37.15  | 7.67   | 69.41  |
| 19    | 94.38  | 28.41  | 13.02  | 75.36  |
| 26    | 88.78  | 22.03  | 20.95  | 82.19  |
| 33    | 86.93  | 19.15  | 29.24  | 90.67  |
| 42    | 80.28  | 16.32  | 51.16  | 101.00 |

**Clear depth gradient:**
- **Q norms decrease** (98 → 80): queries become less sharp with depth
- **K norms decrease** (40 → 16): key projections shrink dramatically
- **V norms increase** (6 → 51): value projections grow 8x from first to last attention
- **O norms increase** (58 → 101): output projection strengthens

**Interpretation:** Early attention layers (L5, L12) act more as matchers (strong Q/K, weak V) — they route but don't transform much. Later layers (L33, L42) act more as transformers (strong V/O) — they carry significant content. The last attention layer (L42) has the largest V and O norms in the entire network.

This V-norm gradient is critical for self-modification planning: modifications to V/O in layer 42 will have outsized impact on output quality compared to layer 5.

## Mamba SSM Parameters

Per-layer stats for key SSM tensors (layer 0 as example):
- **A_log:** mean=-1.31, std=2.96, range=[-7.5, 7.5] — controls state decay rates
- **D:** scalar skip connection per head
- **dt_bias:** timestep bias, affects how much each token can update state

## FP8 Quantization Notes

- Weights stored in FP8 with per-group (group_size=16) scaling factors
- `weight_scale` and `input_scale` tensors accompany each quantized weight
- Conv1d layers and attention boundary layers excluded from FP8 (kept in BF16)
- Statistics computed after dequantization to float32
