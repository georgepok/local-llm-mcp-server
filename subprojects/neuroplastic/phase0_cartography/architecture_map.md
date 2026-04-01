# Nemotron-3-Nano-30B-A3B: Architecture Map

## Model Identity

| Property | Value |
|---|---|
| Model | NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 |
| Architecture | NemotronHForCausalLM |
| Type | Hybrid Mamba-Transformer + MoE |
| Hidden size | 2688 |
| Layers | 52 |
| Vocab | 131,072 |
| Max context | 262,144 tokens |
| Storage | FP8 (~30.4 GiB) / Compute: bfloat16 |

---

## Layer-by-Layer Structure

Pattern: `MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME`

```
Layer   Type    Description
─────────────────────────────────────────────────────────────────
  0      M      Mamba SSM
  1      E      MoE-FFN
  2      M      Mamba SSM
  3      E      MoE-FFN
  4      M      Mamba SSM
  5      *      ATTENTION  ◄── attention checkpoint 1 of 6
  6      E      MoE-FFN
  7      M      Mamba SSM
  8      E      MoE-FFN
  9      M      Mamba SSM
 10      E      MoE-FFN
 11      M      Mamba SSM
 12      *      ATTENTION  ◄── attention checkpoint 2 of 6
 13      E      MoE-FFN
 14      M      Mamba SSM
 15      E      MoE-FFN
 16      M      Mamba SSM
 17      E      MoE-FFN
 18      M      Mamba SSM
 19      *      ATTENTION  ◄── attention checkpoint 3 of 6
 20      E      MoE-FFN
 21      M      Mamba SSM
 22      E      MoE-FFN
 23      M      Mamba SSM
 24      E      MoE-FFN
 25      M      Mamba SSM
 26      *      ATTENTION  ◄── attention checkpoint 4 of 6
 27      E      MoE-FFN
 28      M      Mamba SSM
 29      E      MoE-FFN
 30      M      Mamba SSM
 31      E      MoE-FFN
 32      M      Mamba SSM
 33      *      ATTENTION  ◄── attention checkpoint 5 of 6
 34      E      MoE-FFN
 35      M      Mamba SSM
 36      E      MoE-FFN
 37      M      Mamba SSM
 38      E      MoE-FFN
 39      M      Mamba SSM
 40      E      MoE-FFN
 41      M      Mamba SSM
 42      *      ATTENTION  ◄── attention checkpoint 6 of 6
 43      E      MoE-FFN
 44      M      Mamba SSM
 45      E      MoE-FFN
 46      M      Mamba SSM
 47      E      MoE-FFN
 48      M      Mamba SSM
 49      E      MoE-FFN
 50      M      Mamba SSM
 51      E      MoE-FFN
─────────────────────────────────────────────────────────────────
       23 M + 6 * + 23 E = 52 total
```

**Summary:**
- Mamba (M): layers 0,2,4,7,9,11,14,16,18,21,23,25,28,30,32,35,37,39,41,44,46,48,50 — 23 layers
- MoE-FFN (E): layers 1,3,6,8,10,13,15,17,20,22,24,27,29,31,34,36,38,40,43,45,47,49,51 — 23 layers
- Attention (*): layers 5,12,19,26,33,42 — 6 layers, evenly spaced every ~7 layers

---

## Component Internals

### M — Mamba SSM Layer

```
Input: h [B, L, 2688]
    │
    ├─ LayerNorm (norm.weight [2688])
    │
    └─ Mamba2 SSM block:
        in_proj: [2688] → [5376*2] (expand=2, outputs x and z)
            │
            ├─ x branch: conv1d (kernel=4) → SiLU → chunk SSM
            │       SSM state: [64 heads × 128 states]
            │       A_log:     [64, 128]   (log decay)
            │       D:         [64]        (skip connection)
            │       dt_bias:   [64]        (dt offset)
            │
            └─ z branch (gate): SiLU
                    │
                    └─ out_proj: [5376] → [2688]

Output: h [B, L, 2688]  (residual add)
```

**Key dimensions:**
- inner_dim = 2688 × 2 = 5376
- SSM state per head: 128
- 64 SSM heads, each with head_dim=64
- n_groups=8 (shared SSM params across groups of heads)
- Sequence complexity: O(L) — linear in sequence length

### * — Attention Layer (GQA)

```
Input: h [B, L, 2688]
    │
    ├─ LayerNorm (norm.weight [2688])
    │
    └─ GQA Attention:
        Q: [2688] → [32 × 128] = [4096]    (32 query heads)
        K: [2688] → [ 2 × 128] = [ 256]    ( 2 key heads)
        V: [2688] → [ 2 × 128] = [ 256]    ( 2 value heads)
        │
        RoPE (theta=10000, partial_rotary_factor=1.0)
        │
        Attention: Q·Kᵀ/√128 → softmax → ·V
        (each K/V head serves 16 query heads — 16:1 GQA ratio)
        │
        o_proj: [32 × 128] = [4096] → [2688]

Output: h [B, L, 2688]  (residual add)
```

**Key dimensions:**
- 32 Q heads, 2 KV heads (GQA ratio 16:1)
- head_dim = 128
- Full RoPE applied to Q and K
- Sequence complexity: O(L²) — quadratic, but only 6/52 layers

### E — MoE-FFN Layer

```
Input: h [B, L, 2688]
    │
    ├─ LayerNorm (norm.weight [2688])
    │
    └─ MoE block:
        Router (gate): [2688] → [128] logits → top-6 selection
            │
            ├─ 128 Routed Experts (6 active per token):
            │       up_proj:   [2688] → [1856]
            │       relu²(x)   (squared ReLU activation)
            │       down_proj: [1856] → [2688]
            │       scaling_factor: 2.5 × normalized routing prob
            │
            └─ 1 Shared Expert (always active):
                    up_proj:   [2688] → [3712]   (2× size)
                    relu²(x)
                    down_proj: [3712] → [2688]

        output = sum(routed_expert_outputs) + shared_expert_output

Output: h [B, L, 2688]  (residual add)
```

**Key dimensions:**
- 128 routed experts, top-6 selected per token (4.7% utilization)
- 1 shared expert (always active, 2× intermediate size)
- intermediate_size=1856 for routed, 3712 for shared
- norm_topk_prob=True: routing weights normalized before scaling

---

## Information Flow Diagram

```
Tokens
  │
  ▼
backbone.embeddings [131072 × 2688]
  │
  ▼
┌─────────────────────────────────────────────────┐
│  Layer 0  (M)  Mamba SSM ──────── O(L) recurrence │
│  Layer 1  (E)  MoE-FFN ─────────── top-6 routing │
│  Layer 2  (M)  Mamba SSM                          │
│  Layer 3  (E)  MoE-FFN                            │
│  Layer 4  (M)  Mamba SSM                          │
│  Layer 5  (*)  Attention ──────── O(L²) context  │◄─ attn #1
│  Layer 6  (E)  MoE-FFN                            │
│  ...                                              │
│  Layer 12 (*)  Attention ─────────────────────── │◄─ attn #2
│  ...                                              │
│  Layer 19 (*)  Attention ─────────────────────── │◄─ attn #3
│  ...                                              │
│  Layer 26 (*)  Attention ─────────────────────── │◄─ attn #4
│  ...                                              │
│  Layer 33 (*)  Attention ─────────────────────── │◄─ attn #5
│  ...                                              │
│  Layer 42 (*)  Attention ─────────────────────── │◄─ attn #6
│  ...                                              │
│  Layer 50 (M)  Mamba SSM                          │
│  Layer 51 (E)  MoE-FFN                            │
└─────────────────────────────────────────────────┘
  │
  ▼
backbone.norm_f [2688]
  │
  ▼
lm_head [131072 × 2688]  (untied from embeddings)
  │
  ▼
Logits [vocab=131072]
```

---

## Parameter Count Estimates

### Global Parameters

| Component | Shape | Parameters |
|---|---|---|
| Embeddings | 131072 × 2688 | 352,321,536 |
| Final norm | 2688 | 2,688 |
| LM head | 131072 × 2688 | 352,321,536 |
| **Global subtotal** | | **~704.6M** |

### Per Mamba Layer (× 23 layers)

| Component | Shape | Parameters |
|---|---|---|
| in_proj | 2688 × 10752 | 28,901,376 |
| out_proj | 5376 × 2688 | 14,450,688 |
| conv1d | 64 groups × 4 kernel | ~2,048 |
| A_log | 64 × 128 | 8,192 |
| D | 64 | 64 |
| dt_bias | 64 | 64 |
| norm | 2688 | 2,688 |
| **Per-layer subtotal** | | **~43.4M** |
| **23 Mamba layers** | | **~997M** |

### Per Attention Layer (× 6 layers)

| Component | Shape | Parameters |
|---|---|---|
| q_proj | 2688 × 4096 | 11,010,048 |
| k_proj | 2688 × 256 | 688,128 |
| v_proj | 2688 × 256 | 688,128 |
| o_proj | 4096 × 2688 | 11,010,048 |
| norm | 2688 | 2,688 |
| **Per-layer subtotal** | | **~23.4M** |
| **6 Attention layers** | | **~140M** |

### Per MoE Layer (× 23 layers)

| Component | Shape | Parameters |
|---|---|---|
| gate (router) | 128 × 2688 | 344,064 |
| 128 × up_proj | 128 × (2688 × 1856) | 639,590,400 |
| 128 × down_proj | 128 × (1856 × 2688) | 639,590,400 |
| shared up_proj | 2688 × 3712 | 9,977,856 |
| shared down_proj | 3712 × 2688 | 9,977,856 |
| norm | 2688 | 2,688 |
| **Per-layer subtotal** | | **~1,299M** |
| **23 MoE layers** | | **~29.9B** |

### Total Estimate

| Section | Parameters |
|---|---|
| Global (embed + head + norm) | ~704.6M |
| 23 Mamba layers | ~997M |
| 6 Attention layers | ~140M |
| 23 MoE layers | ~29,877M |
| **TOTAL** | **~31.7B** |

Note: "30B-A3B" naming means ~30B total params, ~3B active per token (6/128 expert activation).

---

## Introspection Targets

For Phase 0 weight baseline, the highest-value tensors to analyze are:

1. **MoE router weights** (`gate.weight`, [128, 2688] per layer): Router specialization — do experts develop distinct routing signatures?
2. **Mamba A_log** (log decay matrix): Controls how fast SSM state decays — encodes temporal dynamics
3. **Mamba dt_bias**: Per-head time step bias — encodes input-independent timing
4. **Attention Q/K projections**: SVD reveals which subspaces are queried vs. keyed
5. **Expert weight norms**: Expert specialization — do experts have different activation scales?
6. **Mamba D** (skip connection): Direct residual strength per SSM head

---

## FP8 Quantization Notes

Most weight tensors come in triplets:
- `*.weight` — FP8 quantized values
- `*.weight_scale` — per-group scale factors (group_size=16)
- `*.input_scale` — per-tensor or per-token input scale

The following are NOT quantized:
- All `conv1d` layers in Mamba blocks
- Attention projection layers adjacent to Mamba blocks
- All `norm.weight` vectors
- `A_log`, `D`, `dt_bias`, `bias` vectors

For introspection: dequantize = `weight.to(float32) * weight_scale` before computing statistics.
