# Nemotron-3-Nano-30B-A3B — Architecture Summary

Source: actual `config.json` from `/home/pokazge/models/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8/`

## Core Dimensions
- **hidden_size:** 2688
- **num_hidden_layers:** 52
- **vocab_size:** 131,072
- **max_position_embeddings:** 262,144
- **Total parameters:** ~30B (30.4GB in FP8)
- **Active parameters per forward pass:** ~3B (A3B designation)

## Layer Pattern

```
hybrid_override_pattern: MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME
                         ─────┬──────┬──────┬──────┬──────┬────────┬────────
                         Block1 Block2 Block3 Block4 Block5 Block6   Block7
```

M = Mamba (23 layers), E = MoE-FFN (23 layers), * = Attention (6 layers)

Pattern: Mamba and MoE alternate, with attention layers inserted every ~8 layers.
The last two blocks are longer (8 layers between attention points vs 6).

## Attention (6 layers: 5, 12, 19, 26, 33, 42)
- **Type:** Grouped Query Attention (GQA)
- **Query heads:** 32, **KV heads:** 2 (16:1 ratio — aggressive compression)
- **Head dimension:** 128
- **Positional encoding:** RoPE (theta=10000, full rotary)
- **No bias**

## Mamba SSM (23 layers: 0, 2, 4, 7, 9, 11, ...)
- **Heads:** 64 (multi-head SSM)
- **Head dimension:** 64
- **State size:** 128
- **Conv kernel:** 4
- **Expand factor:** 2 (projects hidden_size to 2x before SSM)
- **Chunk size:** 128
- **Groups:** 8
- **Activation:** SiLU

## MoE-FFN (23 layers: 1, 3, 6, 8, 10, ...)
- **Routed experts:** 128
- **Active per token:** 6 (top-6 routing)
- **Expert intermediate size:** 1856
- **Shared expert:** 1 (always active, intermediate_size=3712 — 2x routed)
- **Routing scale factor:** 2.5
- **Activation:** relu^2

## Quantization
- **Method:** FP8 via NVIDIA ModelOpt 0.29.0
- **KV cache:** FP8
- **Excluded from FP8:** All conv1d layers + layers adjacent to attention boundaries (layers 4-5, 11-12, 18-19, 25-26, 32-33, 41-42)

## Information Flow

```
Input tokens
    │
    ▼
[Embedding] (131072 × 2688)
    │
    ▼
┌─ Layer 0: Mamba ──┐
│  Layer 1: MoE-FFN │
│  Layer 2: Mamba   │   Block 1
│  Layer 3: MoE-FFN │
│  Layer 4: Mamba   │
├─ Layer 5: ATTN ───┤   ← Global context mixing
│  Layer 6: MoE-FFN │
│  Layer 7: Mamba   │   Block 2
│  ...              │
├─ Layer 12: ATTN ──┤   ← Global context mixing
│  ...              │
├─ Layer 19: ATTN ──┤
│  ...              │
├─ Layer 26: ATTN ──┤
│  ...              │
├─ Layer 33: ATTN ──┤
│  ...              │
├─ Layer 42: ATTN ──┤   ← Last attention (layer 42 of 52)
│  Layer 43-51:     │   Last 9 layers: alternating MoE + Mamba, no attention
└───────────────────┘
    │
    ▼
[RMSNorm] (norm_f)
    │
    ▼
[LM Head] (2688 × 131072)
    │
    ▼
Logits
```

## Key Architectural Observations for Self-Modification

1. **Attention is sparse and precious** — only 6 out of 52 layers. These are the model's only mechanism for global token-to-token interaction. Mamba provides local/sequential context; attention provides global.

2. **Last 9 layers have no attention** — the tail of the network (43-51) relies entirely on Mamba SSM + MoE. Any self-modification targeting long-range reasoning should focus on layers 33-42.

3. **MoE routing is very wide** — 128 experts with top-6 selection. This is unusually high fan-out. Expert specialization analysis (Phase 0 Task 3) will reveal if experts are well-differentiated or homogeneous.

4. **FP8 exclusion zones around attention** — NVIDIA excluded the Mamba layers immediately before each attention layer from FP8 quantization. These transition layers are considered precision-sensitive.

5. **GQA 16:1 ratio is extreme** — 32 query heads share only 2 KV heads. This maximizes compute efficiency but limits the diversity of key-value representations. Understanding how this bottleneck affects information flow is critical for self-modification.
