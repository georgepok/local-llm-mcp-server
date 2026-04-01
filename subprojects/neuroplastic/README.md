# Neuroplastic — Self-Introspecting Neural Substrate

Research into neural network self-modification via weight introspection, targeting Nemotron-3-Nano-30B-A3B on DGX Spark.

## Phases

| Phase | Name | Status | Description |
|-------|------|--------|-------------|
| 0 | Cartography | **In progress** | Map Nemotron architecture, baseline weight statistics |
| 1 | Self-Model | Planned | Build internal model of own architecture |
| 2 | Experiments | Planned | Test targeted weight modifications |
| 3 | Self-Directed | Planned | Model proposes its own modifications |
| 4 | Hebbian | Planned | Activity-dependent plasticity |
| 5 | Awareness | Planned | Self-monitoring and adaptation |
| 6 | Amplification | Planned | Targeted capability enhancement |
| 7 | Auto-Research | Planned | Autonomous research direction discovery |
| 8 | Adaptive MCMC | Planned | Stochastic self-modification |

## Phase 0 Deliverables

1. `architecture_ground_truth.json` — Verified layer types, counts, dimensions
2. `deployment_config.json` — vLLM setup details
3. `weight_baseline.json` — Per-layer statistics (mean, std, norm)
4. `architecture_map.md` — Visual architecture map

## Target Model

- **Nemotron-3-Nano-30B-A3B**: Hybrid Mamba-Transformer with MoE
- **Quantization**: FP8 on DGX Spark
- **Serving**: vLLM container (`vllm-nemotron-serve`)

## Key Documents

- `REQUIREMENTS.md` — Full specification and phase structure
- `INTELLIGENCE_SEARCH_CONSERVATION.md` — Conservation principles for self-modification

## Running

```bash
# Phase 0: map architecture
python phase0_cartography/map_architecture.py

# Fine-tuning experiments
python finetune/run_finetune.py --config finetune/config.yaml
```

**IMPORTANT**: FP8 + LoRA fine-tuning on DGX Spark requires careful memory management. The GB10's unified memory means GPU and CPU compete for the same 128GB.
