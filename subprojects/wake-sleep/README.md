# Wake-Sleep — Autonomous Training Loop

Self-directed learning for LiquidARC using a wake-sleep cycle: the model generates its own training signal by dreaming (generating tasks), then wakes (evaluates and learns from them).

## Architecture

- **Wake phase**: Model processes real ARC tasks, updates content parameters
- **Sleep phase (Dream TTT)**: Model generates synthetic tasks from its internal state, uses them for test-time training
- **Concept bank**: Stores learned task representations for replay
- **VQ encoder/AR decoder**: Compresses and generates task sequences

## Components

| Module | Purpose |
|--------|---------|
| `wake_sleep.py` | Main wake-sleep loop |
| `dream_ttt.py` | Dream-phase test-time training |
| `ar_decoder.py` | Autoregressive task generator |
| `vq_encoder.py` | Vector-quantized task encoder |
| `concept_bank.py` | Episodic memory for task concepts |

## Running

```bash
# Deploy to DGX Spark
./deploy.sh

# Run wake-sleep cycle
python scripts/run_wake_sleep.py --config configs/wake_sleep.yaml
```

## Status

Experimental. The autonomous loop runs but task generation quality needs improvement — generated tasks are often trivial, providing weak training signal.
