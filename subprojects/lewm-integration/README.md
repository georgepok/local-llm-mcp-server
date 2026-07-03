# LeWM × LiquidARC Integration

Drop-in replacement for LeWorldModel's `ARPredictor` that uses LiquidARC's
continuous-time ODE on a learned Riemannian manifold instead of flat causal
self-attention.

Spec: `../liquid-arc/shared/inbox/LEWM_INTEGRATION_SPEC.md` +
`LEWM_SPEC_REVISION.md` (AR interface correction).

## Layout

```
lewm-integration/
├── le-wm/              vendored upstream (commit ca231f9, read-only)
├── liquid_arc_lewm/    LiquidARCPredictor module
├── configs/            Hydra overrides for phased training
├── scripts/            train.py (fork of upstream with predictor swap)
└── tests/              smoke tests
```

## Interface

```python
LiquidARCPredictor(input_dim, action_emb_dim, ode_config, output_dim=None)
forward(emb[B, T, D_in], act_emb[B, T, A]) -> [B, T, D_out]
```

Matches upstream `ARPredictor` signature exactly. Output dim defaults to
input dim; set separately when LeWM's JEPA wants hidden_dim (384) out of
embed_dim (192) in.

## How it replaces attention

| ARPredictor (upstream)                | LiquidARCPredictor              |
|---------------------------------------|---------------------------------|
| Causal self-attention over T frames   | Causal ODE over T frames        |
| QK^T/√d softmax                       | softmax(-D²_g / 4t)             |
| AdaLN(action) per block               | MetricNet(action) as context    |
| Single forward pass                   | 16 Euler steps                  |

Causal mask is an upper-triangular bool `[T, T]` passed via
`dynamics.set_context(ctx, mask=causal)`. `True = BLOCKED` — past can't see
future.

## Phased execution (per spec)

- **Phase 1** — reproduce LeWM baseline on PushT (upstream train.py).
- **Phase 2** — swap predictor, train with `pred + SIGReg` only.
- **Phase 3** — add criticality + tau_quality losses. Enable in
  `configs/lewm_liquid.yaml → loss.criticality.enabled: True`.
- **Phase 4** — ablations A/B/C/D (baseline vs ODE vs +crit vs +metric-weighted).

## Running

```bash
cd subprojects/lewm-integration
export PYTHONPATH=$PWD:$PWD/../liquid-arc:$PWD/le-wm

# Smoke tests
python tests/test_predictor.py

# Phase 2/3 training (on Spark, fgn-train container)
python scripts/train.py --config-path=../configs --config-name=lewm_liquid
```

Data: `$STABLEWM_HOME/pusht_expert_train.h5` from
`huggingface.co/datasets/quentinll/lewm-pusht`.
