# LiquidARC — Continuous-Time Geometric Model for ARC-AGI

Clean-slate continuous-time geometric model. One shared dynamics module applied
16 times via Euler ODE integration. No attention — all routing through
role-conditioned Riemannian heat kernel diffusion.

## Architecture

```
Input → Embedding → h₀ [B, N, d]
                      ↓
           context = ContextPool(h₀)
                      ↓
           h = EulerSolve(dynamics, h₀, t=[0,1], 16 steps)
                      ↓
              OutputHead → [B, N, 10]
```

Each Euler step (shared weights):
1. MetricNet([LN(h) || ctx]) → learned Riemannian metric g
2. Geodesic distance D² from (h, g)
3. Heat kernel K = softmax(-D²/(4t))
4. Diffusion target = tanh(W_o(K @ W_v(h)))
5. τ = softplus(τ_net(h)) + τ_min
6. dh/dt = -(1/τ)(h - target) + FFN(h)/n_steps

~830K parameters at d=256.

## Usage

```bash
# Smoke test (100 steps)
bash scripts/run.sh smoke

# Full training
bash scripts/run.sh liquid

# Flat baseline
bash scripts/run.sh flat

# Evaluation
python scripts/eval.py --checkpoint output_liquid/checkpoints/best.pt
```

## Key differences from sandwich architecture

| Sandwich (old) | LiquidARC (new) |
|----------------|-----------------|
| 4 independent layer weights | 1 shared dynamics module |
| Attention layers for routing | Geometry-only routing |
| Discrete layer stack | Continuous ODE flow |
| 3.8M params | ~830K params |
| Geometry is a component | Geometry IS the computation |
