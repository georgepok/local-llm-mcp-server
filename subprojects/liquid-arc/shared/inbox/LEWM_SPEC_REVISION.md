# LEWM INTEGRATION — Spec Revision (ARPredictor)

## Correction

The original spec assumed LeWM uses an MLP predictor with signature `(z_t, a_t) → ẑ_{t+1}`. The actual shipped model uses ARPredictor — an AdaLN-zero-conditioned causal Transformer over a history window:

```
Actual interface:  predictor(emb[B, T, D], act_emb[B, T, A_emb]) → [B, T, D]
Training call:     model.predict(ctx_emb[:, :ctx_len], ctx_act[:, :ctx_len])
Loss target:       emb[:, n_preds:]
```

This is autoregressive with history, not single-step. The LiquidARCPredictor must match this interface.

## Revised Architecture

```
ARPredictor (LeWM default):
  History [B, T, D] → causal Transformer + AdaLN(action) → predicted [B, T, D]
  Flat attention: softmax(QK^T/√d) with causal mask

LiquidARCPredictor (replacement):
  History [B, T, D] → ODE + causal edge mask + MetricNet(action) → predicted [B, T, D]
  Curved attention: softmax(-D²_g/4t) with causal mask
```

### Implementation

```python
class LiquidARCPredictor(nn.Module):
    """Drop-in replacement for LeWM's ARPredictor.
    
    Matches interface: (emb[B,T,D], act_emb[B,T,A]) → [B,T,D]
    Uses ODE integration with causal mask over history window.
    """
    def __init__(self, latent_dim, action_emb_dim, ode_config):
        super().__init__()
        self.dynamics = ContinuousDynamics(ode_config)
        
        # Action conditioning: per-timestep action → context for MetricNet
        self.action_proj = nn.Linear(action_emb_dim, ode_config.d_model)
        
        # Dimension matching (if latent_dim != ode d_model)
        self.proj_in = nn.Linear(latent_dim, ode_config.d_model)
        self.proj_out = nn.Linear(ode_config.d_model, latent_dim)
        
        self.n_steps = ode_config.n_steps  # 16
        self.dropout = nn.Dropout(0.1)  # match LeWM predictor
    
    def forward(self, emb, act_emb):
        """
        Args:
            emb:     [B, T, D] history of latent embeddings
            act_emb: [B, T, A] history of action embeddings
        Returns:
            pred:    [B, T, D] predicted embeddings
        """
        B, T, D = emb.shape
        
        # Project to ODE space
        h = self.proj_in(emb)  # [B, T, d_model]
        
        # Build causal mask: position t can attend to positions 0..t
        causal_mask = torch.triu(
            torch.ones(T, T, device=h.device) * float('-inf'), diagonal=1
        )  # [T, T] — upper triangle is -inf
        self.dynamics.set_mask(causal_mask)
        
        # Action context: mean-pool action embeddings for global context
        # Or per-position: each timestep's action conditions its own metric
        action_ctx = self.action_proj(act_emb.mean(dim=1))  # [B, d_model]
        self.dynamics.set_context(action_ctx)
        
        # ODE integration: 16 steps over history with causal routing
        dt = 1.0 / self.n_steps
        for step in range(self.n_steps):
            self.dynamics.set_step_index(step, self.n_steps)
            dh = self.dynamics(t=step * dt, h=h)
            h = h + dh * dt
        
        # Project back, dropout
        pred = self.dropout(self.proj_out(h))  # [B, T, D]
        return pred
```

### Key design choices

**Causal mask as edge mask:** The dynamics.py already supports per-example [B, N, N] masks (added for the graph engine). The causal mask ensures the ODE's heat kernel only routes information from past to present — matching the ARPredictor's causal attention.

**Action conditioning:** Two options:
- **Global:** mean-pool action sequence → single context vector. Simpler. MetricNet sees "the overall action tendency" of the history.
- **Per-position:** each timestep's action conditions the metric at that position. Richer but requires extending set_context to per-position. Implement global first, per-position as ablation.

**Sequence as "graph":** Each timestep is a node. Causal edges connect t → t-1, t-2, etc. The heat kernel diffuses information backward through history (past states inform future predictions). The MetricNet determines HOW information flows — which past states matter most for each future prediction, weighted by the learned metric.

## What This Tests

The ARPredictor is a causal transformer over history — flat attention with learned QKV. The LiquidARC replacement is a causal ODE over history — curved attention with learned metric. This directly tests:

**Does curved geometry over temporal history improve dynamics prediction vs flat attention over temporal history?**

This is a stronger test than the single-step spec. We're comparing architectural paradigms (transformer vs ODE) on exactly the task each is designed for (temporal sequence prediction). If the ODE with heat kernel routing outperforms the causal transformer at predicting physical dynamics in latent space, it's a clean result.

## Everything Else From Original Spec Still Applies

- Training protocol (Phase 1-4)
- Loss function (pred + SIGReg + criticality + tau_quality)
- Evaluation metrics (control, prediction, geometry, physical understanding)
- Success criteria (all 6)
- Ablation design (A/B/C/D)
- Timeline and compute estimates

Only the predictor module interface changes. The rest of the integration is identical.

## Action Items

1. Start Spark env prep NOW (data download, deps, LeWM baseline)
2. Implement LiquidARCPredictor with AR interface
3. Verify causal mask works with existing dynamics.py mask support
4. Train Phase 1 (baseline) while building the module
5. Proceed per original spec phases
