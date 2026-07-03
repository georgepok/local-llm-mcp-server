# LAYER-WISE ODE CO-PROCESSING — LiquidARC as Parallel Geometric Processor

## Core Idea

Instead of hooking the ODE at one layer and injecting bias into later layers, run the ODE AS A PARALLEL PROCESSOR alongside the LLM. At every layer, the residual stream enters the ODE, and the ODE produces a bias for the next layer's attention. The ODE progressively accumulates a geometric model of the LLM's computation as it unfolds through depth.

```
Layer 1:  h₁ → ODE receives h₁, step 1/36 → produces B₁
Layer 2:  attn(h₁) + B₁ → h₂ → ODE receives h₂ (has h₁ context), step 2/36 → B₂
Layer 3:  attn(h₂) + B₂ → h₃ → ODE receives h₃ (has h₁,h₂), step 3/36 → B₃
...
Layer 36: attn(h₃₅) + B₃₅ → h₃₆ → ODE has full computation trajectory → output
```

The LLM and ODE are in continuous dialogue through the depth of the network.

## Why This Is Different

**Single-hook approach:** ODE sees ONE snapshot of the residual stream. Produces ONE bias. The LLM's computation above and below the hook is unaware of geometry.

**Layer-wise approach:** ODE sees the ENTIRE computation unfold. Produces DEPTH-DEPENDENT biases that shape how the computation evolves. Early biases are crude (syntactic). Late biases are rich (full reasoning context). The ODE and LLM CO-EVOLVE through depth.

This means the ODE doesn't just modify WHERE attention goes — it modifies HOW THE COMPUTATION UNFOLDS. Bias at layer 5 changes h₅, which changes layer 6's input, which changes the ODE's state at step 6, which changes the bias at layer 7... The system is a coupled dynamical system through the depth dimension.

## Architecture

### ODE Design: One Step Per Layer

The current ODE uses 16 Euler steps in a single integration. The layer-wise architecture distributes integration across the LLM's depth: ONE Euler step per layer, 36 steps total for Qwen3-4B. This is more integration steps than the current architecture, and each step is informed by the actual computation at that depth.

```python
class LayerWiseODE:
    """LiquidARC ODE that co-processes with the LLM layer by layer."""
    
    def __init__(self, dynamics, d_model, n_layers):
        self.dynamics = dynamics           # shared ContinuousDynamics (MetricNet, TauNet, FFN)
        self.n_layers = n_layers
        self.h_ode = None                  # accumulated ODE state
        self.h_state = None                # persistent state across calls [1, K, d]
        self.layer_biases = []             # diagnostic: bias at each layer
        
    def start_forward(self, h_state=None):
        """Called at the start of each LLM forward pass."""
        self.h_ode = None
        self.layer_biases = []
        if h_state is not None:
            self.h_state = h_state
    
    def process_layer(self, layer_idx, h_residual):
        """Called at each LLM layer. Returns attention bias for this layer.
        
        Args:
            layer_idx: 0-based layer index
            h_residual: [B, N, d] residual stream at this layer
        
        Returns:
            B: [N, N] attention bias matrix for this layer
        """
        B, N, d = h_residual.shape
        
        if self.h_ode is None:
            # First layer: initialize ODE state
            # Combine persistent state (if any) with current residual
            if self.h_state is not None:
                K = self.h_state.shape[1]
                self.h_ode = torch.cat([self.h_state, h_residual], dim=1)  # [B, K+N, d]
            else:
                self.h_ode = h_residual.clone()
        
        # Tell dynamics which layer we're at (uses step embeddings)
        self.dynamics.set_step_index(layer_idx, self.n_layers)
        
        # Context for MetricNet: mean of current residual stream
        self.dynamics.set_context(h_residual.mean(dim=1))
        
        # ONE Euler step: dh/dt from dynamics
        dh = self.dynamics(t=layer_idx, h=self.h_ode)
        dt = 1.0 / self.n_layers
        self.h_ode = self.h_ode + dh * dt
        
        # Sensory forcing: blend new residual information into ODE state
        # Only the prompt positions (not persistent state positions)
        K = self.h_state.shape[1] if self.h_state is not None else 0
        alpha = 0.2  # coupling strength (how much new layer info enters ODE)
        self.h_ode[:, K:, :] = (1 - alpha) * self.h_ode[:, K:, :] + alpha * h_residual
        
        # Compute bias from ODE state (prompt positions only)
        h_prompt = self.h_ode[:, K:, :]
        B = compute_bias(h_prompt)  # [N, N]
        
        self.layer_biases.append(B.detach())
        return B
    
    def end_forward(self):
        """Called after the last layer. Updates persistent state."""
        if self.h_state is not None:
            K = self.h_state.shape[1]
            self.h_state = self.h_ode[:, :K, :].detach()
        return self.h_state
```

### LLM Integration (Qwen3-4B)

Register a forward pre-hook on EVERY attention layer:

```python
ode = LayerWiseODE(dynamics, d_model=2048, n_layers=36)

for i, layer in enumerate(model.layers):
    def make_hook(layer_idx):
        def hook(module, args):
            h = args[0]  # residual stream input to this layer
            B = ode.process_layer(layer_idx, h)
            # Store B for injection into this layer's attention
            module._ode_bias = B
            return args
        return hook
    layer.register_forward_pre_hook(make_hook(i))
    
    # Also modify the attention to use the bias
    def attn_hook(module, args, kwargs):
        if hasattr(module, '_ode_bias'):
            # Add bias to attention logits
            kwargs['attn_bias'] = module._ode_bias
        return args, kwargs
    layer.self_attn.register_forward_pre_hook(attn_hook, with_kwargs=True)
```

### Sustained Criticality Per Layer

The criticality system operates on each layer's ODE step:

```python
# In the training loop, after each forward pass:
for layer_idx, B in enumerate(ode.layer_biases):
    # Compute D²/4τ at this layer depth
    g = ode.dynamics.get_metric_at_step(layer_idx)
    tau = ode.dynamics.get_tau_at_step(layer_idx)
    ratio = compute_criticality_ratio(g, tau)
    
    # Per-layer criticality loss (target may vary by depth)
    target = criticality_schedule(layer_idx, n_layers)  # e.g., 18 for all, or depth-varying
    layer_crit_loss += smooth_l1(ratio, target)

total_loss = ce_loss + lambda_crit * layer_crit_loss / n_layers
```

The criticality target could be:
- **Uniform:** same D²/4τ target at every layer (simplest)
- **Depth-varying:** higher targets at early layers (less routing needed), lower at late layers (more structural routing)
- **Adaptive:** each layer finds its own critical point via the D² EMA tracking

Start with uniform. Let the data tell us if depth-varying is needed.

## What This Uniquely Enables

### 1. Computation steering, not just attention steering
Bias at layer 5 changes h₅, which changes what layer 6 computes. The entire computation trajectory is shaped by geometric routing. By layer 36, the output has been continuously co-processed by the ODE.

### 2. Depth-dependent routing
Early layers: syntactic biases ("attend within the same clause")
Middle layers: semantic biases ("attend to the same topic cluster")  
Late layers: structural biases ("attend to the causal chain")
Same ODE weights, different step embeddings → depth-appropriate behavior.

### 3. Active hallucination prevention
The ODE can detect at layer 20 that the model is about to diverge from correct reasoning (residual for "bridge" separating from "shortage"). Bias at layer 21 increases their coupling. The hallucination is prevented DURING computation, not caught afterward.

### 4. Phase transitions through depth
Novel reasoning patterns might trigger a mini phase transition mid-forward-pass. The ODE's CV could spike at layer 15 when the LLM encounters something requiring new routing, reorganize, and settle by layer 25. Structural adaptation DURING inference.

## Computational Cost

### Per forward pass:
- 36 ODE Euler steps (one per layer)
- Each step: MetricNet forward + TauNet forward + SDPA + FFN + tau modulation
- At d=2048, N=500: each step ≈ 1-3ms on Spark GPU
- Total ODE overhead: 36 × 2ms = ~72ms per forward pass
- Qwen3-4B forward pass: ~50-100ms
- Total: ~120-170ms (1.5-2× overhead)

### Optimization opportunities:
- Reduce ODE d_model below LLM d_model (e.g., d_ode=512, project in/out)
- Skip ODE at early layers (layers 1-4 rarely need geometric routing)
- Share MetricNet computation across adjacent layers (metric changes slowly)
- Async ODE computation (pipeline layer L's ODE step with layer L+1's attention)

## Persistent State

Between conversation turns, the ODE's h_state (K=32-64 positions) persists. It carries the geometric structure accumulated from ALL prior turns, processed through ALL layers. This is richer than single-layer h_state because it's been shaped by the full depth of the computation.

On the next turn:
1. h_state enters at layer 0 alongside the new prompt
2. Through 36 layers of co-processing, h_state influences the new computation
3. After layer 36, h_state is updated with this turn's geometric structure
4. Ready for the next turn

## Starting Point: Qwen3-4B

Qwen3-4B is the right model to start with:
- 36 layers (enough depth for meaningful progressive accumulation)
- d_model=2048 (manageable ODE dimensions, or project to d_ode=512)
- 4B params (fits comfortably on Spark alongside LiquidARC)
- Pure attention (every layer has attention to bias — no Mamba layers to skip)
- Already integrated in the current Mind setup

### Phase 1: Proof of concept
- Hook all 36 layers with ODE co-processing
- Use existing ContinuousDynamics with step embeddings set to layer indices
- No criticality loss initially — just observe what the ODE does
- Run the causal chain test: does per-layer routing improve chain reasoning?
- Measure: CV per layer, D² per layer, bias range per layer

### Phase 2: Add criticality scaffolding
- D²/4τ loss per layer (uniform target initially)
- tau_quality_loss per layer
- Convergence coupling per layer
- Measure: does criticality improve at ALL depths or only certain layers?

### Phase 3: Optimize
- Find which layers benefit most from ODE routing (likely middle + late)
- Skip ODE at non-beneficial layers
- Reduce d_ode if full d_model is unnecessary
- Profile and optimize for real-time generation

## Connection to Prior Work

This architecture is what FGN v3's parallel transport was always describing: information transported along geodesics through the manifold of the computation. Each layer is a point on the manifold. The ODE computes the metric at each point. The bias is the connection coefficient that tells the LLM how to transport information from one layer to the next while respecting the geometry.

The phase transition becomes a property of the DEPTH trajectory, not just the training trajectory. The system can reorganize its routing structure mid-forward-pass when it encounters novel structure — exactly the "meta-learning at the geometry level" the reviewer described.

## What Transfers From Current Work

- ContinuousDynamics (MetricNet, TauNet, FFN) — shared weights across all layer steps
- Step embeddings — originally for 16 ODE steps, now for 36 LLM layers
- Sustained criticality losses (D²/4τ, tau_quality, curvature diversity)
- τ-CV coupling and τ-convergence coupling
- State cosine + displacement bias computation
- h_state persistence across calls
- All diagnostic infrastructure (CV, D², entropy, B_across/B_within)

What changes:
- Integration distributed across LLM layers instead of concentrated in 16 steps
- Sensory forcing from residual stream at each layer (not from extracted deltas)
- Bias injection at every layer (not just middle-third layers)
- No delta extraction, no buffer, no window — the LLM IS the input source
