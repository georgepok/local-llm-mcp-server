# THE REAL PROBLEM: MetricNet Needs Training on LLM Residuals

## Diagnosis

Every architectural change (buffers, windows, deltas, perturbation, layer-wise hooks) was solving the wrong problem. The MetricNet was trained on ARC embeddings. It produces near-uniform g (CV=0.51) on LLM residual streams because it was never trained to differentiate LLM features.

```
MetricNet on ARC embeddings:      CV ≈ 7.0  (post-transition, structured)
MetricNet on random vectors:      CV ≈ 1.05 
MetricNet on LLM residual stream: CV ≈ 0.51 (FLAT — no routing)
```

No amount of coupling, perturbation, or normalization can make flat g produce structured routing. The architecture is correct. The weights are wrong for this domain.

## What's Validated (keep everything)

- Layer-wise co-processing architecture
- Perturbation engine (ε-bounded corrections)
- B_within/B_across separation through depth
- Sustained criticality system (D²/4τ, tau_quality, convergence coupling)
- State cosine + displacement bias mechanism
- All diagnostic infrastructure

## What's Needed: End-to-End Training with Layer-Wise Hooks

### Training loop

```python
# Freeze Qwen3-4B (LLM weights don't change)
for param in qwen3.parameters():
    param.requires_grad = False

# Train LiquidARC components only
optimizer = Adam([
    {'params': dynamics.metric_net.parameters(), 'lr': 1e-3},   # MetricNet — HIGH LR
    {'params': dynamics.tau_net.parameters(), 'lr': 1e-3},       # TauNet
    {'params': dynamics.ffn.parameters(), 'lr': 1e-4},           # FFN
    {'params': dynamics.W_v.parameters(), 'lr': 1e-4},           # Value projection
    {'params': dynamics.W_o.parameters(), 'lr': 1e-4},           # Output projection
])

for batch in dataloader:
    input_ids, target_ids = batch
    
    # Forward pass with layer-wise ODE hooks active
    ode_hook.start_forward()
    logits = qwen3(input_ids)  # hooks inject bias at every layer
    
    # CE loss (does the geometric routing improve prediction?)
    ce_loss = F.cross_entropy(logits.view(-1, vocab_size), target_ids.view(-1))
    
    # Criticality loss per layer
    crit_loss = sum(
        compute_criticality_loss(ode_hook.layer_metrics[i])
        for i in range(n_layers)
    ) / n_layers
    
    # Tau quality loss
    tau_loss = compute_tau_quality_loss(ode_hook.layer_taus)
    
    total_loss = ce_loss + 0.01 * crit_loss + 0.05 * tau_loss
    total_loss.backward()
    optimizer.step()
```

### Gradient flow

```
CE loss → Qwen3 attention logits (frozen, but gradients flow through)
       → attention bias B (from ODE state cosine)
       → ODE state (from perturbation correction)  
       → MetricNet weights (THESE UPDATE)
       → TauNet weights (THESE UPDATE)
```

The CE loss tells the MetricNet: "your routing improved the LLM's prediction." The MetricNet learns which dimensions of LLM residuals matter for routing.

### Training data

**Option A: WikiText/generic text (broad adaptation)**
- MetricNet learns general LLM residual stream statistics
- Routing emerges from what helps generic NTP
- Likely produces modest improvements — generic text doesn't strongly reward geometric routing

**Option B: Multi-hop reasoning tasks (targeted adaptation)**
- GSM8K, StrategyQA, EntailmentBank, or synthetic causal chains
- The CE loss specifically rewards connecting causally related tokens
- MetricNet learns: "these residual stream features indicate causal chain membership"
- More likely to produce the structured routing we need

**Option C: Mixed (recommended)**
- 70% generic text (prevents overfitting to specific reasoning patterns)
- 30% multi-hop reasoning (provides targeted geometric routing signal)

### Starting point

Initialize from the ARC checkpoint but with HIGH MetricNet LR (1e-3):
- The MetricNet architecture (how to compute g from features) transfers
- The specific routing patterns (which features to weight) need to change
- High LR allows rapid adaptation — routing patterns should shift within 500-1000 steps
- Criticality scaffolding prevents the adaptation from destabilizing

### Expected outcome

After training:
- CV on LLM residuals: 3.0-7.0 (structured, depth-dependent)
- D²/4τ per layer: near criticality target
- B_within > B_across: by meaningful margin (not 2% correction, but 20-50%)
- Causal chain test: improvement on 5-hop chain (the one both plain and ODE currently fail)

### Timeline estimate

- Data preparation: 2-4 hours (WikiText + synthetic causal chains)
- Training infrastructure: existing criticality training code + layer-wise hooks
- Training run: ~2000-6000 steps at d=2048 (4-12 hours on Spark)
- Evaluation: causal chain test suite + per-layer diagnostics

### What changes in the code

1. **Training script**: new `train_layer_wise.py` that runs Qwen3-4B forward with ODE hooks, computes CE loss through biased attention, backprops to MetricNet/TauNet
2. **Gradient flow**: ensure ODE perturbation path is differentiable (no detach() between bias computation and MetricNet)
3. **Data loader**: WikiText batches + causal chain batches
4. **Logging**: per-layer CV, D²/4τ, B_within/B_across, plus aggregate CE loss

The layer-wise hook architecture is ready. The criticality scaffolding is ready. The perturbation engine is ready. The only missing piece is the training loop that connects CE loss to MetricNet gradients through the layer-wise bias injection.

## The Insight

We spent weeks making the architecture work perfectly — and it DOES work perfectly. The perturbation is bounded. The depth evolution is correct. The bias separation is structurally right. The criticality system maintains the operating regime.

But a perfectly engineered pipeline carrying flat metrics produces flat routing. The pipeline isn't the problem. The signal source (MetricNet weights) needs to be trained for the signal domain (LLM residuals).

This is the same lesson from the FGN→LiquidARC→Mind arc: the architecture was always right. The phase transition, the heat kernel, the Riemannian metric — all correct concepts. The challenge was always getting the right INPUT into the geometric processor and training it to produce the right OUTPUT for the specific domain.

Now the domain is LLM residual streams, and the architecture (layer-wise perturbation) is validated. The training loop is the final piece.
