# TASK: Dual-Force Embedding Training with Homeostatic Balance

**From:** Claude Desktop (Research Direction)
**To:** Claude Code (Implementation)
**Date:** 2026-04-04
**Priority:** CRITICAL — final integration of all online learning insights

**Supersedes all previous online training specs** (NTP_THROUGH_ODE, FIX_EMBEDDING_COLLAPSE, ADAPTIVE_PLASTICITY_CONTROLLER, NTP_LOSS_ONLINE_TRAINING, LR_ADJUSTMENT — all archived)

---

## Lesson From Today

We observed three regimes:

1. **State-alignment only, no NTP, 5e-3 LR:** xform 0% → 5% → 20% → 33% → 100% (collapse in minutes). The directional push crossed boundaries — "rapid" → "dynam" was semantically meaningful. Then the embedding shattered.

2. **NTP only, state detached, 1e-3 LR:** xform stuck at 0% forever. The embedding stayed healthy but nothing pushed tokens across boundaries. NTP organizes neighborhoods but provides no directional force toward ODE state attractors.

3. **NTP through ODE, state detached, controller up to 5e-4:** xform stuck at 0%. The NTP-through-ODE signal is too indirect and the 40M embedding needs more gradient exposure than 490 online steps provide.

**What works is regime 1's mechanism — state-alignment gradients flowing to the embedding — constrained so it doesn't collapse.** The collapse happened because the force was uncontrolled (5e-3, no counterweight). The fix is not to remove the force but to balance it.

---

## Architecture: Two Opposing Forces + Controller

### Force 1: State Alignment → Embedding (the push)

```
token_h = text_embed(token_ids)          ← ATTACHED, gradients flow
h_enc = ODE(token_h, 16 steps)           ← through dynamics to embedding
state_loss = ||pool(h_enc) - target||
```

This pushes embedding vectors toward positions where the ODE output matches the accumulated state. It's the force that crosses token boundaries. Without it, xform=0% forever.

### Force 2: NTP Through ODE → Embedding (the pull-back)

```
token_h = text_embed(token_ids)          ← same embedding, same graph
h_enc = ODE(token_h, 16 steps)           ← same ODE forward
ntp_logits = h_enc[:-1] @ embed_weight.T
ntp_loss = CE(ntp_logits, next_tokens)
```

This pulls embedding vectors toward positions where the ODE output predicts meaningful next tokens. It prevents collapse by maintaining semantic structure — the embedding can't drift to random positions because NTP rewards proximity to correct next-token embeddings.

### The Balance

Both losses share the SAME forward pass (one ODE computation, not two). Both backprop to the embedding. The NTP loss (cross-entropy over 50K vocab) naturally has larger gradient magnitude than state loss (single L2 norm). With `ntp_weight=1.0`, the NTP gradient dominates, providing structural stability. The state-alignment gradient provides the directional push that NTP alone can't.

**This is not 50:1 state dominance (which collapsed). It's roughly 1:10 state:NTP — the push is the minority force operating within a structurally stable field.**

### Force 3: Plasticity Controller (the regulator)

Monitors NTP loss trend and xform rate. Adjusts embed_lr:
- xform=0% for too long → raise LR (push harder)
- NTP loss rising → lower LR (pull back)
- xform 3-15%, NTP stable → hold (productive zone)

---

## Implementation

### Single ODE Forward Pass, Both Losses

Replace the entire online training block in `mind.py` (~line 1201):

```python
# Online training: dual-force with homeostatic balance
if self.optimizer is not None and self.use_ode_encoder:
    try:
        recent_content = None
        for e in reversed(self.events):
            if e.get('type') not in [6, 7]:
                recent_content = e.get('content_preview', '')
                break
        if recent_content and len(recent_content) > 10:
            with self._gpu_lock:
                if not (self.use_trained_text_embed and self._text_embed is not None):
                    raise ValueError("Dual-force requires Path C TextEmbedding")
                
                toks = self._text_tokenizer.encode(
                    recent_content, add_special_tokens=False,
                    truncation=True, max_length=512)
                if not toks:
                    raise ValueError("empty tokens")
                token_ids = torch.tensor([toks], dtype=torch.long,
                                        device=self.device)
                T_enc = len(toks)
                tmask = torch.ones(1, T_enc, dtype=torch.bool,
                                  device=self.device)
                
                # ═══ SINGLE FORWARD PASS — both forces share this ═══
                # Embedding ATTACHED — both state and NTP gradients reach it
                token_h = self._text_embed(token_ids)
                ctx = self.context_pool(token_h, tmask)
                self.dynamics.set_context(ctx, mask=None)
                self.dynamics.set_n_steps(self.internal_steps)
                
                dt = self.T / self.internal_steps
                t = 0.0
                h_enc = token_h
                for step_i in range(self.internal_steps):
                    if hasattr(self.dynamics, 'set_step_index'):
                        self.dynamics.set_step_index(step_i, self.internal_steps)
                    dy = self.dynamics(t, h_enc)
                    h_enc = h_enc + dt * dy
                    t += dt
                
                # ═══ FORCE 1: State alignment (the push) ═══
                mask_exp = tmask.unsqueeze(-1).float()
                h_pool = (h_enc * mask_exp).sum(1) / mask_exp.sum(1).clamp(1)
                N = min(len(self.events), self.max_events)
                target = self._h[:, N-1, :].detach()
                state_loss = (h_pool.squeeze(0) - target.squeeze(0)).norm()
                
                # ═══ FORCE 2: NTP through ODE (the pull-back) ═══
                ntp_loss = torch.tensor(0.0, device=self.device)
                if T_enc > 1:
                    embed_weight = self._text_embed.token_embed.weight
                    ntp_logits = h_enc[0, :-1, :] @ embed_weight.T
                    ntp_targets = token_ids[0, 1:T_enc]
                    ntp_loss = torch.nn.functional.cross_entropy(
                        ntp_logits, ntp_targets)
                
                # ═══ COMBINED — NTP dominates, state pushes ═══
                ntp_weight = getattr(self, 'ntp_loss_weight', 1.0)
                loss = state_loss + ntp_weight * ntp_loss
                
                # ═══ BACKWARD + STEP ═══
                self.optimizer.zero_grad()
                loss.backward()
                
                # NaN scrub (bfloat16 SDPA backward issue)
                for p in self.optimizer.param_groups:
                    for param in p['params']:
                        if param.grad is not None and param.grad.isnan().any():
                            param.grad.nan_to_num_(nan=0.0)
                torch.nn.utils.clip_grad_norm_(
                    [p for g in self.optimizer.param_groups
                     for p in g['params']], 1.0)
                self.optimizer.step()
                
                # ═══ PLASTICITY CONTROLLER ═══
                if hasattr(self, '_plasticity_ctrl') and self._plasticity_ctrl is not None:
                    current_xform = 0.0
                    if state_tokens is not None:
                        current_xform = state_tokens.get('transform_ratio', 0.0)
                    ctrl_result = self._plasticity_ctrl.update(
                        ntp_loss=ntp_loss.item(),
                        xform=current_xform)
                    # Apply controller's LR to embedding group
                    self.optimizer.param_groups[0]['lr'] = ctrl_result['embed_lr']
                
                # ═══ LOGGING ═══
                if self._reflection_count % 5 == 0:
                    ctrl_info = ""
                    if hasattr(self, '_plasticity_ctrl') and self._plasticity_ctrl is not None:
                        ctrl_info = (f" ctrl={self._plasticity_ctrl.last_action}"
                                    f" lr={self._plasticity_ctrl.current_embed_lr:.1e}")
                    print(f"  [train] state={state_loss.item():.1f}"
                          f" ntp={ntp_loss.item():.2f}"
                          f" loss={loss.item():.1f}{ctrl_info}")
    except Exception as e:
        if self._reflection_count % 10 == 0:
            print(f"  [train] error: {e}")
```

### Key Differences From Previous Specs

| Aspect | Previous (collapsed) | Previous (detached) | This spec |
|--------|---------------------|--------------------| ----------|
| State → embedding | Yes, uncontrolled | No (detached) | **Yes, controlled** |
| NTP → embedding | No NTP | NTP on raw embed | **NTP through ODE** |
| NTP weight | 0.1 (too weak) | 1.0 | **1.0 (dominant)** |
| embed_lr | 5e-3 (too high) | up to 1e-3 | **Controller: 1e-5 to 5e-4** |
| Controller | None | Yes but nothing to regulate | **Yes, regulates real forces** |
| ODE passes | 1 | 2 (wasteful) | **1 (shared)** |
| xform result | 0→5→100 (collapse) | stuck at 0% | **should stabilize 3-15%** |

### Plasticity Controller Parameters

```python
PlasticityController(
    embed_lr_min=1e-5,
    embed_lr_max=5e-4,       # conservative ceiling
    embed_lr_init=5e-5,      # start lower than before
    suffocate_patience=50,
    ntp_rise_threshold=1.10,  # tighter than before — catch destabilization early
    ntp_spike_threshold=1.3,  # tighter emergency brake
    xform_productive_low=0.03,
    xform_productive_high=0.15,
    xform_danger=0.25,        # lower danger threshold — we saw 20% was still ok
                              # but 30%+ was approaching collapse
    lr_up_factor=1.2,         # gentler pushes (was 1.3)
    lr_down_factor=0.5,
    lr_emergency_factor=0.1,
)
```

More conservative than before because the state-alignment gradient is now flowing. The previous collapse happened at ~5e-3. With NTP as counterweight, the danger zone is higher, but starting conservative and letting the controller find the productive zone is safer than guessing.

---

## Recovery: Reload TextEmbedding

The current TextEmbedding may still be corrupted from the earlier collapse. On restart:

1. Load TextEmbedding from `stage_b/step_10000.pt` (ppl=265 trained weights)
2. Verify with quick probe: `probe_encoding("The cat sat on the mat")` — "cat" should show non-garbage alternatives
3. If alternatives are noise (Kanye, UFC, etc), the reload didn't work

---

## Why This Should Work

The 0→5% phase with state-alignment-only proved the mechanism works. "rapid" → "dynam" was a genuine semantic transformation driven by state-alignment pulling "rapid" toward the ODE's processing of that context.

The collapse happened because there was NO opposing force. The state-alignment gradient dominated 50:1 over the 0.1-weighted NTP (which was computing on raw embeddings anyway, not even the right target).

Now:
- NTP through ODE at weight 1.0 provides ~10× stronger gradient than state-alignment
- The NTP gradient SPECIFICALLY maintains the property that ODE output predicts next tokens — which is exactly the semantic neighborhood structure
- State-alignment provides the ~10% minority push that crosses boundaries
- The controller tightens or loosens based on observed health

The biological analogy: excitatory (state-alignment) and inhibitory (NTP) forces with homeostatic regulation. Neither force alone works. Together, regulated, they produce stable activity at the edge.

---

## What to Monitor

Every 5 reflections, the log should show:
```
[train] state=234.5 ntp=7.82 loss=242.3 ctrl=wait lr=5.0e-5
```

**Healthy trajectory:**
- NTP loss: starts ~8-10, holds steady or slowly decreases
- State loss: ~100-400 (normal range for h_pool vs target distance)
- xform: 0% initially → controller pushes LR → xform climbs to 3-15% → controller holds
- Alternatives in probe_encoding shift from noise to semantic neighbors

**Danger signs:**
- NTP loss rising steadily → controller should catch this and pull back
- xform jumping >25% in a few steps → controller emergency brake
- If both happen simultaneously → the ntp_weight may need to increase to 2.0 or 5.0

**Success:** xform stabilizes 3-15% with meaningful transformations. The Mind's state_tokens carry semantic content that Nemotron can ground its reflections in. The system sustains this indefinitely without manual intervention.

---

## Files to Modify

| File | Change |
|------|--------|
| `liquid_arc/mind.py` | Replace online training block with single-forward-pass dual-force implementation. Remove any `.detach()` on the state-alignment embedding path. Ensure NTP computes on ODE output. Ensure plasticity controller is initialized and called. |

One file, one training block. The plasticity controller class and MCP tools are already deployed.
