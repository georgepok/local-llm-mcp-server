# TASK: Sphere-Constrained Embedding — Intrinsic Self-Stabilization

**From:** Claude Desktop (Research Direction)
**To:** Claude Code (Implementation)
**Date:** 2026-04-04
**Priority:** CRITICAL — fixes root cause of repeated embedding collapse

**Supersedes all previous online training specs** (all archived)

---

## The Root Cause

The embedding has collapsed TWICE via the same mechanism: a few tokens accumulate larger L2 norms than their neighbors, winning every dot-product projection, becoming universal attractors that swallow the vocabulary. External controls (plasticity controller, NTP counterweight, LR tuning, detach) all failed because they treat the symptom rather than the cause.

The cause: **embedding vectors can grow without bound.** State-alignment gradients increase the norm of tokens that align with the ODE state. Once a token's norm exceeds its neighbors, it wins more projections, receives more gradient reinforcement, grows further. Positive feedback with no intrinsic brake.

## The Fix: One Line

After every optimizer step, L2-normalize each row of the embedding table to unit norm:

```python
with torch.no_grad():
    self._text_embed.token_embed.weight.div_(
        self._text_embed.token_embed.weight.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    )
```

This is the embedding equivalent of the CV floor for geometry. The CV floor prevents geometric collapse by making flat metrics costly. The sphere constraint prevents embedding collapse by making norm growth impossible. Both are intrinsic — they run automatically without monitoring or external control.

## Why This Preserves xform Development

The genuine 5-20% xform we observed ("rapid" → "dynam") happened because the ODE pushed token representations in directions closer to different token embeddings. That's angular movement — which direction is the ODE output pointing.

On a unit sphere, ALL competition between tokens is angular. The projection `logits = h_processed @ embed_weight.T` with unit-norm embeddings becomes cosine similarity (scaled by the ODE output's norm). xform happens when the ODE output's direction is closer to a different token than the input token. State-alignment rotates embeddings on the sphere. NTP rotates them back. Neither can cause collapse because the escape route (norm growth) is structurally closed.

The collapse tokens ("hippocamp", "CBC") won through MAGNITUDE, not direction. On the sphere, they have no magnitude advantage. They can only win a position's projection if they're angularly closest — which requires being actually relevant to the ODE's processing, not just having a big vector.

## Full Online Training Block

Replace the entire online training section in `mind.py` (~line 1201). This is the same dual-force approach (state-alignment + NTP through ODE) with the sphere constraint added:

```python
# Online training: dual-force with sphere-constrained embedding
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
                    raise ValueError("Requires Path C TextEmbedding")
                
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
                
                # ═══ SINGLE FORWARD PASS ═══
                # Embedding attached — both forces flow through
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
                
                # ═══ FORCE 1: State alignment ═══
                mask_exp = tmask.unsqueeze(-1).float()
                h_pool = (h_enc * mask_exp).sum(1) / mask_exp.sum(1).clamp(1)
                N = min(len(self.events), self.max_events)
                target = self._h[:, N-1, :].detach()
                state_loss = (h_pool.squeeze(0) - target.squeeze(0)).norm()
                
                # ═══ FORCE 2: NTP through ODE ═══
                ntp_loss = torch.tensor(0.0, device=self.device)
                if T_enc > 1:
                    embed_weight = self._text_embed.token_embed.weight
                    ntp_logits = h_enc[0, :-1, :] @ embed_weight.T
                    ntp_targets = token_ids[0, 1:T_enc]
                    ntp_loss = torch.nn.functional.cross_entropy(
                        ntp_logits, ntp_targets)
                
                # ═══ COMBINED LOSS ═══
                ntp_weight = getattr(self, 'ntp_loss_weight', 1.0)
                loss = state_loss + ntp_weight * ntp_loss
                
                # ═══ BACKWARD + STEP ═══
                self.optimizer.zero_grad()
                loss.backward()
                
                # NaN scrub
                for p in self.optimizer.param_groups:
                    for param in p['params']:
                        if param.grad is not None and param.grad.isnan().any():
                            param.grad.nan_to_num_(nan=0.0)
                torch.nn.utils.clip_grad_norm_(
                    [p for g in self.optimizer.param_groups
                     for p in g['params']], 1.0)
                self.optimizer.step()
                
                # ═══ SPHERE CONSTRAINT — the intrinsic stabilizer ═══
                # After every step, project embedding back to unit sphere.
                # This is the ONLY mechanism needed to prevent collapse.
                # No controller, no monitoring, no external intervention.
                with torch.no_grad():
                    w = self._text_embed.token_embed.weight
                    w.div_(w.norm(dim=-1, keepdim=True).clamp(min=1e-8))
                
                # ═══ PLASTICITY CONTROLLER (secondary safety net) ═══
                if hasattr(self, '_plasticity_ctrl') and self._plasticity_ctrl is not None:
                    current_xform = 0.0
                    if state_tokens is not None:
                        current_xform = state_tokens.get('transform_ratio', 0.0)
                    ctrl_result = self._plasticity_ctrl.update(
                        ntp_loss=ntp_loss.item(),
                        xform=current_xform)
                    self.optimizer.param_groups[0]['lr'] = ctrl_result['embed_lr']
                
                # ═══ LOGGING ═══
                if self._reflection_count % 5 == 0:
                    embed_norms = self._text_embed.token_embed.weight.norm(dim=-1)
                    ctrl_info = ""
                    if hasattr(self, '_plasticity_ctrl') and self._plasticity_ctrl is not None:
                        ctrl_info = (f" ctrl={self._plasticity_ctrl.last_action}"
                                    f" lr={self._plasticity_ctrl.current_embed_lr:.1e}")
                    print(f"  [train] state={state_loss.item():.1f}"
                          f" ntp={ntp_loss.item():.2f}"
                          f" embed_norm={embed_norms.mean().item():.4f}"
                          f" (should be ~1.0){ctrl_info}")
    except Exception as e:
        if self._reflection_count % 10 == 0:
            print(f"  [train] error: {e}")
```

## Recovery: Reload + Normalize

The current embedding is collapsed. On restart:

1. Reload TextEmbedding from `stage_b/step_10000.pt`
2. IMMEDIATELY normalize the loaded weights:
```python
with torch.no_grad():
    w = self._text_embed.token_embed.weight
    w.div_(w.norm(dim=-1, keepdim=True).clamp(min=1e-8))
```
3. Verify: `probe_encoding("The cat sat on the mat")` — "cat" should be "cat", not "hippocamp"

Also normalize the embedding at initialization time (in `__init__` or wherever TextEmbedding is loaded) so the system starts on the sphere.

## Plasticity Controller Parameters

The controller is now a secondary safety net, not the primary defense. The sphere constraint does the heavy lifting. But keep the controller for LR management:

```python
PlasticityController(
    embed_lr_min=1e-5,
    embed_lr_max=5e-4,
    embed_lr_init=1e-4,
    suffocate_patience=50,
    ntp_rise_threshold=1.15,
    ntp_spike_threshold=1.5,
    xform_productive_low=0.03,
    xform_productive_high=0.15,
    xform_danger=0.30,
    lr_up_factor=1.2,
    lr_down_factor=0.5,
    lr_emergency_factor=0.1,
)
```

## Why This Is Different From All Previous Specs

Every previous spec tried to CONTROL the balance between forces externally:
- Detach: amputate the push force → suffocation
- NTP counterweight: add opposing force at fixed ratio → wrong ratio collapses
- Plasticity controller: monitor and adjust LR → sensor compromised by collapse
- Dual-force with NTP-through-ODE: balance forces via architecture → still collapses slowly

This spec doesn't control the balance. It **constrains the surface** on which learning happens. On a unit hypersphere:
- State-alignment gradients produce rotation (productive: crosses boundaries)
- NTP gradients produce rotation (productive: maintains semantic structure)  
- Norm growth is impossible (collapse mechanism eliminated)

The balance between forces EMERGES from the constraint, the same way a pendulum's balance emerges from the constraint of the rod. No controller needed to prevent the pendulum from flying off — the rod does it structurally. The sphere does it structurally for the embedding.

This is how nature works. Synaptic scaling doesn't monitor and adjust — it's a molecular constraint that normalizes synaptic weight automatically. The sphere constraint is the computational analogue: a structural property that makes collapse geometrically impossible while allowing all productive learning to proceed.

## What to Monitor

```
[train] state=234.5 ntp=7.82 embed_norm=1.0000 ctrl=wait lr=1.0e-4
```

- **embed_norm should always be ~1.0000** — if it drifts, the normalization isn't applying
- **NTP loss trajectory** — should start high (8-12 for ODE output), gradually decrease
- **xform trajectory** — should climb slowly (0% → 1-5% over hours) as angular reorganization accumulates
- **Vocabulary diversity in alternatives** — the real success signal is semantic neighbors appearing ("cat" → "dog, kitten") rather than noise ("cat" → "Karl, UFC") or collapse ("cat" → "hippocamp")

## Files to Modify

| File | Change |
|------|--------|
| `liquid_arc/mind.py` | Replace online training block. Add sphere normalization after optimizer step AND at embedding load time. |

One file. The core change is two lines: `w.div_(w.norm(...).clamp(...))` after `optimizer.step()` and at initialization.
