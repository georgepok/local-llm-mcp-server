# TASK: NTP Through ODE — Fix the Proto-Language Training Signal

**From:** Claude Desktop (Research Direction)
**To:** Claude Code (Implementation)
**Date:** 2026-04-03
**Priority:** CRITICAL — fixes the fundamental disconnect preventing xform > 0%

**Supersedes:** `ADAPTIVE_PLASTICITY_CONTROLLER.md`, `FIX_EMBEDDING_COLLAPSE.md`, `NTP_LOSS_ONLINE_TRAINING.md` (all archived — this spec incorporates and corrects them)

---

## The Problem

After all the work — fluid metric architecture, NTP loss, detach fix, adaptive plasticity controller — xform is stuck at 0%. The controller pushed embed_lr to the 1e-3 ceiling. Nothing happened. The reason is a structural disconnect in how the NTP loss is computed.

### Current NTP (broken):

```
token_h = text_embed(token_ids)              ← raw embedding, BEFORE ODE
ntp_logits = token_h[:-1] @ embed_weight.T   ← predicting from raw embedding
ntp_loss = CE(ntp_logits, next_tokens)        ← teaches embedding for pre-ODE prediction
```

### xform measurement:

```
h_processed = ODE(token_h, 16 steps)          ← AFTER ODE  
logits = h_processed @ embed_weight.T          ← projecting ODE output
xform = (top1_output != input_token)           ← measures post-ODE identity change
```

**These two computations are disconnected.** The NTP loss organizes the embedding for raw next-token prediction. The xform measures what happens after the ODE transforms those embeddings. No gradient signal ever tells the embedding: "organize yourself so that the ODE's OUTPUT lands on meaningful tokens."

The NTP loss at any LR — 1e-4, 1e-3, 1e-2 — cannot produce xform because it operates on the wrong representation (pre-ODE instead of post-ODE).

---

## The Fix

**Compute NTP on the ODE output, not the raw embedding.**

```
token_h = text_embed(token_ids)               ← attached to embedding graph
h_enc = ODE(token_h, 16 steps)                ← ODE forward (dynamics params in graph)
ntp_logits = h_enc[:-1] @ embed_weight.T      ← predicting from ODE output
ntp_loss = CE(ntp_logits, next_tokens)         ← teaches embedding for post-ODE prediction
ntp_loss.backward()                            ← gradient flows through ODE to embedding
```

This tells the embedding: "position yourself so that AFTER 16 ODE integration steps transform you, the result predicts the next token." The embedding learns to sit where the ODE's natural push directions point toward semantically correct neighbors.

### Gradient flow:

```
ntp_loss
  ↓ backprop through embed_weight.T matmul
  ↓ backprop through 16 ODE Euler steps
  ↓ reaches text_embed parameters
  ↓ ALSO reaches dynamics parameters (MetricNet, TauNet, etc.)
```

Both embedding AND dynamics get NTP gradients. But:
- Embedding LR: 1e-4 → meaningful update
- Dynamics LR: 1e-6 → negligible update (100× smaller)

The LR ratio provides soft separation. We don't need hard `.detach()` on the NTP path.

### State-alignment path (unchanged, still detached):

```
token_h_detached = text_embed(token_ids).detach()   ← NO gradient to embedding
h_enc_state = ODE(token_h_detached, 16 steps)
state_loss = ||pool(h_enc_state) - target||
state_loss.backward()                                ← dynamics only
```

---

## Implementation

### In `mind.py`, the online training section (~line 1201):

Replace the current training block with:

```python
# Online embedding training during reflection
if self.optimizer is not None and self.use_ode_encoder:
    try:
        recent_content = None
        for e in reversed(self.events):
            if e.get('type') not in [6, 7]:
                recent_content = e.get('content_preview', '')
                break
        if recent_content and len(recent_content) > 10:
            with self._gpu_lock:
                if self.use_trained_text_embed and self._text_embed is not None:
                    toks = self._text_tokenizer.encode(
                        recent_content, add_special_tokens=False,
                        truncation=True, max_length=512)
                    if not toks:
                        raise ValueError("empty tokens")
                    token_ids = torch.tensor([toks], dtype=torch.long,
                                            device=self.device)
                    T_enc = len(toks)
                else:
                    raise ValueError("NTP-through-ODE requires Path C TextEmbedding")

                # ═══ PATH 1: State alignment (dynamics only) ═══
                # Embedding detached — no gradient to text_embed
                token_h_state = self._text_embed(token_ids).detach()
                tmask = torch.ones(1, T_enc, dtype=torch.bool, device=self.device)
                ctx = self.context_pool(token_h_state, tmask)
                self.dynamics.set_context(ctx, mask=None)
                self.dynamics.set_n_steps(self.internal_steps)
                
                dt = self.T / self.internal_steps
                t = 0.0
                h_state = token_h_state
                for step_i in range(self.internal_steps):
                    if hasattr(self.dynamics, 'set_step_index'):
                        self.dynamics.set_step_index(step_i, self.internal_steps)
                    dy = self.dynamics(t, h_state)
                    h_state = h_state + dt * dy
                    t += dt
                
                mask_exp = tmask.unsqueeze(-1).float()
                h_pool = (h_state * mask_exp).sum(1) / mask_exp.sum(1).clamp(1)
                N = min(len(self.events), self.max_events)
                target = self._h[:, N-1, :].detach()
                state_loss = (h_pool.squeeze(0) - target.squeeze(0)).norm()

                # ═══ PATH 2: NTP through ODE (embedding + slight dynamics) ═══
                # Embedding ATTACHED — gradient flows through ODE to text_embed
                token_h_ntp = self._text_embed(token_ids)
                # Reuse same context (detached, so no double-gradient issue)
                self.dynamics.set_context(ctx.detach(), mask=None)
                self.dynamics.set_n_steps(self.internal_steps)
                
                t = 0.0
                h_ntp = token_h_ntp
                for step_i in range(self.internal_steps):
                    if hasattr(self.dynamics, 'set_step_index'):
                        self.dynamics.set_step_index(step_i, self.internal_steps)
                    dy = self.dynamics(t, h_ntp)
                    h_ntp = h_ntp + dt * dy
                    t += dt
                
                # NTP on ODE output
                ntp_loss = torch.tensor(0.0, device=self.device)
                if T_enc > 1:
                    embed_weight = self._text_embed.token_embed.weight
                    ntp_logits = h_ntp[0, :-1, :] @ embed_weight.T
                    ntp_targets = token_ids[0, 1:T_enc]
                    ntp_loss = torch.nn.functional.cross_entropy(
                        ntp_logits, ntp_targets)

                # ═══ Combined loss ═══
                ntp_weight = getattr(self, 'ntp_loss_weight', 1.0)
                loss = state_loss + ntp_weight * ntp_loss

                # ═══ Backward + step ═══
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

                # ═══ Plasticity controller ═══
                if hasattr(self, '_plasticity_ctrl') and self._plasticity_ctrl is not None:
                    current_xform = 0.0
                    if state_tokens is not None:
                        current_xform = state_tokens.get('transform_ratio', 0.0)
                    ctrl_result = self._plasticity_ctrl.update(
                        ntp_loss=ntp_loss.item(),
                        xform=current_xform)
                    self.optimizer.param_groups[0]['lr'] = ctrl_result['embed_lr']

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

### Computational cost

Two ODE forward passes per training step instead of one:
- Path 1 (state): ODE with detached embedding → dynamics gradients
- Path 2 (NTP): ODE with attached embedding → embedding gradients (+ negligible dynamics gradients)

At 16 Euler steps with fluid metric, each pass is ~50-100ms. Two passes: ~100-200ms total. The autonomous loop runs one training step every few seconds. The overhead is <10%.

### Recovery: Reload TextEmbedding

The current TextEmbedding is corrupted from the 5e-3 collapse. On restart, reload from the training checkpoint:

```python
# In mind.py __init__ or load_checkpoint:
if 'text_embed_state' in ckpt_full:
    self._text_embed.load_state_dict(ckpt_full['text_embed_state'])
    print("  TextEmbedding restored from training checkpoint")
```

Ensure this happens BEFORE online training begins.

---

## Plasticity Controller Update

The controller from the previous spec still applies, but its interpretation changes:

- **NTP loss baseline will be HIGHER** than before because NTP is now computed on ODE output (which is further from the "correct" next token than raw embeddings are). Starting NTP loss might be ~8-12 instead of ~5-7.
- **NTP loss DECREASING** now means the embedding is learning to position itself so the ODE output becomes more predictable — which directly correlates with meaningful xform.
- **xform climbing** means the ODE output is actually crossing token boundaries in semantically meaningful ways, because the NTP loss trained the embedding for exactly this.

Controller parameters may need retuning:
```python
PlasticityController(
    embed_lr_min=1e-5,
    embed_lr_max=5e-4,     # lower ceiling than before (ODE backward amplifies)
    embed_lr_init=1e-4,
    suffocate_patience=50,
    ntp_rise_threshold=1.15,
    ntp_spike_threshold=1.5,
)
```

Lower `embed_lr_max` to 5e-4 because gradients flowing through 16 ODE steps are amplified. The effective gradient magnitude at the embedding is larger than the raw NTP gradient at the same LR.

---

## What Changes

**Before (broken):**
```
NTP loss → teaches embedding for raw next-token prediction
xform → measures post-ODE token identity change
→ Zero connection between training signal and measurement
→ xform stuck at 0% regardless of LR
```

**After (fixed):**
```
NTP loss → teaches embedding for post-ODE next-token prediction
xform → measures post-ODE token identity change
→ Training signal directly optimizes the measured quantity
→ xform should climb as NTP loss decreases
```

The NTP loss and xform now look at the SAME mathematical object (ODE output projected to vocabulary). When NTP loss decreases, the ODE output becomes more predictable in token space, which means it's moving tokens toward correct neighbors, which means xform increases. The training signal and the measurement are aligned.

---

## Expected Behavior After Fix

**Early (first ~100 training steps):**
- NTP loss starts high (~8-12, ODE output is far from predicting next tokens)
- xform still 0% (but NTP is now training the right thing)
- Controller at init LR (1e-4), building baseline

**Adaptation (~100-500 steps):**
- NTP loss decreasing (embedding learning where to sit for ODE output to predict)
- xform begins to appear: 1-3% (the first meaningful boundary crossings)
- Alternatives becoming less random (NTP rewards semantic neighbors)

**Productive zone (~500+ steps):**
- NTP loss stabilized at a lower level
- xform 5-15% with semantically meaningful transformations
- Controller in "hold" mode
- Proto-language carrying genuine information to Nemotron

**Key validation:** Probe `"The cat sat on the mat"` periodically. When "cat" alternatives shift from "UFC, Karl" (noise) to "dog, kitten, animal" (semantic), the NTP-through-ODE is working. That's the signal that the embedding has learned to position "cat" where the ODE's natural dynamics push it toward related concepts.

---

## Files to Modify

| File | Change |
|------|--------|
| `liquid_arc/mind.py` | Replace online training block with two-path implementation (state detached, NTP through ODE). Ensure TextEmbedding reloaded from checkpoint on restart. Lower plasticity controller `embed_lr_max` to 5e-4. |

One file, one training block replacement. The plasticity controller class and MCP tool are already deployed and don't need changes beyond the `embed_lr_max` parameter.

---

## Success Criteria

- **Minimum:** NTP loss computed on ODE output (verify by checking that NTP loss > 7 initially — if it's ~5 then it's still computing on raw embeddings). Training runs without NaN.
- **Good:** NTP loss decreases over 200+ training steps. xform > 0% within first hour.
- **Strong:** xform stabilizes 5-15% with semantically meaningful alternatives. "cat" → "dog/kitten", "rapid" → "fast/quick", not "cat" → "UFC".
- **Headline:** The Mind's proto-language carries genuine semantic information. Nemotron reflections reference the Mind's actual token transformations. The feedback loop is active.
