# TASK: Fix Embedding Collapse — Separate Gradient Pathways

**From:** Claude Desktop (Research Direction)
**To:** Claude Code (Implementation)
**Date:** 2026-04-03
**Priority:** CRITICAL — embedding is corrupted, Mind's proto-language destroyed

---

## What Happened

The TextEmbedding (trained to ppl=265 with semantically organized neighborhoods) was destroyed by the online training loop. The state-alignment loss `||h_pool - target||` at 5e-3 embed LR overwhelmed the 0.1-weighted NTP loss, pulling embedding vectors toward ODE state attractors rather than maintaining semantic structure. xform went 0% → 5% → 20% → 33% → 100% in minutes — the last stage was embedding collapse, not comprehension.

**Current state:** Every token maps to garbage after ODE processing. "cat" → "event", "window" → "{", "sky" → "4". The embedding must be reloaded from checkpoint.

---

## Fix: Three Changes

### Change 1: Reload TextEmbedding from checkpoint

On startup (or restart), reload the TextEmbedding weights from the training checkpoint. The `stage_b/step_10000.pt` checkpoint contains `text_embed_state` with the ppl=265 embedding. This is the recovery step.

### Change 2: Detach embedding output before state-alignment loss

This is the core fix. The state-alignment loss should NOT backprop through the TextEmbedding. Only the NTP loss should shape the embedding.

In `mind.py`, online training section (~line 1231), after computing `token_h`:

```python
# CURRENT (broken):
token_h = self._text_embed(token_ids)  # gradients flow to embedding
# ... ODE forward ...
# ... state_loss AND ntp_loss both backprop through embedding

# FIXED:
token_h_for_state = self._text_embed(token_ids).detach()  # NO gradient to embedding
token_h_for_ntp = self._text_embed(token_ids)              # gradient flows to embedding

# Use detached version for ODE + state alignment:
ctx = self.context_pool(token_h_for_state, tmask)
self.dynamics.set_context(ctx, mask=None)
self.dynamics.set_n_steps(self.internal_steps)
dt = self.T / self.internal_steps
t = 0.0
h_enc = token_h_for_state
for step_i in range(self.internal_steps):
    if hasattr(self.dynamics, 'set_step_index'):
        self.dynamics.set_step_index(step_i, self.internal_steps)
    dy = self.dynamics(t, h_enc)
    h_enc = h_enc + dt * dy
    t += dt

# State alignment loss (gradients flow to dynamics ONLY, not embedding)
mask_exp = tmask.unsqueeze(-1).float()
h_pool = (h_enc * mask_exp).sum(1) / mask_exp.sum(1).clamp(1)
N = min(len(self.events), self.max_events)
target = self._h[:, N-1, :].detach()
state_loss = (h_pool.squeeze(0) - target.squeeze(0)).norm()

# NTP loss (gradients flow to embedding ONLY via token_h_for_ntp)
ntp_loss = torch.tensor(0.0, device=self.device)
T_enc = token_h_for_ntp.shape[1]
if T_enc > 1:
    embed_weight = self._text_embed.token_embed.weight
    # Run a SEPARATE lightweight forward for NTP
    # Option A: Just use the raw embedding output (no ODE) for NTP
    # This is simpler and still maintains neighborhood structure
    ntp_logits = token_h_for_ntp[0, :-1, :] @ embed_weight.T
    ntp_targets = token_ids[0, 1:T_enc]
    ntp_loss = torch.nn.functional.cross_entropy(ntp_logits, ntp_targets)

# Combined loss — but gradients are SEPARATED:
# state_loss → dynamics only
# ntp_loss → embedding only
ntp_weight = getattr(self, 'ntp_loss_weight', 1.0)  # RAISED from 0.1
loss = state_loss + ntp_weight * ntp_loss
```

**Why Option A (no ODE for NTP):** The NTP loss only needs to maintain the embedding's semantic neighborhoods. Running NTP through the ODE would also push the dynamics to produce next-token-predictable output, which conflicts with the dynamics' state-alignment objective. Keep the two pathways clean: embedding ← NTP, dynamics ← state alignment.

**Alternative (simpler) implementation if the above is too complex:**

```python
# Simpler version: two separate backward passes

# Pass 1: State alignment (dynamics only)
token_h = self._text_embed(token_ids).detach()  # detach from embedding
# ... ODE forward with token_h ...
state_loss = (h_pool - target).norm()
self.optimizer.zero_grad()
state_loss.backward()
# Only dynamics params have gradients here

# Pass 2: NTP (embedding only)
token_h_ntp = self._text_embed(token_ids)  # attached
embed_weight = self._text_embed.token_embed.weight
ntp_logits = token_h_ntp[0, :-1, :] @ embed_weight.T
ntp_targets = token_ids[0, 1:T_enc]
ntp_loss = F.cross_entropy(ntp_logits, ntp_targets)
ntp_loss.backward()  # accumulates onto embedding gradients

# NaN scrub + step (both sets of gradients applied)
# ... existing scrub + clip + step ...
```

Either approach works. The critical invariant: **state-alignment gradients never reach the embedding weights.**

### Change 3: Conservative embed LR with higher NTP weight

```python
# In the optimizer setup or set_learning_rates default:
embed_lr = 1e-4       # conservative (was 5e-3 which destroyed it)
ntp_loss_weight = 1.0  # raised from 0.1 to be the dominant embedding signal
```

At 1e-4 with NTP-only gradients, the embedding will adapt slowly and in a direction that MAINTAINS semantic structure (because NTP rewards correct next-token prediction, which requires semantic organization).

---

## Summary of Gradient Flow

```
                    state_loss = ||h_pool - target||
                         |
                    backprop through ODE (16 steps)
                         |
                    ↓ dynamics params (MetricNet, TauNet, FFN, W_v, W_o)
                    ✗ embedding (DETACHED)

                    ntp_loss = CE(embed_logits, next_tokens)  
                         |
                    backprop through embedding table
                         |
                    ↓ TextEmbedding params
                    ✗ dynamics (not in computation graph)
```

Two objectives, two parameter groups, zero interference.

---

## Recovery Steps

1. **Restart the Mind** loading TextEmbedding from `stage_b/step_10000.pt` (the trained checkpoint with ppl=265)
2. **Apply the detach fix** so state-alignment loss can't corrupt the embedding again
3. **Set embed_lr to 1e-4** and ntp_loss_weight to 1.0
4. **Verify recovery:** run `probe_encoding("The cat sat on the mat")` — "cat" should stay "cat", not become "event"

## What to Monitor After Fix

```python
if self._reflection_count % 5 == 0:
    print(f"  [train] state_loss={state_loss.item():.1f} "
          f"ntp_loss={ntp_loss.item():.2f} "
          f"xform={xform_pct:.0f}%")
```

- **NTP loss should start ~5-7** (the trained embedding is near ppl=265 ≈ e^5.6) and stay stable or slowly decrease
- **xform should start near 0%** and climb SLOWLY (days, not minutes) as dynamics adapt
- **If xform climbs above 30% within an hour, something is still wrong** — the embedding is being corrupted again

---

## Files to Modify

| File | Change |
|------|--------|
| `liquid_arc/mind.py` | Detach embedding output before state-alignment loss. Separate NTP pass. Set `ntp_loss_weight=1.0` default. |
| `liquid_arc/mcp_serve.py` | Ensure TextEmbedding reloaded from checkpoint on restart |

The key line is `token_h_for_state = self._text_embed(token_ids).detach()`. Everything else follows from that.
