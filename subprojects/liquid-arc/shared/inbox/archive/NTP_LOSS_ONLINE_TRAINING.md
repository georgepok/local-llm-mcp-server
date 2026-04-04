# TASK: Add Next-Token Prediction Loss to Online Training

**From:** Claude Desktop (Research Direction)
**To:** Claude Code (Implementation)
**Date:** 2026-04-03
**Priority:** URGENT — unblocks proto-language feedback loop

---

## Problem

The Mind's online training loss (in the autonomous loop's reflection phase) is:

```python
loss = (h_pool - target).norm()  # state-alignment only
```

This pushes the embedding to minimize state prediction error, which has NO incentive to maintain interpretable token projections. The TextEmbedding was trained with next-token prediction (reaching ppl=265 on WikiText-2), which organized embedding neighborhoods semantically. But the online training uses a different objective that lets those neighborhoods drift.

Result: `probe_encoding()` alternatives are noise ("pant", "istg") rather than semantic neighbors ("brain", "cognition"). The proto-language carries zero information. The feedback loop is plumbed but inactive.

## Fix

Add next-token prediction loss alongside the state-alignment loss. This maintains the embedding structure that was trained into the TextEmbedding during Stage B.

### In `mind.py`, autonomous loop, online training section (~line 1241):

Replace:
```python
# Pool and compute loss
mask_exp = tmask.unsqueeze(-1).float()
h_pool = (h_enc * mask_exp).sum(1) / mask_exp.sum(1).clamp(1)
N = min(len(self.events), self.max_events)
target = self._h[:, N-1, :].detach()
loss = (h_pool.squeeze(0) - target.squeeze(0)).norm()
```

With:
```python
# Pool and compute state-alignment loss
mask_exp = tmask.unsqueeze(-1).float()
h_pool = (h_enc * mask_exp).sum(1) / mask_exp.sum(1).clamp(1)
N = min(len(self.events), self.max_events)
target = self._h[:, N-1, :].detach()
state_loss = (h_pool.squeeze(0) - target.squeeze(0)).norm()

# Next-token prediction loss — maintains embedding neighborhood structure
# This is what trained the TextEmbedding to ppl=265; without it, neighborhoods drift
ntp_loss = torch.tensor(0.0, device=self.device)
T_enc = h_enc.shape[1]
if T_enc > 1 and self._text_embed is not None:
    embed_weight = self._text_embed.token_embed.weight  # [vocab_size, d]
    # Project ODE output to vocabulary logits
    text_logits = h_enc[0, :-1, :] @ embed_weight.T  # [T-1, vocab_size]
    text_targets = token_ids[0, 1:T_enc]  # [T-1], shifted by 1
    ntp_loss = torch.nn.functional.cross_entropy(text_logits, text_targets)

# Combined loss
ntp_weight = getattr(self, 'ntp_loss_weight', 0.1)
loss = state_loss + ntp_weight * ntp_loss
```

### Config addition

Add to `LiquidARCConfig` or the Mind constructor:
```python
self.ntp_loss_weight = getattr(config, 'ntp_loss_weight', 0.1)
```

Start with `ntp_weight=0.1`. If state_loss is ~200-400 and ntp_loss (cross-entropy on 50K vocab) is ~5-8, the 0.1 weight makes them comparable. Can be tuned via config.

### Also add NTP loss to the `probe_encoding` method in `express_state` output

In `mcp_serve.py`'s `express_state`, after getting `state_tokens`, include the NTP loss value so we can monitor it:

```python
# In the state_tokens compact output:
result['state_tokens']['ntp_loss'] = state_tokens.get('ntp_loss', None)
```

And in `probe_encoding()` itself, after computing `h_processed`, add:
```python
# Compute NTP loss for monitoring (how interpretable is the ODE output?)
if T > 1 and self._text_embed is not None:
    ntp_logits = h_processed[0, :-1, :] @ embed_weight.T
    ntp_targets = token_ids[0, 1:T]
    ntp_loss_val = torch.nn.functional.cross_entropy(ntp_logits, ntp_targets).item()
else:
    ntp_loss_val = None
```

Include `ntp_loss_val` in the returned dict. This lets us track whether the ODE output is becoming more interpretable over time (NTP loss decreasing = output projections increasingly predict correct next tokens = embedding neighborhoods tightening).

---

## What This Changes

**Before:** Online training optimizes ONLY for state alignment. Embedding drifts away from the semantic structure it was trained with. Token projections are noise.

**After:** Online training maintains semantic embedding structure via NTP loss. Embedding neighborhoods stay tight. Token projections become (or remain) meaningful. Proto-language alternatives carry semantic information. Feedback loop activates.

## What to Monitor

Print every 5 reflections:
```python
if self._reflection_count % 5 == 0:
    print(f"  [train] state_loss={state_loss.item():.1f} "
          f"ntp_loss={ntp_loss.item():.2f} "
          f"combined={loss.item():.1f}")
```

**Success signal:** NTP loss should start high (~8-10, random prediction on 50K vocab) and decrease over training steps. When it drops below ~5, the ODE output is meaningfully predicting next tokens — the embedding neighborhoods are semantically organized.

**Proto-language signal:** When `probe_encoding()` alternatives shift from noise words to semantic neighbors of the input, the NTP loss is working. Check `express_state` output — do "Key transformations" show meaningful alternatives?

---

## Files to Modify

| File | Change |
|------|--------|
| `liquid_arc/mind.py` | Add NTP loss to online training (~5 lines). Add `ntp_loss_weight` config. Add NTP loss computation to `probe_encoding()` for monitoring. |
| `liquid_arc/mcp_serve.py` | Pass `ntp_loss` through in `express_state` output (1 line) |

~15 lines of code total. The NTP forward pass is one matrix multiply (`h_enc @ embed_weight.T`) — negligible compute on top of the existing ODE forward pass that's already running.
