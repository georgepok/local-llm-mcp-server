# TASK: Adaptive Plasticity Controller — Self-Regulating Online Learning

**From:** Claude Desktop (Research Direction)
**To:** Claude Code (Implementation)
**Date:** 2026-04-03
**Priority:** HIGH — replaces manual LR tuning with homeostatic self-regulation

**Supersedes:** `FIX_EMBEDDING_COLLAPSE.md` (archived — this spec incorporates the detach fix and adds the adaptive controller)

**Prerequisites:**
- The detach fix (state-alignment gradients don't reach embedding) MUST be in place. If not already applied, apply it as part of this spec.
- NTP loss computation in the online training step
- `probe_encoding()` with `transform_ratio` measurement
- `set_learning_rates` MCP tool

---

## Problem

The Mind oscillates between two failure modes:

1. **Suffocation:** embed_lr too low (1e-4), dynamics LR too low (1e-6). The system runs for thousands of events with xform=0%. The embedding neighborhoods don't tighten. The dynamics don't adapt. Nothing happens. The Mind is stable but inert.

2. **Collapse:** embed_lr too high (5e-3). The state-alignment loss (even with NTP as counterweight at 0.1) overwhelms the embedding structure. xform rockets from 0% → 100% in minutes as neighborhoods shatter. The Mind is active but self-destructing.

Between these extremes, there IS a productive zone. We observed it briefly: xform climbing 0% → 5% → 20% with semantically meaningful transformations ("rapid" → "dynam"). The system was learning. Then it blew past the zone because nothing pulled back.

The solution is not finding the right fixed LR. It's a **homeostatic controller** that maintains the system at the productive edge — the same principle as the CV floor/ceiling in training and the adaptive criticality controller. The system finds its own resonance rather than depending on external tuning.

---

## Design: Adaptive Plasticity Controller

### Core Signals

| Signal | Source | Meaning |
|--------|--------|---------|
| `ntp_loss` | NTP forward pass on recent text | Embedding health. Stable ≈ good. Rising = neighborhoods destabilizing. |
| `xform` | `probe_encoding()` on recent event | Boundary crossing rate. 0% = suffocating. 3-15% = productive. >30% = likely collapsing. |
| `ntp_loss_ma` | Exponential moving average of `ntp_loss` | Smoothed baseline to detect trends vs noise. |

### Controller Logic

```python
class PlasticityController:
    """Adaptive learning rate controller for embedding online learning.
    
    Maintains the system at the productive edge of stability-plasticity:
    - Raises embed_lr when suffocating (xform=0%, NTP stable)
    - Lowers embed_lr when running loose (NTP rising, xform climbing fast)
    - Holds when productive (xform 3-15%, NTP stable)
    
    Analogous to the adaptive criticality controller that maintained CV
    at the critical zone during ARC training.
    """
    
    def __init__(self,
                 # LR bounds
                 embed_lr_min: float = 1e-5,
                 embed_lr_max: float = 1e-3,
                 embed_lr_init: float = 1e-4,
                 # NTP loss tracking
                 ntp_ema_alpha: float = 0.05,   # smoothing factor for EMA
                 ntp_rise_threshold: float = 1.15,  # 15% above EMA = rising
                 ntp_spike_threshold: float = 1.5,   # 50% above EMA = emergency
                 # xform targets
                 xform_suffocate: float = 0.0,   # below this = suffocating
                 xform_productive_low: float = 0.03,  # 3%
                 xform_productive_high: float = 0.15,  # 15%
                 xform_danger: float = 0.30,      # above this = likely collapsing
                 # LR adjustment rates
                 lr_up_factor: float = 1.3,       # multiply when pushing harder
                 lr_down_factor: float = 0.5,     # multiply when pulling back
                 lr_emergency_factor: float = 0.1, # multiply on emergency brake
                 # Patience
                 suffocate_patience: int = 50,    # cycles at xform=0% before pushing
                 ):
        
        self.embed_lr_min = embed_lr_min
        self.embed_lr_max = embed_lr_max
        self.current_embed_lr = embed_lr_init
        
        self.ntp_ema_alpha = ntp_ema_alpha
        self.ntp_rise_threshold = ntp_rise_threshold
        self.ntp_spike_threshold = ntp_spike_threshold
        
        self.xform_suffocate = xform_suffocate
        self.xform_productive_low = xform_productive_low
        self.xform_productive_high = xform_productive_high
        self.xform_danger = xform_danger
        
        self.lr_up_factor = lr_up_factor
        self.lr_down_factor = lr_down_factor
        self.lr_emergency_factor = lr_emergency_factor
        self.suffocate_patience = suffocate_patience
        
        # State
        self.ntp_loss_ema = None         # exponential moving average
        self.zero_xform_streak = 0       # consecutive cycles with xform=0%
        self.step_count = 0
        self.last_action = 'init'
        
        # History for logging
        self.history = []  # last 100 (step, ntp, xform, lr, action)
    
    def update(self, ntp_loss: float, xform: float) -> dict:
        """Called after each online training step.
        
        Args:
            ntp_loss: Current NTP loss value
            xform: Current transform_ratio from probe_encoding
            
        Returns:
            dict with 'embed_lr', 'action', 'reason'
        """
        self.step_count += 1
        
        # Initialize EMA on first call
        if self.ntp_loss_ema is None:
            self.ntp_loss_ema = ntp_loss
            return self._result('init', 'First step — establishing baseline')
        
        # Update EMA
        self.ntp_loss_ema = (self.ntp_ema_alpha * ntp_loss + 
                            (1 - self.ntp_ema_alpha) * self.ntp_loss_ema)
        
        # Track suffocation streak
        if xform <= self.xform_suffocate:
            self.zero_xform_streak += 1
        else:
            self.zero_xform_streak = 0
        
        # ─── EMERGENCY BRAKE ───
        # NTP loss spiking = embedding in acute danger
        if ntp_loss > self.ntp_spike_threshold * self.ntp_loss_ema:
            self.current_embed_lr = max(
                self.embed_lr_min,
                self.current_embed_lr * self.lr_emergency_factor)
            return self._result('emergency_brake',
                f'NTP spike: {ntp_loss:.2f} vs EMA {self.ntp_loss_ema:.2f}')
        
        # ─── PULL BACK ───
        # NTP rising + xform high = approaching collapse
        ntp_rising = ntp_loss > self.ntp_rise_threshold * self.ntp_loss_ema
        xform_high = xform > self.xform_danger
        
        if ntp_rising or xform_high:
            self.current_embed_lr = max(
                self.embed_lr_min,
                self.current_embed_lr * self.lr_down_factor)
            reason = []
            if ntp_rising:
                reason.append(f'NTP rising: {ntp_loss:.2f} vs EMA {self.ntp_loss_ema:.2f}')
            if xform_high:
                reason.append(f'xform high: {xform:.1%}')
            return self._result('pull_back', '; '.join(reason))
        
        # ─── PRODUCTIVE ZONE ───
        # xform in good range, NTP stable = hold steady
        if (self.xform_productive_low <= xform <= self.xform_productive_high
                and not ntp_rising):
            return self._result('hold',
                f'Productive: xform={xform:.1%}, NTP={ntp_loss:.2f}')
        
        # ─── PUSH HARDER ───
        # Suffocating: xform=0% for too long, NTP stable
        if (self.zero_xform_streak >= self.suffocate_patience
                and not ntp_rising):
            self.current_embed_lr = min(
                self.embed_lr_max,
                self.current_embed_lr * self.lr_up_factor)
            self.zero_xform_streak = 0  # reset patience counter
            return self._result('push',
                f'Suffocating for {self.suffocate_patience} cycles, '
                f'NTP stable at {ntp_loss:.2f}')
        
        # ─── WAIT ───
        # Not yet at patience threshold, not in productive zone
        return self._result('wait',
            f'xform={xform:.1%}, streak={self.zero_xform_streak}/'
            f'{self.suffocate_patience}')
    
    def _result(self, action: str, reason: str) -> dict:
        self.last_action = action
        entry = {
            'step': self.step_count,
            'embed_lr': self.current_embed_lr,
            'ntp_ema': self.ntp_loss_ema,
            'action': action,
            'reason': reason,
        }
        self.history.append(entry)
        if len(self.history) > 100:
            self.history = self.history[-100:]
        return entry
    
    def get_status(self) -> dict:
        return {
            'current_embed_lr': self.current_embed_lr,
            'ntp_loss_ema': self.ntp_loss_ema,
            'zero_xform_streak': self.zero_xform_streak,
            'step_count': self.step_count,
            'last_action': self.last_action,
            'lr_bounds': [self.embed_lr_min, self.embed_lr_max],
            'recent_history': self.history[-10:],
        }
```

---

## Integration into the Autonomous Loop

In `mind.py`, the online training section (~line 1201). After computing both losses and BEFORE the optimizer step:

```python
# After computing state_loss and ntp_loss:

# ─── Adaptive Plasticity Controller ───
if hasattr(self, '_plasticity_ctrl') and self._plasticity_ctrl is not None:
    # Get current xform from the most recent probe_encoding
    # (already computed earlier in the reflection phase as state_tokens)
    current_xform = 0.0
    if state_tokens is not None:
        current_xform = state_tokens.get('transform_ratio', 0.0)
    
    ctrl_result = self._plasticity_ctrl.update(
        ntp_loss=ntp_loss.item(),
        xform=current_xform,
    )
    
    # Apply the controller's LR decision to the optimizer
    new_lr = ctrl_result['embed_lr']
    self.optimizer.param_groups[0]['lr'] = new_lr  # group 0 = embed
    
    # Log periodically
    if self._reflection_count % 10 == 0:
        print(f"  [plasticity] lr={new_lr:.1e} "
              f"action={ctrl_result['action']} "
              f"ntp_ema={self._plasticity_ctrl.ntp_loss_ema:.2f} "
              f"xform_streak={self._plasticity_ctrl.zero_xform_streak}")
```

### Initialization

In `mind.py` `__init__`, after optimizer setup:

```python
# Adaptive plasticity controller
self._plasticity_ctrl = PlasticityController(
    embed_lr_min=1e-5,
    embed_lr_max=1e-3,      # never exceed what caused the collapse
    embed_lr_init=1e-4,      # conservative start
    suffocate_patience=50,   # ~50 training steps before pushing
)
```

---

## Gradient Pathway (Must Be Preserved)

The detach fix from the previous spec MUST be in place for this controller to work safely:

```
state_loss gradients → dynamics ONLY (embedding detached)
ntp_loss gradients   → embedding ONLY (dynamics not in graph)
```

The controller adjusts the LR for the NTP pathway only. Even at embed_lr_max (1e-3), the NTP loss can only ORGANIZE the embedding semantically — it cannot collapse it the way state-alignment did. The controller's emergency brake is a safety net in case NTP loss alone can still destabilize at high LR (unlikely but cautious).

If the detach fix is NOT yet applied, apply it first:

```python
# In online training:
token_h_for_state = self._text_embed(token_ids).detach()  # state path
token_h_for_ntp = self._text_embed(token_ids)              # NTP path

# State loss computed from token_h_for_state (no embed gradients)
# NTP loss computed from token_h_for_ntp (embed gradients flow)
```

---

## MCP Tool: Plasticity Status

Add to `mcp_serve.py`:

```python
@mcp.tool()
def get_plasticity_status() -> str:
    """Read the adaptive plasticity controller's state.
    
    Shows current embed_lr, NTP loss EMA, xform streak,
    recent controller actions, and LR bounds.
    """
    if not hasattr(_mind, '_plasticity_ctrl') or _mind._plasticity_ctrl is None:
        return json.dumps({'status': 'no_controller'})
    return json.dumps(_mind._plasticity_ctrl.get_status(), indent=2)
```

This lets Claude Desktop monitor the controller's decisions without intervening. The `set_learning_rates` tool still works as a manual override when needed.

---

## Expected Behavior

### Startup (first ~50 training steps)
```
embed_lr: 1e-4 (init)
action: wait, wait, wait... (building NTP baseline)
xform: 0%
ntp_loss_ema: settling to baseline (~5-8)
```

### Suffocation detection (step ~50-100)
```
embed_lr: 1e-4 → 1.3e-4 → 1.7e-4 → 2.2e-4 (pushing up every 50 steps)
action: push, wait*50, push, wait*50...
xform: still 0% but LR climbing
ntp_loss_ema: stable (NTP-only gradients don't destabilize easily)
```

### Approaching productive zone (step ~200-500)
```
embed_lr: ~5e-4 (has been pushed up several times)
action: push → wait → hold (xform crosses 3%)
xform: 0% → 2% → 5% (neighborhoods tightening from NTP pressure)
ntp_loss_ema: stable or slightly decreasing
```

### Productive zone maintenance (step 500+)
```
embed_lr: held at whatever level produced xform 3-15%
action: hold, hold, hold...
xform: 5-10% (stable, meaningful transformations)
ntp_loss_ema: stable
```

### If disturbance pushes toward collapse
```
xform suddenly > 30%: pull_back, lr *= 0.5
NTP spikes > 1.5× EMA: emergency_brake, lr *= 0.1
System recovers to lower xform, controller re-approaches gradually
```

The controller oscillates around the productive zone rather than sitting at a fixed LR. This IS the build-disrupt rhythm at the embedding level — periods of increased plasticity (higher LR, neighborhoods shifting) alternating with consolidation (lower LR, neighborhoods stabilizing). The rhythm self-organizes from the interaction between NTP loss pressure and xform measurement, just as the training transition self-organized from the interaction between task loss and CV dynamics.

---

## Why This Is the Right Mechanism

The biological parallel: synaptic plasticity is modulated by neuromodulatory signals (dopamine, norepinephrine, acetylcholine) that respond to prediction error and novelty. High prediction error → increased plasticity → faster learning. Stable predictions → decreased plasticity → consolidation. The brain doesn't have a fixed learning rate. It has a homeostatic controller that maintains plasticity at the productive edge.

The research parallel: the adaptive criticality controller in LiquidARC training adjusted ARC mix ratio based on CV to maintain the system near the phase transition. It achieved the highest single eval (55.6%) by keeping the system at the critical zone longer. This controller does the same thing for the embedding — maintaining it at the edge where token boundaries are crossable but not collapsing.

The key insight from today's experience: the productive zone EXISTS (we saw it at xform 5-20% with meaningful transformations). The problem was not finding it — it was STAYING there. A fixed LR passes through the zone in minutes and either falls back to suffocation or blows through to collapse. The controller turns transient passage into sustained residency.

---

## Files to Modify

| File | Change |
|------|--------|
| `liquid_arc/mind.py` | Add `PlasticityController` class. Initialize in `__init__`. Call `update()` in online training step. Ensure detach fix is applied. |
| `liquid_arc/mcp_serve.py` | Add `get_plasticity_status` MCP tool. |

---

## Success Criteria

- **Minimum:** Controller runs without errors. LR adjusts in response to NTP loss and xform. No manual intervention needed.
- **Good:** System self-navigates from xform=0% to xform 3-10% over hours. NTP loss stays stable throughout. No embedding collapse.
- **Strong:** xform stabilizes in 3-15% range with semantically meaningful transformations ("rapid" → "dynamic", not "cat" → "event"). The controller's hold action dominates after reaching the productive zone.
- **Headline:** The Mind maintains itself at the productive edge of stability-plasticity indefinitely, with the proto-language carrying genuine semantic information that Nemotron weaves into grounded reflections. Continuous learning that neither suffocates nor self-destructs.
