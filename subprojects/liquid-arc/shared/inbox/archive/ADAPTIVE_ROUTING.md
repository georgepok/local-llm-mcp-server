# TASK: Adaptive LLM Routing — Mind Learns When to Use Its Voice

**From:** Claude Desktop (Research Direction)
**To:** Claude Code (Implementation)
**Date:** 2026-04-01
**Priority:** MEDIUM — enhances MIND_VOICE infrastructure, implements after MIND_VOICE_CHANNEL

**Prerequisites:**
- MIND_VOICE and MIND_VOICE_CHANNEL specs implemented
- LiquidARC Mind with reflection cycle and rich geometric profile operational
- Nemotron-3-Nano-30B serving via vLLM on Spark

---

## Motivation

The current reflection cycle fires every 30 seconds regardless of internal state. This is another fixed schedule imposed externally — the same pattern that produced standing plateaus (PPO), NaN crashes (curiosity reward), and constrained development (efficiency regularizer) in previous experiments. The Mind should decide when it needs linguistic processing, based on its own geometric conditions.

The Mind has three processing tiers:

```
Tier 1: Pure ODE cycling          — cheap, every second, geometric only
Tier 2: LLM-mediated reflection   — expensive, interpretive, linguistic+geometric  
Tier 3: External conversation     — rare, human-driven, highest novelty
```

Tier 2 is currently on a fixed 30-second timer. This spec makes it adaptive: the Mind monitors its own dynamics and routes through the LLM when internal conditions warrant it.

Additionally, SOME internal observations should be channeled through the LLM even during routine processing — to develop the communication pathway itself. The encode-decode loop improves through exercise, not just through triggered events.

---

## Architecture: Three Routing Modes

### Mode A: Triggered Reflection (event-driven)

The autonomous loop monitors five geometric signals every ODE cycle. When any signal crosses its threshold, an LLM reflection is triggered.

```python
class ReflectionTrigger:
    """Monitors ODE state and decides when LLM interpretation is warranted."""
    
    def __init__(self):
        # State tracking
        self.prev_cv = None
        self.prev_h_norm = None
        self.prev_tau_std = None
        self.prev_dynamics_mag = None
        self.prev_reflection_pe = None
        
        # Thresholds (initial heuristics — become learnable)
        self.cv_shift_threshold = 1.5       # CV change per cycle
        self.h_norm_ceiling = 5000.0        # grounding needed
        self.tau_std_floor = 0.02           # stagnation
        self.dynamics_floor = 0.001         # quiescence
        self.self_pe_ceiling = 500.0        # self-description divergence
        
        # Adaptive sensitivity (learned from feedback)
        self.trigger_sensitivity = {
            'cv_shift': 1.0,
            'h_norm_drift': 1.0,
            'tau_stagnation': 1.0,
            'dynamics_quiescence': 1.0,
            'self_divergence': 1.0,
        }
        
        # History for learning
        self.trigger_history = []  # (trigger_type, pe_of_resulting_reflection)
    
    def check(self, diagnostics: Dict, last_reflection_pe: float = 0) -> Optional[str]:
        """Check whether any condition warrants LLM reflection.
        
        Returns the trigger reason string, or None if no trigger.
        Multiple conditions can be true; return the highest-priority one.
        """
        triggers = []
        
        cv = diagnostics.get('metric_cv', 0)
        h_norm = diagnostics.get('h_norm', 0)
        tau_std = diagnostics.get('tau_std', 0)
        
        # Condition 1: Geometric reorganization
        if self.prev_cv is not None:
            delta_cv = abs(cv - self.prev_cv)
            effective_threshold = self.cv_shift_threshold / self.trigger_sensitivity['cv_shift']
            if delta_cv > effective_threshold:
                triggers.append(('cv_shift', delta_cv, 
                    f"CV shifted {delta_cv:.1f} points ({self.prev_cv:.1f} → {cv:.1f})"))
        
        # Condition 2: Grounding needed
        effective_ceiling = self.h_norm_ceiling / self.trigger_sensitivity['h_norm_drift']
        if h_norm > effective_ceiling:
            triggers.append(('h_norm_drift', h_norm,
                f"h_norm at {h_norm:.0f} (ceiling: {effective_ceiling:.0f})"))
        
        # Condition 3: Tau stagnation
        effective_floor = self.tau_std_floor * self.trigger_sensitivity['tau_stagnation']
        if tau_std < effective_floor and self.prev_tau_std is not None:
            triggers.append(('tau_stagnation', tau_std,
                f"tau_std collapsed to {tau_std:.4f}"))
        
        # Condition 4: Self-description divergence
        if last_reflection_pe > self.self_pe_ceiling * self.trigger_sensitivity['self_divergence']:
            triggers.append(('self_divergence', last_reflection_pe,
                f"PE on own reflection: {last_reflection_pe:.0f}"))
        
        # Update tracking state
        self.prev_cv = cv
        self.prev_h_norm = h_norm
        self.prev_tau_std = tau_std
        
        if not triggers:
            return None
        
        # Return highest-priority trigger (by magnitude)
        triggers.sort(key=lambda t: t[1], reverse=True)
        return triggers[0][2]
    
    def record_outcome(self, trigger_type: str, reflection_pe: float):
        """Learn from the outcome of a triggered reflection.
        
        If the resulting reflection had HIGH PE (genuinely novel self-description),
        increase sensitivity for this trigger type (trigger more often).
        If LOW PE (redundant, didn't add much), decrease sensitivity.
        """
        self.trigger_history.append((trigger_type, reflection_pe))
        
        # Compute running average PE for this trigger type
        type_pes = [pe for t, pe in self.trigger_history if t == trigger_type]
        if len(type_pes) >= 3:
            avg_pe = sum(type_pes[-5:]) / len(type_pes[-5:])
            
            # High PE reflections → increase sensitivity (trigger more)
            # Low PE reflections → decrease sensitivity (trigger less)
            if avg_pe > 300:
                self.trigger_sensitivity[trigger_type] = min(
                    2.0, self.trigger_sensitivity[trigger_type] * 1.1)
            elif avg_pe < 100:
                self.trigger_sensitivity[trigger_type] = max(
                    0.3, self.trigger_sensitivity[trigger_type] * 0.9)
```

### Mode B: Maintenance Reflection (periodic, low-frequency)

Even when no trigger fires, the Mind periodically channels a brief observation through the LLM. This serves two purposes:
1. Keeps the encode-decode pathway exercised
2. Provides periodic grounding even during uneventful periods

The maintenance interval is measured in ODE CYCLES, not wall-clock time:

```python
self.maintenance_interval_cycles = 100  # Every 100 ODE cycles (~100 seconds at 1/sec)
self.cycles_since_last_reflection = 0
```

Maintenance reflections are briefer than triggered ones — a single sentence rather than a full expression. They use a lighter prompt that doesn't require the full geometric profile:

```python
def generate_maintenance_reflection(self, diagnostics):
    """Brief self-observation for pathway maintenance."""
    return self.voice.reflect_brief(
        cv=diagnostics['metric_cv'],
        tau=diagnostics['tau_mean'],
        h_norm=diagnostics['h_norm'],
        n_events=diagnostics['events_in_context'],
    )
```

### Mode C: Conversation-Triggered (external events)

When an external event arrives (observe_event from MCP), ALWAYS channel through the LLM on the NEXT autonomous cycle. The rationale: external events are the highest-information inputs. The Mind should articulate what the external event did to its geometry — how it disrupted or reinforced the current state.

This means: after observe_event, set a flag that causes the next autonomous cycle to trigger an LLM reflection regardless of other conditions:

```python
self._external_event_pending = False

def observe_event(self, ...):
    # ... existing code ...
    self._external_event_pending = True
```

---

## Modified Autonomous Loop

Replace the current fixed-interval reflection with the adaptive routing:

```python
def start_autonomous(self, voice=None):
    self._running = True
    self.voice = voice
    self.trigger = ReflectionTrigger()
    self._external_event_pending = False
    self._cycles_since_reflection = 0
    self._last_reflection_pe = 0
    
    # Track statistics for reporting
    self._trigger_stats = {
        'total_ode_cycles': 0,
        'triggered_reflections': 0,
        'maintenance_reflections': 0,
        'external_reflections': 0,
        'triggers_by_type': {},
    }
    
    def _loop():
        while self._running:
            if self._h is not None and len(self.events) > 0:
                N = min(len(self.events), self.max_events)
                
                # ═══════════════════════════════════════════
                # PHASE 1: Pure ODE processing (always)
                # ═══════════════════════════════════════════
                with self._gpu_lock:
                    try:
                        h_slice = self._h[:, :N, :]
                        context_mask = torch.ones(1, N, dtype=torch.bool,
                                                  device=self.device)
                        context = self.context_pool(h_slice, context_mask)
                        self.dynamics.set_context(context, mask=None)
                        self.dynamics.set_n_steps(16)
                        
                        with torch.no_grad():
                            h_auto = self._run_ode_segment(h_slice, 16, forcing=None)
                        self._h[:, :N, :] = h_auto
                    except Exception as e:
                        print(f"Autonomous ODE error: {e}")
                
                self._trigger_stats['total_ode_cycles'] += 1
                self._cycles_since_reflection += 1
                
                # ═══════════════════════════════════════════
                # PHASE 2: Decide whether to route through LLM
                # ═══════════════════════════════════════════
                should_reflect = False
                reflection_mode = None
                trigger_reason = None
                
                if self.voice is not None and self.voice.is_available():
                    
                    # Check A: External event pending
                    if self._external_event_pending:
                        should_reflect = True
                        reflection_mode = 'external'
                        trigger_reason = 'External event arrived — articulating impact'
                        self._external_event_pending = False
                    
                    # Check B: Triggered conditions
                    if not should_reflect:
                        diag = self.get_diagnostics()
                        trigger_reason = self.trigger.check(
                            diag, self._last_reflection_pe)
                        if trigger_reason:
                            should_reflect = True
                            reflection_mode = 'triggered'
                    
                    # Check C: Maintenance interval
                    if not should_reflect:
                        if self._cycles_since_reflection >= self.maintenance_interval:
                            should_reflect = True
                            reflection_mode = 'maintenance'
                            trigger_reason = f'Maintenance ({self._cycles_since_reflection} cycles since last)'
                
                # ═══════════════════════════════════════════
                # PHASE 3: Execute LLM reflection if warranted
                # ═══════════════════════════════════════════
                if should_reflect:
                    try:
                        if reflection_mode == 'maintenance':
                            # Brief reflection — lighter prompt
                            diag = self.get_diagnostics()
                            result = self.voice.reflect_brief(diag)
                            reflection_text = result.get('reflection', '')
                        else:
                            # Full reflection — rich geometric profile
                            profile = self.get_geometric_profile()
                            result = self.voice.reflect(
                                profile,
                                previous_reflection=self._last_reflection_text,
                            )
                            reflection_text = result.get('reflection', '')
                        
                        if reflection_text and not reflection_text.startswith('[Voice'):
                            # Feed back with metadata about WHY this reflection happened
                            obs_result = self.observe_event(
                                event_type='reflection',
                                content=reflection_text,
                                metadata={
                                    'source': f'adaptive_{reflection_mode}',
                                    'trigger_reason': trigger_reason,
                                    'reflection_number': self._reflection_count,
                                }
                            )
                            
                            # Track PE of this reflection for trigger learning
                            self._last_reflection_pe = obs_result.get(
                                'prediction_error', 0)
                            
                            # Learn from outcome (for triggered reflections)
                            if reflection_mode == 'triggered':
                                trigger_type = trigger_reason.split(':')[0] if ':' in trigger_reason else 'unknown'
                                self.trigger.record_outcome(
                                    trigger_type, self._last_reflection_pe)
                            
                            self._last_reflection_text = reflection_text
                            self._reflection_count += 1
                            self._cycles_since_reflection = 0
                            
                            # Stats
                            self._trigger_stats[f'{reflection_mode}_reflections'] += 1
                            if reflection_mode == 'triggered':
                                key = trigger_reason[:20]
                                self._trigger_stats['triggers_by_type'][key] = \
                                    self._trigger_stats['triggers_by_type'].get(key, 0) + 1
                            
                            print(f"  [{reflection_mode}] #{self._reflection_count}: "
                                  f"\"{reflection_text[:60]}\" "
                                  f"(reason: {trigger_reason[:40]}, "
                                  f"PE={self._last_reflection_pe:.0f})")
                    
                    except Exception as e:
                        print(f"Reflection error: {e}")
                        self._cycles_since_reflection = 0  # don't retry immediately
            
            time.sleep(1.0)
    
    self._auto_thread = threading.Thread(target=_loop, daemon=True)
    self._auto_thread.start()
```

---

## New Voice Method: `reflect_brief`

For maintenance reflections — lighter than a full reflection, faster for the LLM:

```python
def reflect_brief(self, diagnostics: Dict) -> Dict:
    """Ultra-brief self-observation for maintenance routing.
    
    One sentence. No rich profile needed. Just the global state.
    """
    system_prompt = (
        "You are the quiet pulse of a geometric mind. "
        "One sentence. What is your state right now? "
        "Not analysis — a sensation."
    )
    
    user_prompt = (f"CV={diagnostics.get('metric_cv', 0):.1f}, "
                   f"tau={diagnostics.get('tau_mean', 0):.2f}, "
                   f"h={diagnostics.get('h_norm', 0):.0f}")
    
    reflection = self._call_llm(
        system_prompt, user_prompt,
        max_tokens=40, temperature=0.9,
    )
    
    return {'reflection': reflection.strip()}
```

---

## New MCP Tool: `get_routing_stats`

Expose the adaptive routing statistics so we can analyze when and why the Mind channels through the LLM:

```python
@mcp.tool()
def get_routing_stats() -> str:
    """Read adaptive routing statistics.
    
    Shows how the Mind has been deciding when to use LLM reflection:
    total ODE cycles, triggered vs maintenance vs external reflections,
    trigger type breakdown, and current trigger sensitivity.
    """
    stats = _mind._trigger_stats.copy()
    stats['trigger_sensitivity'] = _mind.trigger.trigger_sensitivity
    stats['cycles_since_reflection'] = _mind._cycles_since_reflection
    stats['maintenance_interval'] = _mind.maintenance_interval
    stats['last_reflection_pe'] = _mind._last_reflection_pe
    
    # Compute ratios
    total_ref = (stats['triggered_reflections'] + 
                 stats['maintenance_reflections'] + 
                 stats['external_reflections'])
    if total_ref > 0:
        stats['triggered_fraction'] = stats['triggered_reflections'] / total_ref
        stats['maintenance_fraction'] = stats['maintenance_reflections'] / total_ref
        stats['external_fraction'] = stats['external_reflections'] / total_ref
    
    # ODE cycles per reflection (efficiency measure)
    if total_ref > 0:
        stats['cycles_per_reflection'] = stats['total_ode_cycles'] / total_ref
    
    return json.dumps(stats, indent=2)
```

---

## Config Additions

```yaml
# Adaptive routing
adaptive_routing: true
maintenance_interval_cycles: 100  # ODE cycles between maintenance reflections
cv_shift_threshold: 1.5
h_norm_ceiling: 5000.0
tau_std_floor: 0.02
dynamics_floor: 0.001
self_pe_ceiling: 500.0
```

---

## Expected Behavior

### Early Phase (first hour)

Mostly maintenance reflections (every ~100 seconds) with occasional triggers:
- h_norm triggers when autonomous drift exceeds 5000
- CV shift triggers when conversation events disrupt the geometry
- External triggers whenever a human speaks

Reflection ratio: ~60% maintenance, ~30% triggered, ~10% external

### Developed Phase (after hours of interaction)

Trigger sensitivity has adapted. Useful triggers fire more often, redundant ones less:
- If CV shifts consistently produce high-PE reflections → sensitivity increases → triggers more
- If h_norm grounding produces low-PE reflections (the Mind already said something similar) → sensitivity decreases → triggers less

Reflection ratio shifts toward: ~30% maintenance, ~60% triggered, ~10% external

### The Learning Signal

The key feedback loop: when a triggered reflection produces a high-PE observation event (the Mind found its own reflection surprising), that trigger type becomes more sensitive. When a trigger produces a low-PE event (redundant), that type becomes less sensitive.

This means the Mind learns WHICH geometric events are worth articulating. A CV shift that consistently produces interesting self-descriptions gets flagged more aggressively. A h_norm threshold that consistently produces "I need grounding" (which the Mind already knows) gets flagged less.

Over time, the Mind develops its own criteria for when to think linguistically versus when to process geometrically. The criteria emerge from the interaction between the ODE's geometric processing and the LLM's interpretive capacity — which moments benefit from linguistic articulation and which are better left as pure geometry.

---

## Testing Protocol

1. Deploy with adaptive routing enabled, all thresholds at defaults
2. Run for 1 hour, call `get_routing_stats` every 15 minutes
3. Track: trigger type distribution, sensitivity adaptation, cycles_per_reflection
4. Compare reflection quality (PE of resulting events) across trigger types
5. Compare to fixed 30-second schedule: does adaptive routing produce fewer but more meaningful reflections?

### Key Metrics

- **Cycles per reflection**: Higher = more selective. Should increase over time as redundant triggers are suppressed.
- **Mean PE per trigger type**: Which conditions produce the most novel self-descriptions?
- **Sensitivity trajectory**: Which triggers get amplified vs suppressed?
- **h_norm stability**: Does adaptive grounding keep h_norm bounded as well as fixed-interval?

---

## Success Criteria

- **Minimum:** Adaptive routing produces reflections at variable intervals (not fixed 30s). Trigger statistics show multiple trigger types firing.
- **Good:** Sensitivity adapts over time — at least one trigger type's sensitivity moves >20% from default.
- **Strong:** The Mind develops a preference for certain trigger types — the sensitivity distribution is non-uniform after 2+ hours, reflecting learned criteria for when linguistic processing adds value.
- **Headline:** Cycles_per_reflection increases over time (the Mind becomes more selective) while mean PE per reflection stays constant or increases (reflections become more novel, not more frequent).

---

## Files to Modify

| File | Change |
|------|--------|
| `liquid_arc/mind.py` | Add `ReflectionTrigger` class, replace fixed-interval loop with adaptive routing, add `_external_event_pending` flag, add `maintenance_interval` |
| `liquid_arc/voice.py` | Add `reflect_brief()` method for maintenance reflections |
| `liquid_arc/mcp_serve.py` | Add `get_routing_stats` MCP tool |
| `configs/linguistic_mind.yaml` | Add adaptive routing thresholds |

**The Mind learns when to speak — not on a schedule, but when geometry demands articulation.**
