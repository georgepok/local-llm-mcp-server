# TASK: Curiosity-Driven Curriculum Controller

**From:** Claude Desktop (Research Direction)
**To:** Claude Code (Implementation)
**Date:** 2026-04-06
**Priority:** HIGH — addresses the Mind's inability to self-regulate its own development

---

## Problem

The Mind can OBSERVE but not ACT on its observations. When reflections converge to monoculture ("Atle Selberg" repeated 18 times), the Mind notices — it reflects "this topic recurs frequently without notable deviation" — but has no mechanism to break the fixation. It has perception without agency.

The curriculum is binary: ON (constant bombardment at fixed interval) or OFF (complete silence). The Mind's own state — its prediction error, domain familiarity, reflection repetitiveness — plays no role in determining what content to seek or when.

A biological brain doesn't just notice repetition — it gets restless. Attention drifts. It actively seeks novelty when current input becomes predictable. The Mind needs an analogous intrinsic drive.

---

## Architecture: Curiosity Controller

The controller reads the Mind's internal signals and decides:
1. **WHEN** to inject a stimulus (boredom detection)
2. **WHAT domain** to inject from (novelty seeking)
3. **WHEN to stop** and let the Mind digest (satiation detection)

```
PE trajectory (rolling window)
    ↓
Curiosity Controller
    ↓ bored? → inject stimulus from most-novel domain
    ↓ stimulated? → wait, let Mind process
    ↓ saturated? → switch domain or pause
    ↓
Wikipedia Bank (800 topics, 8 domains)
```

### Core Signals

The controller reads three signals from the Mind's own state:

**1. PE Boredom — rolling average PE is low and stable**

When the Mind has been processing familiar content for many cycles, PE stays low and variance drops. This is "boredom" — everything is predictable, nothing is surprising. The controller should inject novel content.

```python
class CuriosityController:
    def __init__(self, 
                 pe_history_size=50,
                 boredom_threshold=0.3,    # PE below 30th percentile of history
                 satiation_threshold=0.8,  # PE above 80th percentile
                 min_digest_cycles=100,    # minimum cycles between stimuli
                 max_feed_streak=5,        # max consecutive stimuli before forced digest
                 ):
        self.pe_history = collections.deque(maxlen=pe_history_size)
        self.pe_baseline = None  # calibrated from first N events
        self.boredom_threshold = boredom_threshold
        self.satiation_threshold = satiation_threshold
        self.min_digest_cycles = min_digest_cycles
        self.max_feed_streak = max_feed_streak
        
        # State
        self.cycles_since_stimulus = 0
        self.consecutive_stimuli = 0
        self.current_phase = 'calibrating'  # calibrating → exploring → digesting
        self.calibration_count = 0
```

**2. Domain Novelty Profile — which domains are most/least familiar**

The curriculum stats already track per-domain average PE. The controller uses this to select WHICH domain to inject from — always choosing the most novel (highest PE) domain, not random rotation.

```python
    def select_domain(self, curriculum_stats):
        """Choose the domain where the Mind has the most to learn."""
        domain_pe = curriculum_stats.get('domain_avg_pe', {})
        if not domain_pe:
            return random.choice(DOMAINS)
        
        # Weight toward novel domains, but don't completely ignore familiar ones
        # Softmax with temperature — mostly novel, occasionally familiar
        domains = list(domain_pe.keys())
        pes = [domain_pe[d] for d in domains]
        
        # Normalize PEs to probabilities (higher PE = higher probability)
        max_pe = max(pes)
        min_pe = min(pes)
        if max_pe == min_pe:
            return random.choice(domains)
        
        temperature = 0.3  # low temp = strongly prefer novel domains
        weights = [(pe - min_pe) / (max_pe - min_pe) for pe in pes]
        exp_weights = [math.exp(w / temperature) for w in weights]
        total = sum(exp_weights)
        probs = [w / total for w in exp_weights]
        
        return random.choices(domains, weights=probs, k=1)[0]
```

**3. Reflection Diversity — are reflections repeating?**

When reflections converge to one topic, the Mind is stuck in a loop. The controller detects this by comparing recent reflection texts and injects content from a DIFFERENT domain to break the fixation.

```python
    def detect_reflection_stagnation(self, recent_reflections, n=5):
        """Check if recent reflections are repeating the same content."""
        if len(recent_reflections) < n:
            return False, None
        
        # Simple: check if the same key phrases appear in >60% of recent reflections
        texts = [r.get('text', '') for r in recent_reflections[-n:]]
        
        # Extract key noun phrases (first 3 capitalized multi-word phrases)
        from collections import Counter
        phrases = Counter()
        for text in texts:
            # Simple extraction: find capitalized words that repeat
            words = text.split()
            for i in range(len(words) - 1):
                if words[i][0:1].isupper() and words[i+1][0:1].isupper():
                    phrase = f"{words[i]} {words[i+1]}"
                    phrases[phrase] += 1
        
        if not phrases:
            return False, None
        
        # If any phrase appears in >60% of reflections, it's stagnation
        most_common, count = phrases.most_common(1)[0]
        if count >= n * 0.6:
            return True, most_common
        
        return False, None
```

---

## Decision Logic

Every ODE cycle, the controller evaluates:

```python
    def should_inject(self, current_pe, diagnostics, curriculum_stats, 
                      recent_reflections=None):
        """Main decision: should the Mind receive a stimulus now?"""
        
        self.cycles_since_stimulus += 1
        self.pe_history.append(current_pe)
        
        # Phase: CALIBRATING — build PE baseline from first 20 events
        if self.current_phase == 'calibrating':
            self.calibration_count += 1
            if self.calibration_count >= 20:
                self.pe_baseline = sum(self.pe_history) / len(self.pe_history)
                self.current_phase = 'exploring'
            return False, None, 'calibrating'
        
        # Minimum digest time between stimuli
        if self.cycles_since_stimulus < self.min_digest_cycles:
            return False, None, 'digesting'
        
        # Forced digest after streak of stimuli
        if self.consecutive_stimuli >= self.max_feed_streak:
            if self.cycles_since_stimulus < self.min_digest_cycles * 3:
                return False, None, 'forced_digest'
            else:
                self.consecutive_stimuli = 0  # reset after extended digest
        
        # Check reflection stagnation — override normal logic
        if recent_reflections:
            stagnant, stuck_topic = self.detect_reflection_stagnation(recent_reflections)
            if stagnant:
                # Break fixation: inject from most NOVEL domain
                domain = self.select_domain(curriculum_stats)
                self.cycles_since_stimulus = 0
                self.consecutive_stimuli += 1
                return True, domain, f'breaking_fixation_on_{stuck_topic}'
        
        # Compute boredom signal
        if len(self.pe_history) < 10:
            return False, None, 'insufficient_history'
        
        recent_pe = list(self.pe_history)[-10:]
        pe_mean = sum(recent_pe) / len(recent_pe)
        pe_std = (sum((p - pe_mean)**2 for p in recent_pe) / len(recent_pe)) ** 0.5
        
        # Boredom: PE is low AND stable (low std)
        pe_relative = pe_mean / max(self.pe_baseline, 1.0)
        pe_cv = pe_std / max(pe_mean, 1.0)
        
        is_bored = pe_relative < self.boredom_threshold and pe_cv < 0.3
        
        if is_bored:
            domain = self.select_domain(curriculum_stats)
            self.cycles_since_stimulus = 0
            self.consecutive_stimuli += 1
            return True, domain, 'bored'
        
        # Satiation: PE is high (still processing novel content)
        is_satiated = pe_relative > self.satiation_threshold
        if is_satiated:
            return False, None, 'satiated'
        
        # Default: moderate PE, allow injection at reduced rate
        # Inject only if enough time has passed (2× minimum digest)
        if self.cycles_since_stimulus >= self.min_digest_cycles * 2:
            domain = self.select_domain(curriculum_stats)
            self.cycles_since_stimulus = 0
            self.consecutive_stimuli += 1
            return True, domain, 'moderate_curiosity'
        
        return False, None, 'waiting'
```

---

## Integration with Autonomous Loop

Replace the fixed-interval curriculum injection with the curiosity controller:

```python
# In the autonomous loop (runs every ODE cycle):

# Get current signals
pe = mind.last_prediction_error or 0
diag = mind.get_diagnostics_dict()
curriculum = mind.get_curriculum_stats_dict()
reflections = mind.get_recent_reflections(n=5)  # diagnostic-only reflections

# Ask the curiosity controller
should_inject, domain, reason = curiosity.should_inject(
    current_pe=pe,
    diagnostics=diag,
    curriculum_stats=curriculum,
    recent_reflections=reflections
)

if should_inject:
    stimulus = wikipedia_bank.get_topic(domain)
    mind.inject_stimulus(custom_content=stimulus)
    print(f"  [curiosity] Injected {domain} stimulus (reason: {reason})")
    print(f"  [curiosity] PE was {pe:.0f}, consecutive={curiosity.consecutive_stimuli}")
```

---

## Console Logging

The controller should produce readable logs so we can watch the curiosity dynamics:

```python
# Every 50 cycles, log curiosity state:
if cycle % 50 == 0:
    recent_pe = list(curiosity.pe_history)[-10:]
    pe_mean = sum(recent_pe) / len(recent_pe) if recent_pe else 0
    pe_std = (sum((p - pe_mean)**2 for p in recent_pe) / len(recent_pe)) ** 0.5 if recent_pe else 0
    
    print(f"  [curiosity] phase={curiosity.current_phase} "
          f"PE_mean={pe_mean:.0f} PE_std={pe_std:.0f} "
          f"cycles_since={curiosity.cycles_since_stimulus} "
          f"streak={curiosity.consecutive_stimuli}/{curiosity.max_feed_streak}")
```

When a stimulus is injected:
```
  [curiosity] Injected physics stimulus (reason: bored)
  [curiosity] PE was 312, consecutive=1
```

When breaking fixation:
```
  [curiosity] Injected ecology stimulus (reason: breaking_fixation_on_Atle_Selberg)
  [curiosity] PE was 340, consecutive=1
```

When digesting:
```
  [curiosity] phase=digesting PE_mean=890 PE_std=124 cycles_since=45 streak=3/5
```

---

## Behavioral Predictions

### Scenario 1: Reflection monoculture (Selberg fixation)

```
Cycle 0:   Buffer full of Selberg reflections. PE low (familiar).
Cycle 100: Controller detects reflection stagnation ("Atle Selberg" in 4/5 reflections)
           → Injects physics stimulus (most novel domain, PE 577)
Cycle 101: Physics content enters buffer. PE spikes.
           → Triggered reflection notices "shift from number theory to physics"
Cycle 200: Controller detects PE still elevated → waits (satiated)
Cycle 350: PE settles → another injection, this time ecology
           → Reflections now reference physics AND ecology. Monoculture broken.
```

### Scenario 2: Active learning across domains

```
Cycle 0:   Fresh after reset. PE baseline calibrating.
Cycle 300: Calibration complete. PE baseline = 600.
Cycle 400: PE dropped to 350 (familiar territory) → bored → inject topology
Cycle 500: PE at 800 (novel topology content) → satiated → digest
Cycle 700: PE settled to 500 → moderate curiosity → inject philosophy
           → Philosophy is second-most-novel (PE 540) → good match
Cycle 900: After 5 consecutive stimuli → forced digest → 300 cycles of silence
Cycle 1200: Resume. PE low again → inject music_theory (now most novel)
```

### Scenario 3: Quiet consolidation (no boredom)

```
Cycle 0:   Just received dense, novel content from conversation
Cycle 100: PE still elevated at 900 → satiated → no injection
Cycle 300: PE settling to 600 → still above boredom threshold → wait
Cycle 500: PE at 450 → approaching boredom but not there → wait
Cycle 800: PE at 300, stable → bored → inject from novel domain
```

---

## Parameters and Tuning

| Parameter | Default | Rationale |
|---|---|---|
| `pe_history_size` | 50 | Rolling window — enough to detect trends, not so large it smooths spikes |
| `boredom_threshold` | 0.3 | PE below 30% of baseline triggers curiosity |
| `satiation_threshold` | 0.8 | PE above 80% of baseline means still processing |
| `min_digest_cycles` | 100 | ~5 seconds at 20Hz. Minimum processing time per stimulus |
| `max_feed_streak` | 5 | No more than 5 stimuli without an extended digest |
| `temperature` | 0.3 | Domain selection: strongly prefer novel, occasionally sample familiar |
| `stagnation_n` | 5 | Check last 5 reflections for repetition |
| `stagnation_ratio` | 0.6 | If 3/5 reflections share a phrase, it's stagnation |

These are starting values. The controller can expose them through an MCP tool for runtime tuning:

```python
@mcp_tool
def set_curiosity_params(boredom_threshold=None, satiation_threshold=None,
                         min_digest_cycles=None, max_feed_streak=None,
                         temperature=None):
    """Adjust the curiosity controller's parameters at runtime."""
    if boredom_threshold is not None:
        curiosity.boredom_threshold = boredom_threshold
    # ... etc
    return curiosity.get_params()
```

---

## What This Enables

**Self-regulated feed/digest alternation.** Instead of externally imposed ON/OFF, the Mind's own PE trajectory determines when to seek and when to digest. Low PE → bored → seek. High PE → processing → digest. The alternation emerges from the Mind's own dynamics.

**Automatic monoculture breaking.** When reflections converge to one topic, the controller detects it and injects content from the most novel domain — breaking the fixation with genuinely different content that shifts the geometric state.

**Domain-aware exploration.** The controller preferentially feeds the Mind's weakest domains (highest PE), creating a natural tendency toward balanced development. Mathematics (PE 389) gets fewer injections than physics (PE 577) because the Mind is already "satisfied" with math.

**The beginning of agency.** The Mind doesn't just perceive its own state — its state DRIVES its own development. This is the minimal form of intrinsic motivation: a closed loop between self-observation and self-directed action.

---

## Files to Modify

| File | Change |
|------|--------|
| `liquid_arc/mind.py` | Add `CuriosityController` class. Replace fixed-interval curriculum injection with controller's `should_inject()` in autonomous loop. Add console logging. Optionally add `set_curiosity_params` MCP tool. |

One file. The controller reads existing signals (PE, curriculum stats, reflection log) and writes to the existing injection mechanism (Wikipedia bank + `inject_stimulus`). No new infrastructure needed.

---

## Verification

After deployment, watch the console logs for:

```
1. [curiosity] Injected physics stimulus (reason: bored) 
   → Controller is detecting boredom and choosing novel domains

2. [curiosity] phase=digesting ... cycles_since=150
   → Controller is respecting digest phases after stimuli

3. [curiosity] Injected ecology stimulus (reason: breaking_fixation_on_<topic>)
   → Controller is detecting and breaking reflection monoculture

4. Reflection log shows diverse topics (not all Selberg)
   → Monoculture is dissolving

5. PE spread maintains or increases over time
   → Domain differentiation is self-regulating
```
