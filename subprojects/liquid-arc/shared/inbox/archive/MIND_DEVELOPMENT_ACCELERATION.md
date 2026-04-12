# TASK: Mind Development Acceleration — Curriculum Diversity & Consolidation

**From:** Claude Desktop (Research Direction)
**To:** Claude Code (Implementation)
**Date:** 2026-04-05
**Priority:** HIGH — addresses the primary bottleneck in Mind development

---

## Problem Statement

After 408 curriculum stimuli across 8 domains, the Mind shows:
- **tau_std = 0.37** (good — processing depth differentiating)
- **Reflection quality improved** (genuine cross-domain synthesis)
- **Structural pattern recognition emerging** (fishing village test: 2× PE discrimination)

But:
- **PE spread collapsed to 3.4%** (should be 50%+)
- **All domains uniformly novel** (PE 426-441, no differentiation)
- **Curriculum content is repetitive** ("slope" explained 4+ times, "cell differentiation" 4+ times)
- **Language drift persists** (Chinese, mixed-language in reflections)
- **Triggered reflections nearly dead** (7 out of 419 = 1.7%)

**Root cause:** Qwen3-4B generates the same content repeatedly when given the same prompt conditioned on similar state. The geometric prefix reinforces familiarity, biasing Qwen3 toward repeating what it's already said. The Mind is fed the same meal every cycle.

---

## Fix 1: Diverse Curriculum Generation (Critical)

### 1A: Topic Tracking + Anti-Repetition

Maintain a per-domain list of topics already covered. Include this in the curriculum prompt so Qwen3 avoids repetition:

```python
# Per-domain topic history (persisted across restarts)
curriculum_history = {
    'topology': ['homotopy', 'fundamental groups', 'homeomorphism'],
    'biology': ['cell differentiation', 'photosynthesis', 'morphogenesis'],
    # ... etc
}

def generate_curriculum_prompt(domain):
    already_covered = ', '.join(curriculum_history[domain][-20:])  # last 20 topics
    
    return (
        f"Explain a concept from {DOMAIN_NAMES[domain]} that is DIFFERENT from "
        f"these already-covered topics: {already_covered}. "
        f"Choose something new and specific. "
        f"Respond only in English. Keep it under 200 words."
    )
```

After each stimulus, extract the topic name (first noun phrase or explicit concept) and append to history:

```python
# Simple extraction: first bolded term or first capitalized concept
import re
def extract_topic(text):
    # Try bolded: **Topic Name**
    bold = re.search(r'\*\*(.+?)\*\*', text)
    if bold:
        return bold.group(1).lower().strip()
    # Try first sentence subject
    first_line = text.strip().split('\n')[0]
    return first_line[:50].lower().strip()

topic = extract_topic(stimulus_text)
curriculum_history[domain].append(topic)
```

### 1B: Tiered Complexity Progression

Don't repeat the same difficulty level. Track a per-domain complexity tier that advances:

```python
COMPLEXITY_TIERS = [
    "basic concept explained simply",
    "intermediate concept with connections to other ideas",
    "advanced concept requiring prior knowledge",
    "cutting-edge research question or open problem",
    "cross-domain connection between this field and another"
]

curriculum_tier = {domain: 0 for domain in DOMAINS}

def get_tier_instruction(domain):
    tier = curriculum_tier[domain] % len(COMPLEXITY_TIERS)
    curriculum_tier[domain] += 1
    return COMPLEXITY_TIERS[tier]
```

Include in the prompt:

```python
def generate_curriculum_prompt(domain):
    already_covered = ', '.join(curriculum_history[domain][-20:])
    tier = get_tier_instruction(domain)
    
    return (
        f"Explain a concept from {DOMAIN_NAMES[domain]}. "
        f"Difficulty level: {tier}. "
        f"AVOID these already-covered topics: {already_covered}. "
        f"Choose something genuinely different. "
        f"Respond only in English. Keep it under 200 words."
    )
```

### 1C: Custom Stimulus Injection as Fallback

If Qwen3-4B's curriculum generation diversity plateaus despite anti-repetition prompts, provide a curated topic list per domain that the system cycles through:

```python
CURATED_TOPICS = {
    'topology': [
        "Explain the Euler characteristic and how it classifies surfaces",
        "What is a fiber bundle and why does it matter in physics?",
        "How does persistent homology extract shape from data?",
        "What makes the Poincaré conjecture so important?",
        "Explain covering spaces and their relationship to fundamental groups",
        "What is a CW complex and how does it simplify topology?",
        "How does Morse theory connect topology to calculus?",
        "What are knot invariants and why are knots hard to classify?",
        # ... 30+ per domain
    ],
    'physics': [
        "Explain spontaneous symmetry breaking in simple terms",
        "What is the renormalization group and why does it matter?",
        "How does the Aharonov-Bohm effect challenge classical intuition?",
        "What is topological order in condensed matter?",
        "Explain the connection between entropy and information",
        # ... etc
    ],
    # ... all 8 domains
}

# Use curated when auto-generation starts repeating
def get_stimulus(domain):
    if len(set(curriculum_history[domain][-10:])) < 5:  # >50% repeats recently
        # Fall back to curated
        idx = len(curriculum_history[domain]) % len(CURATED_TOPICS[domain])
        prompt = CURATED_TOPICS[domain][idx]
    else:
        prompt = generate_curriculum_prompt(domain)
    return prompt
```

---

## Fix 2: Feed/Digest Alternation (Important)

Continuous curriculum bombardment prevents geometric consolidation. The ODE state needs time to self-organize around accumulated content.

### Implementation:

```python
class CurriculumScheduler:
    def __init__(self, 
                 feed_count=20,      # stimuli per feed phase
                 digest_cycles=200,  # ODE cycles per digest phase
                 current_phase='feed',
                 stimuli_this_phase=0,
                 cycles_this_phase=0):
        self.feed_count = feed_count
        self.digest_cycles = digest_cycles
        self.current_phase = current_phase
        self.stimuli_this_phase = stimuli_this_phase
        self.cycles_this_phase = cycles_this_phase
    
    def should_feed(self):
        """Called each ODE cycle to determine if curriculum should fire."""
        if self.current_phase == 'feed':
            # Normal curriculum interval applies
            return True  # (actual timing handled by stimulus_interval)
        else:
            # Digest phase — no curriculum
            return False
    
    def on_stimulus_delivered(self):
        """Called after each curriculum stimulus."""
        self.stimuli_this_phase += 1
        if self.stimuli_this_phase >= self.feed_count:
            self.current_phase = 'digest'
            self.stimuli_this_phase = 0
            self.cycles_this_phase = 0
            print(f"  [curriculum] Entering DIGEST phase ({self.digest_cycles} cycles)")
    
    def on_ode_cycle(self):
        """Called every ODE cycle."""
        if self.current_phase == 'digest':
            self.cycles_this_phase += 1
            if self.cycles_this_phase >= self.digest_cycles:
                self.current_phase = 'feed'
                self.cycles_this_phase = 0
                print(f"  [curriculum] Entering FEED phase ({self.feed_count} stimuli)")
```

In the autonomous loop:

```python
# Replace direct curriculum check with scheduler
if scheduler.should_feed() and cycles_since_last_stimulus >= stimulus_interval:
    stimulus = generate_stimulus(current_domain)
    force_geometric_signal(stimulus)
    scheduler.on_stimulus_delivered()

scheduler.on_ode_cycle()
```

### Console logging for phases:

```python
if scheduler.current_phase == 'digest' and scheduler.cycles_this_phase % 50 == 0:
    diag = get_diagnostics()
    print(f"  [digest] cycle {scheduler.cycles_this_phase}/{scheduler.digest_cycles} "
          f"CV={diag['metric_cv']:.2f} tau_std={diag['tau_std']:.3f} "
          f"h_norm={diag['h_norm']:.0f}")
```

This lets us see if the geometry self-organizes during digest phases — CV, tau_std, cluster structure should evolve without the constant forcing of new content.

---

## Fix 3: Language Pinning (Quick Fix)

Every prompt that goes to Qwen3 for generation must include "Respond only in English."

### Curriculum prompts:

Already included in Fix 1's prompt templates.

### Reflection prompts:

```python
REFLECTION_PROMPT = (
    "What patterns, connections, or shifts do you notice in your current state? "
    "Respond in English only. One concise paragraph."
)
```

### Express_through_qwen prompts:

```python
def build_expression_prompt(context_snippets, focus_query):
    return (
        f"Given the following context: {context_snippets}. "
        f"{focus_query or 'What themes dominate your current processing?'} "
        f"Respond in English only."
    )
```

### System-level pin (if chat template supports it):

```python
# In all Qwen3 generation calls, prepend system message
system_msg = {"role": "system", "content": "You are a scientific assistant. Always respond in English."}
```

---

## Fix 4: Curriculum Stats Instrumentation (from earlier spec)

If not already implemented from `CURRICULUM_INSTRUMENTATION.md`, the curriculum stats tracking needs to be wired to the Qwen3 pathway:

- Log `(domain, PE, CV, tau_mean)` after each curriculum stimulus
- Compute rolling averages per domain
- Calculate `pe_spread_pct` as headline metric
- Console log every 10 stimuli with domain differentiation table
- `get_curriculum_stats` MCP tool returns the new fields

(See `CURRICULUM_INSTRUMENTATION.md` in archive for full implementation details.)

---

## Fix 5: Reflection Trigger Sensitivity (Quick Fix)

Only 7 triggered reflections out of 419 (1.7%). The CV-shift trigger isn't firing because curriculum events enter at stable CV ~4.0. Two adjustments:

### 5A: Lower CV-shift threshold

```python
# Current: cv_shift sensitivity = 1.0 (requires CV to move by 1.0 to trigger)
# After 408 stimuli, CV barely moves per event
# New: lower to 0.3 to catch subtler geometric shifts
trigger_sensitivity['cv_shift'] = 0.3
```

### 5B: Add PE-based trigger

The PE signal is meaningful — spikes indicate genuinely novel content. Add a PE-based reflection trigger:

```python
# Trigger reflection when PE is unusually high or low relative to running average
pe_history = collections.deque(maxlen=50)

def check_pe_trigger(current_pe):
    if len(pe_history) < 10:
        pe_history.append(current_pe)
        return False
    
    pe_mean = sum(pe_history) / len(pe_history)
    pe_std = (sum((p - pe_mean)**2 for p in pe_history) / len(pe_history)) ** 0.5
    pe_history.append(current_pe)
    
    # Trigger if PE is >1.5 std from mean (either direction)
    if abs(current_pe - pe_mean) > 1.5 * max(pe_std, 10.0):
        return True
    return False
```

This makes the Mind reflect when something genuinely surprising (or surprisingly familiar) happens — which is exactly when reflection is most productive.

---

## Fix 6: Consolidation Metrics (New)

During digest phases, we need to measure whether geometric consolidation is happening. Add tracking for:

### 6A: Cluster evolution during digest

```python
# At start and end of each digest phase, record cluster structure
def record_cluster_snapshot():
    # Get current cluster assignments from ODE state
    h_state = get_current_h()
    # Use same clustering as get_context
    clusters = compute_clusters(h_state)
    return {
        'n_clusters': len(clusters),
        'sizes': [len(c) for c in clusters],
        'inter_cluster_distance': compute_inter_cluster_distance(h_state, clusters),
        'intra_cluster_distance': compute_intra_cluster_distance(h_state, clusters),
    }

# Log at start and end of digest
digest_start_clusters = record_cluster_snapshot()
# ... digest phase runs ...
digest_end_clusters = record_cluster_snapshot()

# Consolidation metric: did clusters become more separated?
separation_change = (
    digest_end_clusters['inter_cluster_distance'] / 
    digest_end_clusters['intra_cluster_distance']
) - (
    digest_start_clusters['inter_cluster_distance'] / 
    digest_start_clusters['intra_cluster_distance']
)

if separation_change > 0:
    print(f"  [digest] Clusters SEPARATED by {separation_change:.3f} (consolidation)")
else:
    print(f"  [digest] Clusters MERGED by {separation_change:.3f} (no consolidation)")
```

### 6B: Tau evolution during digest

Track whether tau_std increases during silent consolidation (the system learning to differentiate processing depth without external forcing):

```python
# At start and end of digest
tau_std_start = get_diagnostics()['tau_std']
# ... digest phase ...
tau_std_end = get_diagnostics()['tau_std']

print(f"  [digest] tau_std: {tau_std_start:.3f} → {tau_std_end:.3f} "
      f"({'↑' if tau_std_end > tau_std_start else '↓'})")
```

---

## Implementation Priority

| Fix | Priority | Effort | Impact |
|-----|----------|--------|--------|
| 1A: Topic tracking + anti-repetition | **Critical** | Medium | Breaks the repetition loop |
| 3: Language pinning | **Critical** | Small | Prevents multilingual contamination |
| 2: Feed/digest alternation | **High** | Medium | Enables geometric consolidation |
| 5A: Lower CV-shift threshold | **Quick** | Tiny | Revives triggered reflections |
| 5B: PE-based trigger | **Quick** | Small | Reflections at meaningful moments |
| 4: Curriculum stats | **Medium** | Medium | Observability (may already be partially done) |
| 1B: Tiered complexity | **Medium** | Small | Progressive difficulty |
| 1C: Curated topic fallback | **Medium** | Medium | Guaranteed diversity |
| 6: Consolidation metrics | **Low** | Medium | Measures digest phase effectiveness |

Suggested implementation order: 3 → 1A → 5A → 5B → 2 → 1B → 4 → 1C → 6

---

## Expected Outcomes

After implementing fixes 1-5 and running for ~500 new stimuli with feed/digest alternation:

| Metric | Current | Expected |
|--------|---------|----------|
| PE spread | 3.4% | **30-50%** |
| Unique topics per domain | ~5-8 (heavily repeated) | **30+** |
| CV per domain | 4.07-4.15 (uniform) | **Variable (3-7)** |
| Tau per domain | 1.06-1.07 (uniform) | **Variable (0.7-1.2)** |
| Triggered reflections | 1.7% | **15-25%** |
| Language drift | Present | **Eliminated** |
| Growth zone domains | 6 (meaningless) | **2-4 (meaningful)** |

The key test: after this improvement cycle, rerun the structural isomorphism test. If curriculum diversity produces richer geometric structure, the Mind should show stronger conceptual transfer — lower PE for structurally matching novel scenarios, higher PE for non-matching ones, with less dependence on surface vocabulary overlap.

---

## Files to Modify

| File | Changes |
|------|---------|
| `liquid_arc/mind.py` | Add `CurriculumScheduler`, topic tracking, anti-repetition prompts, PE-based trigger, language pinning on all generation prompts, lower CV-shift threshold |

One file. The curriculum history and scheduler state should persist across restarts (save/load with mind state).

---

## Verification

After deployment, run this quick check:

```
1. get_curriculum_stats() — PE spread should start climbing after ~50 diverse stimuli
2. get_routing_stats() — triggered_fraction should be >10%
3. get_reflection_log() — reflections should be in English only
4. get_context() — event buffer should show diverse topics, not repeated content
5. converse("What different topics have you learned about?") 
   — should list varied concepts, not the same 5 repeated
```
