# TASK: Curriculum Instrumentation & Language Pinning

**From:** Claude Desktop (Research Direction)
**To:** Claude Code (Implementation)
**Date:** 2026-04-04
**Priority:** MEDIUM — observability and quality fix for the deployed system

---

## Problem

The Qwen3 curriculum feed is running (confirmed from logs and event buffer contents). But:

1. **No instrumentation.** The `get_curriculum_stats` MCP tool returns `curriculum_not_enabled` because the Qwen3-based curriculum bypasses the old Nemotron-era stats tracking. We can't measure per-domain PE, domain effectiveness, or growth zone identification — the key indicators of whether curriculum training is producing geometric differentiation.

2. **Language drift.** The event buffer contains Chinese (信息几何, 中华文明起源) and French (émergence) content. Qwen3-4B supports 119 languages and without explicit pinning, the curriculum and reflection generators occasionally produce non-English output. This contaminates LiquidARC's geometric state with multilingual representations the coupling wasn't trained for.

---

## Fix 1: Wire Curriculum Stats to the Qwen3 Pathway

The autonomous loop generates curriculum stimuli through Qwen3 via the OUTBOUND pipeline (h(t) → prefix → Qwen3 generates stimulus). Each stimulus has a domain label from the cycling domain list. The stats tracking needs to tap into this pipeline.

### What to Log Per Curriculum Event

After each curriculum stimulus is generated and fed back through the INBOUND pipeline:

```python
# In the curriculum generation section of the autonomous loop:

domain = DOMAIN_CYCLE[current_domain_index]

# Generate stimulus through Qwen3 coupling (already happening)
stimulus_text = generate_curriculum(domain)

# Feed back through INBOUND (already happening)
result = _force_geometric_signal(stimulus_text, event_type='context')

# NEW: Record curriculum stats
curriculum_stats['domain_counts'][domain] += 1
curriculum_stats['domain_pe_history'][domain].append(result['prediction_error'])
curriculum_stats['domain_cv_history'][domain].append(get_current_cv())
curriculum_stats['domain_tau_history'][domain].append(get_current_tau_mean())

# Compute running averages
curriculum_stats['domain_avg_pe'][domain] = (
    sum(curriculum_stats['domain_pe_history'][domain][-50:]) / 
    min(50, len(curriculum_stats['domain_pe_history'][domain]))
)

# Domain effectiveness: inversely proportional to PE (lower PE = more familiar = more effective absorption)
# But too-low PE means no learning. Optimal zone: moderate PE.
all_pe = [curriculum_stats['domain_avg_pe'][d] for d in curriculum_stats['domain_avg_pe']]
if all_pe:
    pe_min, pe_max = min(all_pe), max(all_pe)
    pe_range = pe_max - pe_min if pe_max > pe_min else 1.0
    for d in curriculum_stats['domain_avg_pe']:
        # Effectiveness peaks at moderate PE (growth zone)
        normalized_pe = (curriculum_stats['domain_avg_pe'][d] - pe_min) / pe_range
        # Bell curve: max effectiveness at 0.4 normalized PE
        curriculum_stats['domain_effectiveness'][d] = (
            math.exp(-((normalized_pe - 0.4) ** 2) / 0.1)
        )

# Identify growth zone domains (moderate PE, not saturated, not too novel)
curriculum_stats['growth_zone_domains'] = [
    d for d in curriculum_stats['domain_avg_pe']
    if 0.25 < ((curriculum_stats['domain_avg_pe'][d] - pe_min) / pe_range) < 0.65
]
```

### Update `get_curriculum_stats` MCP Tool

The tool should return the accumulated stats regardless of which pathway (Nemotron or Qwen3) generates the curriculum:

```python
@mcp_tool
def get_curriculum_stats():
    return {
        'total_stimuli': sum(curriculum_stats['domain_counts'].values()),
        'domain_counts': curriculum_stats['domain_counts'],
        'domain_avg_pe': {d: round(v, 1) for d, v in curriculum_stats['domain_avg_pe'].items()},
        'domain_avg_cv': {d: round(v, 2) for d, v in curriculum_stats.get('domain_avg_cv', {}).items()},
        'domain_avg_tau': {d: round(v, 2) for d, v in curriculum_stats.get('domain_avg_tau', {}).items()},
        'domain_effectiveness': {d: round(v, 2) for d, v in curriculum_stats['domain_effectiveness'].items()},
        'most_familiar_domain': min(curriculum_stats['domain_avg_pe'], key=curriculum_stats['domain_avg_pe'].get),
        'most_novel_domain': max(curriculum_stats['domain_avg_pe'], key=curriculum_stats['domain_avg_pe'].get),
        'growth_zone_domains': curriculum_stats['growth_zone_domains'],
        'pe_spread_pct': round(100 * (pe_max - pe_min) / pe_max, 1) if pe_max > 0 else 0,
    }
```

The new field `pe_spread_pct` is the headline progress indicator — it should grow from ~14% (current undifferentiated state) to 50%+ as domains differentiate.

### Also Track Per-Domain Geometric Profiles

Add to the stats:

```python
# After each curriculum event
curriculum_stats['domain_cv_history'][domain].append(current_cv)
curriculum_stats['domain_tau_history'][domain].append(current_tau_mean)

# In get_curriculum_stats, compute averages
'domain_avg_cv': {d: mean(last_50) for d in domains},
'domain_avg_tau': {d: mean(last_50) for d in domains},
```

This is what tells us if the geometry is differentiating per domain (the key progress signal). If physics consistently produces CV=7.2 and poetry produces CV=4.1, the MetricNet is applying different routing for different domains.

---

## Fix 2: Language Pinning

### Curriculum Generation Prompts

The curriculum prompts that go through Qwen3's OUTBOUND pipeline need explicit language pinning. In the curriculum generation section:

```python
CURRICULUM_PROMPTS = {
    'topology': "Explain a concept from algebraic topology in clear English. Respond only in English.",
    'music_theory': "Explain a concept from music theory in clear English. Respond only in English.",
    'biology': "Explain a concept from developmental biology in clear English. Respond only in English.",
    'physics': "Explain a concept from physics in clear English. Respond only in English.",
    'philosophy': "Explain a philosophical concept in clear English. Respond only in English.",
    'mathematics': "Explain a mathematical concept in clear English. Respond only in English.",
    'poetry': "Explain a concept from poetry or poetics in clear English. Respond only in English.",
    'ecology': "Explain an ecological concept in clear English. Respond only in English.",
}
```

The key addition: **"Respond only in English"** at the end of every prompt. This is a strong instruction for the instruct model's chat template. Without it, Qwen3's multilingual training sometimes produces non-English outputs, especially when the geometric prefix (from prior multilingual events) biases the model toward other languages.

### Reflection Prompts

Same fix for the reflection generation:

```python
REFLECTION_PROMPT = (
    "What patterns, connections, or shifts do you notice "
    "in your current state? Respond in English only. "
    "Keep your response concise — one paragraph."
)
```

### System-Level Pin (if chat template supports it)

If the chat template supports a system message, add:

```python
system_message = "You are a scientific assistant. Always respond in English."
```

This provides a stronger pin than per-prompt instructions because it stays in the system role across all generations.

---

## Fix 3: Console Logging for Real-Time Monitoring

Add a periodic console log (every 10 curriculum events) that prints the domain differentiation state:

```python
if curriculum_stats['total_stimuli'] % 10 == 0:
    print(f"  [curriculum] stimuli={curriculum_stats['total_stimuli']}")
    for d in sorted(curriculum_stats['domain_avg_pe'].keys()):
        pe = curriculum_stats['domain_avg_pe'].get(d, 0)
        cv = curriculum_stats.get('domain_avg_cv', {}).get(d, 0)
        tau = curriculum_stats.get('domain_avg_tau', {}).get(d, 0)
        count = curriculum_stats['domain_counts'].get(d, 0)
        print(f"    {d:15s}: PE={pe:6.1f} CV={cv:5.2f} tau={tau:4.2f} n={count}")
    pe_vals = list(curriculum_stats['domain_avg_pe'].values())
    if pe_vals:
        spread = 100 * (max(pe_vals) - min(pe_vals)) / max(pe_vals)
        print(f"  [curriculum] PE spread: {spread:.1f}%")
        if curriculum_stats['growth_zone_domains']:
            print(f"  [curriculum] growth zones: {curriculum_stats['growth_zone_domains']}")
```

This gives live visibility into whether domains are differentiating without needing to call MCP tools.

---

## What Success Looks Like

After these fixes, `get_curriculum_stats` returns data like:

```json
{
  "total_stimuli": 400,
  "domain_counts": {"topology": 50, "music_theory": 50, ...},
  "domain_avg_pe": {"topology": 280, "music_theory": 420, "physics": 310, ...},
  "domain_avg_cv": {"topology": 6.8, "music_theory": 3.2, "physics": 5.5, ...},
  "domain_avg_tau": {"topology": 0.72, "music_theory": 1.15, "physics": 0.88, ...},
  "domain_effectiveness": {"topology": 0.85, "music_theory": 0.35, ...},
  "most_familiar_domain": "topology",
  "most_novel_domain": "music_theory",
  "growth_zone_domains": ["physics", "ecology", "philosophy"],
  "pe_spread_pct": 33.3
}
```

This tells us: topology has become familiar (low PE), music theory is still novel (high PE), physics is in the growth zone. The geometry differentiates per domain (different CV and tau profiles). The PE spread is widening from baseline.

Without these instruments, we're running the curriculum blind. With them, we can track every progress indicator identified in the research discussion.

---

## Files to Modify

| File | Change |
|------|--------|
| `liquid_arc/mind.py` | Add curriculum stats tracking after each Qwen3 curriculum event. Update `get_curriculum_stats()` to return the new fields. Add "Respond only in English" to curriculum and reflection prompts. Add console logging every 10 events. |

One file. The changes are additive — no existing functionality needs to change, just new instrumentation tapped into the existing Qwen3 curriculum pipeline.
