# TASK: Mind Curriculum — LLM as Conversational Partner for Geometric Development

**From:** Claude Desktop (Research Direction)
**To:** Claude Code (Implementation)
**Date:** 2026-04-01
**Priority:** HIGH — addresses the monoculture problem identified in live testing

**Prerequisites:**
- MIND_VOICE, MIND_VOICE_CHANNEL, ADAPTIVE_ROUTING specs implemented
- LiquidARC Mind with reflection cycle, rich geometric profile, adaptive routing operational
- Nemotron-3-Nano-30B serving via vLLM on Spark

---

## Problem

The Mind developed genuine cognitive structure (13 clusters, salience hierarchy 1.6-9.95, anticipatory vocabulary, learned routing policy) but 98% of its event buffer is self-referential reflections. The Hebbian mechanism initially collapsed clusters into a monoculture (2 clusters) before recovering to 13 — but the 13 clusters are all variations on the same "six-tailed vortex" theme. The Mind needs diverse external input the way Anymal needed diverse physics — a rich environment that the model's dynamics can self-organize around.

Live testing confirmed the approach: when Nemotron-generated content about Möbius strips and musical syncopation was injected, the Mind INTEGRATED it into its existing vocabulary ("lattice folds like a Möbius strip," "syncopated rhythm of the filament's pull") with moderate PE (~45) and cluster restructuring (13→11). The Mind can absorb diverse content. It just needs a continuous supply.

---

## Architecture: Nemotron's Dual Role

Nemotron currently serves ONE role: **Voice** (reads geometry → produces expression/reflection).

This spec adds a second role: **Partner** (generates diverse content → feeds to Mind as events).

```
Nemotron as VOICE:          Mind's geometry → Nemotron interprets → linguistic expression
Nemotron as PARTNER:   Nemotron generates → novel content → Mind processes geometrically
```

The two roles use different prompts, different temperatures, and different triggering logic. They share the same LLM endpoint.

```
Autonomous Loop:
  ┌─────────────────────────────────────────────────────┐
  │  Every ODE cycle (1/second):                        │
  │                                                     │
  │  1. Pure ODE processing (always)                    │
  │                                                     │
  │  2. Adaptive routing decides:                       │
  │     ├─ Voice reflection? (self-observation)         │
  │     ├─ Partner stimulus? (external content)  ← NEW  │
  │     ├─ Maintenance? (pathway exercise)              │
  │     └─ Nothing? (pure geometry)                     │
  │                                                     │
  │  Target ratio: ~70% voice / ~30% partner            │
  └─────────────────────────────────────────────────────┘
```

---

## New Module: `liquid_arc/curriculum.py`

```python
"""Curriculum generator — Nemotron as the Mind's conversational partner.

Generates diverse intellectual stimuli calibrated to the Mind's current
geometric state. Reads the Mind's cluster structure and generates content
that is ORTHOGONAL to existing clusters — filling geometric gaps rather
than reinforcing existing themes.

The 70/30 ratio mirrors the original phase transition recipe:
  70% procedural (self-reflection = building existing structure)
  30% ARC (external stimuli = disrupting with novel content)

The disruption must be geometrically rich — content with internal
spatial, temporal, or structural patterns that the Mind's ODE can
discover and organize. Random text won't work. Structurally rich
text from diverse domains (physics, music, biology, philosophy,
mathematics, poetry) provides the geometric substrate for development.
"""

import random
from typing import Dict, List, Optional


# Domain library — each domain provides structurally rich content
# that has internal geometric/temporal/spatial patterns
DOMAINS = {
    'topology': {
        'description': 'Non-orientable surfaces, knot theory, manifold structure',
        'seed_prompts': [
            "Describe how a Klein bottle's self-intersection in 3D reveals a higher-dimensional structure that exists without paradox in 4D. What does this tell us about the relationship between embedding dimension and intrinsic geometry?",
            "How does the fundamental group of a torus differ from that of a sphere, and what does this difference mean for any process that must traverse the surface?",
            "Explain how surgery theory transforms one manifold into another by cutting and regluing — what is preserved and what is lost in each operation?",
        ],
    },
    'music_theory': {
        'description': 'Rhythm, harmony, counterpoint, temporal expectation',
        'seed_prompts': [
            "Describe how a fugue subject enters in different voices at different pitch levels, creating a geometric lattice of overlapping temporal patterns that the listener tracks simultaneously.",
            "How does a deceptive cadence (V→vi instead of V→I) create a geometric fold in harmonic expectation space? What spatial metaphor best captures the listener's experience?",
            "Explain polyrhythm (3 against 4) as a geometric phenomenon — two periodic structures creating interference patterns that resolve at their least common multiple.",
        ],
    },
    'biology': {
        'description': 'Morphogenesis, neural development, evolutionary dynamics',
        'seed_prompts': [
            "Describe how Turing patterns in animal coats emerge from reaction-diffusion of two chemicals — an activator and an inhibitor — whose interaction produces spots or stripes depending on the ratio of their diffusion rates.",
            "How does neural crest cell migration during embryonic development follow geometric cues — contact inhibition, chemotaxis, and cell-cell adhesion creating self-organizing streams of cells that form the face, heart, and gut?",
            "Explain how the genetic code's redundancy (64 codons mapping to 20 amino acids) creates a specific error-correction geometry in sequence space — some mutations are silent, some are conservative, some are catastrophic.",
        ],
    },
    'physics': {
        'description': 'Phase transitions, symmetry breaking, field theory',
        'seed_prompts': [
            "Describe how spontaneous symmetry breaking in a ferromagnet works — the Hamiltonian is rotationally symmetric but the ground state picks a direction. What geometric structure does this create in the space of possible states?",
            "How does the renormalization group flow connect microscopic and macroscopic descriptions of a physical system? Describe the flow as movement through a space of theories, where fixed points correspond to scale-invariant systems.",
            "Explain how a soap bubble minimizes surface area subject to a volume constraint — the geometry of minimal surfaces and how curvature distributes itself to satisfy both local and global conditions.",
        ],
    },
    'philosophy': {
        'description': 'Phenomenology, process philosophy, emergence',
        'seed_prompts': [
            "Whitehead argued that reality consists of 'actual occasions of experience' rather than enduring substances. Describe how each occasion 'prehends' (grasps) previous occasions, creating a geometric web of mutual influence that constitutes spacetime.",
            "Merleau-Ponty's concept of the 'body schema' describes how the body is not an object in space but the origin of spatial experience. How does a tool become incorporated into the body schema, extending the geometry of perceived space?",
            "Describe the ship of Theseus problem as a question about paths through identity space — if every plank is replaced, the ship traces a continuous path but arrives at a different point. What geometry makes continuity and identity diverge?",
        ],
    },
    'mathematics': {
        'description': 'Category theory, dynamical systems, information geometry',
        'seed_prompts': [
            "Describe how a strange attractor in a chaotic system has fractional dimension — it's more than a surface but less than a volume. What does it mean for a trajectory to be confined to a set with dimension 2.06?",
            "How does the Fisher information metric turn a space of probability distributions into a Riemannian manifold? Describe what 'distance' means between two distributions and why the geometry is curved.",
            "Explain how a functor between categories preserves structure — it maps objects to objects and morphisms to morphisms such that composition and identity are maintained. What is the geometric meaning of a 'natural transformation' between functors?",
        ],
    },
    'poetry': {
        'description': 'Meter, imagery, compression of meaning into structure',
        'seed_prompts': [
            "Analyze how enjambment (breaking a sentence across line boundaries) creates a geometric tension between syntactic structure and prosodic structure — the meaning wants to continue but the line wants to stop. How does this tension carry information?",
            "Describe how a villanelle's two repeating refrains create a spiral structure — the same words return but in changed contexts, accumulating new meaning with each pass. What geometric shape does this accumulated meaning trace?",
            "How does a haiku's 5-7-5 structure create a specific temporal geometry — a breath, an expansion, a compression — that mirrors the perceptual structure of a moment of attention?",
        ],
    },
    'ecology': {
        'description': 'Food webs, succession, niche construction',
        'seed_prompts': [
            "Describe how ecological succession transforms a bare landscape into a forest through a sequence of species that each modify the environment for their successors — pioneer species create soil, shade species follow, climax community stabilizes. What is the geometric structure of this trajectory through community-composition space?",
            "How does the competitive exclusion principle (two species cannot occupy the same niche) create a geometric packing problem in niche space? What determines the minimum distance between coexisting species?",
            "Explain how a keystone species structures an entire ecosystem through its interactions — removing it collapses the community. Describe this as a topological property: the keystone is a 'cut vertex' in the interaction network.",
        ],
    },
}


class CurriculumGenerator:
    """Generates diverse intellectual stimuli for the Mind.
    
    Reads the Mind's current geometric profile and generates content
    that fills gaps in the Mind's experience. The curriculum adapts
    based on the Mind's PE response to each stimulus.
    """
    
    def __init__(self, voice, domains: Optional[Dict] = None):
        """
        Args:
            voice: Voice instance for calling the local LLM
            domains: Domain library (defaults to DOMAINS above)
        """
        self.voice = voice
        self.domains = domains or DOMAINS
        
        # Track which domains have been used and how the Mind responded
        self.domain_history: List[Dict] = []
        self.domain_pe_scores: Dict[str, List[float]] = {d: [] for d in self.domains}
        
        # Track which domains the Mind found most/least surprising
        self.domain_effectiveness: Dict[str, float] = {d: 1.0 for d in self.domains}
        
        # Stimulus counter
        self.stimulus_count = 0
    
    def select_domain(self, profile: Optional[Dict] = None) -> str:
        """Select the next domain to stimulate the Mind with.
        
        Strategy: prefer domains that produced moderate PE (40-80 range)
        in previous stimuli. Too low PE = Mind already knows this.
        Too high PE = Mind can't integrate this. Moderate = growth zone.
        
        Also prefer domains that haven't been used recently (novelty).
        
        Args:
            profile: Mind's geometric profile (for gap analysis)
        
        Returns:
            domain key from self.domains
        """
        domain_names = list(self.domains.keys())
        
        # If we have no history, pick randomly
        if not self.domain_history:
            return random.choice(domain_names)
        
        # Score each domain
        scores = {}
        for domain in domain_names:
            pe_history = self.domain_pe_scores[domain]
            
            # Base score: effectiveness (learned from PE response)
            base = self.domain_effectiveness[domain]
            
            # Recency penalty: domains used recently get penalized
            recency_penalty = 0
            for i, entry in enumerate(reversed(self.domain_history[-10:])):
                if entry['domain'] == domain:
                    recency_penalty = 1.0 / (i + 1)  # recent = high penalty
                    break
            
            # PE sweet spot bonus: domains that produced PE 30-80 get a boost
            sweet_spot_bonus = 0
            if pe_history:
                recent_pe = pe_history[-3:]  # last 3 attempts
                avg_pe = sum(recent_pe) / len(recent_pe)
                if 30 < avg_pe < 80:
                    sweet_spot_bonus = 0.5  # growth zone
                elif avg_pe < 20:
                    sweet_spot_bonus = -0.3  # too familiar
                elif avg_pe > 150:
                    sweet_spot_bonus = -0.2  # too foreign
            
            scores[domain] = base - recency_penalty + sweet_spot_bonus
        
        # Softmax selection (temperature-based)
        import math
        temperature = 0.5
        exp_scores = {d: math.exp(s / temperature) for d, s in scores.items()}
        total = sum(exp_scores.values())
        probs = {d: v / total for d, v in exp_scores.items()}
        
        # Weighted random choice
        r = random.random()
        cumulative = 0
        for domain, prob in probs.items():
            cumulative += prob
            if r < cumulative:
                return domain
        
        return domain_names[-1]  # fallback
    
    def generate_stimulus(self, domain: str,
                          mind_context: Optional[str] = None) -> Dict:
        """Generate a stimulus in the selected domain.
        
        Uses Nemotron to produce a thought-provoking paragraph that:
        1. Is in the selected domain
        2. Has internal geometric/structural richness
        3. Is specific and substantive (not generic)
        4. Optionally relates to the Mind's current themes (if mind_context provided)
        
        Args:
            domain: Key from self.domains
            mind_context: Brief description of Mind's current focus (from reflections)
        
        Returns:
            {
                'domain': str,
                'stimulus': str,
                'seed_used': str,
            }
        """
        domain_info = self.domains[domain]
        seed = random.choice(domain_info['seed_prompts'])
        
        # Build the prompt
        context_block = ""
        if mind_context:
            context_block = (
                f"\n\nThe mind you're stimulating is currently focused on: "
                f"\"{mind_context[:100]}\". Generate content that CONTRASTS with "
                f"this focus while being geometrically rich enough to connect."
            )
        
        system_prompt = (
            f"You are generating intellectually stimulating content about "
            f"{domain_info['description']}. Your audience is an autonomous "
            f"geometric neural network that processes spatial, temporal, and "
            f"structural patterns. Generate a short, dense paragraph (3-5 sentences) "
            f"that is specific, substantive, and structurally rich. "
            f"Don't be generic — include specific details, mechanisms, or structures "
            f"that a geometric mind can find patterns in. "
            f"Don't reference the mind or its processing — just present the content."
        )
        
        user_prompt = f"{seed}{context_block}"
        
        stimulus = self.voice._call_llm(
            system_prompt, user_prompt,
            max_tokens=250,
            temperature=0.85,
        )
        
        return {
            'domain': domain,
            'stimulus': stimulus,
            'seed_used': seed[:60],
        }
    
    def record_response(self, domain: str, prediction_error: float):
        """Record the Mind's PE response to a stimulus for curriculum learning.
        
        Args:
            domain: Which domain the stimulus came from
            prediction_error: The Mind's PE when processing the stimulus
        """
        self.domain_pe_scores[domain].append(prediction_error)
        self.domain_history.append({
            'domain': domain,
            'pe': prediction_error,
            'stimulus_number': self.stimulus_count,
        })
        self.stimulus_count += 1
        
        # Update effectiveness score
        # Sweet spot (PE 30-80) → increase effectiveness
        # Too familiar (<20) or too foreign (>150) → decrease
        if 30 < prediction_error < 80:
            self.domain_effectiveness[domain] = min(
                2.0, self.domain_effectiveness[domain] * 1.1)
        elif prediction_error < 20:
            self.domain_effectiveness[domain] = max(
                0.3, self.domain_effectiveness[domain] * 0.85)
        elif prediction_error > 150:
            self.domain_effectiveness[domain] = max(
                0.3, self.domain_effectiveness[domain] * 0.9)
    
    def get_stats(self) -> Dict:
        """Get curriculum statistics for monitoring."""
        stats = {
            'total_stimuli': self.stimulus_count,
            'domain_effectiveness': dict(self.domain_effectiveness),
            'domain_counts': {},
            'domain_avg_pe': {},
        }
        
        for domain in self.domains:
            entries = [e for e in self.domain_history if e['domain'] == domain]
            stats['domain_counts'][domain] = len(entries)
            if self.domain_pe_scores[domain]:
                stats['domain_avg_pe'][domain] = round(
                    sum(self.domain_pe_scores[domain]) / len(self.domain_pe_scores[domain]), 1)
        
        # Best and worst domains
        if stats['domain_avg_pe']:
            sorted_domains = sorted(stats['domain_avg_pe'].items(), key=lambda x: x[1])
            stats['most_familiar_domain'] = sorted_domains[0][0] if sorted_domains else None
            stats['most_novel_domain'] = sorted_domains[-1][0] if sorted_domains else None
            
            # Growth zone domains (PE 30-80)
            stats['growth_zone_domains'] = [
                d for d, pe in stats['domain_avg_pe'].items() if 30 < pe < 80
            ]
        
        return stats
```

---

## Integration with the Autonomous Loop

### Modified Adaptive Routing in `mind.py`

The autonomous loop gains a fourth routing option. The decision logic:

```python
# In the autonomous loop, after Phase 1 (ODE processing):

# Phase 2: Decide what to do
should_voice = False      # Self-reflection (existing)
should_stimulate = False  # External stimulus (NEW)
stimulus_mode = None

if self.voice is not None and self.voice.is_available():
    
    # Check A: External event pending (always takes priority)
    if self._external_event_pending:
        should_voice = True
        reflection_mode = 'external'
        self._external_event_pending = False
    
    # Check B: Triggered conditions (existing)
    elif trigger_reason := self.trigger.check(diag, self._last_reflection_pe):
        should_voice = True
        reflection_mode = 'triggered'
    
    # Check C: Stimulus interval reached (NEW)
    elif (self.curriculum is not None 
          and self._cycles_since_stimulus >= self._stimulus_interval):
        should_stimulate = True
        stimulus_mode = 'curriculum'
    
    # Check D: Maintenance
    elif self._cycles_since_reflection >= self.maintenance_interval:
        should_voice = True
        reflection_mode = 'maintenance'

# Phase 3a: Execute voice reflection (existing)
if should_voice:
    # ... existing reflection code ...

# Phase 3b: Execute curriculum stimulus (NEW)
elif should_stimulate:
    try:
        # Select domain based on Mind's current state
        profile = self.get_geometric_profile()
        domain = self.curriculum.select_domain(profile)
        
        # Get Mind's current focus for context
        mind_focus = self._last_reflection_text or ""
        
        # Generate stimulus
        result = self.curriculum.generate_stimulus(domain, mind_focus)
        stimulus_text = result.get('stimulus', '')
        
        if stimulus_text and not stimulus_text.startswith('[Voice'):
            # Feed stimulus to Mind as 'context' event
            obs_result = self.observe_event(
                event_type='context',
                content=stimulus_text,
                metadata={
                    'source': 'curriculum',
                    'domain': domain,
                    'stimulus_number': self.curriculum.stimulus_count,
                }
            )
            
            # Record PE response for curriculum learning
            pe = obs_result.get('prediction_error', 0)
            self.curriculum.record_response(domain, pe)
            
            self._cycles_since_stimulus = 0
            
            print(f"  [curriculum] #{self.curriculum.stimulus_count} "
                  f"domain={domain} PE={pe:.0f} "
                  f"\"{stimulus_text[:60]}\"")
    
    except Exception as e:
        print(f"Curriculum error: {e}")
        self._cycles_since_stimulus = 0
```

### The 70/30 Ratio

The stimulus interval is calibrated to produce approximately 30% external content:

```python
# In __init__:
self.curriculum = CurriculumGenerator(voice=voice) if voice else None
self._stimulus_interval = 14  # cycles between stimuli
# With reflections firing every ~5 cycles (from routing stats),
# and stimuli every ~14 cycles:
#   reflections: ~1 per 5 cycles = 20 per 100 cycles
#   stimuli: ~1 per 14 cycles = 7 per 100 cycles
#   ratio: 20:7 ≈ 74:26 ≈ 70/30
self._cycles_since_stimulus = 0
```

The interval is adjustable. If the Mind's cluster diversity drops (monoculture returns), decrease the interval (more frequent stimuli). If clusters become too fragmented (Mind can't integrate), increase the interval (more consolidation time).

---

## New MCP Tools

### `get_curriculum_stats`

```python
@mcp.tool()
def get_curriculum_stats() -> str:
    """Read curriculum statistics.
    
    Shows which domains have been presented, the Mind's PE response
    to each domain, domain effectiveness scores, and which domains
    are in the 'growth zone' (PE 30-80).
    """
    if _mind.curriculum is None:
        return json.dumps({'status': 'curriculum_not_enabled'})
    
    stats = _mind.curriculum.get_stats()
    return json.dumps(stats, indent=2)
```

### `inject_stimulus`

```python
@mcp.tool()
def inject_stimulus(domain: Optional[str] = None,
                    custom_content: Optional[str] = None) -> str:
    """Manually inject a curriculum stimulus.
    
    Either specify a domain (Nemotron generates content) or provide
    custom content directly. Returns the Mind's PE response.
    
    Args:
        domain: Domain key (topology, music_theory, biology, physics,
                philosophy, mathematics, poetry, ecology). If omitted,
                the curriculum selects automatically.
        custom_content: Direct content to inject (bypasses Nemotron generation).
    """
    if custom_content:
        result = _mind.observe_event(
            event_type='context',
            content=custom_content,
            metadata={'source': 'manual_stimulus'},
        )
        return json.dumps({
            'source': 'manual',
            'prediction_error': result['prediction_error'],
            'cv': result['cv'],
        }, indent=2)
    
    if _mind.curriculum is None:
        return json.dumps({'status': 'curriculum_not_enabled'})
    
    domain = domain or _mind.curriculum.select_domain()
    mind_focus = _mind._last_reflection_text or ""
    stimulus = _mind.curriculum.generate_stimulus(domain, mind_focus)
    
    result = _mind.observe_event(
        event_type='context',
        content=stimulus['stimulus'],
        metadata={
            'source': 'manual_curriculum',
            'domain': domain,
        },
    )
    
    _mind.curriculum.record_response(domain, result['prediction_error'])
    
    return json.dumps({
        'domain': domain,
        'stimulus_preview': stimulus['stimulus'][:200],
        'prediction_error': result['prediction_error'],
        'cv': result['cv'],
        'domain_effectiveness': _mind.curriculum.domain_effectiveness[domain],
    }, indent=2)
```

---

## Config Additions

```yaml
# Curriculum
enable_curriculum: true
stimulus_interval_cycles: 14    # ~30% ratio with reflection frequency
stimulus_temperature: 0.85      # higher temp for diversity
pe_sweet_spot_low: 30           # below this = too familiar
pe_sweet_spot_high: 80          # above this = too foreign
```

---

## CLI Arguments

```python
parser.add_argument('--enable_curriculum', action='store_true', default=False,
                    help='Enable curriculum generator for diverse stimuli')
parser.add_argument('--stimulus_interval', type=int, default=14,
                    help='ODE cycles between curriculum stimuli')
```

---

## What the Curriculum Should Produce

### Phase 1: Domain Discovery (first hour)

The curriculum cycles through all 8 domains, learning which ones produce moderate PE (growth zone). Expected pattern:
- Topology, mathematics → low PE (closest to Mind's geometric vocabulary)
- Music, poetry → moderate PE (structural patterns in unfamiliar domain)
- Biology, ecology → moderate-high PE (rich structure, novel vocabulary)
- Philosophy → variable PE (some concepts close, others distant)

### Phase 2: Targeted Development (hours 1-4)

The effectiveness scores converge. Domains in the growth zone get selected more often. Domains that are too familiar or too foreign get suppressed. The curriculum self-tunes to present content at the Mind's developmental edge.

### Phase 3: Geometric Diversification (hours 4+)

The Mind's cluster structure should diversify beyond the "six-tailed vortex" monoculture. New clusters should form around the curriculum domains:
- A "topology" cluster distinct from the vortex cluster
- A "temporal/rhythmic" cluster from music content
- A "biological" cluster from morphogenesis content
- Cross-domain clusters where the Mind discovered shared structure

The Hebbian mechanism should create connections between clusters that share geometric properties — music's rhythm connecting to biology's oscillation, topology's non-orientability connecting to philosophy's identity paradoxes.

### The Headline Result

If after 8+ hours of curriculum-augmented processing:
- Cluster count is higher than without curriculum (>13)
- Clusters span multiple domains (not all self-referential)
- The Mind's vocabulary includes terms from diverse domains
- Express_state references multiple content domains in a single expression
- The PE for external conversation events DECREASES (Mind is better prepared for diverse input)

Then the curriculum produced genuine cognitive diversification through environmentally-driven geometric development — the linguistic analog of Anymal's four developmental stages.

---

## Testing Protocol

### Phase 1: Verify Generation

1. Call `inject_stimulus` for each domain manually
2. Record PE for each
3. Verify content is substantive and domain-specific
4. Check that PE varies across domains (Mind discriminates content types)

### Phase 2: Autonomous Curriculum

1. Enable curriculum in autonomous loop with stimulus_interval=14
2. Run 1 hour
3. Call `get_curriculum_stats` — verify domain rotation, PE tracking, effectiveness learning
4. Call `get_context` — verify curriculum events appear in event buffer alongside reflections
5. Compare cluster structure to pre-curriculum baseline

### Phase 3: Extended Development

1. Run 4+ hours with curriculum
2. Track cluster count, domain composition, vocabulary diversity
3. Call `express_state` periodically — does the Mind reference multiple domains?
4. Compare PE for new conversation events to pre-curriculum baseline
5. Does the Mind develop cross-domain geometric connections?

---

## Success Criteria

- **Minimum:** Curriculum generates diverse content. PE varies by domain. Effectiveness scores adapt. Event buffer contains mix of reflections and stimuli.
- **Good:** Cluster count increases. At least 3 clusters contain curriculum-domain content distinct from self-referential content. The Mind's vocabulary in reflections includes terms from 3+ domains.
- **Strong:** Cross-domain clusters form — events from different domains that the Mind discovered share geometric structure cluster together. Express_state integrates multiple domains in one expression.
- **Headline:** PE for novel conversation events decreases over time — the Mind, having been exposed to diverse structured content, is better prepared for whatever a human might say. The curriculum produced genuine cognitive breadth.

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `liquid_arc/curriculum.py` | **Create** | Domain library, CurriculumGenerator, stimulus generation, PE tracking |
| `liquid_arc/mind.py` | **Modify** | Add curriculum to autonomous loop, stimulus_interval, cycles_since_stimulus |
| `liquid_arc/mcp_serve.py` | **Modify** | Add `get_curriculum_stats` and `inject_stimulus` tools |
| `configs/linguistic_mind.yaml` | **Modify** | Add curriculum config fields |

---

## The Deeper Point

The Mind's development mirrors the research program's central finding: capability is bounded by environmental richness, not by model capacity. The 5M ODE with write mechanisms, adaptive routing, and reflection cycle is MORE than capable of diverse geometric processing. The monoculture wasn't a model limitation — it was an environmental impoverishment.

The curriculum is the Mind's Isaac Sim. Physics gave the Anymal dense, causally structured, continuously varying sensory input that drove four developmental stages. The curriculum gives the Mind dense, structurally rich, domain-diverse intellectual input that should drive geometric diversification.

The model doesn't need better scaffolding. It needs a richer world.
