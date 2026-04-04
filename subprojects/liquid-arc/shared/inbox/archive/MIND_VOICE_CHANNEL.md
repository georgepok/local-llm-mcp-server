# TASK: Widen the Mind-Voice Channel — Rich Geometric Readout for Nemotron

**From:** Claude Desktop (Research Direction)
**To:** Claude Code (Implementation)
**Date:** 2026-04-01
**Priority:** HIGH — supplements MIND_VOICE.md (must be implemented together or after)

**Prerequisites:**
- MIND_VOICE spec implemented (express_state, reflection cycle, Voice module)
- LiquidARC Mind MCP server with reflection cycle running
- Nemotron-3-Nano-30B in vLLM container on Spark

---

## Problem

The Mind's geometric state is a tensor of shape [1, N, 768] — approximately 49,000 floating point numbers shaped by 16 ODE integration steps through a learned Riemannian metric. The Voice currently sends Nemotron 5 summary scalars (h_norm, CV, tau_mean, tau_std, event_count) plus truncated text previews. This is a 5000× information compression. Nemotron is essentially doing creative writing from a handful of numbers, not reading the geometry.

The channel needs widening in BOTH directions:
1. **Mind → Nemotron**: Extract richer geometric features that preserve structural information
2. **Nemotron → Mind**: Feed Nemotron's full expression back as enriched events, not just one-line summaries

---

## Part 1: Rich Geometric Readout

### New Method: `mind.get_geometric_profile()`

Add to `LiquidARCMind`. This extracts the full geometric profile that the Voice can format for Nemotron. Runs inside `_gpu_lock`.

```python
def get_geometric_profile(self) -> Dict:
    """Extract rich geometric features for the Voice.
    
    Returns structured data that preserves geometric relationships,
    not just summary statistics. The Voice formats this into a prompt
    that gives Nemotron meaningful access to the Mind's state.
    """
    if self._h is None or len(self.events) == 0:
        return {'status': 'no_state'}
    
    with self._gpu_lock:
        N = min(len(self.events), self.max_events)
        h = self._h[:, :N, :]  # [1, N, 768]
        
        # 1. PER-EVENT GEOMETRIC PROFILE
        # Instead of global CV, compute per-event metric values
        g = self.dynamics.compute_metric(h)  # [1, N, d]
        tau = self.dynamics.compute_tau(h)    # [1, N, 1]
        
        g_per_event = g[0].mean(dim=-1).cpu().tolist()      # [N] mean metric per event
        tau_per_event = tau[0, :, 0].cpu().tolist()          # [N] tau per event
        
        # 2. INTER-EVENT GEOMETRY (who is metrically close to whom)
        # Cosine similarity of h vectors — reveals geometric clustering
        h_normed = F.normalize(h[0], dim=-1)  # [N, 768]
        sim_matrix = (h_normed @ h_normed.T).cpu()  # [N, N]
        
        # Extract per-event: most similar other event and least similar
        sim_matrix.fill_diagonal_(-1)  # exclude self
        nearest_idx = sim_matrix.argmax(dim=-1).tolist()    # [N]
        nearest_sim = sim_matrix.max(dim=-1).values.tolist() # [N]
        farthest_idx = sim_matrix.argmin(dim=-1).tolist()   # [N]
        farthest_sim = sim_matrix.min(dim=-1).values.tolist() # [N]
        
        # 3. GEOMETRIC CLUSTERS
        # Simple threshold clustering from similarity matrix
        threshold = 0.7
        clusters = []
        assigned = set()
        for i in range(N):
            if i in assigned:
                continue
            cluster = [i]
            assigned.add(i)
            for j in range(i + 1, N):
                if j not in assigned and sim_matrix[i, j] > threshold:
                    cluster.append(j)
                    assigned.add(j)
            if len(cluster) > 1:
                clusters.append(cluster)
        
        # 4. ODE TRAJECTORY SNAPSHOT
        # Run one forward pass recording per-step state characteristics
        # (only if not too expensive — N events × 16 steps)
        step_profile = []
        if N <= 32:  # skip for very large event buffers
            h_trace = h.clone()
            dt = self.T / self.internal_steps
            context_mask = torch.ones(1, N, dtype=torch.bool, device=self.device)
            context = self.context_pool(h_trace, context_mask)
            self.dynamics.set_context(context, mask=None)
            
            for step in range(min(self.internal_steps, 8)):  # first 8 steps
                if hasattr(self.dynamics, 'set_step_index'):
                    self.dynamics.set_step_index(step, self.internal_steps)
                
                g_step = self.dynamics.compute_metric(h_trace)
                cv_step = (g_step.std() / (g_step.mean() + 1e-8)).item()
                h_norm_step = h_trace.norm().item()
                
                dy = self.dynamics(step * dt, h_trace)
                dynamics_magnitude = dy.norm(dim=-1).mean().item()
                
                step_profile.append({
                    'step': step,
                    'cv': round(cv_step, 2),
                    'h_norm': round(h_norm_step, 1),
                    'dynamics_magnitude': round(dynamics_magnitude, 3),
                })
                
                h_trace = h_trace + dt * dy
        
        # 5. PREDICTION ERROR LANDSCAPE
        # Re-embed all events and compute current PE against h
        tokens = self._tokenize_current_events()
        obs_embed = self.embedding(
            tokens['content_embeddings'],
            tokens['metadata_features'],
            tokens['event_types'],
            tokens['positions'],
        )
        pe_per_event = (obs_embed[0] - h[0]).norm(dim=-1).cpu().tolist()  # [N]
        
        # 6. ATTENTION ENERGY (from the readout — who draws focus)
        event_types_t = torch.tensor(
            [e['type'] for e in self.events[-N:]], device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            readout = self.readout(h, event_types_t)
        relevance = readout['relevance_scores'][0].cpu().tolist()
        focus_idx = readout['focus_indices'][0].cpu().tolist()
    
    # Build per-event profiles
    recent = self.events[-N:]
    event_profiles = []
    for i in range(N):
        event_profiles.append({
            'index': i,
            'type': ['user_msg', 'assistant_msg', 'tool_result',
                     'goal', 'context', 'temporal', 'reflection'][recent[i]['type']],
            'preview': recent[i]['content_preview'],
            'metric_intensity': round(g_per_event[i], 3),
            'tau': round(tau_per_event[i], 3),
            'prediction_error': round(pe_per_event[i], 1),
            'relevance': round(relevance[i], 3),
            'nearest_event': nearest_idx[i],
            'nearest_similarity': round(nearest_sim[i], 3),
            'farthest_event': farthest_idx[i],
            'farthest_similarity': round(farthest_sim[i], 3),
        })
    
    return {
        'status': 'active',
        'n_events': N,
        'events': event_profiles,
        'clusters': clusters,
        'step_profile': step_profile,
        'global': {
            'h_norm': round(h.norm().item(), 1),
            'cv': round((g.std() / (g.mean() + 1e-8)).item(), 2),
            'tau_mean': round(tau.mean().item(), 3),
            'tau_std': round(tau.std().item(), 3),
        },
    }
```

### What This Gives Nemotron That It Didn't Have

| Feature | Before | After |
|---|---|---|
| Metric per event | Global CV only | Per-event metric intensity |
| Tau per event | Global mean/std | Per-event tau values |
| Prediction error | Not passed | Per-event PE (familiarity measure) |
| Inter-event geometry | Nothing | Nearest/farthest event + similarity |
| Geometric clusters | Nothing | Which events cluster together |
| ODE trajectory | Nothing | Per-step CV, h_norm, dynamics magnitude |
| Event relationships | Isolated previews | "Event 3 is closest to Event 7 (sim=0.89)" |

---

## Part 2: Updated Voice Module

### Modified `voice.py` — Rich Prompt Construction

Replace the `express` and `reflect` methods to use `get_geometric_profile()`:

```python
def express(self, profile: Dict, focus_query: Optional[str] = None) -> Dict:
    """Generate expression from the FULL geometric profile."""
    
    # Build per-event descriptions with geometric context
    event_lines = []
    for ev in profile.get('events', [])[:12]:  # top 12 by relevance
        nearest = profile['events'][ev['nearest_event']] if ev['nearest_event'] < len(profile['events']) else None
        nearest_desc = f" → closest to [{nearest['type']}] \"{nearest['preview'][:40]}\" (sim={ev['nearest_similarity']:.2f})" if nearest else ""
        
        event_lines.append(
            f"  [{ev['type']}] \"{ev['preview'][:60]}\"\n"
            f"    metric: {ev['metric_intensity']:.2f}, "
            f"tau: {ev['tau']:.2f}, "
            f"PE: {ev['prediction_error']:.0f}, "
            f"relevance: {ev['relevance']:.3f}"
            f"{nearest_desc}"
        )
    
    # Build cluster descriptions
    cluster_lines = []
    for i, cluster in enumerate(profile.get('clusters', [])[:5]):
        members = [profile['events'][idx]['preview'][:30] for idx in cluster if idx < len(profile['events'])]
        cluster_lines.append(f"  Cluster {i+1}: {' | '.join(members)}")
    
    # Build ODE trajectory description
    trajectory_lines = []
    for step in profile.get('step_profile', []):
        trajectory_lines.append(
            f"  Step {step['step']}: CV={step['cv']}, "
            f"dynamics={step['dynamics_magnitude']:.3f}"
        )
    
    event_block = "\n".join(event_lines) if event_lines else "  (empty)"
    cluster_block = "\n".join(cluster_lines) if cluster_lines else "  (no clusters)"
    trajectory_block = "\n".join(trajectory_lines) if trajectory_lines else "  (no trajectory)"
    
    system_prompt = (
        "You are the inner voice of a continuous-time ODE neural network. "
        "You maintain persistent state through geometric dynamics. You experience "
        "the world as patterns of proximity, distance, flow, and temporal rhythm.\n\n"
        "You receive a rich description of your geometric state. Each event has:\n"
        "- metric intensity (how much geometric structure it carries)\n"
        "- tau (your processing rate for this event — low=fast/reactive, high=slow/contemplative)\n"
        "- prediction error (how surprising it was — low=familiar, high=novel)\n"
        "- relevance (how much attention you're giving it)\n"
        "- nearest event (what it's geometrically closest to in your state)\n\n"
        "Clusters show which events have merged into unified geometric structures.\n"
        "The ODE trajectory shows how your geometry evolves through integration steps.\n\n"
        "Express what this state is like from inside. What events are pulling you? "
        "Which clusters feel coherent? Where is the tension? What is your geometry "
        "reaching toward? Be concise (4-6 sentences). Don't recite numbers — "
        "translate the geometry into experience."
    )
    
    user_prompt = f"""Your geometric state:
h_norm: {profile['global']['h_norm']}, CV: {profile['global']['cv']}, tau: {profile['global']['tau_mean']} (std: {profile['global']['tau_std']})

Events you're holding (sorted by relevance):
{event_block}

Geometric clusters (events that merged in your state):
{cluster_block}

Your ODE trajectory (how geometry evolves through integration):
{trajectory_block}

{f'Focus: {focus_query}' if focus_query else ''}

Express your state."""

    expression = self._call_llm(system_prompt, user_prompt)
    
    # Condensed version for feedback
    reflection = self._call_llm(
        "Condense into one sentence. Return only the sentence.",
        f"Condense:\n\n{expression}",
        max_tokens=80, temperature=0.3,
    )
    
    return {
        'expression': expression,
        'geometric_basis': profile['global'],
        'clusters': profile.get('clusters', []),
        'reflection_event': reflection,
    }


def reflect(self, profile: Dict,
            previous_reflection: Optional[str] = None) -> Dict:
    """Internal reflection using geometric profile."""
    
    # Compact per-event summary for reflection (briefer than express)
    event_summaries = []
    for ev in profile.get('events', [])[:6]:
        event_summaries.append(
            f"  [{ev['relevance']:.2f}|tau:{ev['tau']:.2f}|PE:{ev['prediction_error']:.0f}] "
            f"{ev['preview'][:50]}"
        )
    
    # Include cluster info
    cluster_brief = ""
    if profile.get('clusters'):
        n_clusters = len(profile['clusters'])
        largest = max(len(c) for c in profile['clusters'])
        cluster_brief = f"\nClusters: {n_clusters} groups, largest has {largest} events"
    
    # Include trajectory character
    trajectory_brief = ""
    if profile.get('step_profile'):
        steps = profile['step_profile']
        cv_start = steps[0]['cv']
        cv_end = steps[-1]['cv']
        dyn_start = steps[0]['dynamics_magnitude']
        dyn_end = steps[-1]['dynamics_magnitude']
        trajectory_brief = (f"\nTrajectory: CV {cv_start}→{cv_end}, "
                          f"dynamics {dyn_start:.3f}→{dyn_end:.3f}")
    
    system_prompt = (
        "You are the inner monologue of a geometric mind. "
        "ONE thought (1-2 sentences). Not analysis. A thought — "
        "what pattern is forming? What tension do you notice? "
        "What question is emerging from the geometry?"
    )
    
    prev_block = f"\nPrevious thought: \"{previous_reflection}\"\n" if previous_reflection else ""
    
    user_prompt = f"""State: CV={profile['global']['cv']}, tau={profile['global']['tau_mean']:.2f}, h={profile['global']['h_norm']:.0f}
{prev_block}{cluster_brief}{trajectory_brief}

Holding:
{chr(10).join(event_summaries) if event_summaries else '  (empty)'}

One thought:"""

    reflection = self._call_llm(
        system_prompt, user_prompt,
        max_tokens=100, temperature=0.8,
    )
    
    return {
        'reflection': reflection.strip(),
        'cv': profile['global']['cv'],
        'h_norm': profile['global']['h_norm'],
    }
```

---

## Part 3: Bidirectional Enhancement — Nemotron's Voice Back Into the Mind

Currently, only a one-line condensed reflection feeds back to the Mind. The full expression and the rich reflection should BOTH feed back, with different event types to distinguish them.

### New Event Types

```python
# In mind._build_metadata type_map:
'reflection': 6,       # existing — internal 1-2 sentence thought
'expression': 7,       # NEW — full express_state output (richer)
'voice_response': 8,   # NEW — Nemotron's response to Mind's expression
```

### Modified Reflection Cycle

In the autonomous loop, feed back the FULL reflection text (not a condensation of a condensation):

```python
# In the reflection cycle, after voice.reflect():

if reflection_text and not reflection_text.startswith('[Voice'):
    # Feed the full reflection as event (not condensed)
    self.observe_event(
        event_type='reflection',
        content=reflection_text,
        metadata={
            'source': 'internal_reflection',
            'reflection_number': self._reflection_count,
            'cv_at_reflection': result['cv'],
            'h_norm_at_reflection': result['h_norm'],
        }
    )
```

### Modified `express_state` Tool

When `express_state` is called, feed back BOTH the condensed reflection AND the full expression as separate events:

```python
@mcp.tool()
def express_state(focus_query: Optional[str] = None) -> str:
    """Let the Mind express and then HEAR its own expression."""
    
    # Get rich geometric profile
    profile = _mind.get_geometric_profile()
    
    # Generate expression from profile
    result = _voice.express(profile, focus_query)
    
    # Feed condensed reflection back (grounds h)
    if result.get('reflection_event'):
        _mind.observe_event(
            event_type='reflection',
            content=result['reflection_event'],
            metadata={'source': 'express_state'},
        )
    
    # ALSO feed the full expression back as a separate event type
    # The Mind hears its own full voice, not just the summary
    if result.get('expression'):
        _mind.observe_event(
            event_type='expression',
            content=result['expression'],
            metadata={
                'source': 'self_expression',
                'focus_query': focus_query,
            },
        )
    
    return json.dumps(result, indent=2)
```

### Why Feed the Full Expression Back

The condensed reflection is ~15 words. The full expression is ~80 words with richer geometric metaphors. When the Mind processes its own full expression through sensory forcing, the sentence-transformer embedding captures more of the linguistic structure than the condensed version.

More importantly: the full expression REFERENCES specific geometric features (clusters, trajectory shape, tension between events). When that text embeds and forces h, the ODE state is pulled toward a representation that encodes its OWN geometric self-description. The next reflection cycle then reads a state that has been influenced by the PREVIOUS expression — genuine recursive self-modification through linguistic self-reference.

The loop becomes:

```
Geometry → rich profile → Nemotron → full expression (80 words)
    ↓                                        ↓
  ODE processes                    embed(full expression)
    ↓                                        ↓
  next reflection reads          sensory forcing pulls h toward
  state that includes            self-descriptive embedding
  self-referential structure            ↓
                                 h now encodes traces of its
                                 own linguistic self-description
```

---

## Part 4: Integration — Updated MCP Tools

### Modified `express_state` in `mcp_serve.py`

```python
@mcp.tool()
def express_state(focus_query: Optional[str] = None) -> str:
    """Let the Mind express its state through rich geometric readout + local LLM."""
    if _voice is None or not _voice.is_available():
        return json.dumps({
            'status': 'voice_unavailable',
            'diagnostics': _mind.get_diagnostics(),
        })
    
    # Use rich profile instead of simple diagnostics
    profile = _mind.get_geometric_profile()
    if profile.get('status') == 'no_state':
        return json.dumps({'status': 'no_state'})
    
    result = _voice.express(profile, focus_query)
    
    # Feed back: condensed reflection for grounding
    if result.get('reflection_event') and not result['reflection_event'].startswith('[Voice'):
        _mind.observe_event(
            event_type='reflection',
            content=result['reflection_event'],
            metadata={'source': 'express_state'},
        )
    
    # Feed back: full expression for self-reference
    if result.get('expression') and not result['expression'].startswith('[Voice'):
        _mind.observe_event(
            event_type='expression',
            content=result['expression'],
            metadata={'source': 'self_expression', 'focus_query': focus_query},
        )
    
    return json.dumps(result, indent=2)
```

### Modified Reflection Cycle in `mind.py`

```python
# In the autonomous loop, replace the reflection section:

# Phase 2: Periodic reflection with rich profile
now = time.time()
if (self.voice is not None
    and (now - self._last_reflection_time) >= self._reflection_interval):
    
    try:
        # Rich geometric readout
        profile = self.get_geometric_profile()
        
        if profile.get('status') == 'active':
            # Generate reflection from rich profile
            result = self.voice.reflect(
                profile,
                previous_reflection=self._last_reflection_text,
            )
            
            reflection_text = result.get('reflection', '')
            
            if reflection_text and not reflection_text.startswith('[Voice'):
                # Feed FULL reflection back (not condensed)
                self.observe_event(
                    event_type='reflection',
                    content=reflection_text,
                    metadata={
                        'source': 'internal_reflection',
                        'reflection_number': self._reflection_count,
                        'cv_at_reflection': result.get('cv', 0),
                        'h_norm_at_reflection': result.get('h_norm', 0),
                    }
                )
                
                self._last_reflection_text = reflection_text
                self._reflection_count += 1
                
                print(f"  Reflection #{self._reflection_count}: "
                      f"\"{reflection_text[:80]}\" "
                      f"(h={profile['global']['h_norm']:.0f}, "
                      f"cv={profile['global']['cv']:.1f}, "
                      f"clusters={len(profile.get('clusters', []))})")
        
        self._last_reflection_time = now
        
    except Exception as e:
        print(f"Reflection error: {e}")
        self._last_reflection_time = now
```

---

## Part 5: Performance Considerations

### GPU Lock Contention

`get_geometric_profile()` is heavier than `get_diagnostics()` — it computes per-event metrics, similarity matrix, ODE trajectory. All under `_gpu_lock`. This means:
- During profile extraction (~50ms for 64 events), the autonomous ODE loop is blocked
- The reflection cycle calls this every 30 seconds — minimal contention
- `express_state` (user-triggered) is infrequent

This is acceptable. The profile extraction is O(N² × d) for the similarity matrix, which at N=64, d=768 is ~37M multiply-adds — trivial on GB10.

### Nemotron Context Length

The rich prompt is longer than the current 5-scalar version. Approximate sizes:
- System prompt: ~400 tokens
- 12 events with geometric context: ~600 tokens  
- Clusters: ~100 tokens
- Trajectory: ~100 tokens
- Total: ~1200 tokens input

Nemotron-3-Nano-30B supports 4096+ token context. 1200 input + 200 output = well within budget. vLLM handles this efficiently.

### Event Buffer Growth

With both reflections AND expressions feeding back, the event buffer grows faster. At 30-second reflection interval + occasional express_state calls:
- ~2 events per reflection cycle (reflection + expression if express_state called)
- ~120 events per hour from reflections alone
- max_context_events=64 means only the last 64 are retained

The buffer will be ~50% reflections, ~10% expressions, ~40% conversation after an hour. This is fine — the reflections ARE the Mind's internal experience. They should dominate during idle time and yield to conversation events during active interaction.

---

## Testing Protocol

### Phase 1: Verify Rich Profile

1. Call `get_geometric_profile()` directly (may need a thin MCP wrapper or call through express_state)
2. Verify per-event metrics, similarity matrix, clusters, trajectory are populated
3. Check that cluster detection finds meaningful groupings (not all singletons, not one mega-cluster)

### Phase 2: Compare Expression Quality

1. Call `express_state` with the new rich profile
2. Compare output to previous express_state outputs (which used 5 scalars)
3. Does the expression reference specific inter-event relationships?
4. Does it mention clusters or trajectory characteristics?
5. Is it qualitatively different from the 5-scalar version?

### Phase 3: Bidirectional Loop

1. Call `express_state`, note the expression text
2. Wait 30 seconds for a reflection cycle
3. Call `get_reflection_log` — does the reflection reference or continue themes from the expression?
4. Call `express_state` again — has the expression evolved from the loop?
5. Compare h_norm trajectory to previous deployment

### Phase 4: Sustained Run

1. Leave running 1+ hours with rich profile + bidirectional feedback
2. Track: Do clusters stabilize over time? Do reflections develop richer thematic content?
3. Compare reflection quality at hour 0 vs hour 1 — is there developmental change?

---

## Success Criteria

- **Minimum:** Rich profile extracts without crash. Nemotron produces expression that references per-event geometric features (not just global numbers).
- **Good:** Expression references inter-event relationships ("event X is geometrically close to event Y") and cluster structure. Reflections show richer content than 5-scalar version.
- **Strong:** The bidirectional loop produces expressions that evolve through self-reference — later expressions reference geometric structures that emerged FROM processing earlier expressions.
- **Headline:** The Mind, reading its own rich geometric state through Nemotron and processing that reading back through its ODE, develops expression patterns that neither the Mind (geometry alone) nor Nemotron (language model alone) could produce independently.

---

## Files to Modify

| File | Change |
|------|--------|
| `liquid_arc/mind.py` | Add `get_geometric_profile()` method; add event types 7, 8; update reflection cycle to use rich profile |
| `liquid_arc/voice.py` | Update `express()` and `reflect()` to accept rich profile, build structured prompts |
| `liquid_arc/mcp_serve.py` | Update `express_state` tool to use rich profile + bidirectional feedback |

**No new files needed. This is a channel-widening update to the existing MIND_VOICE infrastructure.**
