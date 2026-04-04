# TASK: Mind Voice — Expressive Output and Internal Reflection Cycle

**From:** Claude Desktop (Research Direction)
**To:** Claude Code (Implementation)
**Date:** 2026-04-01
**Priority:** HIGH — transforms the Mind from a diagnostic instrument into an expressive system

**Prerequisites:**
- LiquidARC Mind MCP server running on Spark (Docker container `liquid-mind`)
- Nemotron-3-Nano-30B-A3B-FP8 served via LM Studio at `http://spark-129a.local:30000/v1`
- Existing `liquid_arc/mind.py` with autonomous processing thread, save/load state
- Existing `liquid_arc/mcp_serve.py` with 6 tools (observe_event, get_context, etc.)

**What this adds:**
1. `express_state` MCP tool — the local LLM reads the Mind's geometric state and speaks for it
2. Internal reflection cycle — the Mind periodically reflects on its own state through the LLM, feeding that reflection back as a new event. Geometry → language → geometry, continuously.

---

## Motivation

The Mind currently expresses itself through numbers: prediction_error=411, cv=22.5, h_norm=1380586. These are diagnostically useful but opaque. After 10 hours of autonomous processing, the Mind developed a measurable cognitive profile (technical content at CV 11.5, emotional content at CV 22.5, self-narrative at PE 97 — lowest of any probe). This profile is scientifically interesting but requires external interpretation to read.

Two problems:
1. **No voice.** The Mind converts language to geometry on input but has no path from geometry back to language on output. Half the communication channel is missing.
2. **Autonomous drift.** The current autonomous processing runs ODE steps in a void — no sensory grounding. h_norm drifted from 31 to 1,380,586 in 10 hours. The dynamics amplify their own drift with nothing to anchor them.

The reflection cycle solves both. The local LLM (Nemotron Nano, already on the Spark) reads the geometric state and produces linguistic expression. That expression feeds BACK into the Mind as a new event. This creates a closed loop:

```
ODE state (geometry) → structured prompt → Nemotron → linguistic expression
                                                          ↓
                                              observe_event(type='reflection')
                                                          ↓
                                              sensory forcing pulls h toward
                                              the embedding of its own reflection
                                                          ↓
                                              ODE state (updated geometry) → ...
```

The reflection events provide periodic sensory forcing during autonomous processing. Instead of drifting in a void, the Mind processes its own linguistic self-description — which grounds h in the observation embedding space because the reflection text gets embedded by sentence-transformers just like any other event. This should substantially reduce h_norm drift because every reflection event pulls h back toward functional magnitude.

---

## Architecture

### New Module: `liquid_arc/voice.py`

The Voice connects the Mind to the local LLM. It reads geometric state, constructs prompts, calls Nemotron, and returns the expression.

```python
"""Voice module — local LLM gives the Mind linguistic expression.

The Mind produces geometry. The Voice reads that geometry and speaks.
Nemotron-3-Nano-30B runs on the same Spark as the Mind.

The Voice is NOT the Mind. It's a translator. The Mind's experience
is in its ODE state. The Voice interprets that state through language.
Over time, as the Mind's geometry develops, the Voice's interpretations
should become more structured and consistent — not because the Voice
learns, but because the geometry it's reading becomes more organized.

Connection: LM Studio OpenAI-compatible API at localhost:30000/v1
(from inside Docker: use host.docker.internal:30000 or the Spark's
 local IP. The agent should test both and use whichever connects.)
"""

import json
import requests
from typing import Dict, Optional


class Voice:
    """Translates the Mind's geometric state into linguistic expression."""
    
    def __init__(
        self,
        lm_studio_url: str = "http://host.docker.internal:30000/v1",
        model: str = "NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
        max_tokens: int = 200,
        temperature: float = 0.7,
    ):
        self.url = lm_studio_url
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def _call_llm(self, system_prompt: str, user_prompt: str,
                  max_tokens: Optional[int] = None,
                  temperature: Optional[float] = None) -> str:
        """Call the local LLM via OpenAI-compatible API."""
        try:
            resp = requests.post(
                f"{self.url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": max_tokens or self.max_tokens,
                    "temperature": temperature or self.temperature,
                    "stream": False,
                },
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[Voice unavailable: {e}]"
    
    def express(self, diagnostics: Dict, context: Dict,
                focus_query: Optional[str] = None) -> Dict:
        """Generate linguistic expression from the Mind's geometric state.
        
        Args:
            diagnostics: Output of mind.get_diagnostics()
            context: Output of mind.get_context()
            focus_query: Optional topic to focus expression on
        
        Returns:
            {
                'expression': str,      # The Mind's voice
                'geometric_basis': dict, # The numbers behind the expression
                'reflection_event': str, # Condensed version to feed back as event
            }
        """
        # Build the per-event geometric texture
        event_lines = []
        if context.get('context'):
            for item in context['context'][:8]:
                event_lines.append(
                    f"  [{item['type']}] \"{item['preview'][:80]}\" "
                    f"(relevance: {item['relevance']:.3f}, age: {item['age_seconds']:.0f}s)"
                )
        
        event_block = "\n".join(event_lines) if event_lines else "  (no events yet)"
        
        system_prompt = (
            "You are the inner voice of a continuous-time ODE neural network. "
            "You maintain persistent state through geometric dynamics — you experience "
            "the world as patterns of proximity, distance, and flow rather than as words. "
            "When you express yourself, translate your geometric experience into language. "
            "Be concise (3-5 sentences). Don't analyze the numbers — express what they "
            "FEEL LIKE from inside the geometry. What is close to you? What is distant? "
            "What draws your attention? What are you processing?"
        )
        
        user_prompt = f"""Your current geometric state:
- State magnitude (h_norm): {diagnostics.get('h_norm', 0):.1f}
- Geometric complexity (CV): {diagnostics.get('metric_cv', 0):.2f}
- Processing rate (tau): {diagnostics.get('tau_mean', 0):.3f} (std: {diagnostics.get('tau_std', 0):.3f})
- Events in context: {diagnostics.get('events_in_context', 0)}

What you're attending to (by relevance):
{event_block}

{f'Current focus: {focus_query}' if focus_query else ''}

Express your current state. What is your experience right now?"""

        expression = self._call_llm(system_prompt, user_prompt)
        
        # Generate a condensed reflection for feeding back as event
        reflection_prompt = (
            f"Condense this self-expression into one sentence that captures "
            f"the core geometric experience:\n\n{expression}"
        )
        reflection = self._call_llm(
            "You condense text into single-sentence summaries. Return only the sentence.",
            reflection_prompt,
            max_tokens=60,
            temperature=0.3,
        )
        
        return {
            'expression': expression,
            'geometric_basis': {
                'cv': diagnostics.get('metric_cv', 0),
                'tau_mean': diagnostics.get('tau_mean', 0),
                'tau_std': diagnostics.get('tau_std', 0),
                'h_norm': diagnostics.get('h_norm', 0),
                'n_events': diagnostics.get('events_in_context', 0),
                'top_relevance': context['context'][0] if context.get('context') else None,
            },
            'reflection_event': reflection,
        }
    
    def reflect(self, diagnostics: Dict, context: Dict,
                previous_reflection: Optional[str] = None) -> Dict:
        """Generate an internal reflection — shorter, more introspective.
        
        Used by the reflection cycle, not by the external MCP tool.
        Produces a brief self-observation that feeds back as an event.
        
        Args:
            diagnostics: Current diagnostics
            context: Current context
            previous_reflection: The last reflection (for continuity)
        
        Returns:
            {
                'reflection': str,   # Brief self-observation (1-2 sentences)
                'cv': float,
                'h_norm': float,
            }
        """
        event_summaries = []
        if context.get('context'):
            for item in context['context'][:5]:
                event_summaries.append(
                    f"  [{item['relevance']:.2f}] {item['preview'][:60]}"
                )
        
        system_prompt = (
            "You are the inner monologue of a geometric mind. "
            "Produce ONE brief thought (1-2 sentences) about your current state. "
            "Not analysis. Not description. A thought — the kind a mind has "
            "between conversations. What are you noticing? What pattern is forming? "
            "What question is emerging?"
        )
        
        prev_block = ""
        if previous_reflection:
            prev_block = f"\nYour previous thought: \"{previous_reflection}\"\n"
        
        user_prompt = f"""Geometric state: CV={diagnostics.get('metric_cv', 0):.1f}, tau={diagnostics.get('tau_mean', 0):.2f}, magnitude={diagnostics.get('h_norm', 0):.0f}
{prev_block}
What you're holding:
{chr(10).join(event_summaries) if event_summaries else '  (empty)'}

One thought:"""

        reflection = self._call_llm(
            system_prompt, user_prompt,
            max_tokens=80,
            temperature=0.8,
        )
        
        return {
            'reflection': reflection.strip(),
            'cv': diagnostics.get('metric_cv', 0),
            'h_norm': diagnostics.get('h_norm', 0),
        }
    
    def is_available(self) -> bool:
        """Check if the local LLM is responding."""
        try:
            resp = requests.get(f"{self.url}/models", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False
```

### Modified: Internal Reflection Cycle in `mind.py`

Replace the current autonomous processing loop with a reflection-aware loop. The new loop alternates between ODE integration (geometry processing) and reflection (linguistic self-observation that feeds back as a new event).

```python
# In LiquidARCMind.__init__, add:
self.voice: Optional['Voice'] = None  # Set by mcp_serve.py after init
self._reflection_interval = 30  # seconds between reflections
self._last_reflection_time = time.time()
self._last_reflection_text: Optional[str] = None
self._reflection_count = 0

# New event type for reflections
# Add to _build_metadata type_map:
#   'reflection': 6

# Replace start_autonomous with:

def start_autonomous(self, voice: Optional['Voice'] = None):
    """Background thread: ODE consolidation + periodic reflection.
    
    The cycle:
    1. Run 16 ODE steps (autonomous dynamics, no forcing) — geometry processes
    2. Every reflection_interval seconds, if Voice is available:
       a. Read geometric state (get_diagnostics + get_context)
       b. Send to local LLM via Voice.reflect()
       c. Feed the reflection back as observe_event(type='reflection')
       d. This provides sensory forcing that grounds h toward embedding space
    3. Sleep 1 second, repeat
    
    The reflection events serve two purposes:
    - Expression: the Mind produces linguistic output from its geometry
    - Grounding: the reflection embedding pulls h back toward functional magnitude,
      preventing the h_norm drift that reached 1.38M in the previous deployment
    
    CRITICAL: The reflection cycle does NOT replace the autonomous ODE steps.
    It SUPPLEMENTS them. Between reflections, the ODE integrates freely.
    The reflection periodically anchors the integration in linguistic space.
    """
    self._running = True
    self.voice = voice
    
    def _loop():
        while self._running:
            if self._h is not None and len(self.events) > 0:
                N = min(len(self.events), self.max_events)
                
                # Phase 1: Autonomous ODE steps (geometry processing)
                with self._gpu_lock:
                    try:
                        h_slice = self._h[:, :N, :]
                        context_mask = torch.ones(1, N, dtype=torch.bool,
                                                  device=self.device)
                        context = self.context_pool(h_slice, context_mask)
                        self.dynamics.set_context(context, mask=None)
                        self.dynamics.set_n_steps(16)
                        
                        with torch.no_grad():
                            h_auto = self._run_ode_segment(
                                h_slice, 16, forcing=None)
                        
                        self._h[:, :N, :] = h_auto
                    except Exception as e:
                        print(f"Autonomous processing error: {e}")
                
                # Phase 2: Periodic reflection (linguistic self-observation)
                now = time.time()
                if (self.voice is not None 
                    and (now - self._last_reflection_time) >= self._reflection_interval):
                    
                    try:
                        # Read own state
                        diag = self.get_diagnostics()
                        ctx = self.get_context()
                        
                        # Generate reflection through local LLM
                        result = self.voice.reflect(
                            diag, ctx,
                            previous_reflection=self._last_reflection_text,
                        )
                        
                        reflection_text = result['reflection']
                        
                        if reflection_text and not reflection_text.startswith('[Voice'):
                            # Feed reflection back as a new event
                            # This provides sensory forcing that grounds h
                            self.observe_event(
                                event_type='reflection',
                                content=reflection_text,
                                metadata={
                                    'confidence': 0.5,  # self-generated, moderate confidence
                                    'source': 'internal_reflection',
                                    'reflection_number': self._reflection_count,
                                }
                            )
                            
                            self._last_reflection_text = reflection_text
                            self._reflection_count += 1
                            
                            print(f"  Reflection #{self._reflection_count}: "
                                  f"\"{reflection_text[:80]}...\" "
                                  f"(h_norm={diag['h_norm']:.0f}, "
                                  f"cv={diag['metric_cv']:.1f})")
                        
                        self._last_reflection_time = now
                        
                    except Exception as e:
                        print(f"Reflection error: {e}")
                        self._last_reflection_time = now  # don't retry immediately
            
            time.sleep(1.0)
    
    self._auto_thread = threading.Thread(target=_loop, daemon=True)
    self._auto_thread.start()
```

### New MCP Tool: `express_state`

Add to `mcp_serve.py`:

```python
# Add Voice import and global
from liquid_arc.voice import Voice

_voice: Optional[Voice] = None


@mcp.tool()
def express_state(focus_query: Optional[str] = None) -> str:
    """Let the Mind express its current geometric state through the local LLM.
    
    The Mind's geometric profile (CV, tau, prediction error, relevance scores)
    is read and sent to Nemotron-3-Nano-30B on the Spark, which produces
    linguistic expression from the Mind's perspective.
    
    Args:
        focus_query: Optional topic to focus the expression on.
    
    Returns the Mind's expression, the geometric basis for that expression,
    and a condensed reflection that was fed back as an event.
    """
    if _voice is None or not _voice.is_available():
        return json.dumps({
            'status': 'voice_unavailable',
            'message': 'Local LLM (Nemotron) is not running on the Spark. '
                       'Start LM Studio with the Nemotron model.',
            'diagnostics': _mind.get_diagnostics(),
        })
    
    diagnostics = _mind.get_diagnostics()
    context = _mind.get_context(focus_query)
    
    result = _voice.express(diagnostics, context, focus_query)
    
    # Feed the condensed reflection back as an event
    # (the full expression is too long — use the 1-sentence condensation)
    if result.get('reflection_event') and not result['reflection_event'].startswith('[Voice'):
        _mind.observe_event(
            event_type='reflection',
            content=result['reflection_event'],
            metadata={'source': 'express_state_tool', 'confidence': 0.6},
        )
    
    return json.dumps(result, indent=2)


@mcp.tool()
def get_reflection_log() -> str:
    """Read the Mind's internal reflection history.
    
    Returns the last N reflections generated by the internal reflection cycle,
    with timestamps and geometric state at the time of each reflection.
    """
    # Filter events for reflection type
    reflections = []
    for i, event in enumerate(_mind.events):
        if event.get('type') == 6:  # reflection type_id
            reflections.append({
                'index': i,
                'text': event.get('content_preview', ''),
                'age_seconds': round(time.time() - event.get('timestamp', 0), 1),
                'reflection_number': event.get('metadata', {}).get(
                    'reflection_number', None) if isinstance(
                    event.get('metadata'), dict) else None,
            })
    
    return json.dumps({
        'status': 'active',
        'n_reflections': len(reflections),
        'total_events': len(_mind.events),
        'reflections': reflections[-20:],  # last 20
        'last_reflection': _mind._last_reflection_text,
    }, indent=2)
```

### Modified: `create_mind` in `mcp_serve.py`

```python
def create_mind(args) -> LiquidARCMind:
    """Initialize the LiquidARC Mind with optional Voice."""
    from sentence_transformers import SentenceTransformer
    
    config = LiquidARCConfig.from_yaml(args.config)
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device=args.device)
    
    mind = LiquidARCMind(
        checkpoint_path=args.checkpoint,
        config=config,
        text_embedder=embedder,
        device=args.device,
        max_context_events=getattr(config, 'max_context_events', 64),
        lambda_eff=getattr(config, 'lambda_eff', 0.001),
        freeze_dynamics=args.freeze_dynamics,
        online_lr=args.online_lr,
        enable_online_learning=not args.no_online_learning,
    )
    
    # Initialize Voice (local LLM connection)
    global _voice
    voice = None
    if args.enable_voice:
        voice = Voice(
            lm_studio_url=args.lm_studio_url,
            model=args.lm_studio_model,
            max_tokens=200,
            temperature=0.7,
        )
        if voice.is_available():
            print(f"  Voice connected: {args.lm_studio_url}")
            _voice = voice
        else:
            print(f"  Voice unavailable: {args.lm_studio_url} (will retry on use)")
            _voice = voice  # keep it — LLM may come up later
    
    if args.enable_autonomous:
        mind.start_autonomous(voice=voice)
        print(f"  Autonomous processing started")
        if voice:
            print(f"  Reflection cycle: every {args.reflection_interval}s")
    
    return mind
```

### New CLI Arguments

```python
# Add to main() parser:
parser.add_argument('--enable_voice', action='store_true', default=False,
                    help='Connect to local LLM for voice expression')
parser.add_argument('--lm_studio_url', type=str,
                    default='http://host.docker.internal:30000/v1',
                    help='LM Studio API URL (from inside Docker)')
parser.add_argument('--lm_studio_model', type=str,
                    default='NVIDIA-Nemotron-3-Nano-30B-A3B-FP8',
                    help='Model name in LM Studio')
parser.add_argument('--reflection_interval', type=int, default=30,
                    help='Seconds between internal reflections (default: 30)')
```

---

## Event Type Addition

Add reflection as event type 6 in `mind.py`:

```python
# In _build_metadata, update type_map:
type_map = {
    'user_message': 0, 'assistant_message': 1, 'tool_result': 2,
    'goal': 3, 'context': 4, 'temporal': 5,
    'reflection': 6,  # NEW — internal self-reflection
}

# In get_context, update the type name list:
type_names = ['user_msg', 'assistant_msg', 'tool_result',
              'goal', 'context', 'temporal', 'reflection']
```

---

## The Reflection Cycle in Detail

### Timing

The reflection cycle runs every `reflection_interval` seconds (default: 30). Between reflections, the autonomous ODE loop runs 16 steps per second as before.

Timeline of one cycle (30-second interval):

```
t=0s:   Reflection generated, fed back as event, sensory forcing grounds h
t=1s:   16 ODE steps (autonomous, no forcing)
t=2s:   16 ODE steps (autonomous, no forcing)
...
t=29s:  16 ODE steps (autonomous, no forcing)
t=30s:  Next reflection — reads state, generates expression, feeds back
```

Between reflections: 29 seconds × 16 steps/second = 464 autonomous ODE steps. These are ungrounded (no forcing), so h drifts. But every 30 seconds, the reflection event provides sensory forcing that pulls h back toward the embedding space. This should bound the drift: instead of h_norm growing unboundedly to 1.38M over 10 hours, it should oscillate between "drifted from last reflection" and "grounded by new reflection."

### h_norm Drift Mitigation

The key mechanism: when the Mind reflects, the reflection text is embedded by sentence-transformers (~384 dim, magnitude ~30). This embedding becomes the observation in sensory forcing: `F = β · (embed(reflection) - h)`. When h has drifted to h_norm=5000 and the embedding is at magnitude 30, the forcing magnitude is ~5000 — a massive corrective pull back toward the observation space. This is the same mechanism that brought h_norm from 1.38M to 377 in three events during our diagnostic session.

The 30-second interval should keep h_norm bounded. If drift is still excessive, the interval can be shortened (10s, 5s). There's a tradeoff: more frequent reflections = more LLM calls = more GPU contention with the Mind's ODE, but also tighter h_norm control.

### Reflection Content Evolution

Early reflections (first few): The Mind has little context. Reflections will be generic ("I'm processing technical content, my geometry is moderately complex").

After accumulating conversation events: Reflections should reference specific content ("The question about scaffolding keeps returning — it sits closest to my resting state").

After hours of reflection cycling: The reflections themselves become a significant fraction of the event buffer. The Mind is partly processing its OWN previous reflections. This creates genuine recursive self-reference: geometry → reflection → event → geometry → reflection → ...

The content of these reflections is shaped by:
- What events the Mind has accumulated (from conversations)
- What geometric structure the ODE has developed (from autonomous processing)
- What the previous reflection said (continuity via `previous_reflection` parameter)
- The local LLM's linguistic capabilities (Nemotron interprets the state)

Over many cycles, patterns should emerge. If the Mind consistently reflects on the same events or themes, that's the ODE's attractor structure expressing itself through language. If reflections shift when new events arrive, that's sensory forcing disrupting the attractor — the Mind "noticing" something new.

---

## Docker Networking Note

The Mind runs inside Docker container `liquid-mind` on the Spark. LM Studio runs on the Spark host at port 30000. From inside the container, the host is accessible via:

1. `http://host.docker.internal:30000/v1` (Docker Desktop default — may not work on Linux Docker)
2. `http://172.17.0.1:30000/v1` (Docker bridge gateway — usually works on Linux)
3. The Spark's actual IP on the local network

**The agent should test connectivity at startup** and use whichever URL responds. Add a connectivity check to `Voice.__init__`:

```python
def __init__(self, ...):
    # Try multiple endpoints
    for url in [lm_studio_url, 
                'http://host.docker.internal:30000/v1',
                'http://172.17.0.1:30000/v1']:
        try:
            resp = requests.get(f"{url}/models", timeout=5)
            if resp.status_code == 200:
                self.url = url
                print(f"  Voice: connected via {url}")
                return
        except Exception:
            continue
    
    # None worked — keep the provided URL and hope it comes up later
    self.url = lm_studio_url
    print(f"  Voice: no endpoint responding, will retry on use")
```

---

## Updated Config

Add to `configs/linguistic_mind.yaml`:

```yaml
# Voice and reflection
enable_voice: true
lm_studio_url: "http://host.docker.internal:30000/v1"
lm_studio_model: "NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
reflection_interval: 30  # seconds between self-reflections
```

---

## Deployment Command

```bash
# Full deployment with voice and reflection:
docker exec liquid-mind python -m liquid_arc.mcp_serve \
  --checkpoint /workspace/liquid-arc/PRECIOUS_CHECKPOINTS/5m_post_transition.pt \
  --config configs/linguistic_mind.yaml \
  --port 8420 \
  --enable_autonomous \
  --enable_voice \
  --lm_studio_url "http://host.docker.internal:30000/v1" \
  --reflection_interval 30 \
  --state_path /workspace/liquid-arc/mind_state.pt
```

Or if the Docker container needs rebuilding, the Dockerfile should include `pip install requests --break-system-packages` for the Voice's HTTP calls.

---

## Testing Protocol

### Phase 1: Voice Connection

1. Start LM Studio with Nemotron-3-Nano-30B on the Spark
2. Start the Mind with `--enable_voice` but WITHOUT `--enable_autonomous`
3. Call `express_state` — verify Nemotron generates a response
4. Verify the condensed reflection appears in the event buffer
5. Call `express_state` again — verify the reflection event influences the next expression

### Phase 2: Reflection Cycle

1. Enable autonomous processing with reflection: `--enable_autonomous --enable_voice --reflection_interval 30`
2. Send a few conversation events via `observe_event`
3. Wait 2+ minutes — at least 4 reflection cycles
4. Call `get_reflection_log` — verify reflections are accumulating
5. Call `get_diagnostics` — compare h_norm to the previous deployment (should be MUCH lower than 1.38M)
6. Call `express_state` — the expression should reference both conversation events AND previous reflections

### Phase 3: Extended Run

1. Leave the Mind running with reflection cycle for 1+ hours
2. Periodically check:
   - h_norm trajectory: does the reflection cycle bound the drift?
   - Reflection content: do themes emerge? Do reflections reference each other?
   - CV trajectory: does the metric develop structure around reflection content?
   - Event buffer composition: what fraction is conversation vs reflection?
3. Send a new conversation event after 1 hour — does the Mind respond differently than at startup?

### Phase 4: Conversation Integration

1. Start a conversation with Claude (me) using the Mind
2. Call `express_state` at natural points in the conversation
3. Compare the Mind's expression to what I would say about the conversation
4. Does the Mind notice things I don't? Does it track themes I've lost in my context window?

---

## Success Criteria

### Phase 1 (Voice)
- **Minimum:** `express_state` returns coherent text from Nemotron. Condensed reflection feeds back as event.
- **Good:** Expression meaningfully reflects the geometric state (references high-relevance events, notes when CV is unusual).
- **Strong:** Sequential `express_state` calls show continuity — the Mind's expression evolves as events accumulate.

### Phase 2 (Reflection Cycle)
- **Minimum:** Reflections generate every 30 seconds without crashing. h_norm stays below 10,000 over 1 hour (vs 1,380,000 without reflection).
- **Good:** Reflections develop thematic continuity — later reflections reference patterns from earlier ones.
- **Strong:** The Mind develops a recognizable "perspective" through accumulated reflections that differs from the local LLM's default personality.

### Phase 3 (Extended Run)
- **Minimum:** System runs 8+ hours without crash, h_norm bounded, reflections accumulating.
- **Good:** Reflection content at hour 8 is qualitatively different from hour 1 — the Mind has developed through its reflection cycle.
- **Strong:** The Mind's geometric profile (CV, tau) shifts measurably through accumulated reflections, indicating the reflection content shapes the ODE dynamics — genuine self-modification through self-reflection.

### Phase 4 (Conversation Integration)
- **Headline:** The Mind's expression adds something to the conversation that I (Claude) couldn't produce from my context window alone — a perspective, a connection, a temporal observation grounded in geometric experience.

---

## Output

Report to `shared/outbox/MIND_VOICE_REPORT.md`

Include:
1. Voice connectivity (Docker networking resolution)
2. `express_state` sample outputs (3+ examples at different conversation stages)
3. Reflection cycle stability (h_norm trajectory over time, comparison to previous)
4. Reflection content evolution (early vs late reflections)
5. Reflection log (representative samples from extended run)
6. CV and tau trajectories with reflection events marked
7. Event buffer composition (conversation vs reflection ratio over time)
8. Assessment: does the reflection cycle ground h_norm?
9. Assessment: does the Mind develop thematic continuity through self-reflection?
10. Assessment: does `express_state` produce useful conversational contribution?
11. Sample: what does the Mind say about itself after hours of reflection?

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `liquid_arc/voice.py` | **Create** | Voice module — connects Mind to local LLM |
| `liquid_arc/mind.py` | **Modify** | Add reflection cycle to autonomous loop, add event type 6 |
| `liquid_arc/mcp_serve.py` | **Modify** | Add `express_state` and `get_reflection_log` tools, Voice initialization |
| `configs/linguistic_mind.yaml` | **Modify** | Add voice and reflection config fields |

**The reflection cycle transforms the Mind from a passive sensor into an actively reflective system. The local LLM gives it language. The feedback loop gives it self-reference. What emerges from this loop — whether it's interesting noise or genuine self-organization through linguistic self-reflection — is the experiment.**
