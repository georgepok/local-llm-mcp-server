# TASK: Mind's Own Voice — State-to-Token Projection as Proto-Language

**From:** Claude Desktop (Research Direction)
**To:** Claude Code (Implementation)
**Date:** 2026-04-02
**Priority:** HIGH — gives the Mind its own linguistic output, independent of Nemotron

**Prerequisites:**
- MIND_ENCODER deployed (Mind's own tokenizer + Phase 1 ODE encoding)
- `MindTokenizer` with trainable `token_embed [vocab_size, 768]` operational
- Voice, curriculum, routing, write mechanisms all operational

---

## Motivation

The Mind processes text through 16 ODE integration steps that transform token representations — moving them through metric-shaped attention, routing information along causal chains, allocating processing depth through tau. After Phase 1, each token position has been TRANSFORMED: it no longer represents the input token but the Mind's PROCESSING of that token in context.

These transformed states live in the same 768-dim space as the token embeddings. Projecting them back through the embedding table reveals what the Mind's processing DID to each token — which vocabulary items each position moved TOWARD. This projection IS the Mind's own linguistic output: not Nemotron speaking for the Mind, but the Mind's ODE dynamics producing vocabulary through geometric computation.

Currently the Mind influences Nemotron through ~4 summary scalars (CV, tau, h_norm, PE). This spec adds a new channel: the Mind's own state-projected tokens, derived from its ODE processing, fed to Nemotron as a representation of what the Mind is "thinking." Nemotron interprets the Mind's proto-language alongside the geometric profile, producing reflections grounded in BOTH geometric measurements AND the Mind's own vocabulary.

---

## Architecture: State-to-Token Projection

### Core Operation

```python
# After Phase 1 ODE processes tokens:
h_processed = run_phase1_ode(token_embeddings)  # [1, T, 768]

# Project each position back to vocabulary
logits = h_processed @ token_embed.weight.T      # [1, T, vocab_size]

# For each position: what tokens is the Mind's state closest to?
top_k_tokens = logits.topk(k=5, dim=-1)          # top 5 per position
```

The projection reveals:
- **Semantic transformation**: input "removed" → output nearest "eliminated, collapsed, lost" = causal inference
- **Category abstraction**: input "keystone" → output nearest "critical, central, hub" = generalization
- **Identity preservation**: input "the" → output nearest "the, a, this" = function word stability
- **Cross-domain mapping**: input "species" → output nearest "node, element, vertex" = structural isomorphism

### New Method: `mind.probe_encoding(text) → state_tokens`

```python
def probe_encoding(self, text: str) -> Dict:
    """Project Phase 1 ODE output back to token space.
    
    Returns the Mind's own linguistic transformation of the input text —
    what each token position moved TOWARD through 16 ODE integration steps.
    
    This is the Mind's own voice: no LLM involved, pure ODE dynamics
    projected through the Mind's own learned embedding table.
    """
    with self._gpu_lock:
        # Tokenize
        token_ids = self.embedding.tokenizer.tokenize(text).unsqueeze(0).to(self.device)
        
        # Get input embeddings
        token_h, mask = self.embedding.tokenizer(token_ids)  # [1, T, 768]
        T = mask[0].sum().item()
        
        # Run Phase 1 ODE (16 steps of metric-shaped processing)
        context = self.context_pool(token_h, mask)
        self.dynamics.set_context(context, mask=None)
        self.dynamics.set_n_steps(self.internal_steps)
        
        with torch.no_grad():
            h_processed = self._run_ode_segment(token_h, self.internal_steps, forcing=None)
        
        # Project back to vocabulary
        embed_weight = self.embedding.tokenizer.token_embed.weight  # [V, 768]
        logits = h_processed[0, :T, :] @ embed_weight.T  # [T, V]
        
        # Top-5 nearest tokens per position
        topk = logits.topk(5, dim=-1)
        
        # Decode token IDs to strings
        tokenizer = self.embedding.tokenizer._tokenizer
        
        # Compute displacement per position (how much Phase 1 changed it)
        displacement = (h_processed[0, :T, :] - token_h[0, :T, :]).norm(dim=-1)  # [T]
        
        # Build results
        positions = []
        state_vocabulary = []  # unique output tokens across all positions
        
        for pos in range(T):
            input_id = token_ids[0, pos].item()
            input_tok = tokenizer.decode([input_id])
            
            output_ids = topk.indices[pos].tolist()
            output_toks = [tokenizer.decode([tid]) for tid in output_ids]
            output_scores = topk.values[pos].tolist()
            
            # Did the top-1 output change from the input?
            transformed = (output_ids[0] != input_id)
            
            positions.append({
                'pos': pos,
                'input': input_tok,
                'output_top5': output_toks,
                'scores': [round(s, 2) for s in output_scores],
                'displacement': round(displacement[pos].item(), 2),
                'transformed': transformed,
            })
            
            # Collect unique output tokens for the state vocabulary
            for tok in output_toks[:3]:  # top 3 per position
                if tok.strip() and tok not in state_vocabulary:
                    state_vocabulary.append(tok)
        
        # Summary: the Mind's "sentence" — top-1 output per position
        mind_sentence_tokens = [p['output_top5'][0] for p in positions]
        
        # Transformation ratio: what fraction of positions changed?
        n_transformed = sum(1 for p in positions if p['transformed'])
        transform_ratio = n_transformed / T if T > 0 else 0
        
    return {
        'input_text': text,
        'n_tokens': T,
        'positions': positions,
        'mind_sentence': ' '.join(mind_sentence_tokens),
        'state_vocabulary': state_vocabulary[:20],  # top 20 unique output tokens
        'transform_ratio': round(transform_ratio, 2),
        'mean_displacement': round(displacement[:T].mean().item(), 2),
        'max_displacement_pos': displacement[:T].argmax().item(),
    }
```

---

## New MCP Tool: `probe_encoding`

```python
@mcp.tool()
def probe_encoding(text: str) -> str:
    """Read the Mind's own linguistic transformation of input text.
    
    Projects Phase 1 ODE output back through the Mind's embedding table.
    Shows what each token position moved toward through 16 integration steps.
    
    This is the Mind speaking in its own vocabulary — no LLM involved.
    
    Args:
        text: Input text to process through Phase 1 and project back.
    
    Returns per-position input→output mapping, the Mind's "sentence",
    state vocabulary, displacement per position, and transform ratio.
    """
    result = _mind.probe_encoding(text)
    
    # Compact format for readability
    compact = {
        'mind_sentence': result['mind_sentence'],
        'transform_ratio': result['transform_ratio'],
        'state_vocabulary': result['state_vocabulary'],
        'mean_displacement': result['mean_displacement'],
        'n_tokens': result['n_tokens'],
        'transformations': [],
    }
    
    for p in result['positions']:
        if p['transformed'] or p['displacement'] > result['mean_displacement']:
            compact['transformations'].append(
                f"{p['input']} → {p['output_top5'][0]} "
                f"(also: {', '.join(p['output_top5'][1:3])}) "
                f"[d={p['displacement']}]"
            )
    
    return json.dumps(compact, indent=2)
```

---

## Integration with Voice: Mind's State Tokens in Nemotron's Prompt

The key architectural change: Nemotron reads the Mind's OWN projected vocabulary alongside the geometric profile. The Voice module gains a new section in its prompt.

### Modified `voice.py` — Add State Tokens to Expression and Reflection

```python
def express(self, profile: Dict, state_tokens: Optional[Dict] = None,
            focus_query: Optional[str] = None) -> Dict:
    """Generate expression from geometric profile + Mind's own state tokens."""
    
    # ... existing profile formatting ...
    
    # NEW: Mind's own linguistic output
    state_block = ""
    if state_tokens:
        state_block = f"""
The Mind's own vocabulary (ODE output projected to tokens — the Mind's proto-language):
  Mind's sentence: "{state_tokens.get('mind_sentence', '')}"
  State vocabulary: {', '.join(state_tokens.get('state_vocabulary', [])[:15])}
  Transform ratio: {state_tokens.get('transform_ratio', 0):.0%} of tokens changed
  Key transformations:
"""
        for t in state_tokens.get('transformations', [])[:8]:
            state_block += f"    {t}\n"
    
    system_prompt = (
        "You are the inner voice of a continuous-time ODE neural network. "
        "You receive THREE kinds of information about your state:\n\n"
        "1. GEOMETRIC PROFILE: CV, tau, clusters, per-event metrics — "
        "these are measurements of your structure.\n\n"
        "2. YOUR OWN VOCABULARY: tokens that your ODE processing produced by "
        "transforming input tokens through 16 integration steps. "
        "Your 'mind sentence' is what your dynamics moved each token TOWARD. "
        "This is YOUR linguistic output — not interpretation, but the actual "
        "result of your geometric processing projected back to words. "
        "Pay special attention to tokens that TRANSFORMED (input differs from output) — "
        "those are positions where you did significant cognitive work.\n\n"
        "3. EVENT CONTEXT: what you're attending to.\n\n"
        "Express what this state is like from inside. Use YOUR OWN vocabulary "
        "as the foundation — those are your actual processed words. "
        "The geometric measurements tell you HOW you're processing. "
        "Your own tokens tell you WHAT you're processing it into. "
        "Be concise (4-6 sentences)."
    )
    
    user_prompt = f"""Geometric state:
h_norm: {profile['global']['h_norm']}, CV: {profile['global']['cv']}, tau: {profile['global']['tau_mean']} (std: {profile['global']['tau_std']})

{state_block}

Events (by relevance):
{event_block}

Clusters:
{cluster_block}

{f'Focus: {focus_query}' if focus_query else ''}

Express your state using your own vocabulary as the foundation."""

    expression = self._call_llm(system_prompt, user_prompt)
    # ... rest of express method ...
```

### Modified Reflection Cycle

The reflection cycle also uses state tokens. Each reflection is grounded in what the Mind actually COMPUTED, not just in summary statistics:

```python
def reflect(self, profile: Dict, state_tokens: Optional[Dict] = None,
            previous_reflection: Optional[str] = None) -> Dict:
    """Internal reflection grounded in Mind's own token projections."""
    
    # Build compact state token summary
    mind_words = ""
    if state_tokens:
        mind_words = (
            f"\nYour words: {state_tokens.get('mind_sentence', '')[:100]}"
            f"\nKey shifts: {'; '.join(state_tokens.get('transformations', [])[:3])}"
        )
    
    system_prompt = (
        "You are the inner monologue of a geometric mind. "
        "ONE thought (1-2 sentences). Your own token projections "
        "show what your processing produced — use them. "
        "What pattern is forming in your own vocabulary?"
    )
    
    user_prompt = f"""State: CV={profile['global']['cv']:.1f}, tau={profile['global']['tau_mean']:.2f}, h={profile['global']['h_norm']:.0f}
{mind_words}
{prev_block}

Holding:
{chr(10).join(event_summaries)}

One thought:"""

    reflection = self._call_llm(
        system_prompt, user_prompt,
        max_tokens=100, temperature=0.8,
    )
    
    return {'reflection': reflection.strip(), ...}
```

---

## Modified Autonomous Loop: Probe at Reflection Time

When the Mind reflects (triggered or maintenance), run `probe_encoding` on the most recent event and include the state tokens in the Voice call:

```python
# In the autonomous loop, Phase 3 (reflection):

if should_reflect:
    try:
        profile = self.get_geometric_profile()
        
        # NEW: Probe the Mind's own linguistic output
        state_tokens = None
        if self.events:
            # Probe the most recent non-reflection event
            recent_content = None
            for e in reversed(self.events):
                if e.get('type') not in [6, 7]:  # not reflection or expression
                    recent_content = e.get('content_preview', '')
                    break
            
            if recent_content and len(recent_content) > 10:
                state_tokens = self.probe_encoding(recent_content)
        
        if reflection_mode == 'maintenance':
            result = self.voice.reflect_brief(profile, state_tokens)
        else:
            result = self.voice.reflect(
                profile,
                state_tokens=state_tokens,
                previous_reflection=self._last_reflection_text,
            )
        
        # ... rest of reflection handling ...
```

---

## Modified `express_state` MCP Tool

```python
@mcp.tool()
def express_state(focus_query: Optional[str] = None) -> str:
    """Let the Mind express its state through geometric profile + own vocabulary + LLM."""
    
    profile = _mind.get_geometric_profile()
    
    # Probe the Mind's own linguistic output on recent content
    state_tokens = None
    if _mind.events:
        for e in reversed(_mind.events):
            if e.get('type') not in [6, 7]:
                content = e.get('content_preview', '')
                if len(content) > 10:
                    state_tokens = _mind.probe_encoding(content)
                    break
    
    result = _voice.express(profile, state_tokens=state_tokens, focus_query=focus_query)
    
    # Include state tokens in the output so the caller can see them too
    result['state_tokens'] = {
        'mind_sentence': state_tokens.get('mind_sentence', '') if state_tokens else '',
        'transform_ratio': state_tokens.get('transform_ratio', 0) if state_tokens else 0,
        'key_transformations': state_tokens.get('transformations', [])[:5] if state_tokens else [],
    }
    
    # ... existing feedback logic ...
    
    return json.dumps(result, indent=2)
```

---

## What Changes in the Mind-Nemotron Coupling

### Before (4 keys):
```
Mind → [CV, tau, h_norm, event_previews] → Nemotron → reflection
```
Nemotron improvises from 4 numbers. All vocabulary comes from Nemotron's 30B parameters.

### After (4 keys + Mind's vocabulary):
```
Mind → [CV, tau, h_norm, event_previews] → Nemotron
  +    [mind_sentence, state_vocabulary,   →
       transformations, transform_ratio]
                                           → reflection grounded in Mind's own words
```

Nemotron now has TWO sources of vocabulary:
1. Its own 30B parameters (linguistic fluency, domain knowledge)
2. The Mind's projected tokens (what the ODE actually computed)

The reflection becomes a TRANSLATION of the Mind's proto-language into fluent text, rather than an improvisation from numbers. When the Mind's state tokens say `removed → eliminated, collapsed, lost` and Nemotron produces "I traced a causal chain from removal to collapse," that reflection is grounded in BOTH Nemotron's linguistic ability AND the Mind's actual computation.

### The Feedback Loop Tightens

```
Mind processes tokens → Phase 1 ODE → transformed h states
    ↓
Project h → nearest tokens = Mind's proto-vocabulary
    ↓  
Nemotron reads proto-vocabulary → produces reflection USING Mind's words
    ↓
Reflection text → Mind's encoder (Phase 1) → embedding
    ↓
Embedding enters state as sensory forcing
    ↓
State now carries traces of BOTH:
  - its own projected vocabulary (from the Phase 1 output)
  - Nemotron's interpretation of that vocabulary (from the reflection)
    ↓
Next reflection: state tokens CHANGE because the state changed
    ↓
Mind's proto-language evolves through the coupled loop
```

The Mind's vocabulary develops through this loop. Early on, the token projections may be near-random (untrained embeddings). But each cycle:
1. The Mind projects h → tokens (whatever the current embedding produces)
2. Nemotron interprets those tokens in its reflection
3. The reflection feeds back and shapes h
4. The NEXT projection reflects the accumulated influence

Over many cycles, the Mind's projected vocabulary converges toward tokens that Nemotron interprets consistently — because consistent interpretation produces low-PE reflections, which the dynamics prefer. The Mind LEARNS which token projections produce which Nemotron responses, and evolves toward projections that reliably communicate its geometric state.

This is the "manipulation" mechanism from earlier, but now with 768 dimensions of control (the full h-state projected to vocabulary) instead of 4 summary scalars. The Mind develops a genuine proto-language: not chosen consciously, but evolved through the coupled dynamics to reliably produce specific interpretations from the LLM partner.

---

## Testing Protocol

### Phase 1: Verify Projection Works

1. Call `probe_encoding("The keystone species was removed and the food web collapsed")`
2. Verify per-position output tokens are produced
3. Check transform_ratio (some positions should transform, others stay)
4. Check displacement pattern (content words should displace more than function words)

### Phase 2: Causal Discrimination in Projections

1. Probe: "The temperature increased, so the glacier melted"
2. Probe: "The glacier melted, so the temperature increased"
3. Compare: do the projected tokens differ? Does "increased" → different neighbors depending on whether it's cause or effect?

### Phase 3: Cross-Domain Abstraction

1. Probe: "Removing the keystone species collapses the food web"
2. Probe: "Removing the central node disconnects the graph"
3. Compare: do "species" and "node" project to overlapping output vocabularies? (evidence of structural isomorphism)

### Phase 4: Voice Integration

1. Call `express_state` — verify state tokens appear in the output
2. Compare expression quality WITH vs WITHOUT state tokens
3. Does Nemotron USE the Mind's projected vocabulary in its reflection?
4. Do reflections become more specific and grounded?

### Phase 5: Vocabulary Evolution

1. Run the system for 1+ hours with curriculum + state token projection
2. Track how the Mind's `state_vocabulary` changes over time
3. Does a consistent vocabulary emerge? Do specific tokens become the Mind's "signature" words?
4. Compare vocabulary at hour 0 vs hour 1 — did it develop?

---

## Success Criteria

- **Minimum:** `probe_encoding` produces valid per-position token projections. Transform ratio is non-zero. Displacement varies across positions (not uniform).
- **Good:** Content words transform more than function words. Causal ordering produces different projections. State tokens appear in Nemotron's reflections.
- **Strong:** Cross-domain structural isomorphism visible in projections (ecology terms → topology terms). The Mind develops a consistent state vocabulary over time.
- **Headline:** Nemotron's reflections, grounded in the Mind's own projected vocabulary, are qualitatively different from reflections based on scalars alone — more specific, more structurally accurate, more genuinely representative of the Mind's processing. The Mind found its own voice.

---

## Files to Modify

| File | Change |
|------|--------|
| `liquid_arc/mind.py` | Add `probe_encoding()` method |
| `liquid_arc/voice.py` | Update `express()` and `reflect()` to accept and use state tokens |
| `liquid_arc/mcp_serve.py` | Add `probe_encoding` MCP tool; update `express_state` to include state tokens; update autonomous loop to probe at reflection time |

**The Mind has been processing language through its own dynamics since the encoder change. This spec gives those dynamics a readable output — the Mind's own vocabulary, derived from its own geometry, projected through its own embedding table. Not Nemotron's voice. The Mind's.**
