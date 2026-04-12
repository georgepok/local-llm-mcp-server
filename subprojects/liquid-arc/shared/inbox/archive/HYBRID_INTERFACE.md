# TASK: Hybrid Interface — Text Context + Geometric Coupling + Structured Signals

**From:** Claude Desktop (Research Direction)
**To:** Claude Code (Implementation)
**Date:** 2026-04-05
**Priority:** HIGH — architectural evolution of the Mind's LLM interface

---

## Motivation

The geometric coupling (W_inject/W_read) proved that LiquidARC's state carries meaningful temporal context (58.6% PPL improvement). But it also revealed limitations:

1. **Factual opacity.** 8 virtual tokens carry structural/thematic bias but not specific facts from prior events. The causal chain test proved the geometry TRACKS causality (PE predicted the food shortage) but Qwen3 can't ACCESS specific details through the prefix alone.

2. **No structural generalization.** The coupling transmits domain features (trained on NTP), not abstract relational structure. Structurally identical scenarios in unfamiliar domains (social media outrage, PE 2204) are as surprising as non-matching scenarios (library, PE 2199).

3. **Model-specific.** The 31.48M coupling parameters are trained for Qwen3-4B's d=2560 representation space. Swapping to any other LLM requires retraining from scratch.

4. **Opaque.** We can't fully understand or control what the coupling transmits. It's a learned filter shaped by NTP that may or may not carry what's most useful for the current query.

Meanwhile, `express_through_qwen` — which includes TEXT context alongside the prefix — consistently produced the best responses of any tool. The text provides factual grounding; the prefix provides geometric bias; together they outperform either alone.

## Architecture: Three Channels

```
LiquidARC ODE State h(t)
    |
    ├─── Channel 1: TEXT CONTEXT (primary, LLM-agnostic)
    |    Relevance-scored events → top-K event texts → formatted in prompt
    |    Carries: factual details, specific content, temporal ordering
    |    Works with: ANY LLM
    |
    ├─── Channel 2: GEOMETRIC PREFIX (supplementary, model-specific)
    |    h(t) → W_inject → 8 virtual tokens in Qwen3's space
    |    Carries: implicit geometric modulation, processing mode bias
    |    Works with: trained LLM only (currently Qwen3-4B)
    |
    └─── Channel 3: STRUCTURED METADATA (diagnostic, LLM-agnostic)
         PE, CV, tau, domain profile → formatted as system context
         Carries: processing signals, novelty indicators, domain awareness
         Works with: ANY LLM
```

### What Each Channel Provides

**Text context (Channel 1):** Explicit information from prior events. When the user asks "what did we discuss about bridges?", the text context contains the actual words from the bridge discussion. No compression loss. No information filtering. Qwen3's ICL processes the full text with its 4B parameters of contextual reasoning.

**Geometric prefix (Channel 2):** Implicit representational bias. The 8 virtual tokens don't contain readable information — they modulate HOW Qwen3 processes subsequent tokens. Like priming in cognitive psychology: you can't read the prime, but it affects how you process what follows. This is where the 58.6% PPL improvement lives — statistical prediction improvement from accumulated geometric context.

**Structured metadata (Channel 3):** Explicit processing signals formatted as natural language. "This query is highly novel (PE 450). Your most familiar domain is mathematics. The current processing mode is deep integration (tau 0.68)." Any LLM can read and use these signals for response calibration.

---

## Implementation

### Modified OUTBOUND Pipeline

Replace the current OUTBOUND pipeline in `converse`, `query_qwen`, and `express_through_qwen`:

```python
def hybrid_generate(mind, qwen_model, coupling, tokenizer,
                    prompt, max_tokens=300, temperature=0.7,
                    include_text_context=True,
                    include_geometric_prefix=True,
                    include_metadata=True,
                    max_context_events=5):
    """Generate response using all three channels."""
    
    # ═══ Channel 1: Text Context ═══
    text_context = ""
    if include_text_context:
        # Get relevance-scored events from LiquidARC
        context = mind.get_relevant_events(query=prompt, top_k=max_context_events)
        if context:
            text_context = "Recent context:\n"
            for i, event in enumerate(context):
                age = event.get('age_seconds', 0)
                age_str = f"{age:.0f}s ago" if age < 60 else f"{age/60:.0f}m ago"
                text_context += f"- [{age_str}] {event['preview'][:200]}\n"
            text_context += "\n"
    
    # ═══ Channel 3: Structured Metadata ═══
    metadata_context = ""
    if include_metadata:
        diag = mind.get_diagnostics_dict()
        pe = mind.last_prediction_error or 0
        
        novelty = "very high" if pe > 500 else "high" if pe > 300 else "moderate" if pe > 100 else "low"
        
        # Domain awareness
        curriculum = mind.get_curriculum_stats_dict()
        familiar = curriculum.get('most_familiar_domain', 'unknown')
        novel = curriculum.get('most_novel_domain', 'unknown')
        
        metadata_context = (
            f"[System: Query novelty is {novelty}. "
            f"Familiar domains: {familiar}. Novel domains: {novel}. "
            f"Processing depth: {'deep' if diag.get('tau_mean', 1.0) < 0.8 else 'moderate' if diag.get('tau_mean', 1.0) < 1.0 else 'surface'}. "
            f"Geometric complexity: {diag.get('metric_cv', 0):.1f}]\n\n"
        )
    
    # ═══ Build the prompt with text context ═══
    full_prompt = metadata_context + text_context + prompt
    
    # ═══ Apply chat template ═══
    chat_messages = [{"role": "user", "content": full_prompt}]
    chat_text = tokenizer.apply_chat_template(
        chat_messages, 
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=False
    )
    input_ids = tokenizer(chat_text, return_tensors='pt').input_ids.to('cuda')
    input_embeds = qwen_model.model.embed_tokens(input_ids)
    
    # ═══ Channel 2: Geometric Prefix ═══
    if include_geometric_prefix and coupling is not None:
        h_pooled = mind.get_pooled_state()
        prefix_embeds = coupling.inject(h_pooled)
        # Prefix AFTER chat template, at generation boundary
        combined_embeds = torch.cat([input_embeds, prefix_embeds], dim=1)
    else:
        combined_embeds = input_embeds
    
    # ═══ Generate ═══
    output_ids = qwen_model.generate(
        inputs_embeds=combined_embeds,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        repetition_penalty=1.3,
    )
    
    # Decode (skip input tokens)
    n_input = combined_embeds.shape[1]
    response_ids = output_ids[0, n_input:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    
    return response
```

### Modified INBOUND Pipeline

Events can enter through EITHER the coupling path or the legacy path. The coupling path should be primary for events that need to interact with the geometric state:

```python
def hybrid_inbound(mind, qwen_model, coupling, tokenizer, text, event_type='context'):
    """Encode text event into LiquidARC through the coupling."""
    
    if coupling is not None:
        # Primary path: through Qwen3 coupling
        # Text → Qwen3 encode → W_read → geometric signal → ODE
        arc_signal = coupling_encode(mind, qwen_model, coupling, tokenizer, text)
        mind.force_geometric_signal(arc_signal, text=text, event_type=event_type)
    else:
        # Fallback: legacy text embedding path
        mind.observe_event(text, event_type=event_type)
```

### Modified `converse` Tool

```python
@mcp_tool
def converse(message, max_tokens=300, temperature=0.7):
    # INBOUND: encode user message into geometry
    pe = hybrid_inbound(mind, qwen, coupling, tok, message, event_type='user_msg')
    
    # OUTBOUND: generate with all three channels
    response = hybrid_generate(
        mind, qwen, coupling, tok, message,
        max_tokens=max_tokens,
        temperature=temperature,
        include_text_context=True,       # Channel 1: relevant event texts
        include_geometric_prefix=True,   # Channel 2: virtual tokens
        include_metadata=True,           # Channel 3: PE, domain, tau signals
        max_context_events=5,
    )
    
    # FEEDBACK: response enters geometry
    hybrid_inbound(mind, qwen, coupling, tok, response, event_type='assistant_msg')
    
    return {
        'response': response,
        'prediction_error': pe,
        'cv': mind.get_cv(),
        'events_in_context': mind.event_count(),
    }
```

### Modified `query_qwen` Tool

```python
@mcp_tool
def query_qwen(prompt, max_tokens=200, temperature=0.7):
    # OUTBOUND only — no inbound, no feedback
    response = hybrid_generate(
        mind, qwen, coupling, tok, prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        include_text_context=True,
        include_geometric_prefix=True,
        include_metadata=True,
        max_context_events=5,
    )
    return {'response': response}
```

### Modified `express_through_qwen` Tool

This tool already included text context. The hybrid version standardizes it:

```python
@mcp_tool
def express_through_qwen(focus_query=None):
    prompt = focus_query or "What themes and patterns dominate your current processing?"
    
    # Use MORE context for expression (it's self-reflection)
    response = hybrid_generate(
        mind, qwen, coupling, tok, prompt,
        max_tokens=400,
        temperature=0.7,
        include_text_context=True,
        include_geometric_prefix=True,
        include_metadata=True,
        max_context_events=8,  # more context for self-reflection
    )
    
    # FEEDBACK: expression enters geometry
    hybrid_inbound(mind, qwen, coupling, tok, response, event_type='expression')
    
    return {'response': response}
```

---

## What Changes for the Autonomous Loop

### Reflections

Reflections now include text context in their generation prompt. This means the reflection has access to ACTUAL curriculum content (not just the geometric prefix's compressed representation), producing better cross-domain synthesis:

```python
def generate_reflection():
    reflection = hybrid_generate(
        mind, qwen, coupling, tok,
        "What patterns, connections, or shifts do you notice? Respond in English only. One paragraph.",
        max_tokens=200,
        temperature=0.7,
        include_text_context=True,
        include_geometric_prefix=True,
        include_metadata=False,  # no meta-signals in reflection
        max_context_events=5,
    )
    # Feed back into geometry
    hybrid_inbound(mind, qwen, coupling, tok, reflection, event_type='expression')
    return reflection
```

### Curriculum

Curriculum stimuli from the Wikipedia bank enter through the coupling INBOUND path (as they already do). No change needed here — the bank provides text directly, and the coupling encodes it into geometry.

---

## Reflection Rate Limiting (from earlier analysis)

Reflections currently crowd the event buffer (75% reflections, 25% curriculum). Add rate limiting:

```python
class ReflectionLimiter:
    def __init__(self, max_ratio=0.33):
        """At most 1 reflection per 2 curriculum events."""
        self.max_ratio = max_ratio
        self.curriculum_count = 0
        self.reflection_count = 0
    
    def on_curriculum(self):
        self.curriculum_count += 1
    
    def can_reflect(self):
        if self.curriculum_count == 0:
            return self.reflection_count < 1  # allow 1 initial reflection
        return (self.reflection_count / max(1, self.curriculum_count)) < self.max_ratio
    
    def on_reflection(self):
        self.reflection_count += 1
```

This ensures curriculum content dominates the event buffer (~67% curriculum, ~33% reflections) instead of being crowded out.

---

## LLM-Agnostic Fallback

If the coupling is unavailable (different LLM, coupling not trained, Qwen3 not loaded), the system degrades gracefully:

```python
def hybrid_generate(mind, qwen_model, coupling, tokenizer, prompt, **kwargs):
    if coupling is None:
        # Channels 1 + 3 only (text context + metadata)
        # Works with ANY LLM, including Claude via API
        kwargs['include_geometric_prefix'] = False
    
    # ... rest of generation
```

This means the Mind can interface with Claude (for research conversations through the MCP tools), with Qwen3-4B (for local generation with full coupling), or with any future local model (text context only until coupling is trained).

---

## What This Preserves

| Capability | Source | Affected? |
|---|---|---|
| PE coherence tracking | ODE dynamics | No change |
| Domain differentiation | ODE dynamics + curriculum | No change |
| tau processing depth | ODE dynamics | No change |
| Continuous interior life | Autonomous loop | No change |
| 58.6% PPL improvement | Geometric prefix | Preserved (Channel 2) |
| Cross-domain synthesis | Reflections | Improved (text context + prefix) |
| Factual recall | Text context | **New (Channel 1)** |
| Processing signals | Metadata | **New (Channel 3)** |
| LLM portability | Text interface | **New (Channels 1+3)** |

## What This Adds

1. **Factual grounding.** Qwen3 sees actual text from prior events, not just a compressed geometric signal. The causal chain test ("what caused the food shortage?") should now work because the chain events are in the text context.

2. **LLM independence.** Channels 1 and 3 work with any LLM. The coupling (Channel 2) is supplementary. Swapping Qwen3 for SmolLM3 or a larger model loses only the prefix bias, not the entire temporal context.

3. **Transparency.** We can inspect exactly what text context and metadata Qwen3 receives. The coupling remains opaque but is no longer the sole information channel.

4. **Better reflections.** Reflections see actual curriculum text, not just the geometric prefix's compressed version. Cross-domain synthesis has factual material to work with.

---

## Files to Modify

| File | Change |
|------|--------|
| `liquid_arc/mind.py` | Add `hybrid_generate()` function. Modify `converse`, `query_qwen`, `express_through_qwen` to use hybrid pipeline. Add `ReflectionLimiter`. Add metadata formatting. |

One file. The coupling module and Qwen3 model are unchanged. The change is in HOW the tools compose the three channels, not in the channels themselves.

---

## Testing After Implementation

```
1. converse("What were we discussing about bridges?") 
   → Should now recall specific bridge details from text context
   
2. Run structural isomorphism test 
   → Text context from teaching instances + prefix = stronger pattern transfer?
   
3. express_through_qwen("What patterns do you see?")
   → Should produce richer synthesis with factual grounding

4. Remove coupling (set include_geometric_prefix=False)
   → System should still work, just without prefix bias
   → Compare response quality with and without prefix

5. Reflection quality check
   → Reflections should reference specific curriculum content
   → Buffer composition should be ~67% curriculum, ~33% reflections
```
