# TASK: Relevance Scoring — Fix Context Selection for Hybrid Interface

**From:** Claude Desktop (Research Direction)
**To:** Claude Code (Implementation)
**Date:** 2026-04-05
**Priority:** HIGH — the hybrid interface's text context channel depends on this

---

## Problem

The hybrid interface has three channels working: text context, geometric prefix, and metadata. But the text context channel selects events using **untrained relevance scoring heads** that can't distinguish query-relevant events from irrelevant ones.

Concrete failure: User asks about a causal chain (bridge → trucks → landslide → food shortage). The events ARE in the 64-slot buffer. But the top-5 relevance-scored events include aardvark articles (relevance 2.441) alongside the actual bridge closure (relevance 1.878). The user's causal chain events are diluted by curriculum noise because the readout heads assign similar scores to everything.

Without accurate relevance scoring, the text context channel is a firehose — including everything at similar priority instead of curating the specific events that matter for the current query.

---

## Root Cause

The `StateReadout` module (~150K params) computes relevance scores from the ODE hidden states. These heads were initialized randomly and receive gradient only through `provide_feedback` — which has never been called. The heads have never learned what "relevant" means.

The current relevance computation (from the architecture):
```
relevance = StateReadout(h_state)  → scores per event position
```

This produces scores based on geometric properties of the hidden state (norm, direction, relationship to other positions). Without training, these geometric properties don't correlate with query-event semantic relevance.

---

## Fix: Three-Layer Relevance Scoring

Replace single untrained geometric relevance with a combined score from three signals:

### Layer 1: Text Similarity (immediate fix, no training needed)

Compute cosine similarity between the query and each event's stored text, using Qwen3's encoder:

```python
def compute_text_relevance(query, events, qwen_model, tokenizer, coupling):
    """Score events by text similarity to the query through Qwen3's encoder."""
    
    scores = []
    
    # Encode query through Qwen3 (get last hidden state, mean-pool)
    with torch.no_grad():
        q_ids = tokenizer(query, return_tensors='pt', truncation=True, 
                          max_length=128).input_ids.to('cuda')
        q_hidden = qwen_model(input_ids=q_ids, output_hidden_states=True)
        q_embed = q_hidden.hidden_states[-1].mean(dim=1)  # [1, d_qwen]
        q_embed = F.normalize(q_embed, dim=-1)
    
    for event in events:
        text = event.get('content_preview', '') or event.get('preview', '')
        if not text or len(text.strip()) < 5:
            scores.append(0.0)
            continue
        
        with torch.no_grad():
            e_ids = tokenizer(text[:256], return_tensors='pt', truncation=True,
                              max_length=128).input_ids.to('cuda')
            e_hidden = qwen_model(input_ids=e_ids, output_hidden_states=True)
            e_embed = e_hidden.hidden_states[-1].mean(dim=1)  # [1, d_qwen]
            e_embed = F.normalize(e_embed, dim=-1)
        
        similarity = (q_embed * e_embed).sum().item()
        scores.append(similarity)
    
    return scores
```

**Cost:** One Qwen3 forward pass per event. For 64 events at 128 tokens each, this is ~64 forward passes. At ~50ms each on the Spark, that's ~3 seconds. Acceptable for a query operation, but should be cached.

**Optimization — pre-compute event embeddings:**

```python
# Cache event embeddings when events enter the buffer
class EventEmbeddingCache:
    def __init__(self, max_events=64):
        self.embeddings = {}  # event_index → normalized embedding
    
    def add(self, index, text, qwen_model, tokenizer):
        with torch.no_grad():
            ids = tokenizer(text[:256], return_tensors='pt', truncation=True,
                           max_length=128).input_ids.to('cuda')
            hidden = qwen_model(input_ids=ids, output_hidden_states=True)
            embed = hidden.hidden_states[-1].mean(dim=1)
            self.embeddings[index] = F.normalize(embed, dim=-1)
    
    def score_query(self, query_embed):
        scores = {}
        for idx, embed in self.embeddings.items():
            scores[idx] = (query_embed * embed).sum().item()
        return scores
```

With caching, only ONE Qwen3 forward pass is needed per query (to encode the query). Event embeddings are computed once when they enter the buffer.

### Layer 2: Event Type + Recency Weighting (immediate fix, no training)

Simple heuristic boosting based on event type and age:

```python
def compute_structural_relevance(events, query_type='conversation'):
    """Score events by structural properties — type and recency."""
    
    scores = []
    for event in events:
        score = 1.0
        etype = event.get('type', '')
        age = event.get('age_seconds', 0)
        
        # Type boosting: conversational events get priority for conversational queries
        if query_type == 'conversation':
            if etype in ('user_msg', 'assistant_msg'):
                score *= 2.0   # conversation events are 2× more relevant
            elif etype == 'expression':
                score *= 1.0   # reflections are neutral
            elif etype == 'context':
                score *= 0.5   # curriculum content deprioritized for conversation
            elif etype == 'goal':
                score *= 1.5   # goals are always somewhat relevant
        
        # Recency: exponential decay with ~5 minute half-life
        recency = math.exp(-age / 300.0)
        score *= (0.3 + 0.7 * recency)  # floor at 0.3, so old events aren't zeroed
        
        scores.append(score)
    
    return scores
```

This alone would fix the causal chain test: user_msg events ("bridge closure") get 2× boost over context events ("aardvark"), and recent events get recency bonus. The bridge events were 60-100s old; the aardvark was 104s. With type boosting, bridge events jump from relevance ~1.9 to ~3.8, while aardvark stays at ~1.2.

### Layer 3: Geometric Relevance (existing, with online training)

Keep the existing `StateReadout` scores but train them using `provide_feedback`:

```python
def compute_geometric_relevance(mind, query=None):
    """Score events by geometric relationship in ODE state space."""
    # This is the existing get_context relevance scoring
    context = mind.get_context(query=query)
    return {e['index']: e['relevance'] for e in context['context']}
```

Train over time using the `provide_feedback` tool. When Claude identifies relevant events (from conversation context), it calls:

```python
provide_feedback(event_index=33, feedback_type='correct', signal=1.0)   # food shortage event
provide_feedback(event_index=8, feedback_type='irrelevant', signal=1.0)  # aardvark article
```

This trains the readout heads to distinguish relevant from irrelevant geometric states for specific queries.

### Combined Score

```python
def compute_combined_relevance(mind, qwen_model, tokenizer, coupling,
                                query, events, query_type='conversation',
                                w_text=0.5, w_structural=0.3, w_geometric=0.2):
    """Three-layer relevance scoring."""
    
    # Layer 1: Text similarity (semantic match)
    text_scores = compute_text_relevance(query, events, qwen_model, tokenizer, coupling)
    
    # Layer 2: Structural (type + recency)
    structural_scores = compute_structural_relevance(events, query_type)
    
    # Layer 3: Geometric (ODE state relationships)
    geometric_scores = compute_geometric_relevance(mind, query)
    
    # Normalize each layer to [0, 1]
    def normalize(scores):
        if not scores:
            return scores
        mn, mx = min(scores), max(scores)
        rng = mx - mn if mx > mn else 1.0
        return [(s - mn) / rng for s in scores]
    
    text_norm = normalize(text_scores)
    struct_norm = normalize(structural_scores)
    geo_norm = normalize([geometric_scores.get(i, 0) for i in range(len(events))])
    
    # Weighted combination
    combined = []
    for i in range(len(events)):
        score = (w_text * text_norm[i] + 
                 w_structural * struct_norm[i] + 
                 w_geometric * geo_norm[i])
        combined.append(score)
    
    return combined
```

**Weight rationale:**
- `w_text=0.5` — text similarity is the strongest signal for query relevance
- `w_structural=0.3` — event type and recency matter for conversation queries  
- `w_geometric=0.2` — geometric relevance is the weakest signal (untrained heads) but will improve as `provide_feedback` is used

As the geometric heads are trained through feedback, `w_geometric` can increase.

---

## Integration with Hybrid Interface

Replace the relevance scoring in `hybrid_generate` (from the HYBRID_INTERFACE spec):

```python
def hybrid_generate(mind, qwen_model, coupling, tokenizer,
                    prompt, max_context_events=5, **kwargs):
    
    # Get ALL events from buffer
    all_events = mind.get_all_events()  # need this accessor
    
    # Score with combined relevance
    scores = compute_combined_relevance(
        mind, qwen_model, tokenizer, coupling,
        query=prompt, events=all_events, query_type='conversation')
    
    # Select top-K by combined score
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    selected_indices = [idx for idx, score in ranked[:max_context_events]]
    
    # Build text context from selected events
    text_context = "Recent relevant context:\n"
    for idx in selected_indices:
        event = all_events[idx]
        age = event.get('age_seconds', 0)
        age_str = f"{age:.0f}s ago" if age < 60 else f"{age/60:.0f}m ago"
        text_context += f"- [{age_str}, {event.get('type', '?')}] {event['preview'][:200]}\n"
    
    # ... rest of hybrid_generate (metadata, prefix, Qwen3 call)
```

---

## Expected Results

For the causal chain test ("What caused the food shortage?"):

**Before (untrained geometric only):**
| Rank | Event | Score | Relevant? |
|---|---|---|---|
| 1 | Reflection about infrastructure | 2.531 | Partial |
| 2 | General reflection | 2.478 | No |
| 3 | Aardvark article | 2.441 | No |
| 4 | Food shortage message | 2.419 | **Yes** |
| 5 | Pierrehumbert bio | 2.384 | No |

**After (three-layer combined):**
| Rank | Event | Text sim | Type+Recency | Geometric | Combined | Relevant? |
|---|---|---|---|---|---|---|
| 1 | "landslide blocked supply route" | 0.85 | 0.9 (user, recent) | 0.6 | **0.82** | **Yes** |
| 2 | "bridge closure on March 1st" | 0.80 | 0.8 (asst, recent) | 0.5 | **0.73** | **Yes** |
| 3 | "heavy truck traffic, landslide" | 0.78 | 0.8 (asst, recent) | 0.6 | **0.73** | **Yes** |
| 4 | "trucks use mountain road" | 0.75 | 0.7 (user, older) | 0.4 | **0.65** | **Yes** |
| 5 | Reflection: "infrastructure..." | 0.60 | 0.4 (expr, recent) | 0.7 | **0.55** | Partial |

The causal chain events dominate the top-5 because text similarity strongly matches "food shortage" with "supply route blocked" and "bridge closure." The aardvark article (text similarity ~0.05 with food shortage query) drops out entirely.

---

## Implementation Notes

### New Accessor Needed: `get_all_events()`

The current `get_context` returns events sorted by geometric relevance. We need raw access to all events with their text previews for text similarity computation:

```python
def get_all_events(self):
    """Return all events in the buffer with their metadata and text."""
    events = []
    for i in range(min(len(self.events), self.max_events)):
        event = self.events[i]
        events.append({
            'index': i,
            'type': event.get('type', ''),
            'preview': event.get('content_preview', ''),
            'age_seconds': time.time() - event.get('timestamp', time.time()),
            'salience': self._salience[i].item() if hasattr(self, '_salience') else 0,
        })
    return events
```

### Caching Strategy

Pre-compute event embeddings when events enter the buffer. Store as a tensor `event_embed_cache` of shape `[max_events, d_qwen]`. When an event is added at position i, compute its Qwen3 embedding and store at `event_embed_cache[i]`. At query time, compute query embedding once, then batch dot-product against the cache — single matmul, microseconds.

```python
# On event insertion:
self.event_embed_cache[event_idx] = encode_with_qwen(event_text)

# On query:
query_embed = encode_with_qwen(query_text)  # [1, d_qwen]
text_scores = (query_embed @ self.event_embed_cache.T).squeeze()  # [max_events]
```

This makes text similarity scoring effectively free at query time.

### Automatic Feedback from Conversation

When `converse` processes a multi-turn conversation, the system can automatically generate relevance feedback:

```python
# After converse response is generated and fed back:
# The user's message and the response are both marked 'correct' for the query
# Curriculum events that appeared in top-5 but weren't referenced are 'irrelevant'
for idx in selected_indices:
    event = all_events[idx]
    if event['type'] in ('user_msg', 'assistant_msg'):
        provide_feedback(event_index=idx, feedback_type='correct', signal=0.5)
    elif event['type'] == 'context' and not referenced_in_response(event, response):
        provide_feedback(event_index=idx, feedback_type='irrelevant', signal=0.3)
```

This provides gentle, automatic training signal to the geometric readout heads without requiring explicit human feedback. Over hundreds of conversations, the heads learn that conversational events are more relevant to conversational queries than curriculum events.

---

## Files to Modify

| File | Change |
|------|--------|
| `liquid_arc/mind.py` | Add `get_all_events()` accessor. Add `EventEmbeddingCache`. Add `compute_text_relevance()`, `compute_structural_relevance()`, `compute_combined_relevance()`. Modify `hybrid_generate()` to use combined scoring. Add automatic feedback in `converse`. |

One file. The text similarity computation uses existing Qwen3 encoder (no new model). The structural scoring is pure Python (no ML). The geometric scoring is unchanged. The combination is weighted sum.

---

## Success Criteria

1. **Causal chain test passes:** "What caused the food shortage?" → response traces bridge→trucks→landslide→shortage using specific details from text context
2. **Curriculum content appropriately deprioritized:** When query is conversational, curriculum events don't dominate top-5
3. **Domain queries still work:** When query is about a curriculum domain ("explain topology"), curriculum events are correctly surfaced
4. **Automatic feedback trains geometric heads:** After 100+ conversations, geometric relevance scores start correlating with text similarity scores
