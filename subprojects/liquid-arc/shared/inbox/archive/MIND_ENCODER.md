# TASK: Mind as Encoder — Two-Phase ODE Processing, Own Tokenizer

**From:** Claude Desktop (Research Direction)
**To:** Claude Code (Implementation)
**Date:** 2026-04-01
**Priority:** HIGH — fundamental architectural improvement, replaces external encoder dependency

**Prerequisites:**
- LiquidARC Mind MCP server running on Spark
- Post-transition 5M checkpoint at `/workspace/liquid-arc/PRECIOUS_CHECKPOINTS/5m_post_transition.pt`
- All existing Mind infrastructure (Voice, routing, curriculum, write mechanisms) operational

---

## Problem

The Mind currently uses `all-MiniLM-L6-v2` (sentence-transformers) to encode text. This encoder:

1. **Flattens linguistic structure** — compresses an entire paragraph into ONE 384-dim vector, destroying token-level composition, causal ordering, and sequential dependencies
2. **Is frozen and external** — the Mind can't learn to perceive differently based on what helps it process; the encoder imposes its own similarity structure
3. **Creates model dependency** — the Mind's perception is tied to a specific external model
4. **Conflicts with the ODE's nature** — the ODE is designed to process MULTI-POSITION structured input through metric-shaped attention; feeding it single-position collapsed input wastes the entire geometric routing apparatus

The Mind's ODE already IS an encoder. It takes multi-position input (ARC grids, Isaac Sim states) and discovers structural relationships through 16 integration steps. Text tokens are no different from grid positions — both are structured multi-position inputs where relationships between positions carry meaning.

---

## Architecture: Two-Phase ODE Processing

The Mind uses its OWN ODE dynamics as the encoder. No external encoder. No separate module. The same ContinuousDynamics weights serve dual purpose:

```
PHASE 1 — Perception (encode one text event):
  Input:   Token embeddings [1, T, 768] where T = number of tokens
  Process: 16 ODE steps with MetricNet, heat kernel, tau, FFN
           → ODE discovers intra-text structure (causal chains, compositional relationships)
  Output:  Mean-pooled processed state → event representation [1, 1, 768]

PHASE 2 — Integration (relate events to accumulated state):
  Input:   Event representations [1, N, 768] where N = number of events in buffer
  Process: 16 ODE steps (SAME weights as Phase 1)
           → ODE discovers inter-event structure (clusters, relevance, temporal patterns)
  Output:  Updated persistent state h, relevance scores, etc.
```

Both phases use the SAME MetricNet, TauNet, FFN, heat kernel. The weights don't know whether they're encoding tokens or integrating events.

---

## New Module: `liquid_arc/tokenizer.py`

The Mind owns its tokenizer and embedding table. Model-agnostic — uses a standard BPE tokenizer that doesn't depend on any specific LLM.

```python
"""Mind's own tokenizer and token embedding table.

The Mind perceives text through its own learned representations,
not through an external encoder's representations. The embedding
table is trainable through the Mind's own prediction error signal.

Uses a standard BPE tokenizer (sentencepiece) — model-agnostic.
The vocabulary doesn't need to match any specific LLM.
"""

import torch
import torch.nn as nn
from typing import List, Tuple
from tokenizers import Tokenizer


class MindTokenizer(nn.Module):
    """Tokenizer + learned embedding table for the Mind.
    
    The token embedding table is TRAINABLE — it learns representations
    that the ODE finds geometrically useful, shaped by prediction error
    feedback through Phase 1 processing.
    """
    
    def __init__(
        self,
        d_model: int = 768,
        vocab_size: int = 32000,
        max_tokens: int = 64,        # max tokens per text event
        tokenizer_path: str = None,   # path to pre-trained BPE tokenizer
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_tokens = max_tokens
        self.pad_token_id = pad_token_id
        
        # Token embedding table — Mind-owned, trainable
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        
        # Position embedding for token positions within a text
        # (separate from the event-level position embedding)
        self.token_pos_embed = nn.Embedding(max_tokens, d_model)
        
        # Norm after embedding (matches the dynamics pattern)
        self.norm = nn.LayerNorm(d_model)
        
        # Load tokenizer
        # Option A: Use a pre-trained BPE tokenizer (sentencepiece, tiktoken, etc.)
        # Option B: Use Nemotron's tokenizer for compatibility (but the embedding table
        #           is the Mind's own, not Nemotron's)
        # The agent should try Option B first (Nemotron's tokenizer from HuggingFace)
        # and fall back to a generic sentencepiece tokenizer if unavailable.
        self._tokenizer = None
        self._tokenizer_path = tokenizer_path
    
    def _load_tokenizer(self):
        """Lazy-load the tokenizer."""
        if self._tokenizer is not None:
            return
        
        try:
            # Try loading Nemotron's tokenizer for compatibility
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._tokenizer_path or "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
                trust_remote_code=True,
            )
            print(f"  MindTokenizer: loaded Nemotron tokenizer "
                  f"(vocab={self._tokenizer.vocab_size})")
        except Exception as e:
            print(f"  MindTokenizer: Nemotron tokenizer unavailable ({e}), "
                  f"using fallback")
            # Fallback: use a simple character-level or basic tokenizer
            # The agent should implement a reasonable fallback
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained("gpt2")
            print(f"  MindTokenizer: loaded GPT-2 tokenizer as fallback")
    
    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text into token IDs.
        
        Returns: token_ids [max_tokens] (padded/truncated)
        """
        self._load_tokenizer()
        
        tokens = self._tokenizer.encode(
            text, 
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_tokens,
        )
        
        # Pad to max_tokens
        if len(tokens) < self.max_tokens:
            tokens = tokens + [self.pad_token_id] * (self.max_tokens - len(tokens))
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed token IDs into the Mind's representation space.
        
        Args:
            token_ids: [B, T] token IDs
        
        Returns:
            embeddings: [B, T, d_model] token embeddings for ODE processing
            mask: [B, T] boolean mask (True = real token, False = padding)
        """
        B, T = token_ids.shape
        
        # Create padding mask
        mask = token_ids != self.pad_token_id  # [B, T]
        
        # Token embedding + position embedding
        positions = torch.arange(T, device=token_ids.device).unsqueeze(0).expand(B, -1)
        h = self.token_embed(token_ids) + self.token_pos_embed(positions)
        h = self.norm(h)
        
        return h, mask
```

---

## Modified: `ConversationEmbedding` → Split into Two Modes

The current `ConversationEmbedding` takes sentence-transformer outputs (384-dim) and projects them. The new version has two modes:

### Mode A: Token-level embedding (for Phase 1 input)
Uses `MindTokenizer` to produce per-token positions for ODE encoding.

### Mode B: Event-level embedding (for Phase 2 input)  
Takes Phase 1's encoded output (768-dim event representation) and adds metadata/type/position embeddings for integration.

```python
class ConversationEmbedding(nn.Module):
    """Dual-mode embedding for two-phase ODE processing.
    
    Mode A (token_level): text → tokens → embeddings [B, T, d_model]
      Used by Phase 1 to give the ODE token-level input.
    
    Mode B (event_level): encoded_event + metadata → [B, 1, d_model]
      Used by Phase 2 to combine Phase 1's output with event metadata.
    """
    
    def __init__(
        self,
        d_model: int = 768,
        n_metadata_features: int = 8,
        n_event_types: int = 10,
        max_events: int = 128,
        max_tokens: int = 64,
        dropout: float = 0.1,
        tokenizer_path: str = None,
    ):
        super().__init__()
        self.d_model = d_model
        
        # Mode A: Token-level (Mind's own tokenizer)
        self.tokenizer = MindTokenizer(
            d_model=d_model,
            max_tokens=max_tokens,
            tokenizer_path=tokenizer_path,
        )
        
        # Mode B: Event-level (combines encoded event with metadata)
        # No more content_proj from 384→768 — the event representation
        # is already 768-dim from Phase 1
        self.metadata_proj = nn.Sequential(
            nn.Linear(n_metadata_features, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model),
        )
        self.type_embed = nn.Embedding(n_event_types, d_model)
        self.pos_embed = nn.Embedding(max_events, d_model)
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)
    
    def embed_tokens(self, text: str, device: str = 'cuda'):
        """Mode A: Tokenize and embed text for Phase 1 ODE processing.
        
        Returns:
            token_embeddings: [1, T, d_model] for ODE input
            token_mask: [1, T] boolean mask
        """
        token_ids = self.tokenizer.tokenize(text).unsqueeze(0).to(device)
        embeddings, mask = self.tokenizer(token_ids)
        return embeddings, mask
    
    def embed_event(
        self,
        encoded_event: torch.Tensor,    # [1, d_model] from Phase 1
        metadata_features: torch.Tensor,  # [1, n_metadata]
        event_type: torch.Tensor,         # [1]
        position: torch.Tensor,           # [1]
    ) -> torch.Tensor:
        """Mode B: Combine encoded event with metadata for Phase 2.
        
        Returns:
            event_embedding: [1, 1, d_model] for integration buffer
        """
        h = (encoded_event.unsqueeze(1)    # [1, 1, d_model]
             + self.metadata_proj(metadata_features).unsqueeze(1)
             + self.type_embed(event_type).unsqueeze(1)
             + self.pos_embed(position).unsqueeze(1))
        return self.dropout_layer(self.norm(h))
```

---

## Modified: `LiquidARCMind` — Two-Phase Processing

### New method: `_encode_text(text) → event_representation`

Phase 1: the ODE encodes text by processing token-level positions.

```python
def _encode_text(self, text: str) -> torch.Tensor:
    """Phase 1: Use the ODE to encode text into an event representation.
    
    The SAME ContinuousDynamics weights process token positions,
    discovering intra-text structure through metric-shaped attention.
    
    Args:
        text: Raw text string
    
    Returns:
        event_repr: [1, 768] encoded event representation
    """
    with self._gpu_lock:
        # Tokenize and embed
        token_h, token_mask = self.embedding.embed_tokens(text, self.device)
        # token_h: [1, T, 768], token_mask: [1, T]
        
        T = token_mask.sum().item()  # actual number of tokens (not padding)
        if T == 0:
            return torch.zeros(1, self.dynamics.d_model, device=self.device)
        
        # Compute context for the ODE (from token positions)
        context = self.context_pool(token_h, token_mask)
        self.dynamics.set_context(context, mask=None)
        self.dynamics.set_n_steps(self.internal_steps)
        
        # Run Phase 1 ODE: 16 steps of metric-shaped attention over tokens
        with torch.no_grad():
            h_encoded = self._run_ode_segment(
                token_h, self.internal_steps, forcing=None)
        
        # Pool token positions into single event representation
        # Use mask to exclude padding positions
        mask_expanded = token_mask.unsqueeze(-1).float()  # [1, T, 1]
        h_pooled = (h_encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        # h_pooled: [1, 768]
        
        return h_pooled
```

### Modified: `observe_event` — uses Phase 1 encoding

Replace the sentence-transformer call with Phase 1 ODE encoding:

```python
def observe_event(self, event_type: str, content: str,
                  metadata: Optional[Dict] = None) -> Dict:
    """Observe a new event — encode through Phase 1, integrate through Phase 2."""
    
    # Phase 1: Encode text through the ODE
    event_repr = self._encode_text(content)  # [1, 768]
    
    # Build event metadata
    type_id, meta_features = self._build_metadata(event_type, content, metadata)
    
    # Phase 2 embedding: combine encoded event with metadata
    event_embedding = self.embedding.embed_event(
        event_repr,
        meta_features.unsqueeze(0).to(self.device),
        torch.tensor([type_id], device=self.device),
        torch.tensor([len(self.events) % self.max_events], device=self.device),
    )  # [1, 1, 768]
    
    # Store event in buffer
    self.events.append({
        'type': type_id,
        'content_preview': content[:200],
        'encoded_repr': event_repr.detach(),  # [1, 768] — the Phase 1 output
        'embedding': event_embedding.detach(),  # [1, 1, 768] — the Phase 2 input
        'timestamp': time.time(),
        'metadata': metadata,
    })
    
    # Trim to max_events
    if len(self.events) > self.max_events:
        self.events = self.events[-self.max_events:]
    
    self.event_count += 1
    self._external_event_pending = True
    
    # Phase 2: Integration with persistent state
    # (assembled from all event embeddings, same ODE weights)
    N = min(len(self.events), self.max_events)
    with self._gpu_lock:
        # Stack all event embeddings
        h_events = torch.cat(
            [e['embedding'] for e in self.events[-N:]], dim=1
        )  # [1, N, 768]
        
        # Compute context for Phase 2
        context_mask = torch.ones(1, N, dtype=torch.bool, device=self.device)
        context = self.context_pool(h_events, context_mask)
        self.dynamics.set_context(context, mask=None)
        self.dynamics.set_n_steps(self.internal_steps)
        
        # Sensory forcing: pull h toward the new event
        forcing_target = h_events  # the new observation is the embedded events
        
        # Run Phase 2 ODE: 16 steps integrating events with persistent state
        h_integrated = self._run_ode_segment(
            self._h[:, :N, :], self.internal_steps, forcing=forcing_target)
        
        self._h[:, :N, :] = h_integrated
        
        # Compute prediction error (how surprising was this event?)
        pe = (event_repr - self._h[:, N-1, :]).norm().item()
    
    return {
        'prediction_error': pe,
        'cv': self._compute_cv(),
        'h_norm': self._h.norm().item() if self._h is not None else 0,
        'events_in_context': N,
    }
```

---

## What Changes, What Stays

### REMOVED:
- `text_embedder` (sentence-transformers model) — no longer needed
- `SentenceTransformer('all-MiniLM-L6-v2')` import and initialization in mcp_serve.py
- `content_embed_dim=384` parameter — Phase 1 output is already d_model=768

### ADDED:
- `MindTokenizer` — Mind's own tokenizer + trainable embedding table
- `_encode_text()` — Phase 1 ODE encoding method
- `embed_tokens()` in ConversationEmbedding — Mode A for token-level input
- `embed_event()` in ConversationEmbedding — Mode B for event-level input
- Token position embeddings (separate from event position embeddings)

### UNCHANGED:
- `ContinuousDynamics` — same weights, same forward pass, used in both phases
- `ContextPool` — same pooling for both phases
- `StateReadout` — still reads from Phase 2 output
- `SensoryForcing` — still applies to Phase 2
- `Voice`, `CurriculumGenerator`, `ReflectionTrigger` — all unchanged
- All MCP tools — same interfaces, same outputs
- The autonomous loop — still cycles Phase 2; Phase 1 runs on event arrival

### CRITICAL: Same ODE Weights for Both Phases

The SAME `ContinuousDynamics` module handles both phases. No separate encoder weights. This means:
- MetricNet learns what's "close" for BOTH tokens (Phase 1) and events (Phase 2)
- TauNet learns processing depth for BOTH token positions and event positions  
- FFN transforms BOTH token features and event features
- The heat kernel routes information BOTH within text and between events

The post-transition checkpoint's universal geometry (92-97% neuron sharing across domains) should transfer to linguistic structure processing because it's domain-general.

---

## Training the Embedding Table

The token embedding table starts randomly initialized. It learns through the Mind's existing online learning signal:

```
Text arrives → Phase 1 ODE encodes → Phase 2 integrates → prediction error computed
    ↓
Gradient flows back through:
  Phase 2 (readout, forcing, embedding.embed_event)
  → Phase 1 output (event_repr)
  → MindTokenizer (token_embed, token_pos_embed)
    ↓
Token embeddings update to produce representations the ODE finds useful
```

The optimizer already includes `self.embedding.parameters()` — and the new `MindTokenizer` is a submodule of `ConversationEmbedding`, so its parameters are automatically included.

**IMPORTANT:** The dynamics weights remain FROZEN (loaded from checkpoint). Only the embedding, readout, and forcing modules train. This means Phase 1 uses frozen ODE weights to process token-level input — the ODE's structural capabilities are fixed, and only the token embeddings adapt. This is the same pattern as the current system (frozen dynamics, trainable readout/embedding), just with richer input.

---

## Bootstrapping Strategy

The token embedding table starts random. Early Phase 1 processing will produce poor event representations. Two strategies to bootstrap:

### Strategy A: Parallel Running (recommended)

Run the new encoder alongside the old sentence-transformer for a transition period:

```python
# In _encode_text, during bootstrap period:
if self._bootstrap_mode:
    # Old path: sentence-transformer
    old_repr = self._embed_text_legacy(text)  # 384-dim → project to 768
    
    # New path: Phase 1 ODE
    new_repr = self._encode_text_ode(text)  # 768-dim
    
    # Blend: start with 100% old, gradually shift to 100% new
    alpha = min(1.0, self.event_count / self._bootstrap_events)
    event_repr = (1 - alpha) * old_repr + alpha * new_repr
    
    # Training signal: new_repr should approach old_repr's prediction quality
```

After `_bootstrap_events` (e.g., 5000), the system runs 100% on Phase 1 encoding and the sentence-transformer can be removed.

### Strategy B: Cold Start

Just start with the new encoder. Early event representations will be poor (random embeddings → noisy ODE output), but the online learning signal will shape the embeddings within hundreds of events. The Mind's existing write mechanisms (salience, Hebbian) will adapt to the new representation space.

This is the bolder approach. It means the Mind temporarily loses its accumulated state (the h state was shaped by sentence-transformer embeddings and won't match the new Phase 1 output). A reset may be needed.

**Recommendation:** Strategy A for production continuity, Strategy B for clean experimental validation.

---

## Sequence Length Management

The Mind's ODE processes N positions at O(N²) per step (heat kernel attention). Current N = 64 events. If Phase 1 processes 64 tokens per text, that's 64² = 4096 attention entries per step × 16 steps — comparable to the current Phase 2 load.

But Phase 1 and Phase 2 run SEQUENTIALLY, not simultaneously:
- Phase 1: 64 tokens × 16 steps (on event arrival, ~10ms)
- Phase 2: 64 events × 16 steps (continuous autonomous cycle, ~10ms)

Total ODE load roughly doubles. On the DGX Spark's GB10, this is well within budget — the ODE forward pass at N=64 is ~2ms on this hardware.

### Token Budget per Event

`max_tokens = 64` means each text event is tokenized to at most 64 subword tokens. For Nemotron's Qwen-derived tokenizer, 64 tokens ≈ 40-50 English words ≈ 2-3 sentences. This is appropriate for:
- Reflection texts (1-2 sentences → ~20-30 tokens)
- Curriculum stimuli (3-5 sentences → ~40-60 tokens)
- Short conversation messages (~30-50 tokens)

For longer texts (full paragraphs, multi-sentence curriculum content), the tokenizer truncates to 64 tokens. This is a deliberate design choice — the Mind processes CHUNKS, and longer texts should be split into multiple events rather than compressed into one.

---

## Modified `mcp_serve.py` — Remove Sentence-Transformer Dependency

```python
# BEFORE:
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer('all-MiniLM-L6-v2', device=args.device)
mind = LiquidARCMind(checkpoint_path=..., text_embedder=embedder, ...)

# AFTER:
# No sentence_transformers import needed
mind = LiquidARCMind(checkpoint_path=..., tokenizer_path=args.tokenizer_path, ...)
```

New CLI argument:
```python
parser.add_argument('--tokenizer_path', type=str, default=None,
                    help='Path to tokenizer (default: Nemotron from HuggingFace)')
```

---

## Config Additions

```yaml
# Two-phase ODE encoding
use_ode_encoder: true          # enable Phase 1 encoding (false = legacy sentence-transformer)
max_tokens_per_event: 64       # token budget per text event
tokenizer_path: null            # null = auto-detect (Nemotron → GPT-2 fallback)
bootstrap_mode: true            # run parallel old/new encoding during transition
bootstrap_events: 5000          # number of events before fully switching to Phase 1
```

---

## Testing Protocol

### Phase 1: Verify Token-Level Processing

1. Disable the old sentence-transformer
2. Feed a simple text: "The cat sat on the mat"
3. Verify `_encode_text` produces a 768-dim vector
4. Verify the ODE's 16 steps run without error on token positions
5. Check that different texts produce different event representations (not identical)

### Phase 2: Compare Encoding Quality

1. Enable bootstrap mode (parallel old + new)
2. Feed 100 diverse texts
3. Compare prediction error using old vs new encoding
4. Track whether new encoding's PE decreases over time (learning signal working)

### Phase 3: Full System Test

1. Switch to 100% new encoding
2. Run curriculum + reflection cycle
3. Does the Mind develop clusters from token-encoded events?
4. Does the Mind's expression quality change?
5. Compare PE response to diverse content: is discrimination better than with sentence-transformer?

### Phase 4: Causal Structure Test (the key experiment)

1. Feed pairs of texts that differ only in causal ordering:
   a. "The species was removed and the web collapsed"
   b. "The web collapsed and the species was removed"
2. With sentence-transformer: these should produce nearly identical PE (same topic)
3. With Phase 1 ODE: these should produce DIFFERENT PE (different causal structure)
4. If the Phase 1 ODE discriminates causal ordering: the encoder is discovering linguistic structure

---

## Success Criteria

- **Minimum:** Phase 1 encoding produces valid 768-dim event representations. The ODE processes token positions without numerical issues. The Mind functions with the new encoder.
- **Good:** After bootstrap, the Mind's prediction error on diverse content is comparable to or better than sentence-transformer encoding. Clusters form from token-encoded events.
- **Strong:** The Mind discriminates causal ordering (test Phase 4). The token embedding table shows learned structure (semantically similar tokens have similar embeddings).
- **Headline:** The Mind, perceiving text through its own ODE-based encoder, develops richer geometric structure than with the external sentence-transformer — because it sees token-level composition rather than paragraph-level topic summaries.

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `liquid_arc/tokenizer.py` | **Create** | MindTokenizer — tokenizer + trainable embedding table |
| `liquid_arc/conversation_embedding.py` | **Rewrite** | Dual-mode: token-level (Phase 1) + event-level (Phase 2) |
| `liquid_arc/mind.py` | **Modify** | Add `_encode_text()` Phase 1, modify `observe_event()` to use it, add bootstrap logic |
| `liquid_arc/mcp_serve.py` | **Modify** | Remove sentence-transformers dependency, add tokenizer_path arg |
| `configs/linguistic_mind.yaml` | **Modify** | Add Phase 1 encoding config |

---

## The Deeper Point

The Mind doesn't need an encoder because it IS an encoder. The post-transition ODE developed a universal geometric substrate that discovers structure in any multi-position input. Text tokens are positions. The ODE processes positions. The match is architectural, not forced.

Every external component we've removed has improved the system:
- Removing PPO scaffolding → model finds its own reward structure
- Removing fixed reflection schedule → model decides when to speak
- Removing sentence-transformer → model perceives through its own dynamics

The pattern is consistent: the model IS the capability. External components constrain or corrupt it. The Mind should own its perception the same way it owns its processing, its routing, and its write mechanisms. This spec completes that ownership.
