# CONTEXT PROMPT SCALING — Architecture for Multi-Turn Conversations

## The Problem

The buffer elimination spec says "build context prompt from conversation history." In a 20-turn conversation:

```
10 user messages × ~30 tokens  = 300 tokens
10 assistant responses × ~200 tokens = 2000 tokens
Total: 2300 tokens
```

Running Qwen3 forward on 2300 tokens to extract deltas: 3-5 seconds. ODE over [1, 2300, 2560]: heavy. Bias [2300 × 2300]: 5.3M entries. Doesn't scale.

And most of those 2000 assistant tokens are the model's OWN verbose responses — reasoning traces, restatements, repetitions. They carry little structural signal. Including them in the ODE input is circular: the ODE shaped the response, now the response shapes the ODE.

## Recommendation: Delta extraction on USER messages only

### Rationale

Structural signal comes from user messages — they introduce facts, entities, causal links, new topics. Model responses are DERIVED from user input + geometric routing. The ODE should process the SOURCES of structure (user input), not the OUTPUTS of structure (model responses).

### Architecture

```
PERSISTENT:
  h_state [1, K, d]          K=32-64, fixed size, accumulated geometric structure
  event_list: [{role, text}]  conversation history (text only, for prompt building)

PER converse() CALL:

  1. PROMPT BUILDING (for Qwen3 generation):
     - Assemble full conversation text: all user + assistant messages
     - This is Qwen3's input — needs full context for ICL
     - Can be up to Qwen3's context window (4K-40K tokens)
  
  2. DELTA EXTRACTION (for ODE, USER MESSAGES ONLY):
     - Extract deltas from user messages only
     - Concatenate: [1, N_user, d] where N_user = total user tokens
     - For 20 turns: ~300 tokens, not 2300
     - Takes ~0.5s, not 3-5s
  
  3. ODE INTEGRATION:
     - Combine: [h_state ; user_deltas] → [1, K + N_user, d]
     - 16 Euler steps, heat kernel routes between all positions
     - h_state positions carry accumulated structure
     - User delta positions carry current structural input
  
  4. BIAS COMPUTATION:
     - Compute state cosine + displacement over user positions → B[N_user × N_user]
     - Map user token positions to their locations in the full prompt
     - Expand B to [N_prompt × N_prompt]:
       * User↔User positions: B values from ODE
       * User↔Assistant positions: zero (native attention handles)
       * Assistant↔Assistant positions: zero
     - This is a SPARSE bias — most entries are zero
  
  5. GENERATION:
     - Inject expanded bias into Qwen3's attention hooks
     - Qwen3 generates with: native attention (full) + geometric bias (user positions)
     - Stop sequences, repetition penalty, strip thinking traces
  
  6. STATE UPDATE:
     - h_state ← ODE_output[:, :K, :] (persistent state evolved by this call)
     - Append user message + assistant response to event_list
     - Optionally: extract deltas from GENERATED response, run one more ODE step
       to update h_state (the model's own output informs the state)
```

### Why user-only deltas are sufficient

The structural relationships the ODE discovers are between USER-introduced concepts:
- "bridge closed" ↔ "trucks rerouted" ↔ "landslide" ↔ "food shortage"
- These are all USER tokens

The assistant's response just TRACES these relationships — it doesn't introduce new structural content. The ODE doesn't need to "discover" that the model's own response is related to the user's input — that's tautological.

Exception: in multi-agent or tool-use scenarios, the assistant's response might introduce genuinely new information (tool results, retrieved documents). These could be treated as pseudo-user events for delta extraction.

### Scaling comparison

| Turns | Full prompt deltas | User-only deltas | Reduction |
|-------|-------------------|-----------------|-----------|
| 5     | 700 tokens        | 150 tokens      | 4.7×      |
| 10    | 2300 tokens       | 300 tokens      | 7.7×      |
| 20    | 5000 tokens       | 600 tokens      | 8.3×      |
| 50    | 12000 tokens      | 1500 tokens     | 8.0×      |

The ODE processes ~8× fewer tokens, making real-time response feasible even for long conversations.

### Bias expansion implementation

```python
def expand_bias_to_prompt(B_user, user_token_positions, n_prompt_tokens):
    """Expand user-only bias to full prompt dimensions.
    
    Args:
        B_user: [N_user, N_user] bias from ODE routing
        user_token_positions: list of ints — which prompt positions are user tokens
        n_prompt_tokens: total tokens in generation prompt
    
    Returns:
        B_full: [N_prompt, N_prompt] sparse bias (zeros for non-user positions)
    """
    B_full = torch.zeros(n_prompt_tokens, n_prompt_tokens, device=B_user.device)
    
    # Map user positions into the full prompt
    for i, pi in enumerate(user_token_positions):
        for j, pj in enumerate(user_token_positions):
            B_full[pi, pj] = B_user[i, j]
    
    return B_full
```

This is a sparse scatter — most of B_full is zero. Qwen3's native attention handles all non-user positions normally. The geometric bias only modulates attention between user-introduced concepts.

### What happens to assistant response tokens in the ODE

Two options:

**Option A: Ignore completely.** The ODE never sees model responses. h_state accumulates from user input only. The model's responses are in the text context prompt for ICL but don't influence geometric state.

**Option B: Lightweight integration.** After generation, extract deltas from the response and run ONE ODE step to update h_state (not for bias computation, just for state accumulation). This lets the persistent state "know" what the model said, which may help PE discrimination on future user messages.

Recommend starting with Option A (simpler, cleaner) and adding Option B if PE discrimination degrades.

### Context prompt building at scale

For the text prompt itself (what Qwen3 sees for ICL):

**Short conversations (<10 turns):** Include everything verbatim. Qwen3 handles 4K tokens easily.

**Medium conversations (10-30 turns):** Include last 5 turns verbatim + summarized earlier turns. The summary can be as simple as first sentence of each earlier message.

**Long conversations (30+ turns):** Include last 5 turns verbatim + topic-based retrieval from earlier turns (using event metadata: timestamps, topics, PE values). This is where the ODE's PE signal becomes useful — events with low PE (highly familiar) can be skipped; events with high PE (novel/surprising) are retained.

The context prompt is for Qwen3's ICL. The ODE is for geometric routing. They serve different functions and should be built differently.

### Migration from current buffer

1. Stop using token_buffer for bias computation
2. Per converse() call: extract deltas from user messages in event_list
3. Compute bias over user tokens, expand to prompt dimensions
4. Remove token_buffer entirely after validation
5. Simplify autonomous loop to h_state consolidation
