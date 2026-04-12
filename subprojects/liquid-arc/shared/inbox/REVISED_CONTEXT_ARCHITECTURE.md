# REVISED: Context Architecture for ODE Processing

## The Correction

User messages only is wrong. Structural signal emerges in the EXCHANGE — the model's reasoning traces, causal chain articulation, corrections, and synthesis carry structural content that user messages alone don't contain.

Example: User provides three disconnected facts. The model's response articulates "A caused B caused C" — that causal chain structure exists in the RESPONSE, not the user input. If the ODE only sees user input, it sees three isolated facts with no chain.

## Revised Architecture: h_state + Recent Window

```
PERSISTENT:
  h_state [1, K, d]     K=32-64, carries compressed history of ALL prior turns

PER-CALL:
  recent_window          Last W full turns (user + assistant), both sides
                         W = 3-5 turns, configurable
                         Includes model's reasoning traces, structural discoveries
  
  current_input          The new user message being responded to
```

### Processing flow

```
1. BUILD GENERATION PROMPT
   - System prompt
   - Full conversation history (for Qwen3 ICL)
   - Current user message
   → N_prompt tokens (can be large, that's fine — Qwen3 handles it natively)

2. EXTRACT DELTAS (for ODE)
   - Recent window: last W turns (user + assistant messages)
   - Current user message
   - Run through Qwen3 forward → layer 18 deltas
   → [1, N_window, d] where N_window = tokens in recent W turns + current message
   - For W=3: typically 300-500 tokens (manageable)
   - For W=5: typically 500-800 tokens (still manageable)

3. ODE INTEGRATION
   - Combine: [h_state ; window_deltas] → [1, K + N_window, d]
   - 16 Euler steps, heat kernel routes between ALL positions
   - h_state positions carry compressed history
   - Window positions carry full recent structure
   - Heat kernel connects history to recent context

4. BIAS COMPUTATION
   - Compute bias over window positions → B[N_window × N_window]
   - Map window token positions to their locations in the generation prompt
   - Expand to prompt: window positions get B values, older positions get zero
   
5. GENERATION
   - Qwen3 generates with:
     * Full context (ICL over entire conversation history)
     * Geometric bias on recent window positions
     * Native attention on older positions
   
6. STATE UPDATE
   - h_state ← ODE_output[:, :K, :]
   - When turns age out of the window, their structural contribution
     persists in h_state (compressed, not lost)
```

### Why the window includes assistant responses

The model's responses contain:
- **Causal articulations**: "A caused B because..." — the chain structure
- **Structural corrections**: "Actually, X is not related to Y" — boundary information
- **Synthesis**: "Both scenarios involve removing a critical element" — cross-domain links
- **Reasoning traces**: The step-by-step reasoning reveals WHICH tokens the model attended to

All of this is structural signal the ODE should process. Excluding it means the ODE is blind to half the conversation's structural evolution.

### Why the window is bounded (not the full history)

- **Computational**: W=5 turns ≈ 500-800 tokens. ODE over [64 + 700] = 764 positions is feasible in <1s. Full history (5000+ tokens) is not.
- **Structural**: Recent turns carry the active structural context. Turn 1's contribution from 20 turns ago has already been absorbed into h_state through incremental updates. Re-processing it adds no new structural information.
- **Quality**: The bias should reflect CURRENT structural relationships. A bias that equally weights turn 1 and turn 20 would dilute the current reasoning context.

### Window aging

When a turn ages out of the window (turn W+1 arrives, oldest turn exits):
- Its structural contribution already lives in h_state (from the ODE update when it WAS in the window)
- It remains in the text prompt for Qwen3 ICL (full history preserved)
- It just stops contributing to the geometric bias (which focuses on recent structure)

This is analogous to working memory in the brain: recent items are held in high-resolution (the window), older items are compressed into long-term storage (h_state), and both are accessible but in different ways.

### h_state update mechanism

Each call, h_state is updated by routing with the current window:

```
call 1: h_state routes with turns [1]         → h_state absorbs turn 1
call 2: h_state routes with turns [1,2]        → h_state absorbs turn 2
call 3: h_state routes with turns [1,2,3]      → h_state absorbs turn 3
call 4: h_state routes with turns [2,3,4]      → turn 1 exits window, lives in h_state
call 5: h_state routes with turns [3,4,5]      → turn 2 exits window, lives in h_state
```

Each ODE integration is 16 heat kernel steps. Information flows from window positions into h_state positions AND from h_state positions into window positions. The h_state is not passive storage — it actively influences how the current window is routed.

### Configuration

```yaml
ode_window_turns: 5          # how many recent turns in the ODE window
ode_window_max_tokens: 800   # hard cap on window tokens (truncate oldest if exceeded)
h_state_size: 64             # persistent state positions
include_assistant: true      # include model responses in window (MUST be true)
```

### What this replaces

- Token buffer (512/1024 slots) → eliminated
- Buffer management (dropping, priority, event_id) → eliminated  
- Curriculum injection into buffer → eliminated (curriculum stays separate)
- Bootstrap tokens → eliminated (h_state initializes from checkpoint)
- Full re-processing each call → only recent window + h_state
