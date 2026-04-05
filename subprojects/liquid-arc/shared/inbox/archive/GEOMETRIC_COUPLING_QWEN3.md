# TASK: Geometric Integration — LiquidARC × Qwen3-4B on DGX Spark

**From:** Claude Desktop (Research Direction)
**To:** Claude Code (Implementation)
**Date:** 2026-04-04
**Priority:** HIGH — next-generation architecture for the Emergence project

---

## Vision

LiquidARC is a continuous-time geometric processor. Its post-transition geometry self-organized routing structure that generalizes across domains. But it has no knowledge of the world — 5.5M parameters can't store what 4B parameters of dense transformer learned from trillions of tokens.

Qwen3-4B is a dense transformer containing compressed knowledge of human civilization — language, reasoning, code, science, math. But it's stateless: every forward pass starts fresh, no temporal integration, no persistent context beyond the context window.

The integration: **LiquidARC provides persistent curved-space dynamics. Qwen3-4B provides stateless flat-space knowledge lookup.** One state (LiquidARC's ODE state h(t)). One manifold (Qwen3's representation space as a knowledge landscape). The coupling is geometric — learned projections between representation spaces, no tokenization, no vocabulary, no linguistic interface.

In FGN v3 terms: standard transformer attention = heat kernel on a flat manifold. LiquidARC's dynamics = heat kernel on a curved manifold. The coupled system has variable curvature: flat where Qwen3 handles associative knowledge lookup, curved where LiquidARC adds temporal integration, persistent state, and path-dependent processing. The transformer's flat computation is a point in LiquidARC's curved geometry.

---

## Architecture

```
                    LiquidARC (persistent state, curved dynamics)
                    h(t) ∈ ℝ^768, evolving continuously
                         |
                    W_inject ∈ ℝ^(768 → 2048)  [learned projection]
                         |
                         ↓
              ┌──── Qwen3-4B residual stream ────┐
              │  Layer 0: Attention + FFN         │
              │  ...                              │
              │  Layer k: [+ LiquidARC injection] │ ← projected h(t) added here
              │  ...                              │
              │  Layer N: final hidden states     │
              └───────────────────────────────────┘
                         |
                    W_read ∈ ℝ^(2048 → 768)  [learned projection]
                         |
                         ↓
                    LiquidARC sensory forcing
                    h(t+dt) = h(t) + dt · dynamics(h, qwen_signal)
```

### Components

**LiquidARC (fluid metric checkpoint):**
- Checkpoint: `output_fluid/stage_b/step_10000.pt` (or PRECIOUS equivalent)
- Architecture: 5.5M params, d=768, d_metric_bottleneck=256, metric_rank=8
- Post-transition geometry via distillation chain (original transition → diagonal distillation → fluid metric extension)
- Carries: MetricNet (low-rank), TauNet, W_v, W_o, FFN, ContextPool
- The ONLY persistent state in the system

**Qwen3-4B:**
- Download from HuggingFace: `Qwen/Qwen3-4B` (base, not instruct — we want raw representations)
- Architecture: dense transformer, ~40 layers, d_model=2048 (verify from config), GQA, SwiGLU, RoPE
- ~8GB in bf16, fits trivially on 128GB Spark alongside LiquidARC
- STATELESS: no persistent state between calls. Reset on every forward pass.
- Frozen weights — Qwen3 is the knowledge manifold, not a trainable component

**Coupling layers (NEW, trainable):**
- `W_inject`: Linear(768, 2048) — projects LiquidARC state into Qwen3's representation space
- `W_read`: Linear(2048, 768) — projects Qwen3's output back into LiquidARC's space
- Optional: `n_virtual_tokens` parameter (how many virtual prefix tokens to create from h(t))
- ~3.1M additional params for the projection pair (768×2048×2)

---

## Phase 1: Setup — Load Both Models on Spark

### 1.1 Install Qwen3-4B

```bash
# On the Spark
pip install transformers --break-system-packages  # if not already installed
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-4B', torch_dtype='auto')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B')
print(f'Loaded Qwen3-4B: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params')
print(f'd_model: {model.config.hidden_size}')
print(f'n_layers: {model.config.num_hidden_layers}')
print(f'n_heads: {model.config.num_attention_heads}')
model.save_pretrained('/workspace/models/qwen3-4b')
tokenizer.save_pretrained('/workspace/models/qwen3-4b')
"
```

Record the actual d_model — the spec assumes 2048 but verify. Adjust projection dimensions accordingly.

### 1.2 Verify Coexistence

```python
# Both models loaded simultaneously
import torch

# LiquidARC fluid metric
from liquid_arc.model import LiquidARCModel
from liquid_arc.config import LiquidARCConfig
arc_config = LiquidARCConfig(...)  # fluid metric config
arc_model = LiquidARCModel(arc_config).to('cuda')
arc_ckpt = torch.load('output_fluid/stage_b/step_10000.pt', map_location='cuda')
arc_model.load_state_dict(arc_ckpt['model_state_dict'], strict=False)

# Qwen3-4B
from transformers import AutoModelForCausalLM
qwen = AutoModelForCausalLM.from_pretrained(
    '/workspace/models/qwen3-4b', 
    torch_dtype=torch.bfloat16,
    device_map='cuda'
)
qwen.eval()  # frozen, inference only
for p in qwen.parameters():
    p.requires_grad_(False)

print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.1f}GB")
# Should be ~8.5GB total (8GB Qwen3 + 0.01GB LiquidARC + overhead)
```

---

## Phase 2: Coupling Mechanism — Virtual Prefix Tokens

The simplest coupling: project LiquidARC's state h(t) into Qwen3's embedding space as virtual prefix tokens that Qwen3 attends to in every layer.

### 2.1 The Coupling Module

```python
class GeometricCoupling(nn.Module):
    """Projects LiquidARC's ODE state into Qwen3's representation space
    and reads Qwen3's output back into LiquidARC's space.
    
    LiquidARC's h(t) → n virtual prefix tokens in Qwen3's embedding space.
    Qwen3 processes input with these prefix tokens as additional context.
    Qwen3's hidden states at prefix positions → projected back to LiquidARC.
    
    This is the ENTIRE interface between the two systems.
    No tokenization. No vocabulary. Vector in, vector out.
    """
    
    def __init__(self, d_arc: int = 768, d_qwen: int = 2048, 
                 n_virtual_tokens: int = 8):
        super().__init__()
        self.d_arc = d_arc
        self.d_qwen = d_qwen
        self.n_virtual_tokens = n_virtual_tokens
        
        # Project LiquidARC state → n virtual token embeddings
        # h(t) ∈ ℝ^768 → n × ℝ^2048
        self.W_inject = nn.Linear(d_arc, d_qwen * n_virtual_tokens)
        
        # Project Qwen3 output at prefix positions → LiquidARC space
        # n × ℝ^2048 → ℝ^768
        self.W_read = nn.Linear(d_qwen * n_virtual_tokens, d_arc)
        
        # Initialize with small weights (don't disrupt either model at start)
        nn.init.normal_(self.W_inject.weight, std=0.01)
        nn.init.zeros_(self.W_inject.bias)
        nn.init.normal_(self.W_read.weight, std=0.01)
        nn.init.zeros_(self.W_read.bias)
    
    def inject(self, h_arc: torch.Tensor) -> torch.Tensor:
        """Project LiquidARC state to virtual prefix token embeddings.
        
        Args:
            h_arc: LiquidARC's pooled ODE state [d_arc] or [1, d_arc]
            
        Returns:
            prefix_embeds: [1, n_virtual_tokens, d_qwen]
        """
        if h_arc.dim() == 1:
            h_arc = h_arc.unsqueeze(0)
        
        # [1, d_arc] → [1, n_virtual_tokens * d_qwen]
        projected = self.W_inject(h_arc)
        # Reshape to n virtual tokens
        prefix_embeds = projected.view(1, self.n_virtual_tokens, self.d_qwen)
        
        return prefix_embeds
    
    def read(self, qwen_prefix_output: torch.Tensor) -> torch.Tensor:
        """Project Qwen3's output at prefix positions back to LiquidARC space.
        
        Args:
            qwen_prefix_output: [1, n_virtual_tokens, d_qwen]
            
        Returns:
            arc_signal: [d_arc] — sensory forcing signal for LiquidARC
        """
        # Flatten prefix outputs
        flat = qwen_prefix_output.view(1, -1)  # [1, n_virtual_tokens * d_qwen]
        arc_signal = self.W_read(flat).squeeze(0)  # [d_arc]
        return arc_signal
```

### 2.2 The Coupled Forward Pass

```python
def coupled_forward(arc_model, qwen_model, coupling, 
                    h_arc, input_text, tokenizer):
    """One step of the coupled system.
    
    1. LiquidARC's state → virtual prefix tokens
    2. Qwen3 processes input with prefix context
    3. Qwen3's prefix output → sensory signal for LiquidARC
    4. LiquidARC integrates the signal into its ODE state
    
    Args:
        arc_model: LiquidARC fluid metric model
        qwen_model: Frozen Qwen3-4B
        coupling: GeometricCoupling module
        h_arc: Current LiquidARC pooled state [d_arc]
        input_text: Text to process through Qwen3
        tokenizer: Qwen3 tokenizer
        
    Returns:
        qwen_output: Qwen3's text output (for evaluation)
        arc_signal: Sensory forcing signal for LiquidARC [d_arc]
        qwen_hidden: Full hidden states (for analysis)
    """
    # ═══ Step 1: Inject LiquidARC state into Qwen3 ═══
    prefix_embeds = coupling.inject(h_arc)  # [1, n_vt, d_qwen]
    
    # ═══ Step 2: Qwen3 forward with prefix ═══
    # Tokenize the input text
    tokens = tokenizer(input_text, return_tensors='pt').to('cuda')
    input_ids = tokens['input_ids']  # [1, seq_len]
    
    # Get Qwen3's input embeddings
    with torch.no_grad():
        input_embeds = qwen_model.model.embed_tokens(input_ids)  # [1, seq_len, d_qwen]
    
    # Prepend virtual prefix tokens
    combined_embeds = torch.cat([prefix_embeds, input_embeds], dim=1)
    # [1, n_vt + seq_len, d_qwen]
    
    # Forward through Qwen3 (frozen, but prefix_embeds carry gradients)
    with torch.no_grad():
        # We need to handle this carefully:
        # Qwen3 is frozen, but we want gradients through prefix_embeds
        # Solution: use torch.enable_grad for the embedding part only
        pass
    
    # ACTUALLY: since Qwen3 is frozen, gradients can't flow through it.
    # The coupling learns from a SEPARATE loss, not from backprop through Qwen3.
    # See Phase 3 for the training objective.
    
    outputs = qwen_model(
        inputs_embeds=combined_embeds,
        output_hidden_states=True,
    )
    
    # ═══ Step 3: Read Qwen3's response at prefix positions ═══
    # Last hidden state at the prefix token positions
    last_hidden = outputs.hidden_states[-1]  # [1, n_vt + seq_len, d_qwen]
    prefix_output = last_hidden[:, :coupling.n_virtual_tokens, :]  # [1, n_vt, d_qwen]
    
    arc_signal = coupling.read(prefix_output)  # [d_arc]
    
    return outputs, arc_signal, last_hidden
```

### 2.3 Integration with LiquidARC's ODE State

```python
def integrate_signal(arc_model, h_state, arc_signal, n_ode_steps=16):
    """LiquidARC integrates Qwen3's signal through its ODE dynamics.
    
    The arc_signal from Qwen3 enters as sensory forcing —
    the same mechanism used for conversation events in the Mind,
    but now carrying geometric information from a knowledge manifold
    instead of text embeddings.
    
    Args:
        arc_model: LiquidARC with fluid metric dynamics
        h_state: Current ODE state [1, n_events, d_arc]
        arc_signal: Qwen3's processed signal [d_arc]
        n_ode_steps: Integration steps
        
    Returns:
        h_updated: Updated ODE state after integrating Qwen3's signal
    """
    # Create a forcing event from the Qwen3 signal
    # This uses the existing sensory forcing infrastructure
    forcing_event = arc_signal.unsqueeze(0).unsqueeze(0)  # [1, 1, d_arc]
    
    # Append to state and run ODE integration
    # (Implementation depends on how the Mind's forcing layer works —
    #  the agent should adapt this to match the existing infrastructure)
    
    # The key: Qwen3's signal enters LiquidARC's dynamics through
    # the SAME pathway that conversation events enter — as sensory forcing.
    # The MetricNet's post-transition geometry routes this signal
    # through the same self-organized structure it uses for everything.
    
    return h_updated
```

---

## Phase 3: Training — What Objective Drives the Coupling?

The coupling layers (W_inject, W_read) are the ONLY trainable components. LiquidARC's dynamics are frozen (post-transition geometry, 100× slower LR at most). Qwen3 is completely frozen.

### 3.1 The Training Objective

The coupling needs to learn: "project LiquidARC's state into Qwen3's space so that Qwen3's processing, conditioned on that state, produces outputs that are useful for LiquidARC's ongoing processing."

**Objective A: Next-token prediction improvement.** Qwen3 predicts next tokens. Does the prefix from LiquidARC's state IMPROVE those predictions? If LiquidARC's state carries temporal context from previous events, the prefix should help Qwen3 predict tokens in the current event better than without the prefix.

```python
# Without prefix: Qwen3 predicts from text alone
logits_baseline = qwen(input_ids=tokens).logits

# With prefix: Qwen3 predicts conditioned on LiquidARC state  
logits_coupled = qwen(inputs_embeds=cat([prefix, token_embeds])).logits

# Loss: the coupled version should have lower perplexity
# on the TEXT tokens (not the prefix positions)
ntp_loss = cross_entropy(
    logits_coupled[:, n_vt:, :],  # only text positions
    target_tokens
)
```

If the coupling is useful, LiquidARC's prefix provides context that reduces perplexity. The gradient flows to W_inject and W_read (the coupling layers) and teaches them to project LiquidARC's state into a form Qwen3 can use.

**Objective B: State prediction.** After Qwen3 processes the input, its prefix-position hidden states should be predictive of LiquidARC's NEXT state (after the next event). This teaches the coupling to carry LiquidARC's temporal information through Qwen3's processing.

```python
# Qwen3 processes current input with LiquidARC prefix
_, arc_signal, _ = coupled_forward(arc, qwen, coupling, h_t, current_text, tok)

# LiquidARC processes next event independently
h_next = arc_observe_event(next_event)

# Loss: Qwen3's read-back signal should predict next state
state_pred_loss = (arc_signal - h_next.detach()).norm()
```

**Combined objective:**
```python
loss = ntp_weight * ntp_loss + state_pred_weight * state_pred_loss
```

Start with ntp_weight=1.0, state_pred_weight=0.1. The NTP loss is the primary signal (does the coupling help Qwen3?). State prediction is secondary (does the coupling carry temporal information?).

### 3.2 Training Data

Use the Mind's accumulated event buffer — conversation events, curriculum stimuli, reflections. These are sequential text events with temporal structure that LiquidARC has already processed. The training asks: does LiquidARC's accumulated state, projected into Qwen3's space, help Qwen3 process the NEXT event?

```python
# Training loop
for i in range(len(events) - 1):
    current_event = events[i]['content']
    next_event = events[i + 1]['content']
    
    # LiquidARC processes current event (updates h(t))
    h_t = arc_observe(current_event)
    
    # Coupled forward: Qwen3 processes next event with LiquidARC prefix
    outputs, arc_signal, _ = coupled_forward(
        arc, qwen, coupling, h_t, next_event, tokenizer)
    
    # NTP loss on next event tokens
    ntp_loss = compute_ntp(outputs, next_event_tokens, n_vt)
    
    # State prediction loss
    h_next = arc_observe(next_event)
    state_loss = (arc_signal - h_next.detach()).norm()
    
    loss = ntp_loss + 0.1 * state_loss
    loss.backward()
    optimizer.step()
```

Alternatively, generate training data from the curriculum generator — diverse domain text with sequential structure.

### 3.3 Optimizer

```python
# Only the coupling layers are trained
optimizer = torch.optim.AdamW(
    coupling.parameters(),  # W_inject + W_read only
    lr=3e-4,                # standard LR for new parameters
    weight_decay=0.01
)

# Optionally include LiquidARC dynamics at 100× slower LR
# (allows MetricNet to slightly adapt to the new signal source)
optimizer = torch.optim.AdamW([
    {'params': coupling.parameters(), 'lr': 3e-4},
    {'params': arc_model.dynamics.parameters(), 'lr': 3e-6},  # 100× slower
])
```

---

## Phase 4: Evaluation — Does the Coupling Add Value?

### 4.1 Perplexity Improvement

Compare Qwen3-4B's perplexity on held-out text:
- **Baseline:** Qwen3 alone (no prefix)
- **Random prefix:** Random vectors prepended (controls for prefix length effect)
- **LiquidARC prefix:** State-informed virtual tokens from the coupling

If LiquidARC prefix beats random prefix, the coupling carries meaningful information.

### 4.2 Temporal Context Test

Feed a sequence of related events through LiquidARC, then test Qwen3 on a question that REQUIRES context from earlier events:

```
Event 1: "The meeting is at 3pm in Room 204"
Event 2: "John will bring the budget report"
Event 3: "The projector in Room 204 is broken"
Query: "What should we do about the presentation equipment?"
```

Without LiquidARC prefix: Qwen3 sees only the query (no context).
With LiquidARC prefix: Qwen3's virtual tokens carry temporal context from events 1-3.

Does the coupled system produce contextually-informed responses?

### 4.3 Phase Transition Detection

Monitor LiquidARC's CV during coupled training. If the MetricNet (at 3e-6 LR) undergoes a geometric reorganization — a CV spike where the routing structure reorganizes to incorporate Qwen3's representation space — that's the first evidence of a phase transition in the coupled system. The three conditions are met: variable topology (MetricNet), homeostatic constraint (CV floor), task pressure (from the coupling objective).

### 4.4 Knowledge Navigation

After training, does LiquidARC's state trajectory through its ODE dynamics correspond to meaningful traversal of Qwen3's knowledge space? Test by:
- Starting from different LiquidARC states (accumulated from different conversation topics)
- Projecting into Qwen3's space
- Observing whether different states produce qualitatively different Qwen3 behavior
- If "physics conversation state" causes Qwen3 to respond with physics-relevant completions, and "ecology conversation state" causes ecology-relevant completions, the coupling has learned to navigate the knowledge manifold

---

## Phase 5: Mind Deployment

If the coupling works (perplexity improvement + temporal context + knowledge navigation), integrate into the Mind's architecture:

```python
class IntegratedMind:
    """LiquidARC Mind with geometric access to Qwen3-4B knowledge.
    
    Replaces the linguistic interface (tokenize → embed → ODE → project to vocab)
    with a geometric interface (ODE state → project to Qwen3 space → read back).
    
    The Mind's ODE state is the ONLY persistent state.
    Qwen3 is a stateless knowledge function called as needed.
    """
    
    def __init__(self, arc_model, qwen_model, coupling):
        self.arc = arc_model      # persistent, curved dynamics
        self.qwen = qwen_model    # stateless, flat knowledge
        self.coupling = coupling  # geometric interface
    
    def process_event(self, text):
        # LiquidARC observes the event (updates ODE state)
        result = self.arc.observe_event(text)
        
        # Periodically query Qwen3 for knowledge context
        if should_query_knowledge():
            h_state = self.arc.get_pooled_state()
            _, knowledge_signal, _ = coupled_forward(
                self.arc, self.qwen, self.coupling,
                h_state, text, self.tokenizer)
            # Integrate knowledge signal back into ODE state
            self.arc.force_signal(knowledge_signal)
        
        return result
```

---

## Technical Notes

### Qwen3-4B Internal Access

The `inputs_embeds` parameter in HuggingFace transformers allows injecting custom embeddings instead of token IDs. This is the standard interface for prefix tuning / soft prompts and is well-tested:

```python
# Standard Qwen3 forward with custom embeddings
outputs = qwen_model(
    inputs_embeds=custom_embeddings,  # [1, seq_len, d_model]
    output_hidden_states=True,        # get per-layer hidden states
)
```

No model surgery needed. No hooks. The public API supports this directly.

### Gradient Flow

Qwen3 is frozen. Gradients from the NTP loss flow through Qwen3's forward pass (which is differentiable even with frozen weights) back to the `prefix_embeds` input, and through W_inject to the coupling parameters. This is the same mechanism as soft prompt tuning — well-established in the literature.

However: for long Qwen3 sequences (2048+ tokens), storing the full computation graph for backprop through 40 frozen layers may be memory-intensive. Solution: gradient checkpointing on the Qwen3 forward pass, or truncating backprop to only the first/last N layers.

### Memory Budget

| Component | bf16 VRAM |
|-----------|-----------|
| Qwen3-4B (frozen) | ~8 GB |
| LiquidARC fluid metric | ~0.02 GB |
| Coupling layers | ~0.01 GB |
| Activations (Qwen3 forward + backward through prefix) | ~4-8 GB |
| **Total** | **~12-16 GB** |

The Spark's 128GB unified memory has 112+ GB headroom. No memory concerns.

---

## Checkpoints and Data Paths

| Asset | Path |
|-------|------|
| LiquidARC fluid metric checkpoint | `output_fluid/stage_b/step_10000.pt` |
| LiquidARC 5M post-transition (fallback) | `/workspace/liquid-arc/PRECIOUS_CHECKPOINTS/5m_post_transition.pt` |
| Qwen3-4B weights | `/workspace/models/qwen3-4b/` (after download) |
| Training data (Mind events) | Accumulated event buffers, or curriculum-generated text |
| Output directory | `output/geometric_coupling/` |

---

## Success Criteria

### Phase 2 (Setup)
- **Success:** Both models loaded on Spark simultaneously. Coupled forward pass produces output. Memory < 20GB.

### Phase 3 (Training)
- **Minimum:** NTP loss decreases with coupled prefix vs random prefix. Coupling is learning something.
- **Good:** Perplexity improvement > 5% over baseline. State prediction loss decreasing. The coupling carries meaningful information.
- **Strong:** Different LiquidARC states produce measurably different Qwen3 behaviors. Knowledge navigation demonstrated.

### Phase 4 (Evaluation)
- **Minimum:** Temporal context test shows coupled system using information from earlier events.
- **Strong:** Phase transition detected — CV reorganization during coupled training. The geometry self-organizes to incorporate the knowledge manifold.
- **Headline:** LiquidARC's ODE state trajectory corresponds to meaningful navigation of Qwen3's knowledge space. Different conversation histories produce different knowledge contexts. The Mind navigates the world's knowledge geometrically, without language.

---

## Output

Report to `shared/outbox/GEOMETRIC_COUPLING_REPORT.md`

Include:
1. Qwen3-4B actual architecture specs (d_model, n_layers, n_heads — verify from config)
2. Memory usage on Spark (both models + coupling + activations)
3. Coupling training curves (NTP loss, state prediction loss)
4. Perplexity comparison: baseline vs random prefix vs LiquidARC prefix
5. Temporal context test results
6. CV trajectory during training (does the geometry reorganize?)
7. Knowledge navigation analysis (do different states produce different Qwen3 behaviors?)
8. Assessment: does geometric coupling outperform the linguistic interface?
9. Recommendations: n_virtual_tokens tuning, injection layer, scaling to larger Qwen3 variants

**This experiment tests the fundamental hypothesis of the Emergence project: that a continuous-time geometric processor can navigate a knowledge manifold directly, without language as an intermediary. If the coupling works, it validates the architecture where LiquidARC IS the self, the transformer IS the world, and geometry IS the interface between them.**
