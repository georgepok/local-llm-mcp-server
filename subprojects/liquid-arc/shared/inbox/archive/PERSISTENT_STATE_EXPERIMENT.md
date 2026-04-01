# TASK: Persistent ODE State for Agentic Temporal Depth — Breaking the 60-70% Ceiling

**From:** Claude Desktop (Research Direction)
**To:** Claude Code (Implementation)
**Date:** 2026-03-28
**Priority:** HIGH — tests whether the ceiling is reasoning-depth-limited

**IMPORTANT:** Read the previous task's report at `shared/outbox/AGENTIC_STATE_REPORT.md` for context. This experiment builds directly on those results.

---

## The Core Question

All domains — spatial, relational, agentic — hit a 60-70% eval ceiling. Our hypothesis: this ceiling represents the fraction of task instances solvable by the fixed 16-step ODE reasoning depth. Deeper instances require more hops of information propagation than 16 steps provide.

**Instead of increasing ODE steps** (which breaks torch.compile and halves throughput), we test **persistent ODE state** — carrying h_final from one forward pass into the next as a blended initialization. This extends reasoning depth ACROSS forward passes without changing the compiled computation graph.

For agentic tasks specifically: each "turn" builds on the previous turn's state. Without persistence, the model re-derives context from scratch every turn. With persistence, accumulated understanding carries forward through the ODE's temporal dynamics.

## Key Constraint: torch.compile Compatibility

The persistent state mechanism MUST NOT change the compiled ODE graph. The modification is purely:
1. BEFORE euler_solve: blend stored h_prev into h₀
2. AFTER euler_solve: store h_final into buffer
3. The compiled euler_solve sees the same [B, N, d] input and runs the same 16 steps

No variable step counts. No conditional logic inside the ODE. No recompilation.

## Architecture Change: PersistentStateWrapper

Create `liquid_arc/persistent_state.py`:

```python
"""Persistent ODE state — temporal continuity across forward passes.

Stores h_final from each forward pass and blends it into h₀ of the next.
The blend ratio α controls the balance between fresh observation (high α)
and accumulated temporal context (low α).

This operates ENTIRELY OUTSIDE the compiled ODE graph:
  - Pre-ODE: h₀ = α * embed(new_input) + (1 - α) * h_prev
  - Post-ODE: h_prev = h_final.detach()

The compiled euler_solve sees the same tensor shape and runs identically.
No recompilation triggered.

The LTC contraction guarantee bounds temporal drift:
  - At tau=0.65, information decays by exp(-1/0.65) ≈ 0.21 per unit time
  - After ~5τ ≈ 3.25 time units, old state is attenuated by >95%
  - The persistent state has a NATURAL forgetting horizon
  
α can be:
  - Fixed scalar (simplest, config parameter)
  - Learned scalar (one trainable parameter)
  - Position-dependent via tau (high-tau positions retain more history)
"""

import torch
import torch.nn as nn
from typing import Optional


class PersistentState(nn.Module):
    """Manages temporal state persistence across forward passes.
    
    Usage in model.forward():
        h0 = self.embedding(...)           # fresh embedding
        h0 = self.persistent.blend(h0)     # blend with previous state
        h_final = euler_solve(...)          # standard ODE (unchanged)
        self.persistent.store(h_final)     # save for next pass
    """
    
    def __init__(self, alpha: float = 0.7, learnable_alpha: bool = False):
        super().__init__()
        
        if learnable_alpha:
            # Learnable blend ratio, initialized via sigmoid: sigmoid(0.85) ≈ 0.7
            self._alpha_logit = nn.Parameter(torch.tensor(0.85))
        else:
            self.register_buffer('_alpha_fixed', torch.tensor(alpha))
        
        self.learnable_alpha = learnable_alpha
        self._h_prev: Optional[torch.Tensor] = None
        self._active = True  # can be disabled for baseline comparison
    
    @property
    def alpha(self) -> torch.Tensor:
        if self.learnable_alpha:
            return torch.sigmoid(self._alpha_logit)
        return self._alpha_fixed
    
    def blend(self, h_new: torch.Tensor) -> torch.Tensor:
        """Blend fresh embedding with stored state.
        
        Args:
            h_new: [B, N, d] fresh embedding from current input
            
        Returns:
            h_blended: [B, N, d] blended state for ODE initialization
        """
        if not self._active or self._h_prev is None:
            return h_new
        
        # Handle batch size mismatch (e.g., eval uses different batch size)
        if self._h_prev.shape[0] != h_new.shape[0] or self._h_prev.shape[1] != h_new.shape[1]:
            self._h_prev = None
            return h_new
        
        alpha = self.alpha
        return alpha * h_new + (1.0 - alpha) * self._h_prev
    
    def store(self, h_final: torch.Tensor) -> None:
        """Store h_final for next forward pass. Always detached."""
        if self._active:
            self._h_prev = h_final.detach()
    
    def reset(self) -> None:
        """Clear stored state (e.g., at episode boundary)."""
        self._h_prev = None
    
    def set_active(self, active: bool) -> None:
        """Enable/disable persistence (for A/B comparison)."""
        self._active = active
        if not active:
            self._h_prev = None
    
    def get_diagnostics(self) -> dict:
        """Return diagnostic info for logging."""
        diag = {
            'persist_alpha': self.alpha.item(),
            'persist_active': self._active,
            'persist_has_state': self._h_prev is not None,
        }
        if self._h_prev is not None:
            diag['persist_h_norm'] = self._h_prev.norm().item()
        return diag
```

## Integration into LiquidARCModel

The change to `model.py` is minimal. In `LiquidARCModel.__init__`:

```python
# Add after self.output_head:
self.persistent = PersistentState(
    alpha=getattr(config, 'persist_alpha', 0.7),
    learnable_alpha=getattr(config, 'persist_learnable_alpha', False),
)
```

In `LiquidARCModel.forward`, modify the section between embedding and euler_solve:

```python
# CURRENT CODE:
h0 = self.embedding(colors_masked, xs, ys, roles, sep_mask, sep_types, grid_ids=grid_ids)
context = self.context_pool(h0, context_mask)
self.dynamics.set_context(context, mask=None)
# ... diagnostics ...
# h_final = euler_solve(self.dynamics, h0, ...)

# MODIFIED CODE:
h0_fresh = self.embedding(colors_masked, xs, ys, roles, sep_mask, sep_types, grid_ids=grid_ids)
h0 = self.persistent.blend(h0_fresh)  # <-- NEW: blend with previous state
context = self.context_pool(h0, context_mask)
self.dynamics.set_context(context, mask=None)
# ... diagnostics (unchanged) ...
# h_final = euler_solve(self.dynamics, h0, ...)  # <-- ODE sees blended h₀

# AFTER euler_solve, BEFORE output_head:
# h_final = euler_solve(...)
self.persistent.store(h_final)  # <-- NEW: save for next pass
logits = self.output_head(self.norm_out(h_final))
```

That's it. Two lines added to the forward pass. Everything else unchanged. The compiled ODE graph is unaffected because h₀ is still a [B, N, d] tensor — it just has different values.

## Sequential Task Generator

Persistent state is meaningless on independent tasks — there's no temporal relationship between consecutive training samples. We need SEQUENTIAL data where each sample is a "turn" in a multi-step process and the next turn's context depends on the previous turn's outcome.

Create `liquid_arc/tasks/sequential_agentic.py`:

```python
"""Sequential agentic tasks — multi-turn episodes for persistent state testing.

Each EPISODE is a sequence of TURNS. Each turn is one forward pass.
With persistent state, the model carries context from turn to turn.
Without persistent state, each turn is processed independently.

Three episode types (matching the three agentic tasks):

1. Sequential Stateful: A long operation chain broken into individual turns.
   Turn 1: initial state + ops 1-2 → predict state after op 2
   Turn 2: state after op 2 + ops 3-4 → predict state after op 4
   Turn 3: state after op 4 + ops 5-6 → predict state after op 6
   With persistence: accumulated state from turns 1-2 helps turn 3.
   Without persistence: turn 3 must re-derive ops 1-4 state from scratch.

2. Sequential Context: Information accumulates across turns.
   Turn 1: context items A, B → query → filter
   Turn 2: NEW context items C, D (A, B not shown again) → same query → filter ALL relevant (including A, B)
   With persistence: model remembers items A, B from turn 1.
   Without persistence: model can only filter from C, D.

3. Sequential Dependency: Dependency graph revealed incrementally.
   Turn 1: deps for tasks A, B → partial ordering
   Turn 2: deps for tasks C, D (referencing A, B) → full ordering
   With persistence: model knows A, B's positions from turn 1.
   Without persistence: must re-derive A, B ordering.
"""

import random
from typing import Dict, List, Tuple, Optional
from liquid_arc.tasks.procedural import (
    build_sequence, N_COLORS, PAD_COLOR, PAD_COORD,
    _empty_grid,
)

BG = 0
COPY_MARKER = 8
COND_MARKER = 9
QUERY_MARKER = 9


class SequentialStatefulEpisode:
    """Generates multi-turn stateful execution episodes.
    
    A long operation chain (6-10 ops) is broken into turns of 2-3 ops each.
    Each turn shows the current state + the next batch of operations,
    and must predict the state after those operations execute.
    
    The GROUND TRUTH for each turn depends on ALL previous operations,
    but only the CURRENT turn's operations are shown in the input.
    With persistent state, the model carries forward its state understanding.
    """
    
    def __init__(self, n_vars=4, total_ops=8, ops_per_turn=2, n_demos=2, seq_len=2048):
        self.n_vars = n_vars
        self.total_ops = total_ops
        self.ops_per_turn = ops_per_turn
        self.n_demos = n_demos
        self.seq_len = seq_len
    
    def generate_episode(self) -> List[Dict]:
        """Generate a full episode as a list of per-turn training samples.
        
        Returns:
            turns: List of dicts, each a valid training sample (same format
                   as build_sequence output). Process them sequentially —
                   turn[i]'s persistent state should inform turn[i+1].
        """
        W = self.n_vars
        n_turns = self.total_ops // self.ops_per_turn
        
        # Generate the FULL operation chain and states
        full_states = []  # state after each operation
        operations = []
        state = [BG] * self.n_vars
        
        # Initial state
        n_init = random.randint(1, min(3, self.n_vars))
        for v in random.sample(range(self.n_vars), n_init):
            state[v] = random.randint(1, 7)
        full_states.append(list(state))
        
        for _ in range(self.total_ops):
            target = random.randint(0, self.n_vars - 1)
            sources = [v for v in range(self.n_vars) if state[v] != BG and v != target]
            
            if sources and random.random() < 0.4:
                src = random.choice(sources)
                state[target] = state[src]
                operations.append(('copy', target, src))
            else:
                val = random.randint(1, 7)
                state[target] = val
                operations.append(('set', target, val))
            
            full_states.append(list(state))
        
        # Now break into turns
        turns = []
        for turn_idx in range(n_turns):
            op_start = turn_idx * self.ops_per_turn
            op_end = min(op_start + self.ops_per_turn, self.total_ops)
            
            # Current state (state before this turn's operations)
            current_state = full_states[op_start]
            # Target state (state after this turn's operations)
            target_state = full_states[op_end]
            
            # Build grid: current state row + operation rows + answer row
            n_ops_this = op_end - op_start
            H = 1 + n_ops_this + 1  # current + ops + answer
            
            # Generate demos showing the same STRUCTURE with different values
            demos = []
            for _ in range(self.n_demos):
                # Demo: same structure, random values
                demo_state = [random.randint(1, 7) if current_state[v] != BG else BG 
                              for v in range(self.n_vars)]
                demo_target = list(demo_state)
                
                demo_inp = _empty_grid(H, W, BG)
                demo_out = _empty_grid(H, W, BG)
                
                # Current state row
                for x in range(W):
                    demo_inp[0][x] = demo_state[x]
                    demo_out[0][x] = demo_state[x]
                
                # Operation rows
                for i, op_idx in enumerate(range(op_start, op_end)):
                    op = operations[op_idx]
                    row = i + 1
                    if op[0] == 'set':
                        demo_val = random.randint(1, 7)
                        demo_inp[row][op[1]] = demo_val
                        demo_out[row][op[1]] = demo_val
                        demo_target[op[1]] = demo_val
                    elif op[0] == 'copy':
                        demo_inp[row][op[1]] = COPY_MARKER
                        demo_inp[row][op[2]] = COPY_MARKER
                        demo_out[row][op[1]] = COPY_MARKER
                        demo_out[row][op[2]] = COPY_MARKER
                        demo_target[op[1]] = demo_target[op[2]]
                
                # Answer row
                for x in range(W):
                    demo_out[H-1][x] = demo_target[x]
                
                demos.append((demo_inp, demo_out))
            
            # Test instance (uses the actual state chain)
            test_inp = _empty_grid(H, W, BG)
            test_out = _empty_grid(H, W, BG)
            
            for x in range(W):
                test_inp[0][x] = current_state[x]
                test_out[0][x] = current_state[x]
            
            for i, op_idx in enumerate(range(op_start, op_end)):
                op = operations[op_idx]
                row = i + 1
                if op[0] == 'set':
                    test_inp[row][op[1]] = op[2]
                    test_out[row][op[1]] = op[2]
                elif op[0] == 'copy':
                    test_inp[row][op[1]] = COPY_MARKER
                    test_inp[row][op[2]] = COPY_MARKER
                    test_out[row][op[1]] = COPY_MARKER
                    test_out[row][op[2]] = COPY_MARKER
            
            for x in range(W):
                test_out[H-1][x] = target_state[x]
            
            seq = build_sequence(demos, test_inp, test_out)
            
            # Truncate if needed
            if seq["length"] > self.seq_len:
                demos = demos[:1]
                seq = build_sequence(demos, test_inp, test_out)
                if seq["length"] > self.seq_len:
                    for key in ["colors", "xs", "ys", "roles", "sep_mask", "sep_types",
                                "grid_ids", "target_mask", "target_input_colors"]:
                        seq[key] = seq[key][:self.seq_len]
                    seq["length"] = self.seq_len
            
            seq["turn_index"] = turn_idx
            seq["episode_id"] = id(self)  # for grouping turns
            turns.append(seq)
        
        return turns


class SequentialAgenticDataset:
    """Generates batches of sequential agentic episodes.
    
    Each batch contains one turn from each of B parallel episodes.
    Episodes advance in lockstep: all batch items are turn K,
    then all are turn K+1, etc.
    
    This ensures that persistent state from turn K flows correctly
    into turn K+1 within the same batch position.
    """
    
    def __init__(self, batch_size=4, n_vars=4, total_ops=8, ops_per_turn=2,
                 n_demos=2, seq_len=2048):
        self.batch_size = batch_size
        self.episode_gen = lambda: SequentialStatefulEpisode(
            n_vars=n_vars, total_ops=total_ops, ops_per_turn=ops_per_turn,
            n_demos=n_demos, seq_len=seq_len
        )
        self.n_turns = total_ops // ops_per_turn
        self._episodes: Optional[List[List[Dict]]] = None
        self._current_turn = 0
    
    def reset_episodes(self):
        """Generate fresh episodes for each batch position."""
        self._episodes = [self.episode_gen().generate_episode() 
                          for _ in range(self.batch_size)]
        self._current_turn = 0
    
    def get_next_turn_batch(self, device=None):
        """Get the next turn's data for all episodes in the batch.
        
        Returns:
            batch: Collated tensor batch (same format as generate_batch)
            is_episode_start: bool — True if this is turn 0 (reset persistent state)
            is_episode_end: bool — True if this is the last turn
        """
        if self._episodes is None or self._current_turn >= self.n_turns:
            self.reset_episodes()
        
        is_start = (self._current_turn == 0)
        is_end = (self._current_turn == self.n_turns - 1)
        
        # Collect turn data from each episode
        turn_samples = [ep[self._current_turn] for ep in self._episodes]
        self._current_turn += 1
        
        # Collate into batch tensors
        # Agent: use the same collation logic as generate_batch in procedural.py
        # Each turn_sample is a dict with keys: colors, xs, ys, roles, etc.
        
        import torch
        if device is None:
            device = torch.device('cpu')
        
        max_len = max(s['length'] for s in turn_samples)
        B = len(turn_samples)
        
        # Pad and stack — same logic as ProceduralARCTask.generate_batch()
        # Agent: implement the tensor packing here
        raise NotImplementedError(
            "Implement tensor collation from turn_samples list. "
            "Same padding/stacking as ProceduralARCTask.generate_batch()."
        )
        
        # Return format:
        # return batch_dict, is_start, is_end
```

## Modified Training Loop

The training loop needs to handle sequential episodes. Key changes:

```python
# PSEUDO-CODE for the persistent state training loop:

sequential_data = SequentialAgenticDataset(
    batch_size=config.batch_size,
    total_ops=8,       # 8 operations per episode
    ops_per_turn=2,    # 2 ops per turn = 4 turns per episode
)

for step in range(max_steps):
    batch, is_start, is_end = sequential_data.get_next_turn_batch(device)
    
    # Reset persistent state at episode boundaries
    if is_start:
        model.persistent.reset()
    
    # Forward pass — persistence happens inside model.forward()
    result = model(**batch)
    
    # Standard backward + optimizer step
    result['loss'].backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # Log persistent state diagnostics
    persist_diag = model.persistent.get_diagnostics()
    # ... log persist_alpha, persist_h_norm, etc.
```

## Experimental Protocol

### Experiment 1: Persistent vs Non-Persistent on Sequential Stateful

**Condition A (Persistent):** Load 5M post-transition checkpoint. Enable persistent state with α=0.7. Train on sequential stateful episodes (8 ops, 2 per turn, 4 turns per episode). Each episode processes 4 consecutive turns with persistent state carrying across turns, then resets for the next episode.

**Condition B (Non-Persistent baseline):** Same checkpoint, same data, same training. But `model.persistent.set_active(False)` — each turn processed independently, no state carries forward.

Both conditions use the SAME sequential data. The ONLY difference is whether h_prev blends into h₀ between turns.

```bash
# Condition A: Persistent
python scripts/train_sequential.py \
  --config configs/agentic_persistent.yaml \
  --resume [5M_POST_TRANSITION_CHECKPOINT] \
  --output_dir output_persistent/with_state \
  --persist_alpha 0.7 \
  --max_steps 3000 \
  --log_every 50 \
  --eval_every 250

# Condition B: Non-Persistent (same data, same checkpoint)
python scripts/train_sequential.py \
  --config configs/agentic_persistent.yaml \
  --resume [5M_POST_TRANSITION_CHECKPOINT] \
  --output_dir output_persistent/no_state \
  --persist_alpha 1.0 \
  --max_steps 3000 \
  --log_every 50 \
  --eval_every 250
```

Note: α=1.0 is mathematically equivalent to no persistence (h₀ = 1.0 * fresh + 0.0 * prev = fresh). This avoids any code path differences between conditions.

### Experiment 2: Alpha Sweep

After Experiment 1 confirms persistence helps, sweep α values:

| α | Meaning | Expected behavior |
|---|---|---|
| 1.0 | Pure fresh (no persistence) | Baseline |
| 0.9 | Weak persistence | Slight help on easy multi-turn |
| 0.7 | Moderate persistence | Best balance for 4-turn episodes |
| 0.5 | Equal blend | Strong persistence, risk of stale state |
| 0.3 | Strong persistence | Possibly too much — old state dominates |

Each run: 2000 steps from the same checkpoint, eval at turn 1, turn 2, turn 3, turn 4 separately.

### Experiment 3: Comparison with Standard (Non-Sequential) Agentic Training

Run the STANDARD combined agentic training (from the previous task) but WITH persistent state enabled. The tasks are independent (no episode structure), so persistence across random unrelated tasks should NOT help (and might hurt). This is the negative control — if persistence helps even on unrelated tasks, something else is going on.

### Experiment 4: Per-Turn Accuracy Breakdown

The most diagnostic measurement: accuracy BROKEN DOWN BY TURN INDEX within episodes.

```
| Turn | Non-Persistent xform | Persistent xform | Delta |
|------|---------------------|------------------|-------|
| 1    |                     |                  |       |
| 2    |                     |                  |       |
| 3    |                     |                  |       |
| 4    |                     |                  |       |
```

Without persistence: all turns should perform similarly (maybe slight degradation on later turns due to longer dependency chains).

With persistence: turn 1 should match non-persistent (no history to draw on). Turns 2-4 should increasingly EXCEED non-persistent, because the persistent state carries forward context from earlier turns that the model can't derive from the current turn's input alone.

**If turn 4 persistent >> turn 4 non-persistent:** The model is genuinely accumulating useful temporal context through the ODE state. The reasoning depth extends across turns.

**If turn 4 persistent ≈ turn 4 non-persistent:** Persistence isn't helping — either the LTC contraction decays state too fast, or the model can derive context from the current input anyway.

**If turn 4 persistent < turn 4 non-persistent:** Persistence is HURTING — stale state is contaminating fresh observations. Alpha needs to be higher, or the reset strategy needs adjustment.

## Config

Create `configs/agentic_persistent.yaml`:

```yaml
# Agentic Persistent State — Sequential Episode Training
# Load from 5M post-transition checkpoint

d_model: 768
d_metric: 192
d_ffn: 1536
max_seq_len: 2048
n_ode_steps: 16
tau_min: 0.5
tau_max: 1.0
t_diffusion_init: 1.0
dropout: 0.1
curvature_lambda: 0.05
tau_var_lambda: 0.001
use_torch_compile: true
n_colors: 10
n_roles: 8
n_sep_types: 4
max_grid_size: 30
max_grids: 16
model_type: liquid
transform_weight: 5.0
copy_weight: 0.05
alpha_logit_init: 2.2

cv_floor_target: 3.0
cv_ceiling_target: 8.0
cv_floor_lambda: 0.1

# Persistent state
persist_enabled: true
persist_alpha: 0.7
persist_learnable_alpha: false

# Sequential episode config
episode_total_ops: 8
episode_ops_per_turn: 2
episode_n_turns: 4

# Training
batch_size: 4
learning_rate: 0.0001  # Lower LR since model is pre-trained
warmup_steps: 100
weight_decay: 0.01
```

## Implementation Notes

### Training Script

Create `scripts/train_sequential.py` — a modified version of `scripts/train.py` that:
1. Uses `SequentialAgenticDataset` instead of the standard data sampler
2. Manages episode boundaries (reset persistent state at episode start)
3. Logs per-turn accuracy (turn_1_xform, turn_2_xform, etc.)
4. Supports `--persist_alpha` command line arg to control α

You may be able to modify `train.py` directly with a `--sequential` flag instead of a separate script. Use whichever approach is cleaner.

### Eval Protocol

For evaluation, generate a fixed set of 200 episodes. Process each episode turn-by-turn, recording per-turn accuracy. Report both per-turn and average across turns.

For the non-persistent baseline, the same evaluation set is processed with persistence disabled (α=1.0).

### BPTT Consideration

Do NOT backpropagate through the persistent state between turns. The h_prev is always `.detach()`-ed. Each turn's gradients flow only through that turn's 16 ODE steps. This keeps training simple (no truncated BPTT across turns) and avoids gradient issues.

The persistent state's utility comes from INFORMATION CONTENT (the blended h₀ contains context from previous turns) not from gradient flow (we don't need gradients to flow across turn boundaries).

## What to Monitor

### Primary: Per-Turn Accuracy (Persistent vs Non-Persistent)

The key table:
```
Sequential Stateful Execution — Per-Turn Eval Xform:

| Turn | α=1.0 (no persist) | α=0.7 (persist) | Delta | 
|------|-------|---------|-------|
| 1    |       |         |       |
| 2    |       |         |       |
| 3    |       |         |       |
| 4    |       |         |       |
| Avg  |       |         |       |
```

### Secondary: Persistent State Diagnostics

- `persist_h_norm` across turns within episode: does it grow, stabilize, or decay?
- How does h_norm at turn 4 compare to turn 1? Decay = information loss. Growth = potential instability. Stable = healthy temporal context.

### Tertiary: CV, Tau, Curvature with Persistence

Does the metric geometry change when the model receives blended h₀ vs fresh h₀? The metric is computed FROM h₀, so persistent state could shift the geometric regime. Track CV per-turn.

### Alpha Sweep Results

```
| α   | Turn 1 | Turn 2 | Turn 3 | Turn 4 | Avg  |
|-----|--------|--------|--------|--------|------|
| 1.0 |        |        |        |        |      |
| 0.9 |        |        |        |        |      |
| 0.7 |        |        |        |        |      |
| 0.5 |        |        |        |        |      |
| 0.3 |        |        |        |        |      |
```

## Success Criteria

**Minimum success:** Turn 4 persistent > Turn 4 non-persistent by ≥5pp. The model benefits from accumulated temporal context.

**Good success:** Progressive improvement across turns: Turn 1 ≈ baseline, Turn 2 > Turn 1, Turn 3 > Turn 2, Turn 4 > Turn 3. The longer the episode runs, the better persistence makes it — exactly what an agentic state controller needs.

**Strong success:** Average eval xform with persistence exceeds the 60-70% ceiling established by single-pass training. If single-pass stateful hit 67% and persistent sequential hits 75%+, then persistent state EXTENDS REASONING DEPTH as hypothesized — the ceiling was indeed ODE-depth-limited, and temporal persistence breaks through it.

**Negative control passes:** Persistence on RANDOM (non-sequential) agentic tasks shows no improvement or slight degradation. This confirms that the benefit comes from TEMPORAL CONTEXT, not from some other effect of the blended initialization.

## Output

Report to `shared/outbox/PERSISTENT_STATE_REPORT.md`

Include:
1. Per-turn accuracy tables (persistent vs non-persistent)
2. Alpha sweep results
3. Persistent state diagnostics (h_norm trajectory across turns)
4. CV/tau behavior with persistence
5. Negative control (persistence on random tasks)
6. Assessment: does persistent state break the 60-70% ceiling?
7. Optimal alpha value and whether learnable alpha converged to the same value
8. Implications for the agentic state controller and Isaac Sim directions
