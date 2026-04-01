# Resonant Geometry v2 — Graph-Distance Structural Energy

## Implementation Specification for Dev

**Version:** v0.2 — Graph-Distance Revision  
**Date:** 2026-02-20  
**Base:** FluidNet v1 codebase in `/workspace/fgn-v3/`  
**Supersedes:** RESONANT_GEOMETRY_SPEC.md (v0.1)  
**Goal:** Replace positional-proxy structural energy with graph-distance structural energy, then run Experiment B (multi-metric worlds).

---

## 0. Context — Why This Revision

### 0.1 Experiment A Results (Completed)

Structural energy prevents geometric flattening. All three λ>0 models maintained CV≈1.1–1.2 at 10K steps versus baseline deflation to 0.26. CE loss was unaffected (0.0000–0.0001 for all models). The mechanism works.

### 0.2 The Problem: Geometry Fidelity Is Zero

A diagnostic measured Spearman correlation between the learned metric's geodesic distances and the world's actual graph distances (shortest-path between rooms):

| Model | Mean ρ | Median ρ | % positive |
|-------|--------|----------|------------|
| λ=0.1 (resonant) | 0.000 | -0.022 | 40% |
| λ=0.0 (baseline) | 0.061 | 0.059 | 67% |

The maintained geometric structure has **no correlation** with world geometry. The metric aligned to **token ordering in text** (the positional proxy), not to the world's spatial/relational structure. Connected rooms described far apart in the episode text get large geodesic distance regardless of their actual graph proximity.

### 0.3 The Fix

Replace the positional proximity proxy with **actual graph distances** computed by the task. The ContinuousGridWorld already runs Dijkstra to solve episodes — we pass the all-pairs shortest-path matrix directly to the structural energy module. The metric then has the correct target to align toward: rooms that are 1-hop apart should be geodesically close, rooms that are 5-hops apart should be geodesically far.

This is not "adding labels." The task generates the world, so it knows the graph. We're giving the metric the right target instead of a broken proxy.

---

## 1. Changes Overview

Four components change. No new files are created; all changes are in-place edits to existing modules.

| File | Change | Scope |
|------|--------|-------|
| `fgn/tasks/continuous_gridworld.py` | Add `room_distances` and `room_token_positions` to batch metadata | Medium |
| `fgn/structural_energy.py` | Replace positional proxy with graph-distance lookup | Medium |
| `fgn/model_fluid.py` | Pass new metadata fields through to structural energy | Small |
| `scripts/train_resonant.py` | Pass new metadata fields from batch to model | Small |

Config (`fgn/config.py`) is unchanged. No new config fields needed.

---

## 2. Component Changes

### 2.1 ContinuousGridWorldTask — Graph Distance Metadata

**File:** `fgn/tasks/continuous_gridworld.py`

#### 2.1.1 Add All-Pairs Shortest Path to ContinuousWorld

Add a method to `ContinuousWorld` that computes the all-pairs shortest-path distance matrix using Dijkstra. This runs once per episode during generation (not during training forward pass).

```python
def all_pairs_shortest_paths(self) -> Dict[Tuple[int, int], float]:
    """Compute shortest-path distance between all room pairs.

    Returns:
        Dict mapping (room_i, room_j) -> shortest path distance.
        Unreachable pairs get distance = float('inf').
    """
    result = {}
    for source in range(self.n_rooms):
        # Dijkstra from source
        dist = {source: 0.0}
        heap = [(0.0, source)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, float('inf')):
                continue
            for v in self.graph.get(u, []):
                edge_d = self.distances.get((u, v), float('inf'))
                new_d = d + edge_d
                if new_d < dist.get(v, float('inf')):
                    dist[v] = new_d
                    heapq.heappush(heap, (new_d, v))
        for target in range(self.n_rooms):
            result[(source, target)] = dist.get(target, float('inf'))
    return result
```

**Note:** `heapq` is already imported at top of file.

#### 2.1.2 Track Room Token Positions During Tokenization

Modify `_tokenize_episode` to record the token position of each room's first mention in the [WORLD] section. This lets the structural energy module know which token positions correspond to which rooms.

Add a new return value `room_token_positions: Dict[int, int]` mapping `room_id -> token_position` (the position of the first token of the room's [WORLD] line).

**Implementation:** Inside the existing line-processing loop, when a line starts with `[WORLD] Room {N}`, record `offset` (the current token count before this line) as the token position for room N.

```python
# Inside _tokenize_episode, at the top of the loop:
room_token_positions = {}  # room_id -> token offset

for line in lines:
    if not line:
        continue

    line_ids = self.tokenizer.encode(line + "\n", add_special_tokens=False)
    offset = len(all_ids)

    # Track room token positions from [WORLD] lines
    if line.startswith("[WORLD] Room "):
        # Extract room number: "[WORLD] Room 7 (kitchen)..." -> 7
        try:
            room_num = int(line.split()[2])
            room_token_positions[room_num] = offset
        except (IndexError, ValueError):
            pass

    # ... rest of existing loop unchanged ...
```

Update `_tokenize_episode` return signature:
```python
def _tokenize_episode(self, text: str) -> Tuple[
    List[int], List[int], int, List[Tuple[int, int, str]], Dict[int, int]
]:
```

Returns: `(all_ids, all_labels, context_end_pos, action_spans, room_token_positions)`

#### 2.1.3 Compute and Pack Distance Matrix into Batch Metadata

Modify `generate_batch` to:

1. Call `world.all_pairs_shortest_paths()` for each episode
2. Build a `[R, R]` distance matrix per episode (R = number of rooms)
3. Pad to max R across the batch
4. Pack into metadata as:
   - `room_distances: [B, R_max, R_max]` — normalized shortest-path distances (float tensor)
   - `room_token_positions: [B, R_max]` — token positions for each room (long tensor, -1 = padding)
   - `n_rooms: [B]` — actual room count per episode (long tensor)

**Normalization:** Divide all distances by the maximum finite distance in each episode. Infinite distances (unreachable rooms) get value 1.0. This puts d_struct in [0, 1].

#### 2.1.4 Plumbing: Return ContinuousWorld from Episode Generation

The episode generation pipeline currently returns `(text, actions, n_steps, cost, step_costs)`. We need to also return the `ContinuousWorld` object so we can call `all_pairs_shortest_paths()`.

**Change `_try_generate_episode` return type** to include the world:

```python
def _try_generate_episode(self, relax: bool = False
                          ) -> Optional[Tuple[str, List[str], int, float, List[float], ContinuousWorld]]:
    # ... existing code ...
    return episode_text, action_strings, n_steps, total_cost, step_costs, world
```

**Change `_generate_valid_episode`** to propagate:

```python
def _generate_valid_episode(self) -> Tuple[str, List[str], int, float, List[float], ContinuousWorld]:
    for _ in range(self.max_retries):
        result = self._try_generate_episode()
        if result is not None:
            return result
    result = self._try_generate_episode(relax=True)
    if result is not None:
        return result
    return self._minimal_episode()
```

**Change `_minimal_episode`** similarly — it already creates a ContinuousWorld internally, just return it.

#### 2.1.5 Full generate_batch Changes

After collecting all episodes, build the distance and position tensors:

```python
def generate_batch(self, batch_size: int,
                   device: Optional[torch.device] = None
                   ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    pad_id = self.tokenizer.eos_token_id or 0
    all_input_ids = []
    all_labels = []
    all_context_masks = []
    all_action_spans = []
    all_optimal_costs = []
    all_step_costs = []
    step_counts = []

    # NEW: collect per-episode graph data
    episode_worlds = []
    episode_room_positions = []

    for _ in range(batch_size):
        result = self._generate_valid_episode()
        episode_text, actions, n_steps, optimal_cost, step_costs, world = result

        input_ids, labels, context_end_pos, action_spans, room_token_pos = \
            self._tokenize_episode(episode_text)

        step_counts.append(n_steps)
        all_optimal_costs.append(optimal_cost)
        all_step_costs.append(step_costs)
        episode_worlds.append(world)
        episode_room_positions.append(room_token_pos)

        # ... existing padding/context_mask logic unchanged ...

        if len(input_ids) > self.seq_len:
            input_ids = input_ids[:self.seq_len]
            labels = labels[:self.seq_len]
        else:
            pad_len = self.seq_len - len(input_ids)
            input_ids += [pad_id] * pad_len
            labels += [-100] * pad_len

        context_mask_row = [False] * self.seq_len
        for i in range(min(context_end_pos, self.seq_len)):
            context_mask_row[i] = True
        all_context_masks.append(context_mask_row)
        all_action_spans.append(action_spans)

        all_input_ids.append(input_ids)
        all_labels.append(labels)

    # Build graph-distance tensors
    R_max = max(w.n_rooms for w in episode_worlds)
    room_distances = torch.ones(batch_size, R_max, R_max)  # default 1.0 (max distance)
    room_positions = torch.full((batch_size, R_max), -1, dtype=torch.long)  # -1 = padding
    n_rooms_tensor = torch.zeros(batch_size, dtype=torch.long)

    for b, (world, rtp) in enumerate(zip(episode_worlds, episode_room_positions)):
        R = world.n_rooms
        n_rooms_tensor[b] = R

        # All-pairs shortest paths
        sp = world.all_pairs_shortest_paths()

        # Find max finite distance for normalization
        finite_dists = [d for d in sp.values() if d < float('inf') and d > 0]
        max_dist = max(finite_dists) if finite_dists else 1.0

        for i in range(R):
            for j in range(R):
                d = sp.get((i, j), float('inf'))
                if d < float('inf'):
                    room_distances[b, i, j] = d / max_dist
                # else stays 1.0

            # Token position for room i
            if i in rtp:
                pos = rtp[i]
                if pos < self.seq_len:
                    room_positions[b, i] = pos

    input_ids_t = torch.tensor(all_input_ids, dtype=torch.long)
    labels_t = torch.tensor(all_labels, dtype=torch.long)

    if device is not None:
        input_ids_t = input_ids_t.to(device)
        labels_t = labels_t.to(device)

    metadata = {
        "task": "continuous_gridworld",
        "n_rooms_min": self.n_rooms_min,
        "n_rooms_max": self.n_rooms_max,
        "space_size": self.space_size,
        "connect_radius": self.connect_radius,
        "avg_steps": sum(step_counts) / max(len(step_counts), 1),
        "context_mask": torch.tensor(all_context_masks, dtype=torch.bool),
        "action_spans": all_action_spans,
        "optimal_costs": all_optimal_costs,
        "step_costs": all_step_costs,
        # NEW: graph distance data
        "room_distances": room_distances,       # [B, R_max, R_max] normalized
        "room_token_positions": room_positions,  # [B, R_max] token positions (-1=pad)
        "n_rooms": n_rooms_tensor,               # [B]
    }

    if device is not None:
        metadata["context_mask"] = metadata["context_mask"].to(device)
        metadata["room_distances"] = metadata["room_distances"].to(device)
        metadata["room_token_positions"] = metadata["room_token_positions"].to(device)
        metadata["n_rooms"] = metadata["n_rooms"].to(device)

    return input_ids_t, labels_t, metadata
```

---

### 2.2 StructuralEnergy — Graph-Distance Mode

**File:** `fgn/structural_energy.py`

Replace the entire module. The new version supports two modes:
- **positional** (legacy, for backward compatibility / testing)
- **graph** (new default, uses room distance matrix)

```python
"""StructuralEnergy — alignment energy between metric geometry and data structure.

v2: Uses precomputed graph distances instead of positional proximity.
The task provides all-pairs shortest-path distances between rooms and the
token positions where each room is described. The energy drives the metric
to make geodesic distances proportional to graph distances.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class StructuralEnergy(nn.Module):
    """Alignment energy between metric geodesic distances and world structure.

    Two modes:
      - "graph": Uses room_distances and room_token_positions from the task.
        d_struct(i,j) = normalized shortest-path distance between rooms.
      - "positional": Legacy mode. d_struct(i,j) = |pos_i - pos_j| / context_length.
    """

    def __init__(self, max_context_pairs: int = 2048, mode: str = "graph"):
        super().__init__()
        self.max_context_positions = int(math.isqrt(max_context_pairs))
        assert mode in ("graph", "positional"), f"Unknown mode: {mode}"
        self.mode = mode

    def forward(
        self,
        h: torch.Tensor,
        g: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        room_distances: Optional[torch.Tensor] = None,
        room_token_positions: Optional[torch.Tensor] = None,
        n_rooms: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute structural energy.

        Args:
            h: [B, N, d] hidden states
            g: [B, N, d] metric field (per-position, from Softplus)
            context_mask: [B, N] True for [WORLD] positions
            room_distances: [B, R_max, R_max] normalized graph distances (graph mode)
            room_token_positions: [B, R_max] token position per room, -1=pad (graph mode)
            n_rooms: [B] actual room count per episode (graph mode)

        Returns:
            energy: scalar tensor
        """
        if self.mode == "graph":
            return self._graph_energy(h, g, room_distances,
                                      room_token_positions, n_rooms)
        else:
            return self._positional_energy(h, g, context_mask)

    def _graph_energy(
        self,
        h: torch.Tensor,
        g: torch.Tensor,
        room_distances: Optional[torch.Tensor],
        room_token_positions: Optional[torch.Tensor],
        n_rooms: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Structural energy using precomputed graph distances.

        For each pair of rooms (i, j):
          - d_geo = geodesic distance between their token positions in the metric
          - d_struct = normalized shortest-path distance from task
          - energy += (d_geo_norm - d_struct)^2

        Only room-to-room pairs are used (not arbitrary context tokens).
        """
        B, N, d = h.shape
        device = h.device

        if room_distances is None or room_token_positions is None or n_rooms is None:
            return torch.tensor(0.0, device=device, dtype=h.dtype)

        energies = []
        for b in range(B):
            R = n_rooms[b].item()
            if R < 2:
                continue

            # Get valid room token positions (exclude -1 padding, exclude out-of-bounds)
            positions = room_token_positions[b, :R]  # [R]
            valid_mask = (positions >= 0) & (positions < N)
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]  # indices into [0..R-1]
            V = valid_indices.shape[0]

            if V < 2:
                continue

            # Subsample if too many rooms
            if V > self.max_context_positions:
                perm = torch.randperm(V, device=device)[:self.max_context_positions]
                valid_indices = valid_indices[perm]
                V = self.max_context_positions

            # Gather token positions for valid rooms
            tok_pos = positions[valid_indices]  # [V] — token positions

            # Gather hidden states and metric at room token positions
            h_rooms = h[b, tok_pos]   # [V, d]
            g_rooms = g[b, tok_pos]   # [V, d]

            # Geodesic distances: D²[i,j] = sum_k((h_i - h_j)² * g_avg_ij)
            diff = h_rooms.unsqueeze(1) - h_rooms.unsqueeze(0)          # [V, V, d]
            g_avg = (g_rooms.unsqueeze(1) + g_rooms.unsqueeze(0)) / 2   # [V, V, d]
            D_geo = (diff * diff * g_avg).sum(-1)                       # [V, V]

            # Normalize geodesic distances to [0, 1]
            D_geo_max = D_geo.max()
            D_geo_norm = D_geo / (D_geo_max + 1e-8)

            # Extract graph distances for valid room pairs
            # valid_indices maps into room IDs [0..R-1]
            room_ids = valid_indices  # these ARE room IDs (0-indexed)
            D_struct = room_distances[b][room_ids][:, room_ids]  # [V, V]

            # MSE between normalized geodesic and graph distances
            energy_b = ((D_geo_norm - D_struct) ** 2).mean()
            energies.append(energy_b)

        if len(energies) == 0:
            return torch.tensor(0.0, device=device, dtype=h.dtype)

        return torch.stack(energies).mean()

    def _positional_energy(
        self,
        h: torch.Tensor,
        g: torch.Tensor,
        context_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Legacy positional-proxy structural energy (v0.1 behavior).

        Kept for backward compatibility and A/B comparison.
        """
        B, N, d = h.shape

        if context_mask is None:
            return torch.tensor(0.0, device=h.device, dtype=h.dtype)

        energies = []
        for b in range(B):
            mask_b = context_mask[b]
            ctx_indices = mask_b.nonzero(as_tuple=True)[0]
            C = ctx_indices.shape[0]

            if C < 2:
                continue

            if C > self.max_context_positions:
                perm = torch.randperm(C, device=h.device)[:self.max_context_positions]
                ctx_indices = ctx_indices[perm]
                C = self.max_context_positions

            h_ctx = h[b, ctx_indices]
            g_ctx = g[b, ctx_indices]

            diff = h_ctx.unsqueeze(1) - h_ctx.unsqueeze(0)
            g_avg = (g_ctx.unsqueeze(1) + g_ctx.unsqueeze(0)) / 2
            D_geo = (diff * diff * g_avg).sum(-1)

            D_geo_max = D_geo.max()
            D_geo_norm = D_geo / (D_geo_max + 1e-8)

            pos = ctx_indices.float()
            pos_diff = (pos.unsqueeze(1) - pos.unsqueeze(0)).abs()
            ctx_len = pos.max() - pos.min() + 1
            D_struct = pos_diff / (ctx_len + 1e-8)

            energy_b = ((D_geo_norm - D_struct) ** 2).mean()
            energies.append(energy_b)

        if len(energies) == 0:
            return torch.tensor(0.0, device=h.device, dtype=h.dtype)

        return torch.stack(energies).mean()


if __name__ == "__main__":
    print("Testing StructuralEnergy v2...")

    B, N, d = 2, 64, 64
    R = 10  # rooms

    # --- Test graph mode ---
    se_graph = StructuralEnergy(max_context_pairs=2048, mode="graph")

    h = torch.randn(B, N, d)
    g = torch.ones(B, N, d)

    # Create fake room data
    room_distances = torch.rand(B, R, R)
    room_distances = (room_distances + room_distances.transpose(1, 2)) / 2  # symmetric
    for b in range(B):
        for i in range(R):
            room_distances[b, i, i] = 0.0  # zero diagonal

    room_token_positions = torch.arange(R).unsqueeze(0).expand(B, -1) * 5  # rooms at pos 0,5,10,...
    n_rooms = torch.full((B,), R, dtype=torch.long)

    energy = se_graph(h, g, room_distances=room_distances,
                      room_token_positions=room_token_positions, n_rooms=n_rooms)
    print(f"  Graph mode, random h: energy={energy.item():.6f} (should be > 0)")
    assert energy.item() > 0.0

    # Test gradient flow
    h_grad = torch.randn(B, N, d, requires_grad=True)
    g_grad = torch.ones(B, N, d, requires_grad=True)
    energy_grad = se_graph(h_grad, g_grad, room_distances=room_distances,
                           room_token_positions=room_token_positions, n_rooms=n_rooms)
    energy_grad.backward()
    assert h_grad.grad is not None and h_grad.grad.abs().sum() > 0, "No grad for h"
    assert g_grad.grad is not None and g_grad.grad.abs().sum() > 0, "No grad for g"
    print("  Graph mode gradient flow: OK")

    # Test with missing data -> zero energy
    energy_none = se_graph(h, g)
    assert energy_none.item() == 0.0
    print("  Graph mode no data: OK (returns 0)")

    # Test with padding (-1 positions)
    room_positions_padded = room_token_positions.clone()
    room_positions_padded[:, -3:] = -1  # last 3 rooms have no valid position
    energy_padded = se_graph(h, g, room_distances=room_distances,
                             room_token_positions=room_positions_padded, n_rooms=n_rooms)
    print(f"  Graph mode with padding: energy={energy_padded.item():.6f}")
    assert energy_padded.item() >= 0.0

    # --- Test positional mode (legacy) ---
    se_pos = StructuralEnergy(max_context_pairs=2048, mode="positional")
    context_mask = torch.zeros(B, N, dtype=torch.bool)
    context_mask[:, :20] = True
    energy_pos = se_pos(h, g, context_mask=context_mask)
    print(f"  Positional mode: energy={energy_pos.item():.6f}")
    assert energy_pos.item() >= 0.0

    print("StructuralEnergy v2 OK")
```

---

### 2.3 FluidNetModel — Pass Graph Data to Structural Energy

**File:** `fgn/model_fluid.py`

#### 2.3.1 Change StructuralEnergy Construction

In `__init__`, change the StructuralEnergy instantiation to use graph mode:

```python
# OLD:
self.structural_energy = StructuralEnergy(
    max_context_pairs=config.structural_energy_max_pairs,
)

# NEW:
self.structural_energy = StructuralEnergy(
    max_context_pairs=config.structural_energy_max_pairs,
    mode="graph",
)
```

#### 2.3.2 Change Forward Signature

Add three optional parameters to `forward()`:

```python
def forward(
    self,
    input_ids: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    context_mask: Optional[torch.Tensor] = None,
    room_distances: Optional[torch.Tensor] = None,         # NEW
    room_token_positions: Optional[torch.Tensor] = None,    # NEW
    n_rooms: Optional[torch.Tensor] = None,                 # NEW
) -> Dict[str, torch.Tensor]:
```

#### 2.3.3 Change Structural Energy Computation

Replace the existing structural energy block:

```python
# OLD:
if self.lambda_struct > 0 and context_mask is not None:
    g_layer0 = self.layers[0].get_current_metric(h, context)
    e_struct = self.structural_energy(h, g_layer0, context_mask)
else:
    e_struct = torch.tensor(0.0, device=device)

# NEW:
if self.lambda_struct > 0 and room_distances is not None:
    g_layer0 = self.layers[0].get_current_metric(h, context)
    e_struct = self.structural_energy(
        h, g_layer0,
        room_distances=room_distances,
        room_token_positions=room_token_positions,
        n_rooms=n_rooms,
    )
else:
    e_struct = torch.tensor(0.0, device=device)
```

**Note:** The guard condition changes from `context_mask is not None` to `room_distances is not None`. When λ_struct=0, structural energy is still skipped entirely.

---

### 2.4 Training Script — Pass Metadata Through

**File:** `scripts/train_resonant.py`

In the training loop, where the model forward call happens, pass the new metadata fields:

```python
# OLD:
if is_fluid:
    result = compiled_model(input_ids, labels=labels,
                            context_mask=context_mask)

# NEW:
if is_fluid:
    result = compiled_model(
        input_ids,
        labels=labels,
        context_mask=context_mask,
        room_distances=meta.get("room_distances"),
        room_token_positions=meta.get("room_token_positions"),
        n_rooms=meta.get("n_rooms"),
    )
```

This is the **only** change to the training script. Logging, checkpointing, and evaluation calls remain identical.

---

## 3. Experiment Plan

### 3.1 Experiment A-v2 — Graph-Distance Structural Energy

**Purpose:** Confirm that graph-distance structural energy (a) still prevents flattening and (b) achieves non-trivial geometry fidelity.

**Setup:** Identical to Experiment A but with the graph-distance structural energy.

| Run | λ_struct | Steps | Task |
|-----|----------|-------|------|
| baseline | 0.0 | 10K | ContinuousGridWorld (10–15 rooms, R=30) |
| resonant | 0.1 | 10K | ContinuousGridWorld (10–15 rooms, R=30) |

Only two runs needed — λ=0.1 was the sweet spot from Experiment A (moderate curvature, balanced CV/|κ|). We don't need a full sweep since the mechanism already works; we're testing the new target.

**Success Criteria:**

| Metric | Baseline (λ=0) | Resonant (λ=0.1) | Threshold |
|--------|----------------|-------------------|-----------|
| CV @ 10K | ~0.26 (deflated) | >0.30 | Must exceed baseline |
| CE @ 10K | ~0.0000 | ≤0.001 | Must not degrade |
| Geometry Fidelity (mean ρ) | ~0.06 | **>0.3** | **Must exceed 0.3** |
| E_struct | ~0.16 (random) | <0.05 | Shows alignment |

**Critical gate:** Geometry fidelity ρ > 0.3. If the metric's geodesic distances now correlate with actual graph distances, the structural energy is working as intended. If ρ stays near zero, the energy formulation has a deeper problem.

**Running the fidelity diagnostic:** Use the same script that produced the v0.1 fidelity results. It computes Spearman correlation between geodesic distances (from the metric) and graph shortest-path distances (from the world) for each episode. The script should work unchanged since it independently computes graph distances from the world.

### 3.2 Experiment B — Multi-Metric Worlds

**Gated on:** Experiment A-v2 achieving fidelity ρ > 0.3.

Experiment B is defined in RESONANT_GEOMETRY_SPEC.md §4 and is **unchanged** from the original spec. The multi-metric world task, evaluation script, and success criteria remain identical. The only difference is that the resonant model entering Experiment B will use graph-distance structural energy instead of positional proxy.

---

## 4. Implementation Order

1. **ContinuousWorld.all_pairs_shortest_paths()** — new method
2. **_tokenize_episode** — add room_token_positions tracking
3. **_try_generate_episode / _generate_valid_episode / _minimal_episode** — return world object
4. **generate_batch** — build room_distances/room_token_positions tensors
5. **StructuralEnergy** — replace module with v2 (graph + positional modes)
6. **FluidNetModel.forward** — new parameters, graph-mode energy call
7. **train_resonant.py** — pass metadata through
8. **Self-test:** Run `python -m fgn.structural_energy` and `python -m fgn.tasks.continuous_gridworld`
9. **Train:** Experiment A-v2 (two runs: λ=0.0 and λ=0.1)
10. **Evaluate:** Geometry fidelity diagnostic on λ=0.1 checkpoint

---

## 5. Testing Checklist

### 5.1 Unit Tests

- [ ] `python -m fgn.structural_energy` passes (both graph and positional modes)
- [ ] `python -m fgn.tasks.continuous_gridworld` passes (generates batches with room_distances)
- [ ] Verify room_distances has correct shape [B, R_max, R_max] and is symmetric with zero diagonal
- [ ] Verify room_token_positions maps to correct token offsets (spot-check 2-3 episodes)
- [ ] Verify gradient flows through graph-mode structural energy to both h and g

### 5.2 Integration Tests

- [ ] `train_resonant.py` with λ=0.0 runs and produces identical results to v0.1 baseline (structural energy is skipped, so behavior should match exactly)
- [ ] `train_resonant.py` with λ=0.1 runs without errors, e_struct decreases over training
- [ ] e_struct starts at a meaningful value (>0.05) and decreases, not starting at ~0 (which would mean the rooms are already aligned, indicating a bug)

### 5.3 Sanity Checks

- [ ] A single batch has room_distances, room_token_positions, and n_rooms in metadata
- [ ] `room_distances[b, i, i] == 0.0` for all rooms (self-distance is zero)
- [ ] `room_distances[b, i, j] == room_distances[b, j, i]` (symmetry)
- [ ] All room_distances values are in [0.0, 1.0] after normalization
- [ ] room_token_positions values are < seq_len (no out-of-bounds positions)

---

## 6. What Does NOT Change

- **Config (fgn/config.py):** No new fields. `structural_energy_lambda` and `structural_energy_max_pairs` are reused.
- **FluidLayer:** No changes.
- **ContextPool:** No changes.
- **Losses (fgn/losses.py):** No changes.
- **Eval script (scripts/eval_resonant.py):** No changes needed — it computes fidelity independently.
- **Flat model:** No changes — structural energy is FluidNet-only.
- **Training hyperparameters:** Same LR, warmup, scheduler, grad clip, batch size as Experiment A.

---

## 7. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Room token position tracking is off by a few tokens (tokenizer artifacts) | Medium | Spot-check 3 episodes manually: decode tokens at recorded positions, verify they start with room descriptions |
| Graph distances are dominated by a few long paths, compressing most distances near 0 | Low | Normalization by max_finite_dist handles this; verify distribution is not degenerate |
| Structural energy converges but geometry fidelity stays low (metric satisfies energy in a degenerate way) | Low | The graph-distance target is much more constrained than positional proxy — there's essentially one correct geometry up to isometry |
| Speed regression from all_pairs_shortest_paths | Very Low | Dijkstra on 10-15 nodes is microseconds; happens during batch generation, not forward pass |
