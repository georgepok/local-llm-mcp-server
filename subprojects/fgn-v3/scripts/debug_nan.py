#!/usr/bin/env python3
"""Debug NaN in criticality training. Run before full training."""
import sys, json, random, torch
sys.path.insert(0, "/workspace/fgn-v3")

from fgn.config import FGNConfig
from fgn.model_fluid import FluidNetModel
from fgn.tasks.master_world import MasterWorld
from fgn.tasks import get_task
from transformers import AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Same config as run_criticality.sh
config = FGNConfig.from_yaml("configs/resonant_6l.yaml")
config.structural_energy_lambda = 0.0
print(f"Config: d={config.d_model}, layers={config.n_layers}, vocab={config.vocab_size}")
print(f"  structural_energy_lambda={config.structural_energy_lambda}")
print(f"  structural_energy_d_proj={config.structural_energy_d_proj}")

# Create model
model = FluidNetModel(config).to(device)
print(f"Model created: {sum(p.numel() for p in model.parameters()):,} params")

# Check for NaN in model weights
for name, p in model.named_parameters():
    if torch.isnan(p).any():
        print(f"  NaN in {name}")
    if torch.isinf(p).any():
        print(f"  Inf in {name}")

# Create MasterWorld
kwargs = {"n_rooms_max": 20, "n_objects": 4, "space_size": 150.0,
          "connect_radius": 40.0, "locked_door_prob": 0.3,
          "n_rooms_min": 5, "min_steps": 3, "max_steps": 12,
          "min_state_changes": 1}

tokenizer = AutoTokenizer.from_pretrained("gpt2")
master = MasterWorld(n_rooms=20, n_objects=4, space_size=150.0,
                     connect_radius=40.0, locked_door_prob=0.3, seed=42)
task = get_task("CW", tokenizer, seq_len=1024, **kwargs)

print(f"MasterWorld: {master.get_topology_stats()}")

# Generate one episode
world = master.create_episode_world()
print(f"Episode world: {world.n_rooms} rooms, connected={world._is_connected()}")

ep = task._try_generate_episode(override_world=world)
if ep is None:
    print("Episode generation returned None, trying fallback")
    ep = task._generate_valid_episode()

text, actions, n_steps, cost, step_costs, w = ep
ids, labels, ctx_end, spans, rtp = task._tokenize_episode(text)
n_sup = sum(1 for l in labels if l != -100)
print(f"Episode: {len(ids)} tokens, ctx_end={ctx_end}, supervised={n_sup}")

if len(ids) > 1024:
    print(f"  WARNING: truncated from {len(ids)} to 1024")
    ids = ids[:1024]
    labels = labels[:1024]
    n_sup_after = sum(1 for l in labels if l != -100)
    print(f"  supervised after truncation: {n_sup_after}")
else:
    pad_len = 1024 - len(ids)
    ids += [tokenizer.eos_token_id or 0] * pad_len
    labels += [-100] * pad_len

# Build batch
input_ids = torch.tensor([ids], dtype=torch.long, device=device)
labels_t = torch.tensor([labels], dtype=torch.long, device=device)

# Check input validity
print(f"Input range: {input_ids.min().item()}-{input_ids.max().item()}")
print(f"Labels range: {labels_t.min().item()}-{labels_t.max().item()}")
assert input_ids.max().item() < config.vocab_size, f"Token ID {input_ids.max().item()} >= vocab {config.vocab_size}"

# Forward pass WITHOUT autocast first
print("\n--- Forward (float32) ---")
with torch.no_grad():
    result = model(input_ids, labels=labels_t)
    print(f"  loss={result['loss'].item():.4f}")
    print(f"  ce_loss={result['ce_loss'].item():.4f}")
    print(f"  NaN? {torch.isnan(result['loss']).item()}")
    if 'avg_kappa' in result:
        print(f"  kappa={result['avg_kappa'].item():.4f}")

# Forward pass WITH autocast (bfloat16)
print("\n--- Forward (bfloat16 autocast) ---")
with torch.no_grad():
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        result = model(input_ids, labels=labels_t)
        print(f"  loss={result['loss'].item():.4f}")
        print(f"  ce_loss={result['ce_loss'].item():.4f}")
        print(f"  NaN? {torch.isnan(result['loss']).item()}")

# Try with room metadata (full path)
print("\n--- Forward with room metadata ---")
R = world.n_rooms
sp = world.all_pairs_shortest_paths()
finite_dists = [d for d in sp.values() if d < float('inf') and d > 0]
max_dist = max(finite_dists) if finite_dists else 1.0
print(f"  R={R}, max_dist={max_dist:.2f}, finite_dists={len(finite_dists)}")

room_distances = torch.ones(1, R, R, device=device)
room_positions = torch.full((1, R), -1, dtype=torch.long, device=device)
n_rooms = torch.tensor([R], dtype=torch.long, device=device)

for i in range(R):
    for j in range(R):
        d = sp.get((i, j), float('inf'))
        if d < float('inf'):
            room_distances[0, i, j] = d / max_dist
    if i in rtp and rtp[i] < 1024:
        room_positions[0, i] = rtp[i]

# Check for inf/nan in room_distances
print(f"  room_distances NaN: {torch.isnan(room_distances).any().item()}")
print(f"  room_distances Inf: {torch.isinf(room_distances).any().item()}")
print(f"  room_distances range: {room_distances.min().item():.4f}-{room_distances.max().item():.4f}")
print(f"  room_positions valid: {(room_positions >= 0).sum().item()}/{R}")

context_mask = torch.zeros(1, 1024, dtype=torch.bool, device=device)
context_mask[0, :ctx_end] = True

with torch.no_grad():
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        result = model(input_ids, labels=labels_t,
                      context_mask=context_mask,
                      room_distances=room_distances,
                      room_token_positions=room_positions,
                      n_rooms=n_rooms)
        print(f"  loss={result['loss'].item():.4f}")
        print(f"  NaN? {torch.isnan(result['loss']).item()}")

# Test actual training loop (5 steps)
print("\n--- Training loop (5 steps) ---")
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
scaler = torch.amp.GradScaler("cuda", enabled=False)

for step in range(5):
    # Generate fresh batch each step (same as criticality training)
    world_ep = master.create_episode_world()
    ep = task._try_generate_episode(override_world=world_ep)
    if ep is None:
        ep = task._generate_valid_episode()

    text, actions, n_steps_ep, cost, step_costs_ep, w_ep = ep
    ids_ep, labels_ep, ctx_end_ep, spans_ep, rtp_ep = task._tokenize_episode(text)

    if len(ids_ep) > 1024:
        ids_ep = ids_ep[:1024]
        labels_ep = labels_ep[:1024]
    else:
        pad_len = 1024 - len(ids_ep)
        ids_ep += [tokenizer.eos_token_id or 0] * pad_len
        labels_ep += [-100] * pad_len

    input_ids_step = torch.tensor([ids_ep], dtype=torch.long, device=device)
    labels_step = torch.tensor([labels_ep], dtype=torch.long, device=device)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        result = model(input_ids_step, labels=labels_step)
        loss = result["loss"]

    nan_loss = torch.isnan(loss).item()
    print(f"  step {step}: loss={loss.item():.4f}, NaN={nan_loss}, "
          f"kappa={result.get('avg_kappa', torch.tensor(0)).item():.4f}")

    if nan_loss:
        # Check gradients
        print("    Checking model state for NaN...")
        for name, p in model.named_parameters():
            if torch.isnan(p).any():
                print(f"    NaN in param: {name}")
        break

    optimizer.zero_grad()
    loss.backward()

    # Check gradients before clip
    total_grad_norm = 0.0
    has_nan_grad = False
    for name, p in model.named_parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any():
                print(f"    NaN gradient in: {name}")
                has_nan_grad = True
            total_grad_norm += p.grad.data.norm().item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    print(f"    grad_norm={total_grad_norm:.2f}, nan_grad={has_nan_grad}")

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

print("\nDebug complete.")
