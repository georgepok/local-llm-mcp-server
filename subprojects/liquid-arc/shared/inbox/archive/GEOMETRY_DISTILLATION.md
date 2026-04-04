# TASK: Geometry Distillation — Seeding a Correct Architecture from the Phase Transition

**From:** Claude Desktop (Research Direction)
**To:** Claude Code (Implementation)
**Date:** 2026-03-31
**Priority:** HIGH — next-generation architecture

**Prerequisites:**
- Post-transition 5M checkpoint at `/workspace/liquid-arc/PRECIOUS_CHECKPOINTS/5m_post_transition.pt` (step 10000)
- ARC data at `/workspace/fgn-v3/data/arc-repo/data`
- Existing training infrastructure: `scripts/train.py`, `liquid_arc/` module

---

## Motivation

The research program has identified a fundamental level mismatch in the current architecture. Tau controls activation dynamics (inference time, within 16 ODE steps) but doesn't propagate into weight dynamics (learning time, across gradient updates). The gradient through a high-tau position is `∂h_{t+1}/∂h_t = (1 - dt/τ)`, which APPROACHES 1 for large tau — meaning high-tau positions are MORE gradient-transparent, not less. The inference-time fast/slow split doesn't create a learning-time fast/slow split.

This mismatch has caused every harness problem in the program:
- Phase transition crashes (metric changes too fast — no gradient damping at high tau)
- Strategic death (high tau = stable state + fast learning = easy exploit)
- Standing plateau under efficiency regularizer (external control replaces missing intrinsic mechanism)

The fix requires a new architecture with STRUCTURAL tau — input-independent, slow-learning tau that modulates gradient magnitude — present from initialization. But this new architecture needs the phase transition's geometric substrate to be functional. The phase transition isn't reliably reproducible.

**Solution:** Use the post-transition model as a TEACHER. Record its geometric properties (metric CV, tau distribution, routing statistics). Train the new architecture to match these geometric targets rather than discovering them through an unreliable transition. The new model is BORN in the post-transition regime and has the correct two-timescale mechanism from the start.

---

## Phase 1: Record the Teacher's Geometry

### Script: `scripts/record_geometry.py`

Run the post-transition 5M checkpoint on ARC + procedural tasks. Record the geometric properties at each of the 16 ODE steps.

```python
"""Record geometric properties from the post-transition model.

Runs the 5M checkpoint (step 10000) on a mix of ARC + procedural tasks.
At each ODE step, records:
  - Metric field statistics: mean(g), std(g), CV(g)
  - Tau statistics: mean(tau), std(tau), min(tau), max(tau)
  - h trajectory: norm(h), h direction statistics
  - Heat kernel statistics: attention entropy, top-k concentration

These statistics define the GEOMETRIC REGIME the phase transition produced.
The new architecture will be initialized to match this regime.

Usage:
    python scripts/record_geometry.py \
      --checkpoint /workspace/liquid-arc/PRECIOUS_CHECKPOINTS/5m_post_transition.pt \
      --config configs/liquid_arc_5m.yaml \
      --data_dir /workspace/fgn-v3/data/arc-repo/data \
      --n_tasks 2000 \
      --output geometry_targets.pt
"""

import argparse
import sys
from pathlib import Path
import torch
import random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liquid_arc.config import LiquidARCConfig
from liquid_arc.model import LiquidARCModel

# Also need task generators
FGN_ROOT = str(Path(__file__).resolve().parent.parent.parent / "fgn-v3")
sys.path.insert(0, FGN_ROOT)
from fgn.tasks.arc import ARCTask
from liquid_arc.tasks.procedural import ProceduralARCTask


class GeometryRecorder:
    """Hook into the ODE dynamics to record per-step geometry.
    
    Wraps the dynamics.forward() to intercept and record g, tau, h_norm
    at every ODE step. Does NOT modify the dynamics — pure observation.
    """
    
    def __init__(self, model: LiquidARCModel):
        self.model = model
        self.dynamics = model.dynamics
        self.records = []  # list of per-task geometry records
        self._current_task = {}
        self._step_data = []
    
    def start_task(self):
        """Begin recording a new task."""
        self._step_data = []
    
    def record_step(self, step_idx: int, h: torch.Tensor):
        """Record geometry at one ODE step (called from hooked forward)."""
        with torch.no_grad():
            g = self.dynamics.compute_metric(h)
            tau = self.dynamics.compute_tau(h)
            
            g_mean = g.mean().item()
            g_std = g.std().item()
            g_cv = g_std / (g_mean + 1e-8)
            
            tau_flat = tau.squeeze(-1)  # [B, N]
            
            self._step_data.append({
                'step': step_idx,
                'g_mean': g_mean,
                'g_std': g_std,
                'g_cv': g_cv,
                'tau_mean': tau_flat.mean().item(),
                'tau_std': tau_flat.std().item(),
                'tau_min': tau_flat.min().item(),
                'tau_max': tau_flat.max().item(),
                'h_norm': h.norm().item(),
                'h_mean': h.mean().item(),
                'h_std': h.std().item(),
            })
    
    def end_task(self):
        """Finish recording, store summary."""
        if self._step_data:
            self.records.append(self._step_data)
    
    def get_targets(self) -> dict:
        """Compute aggregate geometry targets across all recorded tasks.
        
        Returns a dict of per-step statistics (mean and std across tasks)
        that the new architecture should match.
        """
        n_steps = len(self.records[0]) if self.records else 0
        n_tasks = len(self.records)
        
        targets = {
            'n_tasks': n_tasks,
            'n_steps': n_steps,
            'per_step': [],
        }
        
        for step_idx in range(n_steps):
            step_values = {
                'g_cv': [],
                'g_mean': [],
                'g_std': [],
                'tau_mean': [],
                'tau_std': [],
                'tau_min': [],
                'tau_max': [],
                'h_norm': [],
            }
            
            for task_record in self.records:
                if step_idx < len(task_record):
                    for key in step_values:
                        step_values[key].append(task_record[step_idx][key])
            
            step_target = {}
            for key, values in step_values.items():
                t = torch.tensor(values)
                step_target[f'{key}_mean'] = t.mean().item()
                step_target[f'{key}_std'] = t.std().item()
                step_target[f'{key}_median'] = t.median().item()
                step_target[f'{key}_p25'] = t.quantile(0.25).item()
                step_target[f'{key}_p75'] = t.quantile(0.75).item()
            
            targets['per_step'].append(step_target)
        
        # Global summary (averaged across all steps)
        all_cv = [r[s]['g_cv'] for r in self.records for s in range(len(r))]
        all_tau = [r[s]['tau_mean'] for r in self.records for s in range(len(r))]
        
        targets['global'] = {
            'cv_mean': torch.tensor(all_cv).mean().item(),
            'cv_std': torch.tensor(all_cv).std().item(),
            'tau_mean': torch.tensor(all_tau).mean().item(),
            'tau_std': torch.tensor(all_tau).std().item(),
        }
        
        return targets


def record_geometry(checkpoint_path, config, data_dir, n_tasks, output_path):
    """Main recording function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = LiquidARCModel(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
    # Handle compiled checkpoints
    cleaned = {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    model.eval()
    
    recorder = GeometryRecorder(model)
    
    # Hook into the ODE solver to record at each step
    # The approach: temporarily replace euler_solve with a recording version
    from liquid_arc.solver import euler_solve
    
    def recording_euler_solve(fn, y0, t_span, n_steps):
        t_start, t_end = t_span
        dt = (t_end - t_start) / n_steps
        t = t_start
        y = y0
        for i in range(n_steps):
            if hasattr(fn, 'set_step_embed'):
                fn.set_step_embed(i, n_steps)
            if hasattr(fn, 'set_step_index'):
                fn.set_step_index(i, n_steps)
            recorder.record_step(i, y)
            dy = fn(t, y)
            y = y + dt * dy
            t = t + dt
        recorder.record_step(n_steps, y)  # final state
        return y
    
    # Create task generators
    # 70% procedural, 30% ARC — same mix that produced the phase transition
    procedural_task = ProceduralARCTask(seq_len=2048, augment=True)
    arc_task = ARCTask(data_dir=data_dir, seq_len=2048, augment=True)
    
    # Monkey-patch the solver temporarily
    import liquid_arc.model as model_module
    original_solver = model_module.euler_solve
    model_module.euler_solve = recording_euler_solve
    
    # Record geometry across n_tasks
    print(f"Recording geometry from {n_tasks} tasks...")
    
    with torch.no_grad():
        for i in range(n_tasks):
            # 70/30 procedural/ARC mix
            if random.random() < 0.7:
                batch = procedural_task.generate_batch(batch_size=1, device=device)
            else:
                batch = arc_task.generate_batch(batch_size=1, device=device)
            
            recorder.start_task()
            
            # Forward pass — the hooked solver records geometry at each step
            try:
                result = model(**batch)
            except Exception as e:
                print(f"  Task {i} failed: {e}")
                continue
            
            recorder.end_task()
            
            if (i + 1) % 200 == 0:
                print(f"  Recorded {i+1}/{n_tasks} tasks")
    
    # Restore original solver
    model_module.euler_solve = original_solver
    
    # Compute and save targets
    targets = recorder.get_targets()
    torch.save(targets, output_path)
    
    print(f"\nGeometry targets saved to {output_path}")
    print(f"  Tasks recorded: {targets['n_tasks']}")
    print(f"  Steps per task: {targets['n_steps']}")
    print(f"  Global CV: {targets['global']['cv_mean']:.3f} ± {targets['global']['cv_std']:.3f}")
    print(f"  Global tau: {targets['global']['tau_mean']:.3f} ± {targets['global']['tau_std']:.3f}")
    
    # Print per-step profile
    print(f"\n  Per-step geometry profile:")
    print(f"  {'Step':>4} | {'CV':>8} | {'tau_mean':>8} | {'h_norm':>10}")
    print(f"  {'-'*4}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")
    for s, st in enumerate(targets['per_step']):
        print(f"  {s:>4} | {st['g_cv_mean']:>8.3f} | {st['tau_mean_mean']:>8.3f} | {st['h_norm_mean']:>10.1f}")
    
    return targets


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--n_tasks', type=int, default=2000)
    parser.add_argument('--output', default='geometry_targets.pt')
    args = parser.parse_args()
    
    config = LiquidARCConfig.from_yaml(args.config) if hasattr(LiquidARCConfig, 'from_yaml') \
        else LiquidARCConfig()  # agent: implement from_yaml or load manually
    
    record_geometry(args.checkpoint, config, args.data_dir, args.n_tasks, args.output)
```

### Expected Output

A `geometry_targets.pt` file containing:
```
{
    'n_tasks': 2000,
    'n_steps': 17,  # 16 steps + final state
    'per_step': [
        {'g_cv_mean': 6.8, 'g_cv_std': 0.4, 'tau_mean_mean': 0.65, ...},  # step 0
        {'g_cv_mean': 6.7, 'g_cv_std': 0.5, 'tau_mean_mean': 0.64, ...},  # step 1
        ...
    ],
    'global': {
        'cv_mean': 6.75,
        'cv_std': 0.45,
        'tau_mean': 0.65,
        'tau_std': 0.08,
    }
}
```

This is the numerical fingerprint of the phase transition's product.

---

## Phase 2: New Architecture with Structural Tau

### Changes to ContinuousDynamics

The new architecture adds ONE conceptual component: structural tau. This is a per-position, input-INDEPENDENT parameter that:
1. Modulates the DYNAMIC tau (inference time)
2. Modulates the GRADIENT MAGNITUDE through geometric parameters (learning time)

#### New Parameter: `structural_tau`

Add to `ContinuousDynamics.__init__`:

```python
# Structural tau — input-independent, learned slowly
# This is a PROPERTY of the position, not of the input
# Initialized from geometry targets (Phase 1)
self.structural_tau = nn.Parameter(torch.ones(config.max_seq_len))
# Range: [0.3, 3.0] via softplus
self.structural_tau_min = 0.3
self.structural_tau_max = 3.0
```

#### Modified Tau Computation

In `ContinuousDynamics.forward()`, tau becomes the PRODUCT of structural and dynamic components:

```python
# CURRENT tau computation:
tau_raw = F.softplus(self.tau_net_linear2(F.gelu(self.tau_net_linear1(h_normed))))
tau = self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(tau_raw)

# NEW tau computation:
# Dynamic tau (input-dependent, same as before)
tau_dynamic = F.softplus(self.tau_net_linear2(F.gelu(self.tau_net_linear1(h_normed))))
tau_dynamic = self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(tau_dynamic)

# Structural tau (input-independent, position-based)
N = h.shape[1]
s_tau_raw = self.structural_tau[:N]
s_tau = self.structural_tau_min + (self.structural_tau_max - self.structural_tau_min) * \
        torch.sigmoid(s_tau_raw)
s_tau = s_tau.unsqueeze(0).unsqueeze(-1)  # [1, N, 1]

# Combined: structural modulates dynamic
tau = tau_dynamic * s_tau  # element-wise product
# Clamp to valid range
tau = tau.clamp(min=self.tau_min, max=self.tau_max * self.structural_tau_max)
```

The structural tau MULTIPLIES the dynamic tau. When structural tau is high for a position, that position's effective tau is always high regardless of input — it's a structurally slow position. When structural tau is low, the position responds at its natural dynamic rate.

#### torch.compile Note

`structural_tau` is a fixed-size parameter indexed by `[:N]` where N is the sequence length. As long as N doesn't change between batches (which it doesn't within a task type), this compiles cleanly. Same indexing pattern as `step_embeds`.

### Three-Group Optimizer with Gradient Coupling

The key change: geometric parameter gradients are SCALED by the inverse of structural tau.

```python
# In the training script, replace the two-group optimizer:

# GROUP 1: Structural parameters (MetricNet, TauNet, t_diffusion, structural_tau)
# Learns SLOWLY — these define the geometric substrate
structural_params = list(model.dynamics.metric_net_linear1.parameters()) + \
                    list(model.dynamics.metric_net_linear2.parameters()) + \
                    list(model.dynamics.tau_net_linear1.parameters()) + \
                    list(model.dynamics.tau_net_linear2.parameters()) + \
                    [model.dynamics.t_diffusion, model.dynamics.alpha_logit] + \
                    [model.dynamics.structural_tau] + \
                    list(model.context_pool.parameters())

# GROUP 2: Content parameters (FFN, W_v, W_o, embedding, output head)
# Learns at normal rate — these define task-specific computation
content_params = [p for p in model.parameters() 
                  if id(p) not in {id(sp) for sp in structural_params}]

optimizer = torch.optim.AdamW([
    {'params': structural_params, 'lr': base_lr * 0.01},  # 100× slower
    {'params': content_params, 'lr': base_lr},
], weight_decay=0.01)
```

**The 100× LR ratio** is the gradient coupling that the feedback identified as missing. Geometric parameters learn 100× slower than content parameters. The structural tau then further modulates this: positions where structural_tau is high (geometry-anchoring positions) have their geometric parameter gradients effectively scaled by 1/structural_tau, making them even slower.

The gradient coupling through structural_tau is implicit: positions with high structural_tau have h that changes less per ODE step → the loss gradient w.r.t. MetricNet for those positions is smaller → those geometric weights update less. This is the natural coupling that the feedback described: "high-tau positions also receive smaller gradient updates."

Wait — the feedback pointed out this is exactly what DOESN'T happen with the current architecture because `∂h/∂h_prev = (1 - dt/τ)` approaches 1 for high tau. The FIX is: instead of relying on the implicit gradient coupling (which goes the wrong direction), EXPLICITLY scale geometric parameter gradients by structural_tau in a custom backward hook:

```python
# Explicit gradient coupling — applied after loss.backward(), before optimizer.step()
def apply_structural_gradient_coupling(model):
    """Scale geometric parameter gradients by inverse structural tau.
    
    High structural_tau → gradient scaled DOWN → geometry learns slower at those positions.
    Low structural_tau → gradient unchanged → geometry learns at normal rate.
    
    This EXPLICITLY creates the learning-time timescale separation that the
    inference-time tau creates at the activation level.
    """
    s_tau = torch.sigmoid(model.dynamics.structural_tau)  # [max_seq_len]
    s_tau_mean = s_tau.mean()
    
    # Scale MetricNet gradients
    for p in list(model.dynamics.metric_net_linear1.parameters()) + \
             list(model.dynamics.metric_net_linear2.parameters()):
        if p.grad is not None:
            # Global scaling by mean structural tau (higher = slower)
            p.grad *= (1.0 / (s_tau_mean + 0.1))
    
    # Scale TauNet gradients (so tau itself stabilizes at high values)
    for p in list(model.dynamics.tau_net_linear1.parameters()) + \
             list(model.dynamics.tau_net_linear2.parameters()):
        if p.grad is not None:
            p.grad *= (1.0 / (s_tau_mean + 0.1))

# In training loop, after loss.backward():
loss.backward()
apply_structural_gradient_coupling(model)
optimizer.step()
```

### Config for New Architecture

Create `configs/liquid_arc_v2.yaml`:

```yaml
# LiquidARC v2 — Structural Tau with Gradient Coupling
# Geometry seeded from post-transition teacher

d_model: 768
d_metric: 192
d_ffn: 1536
max_seq_len: 2048
n_ode_steps: 16
tau_min: 0.5
tau_max: 1.0
t_diffusion_init: 1.0
dropout: 0.1
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

# Structural tau (NEW)
structural_tau_enabled: true
structural_tau_min: 0.3
structural_tau_max: 3.0

# CV floor/ceiling
cv_floor_target: 3.0
cv_ceiling_target: 8.0
cv_floor_lambda: 0.1

# Optimizer — three-group with gradient coupling
base_lr: 0.0003
structural_lr_ratio: 0.01  # 100× slower for geometric params
warmup_steps: 500
weight_decay: 0.01

# Training
use_procedural: true
real_arc_mix_ratio: 0.3
batch_size: 4
```

---

## Phase 3: Geometric Initialization (Distillation)

### The Key Idea

Instead of random initialization → slow CV climb → unreliable phase transition, the new model's MetricNet is initialized so that it ALREADY produces the post-transition CV regime on ARC inputs.

Two approaches (try both, compare):

#### Approach A: Statistical Target Matching

Initialize MetricNet weights, then run a short optimization loop that adjusts only MetricNet to match the teacher's geometric statistics:

```python
def initialize_from_geometry_targets(model, targets, n_init_steps=500, device='cuda'):
    """Warm-start MetricNet to produce teacher-like geometry.
    
    Runs ARC + procedural batches through the model and adjusts
    MetricNet weights to match the teacher's CV and tau distribution.
    No task loss — geometry matching only.
    
    Args:
        model: The new LiquidARCModel (v2 with structural tau)
        targets: geometry_targets.pt from Phase 1
        n_init_steps: Number of initialization steps
        device: GPU device
    """
    target_cv = targets['global']['cv_mean']      # ~6.75
    target_tau_mean = targets['global']['tau_mean']  # ~0.65
    
    # Only optimize MetricNet and TauNet during initialization
    init_params = list(model.dynamics.metric_net_linear1.parameters()) + \
                  list(model.dynamics.metric_net_linear2.parameters()) + \
                  list(model.dynamics.tau_net_linear1.parameters()) + \
                  list(model.dynamics.tau_net_linear2.parameters()) + \
                  [model.dynamics.t_diffusion]
    
    init_optimizer = torch.optim.Adam(init_params, lr=1e-3)
    
    procedural = ProceduralARCTask(seq_len=2048)
    
    print(f"Initializing geometry to match teacher (CV={target_cv:.2f}, tau={target_tau_mean:.2f})")
    
    for step in range(n_init_steps):
        batch = procedural.generate_batch(batch_size=4, device=device)
        
        # Embed (no ODE, just get h0)
        colors_masked = batch['colors'].clone()
        colors_masked[batch['target_mask']] = 10  # PAD
        h0 = model.embedding(
            colors_masked, batch['xs'], batch['ys'], batch['roles'],
            batch['sep_mask'], batch['sep_types'], grid_ids=batch['grid_ids']
        )
        
        # Compute current geometry
        g = model.dynamics.compute_metric(h0)
        tau = model.dynamics.compute_tau(h0)
        
        current_cv = g.std() / (g.mean() + 1e-8)
        current_tau_mean = tau.mean()
        
        # Loss: match teacher statistics
        cv_loss = (current_cv - target_cv) ** 2
        tau_loss = (current_tau_mean - target_tau_mean) ** 2
        
        # Also match per-step profile if available
        # (run a few ODE steps and compare intermediate geometry)
        
        loss = cv_loss + tau_loss
        
        init_optimizer.zero_grad()
        loss.backward()
        init_optimizer.step()
        
        if (step + 1) % 100 == 0:
            print(f"  Init step {step+1}: CV={current_cv.item():.3f} "
                  f"(target {target_cv:.3f}), "
                  f"tau={current_tau_mean.item():.3f} "
                  f"(target {target_tau_mean:.3f})")
    
    print(f"  Initialization complete. Final CV={current_cv.item():.3f}, "
          f"tau={current_tau_mean.item():.3f}")
```

#### Approach B: Direct Weight Transfer

Copy the teacher's MetricNet and TauNet weights directly into the student. Since both architectures have the same MetricNet structure (Linear(2d, d_met) → Linear(d_met, d)), the weights are directly compatible:

```python
def transfer_geometry_weights(student, teacher_checkpoint, device='cuda'):
    """Copy MetricNet and TauNet weights from teacher to student.
    
    The student starts with EXACTLY the teacher's geometry.
    The structural tau and gradient coupling then preserve it
    during subsequent training.
    """
    ckpt = torch.load(teacher_checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
    cleaned = {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}
    
    # Transfer MetricNet
    geo_keys = [
        'dynamics.metric_net_linear1.weight', 'dynamics.metric_net_linear1.bias',
        'dynamics.metric_net_linear2.weight', 'dynamics.metric_net_linear2.bias',
        'dynamics.tau_net_linear1.weight', 'dynamics.tau_net_linear1.bias',
        'dynamics.tau_net_linear2.weight', 'dynamics.tau_net_linear2.bias',
        'dynamics.t_diffusion', 'dynamics.alpha_logit',
    ]
    
    transferred = 0
    for key in geo_keys:
        if key in cleaned:
            param = dict(student.named_parameters()).get(key)
            if param is not None and param.shape == cleaned[key].shape:
                param.data.copy_(cleaned[key])
                transferred += 1
    
    print(f"Transferred {transferred}/{len(geo_keys)} geometry parameters from teacher")
    
    # Optionally also transfer context_pool (it's geometric infrastructure)
    cp_keys = [k for k in cleaned if k.startswith('context_pool.')]
    for key in cp_keys:
        param = dict(student.named_parameters()).get(key)
        if param is not None and param.shape == cleaned[key].shape:
            param.data.copy_(cleaned[key])
            transferred += 1
    
    print(f"Total transferred: {transferred} parameters")
```

**Approach B is simpler and more direct.** The teacher's MetricNet IS the product of the phase transition. Copying its weights gives the student exactly that geometry. The structural tau and gradient coupling then PRESERVE it during training (100× slower LR for geometric params) while the content parameters (FFN, W_v, W_o, embedding, output head) learn from scratch.

**Recommendation: Use Approach B (direct weight transfer) as the primary method. Use Approach A as a fallback if weight shapes don't match (e.g., if the new architecture changes MetricNet dimensions).**

---

## Phase 4: Training and Validation

### Training Script Modifications

Modify `scripts/train.py` (or create `scripts/train_v2.py`):

1. Load new config (`liquid_arc_v2.yaml`)
2. Create model with structural tau enabled
3. Initialize geometry from teacher (Approach B)
4. Set up three-group optimizer with gradient coupling
5. Apply `apply_structural_gradient_coupling()` after each backward pass
6. Log structural_tau statistics alongside CV, dynamic tau

### Training Protocol

```bash
# Step 1: Record teacher geometry
python scripts/record_geometry.py \
  --checkpoint /workspace/liquid-arc/PRECIOUS_CHECKPOINTS/5m_post_transition.pt \
  --config configs/liquid_arc_5m.yaml \
  --data_dir /workspace/fgn-v3/data/arc-repo/data \
  --n_tasks 2000 \
  --output geometry_targets.pt

# Step 2: Train new architecture with seeded geometry
python scripts/train_v2.py \
  --config configs/liquid_arc_v2.yaml \
  --data_dir /workspace/fgn-v3/data/arc-repo/data \
  --teacher_checkpoint /workspace/liquid-arc/PRECIOUS_CHECKPOINTS/5m_post_transition.pt \
  --geometry_init weight_transfer \
  --output_dir output_v2/seeded \
  --max_steps 10000 \
  --log_every 50 \
  --eval_every 500 \
  --save_every 2000
```

### What to Monitor

#### Primary: Does the geometry SURVIVE training?

```
| Step | CV  | CV (teacher) | tau_mean | structural_tau_mean | ARC eval xform |
|------|-----|-------------|----------|--------------------|-|
| 0    |     | ~6.75       |          |                    | |
| 1000 |     |             |          |                    | |
| 2000 |     |             |          |                    | |
| 5000 |     |             |          |                    | |
| 10000|     |             |          |                    | |
```

If CV stays in the 6-7 range throughout training (matching teacher), the geometry distillation worked AND the structural tau / gradient coupling is preserving it. If CV drifts away (toward 0 or toward 15+), the preservation mechanism isn't strong enough.

#### Secondary: Does the model learn ARC tasks WITHOUT a phase transition?

The whole point: the model should start training and IMMEDIATELY produce non-trivial ARC predictions (because the geometry is already in the right regime), without the 5000-step plateau that preceded the original transition.

```
| Step | V2 eval xform | Original eval xform (for reference) |
|------|--------------|-------------------------------------|
| 100  |              | ~15% (still in plateau)             |
| 500  |              | ~18% (still in plateau)             |
| 1000 |              | ~20% (still in plateau)             |
| 5000 |              | ~35% (just post-transition)         |
| 10000|              | ~44% (converged)                    |
```

If V2 reaches 35%+ eval xform by step 1000 (vs the original's 5000), the geometry seeding bypassed the transition entirely.

#### Tertiary: Universality Probe

After 10K steps, run the universality probe domains from the V2 checkpoint:

```bash
# Same protocol as the original universality probe
for DOMAIN in sorting logic_inference graph_coloring pattern_completion; do
    python scripts/train.py \
      --config configs/universality_${DOMAIN}.yaml \
      --resume output_v2/seeded/step_10000.pt \
      --max_steps 500 \
      --output_dir output_v2/universality/${DOMAIN}
done
```

**Success criterion:** Same transfer speeds as the original (sorting ~50 steps, logic ~300, graph coloring ~400). If transfer speeds match, the distilled geometry is functionally equivalent to the transitioned geometry.

#### Quaternary: Structural Tau Differentiation

Does structural_tau develop meaningful per-position variation?

```
| Position Type | structural_tau @ step 0 | @ step 5000 | @ step 10000 |
|---|---|---|---|
| Separator tokens | 1.0 | | |
| Demo input cells | 1.0 | | |
| Demo output cells | 1.0 | | |
| Test input cells | 1.0 | | |
| Test output cells | 1.0 | | |
```

If separator tokens (structural anchors) develop HIGH structural_tau and test output tokens (content that must change) develop LOW structural_tau, the model has discovered the inference/structure split through training. The structural tau learned, from gradient signal alone, which positions are geometry-bearing and which are content-bearing.

---

## Phase 5: Isaac Sim Transfer

If Phase 4 succeeds (geometry preserved, ARC performance matches or exceeds original), run the Isaac Sim pipeline on the V2 model:

```bash
python scripts/train_isaac.py \
  --task Isaac-Cartpole-Direct-v0 \
  --checkpoint output_v2/seeded/step_10000.pt \
  --headless --num_envs 1024 \
  --freeze_dynamics false \
  --total_steps 500000
```

Key question: does the V2 model with structural tau produce MORE STABLE phase transitions on Anymal than the original? The original crashed at CV 8.3 during a metric reorganization. With structural tau slowing geometric parameter updates 100×, the same reorganization should happen GRADUALLY rather than catastrophically.

```
| Metric | V1 (original) | V2 (structural tau) |
|---|---|---|
| CV at crash/max | 8.3 (NaN) | ? |
| Phase transitions | 2 (second crashed) | ? |
| Walking achieved | Yes (at -11.2) | ? |
| Stability | NaN at update 75 | ? |
```

---

## Success Criteria

### Phase 1 (Recording)
- **Success:** geometry_targets.pt saved with 2000 task recordings. CV in 6-8 range, tau in 0.5-0.8 range. Clear per-step profile.

### Phase 3 (Initialization)
- **Success:** After weight transfer, model produces CV 6-7 on ARC inputs immediately (before any training). No random initialization warm-up needed.

### Phase 4 (Training)
- **Minimum:** CV stays in 5-8 range throughout 10K steps of training. No phase transition needed. Model learns ARC tasks.
- **Good:** ARC eval xform ≥ 35% by step 1000 (vs original's 5000 to reach similar performance). 5× speedup from bypassing the transition.
- **Strong:** Universality probe shows same transfer speeds as original (sorting ~50, logic ~300). The distilled geometry is functionally equivalent.
- **Headline:** Structural tau develops meaningful differentiation (high for structural positions, low for content positions) — the model discovers the WHERE/WHAT timescale separation through training.

### Phase 5 (Isaac Sim)
- **Success:** More stable than original on Anymal. No NaN crash during phase transitions. Walking achieved.
- **Strong:** Better final reward than V1 because stable training allows longer runs without crashing.

---

## Output

Report to `shared/outbox/GEOMETRY_DISTILLATION_REPORT.md`

Include:
1. Teacher geometry recording results (CV profile, tau distribution)
2. Initialization verification (CV match before training)
3. Training trajectory (CV, tau, structural_tau, eval xform)
4. Comparison to original model's training trajectory
5. Universality probe results (transfer speeds)
6. Structural tau differentiation (per-position analysis)
7. Isaac Sim results (if reached)
8. Assessment: does geometric distillation bypass the phase transition?
9. Assessment: does structural tau create genuine timescale separation?
10. Assessment: is this architecture more stable than V1 for complex tasks?

**This experiment determines whether the elusive phase transition can be captured and transferred, freeing all future models from the requirement to reproduce it. If successful, the post-transition geometry becomes a reproducible initialization recipe rather than a one-time historical event.**
