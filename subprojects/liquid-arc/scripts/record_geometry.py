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
import liquid_arc.model as _model_module

FGN_ROOT = str(Path(__file__).resolve().parent.parent.parent / "fgn-v3")
sys.path.insert(0, FGN_ROOT)


class GeometryRecorder:
    """Hook into the ODE dynamics to record per-step geometry.

    Wraps euler_solve via monkey-patch to intercept h at every ODE step.
    Does NOT modify the dynamics — pure observation.
    """

    def __init__(self, model: LiquidARCModel):
        self.model = model
        self.dynamics = model.dynamics
        self.records = []
        self._step_data = []

    def start_task(self):
        self._step_data = []

    def record_step(self, step_idx: int, h: torch.Tensor):
        """Record geometry at one ODE step."""
        with torch.no_grad():
            g = self.dynamics.compute_metric_diag(h)
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
        if self._step_data:
            self.records.append(self._step_data)

    def get_targets(self) -> dict:
        """Compute aggregate geometry targets across all recorded tasks."""
        n_steps = len(self.records[0]) if self.records else 0
        n_tasks = len(self.records)

        targets = {
            'n_tasks': n_tasks,
            'n_steps': n_steps,
            'per_step': [],
        }

        for step_idx in range(n_steps):
            step_values = {
                'g_cv': [], 'g_mean': [], 'g_std': [],
                'tau_mean': [], 'tau_std': [], 'tau_min': [], 'tau_max': [],
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LiquidARCModel(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
    cleaned = {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    model.eval()

    recorder = GeometryRecorder(model)

    # Monkey-patch euler_solve in model's namespace to intercept per-step h
    original_euler = _model_module.euler_solve

    def recording_euler_solve(fn, y0, t_span, n_steps, **kwargs):
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

    _model_module.euler_solve = recording_euler_solve

    # Task generators — 70% procedural, 30% ARC
    from liquid_arc.tasks.procedural import ProceduralARCTask
    procedural_task = ProceduralARCTask(seq_len=config.max_seq_len, augment=True)

    arc_task = None
    if data_dir:
        try:
            sys.path.insert(0, FGN_ROOT)
            from fgn.tasks.arc import ARCTask
            arc_task = ARCTask(data_dir=data_dir, seq_len=config.max_seq_len, augment=True)
            print(f"ARC data loaded from {data_dir}")
        except Exception as e:
            print(f"ARC data unavailable ({e}), using 100% procedural")
            arc_task = None

    print(f"Recording geometry from {n_tasks} tasks on {device}...")

    with torch.no_grad():
        for i in range(n_tasks):
            use_arc = arc_task is not None and random.random() < 0.3
            try:
                if use_arc:
                    _, _, meta = arc_task.generate_batch(batch_size=1, device=device)
                else:
                    _, _, meta = procedural_task.generate_batch(batch_size=1, device=device)
            except Exception as e:
                print(f"  Task {i} batch generation failed: {e}")
                continue

            recorder.start_task()

            try:
                result = model(
                    colors=meta['colors'],
                    xs=meta['xs'],
                    ys=meta['ys'],
                    roles=meta['roles'],
                    sep_mask=meta['sep_mask'],
                    sep_types=meta['sep_types'],
                    target_mask=meta['target_mask'],
                    target_labels=meta.get('target_labels'),
                    grid_ids=meta.get('grid_ids'),
                )
                recorder.end_task()
            except Exception as e:
                print(f"  Task {i} forward failed: {e}")
                continue

            if (i + 1) % 200 == 0:
                n_recorded = len(recorder.records)
                print(f"  Recorded {i+1}/{n_tasks} tasks ({n_recorded} successful)")

    _model_module.euler_solve = original_euler

    if not recorder.records:
        print("ERROR: No tasks recorded successfully.")
        return None

    targets = recorder.get_targets()
    torch.save(targets, output_path)

    print(f"\nGeometry targets saved: {output_path}")
    print(f"  Tasks recorded: {targets['n_tasks']}")
    print(f"  Steps per task: {targets['n_steps']}")
    print(f"  Global CV:  {targets['global']['cv_mean']:.3f} ± {targets['global']['cv_std']:.3f}")
    print(f"  Global tau: {targets['global']['tau_mean']:.3f} ± {targets['global']['tau_std']:.3f}")

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
    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--n_tasks', type=int, default=2000)
    parser.add_argument('--output', default='geometry_targets.pt')
    args = parser.parse_args()

    config = LiquidARCConfig.from_yaml(args.config)
    record_geometry(args.checkpoint, config, args.data_dir, args.n_tasks, args.output)
