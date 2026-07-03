"""Fixed-point convergence diagnostic for LiquidARC.

Tests whether the trained dynamics converges to a stable equilibrium when
iterated. If yes, DEQ-style equilibrium-as-ODE-limit is a rigorous framing
of the project; if no, the SOC training has produced non-convergent
dynamics and the "continuous time" claim is dead.

Method
------
Manually unroll the dynamics for many steps (without the chunked / compiled
solver) so we can observe the trajectory. At each step k, record:
    - h_k (the hidden state)
    - per-step relative residual r_k = ||h_{k+1} - h_k|| / ||h_k||

Then check:
    1. Does r_k decay toward zero as k grows? (fixed point exists)
    2. Does the trajectory stabilise within the budget? (DEQ would converge)
    3. What's the residual floor at k → max? (cleanness of fixed point)

For a true contractive ODE: r_k decays exponentially with rate ~1/τ_min.
For Forward Euler at the trained dt/τ: r_k initially decays then stalls
at a floor proportional to dt (the discretisation error).
For a non-convergent / oscillatory system: r_k stays bounded above some
non-trivial floor and may oscillate.

Usage:
    python scripts/eval_fixed_point_convergence.py \
        --checkpoint output_30m/checkpoints/best.pt \
        --data_dir /workspace/liquid-arc/data/arc \
        --max_steps 512 \
        --batch_size 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liquid_arc.config import LiquidARCConfig
from liquid_arc.model import create_model

FGN_ROOT = str(Path(__file__).resolve().parent.parent.parent / "fgn-v3")
sys.path.insert(0, FGN_ROOT)
from fgn.tasks.arc import ARCTask


@torch.no_grad()
def manual_unroll(model, h0, mask, max_steps: int, integration_time: float,
                  record_every: int = 1):
    """Step the dynamics by hand and record per-step relative residuals.

    Mirrors what euler_solve does but lets us observe each h_k. No
    chunked checkpointing needed because we're inference-only here.

    Returns:
        residuals: list of (step_idx, rel_residual, h_norm) tuples
        h_history: list of [B, N, d] tensors at each recorded step
    """
    dt = integration_time / max_steps
    t = 0.0
    h = h0
    residuals = []
    h_history = []

    # Reset any per-step state in the dynamics module
    inner = model.dynamics._orig_mod if hasattr(model.dynamics, "_orig_mod") else model.dynamics
    if hasattr(inner, "reset_fast_weights"):
        inner.reset_fast_weights(h0.shape[0], h0.device, h0.dtype)
    if hasattr(inner, "reset_id_history"):
        inner.reset_id_history(h0.shape[0], h0.shape[1], h0.device, h0.dtype)

    # Run the dynamics manually for max_steps iterations.
    for k in range(max_steps):
        if hasattr(model.dynamics, "set_step_index"):
            model.dynamics.set_step_index(k, max_steps)
        if hasattr(model.dynamics, "set_step_embed"):
            model.dynamics.set_step_embed(k, max_steps)
        dh = model.dynamics(t, h)
        if isinstance(dh, tuple):
            dh = dh[0]
        h_new = h + dt * dh
        # Per-step relative residual = ||h_{k+1} - h_k|| / ||h_k||
        diff_norm = (h_new - h).norm(dim=-1).mean().item()
        h_norm = h.norm(dim=-1).mean().item()
        rel = diff_norm / max(h_norm, 1e-8)
        if (k + 1) % record_every == 0 or k == 0:
            residuals.append((k + 1, rel, h_norm))
            if len(h_history) < 24:  # cap memory: only keep up to 24 snapshots
                h_history.append(h_new.detach().clone())
        h = h_new
        t = t + dt
    return residuals, h_history, h


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_dir", type=str, default="/workspace/liquid-arc/data/arc")
    p.add_argument("--max_steps", type=int, default=512,
                   help="far beyond trained n_ode_steps; tests if dynamics ever settles")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="fixed_point_convergence.json")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    if isinstance(config, dict):
        config = LiquidARCConfig(**config)
    print(f"Trained config: d_model={config.d_model}, "
          f"n_ode_steps={config.n_ode_steps}, "
          f"integration_time={getattr(config, 'integration_time', 1.0)}, "
          f"tau=[{config.tau_min}, {config.tau_max}], "
          f"routing={config.routing_mode}")

    model = create_model(config, device)
    sd = ckpt["model"]
    sd = {k.replace("._orig_mod.", "."): v for k, v in sd.items()}
    remapped = {}
    for k, v in sd.items():
        if "metric_net_linear2." in k and "metric_net_linear2_diag" not in k:
            k = k.replace("metric_net_linear2.", "metric_net_linear2_diag.")
        remapped[k] = v
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing:
        print(f"  [load] {len(missing)} missing keys: {missing[:3]}")
    print(f"Loaded ckpt step={ckpt.get('step', '?')}\n")

    eval_task = ARCTask(seq_len=config.max_seq_len, data_dir=args.data_dir,
                        split="eval", augment=False)

    # Get one representative batch
    _, _, meta = eval_task.generate_batch(args.batch_size, device=device)

    # Compute h0 and set up dynamics context exactly as model.forward does
    model.eval()
    colors_masked = meta["colors"].clone()
    target_input_colors = meta.get("target_input_colors")
    if target_input_colors is not None:
        colors_masked[meta["target_mask"]] = target_input_colors[meta["target_mask"]]

    h0 = model.embedding(colors_masked, meta["xs"], meta["ys"], meta["roles"],
                          meta["sep_mask"], meta["sep_types"],
                          grid_ids=meta.get("grid_ids"))
    h0 = model.persistent.blend(h0)
    context = model.context_pool(h0, meta["context_mask"])
    model.dynamics.set_context(context, mask=None)

    integration_time = getattr(config, "integration_time", 1.0)
    print(f"Manual unroll: max_steps={args.max_steps}, "
          f"dt={integration_time / args.max_steps:.5f}, "
          f"dt/tau ∈ [{integration_time / args.max_steps / config.tau_max:.4f}, "
          f"{integration_time / args.max_steps / config.tau_min:.4f}]\n")

    record_every = max(1, args.max_steps // 64)  # ~64 sample points
    with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                             enabled=(device.type == "cuda")):
        residuals, h_history, h_final = manual_unroll(
            model, h0, mask=None, max_steps=args.max_steps,
            integration_time=integration_time, record_every=record_every,
        )

    # Print per-step residual table
    print(f"Per-step relative residual ||h_{{k+1}} - h_k|| / ||h_k|| over "
          f"{args.max_steps} iterations:")
    print(f"  {'step':>6}  {'rel_residual':>15}  {'h_norm':>12}")
    log_residuals = residuals[:]
    for (k, rel, h_norm) in residuals:
        marker = ""
        if k == config.n_ode_steps:
            marker = " ← trained n"
        print(f"  {k:>6d}  {rel:>15.6f}  {h_norm:>12.4f}{marker}")

    # Decay analysis: is residual decaying as 1/k? exponentially? not at all?
    if len(residuals) >= 4:
        # Compare rel_residual at 25%, 50%, 75%, 100% of max
        idx_25 = len(residuals) // 4
        idx_50 = len(residuals) // 2
        idx_75 = 3 * len(residuals) // 4
        rel_25 = residuals[idx_25][1]
        rel_50 = residuals[idx_50][1]
        rel_75 = residuals[idx_75][1]
        rel_end = residuals[-1][1]
        print(f"\nDecay across the run:")
        print(f"  at 25% of budget (step {residuals[idx_25][0]:>3d}): "
              f"rel_residual = {rel_25:.5f}")
        print(f"  at 50% of budget (step {residuals[idx_50][0]:>3d}): "
              f"rel_residual = {rel_50:.5f}")
        print(f"  at 75% of budget (step {residuals[idx_75][0]:>3d}): "
              f"rel_residual = {rel_75:.5f}")
        print(f"  at end   (step {residuals[-1][0]:>3d}): "
              f"rel_residual = {rel_end:.5f}")

    # Pairwise stability between recorded snapshots
    print(f"\nSnapshot pairwise distance ||h_a - h_b|| / ||h_b|| (lower = more "
          f"stable):")
    print(f"  {'step_a':>6} -> {'step_b':>6}  {'rel':>10}")
    if len(h_history) >= 4:
        # Compare 25%, 50%, 75% to final
        steps_recorded = [residuals[i*record_every//record_every][0] for i in range(min(len(h_history), len(residuals)))]
        # Use the indices of the snapshots we have
        n_h = len(h_history)
        idxs = [n_h // 4, n_h // 2, 3 * n_h // 4, n_h - 1]
        h_final_snap = h_history[-1]
        norm_final = h_final_snap.norm(dim=-1).mean().item()
        for idx in idxs:
            if idx < 0 or idx >= n_h:
                continue
            ha = h_history[idx]
            diff = (ha - h_final_snap).norm(dim=-1).mean().item()
            rel = diff / max(norm_final, 1e-8)
            # find which step this snapshot is from
            step_a = residuals[min(idx, len(residuals) - 1)][0] if idx * record_every < args.max_steps else args.max_steps
            print(f"  {step_a:>6d} -> {args.max_steps:>6d}  {rel:>10.4f}")

    summary = {
        "checkpoint": args.checkpoint,
        "step": ckpt.get("step", None),
        "trained_n_ode_steps": config.n_ode_steps,
        "max_steps_diagnostic": args.max_steps,
        "dt": integration_time / args.max_steps,
        "integration_time": integration_time,
        "tau_min": config.tau_min,
        "tau_max": config.tau_max,
        "residuals": [(int(s), float(r), float(n)) for (s, r, n) in residuals],
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {args.out}")

    # Verdict
    if len(residuals) >= 4:
        rel_at_trained = next(
            (r for (s, r, _) in residuals if s == config.n_ode_steps), None)
        rel_at_end = residuals[-1][1]
        print(f"\n{'=' * 60}")
        print(f"Verdict")
        print(f"{'=' * 60}")
        if rel_at_end < 0.001:
            print(f"  Fixed point: STRONG (rel_residual at end = {rel_at_end:.5f})")
            print(f"  → Dynamics converges. DEQ framing is rigorous.")
        elif rel_at_end < 0.01:
            print(f"  Fixed point: WEAK (rel_residual at end = {rel_at_end:.5f})")
            print(f"  → Dynamics nearly converges; DEQ likely usable.")
        elif rel_at_end < 0.1:
            print(f"  Fixed point: MARGINAL (rel_residual at end = {rel_at_end:.5f})")
            print(f"  → Dynamics doesn't fully converge but residual is bounded.")
            print(f"     Could be discretisation floor or genuine non-convergence.")
        else:
            print(f"  Fixed point: ABSENT (rel_residual at end = {rel_at_end:.5f})")
            print(f"  → Dynamics is non-convergent. SOC has trained the model")
            print(f"     into a non-equilibrium regime.")
        if rel_at_trained is not None:
            print(f"\n  Per-step residual at trained n={config.n_ode_steps}: "
                  f"{rel_at_trained:.5f}")
            print(f"  Per-step residual at {args.max_steps} steps: "
                  f"{rel_at_end:.5f}")
            ratio = rel_at_end / rel_at_trained if rel_at_trained > 0 else float('nan')
            print(f"  Ratio (end / trained): {ratio:.3f}")
            if ratio < 0.1:
                print(f"  → Strong decay; convergent dynamics.")
            elif ratio < 0.5:
                print(f"  → Modest decay; dynamics slowly settling.")
            else:
                print(f"  → Little/no decay; non-convergent or chaotic dynamics.")


if __name__ == "__main__":
    main()
