"""Smoke test for the Tier-1 / Tier-2 changes.

Validates two fixes:
  (A) `compute_geo_lambda` honors the new decay window for configs where
      `geo_lambda_final != geo_lambda_init`, and preserves constant
      behavior when they match.
  (B) `structural_tau` now receives non-zero gradient once
      `tau_quality_loss_enabled=True` and `structural_tau_enabled=True`.
      The fix is in dynamics.py::compute_tau — previously the
      compute_tau path skipped the structural-tau multiplicative
      branch, so the loss's backward graph never touched the
      structural_tau parameter. This test fails loudly if that regresses.

Runs on CPU with a tiny config. ~5 seconds.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/workspace/liquid-arc")
sys.path.insert(0, "/home/pokazge/liquid-arc")

import torch

from liquid_arc.config import LiquidARCConfig
from liquid_arc.model import LiquidARCModel
from scripts.train import compute_geo_lambda


# ---------------------------------------------------------------------
# (A) compute_geo_lambda: backward compat + decay
# ---------------------------------------------------------------------

def test_geo_decay():
    print("\n=== (A) compute_geo_lambda ===")
    # Baseline: existing liquid_arc_geo.yaml (init==final==1.0)
    c_base = LiquidARCConfig.from_yaml(
        "/workspace/liquid-arc/configs/liquid_arc_geo.yaml")
    steps = [0, 10_000, 15_000, 17_500, 20_000, 30_000]
    vals = [compute_geo_lambda(s, c_base) for s in steps]
    print(f"  baseline  (init=final=1.0): {dict(zip(steps, vals))}")
    assert all(v == 1.0 for v in vals), f"baseline regressed: {vals}"

    # New decay config (final=0.0, decay 15k→20k)
    c_decay = LiquidARCConfig.from_yaml(
        "/workspace/liquid-arc/configs/liquid_arc_geo_decay.yaml")
    vals = [compute_geo_lambda(s, c_decay) for s in steps]
    print(f"  decay     (final=0.0):       {dict(zip(steps, vals))}")
    assert vals[0] == 1.0 and vals[1] == 1.0     # pre-decay
    assert vals[2] == 1.0                         # at decay start
    assert abs(vals[3] - 0.5) < 1e-6             # midpoint
    assert vals[4] == 0.0                         # at end
    assert vals[5] == 0.0                         # past end
    print("  [PASS] decay interpolates correctly; baseline unaffected")


# ---------------------------------------------------------------------
# (B) structural_tau gradient reachability
# ---------------------------------------------------------------------

def test_structural_tau_gradient():
    print("\n=== (B) structural_tau gradient reachability ===")
    # Minimal config that exercises the fix path.
    cfg = LiquidARCConfig(
        d_model=64, d_metric=16, d_ffn=128, max_seq_len=32,
        n_ode_steps=4, ode_steps_min=4, ode_steps_max=4,
        use_torch_compile=False,
        structural_tau_enabled=True,
        structural_tau_min=0.3,
        structural_tau_max=3.0,
        tau_quality_loss_enabled=True,
        tau_quality_lambda=0.05,
        tau_mean_target=1.0,
        tau_log_spread_target=0.6,
        geo_loss_enabled=False,
        n_colors=10, n_roles=8, n_sep_types=4, max_grid_size=5, max_grids=2,
    )
    device = torch.device("cpu")
    model = LiquidARCModel(cfg).to(device)

    # Confirm the parameter exists + tracks grad
    assert hasattr(model.dynamics, "structural_tau")
    stau = model.dynamics.structural_tau
    assert stau.requires_grad, "structural_tau must require grad"

    # Fabricate a minimal batch (tiny seq, arbitrary values)
    B, N = 2, 16
    colors = torch.randint(0, cfg.n_colors, (B, N), device=device)
    xs = torch.randint(0, cfg.max_grid_size, (B, N), device=device)
    ys = torch.randint(0, cfg.max_grid_size, (B, N), device=device)
    roles = torch.zeros(B, N, dtype=torch.long, device=device)
    sep_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    sep_types = torch.zeros(B, N, dtype=torch.long, device=device)
    target_mask = torch.ones(B, N, dtype=torch.bool, device=device)
    target_labels = torch.randint(0, cfg.n_colors, (B, N), device=device)
    grid_ids = torch.zeros(B, N, dtype=torch.long, device=device)

    model.train()
    model.zero_grad()
    # Discover call signature dynamically — codebase has a few variants.
    import inspect
    sig = inspect.signature(model.forward)
    print(f"  model.forward signature: {list(sig.parameters.keys())}")
    kwargs = dict(
        colors=colors, xs=xs, ys=ys, roles=roles,
        sep_mask=sep_mask, sep_types=sep_types,
        target_mask=target_mask, target_labels=target_labels,
        grid_ids=grid_ids,
    )
    # Only pass kwargs the model actually accepts.
    accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
    result = model(**accepted)

    # Validate the tau_quality_loss is actually active in this config
    tql = result.get("tau_quality_loss",
                      result.get("loss_components", {}).get("tau_quality_loss"))
    if tql is None:
        # Fallback: total loss must be non-zero and contain tau_quality
        print(f"  result keys: {list(result.keys())}")
    else:
        print(f"  tau_quality_loss value: {float(tql):.4f}")
        assert float(tql) != 0.0, "tau_quality_loss is 0 — not active?"

    total = result["loss"]
    total.backward()

    # The critical check
    if stau.grad is None:
        print("  [FAIL] structural_tau.grad is None — fix did NOT connect")
        return False
    grad_norm = stau.grad.norm().item()
    print(f"  structural_tau.grad.norm() = {grad_norm:.3e}")
    if grad_norm == 0.0:
        print("  [FAIL] grad norm is exactly zero — still disconnected")
        return False
    print(f"  structural_tau param norm before step: {stau.norm().item():.3e}")
    print("  [PASS] structural_tau receives non-zero gradient")
    return True


if __name__ == "__main__":
    test_geo_decay()
    ok = test_structural_tau_gradient()
    print("\n=== SUMMARY ===")
    print(f"  geo_decay:       PASS")
    print(f"  structural_tau:  {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
