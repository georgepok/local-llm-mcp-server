"""Model utilities — forward_with_external_context wrapper.

Wraps LiquidARCModel.forward() to bypass ContextPool and use
externally-provided z_context instead. Zero modifications to liquid-arc.
"""

import sys
from pathlib import Path
from typing import Dict, Optional

import torch

# Import from liquid-arc
_LIQUID_ARC_ROOT = str(Path(__file__).resolve().parent.parent.parent / "liquid-arc")
if _LIQUID_ARC_ROOT not in sys.path:
    sys.path.insert(0, _LIQUID_ARC_ROOT)

from liquid_arc.model import LiquidARCModel, PAD_COLOR
from liquid_arc.solver import euler_solve, deq_solve, invertible_euler_solve


def forward_with_external_context(
    model: LiquidARCModel,
    z_context: torch.Tensor,
    colors: torch.Tensor,
    xs: torch.Tensor,
    ys: torch.Tensor,
    roles: torch.Tensor,
    sep_mask: torch.Tensor,
    sep_types: torch.Tensor,
    target_mask: torch.Tensor,
    target_labels: Optional[torch.Tensor] = None,
    context_mask: Optional[torch.Tensor] = None,
    target_input_colors: Optional[torch.Tensor] = None,
    grid_ids: Optional[torch.Tensor] = None,
    n_steps: Optional[int] = None,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """Run LiquidARCModel forward with externally-provided context.

    Duplicates the forward() flow but replaces:
        context = self.context_pool(h0, context_mask)
    with:
        context = z_context  (externally provided [B, d_model] tensor)

    Everything else is identical: same embedding, ODE solver, output head, loss.
    Does NOT touch the compiled dynamics path — set_context() is called before ODE.

    Args:
        model: LiquidARCModel instance
        z_context: [B, d_model] external context vector (from z_to_context projection)
        colors, xs, ys, roles, sep_mask, sep_types, target_mask: standard model inputs
        target_labels, context_mask, target_input_colors, grid_ids: optional
        n_steps: ODE step count override

    Returns:
        Same result dict as LiquidARCModel.forward()
    """
    config = model.config
    device = colors.device
    actual_steps = n_steps if n_steps is not None else config.n_ode_steps

    # Mask test output colors with test input colors (or PAD)
    colors_masked = colors.clone()
    if target_input_colors is not None:
        colors_masked[target_mask] = target_input_colors[target_mask]
    else:
        colors_masked[target_mask] = PAD_COLOR

    # Embed once
    h0 = model.embedding(colors_masked, xs, ys, roles, sep_mask, sep_types,
                          grid_ids=grid_ids)

    # Use external context instead of context_pool
    model.dynamics.set_context(z_context, mask=None)
    model.dynamics.set_n_steps(actual_steps)

    # Diagnostics from initial state
    g_init = model.dynamics.compute_metric(h0)
    kappa = model.curvature_engine(g_init)
    metric_cv = g_init.std() / (g_init.mean() + 1e-8)

    if config.channel_gate_enabled:
        gate_init = model.dynamics.compute_gate(h0)
        tau_avg_val = gate_init.mean()
        tau_std_val = gate_init.mean(dim=-1).std(dim=1).mean()
        tau_var_val = gate_init.mean(dim=-1).var(dim=1).mean()
        tau_min_val = gate_init.min()
        tau_max_val = gate_init.max()
    else:
        tau_init = model.dynamics.compute_tau(h0)
        tau_avg_val = tau_init.mean(dim=1).mean()
        tau_flat = tau_init.squeeze(-1)
        tau_std_val = tau_flat.std(dim=1).mean()
        tau_var_val = tau_flat.var(dim=1).mean()
        tau_min_val = tau_flat.min()
        tau_max_val = tau_flat.max()

    # ODE integration — mirror solver dispatch from LiquidARCModel.forward()
    if config.deq_solver:
        h = deq_solve(
            model.dynamics, h0, t_span=(0.0, 1.0),
            n_steps=actual_steps, n_ift_iters=config.deq_ift_iters,
        )
    elif config.invertible_solver:
        h = invertible_euler_solve(
            model.dynamics, h0, t_span=(0.0, 1.0),
            n_steps=actual_steps, n_fp_iters=config.n_fp_iters,
        )
    else:
        h = euler_solve(model.dynamics, h0, t_span=(0.0, 1.0), n_steps=actual_steps)

    # Output head
    logits = model.output_head(model.norm_out(h))

    result = {
        "logits": logits,
        "h_final": h,
        "metric_cv": metric_cv,
        "avg_kappa": kappa.abs().mean(),
        "tau_avg": tau_avg_val,
        "tau_std": tau_std_val,
        "tau_min": tau_min_val,
        "tau_max": tau_max_val,
        "geo_loss": torch.tensor(0.0, device=device),
        "geo_mse": torch.tensor(0.0, device=device),
    }

    # Loss + accuracy (reuse model's internal method)
    if target_labels is not None:
        result.update(model._compute_loss(
            logits, target_labels, target_mask, target_input_colors,
            kappa, tau_var_val, metric_cv, device,
        ))

    return result
