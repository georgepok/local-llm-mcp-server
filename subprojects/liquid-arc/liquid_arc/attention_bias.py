"""Attention Bias — computes geometric bias matrix from LiquidARC ODE state.

LiquidARC → LLM bridge: the ODE's learned Riemannian metric produces
an attention bias B_ij = q_i·k_j/(2t) - ||k_j||²/(4t) that tells the LLM
which positions should attend to which. The LLM adds this to its attention
logits: attn_logits += λ * B.

Uses SDPA factorization — same trick as the internal ODE heat kernel.
The N×N bias matrix is computed as q·k^T without materializing D²_ij directly,
matching the factorization used during training.

This is a pure computation module — no model weights, no state.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple


def compute_attention_bias(
    dynamics,
    h_ode: torch.Tensor,
    token_sources: list = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute geometric attention bias from LiquidARC ODE state.

    Instead of materializing D²_ij = sum_k g_k(h_i - h_j)², factorize as:
        B_ij = q_i·k_j/(2t) - ||k_j||²/(4t)
    where q = k = h_normed * sqrt(g)

    This is the pre-softmax logit of the heat kernel — same as what
    the ODE computes internally, but extracted before softmax.

    Args:
        dynamics: ContinuousDynamics module (has MetricNet, TauNet, t_diffusion)
        h_ode: [B, N, d] ODE state (token-level, N can be large)

    Returns:
        bias: [N, N] attention bias matrix (from first batch element)
        diagnostics: dict with cv, D_sq_4tau, tau_mean, criticality_flag
    """
    with torch.no_grad():
        B, N, d = h_ode.shape
        # Ensure dtype matches dynamics weights
        param_dtype = next(dynamics.parameters()).dtype
        h_ode = h_ode.to(param_dtype)

        # MetricNet forward
        h_normed = dynamics.norm_geo(h_ode)
        context = (dynamics._context if dynamics._context is not None
                   else torch.zeros(B, d, device=h_ode.device, dtype=param_dtype))
        ctx_exp = context.unsqueeze(1).expand(-1, N, -1)
        metric_input = torch.cat([h_normed, ctx_exp], dim=-1)
        hidden = F.gelu(dynamics.metric_net_linear1(metric_input))
        g = F.softplus(dynamics.metric_net_linear2_diag(hidden))  # [B, N, d]

        # SDPA factorization: q = k = h_normed * sqrt(g)
        sqrt_g = g.sqrt()
        qk = h_normed * sqrt_g  # [B, N, d]

        t_diff = F.softplus(dynamics.t_diffusion)

        # B_ij = q_i·k_j/(2t) - ||k_j||²/(4t)
        dot_qk = torch.bmm(qk, qk.transpose(1, 2)) / (2.0 * t_diff)  # [B, N, N]
        k_norm_sq = (qk * qk).sum(dim=-1, keepdim=True)               # [B, N, 1]
        bias = dot_qk - k_norm_sq.transpose(1, 2) / (4.0 * t_diff)    # [B, N, N]

        # Diagnostics — tau
        tau = dynamics.compute_tau(h_ode)  # [B, N, 1]
        tau_mean = tau.mean()
        tau_std = tau.std()
        cv = (g.std() / (g.mean() + 1e-8)).item()

        # Actual pairwise D² — GUARANTEED cross-event sampling
        D_sq_within = D_sq_across = D_sq_all_median = 0.0
        D_sq_all_mean = D_sq_all_std = 0.0
        ratio_actual = 0.0
        n_cross_pairs = 0

        if N > 1 and token_sources and len(token_sources) == N:
            # Group token indices by event_id
            event_groups = {}
            for idx, eid in enumerate(token_sources):
                event_groups.setdefault(eid, []).append(idx)
            unique_events = list(event_groups.keys())

            # Cross-event pairs: sample one token from each pair of distinct events
            cross_i, cross_j = [], []
            within_i, within_j = [], []
            import random as _rng
            for a_idx in range(len(unique_events)):
                for b_idx in range(a_idx + 1, len(unique_events)):
                    grp_a = event_groups[unique_events[a_idx]]
                    grp_b = event_groups[unique_events[b_idx]]
                    # Sample up to 10 pairs per event combination
                    n_sample = min(10, len(grp_a), len(grp_b))
                    for _ in range(n_sample):
                        cross_i.append(_rng.choice(grp_a))
                        cross_j.append(_rng.choice(grp_b))

            # Within-event pairs: sample from same event
            for eid, indices in event_groups.items():
                if len(indices) > 1:
                    n_sample = min(10, len(indices) * (len(indices) - 1) // 2)
                    for _ in range(n_sample):
                        a = _rng.choice(indices)
                        b = _rng.choice(indices)
                        if a != b:
                            within_i.append(a)
                            within_j.append(b)

            # Compute D² for cross-event pairs
            if cross_i:
                ci = torch.tensor(cross_i, device=h_ode.device)
                cj = torch.tensor(cross_j, device=h_ode.device)
                delta_c = h_ode[0, ci, :] - h_ode[0, cj, :]
                g_avg_c = (g[0, ci, :] + g[0, cj, :]) * 0.5
                D_sq_cross = (delta_c * g_avg_c * delta_c).sum(dim=-1)
                D_sq_across = D_sq_cross.median().item()
                n_cross_pairs = len(cross_i)

            # Compute D² for within-event pairs
            if within_i:
                wi = torch.tensor(within_i, device=h_ode.device)
                wj = torch.tensor(within_j, device=h_ode.device)
                delta_w = h_ode[0, wi, :] - h_ode[0, wj, :]
                g_avg_w = (g[0, wi, :] + g[0, wj, :]) * 0.5
                D_sq_with = (delta_w * g_avg_w * delta_w).sum(dim=-1)
                D_sq_within = D_sq_with.median().item()

            ratio_actual = D_sq_across / (4.0 * tau_mean.item() + 1e-8)

        elif N > 1:
            # No event_ids — fall back to random sampling
            n_pairs = min(200, N * (N - 1) // 2)
            idx_i = torch.randint(0, N, (n_pairs,), device=h_ode.device)
            idx_j = (idx_i + torch.randint(1, N, (n_pairs,), device=h_ode.device)) % N
            delta = h_ode[0, idx_i, :] - h_ode[0, idx_j, :]
            g_avg_f = (g[0, idx_i, :] + g[0, idx_j, :]) * 0.5
            D_sq_actual = (delta * g_avg_f * delta).sum(dim=-1)
            D_sq_across = D_sq_actual.median().item()
            ratio_actual = D_sq_across / (4.0 * tau_mean.item() + 1e-8)

        # TauNet logit diagnostics — pre-sigmoid values
        tau_logits = dynamics.tau_net_linear2(
            torch.nn.functional.gelu(dynamics.tau_net_linear1(h_normed))
        )  # [B, N, 1]
        tau_logit_mean = tau_logits.mean().item()
        tau_logit_std = tau_logits.std().item()

        # Bias matrix statistics — overall and by event pair type
        B_mat = bias[0]
        B_flat = B_mat.flatten()
        B_max = B_flat.max().item()
        B_min = B_flat.min().item()
        B_range = B_max - B_min
        B_std = B_flat.std().item()

        # B breakdown by within/across event
        B_within_mean = 0.0
        B_across_mean = 0.0
        B_across_max = float('-inf')
        if token_sources and len(token_sources) == N:
            src = token_sources
            # Build masks for within-event and cross-event pairs
            within_vals = []
            across_vals = []
            # Sample instead of full N×N for efficiency
            n_sample = min(500, N * N)
            si = torch.randint(0, N, (n_sample,), device=B_mat.device)
            sj = torch.randint(0, N, (n_sample,), device=B_mat.device)
            for k in range(n_sample):
                i_idx, j_idx = si[k].item(), sj[k].item()
                if i_idx == j_idx:
                    continue
                val = B_mat[i_idx, j_idx].item()
                if src[i_idx] == src[j_idx]:
                    within_vals.append(val)
                else:
                    across_vals.append(val)
                    if val > B_across_max:
                        B_across_max = val
            B_within_mean = sum(within_vals) / max(len(within_vals), 1)
            B_across_mean = sum(across_vals) / max(len(across_vals), 1)
            if not across_vals:
                B_across_max = 0.0

        # Attention entropy on the NORMALIZED bias (what Qwen3 actually sees)
        # after bias_lambda scaling. The raw bias has extreme values that make
        # softmax degenerate; the normalized version is what matters.
        import math as _math
        sample_n = min(64, N)
        # Sample rows spread across the full token range, not just first 64
        if N > sample_n:
            step = N // sample_n
            sample_idx = list(range(0, N, step))[:sample_n]
        else:
            sample_idx = list(range(N))
        B_rows = bias[0, sample_idx, :]  # [sample_n, N]
        # Normalize like QwenBridge does
        b_mean = bias[0].mean()
        b_std = bias[0].std().clamp(min=1e-8)
        target_range = 2.0 * _math.log(max(N, 2))
        # Per-row normalization (matching QwenBridge)
        row_mean = B_rows.mean(dim=-1, keepdim=True)
        row_centered = B_rows - row_mean
        row_range = (row_centered.max(dim=-1, keepdim=True).values
                     - row_centered.min(dim=-1, keepdim=True).values).clamp(min=1e-8)
        B_norm_rows = row_centered / row_range * target_range
        K_sample = torch.softmax(B_norm_rows, dim=-1)  # [sample_n, N]
        attn_entropy = -(K_sample * (K_sample + 1e-10).log()).sum(dim=-1).mean().item()
        max_entropy = _math.log(N) if N > 1 else 1.0
        entropy_ratio = attn_entropy / max_entropy  # 0=identity, 1=uniform

    print(f"  [bias] [{N}x{N}] CV={cv:.2f} "
          f"Bx={B_across_mean:.0f} Bx_max={B_across_max:.0f} Br={B_range:.0f} "
          f"H={attn_entropy:.2f}/{max_entropy:.2f}={entropy_ratio:.2f} "
          f"tau={tau_mean.item():.2f}±{tau_std.item():.4f}")

    return bias[0], {
        'cv': cv,
        'attn_entropy': attn_entropy,
        'entropy_ratio': entropy_ratio,
        'D_sq_4tau': ratio_actual,
        'D_sq_across': D_sq_across,
        'D_sq_within': D_sq_within,
        'B_max': B_max,
        'B_min': B_min,
        'B_range': B_range,
        'B_std': B_std,
        'B_within_mean': B_within_mean,
        'B_across_mean': B_across_mean,
        'B_across_max': B_across_max,
        'tau_mean': tau_mean.item(),
        'tau_std': tau_std.item(),
        # Criticality assessment: entropy_ratio in [0.3, 0.7] = structured but not degenerate
        'criticality_flag': 0.2 < entropy_ratio < 0.8,
    }


def extend_bias(
    bias_existing: torch.Tensor,
    dynamics,
    h_ode: torch.Tensor,
    new_idx: int,
) -> torch.Tensor:
    """Extend bias matrix by one row/column for a newly added token.

    Instead of recomputing the full N×N matrix, compute only the new row
    and column corresponding to the token at new_idx. This is O(N) rather
    than O(N²) for incremental generation updates.

    Args:
        bias_existing: [N_old, N_old] existing bias matrix
        dynamics: ContinuousDynamics module
        h_ode: [1, N_new, d] ODE state including the new token at new_idx
        new_idx: index of the new token in h_ode (typically N_new - 1)

    Returns:
        bias_new: [N_new, N_new] extended bias matrix
    """
    with torch.no_grad():
        N_old = bias_existing.shape[0]
        N_new = h_ode.shape[1]
        d = h_ode.shape[2]

        param_dtype = next(dynamics.parameters()).dtype
        h_ode = h_ode.to(param_dtype)

        # Compute metric for all positions (needed for new row/col)
        h_normed = dynamics.norm_geo(h_ode)
        context = (dynamics._context if dynamics._context is not None
                   else torch.zeros(1, d, device=h_ode.device, dtype=param_dtype))
        ctx_exp = context.unsqueeze(1).expand(-1, N_new, -1)
        metric_input = torch.cat([h_normed, ctx_exp], dim=-1)
        hidden = F.gelu(dynamics.metric_net_linear1(metric_input))
        g = F.softplus(dynamics.metric_net_linear2_diag(hidden))  # [1, N_new, d]

        sqrt_g = g.sqrt()
        qk = (h_normed * sqrt_g)[0]  # [N_new, d]

        t_diff = F.softplus(dynamics.t_diffusion)

        # New row: bias[new_idx, j] for all j in [0, N_new)
        # New col: bias[i, new_idx] for all i in [0, N_new)
        q_new = qk[new_idx]          # [d]
        k_norm_sq_all = (qk * qk).sum(dim=-1)  # [N_new]

        # Row: q_new · k_j / (2t) - ||k_j||² / (4t)
        new_row = (qk @ q_new) / (2.0 * t_diff) - k_norm_sq_all / (4.0 * t_diff)  # [N_new]

        # Col: q_i · k_new / (2t) - ||k_new||² / (4t)
        k_norm_sq_new = (q_new * q_new).sum()
        new_col = (qk @ q_new) / (2.0 * t_diff) - k_norm_sq_new / (4.0 * t_diff)  # [N_new]

        # Assemble N_new × N_new matrix
        bias_new = torch.zeros(N_new, N_new, device=bias_existing.device,
                               dtype=bias_existing.dtype)
        # Copy existing block
        bias_new[:N_old, :N_old] = bias_existing
        # Fill new row and column
        bias_new[new_idx, :] = new_row.to(bias_existing.dtype)
        bias_new[:, new_idx] = new_col.to(bias_existing.dtype)

    return bias_new
