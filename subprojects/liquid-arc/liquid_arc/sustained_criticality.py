"""Sustained Criticality losses for LiquidARC.

Self-organized phase transition maintenance via three mechanisms:
1. Criticality loss: keeps D²/4τ ratio near the critical point (target ~18.0)
2. Curvature diversity loss: CV floor/ceiling + metric entropy reward
3. τ-CV coupling: tau adapts to local metric complexity (applied in dynamics.py)

These losses operate on the initial metric/tau state (before ODE), which is the
most direct signal to the geometric parameters. They are computed outside the
compiled ODE loop to avoid torch.compile interaction.
"""

import math
import torch
import torch.nn.functional as F
from typing import Dict, Tuple


def compute_tau_quality_loss(
    tau: torch.Tensor,
    mean_target: float = 1.0,
    log_spread_target: float = 0.6,
) -> torch.Tensor:
    """Replace tau_var_loss with dynamics-aware tau regulation.

    Two components:
    1. Mean anchor: smooth_l1(tau_mean, mean_target) — keep tau in productive ODE range
    2. Log-space spread: (log_tau_std - log_spread_target)² — encourage ~2× ratio between positions

    Args:
        tau: [B, N, 1] per-position time constants
        mean_target: target mean tau (default 1.0 — natural ODE timescale)
        log_spread_target: target std of log(tau) (0.6 → positions differ by ~1.8×)

    Returns:
        loss: scalar tau quality loss
    """
    tau_flat = tau.squeeze(-1)  # [B, N]

    # 1. Anchor mean to productive range
    tau_mean = tau_flat.mean(dim=-1)  # [B]
    mean_anchor = F.smooth_l1_loss(
        tau_mean,
        torch.ones_like(tau_mean) * mean_target
    )

    # 2. Log-space spread (multiplicative differentiation)
    log_tau = torch.log(tau_flat + 1e-8)
    log_tau_std = log_tau.std(dim=-1)  # [B]
    spread_loss = (log_tau_std - log_spread_target) ** 2

    return mean_anchor + 0.5 * spread_loss.mean()


def compute_criticality_loss(
    h: torch.Tensor,
    g: torch.Tensor,
    tau: torch.Tensor,
    t_diffusion_param: torch.Tensor,
    target_ratio: float = 18.0,
    n_pairs: int = 256,
    d_sq_target: float = 60.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute criticality loss driving D²/4τ toward the critical point.

    The critical point is characterized by the ratio median(D²) / (4 * median(τ))
    being near a specific value (default 18.0) at which the heat kernel neither
    over-concentrates (below critical) nor washes out (above critical).

    Args:
        h: [B, N, d] hidden states (used for pair sampling)
        g: [B, N, d] diagonal metric (positive, from Softplus)
        tau: [B, N, 1] per-position time constants
        t_diffusion_param: scalar Parameter (raw, before Softplus)
        target_ratio: target D²/(4τ) ratio (default 18.0)
        n_pairs: number of random position pairs to sample for D² estimate
        d_sq_target: target D² median value for scale anchor (default 60.0)

    Returns:
        loss: scalar criticality loss (smooth_l1 on ratio vs target + D² anchor)
        diagnostics: dict with D²_median, ratio, attn_entropy, entropy_ratio, amp, D_sq_anchor
    """
    B, N, d = h.shape
    device = h.device

    # Actual diffusion time from raw parameter
    t_diff = F.softplus(t_diffusion_param)

    # Sample random position pairs for D² estimation
    # Clamp n_pairs to avoid exceeding available pairs
    actual_pairs = min(n_pairs, N * (N - 1) // 2)
    # Use first batch element for stable gradient path
    g_b = g[0]  # [N, d]
    tau_b = tau[0].squeeze(-1)  # [N]

    # Sample random pairs without replacement (capped at actual_pairs)
    pair_count = min(actual_pairs, N * N)
    i_idx = torch.randint(0, N, (pair_count,), device=device)
    j_idx = torch.randint(0, N, (pair_count,), device=device)

    # Geodesic D²: sum_k g_k(x) * (h_ik - h_jk)^2
    # Use mean metric at each position: g_ij = (g_i + g_j) / 2
    h_b = h[0]  # [N, d]
    diff = h_b[i_idx] - h_b[j_idx]  # [pairs, d]
    g_mean = (g_b[i_idx] + g_b[j_idx]) * 0.5  # [pairs, d]
    d_sq = (diff * diff * g_mean).sum(dim=-1)  # [pairs]

    # Tau at each pair: mean of the two positions
    tau_pairs = (tau_b[i_idx] + tau_b[j_idx]) * 0.5  # [pairs]

    # Compute ratio: median(D²) / (4 * median(τ))
    d_sq_median = d_sq.median()
    tau_median = tau_pairs.median()
    ratio = d_sq_median / (4.0 * tau_median + 1e-8)

    # Smooth L1 loss on log ratio (more stable than raw ratio)
    # Log ratio: log(ratio/target) — zero when ratio == target
    log_ratio = torch.log(ratio / target_ratio + 1e-8)
    ratio_loss = F.smooth_l1_loss(log_ratio, torch.zeros_like(log_ratio), beta=0.5)

    # D² scale anchor: prevent scale drift (optional — set d_sq_target=0 to disable)
    if d_sq_target > 0:
        D_sq_log = torch.log(d_sq_median + 1e-8)
        D_sq_target_log = math.log(d_sq_target)
        D_sq_anchor = 0.1 * (D_sq_log - D_sq_target_log) ** 2
        loss = ratio_loss + D_sq_anchor
    else:
        D_sq_anchor = torch.tensor(0.0, device=h.device)
        loss = ratio_loss

    # Attention entropy diagnostic: compute approximate heat kernel entropy
    # Use a small sample for efficiency
    sample_n = min(32, N)
    g_sample = g_b[:sample_n]  # [sample_n, d]
    h_sample = h_b[:sample_n]  # [sample_n, d]
    sqrt_g_sample = torch.sqrt(g_sample)
    q = h_sample * sqrt_g_sample  # [sample_n, d]
    # Compute kernel: K_ij = exp(-D²_ij / (4t))
    # Factored: logK_ij = q_i·q_j/(2t) - ||q_j||²/(4t)
    dot_qk = torch.mm(q, q.t()) / (2.0 * t_diff)  # [sample_n, sample_n]
    k_norm_sq = (q * q).sum(dim=-1)  # [sample_n]
    bias = -k_norm_sq.unsqueeze(0) / (4.0 * t_diff)  # [1, sample_n]
    log_k = dot_qk + bias  # [sample_n, sample_n]
    attn_weights = F.softmax(log_k, dim=-1)  # [sample_n, sample_n]
    # Entropy per row: H = -sum(p * log(p))
    attn_entropy = -(attn_weights * (attn_weights + 1e-10).log()).sum(dim=-1).mean()

    # Max possible entropy for uniform: log(sample_n)
    max_entropy = math.log(sample_n)
    entropy_ratio = attn_entropy / (max_entropy + 1e-8)

    # Amplitude: std of routed values as proxy for information transmission
    amp = g_b.std() / (g_b.mean() + 1e-8)

    diagnostics = {
        "D_sq_median": d_sq_median.item(),
        "ratio": ratio.item(),
        "attn_entropy": attn_entropy.item(),
        "entropy_ratio": entropy_ratio.item(),
        "amp": amp.item(),
        "D_sq_anchor": D_sq_anchor.item(),
    }

    return loss, diagnostics


def compute_curvature_diversity_loss(
    g: torch.Tensor,
    cv_floor: float = 2.0,
    cv_ceiling: float = 10.0,
    n_bins: int = 32,
) -> torch.Tensor:
    """Compute curvature diversity loss: CV floor/ceiling + metric entropy reward.

    Soft quadratic penalties keep metric CV in [cv_floor, cv_ceiling] band.
    Metric entropy reward (binned histogram) encourages diverse metric values,
    preventing the metric from collapsing to a single value even within the CV band.

    Args:
        g: [B, N, d] diagonal metric tensor (positive, from Softplus)
        cv_floor: minimum allowed CV (soft hinge below this)
        cv_ceiling: maximum allowed CV (soft hinge above this)
        n_bins: number of histogram bins for entropy reward

    Returns:
        loss: scalar combined diversity loss (penalties - entropy_reward)
    """
    # Flatten metric for global statistics
    g_flat = g.reshape(-1)  # [B*N*d]
    g_mean = g_flat.mean()
    g_std = g_flat.std()
    cv = g_std / (g_mean + 1e-8)

    # Soft quadratic floor penalty: (max(0, floor - cv))²
    floor_deficit = F.relu(cv_floor - cv)
    floor_loss = floor_deficit ** 2

    # Soft quadratic ceiling penalty: (max(0, cv - ceiling))²
    ceiling_excess = F.relu(cv - cv_ceiling)
    ceiling_loss = ceiling_excess ** 2

    # Metric entropy reward: encourage diverse metric values
    # Bin g values into histogram and compute entropy
    # Use soft binning via piecewise linear assignment for gradient flow
    g_min = g_flat.detach().min()
    g_max = g_flat.detach().max() + 1e-8
    g_range = g_max - g_min

    # Normalize to [0, 1] for binning
    g_norm = (g_flat - g_min) / g_range  # [B*N*d], values in [0, 1]

    # Soft histogram: each value contributes to adjacent bins
    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1, device=g.device)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5  # [n_bins]
    bin_width = 1.0 / n_bins

    # Distance from each value to each bin center, then triangle kernel
    dists = (g_norm.unsqueeze(-1) - bin_centers.unsqueeze(0)).abs()  # [N_flat, n_bins]
    weights = F.relu(1.0 - dists / bin_width)  # triangle kernel, [N_flat, n_bins]
    bin_counts = weights.sum(dim=0)  # [n_bins]
    bin_probs = bin_counts / (bin_counts.sum() + 1e-8)

    # Entropy: H = -sum(p * log(p))
    entropy = -(bin_probs * (bin_probs + 1e-10).log()).sum()
    max_entropy = math.log(n_bins)
    # Normalize to [0, 1] and negate (we want to maximize entropy)
    entropy_reward = entropy / (max_entropy + 1e-8)

    # Combined: floor + ceiling penalties, minus entropy reward (scaled)
    # Scale entropy reward to be smaller than penalties (0.1x)
    loss = floor_loss + ceiling_loss - 0.1 * entropy_reward

    return loss


def compute_cv_tau_product(cv: float, tau_mean: float) -> float:
    """Compute CV·τ product for logging.

    The CV·τ product is a joint diagnostic for the criticality state.
    Near-critical behavior is associated with specific ranges of this product.

    Args:
        cv: metric coefficient of variation (dimensionless)
        tau_mean: mean tau across positions

    Returns:
        product: cv * tau_mean
    """
    return cv * tau_mean
