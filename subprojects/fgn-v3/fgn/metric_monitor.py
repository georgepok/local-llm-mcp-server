"""MetricMonitor — shrink-and-perturb for crystallized metric dimensions.

Tracks per-dimension variance of the metric tensor g across positions.
When a dimension's variance falls in the bottom percentile (crystallization),
perturbs the metric network weights to revive it.

Also tracks metric volatility for plasticity telemetry.
"""

import torch
import torch.nn as nn


class MetricMonitor:
    """Monitor and maintain metric tensor plasticity.

    Attaches to FluidNet layers to track per-dimension metric health
    and apply targeted perturbations when dimensions crystallize.
    """

    def __init__(self, crystal_percentile: float = 0.05,
                 perturb_alpha: float = 0.9, perturb_sigma: float = 0.01,
                 check_every: int = 500):
        """
        Args:
            crystal_percentile: bottom fraction of dimensions to consider
                crystallized (0.05 = bottom 5%)
            perturb_alpha: mixing coefficient for shrink-and-perturb
                W_new = alpha * W_old + (1-alpha) * N(0, sigma)
            perturb_sigma: std of noise injection for dead dimensions
            check_every: how often to check and perturb (in training steps)
        """
        self.crystal_percentile = crystal_percentile
        self.perturb_alpha = perturb_alpha
        self.perturb_sigma = perturb_sigma
        self.check_every = check_every

        # Tracking state
        self._prev_g_mean = None  # for volatility computation
        self.last_volatility = 0.0
        self.last_n_crystallized = 0
        self.last_n_perturbed = 0
        self.total_perturbations = 0

    @torch.no_grad()
    def check_and_perturb(self, model, step: int) -> dict:
        """Check metric health across all layers and perturb dead dimensions.

        Uses relative threshold: only the bottom crystal_percentile of
        dimensions (by health/variance) are considered crystallized.

        Args:
            model: FluidNetModel (unwrapped, not compiled)
            step: current training step

        Returns:
            Dict with monitoring stats
        """
        if step % self.check_every != 0:
            return {}

        stats = {
            "n_crystallized": 0,
            "n_perturbed": 0,
            "n_total_dims": 0,
            "per_layer": [],
        }

        for layer_idx, layer in enumerate(model.layers):
            # Get the metric output layer weights
            metric_linear = layer.metric_net_linear2

            # Compute per-dimension variance of the output weights
            # metric_net_linear2.weight shape: [d_model, d_metric]
            w = metric_linear.weight.data  # [d_model, d_metric]

            # Per output-dimension variance
            dim_var = w.var(dim=1)  # [d_model]

            # Combine with bias spread if present
            if metric_linear.bias is not None:
                b = metric_linear.bias.data  # [d_model]
                bias_spread = (b - b.mean()).abs()
                health = dim_var + bias_spread * 0.1
            else:
                health = dim_var

            n_dims = health.shape[0]
            stats["n_total_dims"] += n_dims

            # Relative threshold: bottom percentile of health
            threshold = health.quantile(self.crystal_percentile)
            crystallized = health < threshold
            n_crystal = crystallized.sum().item()
            stats["n_crystallized"] += n_crystal

            layer_stat = {
                "layer": layer_idx,
                "n_crystallized": n_crystal,
                "n_perturbed": 0,
                "health_min": health.min().item(),
                "health_mean": health.mean().item(),
                "health_max": health.max().item(),
                "threshold": threshold.item(),
            }

            if n_crystal > 0:
                # Shrink-and-perturb: W_new = alpha * W_old + (1-alpha) * N(0, sigma)
                crystal_mask = crystallized.unsqueeze(1).expand_as(w)  # [d, d_metric]
                noise = torch.randn_like(w) * self.perturb_sigma
                w_new = torch.where(
                    crystal_mask,
                    self.perturb_alpha * w + (1 - self.perturb_alpha) * noise,
                    w,
                )
                metric_linear.weight.data.copy_(w_new)

                # Also perturb bias for crystallized dims
                if metric_linear.bias is not None:
                    b_noise = torch.randn_like(b) * self.perturb_sigma
                    b_new = torch.where(
                        crystallized,
                        self.perturb_alpha * b + (1 - self.perturb_alpha) * b_noise,
                        b,
                    )
                    metric_linear.bias.data.copy_(b_new)

                layer_stat["n_perturbed"] = n_crystal
                stats["n_perturbed"] += n_crystal
                self.total_perturbations += n_crystal

            stats["per_layer"].append(layer_stat)

        self.last_n_crystallized = stats["n_crystallized"]
        self.last_n_perturbed = stats["n_perturbed"]

        return stats

    @torch.no_grad()
    def compute_volatility(self, model, h: torch.Tensor, context: torch.Tensor) -> float:
        """Compute metric volatility: rate of change of g between calls.

        Args:
            model: FluidNetModel (unwrapped)
            h: current hidden states [B, N, d]
            context: context vector [B, d]

        Returns:
            Metric volatility (mean absolute change in g from last call)
        """
        # Compute current g from last layer
        g = model.layers[-1].get_current_metric(h, context)
        g_mean = g.mean(dim=(0, 1))  # [d] average across batch and positions

        if self._prev_g_mean is not None:
            volatility = (g_mean - self._prev_g_mean).abs().mean().item()
        else:
            volatility = 0.0

        self._prev_g_mean = g_mean.clone()
        self.last_volatility = volatility
        return volatility


class DynamicWeightDecay:
    """Adjust weight decay based on EMA-smoothed CE loss velocity.

    When CE drops rapidly (model is learning) → increase decay (push toward compression)
    When CE spikes (topology shifted) → decrease decay (allow exploration)

    Uses EMA smoothing to prevent oscillation between extremes.
    """

    def __init__(self, base_decay: float = 0.1,
                 min_decay: float = 0.01, max_decay: float = 0.3,
                 velocity_window: int = 100,
                 increase_rate: float = 1.02, decrease_rate: float = 0.95,
                 ema_alpha: float = 0.1):
        self.base_decay = base_decay
        self.min_decay = min_decay
        self.max_decay = max_decay
        self.current_decay = base_decay
        self.velocity_window = velocity_window
        self.increase_rate = increase_rate
        self.decrease_rate = decrease_rate
        self.ema_alpha = ema_alpha
        self._loss_history = []
        self._ema_velocity = 0.0

    def update(self, ce_loss: float, optimizer) -> float:
        """Update weight decay based on EMA-smoothed loss velocity.

        Args:
            ce_loss: current CE loss value
            optimizer: AdamW optimizer to update

        Returns:
            New weight decay value
        """
        self._loss_history.append(ce_loss)
        if len(self._loss_history) > self.velocity_window * 2:
            self._loss_history = self._loss_history[-self.velocity_window * 2:]

        if len(self._loss_history) >= self.velocity_window:
            # Compute raw loss velocity (negative = improving)
            recent = self._loss_history[-self.velocity_window // 2:]
            earlier = self._loss_history[-self.velocity_window:-self.velocity_window // 2]
            raw_velocity = sum(recent) / len(recent) - sum(earlier) / len(earlier)

            # EMA smooth the velocity to prevent oscillation
            self._ema_velocity = (self.ema_alpha * raw_velocity +
                                  (1 - self.ema_alpha) * self._ema_velocity)

            if self._ema_velocity < -0.001:
                # Loss dropping → increase decay (push compression)
                self.current_decay = min(
                    self.current_decay * self.increase_rate, self.max_decay)
            elif self._ema_velocity > 0.01:
                # Loss spiking → decrease decay (allow exploration)
                self.current_decay = max(
                    self.current_decay * self.decrease_rate, self.min_decay)

        # Apply to optimizer
        for group in optimizer.param_groups:
            group['weight_decay'] = self.current_decay

        return self.current_decay


if __name__ == "__main__":
    print("Testing MetricMonitor...")

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from fgn.config import FGNConfig
    from fgn.model_fluid import FluidNetModel

    cfg = FGNConfig(
        d_model=64, n_heads=4, d_ff=256, n_layers=4,
        vocab_size=100, max_seq_len=32,
        d_metric=16, d_ffn_fluid=128, n_scales=3,
        architecture_version="fluid",
        geo_metric_type="learned",
    )
    model = FluidNetModel(cfg)

    # Test with relative threshold
    monitor = MetricMonitor(crystal_percentile=0.05, check_every=1)
    stats = monitor.check_and_perturb(model, step=0)
    print(f"  Total dims: {stats.get('n_total_dims', 0)}")
    print(f"  Crystallized: {stats.get('n_crystallized', 0)} "
          f"(should be ~5% of total)")
    print(f"  Perturbed: {stats.get('n_perturbed', 0)}")
    for ls in stats.get("per_layer", []):
        print(f"    Layer {ls['layer']}: {ls['n_crystallized']} crystal, "
              f"health=[{ls['health_min']:.6f}, {ls['health_mean']:.6f}, "
              f"{ls['health_max']:.6f}], threshold={ls['threshold']:.6f}")

    # Test volatility
    x = torch.randint(0, 100, (2, 16))
    with torch.no_grad():
        pos = torch.arange(16).unsqueeze(0)
        h = model.embed(x) + model.pos_embed(pos)
        context = model.context_pool(h, None)
    v = monitor.compute_volatility(model, h, context)
    print(f"  Volatility (first call): {v:.6f}")
    v = monitor.compute_volatility(model, h, context)
    print(f"  Volatility (same input): {v:.6f}")

    # Test EMA dynamic decay
    import torch.optim
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    dwd = DynamicWeightDecay(base_decay=0.1, ema_alpha=0.1)
    for i in range(200):
        loss = 5.0 - i * 0.02
        decay = dwd.update(loss, opt)
    print(f"  Dynamic decay after 200 decreasing steps: {decay:.4f}")
    print(f"  EMA velocity: {dwd._ema_velocity:.6f}")

    print("MetricMonitor OK")
