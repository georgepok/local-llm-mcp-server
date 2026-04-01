"""FGNModel — full Fluid Geometry Network language model."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FGNConfig
from .layer import FGNTransformerLayer
from .losses import CurvatureRegularization


class FGNModel(nn.Module):
    def __init__(self, config: FGNConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            FGNTransformerLayer(config, i) for i in range(config.n_layers)
        ])

        # Output
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Curvature regularization
        self.curv_reg = CurvatureRegularization(
            n_layers=config.n_layers,
            curvature_lambda=config.curvature_lambda,
            correlation_length_init=config.correlation_length_init,
            curvature_reward_mu=config.curvature_reward_mu,
        )

        # Scale entropy weight
        self.scale_entropy_alpha = config.scale_entropy_alpha

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: [B, N] token indices
            labels: [B, N] target token indices (optional)

        Returns:
            Dict with 'logits', and optionally 'loss', 'ce_loss',
            'curv_loss', 'scale_loss', 'curvatures'
        """
        B, N = input_ids.shape
        device = input_ids.device

        # Embeddings
        pos = torch.arange(N, device=device).unsqueeze(0)
        h = self.embed(input_ids) + self.pos_embed(pos)

        # Causal mask: True = masked (future positions)
        mask = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)

        # Forward through layers — collect returned values (compile-safe)
        curvatures = []
        metric_cvs = []
        scale_entropies = []

        for layer in self.layers:
            h, kappa, m_cv, s_ent = layer(h, mask=mask)
            curvatures.append(kappa)
            metric_cvs.append(m_cv)
            scale_entropies.append(s_ent)

        # LM head
        h = self.norm(h)
        logits = self.lm_head(h)

        result = {"logits": logits}

        # Monitoring stats (always computed, cheap tensor ops)
        result["metric_cv"] = sum(metric_cvs) / len(metric_cvs)
        result["avg_kappa"] = sum(k.abs().mean() for k in curvatures) / len(curvatures)

        if labels is not None:
            # Cross-entropy loss
            ce_loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )
            result["ce_loss"] = ce_loss

            # Curvature regularization (uses returned values, not cached)
            curv_loss = self.curv_reg(curvatures)
            result["curv_loss"] = curv_loss
            result["curvatures"] = curvatures

            # Scale entropy regularization (uses returned values, not cached)
            scale_loss = self.scale_entropy_alpha * sum(scale_entropies) / len(self.layers)
            result["scale_loss"] = scale_loss

            # Total loss
            result["loss"] = ce_loss + curv_loss + scale_loss

        return result

    def get_curvatures(self) -> List[torch.Tensor]:
        """Collect curvature tensors from all layers."""
        return [layer.last_curvature for layer in self.layers
                if layer.last_curvature is not None]

    def get_metrics(self) -> List[torch.Tensor]:
        """Collect metric tensors from all layers."""
        return [layer.last_metric for layer in self.layers
                if layer.last_metric is not None]

    def get_scale_weights(self, h: torch.Tensor) -> List[torch.Tensor]:
        """Get per-layer scale selection weights."""
        weights = []
        for layer in self.layers:
            w = F.softmax(layer.attention.W_scale(h), dim=-1)
            weights.append(w)
        return weights

    def slow_parameters(self) -> List[nn.Parameter]:
        """Parameters that use 0.1x learning rate (metric, diffusion times, correlation lengths)."""
        params = []
        for layer in self.layers:
            params.extend(layer.metric.parameters())
            params.append(layer.attention.log_t)
        params.extend(self.curv_reg.parameters())
        return params

    def fast_parameters(self) -> List[nn.Parameter]:
        """Parameters that use base learning rate."""
        slow_ids = {id(p) for p in self.slow_parameters()}
        return [p for p in self.parameters() if id(p) not in slow_ids]


if __name__ == "__main__":
    cfg = FGNConfig(d_model=64, n_heads=4, d_ff=256, n_layers=2,
                    vocab_size=100, max_seq_len=32)
    model = FGNModel(cfg)

    B, N = 2, 16
    input_ids = torch.randint(0, 100, (B, N))
    labels = torch.randint(0, 100, (B, N))

    # Forward pass with loss
    result = model(input_ids, labels=labels)
    assert "loss" in result
    assert "logits" in result
    assert result["logits"].shape == (B, N, 100)
    print(f"loss={result['loss'].item():.4f}, ce={result['ce_loss'].item():.4f}, "
          f"curv={result['curv_loss'].item():.4f}, scale={result['scale_loss'].item():.4f}")

    # Gradient flow test: verify metric gets gradients from all 3 paths
    result["loss"].backward()
    for i, layer in enumerate(model.layers):
        for name, p in layer.metric.named_parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0, \
                f"No grad for layer {i} metric.{name}"
    print("Gradient flow through metric: OK")

    # Parameter groups
    slow = model.slow_parameters()
    fast = model.fast_parameters()
    total = sum(p.numel() for p in model.parameters())
    slow_n = sum(p.numel() for p in slow)
    fast_n = sum(p.numel() for p in fast)
    assert slow_n + fast_n == total, f"Parameter split mismatch: {slow_n}+{fast_n} != {total}"
    print(f"Parameters: {total:,} total ({slow_n:,} slow, {fast_n:,} fast)")

    # Correlation lengths
    lengths = model.curv_reg.correlation_lengths()
    print(f"Correlation lengths: {[f'{l:.2f}' for l in lengths]}")

    print("FGNModel OK")
