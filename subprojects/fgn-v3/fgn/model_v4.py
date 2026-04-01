"""FGNv4Model — Geometry-First Architecture language model.

Dual-pathway architecture: GeoRoute (metric-based) + StandardAttention (dot-product),
connected by a learned gate per layer.

Supports Phase 0 (geometric pre-training) and Phase 1 (joint training) modes.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FGNConfig
from .layer_v4 import FGNv4Layer
from .losses import CurvatureRegularization


class FGNv4Model(nn.Module):
    def __init__(self, config: FGNConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            FGNv4Layer(config, i) for i in range(config.n_layers)
        ])

        # Output
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Curvature regularization (disabled by default for v4)
        self.curv_reg = CurvatureRegularization(
            n_layers=config.n_layers,
            curvature_lambda=config.curvature_lambda,
            correlation_length_init=config.correlation_length_init,
            curvature_reward_mu=config.curvature_reward_mu,
        )

        # Geometric auxiliary loss weight
        self.geo_aux_alpha = config.geo_aux_loss_alpha

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
            'curv_loss', 'metric_cv', 'avg_kappa', 'avg_gate'
        """
        B, N = input_ids.shape
        device = input_ids.device

        pos = torch.arange(N, device=device).unsqueeze(0)
        h = self.embed(input_ids) + self.pos_embed(pos)

        # Causal mask
        mask = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)

        # Forward through layers
        curvatures = []
        metric_cvs = []
        gate_values = []

        for layer in self.layers:
            h, kappa, m_cv, gate_val = layer(h, mask=mask)
            curvatures.append(kappa)
            metric_cvs.append(m_cv)
            gate_values.append(gate_val)

        # LM head
        h = self.norm(h)
        logits = self.lm_head(h)

        result = {"logits": logits}

        # Monitoring stats
        result["metric_cv"] = sum(metric_cvs) / len(metric_cvs)
        result["avg_kappa"] = sum(k.abs().mean() for k in curvatures) / len(curvatures)
        # avg_gate as tensor (avoid .item() in compiled path)
        result["avg_gate"] = sum(gate_values) / len(gate_values)

        # Compatibility with v3 training script logging
        result["scale_loss"] = torch.tensor(0.0, device=device)

        if labels is not None:
            ce_loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )
            result["ce_loss"] = ce_loss

            # Curvature regularization
            curv_loss = self.curv_reg(curvatures)
            result["curv_loss"] = curv_loss

            # Geometric auxiliary loss: -alpha * mean(CV across layers)
            geo_aux_loss = torch.tensor(0.0, device=device)
            if self.geo_aux_alpha > 0:
                avg_cv = sum(metric_cvs) / len(metric_cvs)
                geo_aux_loss = -self.geo_aux_alpha * avg_cv
            result["geo_aux_loss"] = geo_aux_loss

            result["loss"] = ce_loss + curv_loss + geo_aux_loss

        return result

    # --- Phase 0 / Phase 1 training mode switches ---

    def freeze_attention(self):
        """Phase 0: freeze StandardAttention parameters."""
        for layer in self.layers:
            for p in layer.attention.parameters():
                p.requires_grad = False

    def unfreeze_attention(self):
        """Phase 1: unfreeze StandardAttention parameters."""
        for layer in self.layers:
            for p in layer.attention.parameters():
                p.requires_grad = True

    def force_gate(self, value: float = 10.0):
        """Phase 0: force gate to near-1.0 and freeze it."""
        for layer in self.layers:
            layer.gate_geo_raw.data.fill_(value)
            layer.gate_geo_raw.requires_grad = False

    def init_gate(self, value: float = 3.0):
        """Phase 1: reset gate to initial value and unfreeze."""
        for layer in self.layers:
            layer.gate_geo_raw.data.fill_(value)
            layer.gate_geo_raw.requires_grad = True

    # --- Parameter groups ---

    def geo_parameters(self) -> List[nn.Parameter]:
        """Parameters for geometric pathway (MetricNetwork, GeoRoute, gate, log_t)."""
        params = []
        for layer in self.layers:
            if hasattr(layer, 'metric'):
                params.extend(layer.metric.parameters())
            params.extend(layer.geo_route.parameters())
            params.append(layer.gate_geo_raw)
        params.extend(self.curv_reg.parameters())
        return params

    def attn_parameters(self) -> List[nn.Parameter]:
        """Parameters for attention pathway (StandardAttention)."""
        params = []
        for layer in self.layers:
            params.extend(layer.attention.parameters())
        return params

    def other_parameters(self) -> List[nn.Parameter]:
        """Parameters not in geo or attn groups (embeddings, FFN, norms, lm_head)."""
        geo_ids = {id(p) for p in self.geo_parameters()}
        attn_ids = {id(p) for p in self.attn_parameters()}
        return [p for p in self.parameters()
                if id(p) not in geo_ids and id(p) not in attn_ids]

    def slow_parameters(self) -> List[nn.Parameter]:
        """Compatibility with v3 training script. Returns geo parameters."""
        return self.geo_parameters()

    def fast_parameters(self) -> List[nn.Parameter]:
        """Compatibility with v3 training script. Returns non-geo parameters."""
        slow_ids = {id(p) for p in self.slow_parameters()}
        return [p for p in self.parameters() if id(p) not in slow_ids]


if __name__ == "__main__":
    cfg = FGNConfig(d_model=64, n_heads=4, d_ff=256, n_layers=2,
                    vocab_size=100, max_seq_len=32, geo_heads=1,
                    architecture_version="v4", gate_init=3.0,
                    curvature_lambda=0.0, curvature_reward_mu=0.0)
    model = FGNv4Model(cfg)

    B, N = 2, 16
    input_ids = torch.randint(0, 100, (B, N))
    labels = torch.randint(0, 100, (B, N))

    # Forward pass with loss
    result = model(input_ids, labels=labels)
    assert "loss" in result
    assert "logits" in result
    assert result["logits"].shape == (B, N, 100)
    assert "avg_gate" in result
    print(f"loss={result['loss'].item():.4f}, ce={result['ce_loss'].item():.4f}, "
          f"gate={result['avg_gate']:.4f}, cv={result['metric_cv'].item():.4f}")

    # Gradient flow
    result["loss"].backward()
    for i, layer in enumerate(model.layers):
        for name, p in layer.metric.named_parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0, \
                f"No grad for layer {i} metric.{name}"
        assert layer.gate_geo_raw.grad is not None, f"No grad for layer {i} gate"
    print("Gradient flow: OK")

    # Phase 0 mode
    model.freeze_attention()
    model.force_gate(10.0)
    for layer in model.layers:
        assert not layer.gate_geo_raw.requires_grad
        for p in layer.attention.parameters():
            assert not p.requires_grad
    print("Phase 0 freeze: OK")

    # Phase 1 mode
    model.unfreeze_attention()
    model.init_gate(3.0)
    for layer in model.layers:
        assert layer.gate_geo_raw.requires_grad
        for p in layer.attention.parameters():
            assert p.requires_grad
    print("Phase 1 unfreeze: OK")

    # Parameter groups
    geo = model.geo_parameters()
    attn = model.attn_parameters()
    other = model.other_parameters()
    total = sum(p.numel() for p in model.parameters())
    geo_n = sum(p.numel() for p in geo)
    attn_n = sum(p.numel() for p in attn)
    other_n = sum(p.numel() for p in other)
    assert geo_n + attn_n + other_n == total, \
        f"Param split mismatch: {geo_n}+{attn_n}+{other_n} != {total}"
    print(f"Parameters: {total:,} total (geo={geo_n:,}, attn={attn_n:,}, other={other_n:,})")

    print("FGNv4Model OK")
