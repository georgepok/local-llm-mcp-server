"""FGNv5Model — Hierarchical Geometric Routing language model.

GeoRoute runs first on every token. Entropy-based escalation triggers
attention only where GeoRoute is uncertain. No blend gate, no Phase 0/1.

Single-phase training with sharpness annealing for soft escalation.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FGNConfig
from .layer_v5 import FGNv5Layer
from .losses import CurvatureRegularization


class FGNv5Model(nn.Module):
    def __init__(self, config: FGNConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            FGNv5Layer(config, i) for i in range(config.n_layers)
        ])

        # Output
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Curvature regularization (disabled by default for v5)
        self.curv_reg = CurvatureRegularization(
            n_layers=config.n_layers,
            curvature_lambda=config.curvature_lambda,
            correlation_length_init=config.correlation_length_init,
            curvature_reward_mu=config.curvature_reward_mu,
        )

        # Escalation penalty weight
        self.esc_penalty_alpha = config.escalation_penalty_alpha

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

        Returns:
            Dict with 'logits', and optionally 'loss', 'ce_loss',
            'curv_loss', 'esc_penalty', 'metric_cv', 'avg_kappa',
            'escalation_rate', 'avg_entropy'
        """
        B, N = input_ids.shape
        device = input_ids.device

        pos = torch.arange(N, device=device).unsqueeze(0)
        h = self.embed(input_ids) + self.pos_embed(pos)

        # Causal mask
        mask = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool),
                          diagonal=1)

        # Forward through layers
        curvatures = []
        metric_cvs = []
        escalation_rates = []
        entropies = []

        for layer in self.layers:
            h, kappa, m_cv, avg_ent, esc_rate = layer(h, mask=mask)
            curvatures.append(kappa)
            metric_cvs.append(m_cv)
            entropies.append(avg_ent)
            escalation_rates.append(esc_rate)

        # LM head
        h = self.norm(h)
        logits = self.lm_head(h)

        result = {"logits": logits}

        # Monitoring stats
        result["metric_cv"] = sum(metric_cvs) / len(metric_cvs)
        result["avg_kappa"] = sum(k.abs().mean() for k in curvatures) / len(curvatures)
        result["escalation_rate"] = sum(escalation_rates) / len(escalation_rates)
        result["avg_entropy"] = sum(entropies) / len(entropies)

        # Per-layer stats for logging
        result["esc_rates_per_layer"] = escalation_rates
        result["entropies_per_layer"] = entropies

        # Compatibility with v4 training script
        result["scale_loss"] = torch.tensor(0.0, device=device)
        result["avg_gate"] = torch.tensor(0.0, device=device)

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

            # Escalation penalty: penalize high escalation rate
            # This rewards the metric for being useful (reducing attention usage)
            avg_esc_rate = sum(escalation_rates) / len(escalation_rates)
            esc_penalty = self.esc_penalty_alpha * avg_esc_rate
            result["esc_penalty"] = esc_penalty

            result["loss"] = ce_loss + curv_loss + esc_penalty

        return result

    # --- Parameter groups ---

    def geo_parameters(self) -> List[nn.Parameter]:
        """Parameters for geometric pathway (MetricNetwork, GeoRoute, threshold)."""
        params = []
        for layer in self.layers:
            if hasattr(layer, 'metric'):
                params.extend(layer.metric.parameters())
            params.extend(layer.geo_route.parameters())
            if hasattr(layer, 'threshold_raw'):
                params.append(layer.threshold_raw)
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

    def set_sharpness(self, sharpness: float):
        """Set escalation sharpness for all layers (soft mode only)."""
        for layer in self.layers:
            if hasattr(layer, 'sharpness'):
                layer.sharpness = sharpness


if __name__ == "__main__":
    cfg = FGNConfig(d_model=64, n_heads=4, d_ff=256, n_layers=2,
                    vocab_size=100, max_seq_len=32, geo_heads=1,
                    architecture_version="v5",
                    curvature_lambda=0.0, curvature_reward_mu=0.0,
                    escalation_mode="soft", escalation_penalty_alpha=0.01)
    model = FGNv5Model(cfg)

    B, N = 2, 16
    input_ids = torch.randint(0, 100, (B, N))
    labels = torch.randint(0, 100, (B, N))

    # Forward pass with loss
    result = model(input_ids, labels=labels)
    assert "loss" in result
    assert "logits" in result
    assert result["logits"].shape == (B, N, 100)
    assert "escalation_rate" in result
    assert "avg_entropy" in result
    print(f"loss={result['loss'].item():.4f}, ce={result['ce_loss'].item():.4f}, "
          f"esc_rate={result['escalation_rate'].item():.4f}, "
          f"avg_entropy={result['avg_entropy'].item():.4f}, "
          f"cv={result['metric_cv'].item():.4f}")

    # Per-layer stats
    esc_rates = result["esc_rates_per_layer"]
    entropies = result["entropies_per_layer"]
    print(f"Per-layer esc_rate: [{', '.join(f'{r.item():.3f}' for r in esc_rates)}]")
    print(f"Per-layer entropy: [{', '.join(f'{e.item():.3f}' for e in entropies)}]")

    # Gradient flow
    result["loss"].backward()
    if hasattr(model.layers[0], 'metric'):
        for i, layer in enumerate(model.layers):
            for name, p in layer.metric.named_parameters():
                assert p.grad is not None and p.grad.abs().sum() > 0, \
                    f"No grad for layer {i} metric.{name}"
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'threshold_raw'):
            assert layer.threshold_raw.grad is not None, \
                f"No grad for layer {i} threshold"
    print("Gradient flow: OK")

    # Sharpness control
    model.set_sharpness(5.0)
    for layer in model.layers:
        assert layer.sharpness == 5.0
    print("Sharpness control: OK")

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

    # Test flat metric mode
    cfg_flat = FGNConfig(d_model=64, n_heads=4, d_ff=256, n_layers=2,
                         vocab_size=100, max_seq_len=32, geo_heads=1,
                         architecture_version="v5", geo_metric_type="flat",
                         escalation_mode="soft")
    model_flat = FGNv5Model(cfg_flat)
    result_flat = model_flat(input_ids, labels=labels)
    assert result_flat["metric_cv"].item() == 0.0
    print(f"Flat metric: cv={result_flat['metric_cv'].item():.4f}, "
          f"esc_rate={result_flat['escalation_rate'].item():.4f}")
    print("FGNv5Model OK")
