"""FGNv6Model — Budget-based attention escalation with shared context pooling."""
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import FGNConfig
from .context_pool import ContextPool
from .layer_v6 import FGNv6Layer
from .losses import CurvatureRegularization


class FGNv6Model(nn.Module):
    """
    FGNv6Model — Hierarchical Geometric Routing with Budget-based Escalation.

    Key differences from v5:
    - Shared ContextPool computed once from initial embeddings
    - Fixed per-layer attention budgets (no learned escalation)
    - No sharpness annealing
    - No escalation penalty in loss
    """

    def __init__(self, config: FGNConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)

        # Context pooling (shared, computed once from embeddings)
        # Only needed when geo_metric_type == "learned"
        if config.geo_metric_type == "learned":
            self.context_pool = ContextPool(config)

        # Layers with per-layer budgets
        budgets = config.attention_budgets  # tuple of floats, one per layer
        assert len(budgets) == config.n_layers, \
            f"attention_budgets length {len(budgets)} != n_layers {config.n_layers}"

        self.layers = nn.ModuleList([
            FGNv6Layer(config, layer_idx=i, budget=budgets[i])
            for i in range(config.n_layers)
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

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with small values for stability."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through FGNv6 model.

        Args:
            input_ids: [B, N] token indices
            labels: [B, N] target tokens for training (optional)
            context_mask: [B, N] boolean mask marking context tokens (optional)

        Returns:
            Dictionary containing:
                - logits: [B, N, vocab_size]
                - loss: total loss (if labels provided)
                - ce_loss: cross-entropy loss
                - curv_loss: curvature regularization loss
                - esc_penalty: 0.0 (for compatibility)
                - metric_cv: average metric coefficient of variation
                - avg_kappa: average absolute curvature
                - escalation_rate: average escalation rate across layers
                - avg_entropy: average routing entropy
                - esc_rates_per_layer: list of per-layer escalation rates
                - entropies_per_layer: list of per-layer entropies
                - scale_loss: 0.0 (for compatibility with v4 scripts)
                - avg_gate: 0.0 (for compatibility with v4 scripts)
        """
        _, N = input_ids.shape
        device = input_ids.device

        # Compute embeddings
        pos = torch.arange(N, device=device).unsqueeze(0)
        h = self.embed(input_ids) + self.pos_embed(pos)

        # Compute shared context (once, from initial embeddings)
        context = None
        if hasattr(self, 'context_pool'):
            context = self.context_pool(h, context_mask)  # [B, d]

        # Causal mask
        mask = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)

        # Forward through layers
        curvatures = []
        metric_cvs = []
        escalation_rates = []
        entropies = []

        for layer in self.layers:
            h, kappa, m_cv, avg_ent, esc_rate = layer(h, mask=mask, context=context)
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

        # Per-layer stats
        result["esc_rates_per_layer"] = escalation_rates
        result["entropies_per_layer"] = entropies

        # Compatibility with v4 training scripts
        result["scale_loss"] = torch.tensor(0.0, device=device)
        result["avg_gate"] = torch.tensor(0.0, device=device)

        if labels is not None:
            # Cross-entropy loss
            ce_loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )
            result["ce_loss"] = ce_loss

            # Curvature regularization (if enabled)
            curv_loss = self.curv_reg(curvatures)
            result["curv_loss"] = curv_loss

            # No escalation penalty in v6 (budgets are fixed)
            result["esc_penalty"] = torch.tensor(0.0, device=device)

            result["loss"] = ce_loss + curv_loss

        return result

    def geo_parameters(self) -> List[nn.Parameter]:
        """Parameters for geometric pathway (MetricNetwork, GeoRoute, ContextPool)."""
        params = []
        if hasattr(self, 'context_pool'):
            params.extend(self.context_pool.parameters())
        for layer in self.layers:
            if hasattr(layer, 'metric'):
                params.extend(layer.metric.parameters())
            params.extend(layer.geo_route.parameters())
        params.extend(self.curv_reg.parameters())
        return params

    def attn_parameters(self) -> List[nn.Parameter]:
        """Parameters for attention pathway (StandardAttention)."""
        params = []
        for layer in self.layers:
            if hasattr(layer, 'attention'):
                params.extend(layer.attention.parameters())
        return params

    def other_parameters(self) -> List[nn.Parameter]:
        """Parameters not in geo or attn groups (embeddings, FFN, norms, lm_head)."""
        geo_ids = {id(p) for p in self.geo_parameters()}
        attn_ids = {id(p) for p in self.attn_parameters()}
        return [p for p in self.parameters()
                if id(p) not in geo_ids and id(p) not in attn_ids]


if __name__ == "__main__":
    print("Testing FGNv6Model...")

    # Test 1: Learned metric with context pooling
    print("\n=== Test 1: Learned metric with context pooling ===")
    cfg = FGNConfig(
        d_model=64,
        n_heads=4,
        d_ff=256,
        n_layers=6,
        vocab_size=100,
        max_seq_len=32,
        geo_heads=4,
        architecture_version="v6",
        geo_metric_type="learned",
        curvature_lambda=0.01,
        curvature_reward_mu=0.0,
        attention_budgets=(0.0, 0.05, 0.1, 0.1, 0.2, 0.3)
    )
    model = FGNv6Model(cfg)

    B, N = 2, 16
    input_ids = torch.randint(0, 100, (B, N))
    labels = torch.randint(0, 100, (B, N))
    context_mask = torch.zeros(B, N, dtype=torch.bool)
    context_mask[:, :4] = True  # First 4 tokens are context

    # Forward pass
    result = model(input_ids, labels=labels, context_mask=context_mask)

    # Verify output keys
    expected_keys = {
        "logits", "loss", "ce_loss", "curv_loss", "esc_penalty",
        "metric_cv", "avg_kappa", "escalation_rate", "avg_entropy",
        "esc_rates_per_layer", "entropies_per_layer",
        "scale_loss", "avg_gate"
    }
    assert set(result.keys()) == expected_keys, f"Missing keys: {expected_keys - set(result.keys())}"

    # Verify shapes
    assert result["logits"].shape == (B, N, cfg.vocab_size), \
        f"Wrong logits shape: {result['logits'].shape}"
    assert len(result["esc_rates_per_layer"]) == cfg.n_layers, \
        f"Wrong number of per-layer esc_rates: {len(result['esc_rates_per_layer'])}"
    assert len(result["entropies_per_layer"]) == cfg.n_layers, \
        f"Wrong number of per-layer entropies: {len(result['entropies_per_layer'])}"

    # Verify losses are scalars
    assert result["loss"].ndim == 0, "loss should be scalar"
    assert result["ce_loss"].ndim == 0, "ce_loss should be scalar"
    assert result["curv_loss"].ndim == 0, "curv_loss should be scalar"

    # Verify escalation penalty is zero
    assert result["esc_penalty"].item() == 0.0, "esc_penalty should be 0.0 in v6"

    # Verify gradient flow
    result["loss"].backward()
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for _ in model.parameters())
    print(f"Parameters with gradients: {has_grad}/{total_params}")
    assert has_grad > 0, "No gradients computed"

    # Verify parameter groups are disjoint
    geo_params = set(id(p) for p in model.geo_parameters())
    attn_params = set(id(p) for p in model.attn_parameters())
    other_params = set(id(p) for p in model.other_parameters())
    all_params = set(id(p) for p in model.parameters())

    assert len(geo_params & attn_params) == 0, "geo and attn params overlap"
    assert len(geo_params & other_params) == 0, "geo and other params overlap"
    assert len(attn_params & other_params) == 0, "attn and other params overlap"
    assert geo_params | attn_params | other_params == all_params, "param groups don't cover all params"

    print(f"Parameter groups: geo={len(geo_params)}, attn={len(attn_params)}, other={len(other_params)}")

    # Print stats
    print(f"Loss: {result['loss'].item():.4f}")
    print(f"CE Loss: {result['ce_loss'].item():.4f}")
    print(f"Curv Loss: {result['curv_loss'].item():.4f}")
    print(f"Metric CV: {result['metric_cv'].item():.4f}")
    print(f"Avg Kappa: {result['avg_kappa'].item():.4f}")
    print(f"Escalation Rate: {result['escalation_rate'].item():.4f}")
    print(f"Avg Entropy: {result['avg_entropy'].item():.4f}")
    print(f"Per-layer escalation rates: {[f'{r:.3f}' for r in result['esc_rates_per_layer']]}")
    print(f"Per-layer entropies: {[f'{e:.3f}' for e in result['entropies_per_layer']]}")

    # Test 2: Flat metric (no context pooling)
    print("\n=== Test 2: Flat metric (no context pooling) ===")
    cfg_flat = FGNConfig(
        d_model=64,
        n_heads=4,
        d_ff=256,
        n_layers=6,
        vocab_size=100,
        max_seq_len=32,
        geo_heads=4,
        architecture_version="v6",
        geo_metric_type="flat",
        curvature_lambda=0.0,
        curvature_reward_mu=0.0,
        attention_budgets=(0.0, 0.05, 0.1, 0.1, 0.2, 0.3)
    )
    model_flat = FGNv6Model(cfg_flat)

    # Verify no context_pool
    assert not hasattr(model_flat, 'context_pool'), "Flat metric should not have context_pool"

    # Forward pass without context_mask
    result_flat = model_flat(input_ids, labels=labels)

    # Verify flat metric has cv=0.0
    assert result_flat["metric_cv"].item() == 0.0, \
        f"Flat metric should have cv=0.0, got {result_flat['metric_cv'].item()}"

    print(f"Flat metric CV: {result_flat['metric_cv'].item():.4f} (should be 0.0)")
    print(f"Flat metric avg_kappa: {result_flat['avg_kappa'].item():.4f} (should be 0.0)")

    # Test 3: Budget validation
    print("\n=== Test 3: Budget validation ===")
    try:
        bad_cfg = FGNConfig(
            d_model=64,
            n_heads=4,
            d_ff=256,
            n_layers=6,
            vocab_size=100,
            max_seq_len=32,
            geo_heads=4,
            architecture_version="v6",
            geo_metric_type="learned",
            attention_budgets=(0.0, 0.1, 0.2)  # Wrong length
        )
        bad_model = FGNv6Model(bad_cfg)
        print("ERROR: Should have raised assertion error for wrong budget length")
    except AssertionError as e:
        print(f"Correctly raised error: {e}")

    print("\n=== All tests passed! ===")
