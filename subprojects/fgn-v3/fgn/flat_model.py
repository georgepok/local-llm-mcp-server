"""Flat Transformer Baseline — standard attention without Riemannian geometry.

This is a parameter-matched baseline (~31M params at d=256, 6 layers, 8 heads)
that uses vanilla dot-product attention instead of heat kernel attention on
a Riemannian manifold. NO MetricNetwork, NO geodesic distances, NO scale mixing.

Architecture matches FGNModel parameter budget for fair comparison.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FGNConfig


class FlatAttention(nn.Module):
    """Standard multi-head dot-product attention with causal masking."""

    def __init__(self, config: FGNConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_model = config.d_model

        # Q, K, V, O projections (same structure as HeatKernelAttention)
        self.W_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_o = nn.Linear(config.d_model, config.d_model, bias=False)

        # Attention dropout
        self.attn_drop = nn.Dropout(config.dropout)

        # Scaling factor for attention scores
        self.scale = config.d_head ** -0.5

    def forward(self, h: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Standard scaled dot-product attention.

        Args:
            h: [B, N, d_model] hidden states
            mask: [N, N] causal mask (True = masked positions)

        Returns:
            attn_output [B, N, d_model]
        """
        B, N, _ = h.shape

        # Project Q, K, V -> [B, H, N, d_head]
        Q = self.W_q(h).view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        K = self.W_k(h).view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        V = self.W_v(h).view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        # Scaled dot-product attention: softmax(QK^T / sqrt(d_k))
        scores = (Q @ K.transpose(-2, -1)) * self.scale  # [B, H, N, N]

        # Apply causal mask
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        # Apply attention to values
        attn_out = attn_weights @ V  # [B, H, N, d_head]

        # Reshape and project
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, N, self.d_model)
        return self.W_o(attn_out)


class FlatTransformerLayer(nn.Module):
    """Standard pre-norm transformer layer with dot-product attention."""

    def __init__(self, config: FGNConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # Pre-norm layers (only 2, no metric norm)
        self.norm_attn = nn.LayerNorm(config.d_model)
        self.norm_ff = nn.LayerNorm(config.d_model)

        # Standard attention (no geometry)
        self.attention = FlatAttention(config)

        # FFN with dropout (same as FGNTransformerLayer)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
        )

        # Residual dropout
        self.resid_drop = nn.Dropout(config.dropout)

    def forward(self, h: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            h: [B, N, d_model]
            mask: [N, N] causal mask

        Returns:
            h [B, N, d_model]
        """
        # Attention + residual dropout
        attn_out = self.attention(self.norm_attn(h), mask=mask)
        h = h + self.resid_drop(attn_out)

        # FFN + residual dropout
        h = h + self.resid_drop(self.ffn(self.norm_ff(h)))

        return h


class FlatTransformerModel(nn.Module):
    """Full flat transformer language model (parameter-matched baseline)."""

    def __init__(self, config: FGNConfig):
        super().__init__()
        self.config = config

        # Embeddings (same as FGNModel)
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)

        # Transformer layers (flat, no geometry)
        self.layers = nn.ModuleList([
            FlatTransformerLayer(config, i) for i in range(config.n_layers)
        ])

        # Output (same as FGNModel)
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights (same as FGNModel)."""
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
            Dict with 'logits', and optionally 'loss', 'ce_loss'.
            Also includes zero-valued 'curv_loss', 'scale_loss', 'metric_cv',
            'avg_kappa' for compatibility with FGN training script logging.
        """
        B, N = input_ids.shape
        device = input_ids.device

        # Embeddings
        pos = torch.arange(N, device=device).unsqueeze(0)
        h = self.embed(input_ids) + self.pos_embed(pos)

        # Causal mask: True = masked (future positions)
        mask = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)

        # Forward through layers
        for layer in self.layers:
            h = layer(h, mask=mask)

        # LM head
        h = self.norm(h)
        logits = self.lm_head(h)

        result = {"logits": logits}

        # Zero-valued compatibility stats (for training script logging)
        result["metric_cv"] = torch.tensor(0.0, device=device)
        result["avg_kappa"] = torch.tensor(0.0, device=device)

        if labels is not None:
            # Cross-entropy loss
            ce_loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )
            result["ce_loss"] = ce_loss

            # Zero-valued regularization losses (for compatibility)
            result["curv_loss"] = torch.tensor(0.0, device=device)
            result["scale_loss"] = torch.tensor(0.0, device=device)

            # Total loss (only CE for flat baseline)
            result["loss"] = ce_loss

        return result

    def slow_parameters(self) -> List[nn.Parameter]:
        """Parameters with reduced learning rate (none for flat baseline)."""
        return []

    def fast_parameters(self) -> List[nn.Parameter]:
        """Parameters with base learning rate (all parameters)."""
        return list(self.parameters())


if __name__ == "__main__":
    # Smoke test with small config
    cfg = FGNConfig(d_model=64, n_heads=4, d_ff=256, n_layers=2,
                    vocab_size=100, max_seq_len=32)
    model = FlatTransformerModel(cfg)

    B, N = 2, 16
    input_ids = torch.randint(0, 100, (B, N))
    labels = torch.randint(0, 100, (B, N))

    # Forward pass with loss
    result = model(input_ids, labels=labels)
    assert "loss" in result
    assert "logits" in result
    assert result["logits"].shape == (B, N, 100)
    assert "ce_loss" in result
    assert "curv_loss" in result
    assert "scale_loss" in result
    assert "metric_cv" in result
    assert "avg_kappa" in result
    print(f"loss={result['loss'].item():.4f}, ce={result['ce_loss'].item():.4f}, "
          f"curv={result['curv_loss'].item():.4f}, scale={result['scale_loss'].item():.4f}")

    # Gradient flow test
    result["loss"].backward()
    for i, layer in enumerate(model.layers):
        for name, p in layer.named_parameters():
            if p.requires_grad:
                assert p.grad is not None and p.grad.abs().sum() > 0, \
                    f"No grad for layer {i}.{name}"
    print("Gradient flow: OK")

    # Parameter groups
    slow = model.slow_parameters()
    fast = model.fast_parameters()
    total = sum(p.numel() for p in model.parameters())
    slow_n = sum(p.numel() for p in slow)
    fast_n = sum(p.numel() for p in fast)
    assert slow_n + fast_n == total, f"Parameter split mismatch: {slow_n}+{fast_n} != {total}"
    assert slow_n == 0, f"Flat baseline should have 0 slow params, got {slow_n}"
    print(f"Parameters: {total:,} total ({slow_n:,} slow, {fast_n:,} fast)")

    # Compare parameter count with FGN at same config
    from .model import FGNModel
    fgn_model = FGNModel(cfg)
    fgn_total = sum(p.numel() for p in fgn_model.parameters())
    print(f"FGN parameters: {fgn_total:,}")
    print(f"Flat parameters: {total:,}")
    print(f"Difference: {fgn_total - total:,} ({100*(fgn_total-total)/fgn_total:.1f}% more in FGN)")

    print("\nFlatTransformerModel OK")
