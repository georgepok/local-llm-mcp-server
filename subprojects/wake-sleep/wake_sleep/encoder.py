"""ConceptEncoder — CNN on raw 2D grids -> z in R^128.

Extracts latent concept z from (input_grid, output_grid) demo pairs.
Information bottleneck: forces compression of transformation rules.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConceptEncoder(nn.Module):
    """Extract latent concept z in R^z_dim from (input_grid, output_grid) demo pairs.

    CNN processes stacked input/output grids as 2D spatial objects.
    Mean-pool z across K demo pairs per task -> z_task [B, z_dim].
    """

    def __init__(self, z_dim: int = 128, d_enc: int = 32, n_colors: int = 11):
        super().__init__()
        self.z_dim = z_dim
        self.color_embed = nn.Embedding(n_colors, d_enc)  # 11 colors (10 + PAD)
        # CNN on stacked [input, output]: [B, 2*d_enc, H, W]
        self.conv1 = nn.Conv2d(2 * d_enc, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)  # -> [B, 128, 1, 1]
        self.proj = nn.Sequential(
            nn.Linear(128, z_dim),
            nn.LayerNorm(z_dim),
        )

    def encode_pair(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> torch.Tensor:
        """Encode one (input, output) pair -> z [B, z_dim].

        Args:
            input_grid: [B, H, W] int tensor (colors 0-9, pad=10)
            output_grid: [B, H, W] int tensor

        Returns:
            z: [B, z_dim]
        """
        # Embed: [B, H, W] -> [B, H, W, d_enc] -> permute to [B, d_enc, H, W]
        in_emb = self.color_embed(input_grid).permute(0, 3, 1, 2)
        out_emb = self.color_embed(output_grid).permute(0, 3, 1, 2)
        x = torch.cat([in_emb, out_emb], dim=1)  # [B, 2*d_enc, H, W]
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = self.pool(x).flatten(1)  # [B, 128]
        return self.proj(x)            # [B, z_dim]

    def forward(self, demo_pairs: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """Encode K demo pairs and aggregate -> z_task [B, z_dim].

        Args:
            demo_pairs: list of (input_grid [B, H, W], output_grid [B, H, W]) tuples.
                        Grids should be padded to max(H) x max(W) within batch with PAD_COLOR=10.

        Returns:
            z_task: [B, z_dim] mean-pooled concept vector
        """
        zs = [self.encode_pair(inp, out) for inp, out in demo_pairs]
        return torch.stack(zs).mean(dim=0)  # mean-pool across demos


if __name__ == "__main__":
    print("Testing ConceptEncoder...")
    enc = ConceptEncoder(z_dim=128, d_enc=32)
    n_params = sum(p.numel() for p in enc.parameters())
    print(f"  Params: {n_params:,}")

    B = 4
    # 2 demo pairs, different grid sizes padded to 5x5
    pairs = [
        (torch.randint(0, 10, (B, 5, 5)), torch.randint(0, 10, (B, 5, 5))),
        (torch.randint(0, 10, (B, 5, 5)), torch.randint(0, 10, (B, 5, 5))),
    ]

    z = enc(pairs)
    assert z.shape == (B, 128), f"Got {z.shape}"

    # Gradient flow
    z.sum().backward()
    has_grad = sum(1 for p in enc.parameters() if p.grad is not None)
    print(f"  Gradients: {has_grad}/{sum(1 for _ in enc.parameters())}")

    # z diversity: different inputs should give different z
    pairs2 = [
        (torch.randint(0, 10, (B, 5, 5)), torch.randint(0, 10, (B, 5, 5))),
        (torch.randint(0, 10, (B, 5, 5)), torch.randint(0, 10, (B, 5, 5))),
    ]
    enc.zero_grad()
    z2 = enc(pairs2)
    diff = (z.detach() - z2.detach()).norm(dim=-1).mean()
    print(f"  z diff between different inputs: {diff:.4f}")

    print("ConceptEncoder OK")
