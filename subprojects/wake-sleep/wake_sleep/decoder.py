"""DreamDecoder — OutputPredictor: z + input_grid -> output_grid.

Conditional CNN that predicts output grid given concept z and input grid.
Used in Wake phase (learn from real ARC) and Sleep phase (generate dreams).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OutputPredictor(nn.Module):
    """Predict output grid given concept z and input grid.

    Conditional CNN: embed input, condition on z, predict output colors.
    """

    def __init__(self, z_dim: int = 128, d_dec: int = 64, n_colors: int = 11):
        super().__init__()
        self.color_embed = nn.Embedding(n_colors, d_dec)
        self.z_proj = nn.Linear(z_dim, d_dec)
        # Input conv: [d_dec + d_dec] = 2*d_dec channels -> d_dec
        self.input_conv = nn.Conv2d(2 * d_dec, d_dec, 3, padding=1)
        # 3 residual Conv2d blocks
        self.res_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d_dec, d_dec, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(d_dec, d_dec, 3, padding=1),
            ) for _ in range(3)
        ])
        self.head = nn.Conv2d(d_dec, n_colors, 1)  # 1x1 conv -> per-cell logits

    def forward(self, z: torch.Tensor, input_grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, z_dim] concept vector
            input_grid: [B, H, W] int tensor (colors 0-9)

        Returns:
            logits: [B, n_colors, H, W]
        """
        B, H, W = input_grid.shape
        # Embed input: [B, H, W, d_dec] -> [B, d_dec, H, W]
        x = self.color_embed(input_grid).permute(0, 3, 1, 2)
        # Broadcast z: [B, d_dec] -> [B, d_dec, H, W]
        z_feat = self.z_proj(z).unsqueeze(-1).unsqueeze(-1).expand(B, -1, H, W)
        # Concat and initial conv
        x = F.gelu(self.input_conv(torch.cat([x, z_feat], dim=1)))
        # Residual blocks
        for res in self.res_convs:
            x = x + res(x)
        return self.head(x)  # [B, n_colors, H, W]


class DreamDecoder(nn.Module):
    """Wrapper holding OutputPredictor."""

    def __init__(self, z_dim: int = 128, d_dec: int = 64, n_colors: int = 11):
        super().__init__()
        self.output_pred = OutputPredictor(z_dim, d_dec, n_colors)

    def predict(self, z: torch.Tensor, input_grid: torch.Tensor) -> torch.Tensor:
        """Generate output grid logits from z + input."""
        return self.output_pred(z, input_grid)

    def dream(self, z: torch.Tensor, input_grid: torch.Tensor) -> torch.Tensor:
        """Generate hard output grid (argmax) for Sleep phase."""
        logits = self.output_pred(z, input_grid)
        return logits.argmax(dim=1)  # [B, H, W]


if __name__ == "__main__":
    print("Testing DreamDecoder...")
    dec = DreamDecoder(z_dim=128, d_dec=64)
    n_params = sum(p.numel() for p in dec.parameters())
    print(f"  Params: {n_params:,}")

    B = 4
    z = torch.randn(B, 128)
    grid = torch.randint(0, 10, (B, 5, 7))

    logits = dec.predict(z, grid)
    assert logits.shape == (B, 11, 5, 7), f"Got {logits.shape}"

    dream = dec.dream(z, grid)
    assert dream.shape == (B, 5, 7), f"Got {dream.shape}"
    assert dream.min() >= 0 and dream.max() <= 10

    # Loss decreases over 50 optim steps on one task
    target = torch.randint(0, 10, (B, 5, 7))
    opt = torch.optim.Adam(dec.parameters(), lr=1e-3)
    loss0 = None
    for step in range(50):
        opt.zero_grad()
        logits = dec.predict(z, grid)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        opt.step()
        if step == 0:
            loss0 = loss.item()
    print(f"  Loss: {loss0:.4f} -> {loss.item():.4f} (50 steps)")
    assert loss.item() < loss0, "Loss did not decrease"

    print("DreamDecoder OK")
