"""ARC embedding — additive cell-as-token representation.

h = ColorEmbed(color) + PosX(x) + PosY(y) + RoleEmbed(role)
    + SepEmbed(sep_type) * is_sep + GridIdEmbed(grid_id)
    → LayerNorm → Dropout

v2: Removed seq_pos_embed (262K params, 31% of model at d=256). The 1D sequential
position injected harmful bias — ARC grids are 2D, and the model already has
(pos_x, pos_y) for spatial structure. Replaced with small grid_id_embed (4K params)
that tells the model which grid a token belongs to, using grid_ids from the ARC task.
"""

import torch
import torch.nn as nn

from .config import LiquidARCConfig


class ARCEmbedding(nn.Module):
    """Additive embedding for ARC cell-as-token representation.

    Role embedding shared between test_input and test_output (role 3 → 2)
    so the model learns spatial reasoning that transfers from input to output.
    """

    def __init__(self, config: LiquidARCConfig):
        super().__init__()
        d = config.d_model

        self.color_embed = nn.Embedding(config.n_colors + 1, d)  # +1 for PAD_COLOR
        self.pos_x_embed = nn.Embedding(config.max_grid_size + 1, d)  # +1 for PAD_COORD
        self.pos_y_embed = nn.Embedding(config.max_grid_size + 1, d)
        self.role_embed = nn.Embedding(4, d)  # input_demo/output_demo/test_input/test_output
        self.sep_embed = nn.Embedding(config.n_sep_types, d)
        self.grid_id_embed = nn.Embedding(config.max_grids, d)  # which grid a token belongs to

        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, colors, xs, ys, roles, sep_mask, sep_types, grid_ids=None):
        """
        Args:
            colors: [B, N] color indices (0-10)
            xs: [B, N] x coordinates (0-30)
            ys: [B, N] y coordinates (0-30)
            roles: [B, N] role indices (0-3)
            sep_mask: [B, N] bool — True for separator tokens
            sep_types: [B, N] separator type indices (0-3)
            grid_ids: [B, N] optional grid identifiers (-1 for separators)

        Returns:
            h: [B, N, d] embedded hidden states
        """
        # Share role embedding: test_output (3) → test_input (2)
        roles_shared = roles.clone()
        roles_shared[roles == 3] = 2

        h = (self.color_embed(colors) +
             self.pos_x_embed(xs) +
             self.pos_y_embed(ys) +
             self.role_embed(roles_shared))

        # Add separator embedding only at separator positions
        sep_h = self.sep_embed(sep_types)
        h = h + sep_h * sep_mask.unsqueeze(-1).float()

        # Grid identity embedding (replaces seq_pos_embed)
        if grid_ids is not None:
            # Separators have grid_id=-1; tasks with many demos can exceed max_grids
            gids = grid_ids.clamp(min=0, max=self.grid_id_embed.num_embeddings - 1)
            h = h + self.grid_id_embed(gids)

        return self.dropout(self.norm(h))


if __name__ == "__main__":
    print("Testing ARCEmbedding...")
    config = LiquidARCConfig(d_model=64, max_seq_len=128)
    emb = ARCEmbedding(config)

    B, N = 2, 32
    colors = torch.randint(0, 10, (B, N))
    xs = torch.randint(0, 10, (B, N))
    ys = torch.randint(0, 10, (B, N))
    roles = torch.randint(0, 4, (B, N))
    sep_mask = torch.zeros(B, N, dtype=torch.bool)
    sep_mask[:, [7, 15, 23]] = True
    sep_types = torch.zeros(B, N, dtype=torch.long)
    grid_ids = torch.zeros(B, N, dtype=torch.long)
    grid_ids[:, :8] = 0
    grid_ids[:, 7] = -1  # separator
    grid_ids[:, 8:16] = 1
    grid_ids[:, 15] = -1

    h = emb(colors, xs, ys, roles, sep_mask, sep_types, grid_ids=grid_ids)
    assert h.shape == (B, N, 64), f"Got {h.shape}"

    # Without grid_ids (backwards compat)
    h2 = emb(colors, xs, ys, roles, sep_mask, sep_types)
    assert h2.shape == (B, N, 64)

    # Gradient flow
    h.sum().backward()
    print(f"  Output shape: {h.shape}")
    print(f"  Params: {sum(p.numel() for p in emb.parameters()):,}")
    print("ARCEmbedding OK")
