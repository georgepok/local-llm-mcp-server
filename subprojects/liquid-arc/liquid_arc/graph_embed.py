"""Graph node embedding — additive categorical representation for LiquidARC ODE.

Directly mirrors ARC embedding structure (liquid_arc/embedding.py):

    ARC:   ColorEmbed(color) + PosX(x) + PosY(y) + RoleEmbed(role)
    Graph: TypeEmbed(type) + RoleEmbed(role) + StructProj(struct_features)

The categorical components (type_embed, role_embed) create the cluster
structure the MetricNet trained on ARC already knows how to route on:
same type → same TypeEmbed component → low D² → heat kernel connects.

See GRAPH_REASONING_ENGINE_SPEC.md Phase 1 for the architectural rationale.
"""

import torch
import torch.nn as nn


class GraphNodeEmbedding(nn.Module):
    """Encode graph nodes for LiquidARC ODE processing.

    Spec signature (GRAPH_REASONING_ENGINE_SPEC.md lines 52-81):
        __init__(d_model, n_node_types=32, n_edge_types=16, n_roles=8)
        forward(node_types, roles, struct_features) -> h [B, N, d_model]
    """

    def __init__(self, d_model: int, n_node_types: int = 32,
                 n_edge_types: int = 16, n_roles: int = 8):
        super().__init__()
        self.d_model = d_model
        self.type_embed = nn.Embedding(n_node_types, d_model)       # like ColorEmbed
        self.role_embed = nn.Embedding(n_roles, d_model)            # like RoleEmbed
        self.struct_proj = nn.Linear(16, d_model)                   # structural features
        self.edge_type_embed = nn.Embedding(n_edge_types, d_model)  # relation encoding
        self.norm = nn.LayerNorm(d_model)

    def forward(self, node_types: torch.Tensor, roles: torch.Tensor,
                struct_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_types:      [B, N] long — node type indices in [0, n_node_types)
            roles:           [B, N] long — role indices in [0, n_roles)
            struct_features: [B, N, 16] float — per-node topology features

        Returns:
            h: [B, N, d_model] — node embeddings with categorical cluster structure
        """
        h = (self.type_embed(node_types)
             + self.role_embed(roles)
             + self.struct_proj(struct_features))
        return self.norm(h)


if __name__ == "__main__":
    print("Testing GraphNodeEmbedding...")
    emb = GraphNodeEmbedding(d_model=768)
    B, N = 2, 10
    node_types = torch.randint(0, 32, (B, N))
    roles = torch.randint(0, 8, (B, N))
    struct_features = torch.randn(B, N, 16)
    h = emb(node_types, roles, struct_features)
    assert h.shape == (B, N, 768), f"expected ({B},{N},768), got {h.shape}"
    h.sum().backward()
    print(f"  output shape: {h.shape}")
    print(f"  params: {sum(p.numel() for p in emb.parameters()):,}")
    print("GraphNodeEmbedding OK")
