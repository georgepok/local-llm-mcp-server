"""GeometricCoupling — projects between LiquidARC's ODE space and an LLM's representation space.

LiquidARC provides persistent curved-space dynamics (d=768).
The LLM provides stateless knowledge lookup (d varies by model).
The coupling is geometric — learned projections, no tokenization, no vocabulary.

Supported LLMs (d_qwen parameter):
  - Qwen3-4B: d=2560
  - Nemotron-3-Nano-30B-A3B: d=2688
  - Qwen3.5-9B: d=4096

LiquidARC's h(t) → n virtual prefix tokens in the LLM's embedding space.
The LLM processes input with these prefix tokens as additional context.
"""

import torch
import torch.nn as nn


class GeometricCoupling(nn.Module):
    """Projects LiquidARC's ODE state into Qwen3's representation space
    and reads Qwen3's output back into LiquidARC's space.

    The ENTIRE interface between the two systems.
    Vector in, vector out. No tokenization.
    """

    def __init__(self, d_arc: int = 768, d_qwen: int = 2560,
                 n_virtual_tokens: int = 8):
        super().__init__()
        self.d_arc = d_arc
        self.d_qwen = d_qwen
        self.n_virtual_tokens = n_virtual_tokens

        # Project LiquidARC state → n virtual token embeddings
        # h(t) ∈ ℝ^768 → n × ℝ^2560
        self.W_inject = nn.Linear(d_arc, d_qwen * n_virtual_tokens)

        # Project Qwen3 output at prefix positions → LiquidARC space
        # n × ℝ^2560 → ℝ^768
        self.W_read = nn.Linear(d_qwen * n_virtual_tokens, d_arc)

        # Small init — don't disrupt either model at start
        nn.init.normal_(self.W_inject.weight, std=0.01)
        nn.init.zeros_(self.W_inject.bias)
        nn.init.normal_(self.W_read.weight, std=0.01)
        nn.init.zeros_(self.W_read.bias)

    def inject(self, h_arc: torch.Tensor) -> torch.Tensor:
        """Project LiquidARC state to virtual prefix token embeddings.

        Args:
            h_arc: LiquidARC's pooled ODE state [d_arc] or [1, d_arc]

        Returns:
            prefix_embeds: [1, n_virtual_tokens, d_qwen]
        """
        if h_arc.dim() == 1:
            h_arc = h_arc.unsqueeze(0)
        # [1, d_arc] → [1, n_vt * d_qwen]
        projected = self.W_inject(h_arc)
        # Reshape to n virtual tokens
        return projected.view(1, self.n_virtual_tokens, self.d_qwen)

    def read(self, qwen_prefix_output: torch.Tensor) -> torch.Tensor:
        """Project Qwen3's output at prefix positions back to LiquidARC space.

        Args:
            qwen_prefix_output: [1, n_virtual_tokens, d_qwen]

        Returns:
            arc_signal: [d_arc] — sensory forcing signal for LiquidARC
        """
        flat = qwen_prefix_output.view(1, -1)  # [1, n_vt * d_qwen]
        return self.W_read(flat).squeeze(0)  # [d_arc]

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
