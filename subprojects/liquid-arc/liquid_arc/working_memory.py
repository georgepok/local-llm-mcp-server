"""WorkingMemory V4 — Observe During ODE, Correct at Output.

The memory NEVER modifies the base model's internal processing.
It passively observes h at every ODE step, accumulates into slots,
then produces a logit correction AFTER the ODE completes.

This is structurally immune to copy bias because:
1. The dynamics runs identically to the frozen base (no h residual, no overlay)
2. The correction targets logits, not hidden states
3. With xform-only loss, the correction only gets gradient from transform cells
4. Mean-pooled observations lose position-specific input color information

Parameter count (~42K):
    write_proj:      256 × 64 + 64 = 16,448
    write_gate:      256 × 8 + 8   = 2,056
    step_embed:      20 × 64       = 1,280
    read_query:      256 × 64 + 64 = 16,448
    correction_head: 64×64+64 + 64×10+10 = 4,810
    slot_init:       8 × 64        = 512
    Total:           ~41,554
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class WorkingMemory(nn.Module):
    """Observe-only memory with output correction.

    Args:
        d_model: Hidden dimension of base model.
        n_slots: Number of memory slots.
        d_memory: Slot embedding dimension.
        n_colors: Number of output classes (10 for ARC).
    """

    def __init__(self, d_model: int, n_slots: int = 8, d_memory: int = 64,
                 n_colors: int = 10):
        super().__init__()
        self.d_model = d_model
        self.n_slots = n_slots
        self.d_memory = d_memory

        # Learned initial slot state
        self.slot_init = nn.Parameter(torch.zeros(n_slots, d_memory))

        # Write path: observe h → update slots
        self.write_proj = nn.Linear(d_model, d_memory, bias=True)
        self.write_gate = nn.Linear(d_model, n_slots, bias=True)

        # Step embedding: tells memory WHICH ODE step it's observing
        self.step_embed = nn.Embedding(20, d_memory)

        # Output correction: slots → per-position logit correction
        self.read_query = nn.Linear(d_model, d_memory, bias=True)
        self.correction_head = nn.Sequential(
            nn.Linear(d_memory, d_memory),
            nn.GELU(),
            nn.Linear(d_memory, n_colors),
        )
        # Zero-init final layer → starts as no-op
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)

        self._slots: Optional[torch.Tensor] = None
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.write_proj.weight, std=0.02)
        nn.init.zeros_(self.write_proj.bias)
        nn.init.normal_(self.write_gate.weight, std=0.02)
        nn.init.zeros_(self.write_gate.bias)
        nn.init.normal_(self.read_query.weight, std=0.02)
        nn.init.zeros_(self.read_query.bias)
        nn.init.normal_(self.correction_head[0].weight, std=0.02)
        nn.init.zeros_(self.correction_head[0].bias)

    def reset(self, batch_size: int, device: torch.device) -> None:
        """Initialize slot state for a new sequence."""
        self._slots = self.slot_init.unsqueeze(0).expand(batch_size, -1, -1).clone()

    def observe(self, h_detached: torch.Tensor, step_index: int) -> None:
        """Passively observe hidden state. NO modification to h.

        Accumulates information into slots, tagged with step embedding
        so the memory knows WHEN it observed each pattern.

        Args:
            h_detached: [B, N, d_model] — detached, no grad flows back
            step_index: which ODE step (0-16)
        """
        if self._slots is None:
            raise RuntimeError("WorkingMemory.reset() must be called before observe()")

        h_summary = h_detached.mean(dim=1)  # [B, d_model]

        # Project + add step context
        step_idx = min(step_index, 19)
        step_ctx = self.step_embed(
            torch.tensor(step_idx, device=h_detached.device))  # [d_memory]
        write_val = self.write_proj(h_summary) + step_ctx  # [B, d_memory]

        # Soft gate over slots
        write_weights = F.softmax(self.write_gate(h_summary), dim=-1)  # [B, n_slots]
        write_update = write_weights.unsqueeze(-1) * write_val.unsqueeze(1)

        # EMA update (detached — observation is non-differentiable)
        self._slots = (0.9 * self._slots + 0.1 * write_update).detach()

    def get_output_correction(self, h_final_detached: torch.Tensor) -> torch.Tensor:
        """Compute logit corrections from accumulated observations.

        Called ONCE after ODE completes. This is the ONLY place
        the memory affects predictions. Gradients flow through this
        to the read_query and correction_head parameters.

        Args:
            h_final_detached: [B, N, d_model] — final hidden state (detached)

        Returns:
            correction: [B, N, n_colors] — additive logit correction
        """
        if self._slots is None:
            raise RuntimeError("No observations recorded")

        # Per-position query into accumulated memory
        query = self.read_query(h_final_detached)  # [B, N, d_memory]

        # Attend over slots
        scores = torch.matmul(
            query, self._slots.transpose(1, 2)) / (self.d_memory ** 0.5)
        attn = F.softmax(scores, dim=-1)  # [B, N, n_slots]

        # Read content
        read_val = torch.matmul(attn, self._slots)  # [B, N, d_memory]

        # Project to logit corrections
        return self.correction_head(read_val)  # [B, N, n_colors]

    def get_diagnostics(self) -> Dict[str, torch.Tensor]:
        """Return diagnostic tensors for logging."""
        device = self.slot_init.device
        if self._slots is None:
            return {
                "mem_slot_norm": torch.tensor(0.0, device=device),
                "mem_slot_var": torch.tensor(0.0, device=device),
            }
        return {
            "mem_slot_norm": self._slots.norm(dim=-1).mean().detach(),
            "mem_slot_var": self._slots.var(dim=1).mean().detach(),
        }
