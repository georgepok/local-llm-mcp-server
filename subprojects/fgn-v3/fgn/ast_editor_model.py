"""AST editor model: LiquidARC ODE substrate with three independent emit
heads (pointer, op, payload) on edit-slot positions.

Differs from LiquidSequenceModel in two ways:
  1. Output side has three independent linear heads instead of one LM head
  2. Loss is computed at designated [SLOT] positions only, summing three CE
     losses per slot — no AR teacher-forcing easy path

Architecture mirrors LiquidSequenceModel for K=1 vs K=2 dispatch and uses the
same ContinuousDynamics / MultiSubstrateDynamics modules + same Euler solver.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Imports mirror LiquidSequenceModel — keep paths consistent so this file can
# live alongside it in the fgn package.
import sys
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_LIQUID_ARC = os.path.normpath(os.path.join(_HERE, "..", "..", "..",
                                              "liquid-arc"))
if _LIQUID_ARC not in sys.path:
    sys.path.insert(0, _LIQUID_ARC)

from liquid_arc.config import LiquidARCConfig
from liquid_arc.dynamics import ContinuousDynamics
from liquid_arc.multi_substrate import MultiSubstrateDynamics
from liquid_arc.solver import euler_solve

from .config import FGNConfig


class ASTEditorModel(nn.Module):
    """LiquidARC ODE + 3-head emit at edit-slot positions.

    Input layout (matches synthetic_ast.py single-slot encoding):
        BOS src(N_NODES) SEP_TGT tgt(N_NODES) SEP_EDIT [SLOT_1..SLOT_K_FIX]

    The K_FIX [SLOT] positions each carry a special token in input_ids; after
    embedding + ODE, three heads read the hidden state at those positions and
    emit (pointer, op, payload). All three predictions for one edit come from
    a SINGLE hidden vector — that's the (where, what) decomposition test.
    """

    def __init__(self, config: FGNConfig,
                 n_nodes: int, n_edit_ops: int, payload_range: int,
                 script_start: int, k_fix: int):
        super().__init__()
        self.config = config
        self.n_nodes = n_nodes
        self.n_edit_ops = n_edit_ops
        self.payload_range = payload_range
        self.script_start = script_start
        self.k_fix = k_fix
        d = config.d_model

        la_cfg = LiquidARCConfig(
            d_model=d,
            d_metric=getattr(config, 'd_liquid_metric', 0) or int(d * 0.75),
            d_ffn=getattr(config, 'd_liquid_ffn', 0) or 4 * d,
            max_seq_len=config.max_seq_len,
            n_ode_steps=getattr(config, 'n_ode_steps', 8),
            ode_steps_min=getattr(config, 'n_ode_steps', 8),
            ode_steps_max=getattr(config, 'n_ode_steps', 8),
            integration_time=2.0,
            tau_min=getattr(config, 'liquid_tau_min', 0.5),
            tau_max=getattr(config, 'liquid_tau_max', 1.0),
            t_diffusion_init=1.0,
            routing_mode=getattr(config, 'liquid_routing', 'metric'),
            dropout=config.dropout,
            alpha_logit_init=2.2,
            tau_freeze_steps=0,
            use_torch_compile=False,
            chunk_size=256,
            metric_rank=getattr(config, 'metric_rank', 0),
            structural_tau_enabled=False,
            tau_quality_loss_enabled=False,
            criticality_loss_enabled=False,
            halting_enabled=False,
            rezero_enabled=False,
        )
        self.la_cfg = la_cfg
        self.n_ode_steps = la_cfg.n_ode_steps

        self.embed = nn.Embedding(config.vocab_size, d)
        self.pos_embed = nn.Embedding(config.max_seq_len, d)

        self.K_substrates = int(getattr(config, 'k_substrates', 1))
        self.lateral_weight = float(getattr(config, 'lateral_weight', 0.5))
        if self.K_substrates > 1:
            self.dynamics = MultiSubstrateDynamics(
                la_cfg, K=self.K_substrates,
                lateral_weight=self.lateral_weight)
            self._fuse_proj = nn.Linear(self.K_substrates * d, d, bias=False)
        else:
            self.dynamics = ContinuousDynamics(la_cfg)
            self._fuse_proj = None

        self.norm = nn.LayerNorm(d)
        # Three independent heads — the (where, what) decomposition lives here
        self.ptr_head = nn.Linear(d, n_nodes)
        self.op_head = nn.Linear(d, n_edit_ops)
        self.pay_head = nn.Linear(d, payload_range)

    def _pool_context(self, h: torch.Tensor) -> torch.Tensor:
        return h.mean(dim=1)

    def forward(self, input_ids: torch.Tensor,
                gt_ptr: Optional[torch.Tensor] = None,
                gt_op: Optional[torch.Tensor] = None,
                gt_pay: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """input_ids: [B, SEQ_LEN] of token ids (slot tokens at known positions)
        gt_ptr/gt_op/gt_pay: [B, K_FIX] integer labels for each slot (training)
        Returns dict with 'ptr_logits', 'op_logits', 'pay_logits' (all
        [B, K_FIX, *]) and optionally 'loss' if GT supplied.
        """
        B, N = input_ids.shape
        device = input_ids.device
        pos = torch.arange(N, device=device).unsqueeze(0)
        h0 = self.embed(input_ids) + self.pos_embed(pos)
        mask = torch.triu(
            torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)
        context = self._pool_context(h0)
        self.dynamics.set_context(context, mask=mask)
        self.dynamics.set_n_steps(self.n_ode_steps)

        if self.K_substrates > 1:
            h0_evolved = h0.repeat(1, 1, self.K_substrates)
        else:
            h0_evolved = h0
        h = euler_solve(self.dynamics, h0_evolved,
                         t_span=(0.0, self.la_cfg.integration_time),
                         n_steps=self.n_ode_steps)
        if isinstance(h, tuple):
            h = h[0]
        if self.K_substrates > 1:
            h = self._fuse_proj(h)
        h = self.norm(h)

        # Extract hidden vectors at the K_FIX slot positions
        slot_idx = torch.arange(self.script_start,
                                  self.script_start + self.k_fix,
                                  device=device)
        h_slots = h[:, slot_idx, :]  # [B, K_FIX, d]

        ptr_logits = self.ptr_head(h_slots)  # [B, K_FIX, N_NODES]
        op_logits = self.op_head(h_slots)    # [B, K_FIX, N_EDIT_OPS]
        pay_logits = self.pay_head(h_slots)  # [B, K_FIX, PAY_RANGE]

        result: Dict[str, torch.Tensor] = {
            "ptr_logits": ptr_logits,
            "op_logits": op_logits,
            "pay_logits": pay_logits,
        }

        if gt_ptr is not None and gt_op is not None and gt_pay is not None:
            ce_ptr = F.cross_entropy(
                ptr_logits.reshape(-1, self.n_nodes),
                gt_ptr.reshape(-1), ignore_index=-100)
            ce_op = F.cross_entropy(
                op_logits.reshape(-1, self.n_edit_ops),
                gt_op.reshape(-1), ignore_index=-100)
            ce_pay = F.cross_entropy(
                pay_logits.reshape(-1, self.payload_range),
                gt_pay.reshape(-1), ignore_index=-100)
            result["loss"] = ce_ptr + ce_op + ce_pay
            result["ce_ptr"] = ce_ptr.detach()
            result["ce_op"] = ce_op.detach()
            result["ce_pay"] = ce_pay.detach()
        return result
