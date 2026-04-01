"""WakeSleepModel — orchestrates Encoder + Decoder + ODE with z projection.

Does NOT own the base ODE model — takes it as a reference.
Owns the z_to_context projection (trained in both Wake and Sleep).
"""

import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import from liquid-arc
_LIQUID_ARC_ROOT = str(Path(__file__).resolve().parent.parent.parent / "liquid-arc")
if _LIQUID_ARC_ROOT not in sys.path:
    sys.path.insert(0, _LIQUID_ARC_ROOT)

from liquid_arc.model import LiquidARCModel
from liquid_arc.tasks.procedural import build_sequence, PAD_COLOR, PAD_COORD

from .config import WakeSleepConfig
from .encoder import ConceptEncoder
from .decoder import DreamDecoder
from .concept_bank import ConceptBank
from .model import forward_with_external_context


def collate_sequences(seqs: list, device: torch.device, max_seq_len: int) -> dict:
    """Collate list of build_sequence() outputs into padded batch tensors.

    Same format as ProceduralARCTask.generate_batch() metadata dict.
    """
    batch_size = len(seqs)
    max_N = max_seq_len

    colors = torch.full((batch_size, max_N), PAD_COLOR, dtype=torch.long, device=device)
    xs_t = torch.full((batch_size, max_N), PAD_COORD, dtype=torch.long, device=device)
    ys_t = torch.full((batch_size, max_N), PAD_COORD, dtype=torch.long, device=device)
    roles = torch.zeros(batch_size, max_N, dtype=torch.long, device=device)
    sep_mask = torch.ones(batch_size, max_N, dtype=torch.bool, device=device)
    sep_types = torch.zeros(batch_size, max_N, dtype=torch.long, device=device)
    grid_ids = torch.full((batch_size, max_N), -1, dtype=torch.long, device=device)
    target_mask = torch.zeros(batch_size, max_N, dtype=torch.bool, device=device)
    target_labels = torch.full((batch_size, max_N), -100, dtype=torch.long, device=device)
    target_input_colors = torch.full((batch_size, max_N), PAD_COLOR, dtype=torch.long, device=device)
    context_mask = torch.ones(batch_size, max_N, dtype=torch.bool, device=device)
    lengths = torch.zeros(batch_size, dtype=torch.long, device=device)

    for i, s in enumerate(seqs):
        N = s["length"]
        if N > max_N:
            N = max_N  # truncate if too long
        lengths[i] = N
        colors[i, :N] = torch.tensor(s["colors"][:N], dtype=torch.long)
        xs_t[i, :N] = torch.tensor(s["xs"][:N], dtype=torch.long)
        ys_t[i, :N] = torch.tensor(s["ys"][:N], dtype=torch.long)
        roles[i, :N] = torch.tensor(s["roles"][:N], dtype=torch.long)
        sep_mask[i, :N] = torch.tensor(s["sep_mask"][:N], dtype=torch.bool)
        sep_types[i, :N] = torch.tensor(s["sep_types"][:N], dtype=torch.long)
        grid_ids[i, :N] = torch.tensor(s["grid_ids"][:N], dtype=torch.long)
        target_mask[i, :N] = torch.tensor(s["target_mask"][:N], dtype=torch.bool)
        target_input_colors[i, :N] = torch.tensor(s["target_input_colors"][:N], dtype=torch.long)

        tgt_positions = [j for j, m in enumerate(s["target_mask"][:N]) if m]
        for j, pos in enumerate(tgt_positions):
            if j < len(s["target_colors"]):
                target_labels[i, pos] = s["target_colors"][j]

        context_mask[i, :N] = ~target_mask[i, :N]

    return {
        "colors": colors,
        "xs": xs_t,
        "ys": ys_t,
        "roles": roles,
        "sep_mask": sep_mask,
        "sep_types": sep_types,
        "grid_ids": grid_ids,
        "target_mask": target_mask,
        "target_labels": target_labels,
        "target_input_colors": target_input_colors,
        "context_mask": context_mask,
        "lengths": lengths,
    }


class WakeSleepModel(nn.Module):
    """Orchestrates Encoder + Decoder + ODE with z projection.

    Does NOT own the base ODE model — takes it as a reference.
    Owns the z_to_context projection (trained in both Wake and Sleep).
    """

    def __init__(self, config: WakeSleepConfig, base_model: LiquidARCModel):
        super().__init__()
        self.encoder = ConceptEncoder(
            z_dim=config.ws_z_dim, d_enc=config.ws_d_enc)
        self.decoder = DreamDecoder(
            z_dim=config.ws_z_dim, d_dec=config.ws_d_dec)
        self.z_to_context = nn.Sequential(
            nn.Linear(config.ws_z_dim, config.d_model),
            nn.LayerNorm(config.d_model),
        )
        self.concept_bank = ConceptBank(config.ws_concept_bank_size, config.ws_z_dim)
        self.base_model = base_model  # reference, not copy
        self.config = config

    def wake_step(
        self,
        demo_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Wake: Encoder+Decoder learn from real ARC. ODE frozen.

        Args:
            demo_pairs: list of (input_grid [B, H, W], output_grid [B, H, W]) tuples

        Returns:
            dict with wake_loss, z_norm
        """
        # 1. Encode demos -> z_task [B, z_dim]
        z = self.encoder(demo_pairs)

        # 2. Reconstruct each demo's output from z + input
        total_loss = torch.tensor(0.0, device=device)
        for inp, out in demo_pairs:
            logits = self.decoder.predict(z, inp)  # [B, n_colors, H, W]
            loss = F.cross_entropy(logits, out)     # out is [B, H, W] longs
            total_loss = total_loss + loss
        total_loss = total_loss / len(demo_pairs)

        # 3. Store z in concept bank
        self.concept_bank.add(z)

        return {
            "wake_loss": total_loss,
            "z_norm": z.norm(dim=-1).mean().detach(),
        }

    def sleep_step(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Sleep: ODE trains on Decoder-generated dreams. Encoder+Decoder frozen.

        Returns:
            dict with ODE result (loss, ce_loss, metric_cv, avg_kappa, etc.)
        """
        config = self.config

        # 1. Hallucinate novel rule via interpolation
        z_dream = self.concept_bank.sample_interpolated(
            batch_size, device,
            alpha_min=config.ws_interp_alpha_min,
            alpha_max=config.ws_interp_alpha_max,
            noise_std=config.ws_z_noise_std,
        )

        # 2. Generate random input grids for demo pair
        H = random.randint(config.ws_dream_grid_min, config.ws_dream_grid_max)
        W = random.randint(config.ws_dream_grid_min, config.ws_dream_grid_max)
        dream_input = torch.randint(0, 10, (batch_size, H, W), device=device)

        # 3. Top-down: frozen Decoder generates target
        #    Clamp to [0, 9] — decoder predicts 11 classes (incl. PAD=10)
        #    but ODE output head only has 10 classes (actual colors).
        with torch.no_grad():
            dream_output = self.decoder.dream(z_dream, dream_input).clamp(0, 9)

        # 4. Generate a second dream pair for the test portion
        H2 = random.randint(config.ws_dream_grid_min, config.ws_dream_grid_max)
        W2 = random.randint(config.ws_dream_grid_min, config.ws_dream_grid_max)
        test_input = torch.randint(0, 10, (batch_size, H2, W2), device=device)
        with torch.no_grad():
            test_output = self.decoder.dream(z_dream, test_input).clamp(0, 9)

        # 5. Serialize dream pairs for ODE (per batch element)
        seqs = []
        for b in range(batch_size):
            demo = (dream_input[b].cpu().tolist(), dream_output[b].cpu().tolist())
            t_in = test_input[b].cpu().tolist()
            t_out = test_output[b].cpu().tolist()
            seqs.append(build_sequence([demo], t_in, t_out))
        batch_tensors = collate_sequences(seqs, device, config.max_seq_len)

        # 6. Project z_dream -> ODE context space
        z_context = self.z_to_context(z_dream)  # [B, d_model]

        # 7. ODE forward with external context
        result = forward_with_external_context(
            self.base_model,
            z_context=z_context,
            colors=batch_tensors["colors"],
            xs=batch_tensors["xs"],
            ys=batch_tensors["ys"],
            roles=batch_tensors["roles"],
            sep_mask=batch_tensors["sep_mask"],
            sep_types=batch_tensors["sep_types"],
            target_mask=batch_tensors["target_mask"],
            target_labels=batch_tensors["target_labels"],
            context_mask=batch_tensors["context_mask"],
            target_input_colors=batch_tensors["target_input_colors"],
            grid_ids=batch_tensors["grid_ids"],
            n_steps=config.n_ode_steps,
        )

        return result

    def wake_parameters(self) -> list:
        """Parameters trained during Wake: encoder + decoder only.

        z_to_context is trained exclusively during Sleep to keep Adam state
        consistent (it bridges encoder z-space and ODE context-space).
        Gradients from Wake still flow through z_to_context when sleep_opt
        updates it, but its optimizer state is maintained in one place.
        """
        params = list(self.encoder.parameters())
        params += list(self.decoder.parameters())
        return params

    def sleep_parameters(self) -> list:
        """Parameters trained during Sleep: ODE + z_to_context."""
        params = list(self.base_model.parameters())
        params += list(self.z_to_context.parameters())
        return params

    def z_proj_parameters(self) -> list:
        """z_to_context projection parameters (for separate optimizer if needed)."""
        return list(self.z_to_context.parameters())
