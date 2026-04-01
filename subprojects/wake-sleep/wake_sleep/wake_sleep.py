"""WakeSleepModel V2 — VQ-VAE + Autoregressive Transformer + Hybrid Sleep.

V2 changes over V1:
1. VQEncoder replaces ConceptEncoder — discrete codebook forces crisp concepts
2. ARDecoder replaces DreamDecoder — autoregressive generation for exact integer outputs
3. Hybrid Sleep: 50% dreams + 50% real ARC — anchors ODE to real distribution
4. W_o explicitly in melt set for full WHERE+WHEN+WHAT plasticity

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

# Import from fgn-v3 for real ARC sequences
_FGN_ROOT = str(Path(__file__).resolve().parent.parent.parent / "fgn-v3")
if _FGN_ROOT not in sys.path:
    sys.path.insert(0, _FGN_ROOT)

from liquid_arc.model import LiquidARCModel
from liquid_arc.tasks.procedural import build_sequence, PAD_COLOR, PAD_COORD
from fgn.tasks.arc import build_sequence as arc_build_sequence, pad_single_to_batch

from .config import WakeSleepConfig
from .vq_encoder import VQEncoder
from .ar_decoder import ARDecoder
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


def extract_grid_pairs(
    task: dict, device: torch.device
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Extract (input_grid, output_grid) tensor pairs from ARC task dict.

    Args:
        task: ARC task dict with "train" key containing demo pairs.
        device: target device

    Returns:
        List of (input_grid [1, H, W], output_grid [1, H, W]) tuples.
        Grids padded to max H and max W across all demos, with PAD_COLOR=10.
    """
    demos = task["train"]

    max_h = max(max(len(d["input"]), len(d["output"])) for d in demos)
    max_w = max(max(len(d["input"][0]), len(d["output"][0])) for d in demos)

    pairs = []
    for d in demos:
        inp = d["input"]
        out = d["output"]

        inp_t = torch.full((1, max_h, max_w), 10, dtype=torch.long, device=device)
        out_t = torch.full((1, max_h, max_w), 10, dtype=torch.long, device=device)

        for y, row in enumerate(inp):
            for x, c in enumerate(row):
                inp_t[0, y, x] = c
        for y, row in enumerate(out):
            for x, c in enumerate(row):
                out_t[0, y, x] = c

        pairs.append((inp_t, out_t))

    return pairs


class WakeSleepModel(nn.Module):
    """V2: VQ-VAE Encoder + AR Decoder + Hybrid Sleep.

    Does NOT own the base ODE model — takes it as a reference.
    Owns the z_to_context projection (trained in both Wake and Sleep).
    """

    def __init__(self, config: WakeSleepConfig, base_model: LiquidARCModel):
        super().__init__()
        self.encoder = VQEncoder(
            z_dim=config.ws_z_dim,
            d_enc=config.ws_d_enc,
            n_embeddings=config.ws_vq_n_embeddings,
            n_tokens=config.ws_vq_n_tokens,
            beta=config.ws_vq_beta,
            decay=config.ws_vq_decay,
            entropy_weight=config.ws_vq_entropy_weight,
        )
        self.decoder = ARDecoder(
            z_dim=config.ws_z_dim,
            d_ar=config.ws_ar_d_model,
            n_heads=config.ws_ar_n_heads,
            n_layers=config.ws_ar_n_layers,
            n_colors=10,
            dropout=config.ws_ar_dropout,
            n_rule_tokens=config.ws_vq_n_tokens,
        )
        # z_to_context receives mean-pooled z_q: z_q.mean(dim=1) -> [B, z_dim]
        self.z_to_context = nn.Sequential(
            nn.Linear(config.ws_z_dim, config.d_model),
            nn.LayerNorm(config.d_model),
        )
        self.concept_bank = ConceptBank(
            config.ws_concept_bank_size,
            config.ws_z_dim,
            n_tokens=config.ws_vq_n_tokens,
        )
        self.base_model = base_model  # reference, not copy
        self.config = config

    def wake_step(
        self,
        demo_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Wake: single-task wake step (backward compatible). See wake_step_batched."""
        return self.wake_step_batched([demo_pairs], device)

    def wake_step_batched(
        self,
        task_demo_pairs_list: List[List[Tuple[torch.Tensor, torch.Tensor]]],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Wake: VQ-Encoder + AR Decoder learn from N tasks simultaneously.

        Processes N tasks, batches their z_e sequences through VQ together so
        entropy regularization sees N*L assignments per step (not just N).

        Args:
            task_demo_pairs_list: list of N task demo_pairs, each is
                list of (input_grid [1, H, W], output_grid [1, H, W]) tuples

        Returns:
            dict with wake_loss, vq_loss, z_norm, codebook_usage
        """
        N = len(task_demo_pairs_list)

        # 1. Encode each task's demos separately (different grid sizes)
        # encode_pair returns [1, L, z_dim]; mean-pool across demos per task
        z_es = []
        for demo_pairs in task_demo_pairs_list:
            zs = [self.encoder.encode_pair(inp, out) for inp, out in demo_pairs]
            z_e = torch.stack(zs).mean(dim=0)  # [1, L, z_dim] mean-pool across demos
            z_es.append(z_e)
        z_e_batch = torch.cat(z_es, dim=0)  # [N, L, z_dim]

        B, L, z_dim = z_e_batch.shape

        # 2. Flatten spatial tokens for VQ: [N*L, z_dim] — entropy regularization
        #    now sees N*L assignments per batch (much richer signal than N alone)
        z_e_flat = z_e_batch.reshape(B * L, z_dim)
        z_q_flat, vq_loss, indices_flat = self.encoder.vq(z_e_flat)

        # Reshape back to sequence format
        z_q = z_q_flat.reshape(B, L, z_dim)     # [N, L, z_dim]
        indices = indices_flat.reshape(B, L)    # [N, L]

        # 3. AR Decoder: reconstruct each task's outputs from its z_q sequence
        total_recon_loss = torch.tensor(0.0, device=device)
        n_pairs = 0
        for i, demo_pairs in enumerate(task_demo_pairs_list):
            z_q_i = z_q[i:i+1]  # [1, L, z_dim]
            for inp, out in demo_pairs:
                logits = self.decoder(z_q_i, inp, out)  # [1, H*W, 10]
                target_flat = out.clamp(0, 9).reshape(1, -1)  # [1, H*W]
                loss = F.cross_entropy(
                    logits.reshape(-1, 10), target_flat.reshape(-1)
                )
                total_recon_loss = total_recon_loss + loss
                n_pairs += 1
        total_recon_loss = total_recon_loss / max(n_pairs, 1)

        # 4. Total wake loss = reconstruction + VQ (commitment + entropy)
        wake_loss = total_recon_loss + vq_loss

        # 5. Store z_q sequences in concept bank — add handles [N, L, z_dim]
        self.concept_bank.add(z_q)

        return {
            "wake_loss": wake_loss,
            "recon_loss": total_recon_loss.detach(),
            "vq_loss": vq_loss.detach(),
            "z_norm": z_q.norm(dim=-1).mean().detach(),
            "codebook_usage": self.encoder.vq.codebook_usage(),
        }

    def sleep_step(
        self,
        batch_size: int,
        device: torch.device,
        arc_tasks: list = None,
    ) -> Dict[str, torch.Tensor]:
        """Sleep: ODE trains on dreams (50%) + real ARC (50%). Encoder+Decoder frozen.

        Args:
            batch_size: number of sequences
            device: torch device
            arc_tasks: list of ARC task dicts for real ARC mixing (required for V2)

        Returns:
            dict with ODE result (loss, ce_loss, metric_cv, avg_kappa, etc.)
            plus 'sleep_source' key ('dream' or 'real')
        """
        config = self.config
        use_real = (
            arc_tasks is not None
            and len(arc_tasks) > 0
            and random.random() < config.ws_real_arc_mix_ratio
        )

        if use_real:
            return self._sleep_step_real(batch_size, device, arc_tasks)
        else:
            return self._sleep_step_dream(batch_size, device)

    def _sleep_step_dream(
        self, batch_size: int, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Sleep with dream-generated data."""
        config = self.config

        # 1. Hallucinate novel rule via interpolation
        # sample_interpolated returns [batch_size, L, z_dim]
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

        # 3. AR Decoder generates crisp integer targets — z_dream is [B, L, z_dim]
        with torch.no_grad():
            dream_output = self.decoder.dream(z_dream, dream_input, temperature=0.0)

        # 4. Generate a second dream pair for the test portion
        H2 = random.randint(config.ws_dream_grid_min, config.ws_dream_grid_max)
        W2 = random.randint(config.ws_dream_grid_min, config.ws_dream_grid_max)
        test_input = torch.randint(0, 10, (batch_size, H2, W2), device=device)
        with torch.no_grad():
            test_output = self.decoder.dream(z_dream, test_input, temperature=0.0)

        # 5. Serialize dream pairs for ODE (per batch element)
        seqs = []
        for b in range(batch_size):
            demo = (dream_input[b].cpu().tolist(), dream_output[b].cpu().tolist())
            t_in = test_input[b].cpu().tolist()
            t_out = test_output[b].cpu().tolist()
            seqs.append(build_sequence([demo], t_in, t_out))
        batch_tensors = collate_sequences(seqs, device, config.max_seq_len)

        # 6. Project mean-pooled z_dream -> ODE context space
        # z_dream: [B, L, z_dim] -> mean over L -> [B, z_dim] -> z_to_context -> [B, d_model]
        z_context = self.z_to_context(z_dream.mean(dim=1))

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
        result["sleep_source"] = "dream"
        return result

    def _sleep_step_real(
        self, batch_size: int, device: torch.device, arc_tasks: list
    ) -> Dict[str, torch.Tensor]:
        """Sleep with real ARC data — anchors ODE to real distribution.

        Uses pad_single_to_batch from fgn-v3 (returns tensor-based dict)
        then stacks across batch elements.
        """
        config = self.config

        # Build batch from random ARC tasks
        padded_metas = []
        z_contexts = []
        for _ in range(batch_size):
            task = random.choice(arc_tasks)

            # Build ODE sequence from real ARC task
            d4_idx = random.randint(0, 7)
            test_idx = 0
            if len(task.get("test", [])) > 1:
                test_idx = random.randint(0, len(task["test"]) - 1)

            seq = arc_build_sequence(
                task, d4_idx=d4_idx, test_idx=test_idx,
                max_seq_len=config.max_seq_len,
            )
            if seq is None:
                seq = arc_build_sequence(
                    task, d4_idx=0, test_idx=0,
                    max_seq_len=config.max_seq_len,
                )
            if seq is None:
                continue

            # pad_single_to_batch returns [1, max_seq_len] tensors on device
            meta = pad_single_to_batch(seq, config.max_seq_len, device)
            padded_metas.append(meta)

            # Encode task's demos for z_context
            demo_pairs = extract_grid_pairs(task, device)
            with torch.no_grad():
                _z_e, z_q, _vq_loss, _indices = self.encoder(demo_pairs)
            z_contexts.append(z_q)  # [1, L, z_dim]

        if len(padded_metas) == 0:
            return self._sleep_step_dream(batch_size, device)

        # Stack all padded sequences into a batch
        actual_B = len(padded_metas)

        def _stack(key):
            tensors = [m[key] for m in padded_metas if key in m and m[key] is not None]
            if not tensors:
                return None
            return torch.cat(tensors, dim=0)

        # z_contexts contains [1, L, z_dim] tensors; cat -> [actual_B, L, z_dim]
        z_q_batch = torch.cat(z_contexts, dim=0)  # [actual_B, L, z_dim]
        # Mean-pool spatial tokens before projecting to ODE context space
        z_context = self.z_to_context(z_q_batch.mean(dim=1))  # [actual_B, d_model]

        result = forward_with_external_context(
            self.base_model,
            z_context=z_context,
            colors=_stack("colors"),
            xs=_stack("xs"),
            ys=_stack("ys"),
            roles=_stack("roles"),
            sep_mask=_stack("sep_mask"),
            sep_types=_stack("sep_types"),
            target_mask=_stack("target_mask"),
            target_labels=_stack("target_labels"),
            context_mask=_stack("context_mask"),
            target_input_colors=_stack("target_input_colors"),
            grid_ids=_stack("grid_ids"),
            n_steps=config.n_ode_steps,
        )
        result["sleep_source"] = "real"
        return result

    def wake_parameters(self) -> list:
        """Parameters trained during Wake: encoder + decoder only.

        z_to_context is trained exclusively during Sleep to keep Adam state
        consistent (it bridges encoder z-space and ODE context-space).
        """
        params = list(self.encoder.parameters())
        params += list(self.decoder.parameters())
        return params

    def sleep_parameters(self) -> list:
        """Parameters trained during Sleep: ODE + z_to_context.

        W_o is already part of base_model.parameters() — no extra code needed.
        The full WHERE+WHEN+WHAT plasticity comes from unfreezing all of base_model.
        """
        params = list(self.base_model.parameters())
        params += list(self.z_to_context.parameters())
        return params

    def z_proj_parameters(self) -> list:
        """z_to_context projection parameters (for separate optimizer if needed)."""
        return list(self.z_to_context.parameters())
