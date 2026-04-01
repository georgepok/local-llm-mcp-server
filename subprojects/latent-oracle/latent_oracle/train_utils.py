"""Training utilities — OracleArcDataset, forward_with_oracle, helpers.

OracleArcDataset indexes precomputed oracle embeddings by (task_id, d4_idx)
and pairs them with ARC task sequences built via fgn-v3's build_sequence().

forward_with_oracle wraps the LiquidARC forward pass, replacing ContextPool
with oracle-projected context and adding kappa distillation loss.
"""

import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# Import from sibling subprojects
_LIQUID_ARC_ROOT = str(Path(__file__).resolve().parent.parent.parent / "liquid-arc")
if _LIQUID_ARC_ROOT not in sys.path:
    sys.path.insert(0, _LIQUID_ARC_ROOT)

_FGN_ROOT = str(Path(__file__).resolve().parent.parent.parent / "fgn-v3")
if _FGN_ROOT not in sys.path:
    sys.path.insert(0, _FGN_ROOT)

from liquid_arc.model import LiquidARCModel, PAD_COLOR
from liquid_arc.solver import euler_solve, deq_solve, invertible_euler_solve
from fgn.tasks.arc import load_arc_tasks, build_sequence, pad_single_to_batch


class OracleArcDataset:
    """Pairs precomputed oracle embeddings with ARC task sequences.

    Loads embeddings.pt and ARC task JSON files. For each sample, looks up
    the precomputed embedding by (task_id, d4_idx) and builds the ARC sequence
    using fgn-v3's build_sequence().

    Optionally loads precomputed similarity matrices for representation
    distillation (oracle → heat kernel geometry).
    """

    def __init__(
        self,
        embeddings_path: str,
        data_dir: str,
        max_seq_len: int = 2048,
        similarity_path: str = "",
    ):
        # Load precomputed oracle embeddings
        emb_data = torch.load(embeddings_path, map_location="cpu", weights_only=True)
        self.embeddings = emb_data["embeddings"]        # [N, oracle_dim]
        self.task_ids = emb_data["task_ids"]             # list[str]
        self.d4_indices = emb_data["d4_indices"]         # [N]
        self.test_indices = emb_data["test_indices"]     # [N]
        self.splits = emb_data["splits"]                 # list[str]
        self.oracle_dim = emb_data["oracle_dim"]

        # Build lookup index: (task_id, d4_idx, test_idx) → row index
        self._index: Dict[Tuple[str, int, int], int] = {}
        for i in range(len(self.task_ids)):
            key = (self.task_ids[i], int(self.d4_indices[i]), int(self.test_indices[i]))
            self._index[key] = i

        # Build split → task_id list
        self._train_ids: List[str] = []
        self._eval_ids: List[str] = []
        seen_train, seen_eval = set(), set()
        for i, tid in enumerate(self.task_ids):
            if self.splits[i] == "train" and tid not in seen_train:
                self._train_ids.append(tid)
                seen_train.add(tid)
            elif self.splits[i] == "eval" and tid not in seen_eval:
                self._eval_ids.append(tid)
                seen_eval.add(tid)

        # Load ARC tasks
        all_tasks = load_arc_tasks(data_dir)
        self._tasks: Dict[str, dict] = {}
        for split_key in ("train", "eval"):
            for t in all_tasks.get(split_key, []):
                self._tasks[t["task_id"]] = t

        self.max_seq_len = max_seq_len

        # Load precomputed similarity matrices (optional)
        self._sim_data: Optional[dict] = None
        self._sim_index: Dict[Tuple[str, int, int], int] = {}
        if similarity_path:
            self._load_similarities(similarity_path)

        print(f"  OracleArcDataset: {len(self.embeddings)} embeddings, "
              f"{len(self._train_ids)} train tasks, {len(self._eval_ids)} eval tasks, "
              f"oracle_dim={self.oracle_dim}, "
              f"similarities={'yes' if self._sim_data else 'no'}")

    def _load_similarities(self, path: str):
        """Load precomputed similarity matrices from precompute_similarity.py output."""
        sim_data = torch.load(path, map_location="cpu", weights_only=False)
        self._sim_data = sim_data

        # Build index: (task_id, d4_idx, test_idx) → row in similarity data
        task_ids = sim_data["task_ids"]
        d4_indices = sim_data["d4_indices"]
        test_indices = sim_data["test_indices"]
        for i in range(len(task_ids)):
            key = (task_ids[i], int(d4_indices[i]), int(test_indices[i]))
            self._sim_index[key] = i

        n_sims = len(sim_data["similarities"])
        print(f"  Loaded {n_sims} similarity matrices from {path}")

    @property
    def has_similarities(self) -> bool:
        return self._sim_data is not None

    def get_similarity(
        self, task_id: str, d4_idx: int, test_idx: int,
    ) -> Optional[Tuple[torch.Tensor, list, list]]:
        """Look up precomputed similarity matrix for a specific task variant.

        Does NOT fall back to d4=0 — the similarity must match the exact
        D4 variant used for the sequence, otherwise cell coordinates won't align.

        Returns:
            (similarity [N_cells, N_cells], cell_coords [(r,c,gid),...],
             grid_dims [(H,W),...]) or None if not found.
        """
        if self._sim_data is None:
            return None

        key = (task_id, d4_idx, test_idx)
        idx = self._sim_index.get(key)
        if idx is None:
            return None

        return (
            self._sim_data["similarities"][idx],
            self._sim_data["cell_coords"][idx],
            self._sim_data["grid_dims"][idx],
        )

    def sample_batch(
        self,
        batch_size: int,
        split: str,
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Sample a batch of oracle embeddings + ARC sequences.

        Args:
            batch_size: number of samples
            split: "train" or "eval"
            device: target device

        Returns:
            oracle_embs: [B, oracle_dim] precomputed oracle embeddings
            batch: dict with padded ARC tensors (same format as model.forward())
                   If similarities loaded, batch also includes:
                     - "oracle_sim": [B, max_cells, max_cells] similarity matrices
                     - "cell_to_seq": [B, max_cells] cell→sequence index mapping
                     - "sim_valid_mask": [B, max_cells, max_cells] same-grid pair mask
                     - "n_cells": [B] actual number of cells per sample
        """
        task_pool = self._train_ids if split == "train" else self._eval_ids

        oracle_rows = []
        sequences = []
        sim_info: List[Optional[Tuple]] = []  # (sim, cell_coords, grid_dims) or None
        sample_keys: List[Tuple[str, int, int]] = []  # (task_id, d4_idx, test_idx)

        attempts = 0
        while len(sequences) < batch_size and attempts < batch_size * 10:
            attempts += 1
            task_id = random.choice(task_pool)
            task = self._tasks.get(task_id)
            if task is None:
                continue

            # Random D4 augmentation
            d4_idx = random.randint(0, 7)

            # Random test pair
            test_idx = random.randint(0, len(task["test"]) - 1)

            # Look up precomputed embedding
            key = (task_id, d4_idx, test_idx)
            row_idx = self._index.get(key)
            if row_idx is None:
                # Fall back to d4=0, test=0 if exact variant not precomputed
                key = (task_id, 0, 0)
                row_idx = self._index.get(key)
                if row_idx is None:
                    continue
                d4_idx = 0
                test_idx = 0

            # Build ARC sequence
            seq = build_sequence(
                task, d4_idx=d4_idx, test_idx=test_idx,
                max_seq_len=self.max_seq_len,
            )
            if seq is None:
                continue

            oracle_rows.append(row_idx)
            sequences.append(seq)
            sample_keys.append((task_id, d4_idx, test_idx))

            # Look up similarity (must use same d4_idx/test_idx as the sequence)
            if self.has_similarities:
                sim_result = self.get_similarity(task_id, d4_idx, test_idx)
                if sim_result is None:
                    # No similarity for this variant — skip distillation for this sample
                    sim_info.append(None)
                else:
                    sim_info.append(sim_result)
            else:
                sim_info.append(None)

        if len(sequences) == 0:
            raise RuntimeError(f"Could not build any sequences for split={split}")

        # Pad sequences to batch
        batch = self._collate(sequences, device)

        # Stack oracle embeddings
        oracle_embs = self.embeddings[oracle_rows].to(device=device, dtype=torch.float32)

        # Add similarity data to batch if available
        if self.has_similarities and any(s is not None for s in sim_info):
            self._add_similarity_to_batch(batch, sim_info, device)

        return oracle_embs, batch

    def _add_similarity_to_batch(
        self,
        batch: Dict[str, torch.Tensor],
        sim_info: List[Optional[Tuple]],
        device: torch.device,
    ):
        """Add padded similarity matrices and cell-to-sequence mappings to batch.

        Pads variable-size similarity matrices to the max cell count in the batch.
        Builds cell_to_seq mapping by matching cell_coords (r, c, grid_id) to
        the batch's xs, ys, grid_ids tensors.
        """
        from .oracle_distill import build_cell_to_seq_map, build_valid_mask

        B = batch["xs"].shape[0]
        N_seq = batch["xs"].shape[1]

        # Find max number of cells across batch
        max_cells = 0
        for info in sim_info:
            if info is not None:
                sim, coords, _ = info
                max_cells = max(max_cells, sim.shape[0])

        if max_cells == 0:
            return

        # Build padded tensors
        oracle_sim = torch.zeros(B, max_cells, max_cells, device=device)
        cell_to_seq = torch.zeros(B, max_cells, dtype=torch.long, device=device)
        sim_valid_mask = torch.zeros(B, max_cells, max_cells, device=device)
        n_cells = torch.zeros(B, dtype=torch.long, device=device)

        for b, info in enumerate(sim_info):
            if info is None:
                continue

            sim, cell_coords, grid_dims = info
            nc = sim.shape[0]
            assert nc == len(cell_coords), (
                f"Similarity matrix size {nc} != cell_coords length {len(cell_coords)}"
            )
            n_cells[b] = nc

            # Pad similarity matrix
            oracle_sim[b, :nc, :nc] = sim.to(device)

            # Build cell→sequence mapping for this sample
            mapping = build_cell_to_seq_map(
                cell_coords,
                batch["xs"][b:b+1],
                batch["ys"][b:b+1],
                batch["grid_ids"][b:b+1],
                batch["sep_mask"][b:b+1],
            )  # [1, nc]
            cell_to_seq[b, :nc] = mapping[0, :nc]

            # Build valid mask (same-grid pairs)
            vmask = build_valid_mask(cell_coords)  # [nc, nc]
            sim_valid_mask[b, :nc, :nc] = vmask.float().to(device)

            # Zero out unmapped cells (sentinel = -1 from build_cell_to_seq_map)
            mapped = (cell_to_seq[b, :nc] >= 0)  # [nc]
            mapped_2d = mapped.unsqueeze(1) & mapped.unsqueeze(0)  # [nc, nc]
            sim_valid_mask[b, :nc, :nc] *= mapped_2d.float()

        batch["oracle_sim"] = oracle_sim
        batch["cell_to_seq"] = cell_to_seq
        batch["sim_valid_mask"] = sim_valid_mask
        batch["n_cells"] = n_cells

    def _collate(
        self, sequences: List[Dict[str, torch.Tensor]], device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Pad list of single-sequence dicts to batched tensors."""
        padded = [
            pad_single_to_batch(seq, self.max_seq_len, device)
            for seq in sequences
        ]

        batch = {}
        for key in padded[0]:
            tensors = [p[key] for p in padded]
            if isinstance(tensors[0], torch.Tensor):
                batch[key] = torch.cat(tensors, dim=0)
            else:
                batch[key] = tensors
        return batch


def forward_with_oracle(
    model: LiquidARCModel,
    z_context: torch.Tensor,
    kappa_target: torch.Tensor,
    colors: torch.Tensor,
    xs: torch.Tensor,
    ys: torch.Tensor,
    roles: torch.Tensor,
    sep_mask: torch.Tensor,
    sep_types: torch.Tensor,
    target_mask: torch.Tensor,
    target_labels: Optional[torch.Tensor] = None,
    context_mask: Optional[torch.Tensor] = None,
    target_input_colors: Optional[torch.Tensor] = None,
    grid_ids: Optional[torch.Tensor] = None,
    n_steps: Optional[int] = None,
    lambda_kappa: float = 0.1,
    delta_W_o: Optional[torch.Tensor] = None,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """Run LiquidARCModel forward with oracle-projected context + kappa distillation.

    Mirrors wake_sleep/model.py:forward_with_external_context but adds:
      - kappa_distill_loss: MSE(|κ|_actual, κ_target)
      - delta_W_o: optional HyperNet-predicted W_o weight delta

    Does NOT touch the compiled dynamics path — set_context() is called before ODE.

    Args:
        model: LiquidARCModel instance (possibly torch.compiled dynamics)
        z_context: [B, d_model] projected oracle context
        kappa_target: [B, 1] target curvature from oracle projection head
        ... (standard ARC model inputs)
        lambda_kappa: weight for kappa distillation loss
        delta_W_o: [d_model, d_model] optional HyperNet W_o delta

    Returns:
        Result dict with all standard fields + kappa_distill_loss
    """
    config = model.config
    device = colors.device
    actual_steps = n_steps if n_steps is not None else config.n_ode_steps

    # Mask test output colors
    colors_masked = colors.clone()
    if target_input_colors is not None:
        colors_masked[target_mask] = target_input_colors[target_mask]
    else:
        colors_masked[target_mask] = PAD_COLOR

    # Embed
    h0 = model.embedding(colors_masked, xs, ys, roles, sep_mask, sep_types,
                          grid_ids=grid_ids)

    # Inject oracle context (bypass ContextPool)
    model.dynamics.set_context(z_context, mask=None)
    model.dynamics.set_n_steps(actual_steps)

    # Inject HyperNet W_o delta (None resets to zero = original behavior)
    model.dynamics.set_delta_W_o(delta_W_o)

    # Diagnostics from initial state
    g_init = model.dynamics.compute_metric(h0)
    kappa = model.curvature_engine(g_init)
    metric_cv = g_init.std() / (g_init.mean() + 1e-8)

    if config.channel_gate_enabled:
        gate_init = model.dynamics.compute_gate(h0)
        tau_avg_val = gate_init.mean()
        tau_std_val = gate_init.mean(dim=-1).std(dim=1).mean()
        tau_var_val = gate_init.mean(dim=-1).var(dim=1).mean()
        tau_min_val = gate_init.min()
        tau_max_val = gate_init.max()
    else:
        tau_init = model.dynamics.compute_tau(h0)
        tau_avg_val = tau_init.mean(dim=1).mean()
        tau_flat = tau_init.squeeze(-1)
        tau_std_val = tau_flat.std(dim=1).mean()
        tau_var_val = tau_flat.var(dim=1).mean()
        tau_min_val = tau_flat.min()
        tau_max_val = tau_flat.max()

    # ODE integration
    if config.deq_solver:
        h = deq_solve(
            model.dynamics, h0, t_span=(0.0, 1.0),
            n_steps=actual_steps, n_ift_iters=config.deq_ift_iters,
        )
    elif config.invertible_solver:
        h = invertible_euler_solve(
            model.dynamics, h0, t_span=(0.0, 1.0),
            n_steps=actual_steps, n_fp_iters=config.n_fp_iters,
        )
    else:
        h = euler_solve(model.dynamics, h0, t_span=(0.0, 1.0), n_steps=actual_steps)

    # Output head
    logits = model.output_head(model.norm_out(h))

    result = {
        "logits": logits,
        "h0": h0,
        "h_final": h,
        "metric_cv": metric_cv,
        "avg_kappa": kappa.abs().mean(),
        "tau_avg": tau_avg_val,
        "tau_std": tau_std_val,
        "tau_min": tau_min_val,
        "tau_max": tau_max_val,
        "geo_loss": torch.tensor(0.0, device=device),
        "geo_mse": torch.tensor(0.0, device=device),
    }

    # CE + task losses (reuse model's internal loss computation)
    if target_labels is not None:
        result.update(model._compute_loss(
            logits, target_labels, target_mask, target_input_colors,
            kappa, tau_var_val, metric_cv, device,
        ))

    # Kappa distillation loss: log-space MSE (curvature is exponential)
    # In linear space, |0.006 - 0.04| = 0.034 (tiny gradient).
    # In log space, |log(0.006) - log(0.04)| = 1.90 (massive gradient).
    actual_kappa = kappa.abs().mean()  # scalar
    kappa_t = kappa_target.squeeze(-1).mean().detach()  # scalar, detached
    eps = 1e-6
    kappa_distill = F.mse_loss(
        torch.log(actual_kappa + eps), torch.log(kappa_t + eps)
    )
    result["kappa_distill_loss"] = kappa_distill
    result["kappa_target_mean"] = kappa_t

    return result
