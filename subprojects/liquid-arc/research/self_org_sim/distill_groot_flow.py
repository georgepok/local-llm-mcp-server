"""Flow-matching distillation: Liquid encoder -> conditional flow head.

Matches GR00T's own training paradigm (flow-matching DiT). Replaces the BC
linear action head with a small conditional transformer that predicts a velocity
field over the action chunk. Training learns the velocity at a random t in [0,1];
inference integrates the ODE from random noise to the action chunk.

Why this is different from BC v1/v2/v3:
  - Loss is MSE on velocity field, not on actions directly
  - The model learns the *score* of the action distribution, not point predictions
  - Empirically shows phase transitions when the score-function structure crystallizes
  - Sampling many denoising steps is naturally robust to compound error in closed loop
  - Industry SOTA for robotic BC (Diffusion Policy, Chi et al; GR00T uses this)

Run on Spark (uses our 3.12 venv with CUDA torch):
  cd /home/pokazge/liquid-arc/research/self_org_sim
  source /home/pokazge/Isaac-GR00T/.venv/bin/activate
  source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
  python distill_groot_flow.py \\
    --data_dir /home/pokazge/datasets/libero-10-r-decoded \\
    --teacher_labels_dir /home/pokazge/datasets/libero-10-r-groot-labels \\
    --out_dir /tmp/distill_groot_flow \\
    --max_steps 30000 --batch_size 256 --d 512 --augment --n_tasks 10 --compile
"""

from __future__ import annotations

import argparse
import csv
import functools
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from distill_groot import (
    LiberoMemmapDataset,
    TeacherLabelDataset,
    VisionEncoder,
    augment_imgs,
    collate_batch,
    soc_penalty,
)

print = functools.partial(print, flush=True)
torch.set_float32_matmul_precision("high")

# GB10 (sm121) doesn't ship the cutlass mem-efficient SDPA kernels that PyTorch
# tried to use. Disable them so SDPA falls back to flash-attn (built for sm121).
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)


# ---------------------------------------------------------------------------
# Liquid encoder (vision + state + ODE) -> condition vector
# ---------------------------------------------------------------------------

class LiquidEncoder(nn.Module):
    """Same encoder as LiquidStudent, but exposes the post-ODE hidden as condition."""

    def __init__(self, state_dim=8, d=256, d_vis=256, img_size=96, k_max=16,
                 halt_mode="learned", min_steps=4, dt=0.5,
                 n_tasks=0, d_task=32, z_groot_dim=0,
                 z_channel_dims=None, gated_mixture=False, s2_halt=False,
                 query_bank=False, query_dim=8, forward_model=False,
                 action_dim=7, cadence_head=False):
        super().__init__()
        self.k_max = k_max
        self.halt_mode = halt_mode
        self.min_steps = min_steps
        self.dt = dt
        self.n_tasks = n_tasks
        self.z_groot_dim = z_groot_dim
        # z_channel_dims: list of per-channel dims, e.g. [2048, 1536] for z_vl+z_state.
        # When set, encoder splits z_groot internally into channels of those sizes.
        self.z_channel_dims = z_channel_dims or []
        self.gated_mixture = gated_mixture and len(self.z_channel_dims) >= 2
        self.s2_halt = s2_halt
        # V6a: bidirectional integration via query bank. Liquid emits Δs that
        # selects (via soft attention) over K pre-computed perturbed-z entries.
        self.query_bank = query_bank

        self.vis_enc = VisionEncoder(in_ch=3, d_out=d_vis, img_size=img_size)
        self.wrist_enc = VisionEncoder(in_ch=3, d_out=d_vis, img_size=img_size)
        self.state_enc = nn.Sequential(
            nn.Linear(state_dim, d), nn.SiLU(),
            nn.Linear(d, d), nn.SiLU(),
        )
        if n_tasks > 0:
            self.task_emb = nn.Embedding(n_tasks, d_task)
            fuse_in = d_vis * 2 + d + d_task
        else:
            self.task_emb = None
            fuse_in = d_vis * 2 + d
        self.fuse = nn.Linear(fuse_in, d)
        # System-2 intent injection. Zero-init means initial behavior is
        # identical to z_groot=None — the network must learn to use the signal.
        # Two modes:
        #   gated_mixture=False: single z_groot_proj over the (concatenated) z_groot
        #   gated_mixture=True (V5b): separate per-channel projections + softmax gate
        #     over h_pre selecting per-position which channel to lean on.
        if z_groot_dim > 0:
            if self.gated_mixture:
                self.z_channel_projs = nn.ModuleList([
                    nn.Linear(cd, d) for cd in self.z_channel_dims
                ])
                for proj in self.z_channel_projs:
                    nn.init.zeros_(proj.weight)
                    nn.init.zeros_(proj.bias)
                self.channel_gate = nn.Linear(d, len(self.z_channel_dims))
                nn.init.zeros_(self.channel_gate.weight)
                nn.init.zeros_(self.channel_gate.bias)
                self.z_groot_proj = None
            else:
                self.z_groot_proj = nn.Linear(z_groot_dim, d)
                nn.init.zeros_(self.z_groot_proj.weight)
                nn.init.zeros_(self.z_groot_proj.bias)
                self.z_channel_projs = None
                self.channel_gate = None
        else:
            self.z_groot_proj = None
            self.z_channel_projs = None
            self.channel_gate = None

        self.drift = nn.Sequential(
            nn.Linear(d, d * 2), nn.SiLU(),
            nn.Linear(d * 2, d),
        )
        nn.init.zeros_(self.drift[-1].weight)
        nn.init.zeros_(self.drift[-1].bias)

        # FiLM modulation: z_vl → (γ, β) applied to drift output at EVERY ODE step.
        # Diagnosed (May 2026): single residual injection of z_vl into h_pre then 16
        # ODE drift steps washes out z_vl content — model became insensitive to
        # which z_vl it gets (only sensitive to presence/absence). FiLM lets z_vl
        # modulate drift dynamics at each step, preserving task-conditioning all
        # the way through the ODE. Zero-init: γ=1, β=0 → no modulation initially,
        # learned during training. Only created when z_groot_dim > 0.
        if z_groot_dim > 0:
            self.z_vl_film = nn.Linear(d, 2 * d)
            nn.init.zeros_(self.z_vl_film.weight)
            nn.init.zeros_(self.z_vl_film.bias)
        else:
            self.z_vl_film = None

        self.tau_raw = nn.Parameter(torch.zeros(d))

        if halt_mode == "learned":
            self.halt_head = nn.Linear(d, 1)
            with torch.no_grad():
                self.halt_head.bias.fill_(-3.0)

        # V5a: predicts P(call System-2 next chunk) from post-ODE h.
        # Trained via consistency loss between fresh and stale z_groot conds.
        if s2_halt:
            self.s2_halt_head = nn.Linear(d, 1)
            nn.init.zeros_(self.s2_halt_head.weight)
            self.s2_halt_head.bias.fill_(-2.0)  # init halt prob ≈ 0.12

        # V6a/V6c: query head emits a query vector that indexes into a K-entry
        # pre-computed bank via softmax attention. Query dim depends on channel:
        #   V6a (state perturbation):  query_dim = 8
        #   V6c (image-grid attention): query_dim = G*G (e.g. 256 for 16×16)
        # At deployment, the emitted query is applied to GR00T's input via the
        # appropriate channel adapter (state addition or image modulation).
        if query_bank:
            self.query_dim = query_dim
            self.request_head = nn.Linear(d, query_dim)
            nn.init.zeros_(self.request_head.weight)
            nn.init.zeros_(self.request_head.bias)
            # Learned temperature controls bank-attention sharpness.
            self.attn_log_temp = nn.Parameter(torch.zeros(1))

        # V9: forward-model head for self-supervised online adaptation.
        # Predicts cond_{t+1} from (cond_t, chunk_t.mean(dim=0)). At deployment,
        # adaptation gradient comes from prediction error against the actual
        # cond computed at the next chunk decision. Fires every chunk regardless
        # of System-2 cadence — enables K=0 adaptive learning.
        self.forward_model = forward_model
        if forward_model:
            self.forward_pred_head = nn.Sequential(
                nn.Linear(d + action_dim, d * 2), nn.SiLU(),
                nn.Linear(d * 2, d),
            )
            # Identity init: at start, predict cond_{t+1} ≈ cond_t (ignore chunk).
            # Last linear's bias-only path is zero-init so output starts as the
            # SiLU-activated linear projection of cond_t — close to identity.
            nn.init.zeros_(self.forward_pred_head[-1].weight)
            nn.init.zeros_(self.forward_pred_head[-1].bias)

        # Stage 2: cadence head — emits P(call System-2 this chunk) as natural
        # model output. Trained end-to-end via RL reward (success − λ × fires)
        # in Phase B. In Phase A it sits dormant (no direct supervision; gets
        # gradient from the joint loss only if connected). Initial bias chosen
        # so initial firing rate is mid-range (sigmoid(-1)≈0.27).
        self.cadence_head_enabled = cadence_head
        if cadence_head:
            self.cadence_head = nn.Linear(d, 1)
            nn.init.zeros_(self.cadence_head.weight)
            with torch.no_grad():
                self.cadence_head.bias.fill_(-1.0)

    def forward(self, img, wrist_img, state, task_id=None, z_groot=None,
                z_bank=None, delta_bank=None, request_only=False):
        v = self.vis_enc(img)
        w = self.wrist_enc(wrist_img)
        s = self.state_enc(state)
        feats = [v, w, s]
        if self.task_emb is not None:
            if task_id is None:
                t = torch.zeros(v.shape[0], self.task_emb.embedding_dim, device=v.device)
            else:
                t = self.task_emb(task_id)
            feats.append(t)
        h = F.silu(self.fuse(torch.cat(feats, dim=-1)))

        # V6a query-bank path: Liquid emits Δs from h_pre. At training time, soft-
        # attends over a pre-computed K-entry bank to select z. At rollout time,
        # the emitted Δs is sent to GR00T as the actual perturbation and the
        # returned z is fed back via z_groot. Always populate delta_emit so the
        # rollout loop can read it.
        info_extra = {}
        # Capture the d-dim projection of injected z for FiLM use (set if any z is injected).
        z_vl_proj_for_film = None
        if self.query_bank:
            delta_emit = self.request_head(h)  # [B, 8]
            info_extra["delta_emit"] = delta_emit
            if request_only:
                # Rollout pass 1: return after computing delta_emit; skip ODE.
                # cond is undefined but the caller doesn't use it.
                return h, {"delta_emit": delta_emit}
            if z_bank is not None and delta_bank is not None:
                # Training: differentiable selection over bank.
                dist = ((delta_emit.unsqueeze(1) - delta_bank) ** 2).sum(-1)  # [B, K]
                attn = F.softmax(-dist / torch.exp(self.attn_log_temp), dim=-1)  # [B, K]
                z_selected = (attn.unsqueeze(-1) * z_bank).sum(dim=1)  # [B, dim]
                if self.z_groot_proj is not None:
                    z_vl_proj_for_film = self.z_groot_proj(z_selected)
                    h = h + z_vl_proj_for_film
                info_extra["attn"] = attn
            elif z_groot is not None and self.z_groot_proj is not None:
                # Rollout: GR00T already queried with this delta_emit, response is z_groot.
                z_vl_proj_for_film = self.z_groot_proj(z_groot)
                h = h + z_vl_proj_for_film
        elif z_groot is not None:
            if self.gated_mixture and self.z_channel_projs is not None:
                # Split z_groot into per-channel slices, project each separately,
                # then mix via softmax gate over h_pre.
                gates = F.softmax(self.channel_gate(h), dim=-1)  # [B, n_channels]
                offset = 0
                mix = 0
                for i, (proj, cd) in enumerate(zip(self.z_channel_projs, self.z_channel_dims)):
                    z_slice = z_groot[..., offset:offset + cd]
                    mix = mix + gates[..., i:i + 1] * proj(z_slice)
                    offset += cd
                h = h + mix
                z_vl_proj_for_film = mix
            elif self.z_groot_proj is not None:
                z_vl_proj_for_film = self.z_groot_proj(z_groot)
                h = h + z_vl_proj_for_film

        B = h.shape[0]
        tau = F.softplus(self.tau_raw) + 0.1
        # FiLM (γ, β) computed once from injected z_vl projection — used at every
        # ODE step so z_vl content shapes drift dynamics throughout the loop, not
        # just via the residual injection that gets washed out over 16 iterations.
        film_gamma = None
        film_beta = None
        if z_vl_proj_for_film is not None and self.z_vl_film is not None:
            gb = self.z_vl_film(z_vl_proj_for_film)
            gamma_raw, film_beta = gb.chunk(2, dim=-1)
            film_gamma = 1.0 + gamma_raw  # init γ=1 (unmodulated when zero-init)
        steps_used = torch.zeros(B, 1, device=h.device)
        still_active = torch.ones(B, 1, device=h.device)
        for k in range(self.k_max):
            dh_raw = self.drift(h) / tau
            if film_gamma is not None:
                dh = film_gamma * dh_raw + film_beta
            else:
                dh = dh_raw
            if self.halt_mode == "learned":
                h_new = h + self.dt * dh
                h = still_active * h_new + (1.0 - still_active) * h
                p_halt = torch.sigmoid(self.halt_head(h))
                steps_used = steps_used + still_active
                if k >= self.min_steps:
                    still_active = still_active * (1.0 - p_halt)
            else:
                h = h + self.dt * dh
                steps_used = steps_used + 1.0

        h_norms = h.norm(dim=-1)
        h_cv = h_norms.std() / (h_norms.mean() + 1e-8)
        info = {"steps_mean": steps_used.mean().detach(), "h_cv": h_cv}
        if self.s2_halt:
            info["s2_halt_logit"] = self.s2_halt_head(h)  # [B, 1]
        if self.cadence_head_enabled:
            # Stage 2: P(call System-2 this chunk). Sigmoid in caller.
            info["cadence_logit"] = self.cadence_head(h)  # [B, 1]
        info.update(info_extra)
        return h, info


# ---------------------------------------------------------------------------
# Flow-matching head: small conditional transformer over action_chunk positions
# ---------------------------------------------------------------------------

class FlowMatchingHead(nn.Module):
    """Predicts velocity v(noisy_chunk, t, cond) for rectified-flow training."""

    def __init__(self, d_cond: int, action_horizon: int, action_dim: int = 7,
                 d_model: int = 256, n_layers: int = 4, n_heads: int = 4,
                 d_t: int = 64):
        super().__init__()
        self.K = action_horizon
        self.A = action_dim
        self.d_model = d_model

        self.action_proj_in = nn.Linear(action_dim, d_model)
        self.cond_proj = nn.Linear(d_cond, d_model)
        self.t_embed = nn.Sequential(
            nn.Linear(1, d_t), nn.SiLU(),
            nn.Linear(d_t, d_model), nn.SiLU(),
        )
        self.pos_embed = nn.Parameter(torch.zeros(action_horizon, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            activation="gelu", batch_first=True, norm_first=True,
            dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.action_proj_out = nn.Linear(d_model, action_dim)
        nn.init.zeros_(self.action_proj_out.weight)
        nn.init.zeros_(self.action_proj_out.bias)

    def forward(self, noisy_chunk: torch.Tensor, t: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        # noisy_chunk: [B, K, A], t: [B], cond: [B, d_cond]
        x = self.action_proj_in(noisy_chunk) + self.pos_embed.unsqueeze(0)
        cond_emb = self.cond_proj(cond) + self.t_embed(t.unsqueeze(-1))
        x = x + cond_emb.unsqueeze(1)
        x = self.transformer(x)
        v = self.action_proj_out(x)
        return v


class LiquidFlowPolicy(nn.Module):
    """Encoder (vision+state+ODE) + flow-matching denoiser head(s).

    If `n_task_heads > 0`, builds N specialist flow heads sharing the encoder.
    forward()/sample() route by task_id to the right head. Each head only sees
    gradients from samples of its task — true per-task specialization.
    """

    def __init__(self, state_dim=8, action_dim=7, action_horizon=16,
                 d=512, d_vis=512, img_size=96, k_max=16,
                 halt_mode="learned", min_steps=4, dt=0.5,
                 n_tasks=0, d_task=32,
                 head_d=256, head_layers=4, head_heads=4,
                 n_task_heads=0, z_groot_dim=0,
                 z_channel_dims=None, gated_mixture=False, s2_halt=False,
                 query_bank=False, query_dim=8, forward_model=False,
                 cadence_head=False):
        super().__init__()
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.n_task_heads = n_task_heads
        self.z_groot_dim = z_groot_dim
        self.query_bank = query_bank
        self.encoder = LiquidEncoder(
            state_dim=state_dim, d=d, d_vis=d_vis, img_size=img_size,
            k_max=k_max, halt_mode=halt_mode, min_steps=min_steps, dt=dt,
            n_tasks=n_tasks, d_task=d_task, z_groot_dim=z_groot_dim,
            z_channel_dims=z_channel_dims, gated_mixture=gated_mixture,
            s2_halt=s2_halt, query_bank=query_bank, query_dim=query_dim,
            forward_model=forward_model, action_dim=action_dim,
            cadence_head=cadence_head,
        )
        if n_task_heads > 0:
            self.heads = nn.ModuleList([
                FlowMatchingHead(
                    d_cond=d, action_horizon=action_horizon, action_dim=action_dim,
                    d_model=head_d, n_layers=head_layers, n_heads=head_heads,
                )
                for _ in range(n_task_heads)
            ])
            self.head = None
        else:
            self.head = FlowMatchingHead(
                d_cond=d, action_horizon=action_horizon, action_dim=action_dim,
                d_model=head_d, n_layers=head_layers, n_heads=head_heads,
            )
            self.heads = None

    def _route_velocity(self, noisy_chunk, t, cond, task_id):
        """Route each batch element to its task-specialist head."""
        if self.heads is None:
            return self.head(noisy_chunk, t, cond)
        if task_id is None:
            # Default to head 0 if we have specialists but no task_id
            return self.heads[0](noisy_chunk, t, cond)
        # Loop over task buckets in the batch (batches typically have 1-3 tasks)
        out = torch.empty_like(noisy_chunk)
        for ti in torch.unique(task_id):
            ti_int = int(ti.item())
            mask = (task_id == ti)
            if not mask.any():
                continue
            head = self.heads[ti_int % len(self.heads)]
            out[mask] = head(noisy_chunk[mask], t[mask], cond[mask])
        return out

    def forward_encoder(self, img, wrist_img, state, task_id=None, z_groot=None,
                        z_bank=None, delta_bank=None, request_only=False):
        return self.encoder(img, wrist_img, state, task_id=task_id, z_groot=z_groot,
                            z_bank=z_bank, delta_bank=delta_bank,
                            request_only=request_only)

    @torch.no_grad()
    def emit_query(self, img, wrist_img, state, task_id=None):
        """V6a rollout helper: returns Liquid's emitted Δs query (8-d) for GR00T."""
        _, info = self.forward_encoder(img, wrist_img, state, task_id=task_id,
                                        request_only=True)
        return info["delta_emit"]

    @torch.no_grad()
    def emit_cadence(self, img, wrist_img, state, task_id=None,
                     z_groot=None, z_bank=None, delta_bank=None):
        """Stage 2 rollout helper: returns P(call System-2 this chunk) ∈ [0,1].
        Uses Liquid's cadence_head — the cognitive-system way of deciding K.
        """
        _, info = self.forward_encoder(
            img, wrist_img, state, task_id=task_id,
            z_groot=z_groot, z_bank=z_bank, delta_bank=delta_bank,
        )
        return torch.sigmoid(info["cadence_logit"]).item()

    def predict_next_cond(self, cond, chunk):
        """V9 forward model: predict next cond from current cond + chunk plan.

        cond:  [B, d]    — current encoder output
        chunk: [B, K, A] — predicted action chunk
        returns: [B, d]  — predicted cond at next chunk-decision step
        """
        chunk_summary = chunk.mean(dim=1)  # [B, A]
        x = torch.cat([cond, chunk_summary], dim=-1)  # [B, d + A]
        return self.encoder.forward_pred_head(x)

    def velocity(self, noisy_chunk, t, cond, task_id=None):
        return self._route_velocity(noisy_chunk, t, cond, task_id)

    @torch.no_grad()
    def sample(self, img, wrist_img, state, task_id=None, n_steps: int = 10,
               z_groot=None, z_bank=None, delta_bank=None):
        cond, _ = self.forward_encoder(img, wrist_img, state, task_id=task_id,
                                       z_groot=z_groot, z_bank=z_bank,
                                       delta_bank=delta_bank)
        B = cond.shape[0]
        x = torch.randn(B, self.action_horizon, self.action_dim, device=cond.device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=cond.device)
            v = self._route_velocity(x, t, cond, task_id)
            x = x + dt * v
        return x


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def flow_loss(model: LiquidFlowPolicy, img, wrist, state, chunks,
               task_id=None, z_groot=None, z_bank=None,
               delta_bank=None) -> tuple[torch.Tensor, dict]:
    """Rectified-flow loss: predict velocity field along noise→action straight path."""
    cond, info = model.forward_encoder(img, wrist, state, task_id=task_id,
                                        z_groot=z_groot, z_bank=z_bank,
                                        delta_bank=delta_bank)
    B = chunks.shape[0]
    t = torch.rand(B, device=chunks.device)
    noise = torch.randn_like(chunks)
    t_b = t.view(-1, 1, 1)
    noisy = (1.0 - t_b) * noise + t_b * chunks
    v_target = chunks - noise
    v_pred = model.velocity(noisy, t, cond, task_id=task_id)
    loss = F.mse_loss(v_pred, v_target)
    info["cond"] = cond  # exposed for V5a halt consistency loss
    return loss, info


@torch.no_grad()
def evaluate(model, val_loader, device, max_batches: int = 20, n_steps: int = 10):
    """Sample action chunks via ODE solve; report MSE/MAE vs ground truth."""
    model.eval()
    sq, ab, n = 0.0, 0.0, 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        z_groot = None
        if len(batch) == 6:
            imgs, wrists, states, chunks, task_ids, z_groot = batch
            task_ids = task_ids.to(device)
            z_groot = z_groot.to(device)
            if (task_ids < 0).all():
                task_ids = None
        elif len(batch) == 5:
            imgs, wrists, states, chunks, task_ids = batch
            task_ids = task_ids.to(device)
        else:
            imgs, wrists, states, chunks = batch
            task_ids = None
        imgs = imgs.to(device)
        wrists = wrists.to(device)
        states = states.to(device)
        chunks = chunks.to(device)
        pred = model.sample(imgs, wrists, states, task_id=task_ids,
                            n_steps=n_steps, z_groot=z_groot)
        diff = pred - chunks
        sq += diff.pow(2).mean().item() * imgs.shape[0]
        ab += diff.abs().mean().item() * imgs.shape[0]
        n += imgs.shape[0]
    model.train()
    return sq / max(n, 1), ab / max(n, 1)


def train(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir = {out_dir}")
    print(f"args: {vars(args)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    idx = np.load(Path(args.data_dir) / "index.npz")
    n_eps = len(idx["episode_lengths"])
    n_val = max(10, n_eps // 10)
    val_indices = list(range(n_eps - n_val, n_eps))
    train_indices = list(range(n_eps - n_val))
    use_lang = args.n_tasks > 0

    cadence_list = [int(x) for x in args.cadence_dropout.split(",") if x.strip()]
    if cadence_list:
        print(f"  cadence_dropout: K-offsets {cadence_list} (chunks)")
    if args.teacher_labels_dir:
        # Pass --img_size as target_img_size so the dataset resizes on read
        # (matters when teacher data was collected at native LIBERO 256×256
        # but Liquid's VisionEncoder is sized for 96×96).
        primary_ds = TeacherLabelDataset(
            args.data_dir, args.teacher_labels_dir,
            action_horizon=args.action_horizon, return_task_id=use_lang,
            target_img_size=args.img_size, return_z_groot=args.use_z_groot,
            cadence_dropout=cadence_list, return_z_fresh=args.s2_halt,
            use_query_bank=args.use_query_bank,
            z_groot_drop_prob=args.z_groot_drop_prob,
        )
        train_img_size = primary_ds.target_img_size
        args.z_groot_dim = primary_ds.z_groot_dim
        args.z_channel_dims = primary_ds.z_dims
        # Auto-detect query_dim from dataset (8 for V6a state, 256 for V6c image-grid)
        args.query_dim = getattr(primary_ds, "query_dim", 8)
        args.query_channel = getattr(primary_ds, "query_channel", "state")
        if primary_ds.target_img_size != primary_ds.img_size:
            print(f"  resize: stored {primary_ds.img_size} → model {train_img_size}")
        if args.use_query_bank:
            print(f"  query channel: {args.query_channel}, query_dim: {args.query_dim}")
        extra_datasets = [primary_ds]
        if args.dagger_data_dir and args.dagger_labels_dir:
            d = TeacherLabelDataset(
                args.dagger_data_dir, args.dagger_labels_dir,
                action_horizon=args.action_horizon, return_task_id=use_lang,
                target_img_size=train_img_size, return_z_groot=args.use_z_groot,
                cadence_dropout=cadence_list,
            )
            print(f"  + dagger samples: {len(d):,} (stored {d.img_size}, resize→{train_img_size})")
            extra_datasets.append(d)
        if args.dagger2_data_dir and args.dagger2_labels_dir:
            d2 = TeacherLabelDataset(
                args.dagger2_data_dir, args.dagger2_labels_dir,
                action_horizon=args.action_horizon, return_task_id=use_lang,
                target_img_size=train_img_size, return_z_groot=args.use_z_groot,
                cadence_dropout=cadence_list,
            )
            print(f"  + dagger2 samples: {len(d2):,} (stored {d2.img_size}, resize→{train_img_size})")
            extra_datasets.append(d2)
        # Multi-suite expert demos. Each path is used as both data_dir and
        # teacher_labels_dir (the libero-*-expert-v1 datasets store both inputs
        # and chunks in one directory). z_vl/z_bank are zeros in these datasets,
        # so combined with z_groot_drop_prob the model sees both regimes:
        # "I have GR00T advice" (primary_ds, libero_10 GR00T-distilled) and
        # "I don't have GR00T advice" (the new suites, expert-only).
        if args.extra_teacher_dirs:
            for extra_path in [p.strip() for p in args.extra_teacher_dirs.split(",") if p.strip()]:
                de = TeacherLabelDataset(
                    extra_path, extra_path,
                    action_horizon=args.action_horizon, return_task_id=use_lang,
                    target_img_size=train_img_size, return_z_groot=args.use_z_groot,
                    cadence_dropout=cadence_list,
                    use_query_bank=args.use_query_bank,
                    z_groot_drop_prob=args.z_groot_drop_prob,
                )
                print(f"  + extra suite {Path(extra_path).name}: {len(de):,} samples "
                      f"(stored {de.img_size}, resize→{train_img_size})")
                extra_datasets.append(de)
        if len(extra_datasets) > 1:
            train_ds = ConcatDataset(extra_datasets)
            print(f"  combined train size: {len(train_ds):,}")
        else:
            train_ds = primary_ds
    else:
        train_ds = LiberoMemmapDataset(
            args.data_dir, args.action_horizon, train_indices,
            return_task_id=use_lang,
        )
        train_img_size = train_ds.img_size
    # val_ds: ground-truth demos. When z_groot conditioning is on, the val set
    # has no z_groot fields → val_mse would test a different distribution than
    # training. Skip val in that case (closed-loop sim is the real eval).
    val_ds = None
    val_loader = None
    if args.use_z_groot:
        print(f"train samples: {len(train_ds):,}  (val skipped — z_groot conditioning on; use closed-loop)")
    else:
        val_ds = LiberoMemmapDataset(
            args.data_dir, args.action_horizon, val_indices,
            return_task_id=use_lang,
        )
        print(f"train samples: {len(train_ds):,}  val samples: {len(val_ds):,}")
    args.img_size = train_img_size

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_batch,
        pin_memory=True,
        prefetch_factor=4 if args.num_workers else None,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=2, collate_fn=collate_batch, pin_memory=True,
        )

    halt_mode = "learned" if args.policy == "liquid_halt" else "none"
    z_groot_dim = getattr(args, "z_groot_dim", 0)
    z_channel_dims = getattr(args, "z_channel_dims", None)
    policy = LiquidFlowPolicy(
        state_dim=8, action_dim=7, action_horizon=args.action_horizon,
        d=args.d, d_vis=args.d, img_size=args.img_size,
        k_max=args.k, halt_mode=halt_mode, min_steps=args.halting_min_steps,
        n_tasks=args.n_tasks, d_task=args.d_task,
        head_d=args.head_d, head_layers=args.head_layers, head_heads=args.head_heads,
        n_task_heads=args.n_task_heads, z_groot_dim=z_groot_dim,
        z_channel_dims=z_channel_dims,
        gated_mixture=args.gated_mixture,
        s2_halt=args.s2_halt,
        query_bank=args.use_query_bank,
        query_dim=getattr(args, "query_dim", 8),
        cadence_head=args.cadence_head,
    ).to(device)
    if z_groot_dim > 0:
        mode_tag = "gated" if args.gated_mixture else "concat"
        halt_tag = " +s2_halt" if args.s2_halt else ""
        print(f"  z_groot conditioning: {args.use_z_groot} (dim={z_groot_dim}) "
              f"mode={mode_tag}{halt_tag}")
    n_params = sum(p.numel() for p in policy.parameters())
    n_enc = sum(p.numel() for p in policy.encoder.parameters())
    if policy.heads is not None:
        n_head = sum(p.numel() for p in policy.heads.parameters())
        print(f"policy params: {n_params/1e6:.2f}M ({n_params:,})  "
              f"= encoder {n_enc/1e6:.2f}M + {len(policy.heads)} task-specialist heads "
              f"{n_head/1e6:.2f}M total ({n_head/len(policy.heads)/1e6:.2f}M each)")
    else:
        n_head = sum(p.numel() for p in policy.head.parameters())
        print(f"policy params: {n_params/1e6:.2f}M ({n_params:,})  "
              f"= encoder {n_enc/1e6:.2f}M + flow_head {n_head/1e6:.2f}M")

    if args.resume:
        print(f"warm-starting from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["policy"].items()}
        own = policy.state_dict()
        loaded = 0
        for k, v in sd.items():
            if k in own and own[k].shape == v.shape:
                own[k].copy_(v); loaded += 1
        print(f"  loaded {loaded}/{len(own)} tensors from step={ckpt.get('step')}")

    if args.compile:
        print("torch.compile(policy) ...")
        policy = torch.compile(policy)

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.max_steps, eta_min=args.lr * 0.1)

    csv_path = out_dir / "log.csv"
    csv_f = open(csv_path, "w")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["step", "loss", "soc", "h_cv", "lr", "n_used", "step_per_s",
                    "val_mse", "val_mae"])

    train_iter = iter(train_loader)
    t_start = time.time()
    t_last = t_start

    for step in range(args.max_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        z_groot = None
        z_fresh = None
        z_bank = None
        delta_bank = None
        if len(batch) == 8:
            # V6a: (img, wrist, state, chunk, ti, z_groot, z_bank, delta_bank)
            imgs, wrists, states, chunks, task_ids, z_groot, z_bank, delta_bank = batch
            task_ids = task_ids.to(device, non_blocking=True)
            z_groot = z_groot.to(device, non_blocking=True)
            z_bank = z_bank.to(device, non_blocking=True)
            delta_bank = delta_bank.to(device, non_blocking=True)
            if (task_ids < 0).all():
                task_ids = None
        elif len(batch) == 7:
            imgs, wrists, states, chunks, task_ids, z_groot, z_fresh = batch
            task_ids = task_ids.to(device, non_blocking=True)
            z_groot = z_groot.to(device, non_blocking=True)
            z_fresh = z_fresh.to(device, non_blocking=True)
            if (task_ids < 0).all():
                task_ids = None
        elif len(batch) == 6:
            imgs, wrists, states, chunks, task_ids, z_groot = batch
            task_ids = task_ids.to(device, non_blocking=True)
            z_groot = z_groot.to(device, non_blocking=True)
            if (task_ids < 0).all():
                task_ids = None
        elif use_lang:
            imgs, wrists, states, chunks, task_ids = batch
            task_ids = task_ids.to(device, non_blocking=True)
        else:
            imgs, wrists, states, chunks = batch
            task_ids = None

        imgs = imgs.to(device, non_blocking=True)
        wrists = wrists.to(device, non_blocking=True)
        states = states.to(device, non_blocking=True)
        chunks = chunks.to(device, non_blocking=True)

        if args.augment:
            imgs = augment_imgs(imgs, crop_pct=args.crop_pct,
                                brightness=args.aug_brightness,
                                contrast=args.aug_contrast,
                                saturation=args.aug_saturation)
            wrists = augment_imgs(wrists, crop_pct=args.crop_pct,
                                   brightness=args.aug_brightness,
                                   contrast=args.aug_contrast,
                                   saturation=args.aug_saturation)

        loss_main, info = flow_loss(policy, imgs, wrists, states, chunks,
                                     task_id=task_ids, z_groot=z_groot,
                                     z_bank=z_bank, delta_bank=delta_bank)
        soc = soc_penalty(info["h_cv"], cv_target=args.cv_target) if args.crit_lambda > 0 else torch.tensor(0.0, device=device)
        loss = loss_main + args.crit_lambda * soc

        # V5a: s2_halt consistency loss. Compare encoder output with stale vs fresh
        # z_groot. The halt head learns to predict "is fresh different from stale?".
        halt_loss = torch.tensor(0.0, device=device)
        if args.s2_halt and z_fresh is not None:
            with torch.no_grad():
                cond_fresh, _ = policy.forward_encoder(imgs, wrists, states,
                                                       task_id=task_ids, z_groot=z_fresh)
            cond_stale = info.get("cond")  # see below — flow_loss must expose this
            if cond_stale is None:
                # Fallback: re-run encoder with stale z (already used, cheap on cache miss)
                cond_stale, _ = policy.forward_encoder(imgs, wrists, states,
                                                       task_id=task_ids, z_groot=z_groot)
            divergence = ((cond_fresh - cond_stale) ** 2).mean(dim=-1)  # [B]
            halt_target = (divergence > divergence.median()).float().unsqueeze(-1)  # [B,1]
            halt_pred_logit = info["s2_halt_logit"]  # encoder must populate this
            halt_loss = F.binary_cross_entropy_with_logits(halt_pred_logit, halt_target)
            loss = loss + args.halt_lambda * halt_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        opt.step()
        scheduler.step()

        if step % args.log_every == 0:
            now = time.time()
            sps = args.log_every / (now - t_last) if step > 0 else 0
            t_last = now
            halt_str = f"  halt={halt_loss.item():.4f}" if args.s2_halt else ""
            print(f"step {step:6d}/{args.max_steps}  loss={loss.item():.5f}  "
                  f"soc={soc.item():.4f}  cv={info['h_cv'].item():.2f}  "
                  f"n_used={info['steps_mean'].item():.1f}{halt_str}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  {sps:.1f} step/s")

            val_mse, val_mae = float("nan"), float("nan")
            if step % args.eval_every == 0 and step > 0 and val_loader is not None:
                val_mse, val_mae = evaluate(policy, val_loader, device,
                                            max_batches=10, n_steps=args.infer_steps)
                print(f"  [eval] val_mse={val_mse:.5f}  val_mae={val_mae:.5f} "
                      f"(sampled with {args.infer_steps} flow steps)")

            csv_w.writerow([step, loss.item(), soc.item(), info["h_cv"].item(),
                            scheduler.get_last_lr()[0], info["steps_mean"].item(), sps,
                            val_mse, val_mae])
            csv_f.flush()

        if step > 0 and step % args.save_every == 0:
            ckpt_path = out_dir / f"step_{step:06d}.pt"
            torch.save({
                "step": step,
                "policy": policy.state_dict(),
                "opt": opt.state_dict(),
                "args": vars(args),
            }, ckpt_path)
            print(f"  saved ckpt -> {ckpt_path}")

    csv_f.close()
    print(f"done. total time: {(time.time() - t_start)/60:.1f} min")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, type=str)
    p.add_argument("--teacher_labels_dir", default="", type=str)
    p.add_argument("--dagger_data_dir", default="", type=str,
                   help="If set together with --dagger_labels_dir, mix in DAgger samples")
    p.add_argument("--dagger_labels_dir", default="", type=str)
    p.add_argument("--dagger2_data_dir", default="", type=str,
                   help="Optional second DAgger pair (e.g. for combining ensemble + apprentice data)")
    p.add_argument("--dagger2_labels_dir", default="", type=str)
    p.add_argument("--extra_teacher_dirs", default="", type=str,
                   help="Comma-separated paths to additional teacher-label datasets "
                        "(libero-*-expert-v1 directories). Each used as both data_dir "
                        "and teacher_labels_dir. z_vl/z_bank can be zeros — combine with "
                        "z_groot_drop_prob > 0 to train Liquid in both 'has-z_vl' and "
                        "'no-z_vl' regimes simultaneously (multi-suite generalization).")
    p.add_argument("--resume", default="", type=str,
                   help="Path to .pt checkpoint to warm-start from (weights only, fresh optimizer)")
    p.add_argument("--out_dir", type=str, default="/tmp/distill_groot_flow")
    p.add_argument("--max_steps", type=int, default=30000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--policy", choices=["flat", "liquid_fixed", "liquid_halt"], default="liquid_halt")
    p.add_argument("--d", type=int, default=512)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--halting_min_steps", type=int, default=4)
    p.add_argument("--action_horizon", type=int, default=16)
    p.add_argument("--img_size", type=int, default=96)
    p.add_argument("--compile", action="store_true")

    p.add_argument("--head_d", type=int, default=256)
    p.add_argument("--head_layers", type=int, default=4)
    p.add_argument("--head_heads", type=int, default=4)
    p.add_argument("--n_task_heads", type=int, default=0,
                   help="If >0, build N specialist flow heads (one per task) "
                        "sharing the encoder. Routed by task_id at fwd/sample.")
    p.add_argument("--infer_steps", type=int, default=10,
                   help="Number of ODE steps for inference sampling")

    p.add_argument("--crit_lambda", type=float, default=0.1)
    p.add_argument("--cv_target", type=float, default=0.5)

    p.add_argument("--augment", action="store_true")
    p.add_argument("--crop_pct", type=float, default=0.9)
    p.add_argument("--aug_brightness", type=float, default=0.15)
    p.add_argument("--aug_contrast", type=float, default=0.15)
    p.add_argument("--aug_saturation", type=float, default=0.15)
    p.add_argument("--n_tasks", type=int, default=0)
    p.add_argument("--d_task", type=int, default=32)
    p.add_argument("--use_z_groot", default="", type=str,
                   help="System 1/2 mode: condition Liquid encoder on GR00T's "
                        "internal state. Single: 'z_vl' / 'z_state' / 'z_motor'. "
                        "Composite (concat): 'z_vl,z_state' etc.")
    p.add_argument("--cadence_dropout", default="", type=str,
                   help="Comma-separated K-offsets for cadence dropout (chunks). "
                        "E.g. '0,1,3,7' samples z_groot from {fresh, 1-stale, 3-stale, "
                        "7-stale} per __getitem__. Forces ODE to bridge System-2 gaps.")
    p.add_argument("--gated_mixture", action="store_true",
                   help="V5b: with composite z_groot, use per-channel projections + "
                        "softmax gate over h_pre instead of single concat projection.")
    p.add_argument("--s2_halt", action="store_true",
                   help="V5a: train s2_halt_head to predict whether System-2 refresh "
                        "is needed per chunk. Adds consistency loss on (fresh, stale) "
                        "z_groot pair. Doubles encoder compute per step.")
    p.add_argument("--halt_lambda", type=float, default=0.1,
                   help="Weight on s2_halt consistency loss when --s2_halt is enabled.")
    p.add_argument("--use_query_bank", action="store_true",
                   help="V6a: bidirectional integration. Liquid emits Δs queries; "
                        "trains via soft attention over K-entry pre-computed bank "
                        "of (Δs, z_vl) pairs. Dataset must have z_vl_bank.dat / "
                        "delta_s_bank.dat from gen_groot_with_query_bank.py.")
    p.add_argument("--z_groot_drop_prob", type=float, default=0.0,
                   help="Stage 1 pressure-landscape: per-sample probability of "
                        "zeroing z_groot+bank during training. Direction-only — "
                        "deployment unchanged. Pressure for the model to act "
                        "robustly with or without System-2 input.")
    p.add_argument("--cadence_head", action="store_true",
                   help="Stage 2: add a small cadence_head to LiquidEncoder. "
                        "Outputs P(call System-2 this chunk). Phase A: head "
                        "structurally present but receives no direct supervision "
                        "(ready for Phase B RL fine-tuning).")

    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--save_every", type=int, default=5000)
    args = p.parse_args()

    train(args)


if __name__ == "__main__":
    main()
