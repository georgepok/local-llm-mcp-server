"""LIBERO behavior cloning into a Liquid student (distillation from GR00T-N1.7 dataset).

Trains a small Liquid policy (vision + state -> action chunk) on the
LeRobot-format LIBERO-10-r dataset that GR00T-N1.7-LIBERO was trained on.
Goal: match GR00T's action accuracy (MSE 0.0025 on demo) at >=10x faster
inference (target 50-100 Hz vs GR00T's 6.9 Hz on Spark).

Run on Spark (after running prep_libero.py to produce the memmap dataset):
  cd /home/pokazge/liquid-arc/research/self_org_sim
  source /home/pokazge/Isaac-GR00T/.venv/bin/activate
  python distill_groot.py \\
    --data_dir /home/pokazge/datasets/libero-10-r-decoded \\
    --out_dir /tmp/distill_groot_v1 \\
    --max_steps 30000 --batch_size 256 \\
    --policy liquid_halt --d 256 --k 16 \\
    --action_horizon 16 --compile
"""

from __future__ import annotations

import argparse
import csv
import functools
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

print = functools.partial(print, flush=True)
torch.set_float32_matmul_precision("high")


# ---------------------------------------------------------------------------
# Memmap-backed dataset (no JPEG decode in hot path; produced by prep_libero.py)
# ---------------------------------------------------------------------------

class LiberoMemmapDataset(Dataset):
    """Map-style dataset over pre-decoded memmap files.

    Yields (img, wrist_img, state, action_chunk[K, 7]) per index.
    Indices restricted to (frame_index, episode_index) pairs where the chunk
    fits within the episode (or padding is permitted).
    """

    def __init__(
        self,
        data_dir: Path,
        action_horizon: int = 16,
        episode_indices: list[int] | None = None,
        return_task_id: bool = False,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.action_horizon = action_horizon
        self.return_task_id = return_task_id

        idx = np.load(self.data_dir / "index.npz")
        self.starts = idx["episode_starts"]
        self.lengths = idx["episode_lengths"]
        self.task_indices = idx["task_indices"]
        self.n_total = int(idx["n_total"])
        self.img_size = int(idx["img_size"])

        if episode_indices is None:
            episode_indices = list(range(len(self.lengths)))
        self.episode_indices = np.asarray(episode_indices, dtype=np.int64)

        # Build flat (episode, t) indexing
        sample_ep, sample_t = [], []
        for ep_i in self.episode_indices:
            n = int(self.lengths[ep_i])
            sample_ep.extend([ep_i] * n)
            sample_t.extend(range(n))
        self.sample_ep = np.asarray(sample_ep, dtype=np.int64)
        self.sample_t = np.asarray(sample_t, dtype=np.int64)

        self.imgs = np.memmap(
            self.data_dir / "imgs.dat", dtype=np.uint8, mode="r",
            shape=(self.n_total, self.img_size, self.img_size, 3),
        )
        self.wrists = np.memmap(
            self.data_dir / "wrists.dat", dtype=np.uint8, mode="r",
            shape=(self.n_total, self.img_size, self.img_size, 3),
        )
        self.states = np.memmap(
            self.data_dir / "states.dat", dtype=np.float32, mode="r",
            shape=(self.n_total, 8),
        )
        self.actions = np.memmap(
            self.data_dir / "actions.dat", dtype=np.float32, mode="r",
            shape=(self.n_total, 7),
        )
        print(f"LiberoMemmapDataset: {len(self.episode_indices)} episodes, "
              f"{len(self.sample_ep):,} samples, img_size={self.img_size}")

    def __len__(self):
        return len(self.sample_ep)

    def _build_chunk(self, ep_start: int, ep_len: int, t: int) -> np.ndarray:
        K = self.action_horizon
        end = min(t + K, ep_len)
        chunk = np.array(self.actions[ep_start + t:ep_start + end])
        if len(chunk) < K:
            pad = np.tile(chunk[-1:], (K - len(chunk), 1))
            chunk = np.concatenate([chunk, pad], axis=0)
        return chunk

    def __getitem__(self, idx):
        ep_i = int(self.sample_ep[idx])
        t = int(self.sample_t[idx])
        ep_start = int(self.starts[ep_i])
        ep_len = int(self.lengths[ep_i])
        global_idx = ep_start + t
        img = np.array(self.imgs[global_idx])
        wrist = np.array(self.wrists[global_idx])
        state = np.array(self.states[global_idx])
        chunk = self._build_chunk(ep_start, ep_len, t)
        if self.return_task_id:
            task_id = int(self.task_indices[ep_i])
            return img, wrist, state, chunk, task_id
        return img, wrist, state, chunk


def collate_batch(batch):
    n_fields = len(batch[0])
    imgs = torch.from_numpy(np.stack([b[0] for b in batch])).float() / 255.0
    wrists = torch.from_numpy(np.stack([b[1] for b in batch])).float() / 255.0
    states = torch.from_numpy(np.stack([b[2] for b in batch]))
    chunks = torch.from_numpy(np.stack([b[3] for b in batch]))
    imgs = imgs.permute(0, 3, 1, 2).contiguous()
    wrists = wrists.permute(0, 3, 1, 2).contiguous()
    if n_fields == 8:
        # V6a: (img, wrist, state, chunk, ti, z_groot, z_bank, delta_bank)
        task_ids = torch.tensor([b[4] for b in batch], dtype=torch.long)
        z_groot = torch.from_numpy(np.stack([b[5] for b in batch])).float()
        z_bank = torch.from_numpy(np.stack([b[6] for b in batch])).float()
        delta_bank = torch.from_numpy(np.stack([b[7] for b in batch])).float()
        return imgs, wrists, states, chunks, task_ids, z_groot, z_bank, delta_bank
    if n_fields == 7:
        # V5a: (img, wrist, state, chunk, ti, z_stale, z_fresh)
        task_ids = torch.tensor([b[4] for b in batch], dtype=torch.long)
        z_stale = torch.from_numpy(np.stack([b[5] for b in batch])).float()
        z_fresh = torch.from_numpy(np.stack([b[6] for b in batch])).float()
        return imgs, wrists, states, chunks, task_ids, z_stale, z_fresh
    if n_fields == 6:
        # (img, wrist, state, chunk, task_id_or_neg1, z_groot) — System 1/2 path
        task_ids = torch.tensor([b[4] for b in batch], dtype=torch.long)
        z_groot = torch.from_numpy(np.stack([b[5] for b in batch])).float()
        return imgs, wrists, states, chunks, task_ids, z_groot
    if n_fields == 5:
        task_ids = torch.tensor([b[4] for b in batch], dtype=torch.long)
        return imgs, wrists, states, chunks, task_ids
    return imgs, wrists, states, chunks


class TeacherLabelDataset(Dataset):
    """Dataset over (obs, teacher_action_chunk) pairs.

    Reuses the LiberoMemmapDataset's image/state memmaps, replaces the
    demonstrator action chunks with chunks predicted by GR00T (saved by
    gen_groot_labels.py).
    """

    def __init__(self, decoded_dir: Path, teacher_labels_dir: Path,
                 action_horizon: int = 16, return_task_id: bool = False,
                 target_img_size: int = 0, return_z_groot: str = "",
                 cadence_dropout=None, return_z_fresh: bool = False,
                 use_query_bank: bool = False,
                 z_groot_drop_prob: float = 0.0):
        super().__init__()
        decoded_dir = Path(decoded_dir)
        teacher_labels_dir = Path(teacher_labels_dir)
        self.action_horizon = action_horizon
        self.return_task_id = return_task_id
        # return_z_groot: "" disables; otherwise one of "z_vl", "z_state", "z_motor".
        # When enabled, the dataset emits an extra trailing field with the named
        # GR00T internal-state vector. Source dir must contain {name}.dat written
        # by gen_groot_with_states.py.
        self.return_z_groot = return_z_groot
        # cadence_dropout: list of K-offsets in chunks. Each __getitem__ samples
        # one K uniformly; the obs/chunk are from the requested step but z_groot
        # is fetched from K chunks earlier in the same episode (clipped to
        # episode start). [0] disables (always fresh). [0,1,3,7] simulates the
        # K∈{1,2,4,8} deployment cadences. Forces ODE to bridge System-2 gaps.
        self.cadence_dropout = cadence_dropout if cadence_dropout else None
        # return_z_fresh: V5a — also emit the (no-dropout) fresh z_groot so the
        # train loop can compute fresh-vs-stale divergence for s2_halt training.
        self.return_z_fresh = return_z_fresh
        # use_query_bank: V6a — load z_vl_bank.dat (K alternative z's per sample
        # from perturbed GR00T queries) plus delta_s_bank.dat. Liquid then learns
        # to emit a query Δs that selects via soft attention over the bank.
        self.use_query_bank = use_query_bank
        self.query_bank_K = 0
        self.z_vl_bank_mm = None
        self.delta_bank_mm = None
        # Stage 1 pressure-landscape: with probability p, zero out z_groot+bank
        # for a sample. Trains the model to be robust to System-2 absence
        # without runtime controllers — pressure-only direction.
        self.z_groot_drop_prob = z_groot_drop_prob

        # Load decoded dataset memmaps for inputs
        idx = np.load(decoded_dir / "index.npz")
        self.starts = idx["episode_starts"]
        self.lengths = idx["episode_lengths"]
        self.task_indices = idx["task_indices"]
        self.n_total = int(idx["n_total"])
        self.img_size = int(idx["img_size"])
        # Optional on-the-fly resize: if target_img_size > 0 and != stored img_size,
        # imgs are resized in __getitem__. Used to mix multi-resolution datasets
        # (e.g., 96x96 demos + 256x256 dagger collection).
        self.target_img_size = target_img_size if target_img_size > 0 else self.img_size
        self.imgs = np.memmap(decoded_dir / "imgs.dat", dtype=np.uint8, mode="r",
                              shape=(self.n_total, self.img_size, self.img_size, 3))
        self.wrists = np.memmap(decoded_dir / "wrists.dat", dtype=np.uint8, mode="r",
                                shape=(self.n_total, self.img_size, self.img_size, 3))
        self.states = np.memmap(decoded_dir / "states.dat", dtype=np.float32, mode="r",
                                shape=(self.n_total, 8))

        # Load teacher labels memmap + sample index
        lbl_idx = np.load(teacher_labels_dir / "labels_index.npz")
        self.sample_idx = lbl_idx["sample_idx"]   # [N, 3] = (ep_i, t, task_idx)
        self.n_samples = int(lbl_idx["n_samples"])
        K = int(lbl_idx["action_horizon"])
        assert K == action_horizon, f"label horizon {K} != requested {action_horizon}"

        # Auto-detect ensemble vs single-sample format. Ensemble has
        # `ensemble_k` in index and a different filename.
        ensemble_path = teacher_labels_dir / "teacher_chunks_ensemble.dat"
        if "ensemble_k" in lbl_idx and ensemble_path.exists():
            self.ensemble_k = int(lbl_idx["ensemble_k"])
            self.teacher_chunks = np.memmap(
                ensemble_path, dtype=np.float32, mode="r",
                shape=(self.n_samples, self.ensemble_k, action_horizon, 7),
            )
            print(f"TeacherLabelDataset (ensemble): {self.n_samples:,} obs × "
                  f"{self.ensemble_k} samples each, img_size={self.img_size}")
        else:
            self.ensemble_k = 1
            self.teacher_chunks = np.memmap(
                teacher_labels_dir / "teacher_chunks.dat", dtype=np.float32, mode="r",
                shape=(self.n_samples, action_horizon, 7),
            )
            print(f"TeacherLabelDataset: {self.n_samples:,} (obs, teacher_chunk) pairs, "
                  f"img_size={self.img_size}")

        if self.return_z_groot:
            # Comma-separated allows composite mode: "z_vl,z_state" concatenates
            # both channels into one z_groot vector. Single channel is the legacy
            # behavior. Order in the list determines concat order.
            channels = [c.strip() for c in self.return_z_groot.split(",") if c.strip()]
            for c in channels:
                assert c in ("z_vl", "z_state", "z_motor"), \
                    f"channel '{c}' not in 'z_vl', 'z_state', 'z_motor'"
            top_idx = np.load(decoded_dir / "index.npz")
            self.z_channels = channels
            self.z_mms = []
            self.z_dims = []
            for c in channels:
                z_path = teacher_labels_dir / f"{c}.dat"
                assert z_path.exists(), (
                    f"z_groot file missing: {z_path}. "
                    "Generate dataset with gen_groot_with_states.py."
                )
                if c == "z_motor":
                    dim = 7
                elif c == "z_vl":
                    dim = int(top_idx["z_vl_dim"])
                else:
                    dim = int(top_idx["z_state_dim"])
                mm = np.memmap(z_path, dtype=np.float32, mode="r",
                               shape=(self.n_samples, dim))
                self.z_mms.append(mm)
                self.z_dims.append(dim)
            self.z_groot_dim = sum(self.z_dims)
            self.z_groot_mm = self.z_mms[0] if len(self.z_mms) == 1 else None
            print(f"  + z_groot: {self.return_z_groot} dim={self.z_groot_dim}"
                  + (f" (composite: {list(zip(channels, self.z_dims))})" if len(channels) > 1 else ""))
        else:
            self.z_channels = []
            self.z_mms = []
            self.z_dims = []
            self.z_groot_mm = None
            self.z_groot_dim = 0

        if self.use_query_bank:
            top_idx = np.load(decoded_dir / "index.npz")
            assert "query_bank_K" in top_idx, (
                "use_query_bank requires dataset generated by gen_groot_with_query_bank.py "
                "or gen_groot_with_image_queries.py"
            )
            self.query_bank_K = int(top_idx["query_bank_K"])
            zvl_dim = int(top_idx["z_vl_dim"])
            # query_dim auto-detected: 8 for V6a (state), G*G for V6c (image)
            self.query_channel = str(top_idx["query_channel"]) if "query_channel" in top_idx else "state"
            self.query_dim = int(top_idx["query_dim"]) if "query_dim" in top_idx else 8
            self.z_vl_bank_mm = np.memmap(
                teacher_labels_dir / "z_vl_bank.dat", dtype=np.float32, mode="r",
                shape=(self.n_samples, self.query_bank_K, zvl_dim),
            )
            self.delta_bank_mm = np.memmap(
                teacher_labels_dir / "delta_s_bank.dat", dtype=np.float32, mode="r",
                shape=(self.n_samples, self.query_bank_K, self.query_dim),
            )
            print(f"  + query_bank: K={self.query_bank_K}, channel={self.query_channel}, "
                  f"query_dim={self.query_dim}, z_vl_bank dim={zvl_dim}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        from PIL import Image
        ep_i, t, ti = self.sample_idx[idx]
        ep_i = int(ep_i); t = int(t); ti = int(ti)
        global_idx = int(self.starts[ep_i]) + t
        img = np.array(self.imgs[global_idx])
        wrist = np.array(self.wrists[global_idx])
        if self.target_img_size != self.img_size:
            img = np.array(Image.fromarray(img).resize(
                (self.target_img_size, self.target_img_size)), dtype=np.uint8)
            wrist = np.array(Image.fromarray(wrist).resize(
                (self.target_img_size, self.target_img_size)), dtype=np.uint8)
        state = np.array(self.states[global_idx])
        # For ensemble datasets, randomly sample one of K teacher chunks per call.
        if self.ensemble_k > 1:
            k = int(np.random.randint(0, self.ensemble_k))
            chunk = np.array(self.teacher_chunks[idx, k])
        else:
            chunk = np.array(self.teacher_chunks[idx])
        if self.z_channels:
            if self.cadence_dropout:
                k_offset = int(np.random.choice(self.cadence_dropout))
                stale_t = max(0, t - k_offset)
                z_idx = int(self.starts[ep_i]) + stale_t
            else:
                z_idx = idx

            def _read_z(zi):
                if len(self.z_mms) == 1:
                    return np.array(self.z_mms[0][zi])
                return np.concatenate(
                    [np.array(mm[zi]) for mm in self.z_mms]
                ).astype(np.float32)

            z_groot = _read_z(z_idx)
            # Stage 1 z_groot dropout: pressure for "act without System-2 input"
            drop_z = (self.z_groot_drop_prob > 0
                      and np.random.random() < self.z_groot_drop_prob)
            if drop_z:
                z_groot = np.zeros_like(z_groot)
            ti_field = ti if self.return_task_id else -1
            if self.use_query_bank and self.z_vl_bank_mm is not None:
                # 8-tuple: (img, wrist, state, chunk, ti, z_groot, z_bank, delta_bank)
                # Bank is read from the SAME stale index as z_groot so cadence
                # dropout makes the bank stale too. Without this, V7b/V6d-class
                # variants can't actually learn cadence robustness — they always
                # see a current-chunk bank during training, then OOD at K>1.
                z_bank = np.array(self.z_vl_bank_mm[z_idx])  # [K, dim]
                delta_bank = np.array(self.delta_bank_mm[z_idx])  # [K, query_dim]
                if drop_z:
                    z_bank = np.zeros_like(z_bank)
                return img, wrist, state, chunk, ti_field, z_groot, z_bank, delta_bank
            if self.return_z_fresh:
                z_fresh = _read_z(idx)  # fresh z (no dropout shift)
                return img, wrist, state, chunk, ti_field, z_groot, z_fresh
            return img, wrist, state, chunk, ti_field, z_groot
        if self.return_task_id:
            return img, wrist, state, chunk, ti
        return img, wrist, state, chunk


# ---------------------------------------------------------------------------
# GPU augmentation (no-op outside training; applied after CPU->GPU transfer)
# ---------------------------------------------------------------------------

def random_crop_resize(imgs: torch.Tensor, crop_pct: float = 0.85) -> torch.Tensor:
    """Random crop a (1-crop_pct) margin and resize back to original size.

    imgs: [B, C, H, W] float in [0, 1]. Returns same shape.
    """
    B, C, H, W = imgs.shape
    cH = int(H * crop_pct)
    cW = int(W * crop_pct)
    # One crop offset per batch element (different views per sample)
    top = torch.randint(0, H - cH + 1, (B,), device=imgs.device)
    left = torch.randint(0, W - cW + 1, (B,), device=imgs.device)
    out = torch.empty_like(imgs)
    for i in range(B):
        crop = imgs[i:i + 1, :, top[i]:top[i] + cH, left[i]:left[i] + cW]
        out[i:i + 1] = F.interpolate(crop, size=(H, W), mode="bilinear", align_corners=False)
    return out


def color_jitter_gpu(imgs: torch.Tensor,
                      brightness: float = 0.2,
                      contrast: float = 0.2,
                      saturation: float = 0.2) -> torch.Tensor:
    """Per-batch random color jitter on GPU. imgs in [0,1]."""
    B = imgs.shape[0]
    dev = imgs.device
    # brightness: x = x + b, b ~ U[-brightness, brightness]
    b = (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * brightness
    out = imgs + b
    # contrast: x = (x - mean) * c + mean, c ~ U[1-contrast, 1+contrast]
    c = 1.0 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * contrast
    mean = out.mean(dim=(2, 3), keepdim=True)
    out = (out - mean) * c + mean
    # saturation: blend with grayscale
    s = 1.0 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * saturation
    gray = out.mean(dim=1, keepdim=True).expand_as(out)
    out = gray + (out - gray) * s
    return out.clamp(0.0, 1.0)


def augment_imgs(imgs: torch.Tensor, crop_pct: float = 0.85,
                 brightness: float = 0.2, contrast: float = 0.2,
                 saturation: float = 0.2) -> torch.Tensor:
    out = random_crop_resize(imgs, crop_pct=crop_pct)
    out = color_jitter_gpu(out, brightness=brightness, contrast=contrast, saturation=saturation)
    return out


# ---------------------------------------------------------------------------
# Vision encoder — small CNN (NatureCNN-style)
# ---------------------------------------------------------------------------

class VisionEncoder(nn.Module):
    """Small CNN -> d_out feature vector. Resolution-independent param count via
    AdaptiveAvgPool2d before flatten — the FC layer always sees a fixed 8x8x64
    spatial summary regardless of input image size. Means we can train at any
    resolution without parameter explosion in the projection layer."""

    def __init__(self, in_ch: int = 3, d_out: int = 128, img_size: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=8, stride=4),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((8, 8)),  # ← pool to fixed 8×8 regardless of input size
            nn.Flatten(),
        )
        conv_dim = 8 * 8 * 64  # 4096, constant
        self.fc = nn.Linear(conv_dim, d_out)

    def forward(self, x):
        return F.silu(self.fc(self.net(x)))


# ---------------------------------------------------------------------------
# Liquid policy — encoder -> LiquidLayer -> action chunk head
# ---------------------------------------------------------------------------

class LiquidStudent(nn.Module):
    """vision + state -> Liquid ODE -> action chunk (action_horizon × 7)."""

    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 7,
        action_horizon: int = 16,
        d: int = 128,
        d_vis: int = 128,
        img_size: int = 96,
        k_max: int = 16,
        halt_mode: str = "learned",  # "none" | "learned"
        min_steps: int = 4,
        dt: float = 0.5,
        n_tasks: int = 0,        # 0 = no language conditioning
        d_task: int = 32,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.k_max = k_max
        self.halt_mode = halt_mode
        self.min_steps = min_steps
        self.dt = dt
        self.n_tasks = n_tasks

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

        # Liquid ODE drift module (zero-init last layer for stable bootstrap)
        self.drift = nn.Sequential(
            nn.Linear(d, d * 2), nn.SiLU(),
            nn.Linear(d * 2, d),
        )
        nn.init.zeros_(self.drift[-1].weight)
        nn.init.zeros_(self.drift[-1].bias)

        # Per-channel adaptive timescale
        self.tau_raw = nn.Parameter(torch.zeros(d))

        if halt_mode == "learned":
            self.halt_head = nn.Linear(d, 1)
            with torch.no_grad():
                self.halt_head.bias.fill_(-3.0)  # bias toward continuing initially

        # Action chunk head: emits all `action_horizon` future actions at once
        self.action_head = nn.Linear(d, action_horizon * action_dim)

    def encode(self, img, wrist_img, state, task_id=None):
        v = self.vis_enc(img)
        w = self.wrist_enc(wrist_img)
        s = self.state_enc(state)
        feats = [v, w, s]
        if self.task_emb is not None:
            if task_id is None:
                # No task signal at inference: zero embedding
                t = torch.zeros(v.shape[0], self.task_emb.embedding_dim, device=v.device)
            else:
                t = self.task_emb(task_id)
            feats.append(t)
        return F.silu(self.fuse(torch.cat(feats, dim=-1)))

    def forward(self, img, wrist_img, state, task_id=None):
        h = self.encode(img, wrist_img, state, task_id=task_id)
        B = h.shape[0]
        tau = F.softplus(self.tau_raw) + 0.1
        steps_used = torch.zeros(B, 1, device=h.device)
        still_active = torch.ones(B, 1, device=h.device)
        for k in range(self.k_max):
            dh = self.drift(h) / tau
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
        actions_flat = self.action_head(h)
        actions = actions_flat.view(B, self.action_horizon, self.action_dim)
        h_norms = h.norm(dim=-1)  # [B]
        h_cv = h_norms.std() / (h_norms.mean() + 1e-8)
        info = {
            "steps_mean": steps_used.mean().detach(),
            "steps_min": steps_used.min().detach(),
            "steps_max": steps_used.max().detach(),
            "h_cv": h_cv,
            "h_norm_mean": h_norms.mean().detach(),
        }
        return actions, info


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def chunk_loss(pred: torch.Tensor, target: torch.Tensor, weighting: str = "exp") -> torch.Tensor:
    """MSE on action chunks with optional position weighting.

    pred, target: [B, K, A]
    weighting: "uniform" | "linear" | "exp" (decay over chunk position)
    """
    K = pred.shape[1]
    if weighting == "uniform":
        w = torch.ones(K, device=pred.device)
    elif weighting == "linear":
        w = torch.linspace(1.0, 0.5, K, device=pred.device)
    elif weighting == "exp":
        w = torch.exp(-torch.arange(K, device=pred.device).float() / (K * 0.5))
    else:
        raise ValueError(weighting)
    w = w / w.sum() * K  # normalize so sum = K (same as uniform)
    sq = (pred - target).pow(2).mean(-1)  # [B, K]
    return (sq * w).mean()


def soc_penalty(h_cv: torch.Tensor, cv_target: float = 0.5) -> torch.Tensor:
    """Asymmetric SoC penalty: only push CV up when below target.

    h_cv is std/mean of per-sample hidden-state norms across batch (always >=0).
    Penalty is (target - cv).clamp(min=0)^2 — quiet when cv >= target.
    """
    return (cv_target - h_cv).clamp(min=0).pow(2)


def train(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir = {out_dir}")
    print(f"args: {vars(args)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train/val split: last 10% of episodes held out
    idx = np.load(Path(args.data_dir) / "index.npz")
    n_eps = len(idx["episode_lengths"])
    n_val = max(10, n_eps // 10)
    val_indices = list(range(n_eps - n_val, n_eps))
    train_indices = list(range(n_eps - n_val))
    print(f"train episodes: {len(train_indices)}, val episodes: {len(val_indices)}")

    use_lang = args.n_tasks > 0
    if args.teacher_labels_dir:
        train_ds = TeacherLabelDataset(
            args.data_dir, args.teacher_labels_dir,
            action_horizon=args.action_horizon, return_task_id=use_lang,
        )
        # Always evaluate on demo labels (the "ground truth" we ultimately care about)
        val_ds = LiberoMemmapDataset(args.data_dir, args.action_horizon, val_indices,
                                      return_task_id=use_lang)
    else:
        train_ds = LiberoMemmapDataset(args.data_dir, args.action_horizon, train_indices,
                                        return_task_id=use_lang)
        val_ds = LiberoMemmapDataset(args.data_dir, args.action_horizon, val_indices,
                                      return_task_id=use_lang)
    args.img_size = train_ds.img_size
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=True,
        prefetch_factor=4 if args.num_workers else None,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_batch,
        pin_memory=True,
    )

    # Build model
    halt_mode = "learned" if args.policy == "liquid_halt" else "none"
    policy = LiquidStudent(
        state_dim=8,
        action_dim=7,
        action_horizon=args.action_horizon,
        d=args.d,
        d_vis=args.d,
        img_size=args.img_size,
        k_max=args.k,
        halt_mode=halt_mode,
        min_steps=args.halting_min_steps,
        n_tasks=args.n_tasks,
        d_task=args.d_task,
    ).to(device)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"policy params: {n_params/1e6:.2f}M ({n_params:,})")

    if args.compile:
        print("torch.compile(policy) ...")
        policy = torch.compile(policy)

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_steps, eta_min=args.lr * 0.1)

    # Logging
    csv_path = out_dir / "log.csv"
    csv_f = open(csv_path, "w")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["step", "loss", "mse", "soc", "h_cv", "lr", "n_used", "step_per_s", "val_mse", "val_mae"])

    train_iter = iter(train_loader)
    t_start = time.time()
    t_last = t_start

    for step in range(args.max_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        if use_lang:
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

        pred, info = policy(imgs, wrists, states, task_id=task_ids)
        mse = chunk_loss(pred, chunks, weighting=args.chunk_weighting)
        soc = soc_penalty(info["h_cv"], cv_target=args.cv_target) if args.crit_lambda > 0 else torch.tensor(0.0, device=device)
        loss = mse + args.crit_lambda * soc

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        opt.step()
        scheduler.step()

        if step % args.log_every == 0:
            now = time.time()
            sps = args.log_every / (now - t_last) if step > 0 else 0
            t_last = now
            print(f"step {step:6d}/{args.max_steps}  loss={loss.item():.5f}  mse={mse.item():.5f}  soc={soc.item():.4f}  "
                  f"cv={info['h_cv'].item():.2f}  n_used={info['steps_mean'].item():.1f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  {sps:.1f} step/s")

            val_mse, val_mae = float("nan"), float("nan")
            if step % args.eval_every == 0 and step > 0:
                val_mse, val_mae = evaluate(policy, val_loader, device, max_batches=20)
                print(f"  [eval] val_mse={val_mse:.5f}  val_mae={val_mae:.5f}")

            csv_w.writerow([step, loss.item(), mse.item(), soc.item(), info["h_cv"].item(),
                            scheduler.get_last_lr()[0], info["steps_mean"].item(), sps, val_mse, val_mae])
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


@torch.no_grad()
def evaluate(policy, val_loader, device, max_batches: int = 20):
    policy.eval()
    sq_total = 0.0
    abs_total = 0.0
    n = 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        if len(batch) == 5:
            imgs, wrists, states, chunks, task_ids = batch
            task_ids = task_ids.to(device)
        else:
            imgs, wrists, states, chunks = batch
            task_ids = None
        imgs = imgs.to(device)
        wrists = wrists.to(device)
        states = states.to(device)
        chunks = chunks.to(device)
        pred, _ = policy(imgs, wrists, states, task_id=task_ids)
        diff = pred - chunks
        sq_total += diff.pow(2).mean().item() * imgs.shape[0]
        abs_total += diff.abs().mean().item() * imgs.shape[0]
        n += imgs.shape[0]
    policy.train()
    return sq_total / max(n, 1), abs_total / max(n, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True,
                   help="Path to memmap dataset produced by prep_libero.py")
    p.add_argument("--out_dir", type=str, default="/tmp/distill_groot")
    p.add_argument("--max_steps", type=int, default=30000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--policy", choices=["flat", "liquid_fixed", "liquid_halt"], default="liquid_halt")
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--halting_min_steps", type=int, default=4)
    p.add_argument("--action_horizon", type=int, default=16)
    p.add_argument("--img_size", type=int, default=96, help="Inferred from dataset; ignored.")
    p.add_argument("--compile", action="store_true", help="torch.compile(policy)")

    p.add_argument("--chunk_weighting", choices=["uniform", "linear", "exp"], default="exp")
    p.add_argument("--crit_lambda", type=float, default=0.1)
    p.add_argument("--cv_target", type=float, default=0.5)

    # v2 features
    p.add_argument("--augment", action="store_true", help="Enable random crop + color jitter on training images")
    p.add_argument("--crop_pct", type=float, default=0.9)
    p.add_argument("--aug_brightness", type=float, default=0.15)
    p.add_argument("--aug_contrast", type=float, default=0.15)
    p.add_argument("--aug_saturation", type=float, default=0.15)
    p.add_argument("--n_tasks", type=int, default=0,
                   help="If >0, enable language conditioning via task_id embedding (0 = disabled)")
    p.add_argument("--d_task", type=int, default=32)
    p.add_argument("--teacher_labels_dir", type=str, default="",
                   help="If set, train on (obs, GR00T_action_chunk) pairs from this dir. "
                        "Validation still uses demo labels.")

    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--save_every", type=int, default=5000)
    args = p.parse_args()

    train(args)


if __name__ == "__main__":
    main()
