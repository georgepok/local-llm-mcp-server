"""v11 episodic memory retrieval: kNN over DINOv2 features.

Stand-alone module imported by rollout_libero_s1s2_retrieval.py.

Bank format (.npz produced by build_memory_bank_v11.py):
  features    [N, 776]  L2-normalized (DINOv2(agent) || DINOv2(wrist) || state[8])
  actions     [N, K, A] teacher action chunks
  suite_idx   [N] int   (0=libero_10, 1=libero_goal, 2=libero_object, 3=libero_spatial)
  task        [N] int   per-suite task index
  ep, t       [N] int   trajectory provenance
  success     [N] int   1 if episode succeeded, 0 if failed, -1 if unknown

Retrieval feature is RAW DINOv2 CLS (pre-projection) so the bank is
student-independent: the same bank works with any future LIBERO student
that uses DINOv2-small as its frozen vision backbone.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SUITE_TO_IDX = {
    "libero_10": 0,
    "libero_goal": 1,
    "libero_object": 2,
    "libero_spatial": 3,
}


def load_dinov2_for_retrieval(device):
    """Load frozen DINOv2-small. Same backbone the LIBERO student uses."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14", verbose=False
        )
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval().to(device)
    return backbone


class RetrievalBank:
    """Episodic memory bank with optional per-suite filtering.

    Args:
        bank_path: path to .npz bank produced by build_memory_bank_v11.py
        device: torch.device for DINOv2 inference
        top_k: how many nearest neighbors to retrieve
        alpha_base: max retrieval weight when sim is high
        adaptive_alpha: if True, scale alpha by mean retrieval similarity
        filter_suite: if not None, restrict retrieval to this suite name
        success_only: if True, drop failed-episode entries at load time
        softmax_temperature: temperature for weighted-mean of retrieved chunks
    """

    def __init__(
        self,
        bank_path: str,
        device,
        top_k: int = 3,
        alpha_base: float = 0.5,
        adaptive_alpha: bool = True,
        filter_suite: Optional[str] = None,
        success_only: bool = False,
        softmax_temperature: float = 8.0,
        load_z_bank_z_state: bool = False,
        dataset_root: str = "/home/pokazge/datasets",
    ):
        self.device = device
        self.top_k = top_k
        self.alpha_base = alpha_base
        self.adaptive_alpha = adaptive_alpha
        self.softmax_temperature = softmax_temperature
        self.load_z_bank_z_state = load_z_bank_z_state

        bank_p = Path(bank_path)
        print(f"[retrieval] loading bank from {bank_p}")
        bank = np.load(bank_p, allow_pickle=True)
        feats = bank["features"]      # [N, d] L2-normalized
        actions = bank["actions"]     # [N, K, A]
        suite_idx = bank["suite_idx"]
        success = bank["success"]
        task = bank["task"]

        # BEFORE filter: compute per-suite sample-position for z_bank lookup.
        # Within each suite, bank entries are in the same order as per-suite
        # samples (see build_memory_bank_v11.py process_suite()).
        per_suite_pos_full = np.zeros(len(feats), dtype=np.int64)
        for s_int in range(len(SUITE_TO_IDX)):
            mask_s = suite_idx == s_int
            indices = np.where(mask_s)[0]
            per_suite_pos_full[indices] = np.arange(len(indices))

        mask = np.ones(len(feats), dtype=bool)
        if success_only:
            mask &= (success > 0)
        if filter_suite is not None:
            target_idx = SUITE_TO_IDX[filter_suite]
            mask &= (suite_idx == target_idx)

        kept = int(mask.sum())
        feats = feats[mask]
        actions = actions[mask]
        suite_idx = suite_idx[mask]
        task = task[mask]
        success = success[mask]
        per_suite_pos = per_suite_pos_full[mask]

        # Move to device for fast matmul retrieval
        self.features_gpu = torch.from_numpy(feats).to(device).float()  # [N, d]
        self.actions = actions.astype(np.float32)  # keep on CPU
        self.suite_idx = suite_idx.astype(np.int64)
        self.task = task.astype(np.int64)
        self.success = success.astype(np.int64)
        self.per_suite_pos = per_suite_pos.astype(np.int64)
        self.N = kept
        self.d_feat = feats.shape[1]
        self.K = actions.shape[1]
        self.A = actions.shape[2]

        # Optional: lazy-load per-suite z_vl_bank.dat + z_state.dat memmaps for
        # substrate-input retrieval (v15-zretrieve). At inference we look up
        # top-K nearest training z_banks to project live GR00T z_bank onto the
        # training manifold (mitigates compound-error distribution shift).
        self.zbank_memmaps = None
        self.zstate_memmaps = None
        if load_z_bank_z_state:
            print(f"[retrieval] loading per-suite z_vl_bank.dat + z_state.dat memmaps for substrate retrieval")
            self.zbank_memmaps = {}
            self.zstate_memmaps = {}
            for s_name, s_int in SUITE_TO_IDX.items():
                d = Path(dataset_root) / f"libero-{s_name.split('_')[-1]}-expert-v1"
                if not d.exists():
                    continue
                # z_vl_bank.dat: [n_samples, K_bank=4, hidden=1024]
                zb_path = d / "z_vl_bank.dat"
                zs_path = d / "z_state.dat"
                if not (zb_path.exists() and zs_path.exists()):
                    print(f"  [skip] {s_name}: missing z_vl_bank.dat or z_state.dat")
                    continue
                zb_bytes = zb_path.stat().st_size
                n_samples = zb_bytes // (4 * 1024 * 4)  # 4 bytes/float × K_bank × hidden
                zs_bytes = zs_path.stat().st_size
                n_samples_zs = zs_bytes // (1536 * 4)
                if n_samples != n_samples_zs:
                    print(f"  [warn] {s_name}: n_samples mismatch zbank={n_samples} zstate={n_samples_zs}")
                self.zbank_memmaps[s_int] = np.memmap(
                    zb_path, dtype=np.float32, mode="r",
                    shape=(n_samples, 4, 1024),
                )
                self.zstate_memmaps[s_int] = np.memmap(
                    zs_path, dtype=np.float32, mode="r",
                    shape=(n_samples_zs, 1536),
                )
                print(f"  [load] {s_name}: n_samples={n_samples}, zbank+zstate memmaps ready")

        # Diagnostics
        suite_counts = {n: int((suite_idx == i).sum()) for n, i in SUITE_TO_IDX.items()}
        print(f"[retrieval] bank loaded: N={kept}, d={self.d_feat}, K={self.K}, A={self.A}")
        print(f"[retrieval] per-suite counts: {suite_counts}")
        print(f"[retrieval] top_k={top_k} alpha_base={alpha_base} "
              f"adaptive={adaptive_alpha} filter={filter_suite} success_only={success_only}")

        # DINOv2 for query encoding
        print(f"[retrieval] loading DINOv2-small for query encoding...")
        self.dino = load_dinov2_for_retrieval(device)
        self._mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(device)
        self._std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(device)

    @torch.no_grad()
    def encode_query(self, img_raw, wrist_raw, state8) -> np.ndarray:
        """Encode an observation into the bank's feature space.

        img_raw, wrist_raw: [H, W, 3] uint8 (raw, e.g. 256×256 from sim)
        state8: [8] float32

        Returns: [d_feat] L2-normalized numpy array
        """
        # Stack both cameras into one batch of 2
        imgs = np.stack([img_raw, wrist_raw])  # [2, H, W, 3]
        x = torch.from_numpy(imgs).to(self.device).float().permute(0, 3, 1, 2) / 255.0
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self._mean) / self._std
        feats = self.dino(x)  # [2, 384]
        agent_feat, wrist_feat = feats[0], feats[1]
        state_t = torch.from_numpy(np.asarray(state8, dtype=np.float32)).to(self.device)
        full = torch.cat([agent_feat, wrist_feat, state_t], dim=-1)  # [776]
        full = full / (full.norm() + 1e-8)
        return full

    @torch.no_grad()
    def retrieve(self, query_t):
        """Top-k retrieval. query_t: [d_feat] GPU tensor (normalized).

        Returns:
            blended_chunk [K, A] numpy — weighted mean of top-k action chunks
            diagnostics dict with top_sims, top_meta, weights
        """
        # Cosine sim = dot product (both normalized)
        sims = self.features_gpu @ query_t  # [N]
        topk = torch.topk(sims, k=self.top_k)
        top_sims = topk.values.cpu().numpy()
        top_idx = topk.indices.cpu().numpy()

        # Softmax weights with temperature
        logits = top_sims * self.softmax_temperature
        # numerically stable softmax
        logits -= logits.max()
        w = np.exp(logits)
        w = w / w.sum()

        # Weighted mean of action chunks
        chunks = self.actions[top_idx]  # [k, K, A]
        blended = (w[:, None, None] * chunks).sum(axis=0).astype(np.float32)  # [K, A]

        return blended, {
            "top_sims": top_sims,
            "top_idx": top_idx,
            "weights": w,
            "mean_sim": float(top_sims.mean()),
        }

    @torch.no_grad()
    def retrieve_zbank_zstate(self, query_t):
        """Top-k retrieval of z_vl_bank + z_state from training manifold.

        Used by v15-zretrieve: projects live GR00T z_bank (which may be OOD
        due to Liquid's drift) back onto the training distribution by
        averaging the K nearest training z_banks.

        query_t: [d_feat] GPU tensor (DINOv2-derived feature, normalized).
        Returns:
            z_bank_avg [K_bank=4, 1024] numpy
            z_state_avg [1536] numpy
            diag dict
        """
        if self.zbank_memmaps is None:
            raise RuntimeError("retrieve_zbank_zstate requires load_z_bank_z_state=True at init")
        sims = self.features_gpu @ query_t
        topk = torch.topk(sims, k=self.top_k)
        top_sims = topk.values.cpu().numpy()
        top_idx = topk.indices.cpu().numpy()
        logits = top_sims * self.softmax_temperature
        logits = logits - logits.max()
        w = np.exp(logits); w = w / w.sum()

        zbanks = []
        zstates = []
        for j in top_idx:
            s_int = int(self.suite_idx[j])
            pos = int(self.per_suite_pos[j])
            zbanks.append(np.asarray(self.zbank_memmaps[s_int][pos]))
            zstates.append(np.asarray(self.zstate_memmaps[s_int][pos]))
        zbanks = np.stack(zbanks)        # [k, 4, 1024]
        zstates = np.stack(zstates)      # [k, 1536]
        z_bank_avg = (w[:, None, None] * zbanks).sum(axis=0).astype(np.float32)
        z_state_avg = (w[:, None] * zstates).sum(axis=0).astype(np.float32)
        return z_bank_avg, z_state_avg, {
            "top_sims": top_sims, "top_idx": top_idx, "weights": w,
            "mean_sim": float(top_sims.mean()),
        }

    @torch.no_grad()
    def query_and_blend(self, img_raw, wrist_raw, state8, liquid_chunk):
        """End-to-end: encode observation, retrieve, blend with Liquid chunk.

        Returns:
            final [K, A] numpy
            diag dict
        """
        query_t = self.encode_query(img_raw, wrist_raw, state8)
        retrieved, diag = self.retrieve(query_t)

        # Adaptive alpha: trust memory when retrieval similarity is high
        if self.adaptive_alpha:
            # Sigmoid-shaped trust: mean_sim near 1 → full alpha; near 0 → zero
            mean_sim = diag["mean_sim"]
            trust = max(0.0, min(1.0, mean_sim))  # cosine ∈ [-1,1] but typical 0..1
            alpha = self.alpha_base * trust
        else:
            alpha = self.alpha_base

        final = alpha * retrieved + (1.0 - alpha) * liquid_chunk
        diag["alpha_used"] = float(alpha)
        return final.astype(np.float32), diag
