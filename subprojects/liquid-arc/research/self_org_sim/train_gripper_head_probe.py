"""Pre-validation for v17: does a substrate-based gripper head outperform
a simple MLP head on the SAME input features?

Trains two heads on expert gripper labels, using DINOv2-derived features
(memory_bank_v11.npz features [776-d]) as input. The substrate-head uses
canonical LiquidARC ContinuousDynamics over K=4 belief positions; the MLP
head is a 3-layer perceptron with similar param budget.

Output: validation accuracy comparison, plus saved checkpoints for closed-
loop swap-in test.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_LA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_LA_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from liquid_arc.config import LiquidARCConfig  # type: ignore
from liquid_arc.dynamics import ContinuousDynamics  # type: ignore
from liquid_arc.solver import euler_solve_halting  # type: ignore
from liquid_arc.context_pool import ContextPool  # type: ignore

torch.set_float32_matmul_precision("high")


class MLPGripperHead(nn.Module):
    """Baseline: simple MLP. Same param budget as substrate head."""

    def __init__(self, d_in=776, d_hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.SiLU(),
            nn.Linear(d_hidden, d_hidden), nn.SiLU(),
            nn.Linear(d_hidden, d_hidden), nn.SiLU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_gripper_substrate_config(d=128):
    """Small substrate config tuned for binary state maintenance, not feature
    extraction. Halting min=2 (faster convergence for binary decisions)."""
    return LiquidARCConfig(
        d_model=d, d_metric=32, d_ffn=128, max_seq_len=4,
        n_ode_steps=8, ode_steps_min=4, ode_steps_max=12,
        integration_time=1.0,
        tau_min=0.3, tau_max=1.0, t_diffusion_init=0.5,
        routing_mode="metric",
        tau_freeze_steps=2000,
        halting_enabled=True, halting_min_steps=2,
        halting_ponder_lambda=0.001 * 0.1,
        rezero_enabled=True, rezero_gate_init=-3.0,
        metric_bias_init_std=0.1,
        deep_supervision_enabled=False,
        ponder_kl_lambda=0.0,
        criticality_loss_enabled=True,
        criticality_loss_lambda=0.0001,
        criticality_target_ratio=18.0, criticality_D_sq_target=30.0,
        curvature_diversity_loss_enabled=True,
        curvature_diversity_lambda=0.0001,
        curvature_cv_floor=1.5, curvature_cv_ceiling=8.0,
        tau_quality_loss_enabled=False,
        tau_mean_target=0.0, tau_log_spread_target=0.4,
        step_embed_enabled=False,
        step_conditional_operator=False,
        structural_tau_enabled=False,
        norm_ref=20.0, norm_lambda=0.1,
        base_lr=3e-4, structural_lr_ratio=0.1,
        warmup_steps=200, weight_decay=0.01,
        use_torch_compile=False,
    )


class SubstrateGripperHead(nn.Module):
    """K=4 belief-position substrate over input features.

    Positions:
      0: approach_belief — feature → linear
      1: contact_belief  — feature → linear (different init)
      2: hold_belief     — feature → linear
      3: target_state    — feature → linear
    Substrate evolves these via Euler ODE with halting; output read from pos 3.
    """

    def __init__(self, d_in=776, d=128, K=4):
        super().__init__()
        self.d = d
        self.K = K
        self.config = make_gripper_substrate_config(d=d)
        # Per-position projections from input feature
        self.pos_projs = nn.ModuleList([nn.Linear(d_in, d) for _ in range(K)])
        for proj in self.pos_projs:
            nn.init.normal_(proj.weight, std=0.02)
            nn.init.zeros_(proj.bias)
        self.pos_embed = nn.Parameter(torch.zeros(K, d))
        nn.init.normal_(self.pos_embed, std=0.02)
        # Canonical substrate
        self.context_pool = ContextPool(self.config)
        self.dynamics = ContinuousDynamics(self.config)
        # Readout from target_state (position 3)
        self.readout = nn.Linear(d, 1)

    def forward(self, x):
        # x: [B, d_in]
        B = x.shape[0]
        h0 = torch.stack([proj(x) for proj in self.pos_projs], dim=1)  # [B, K, d]
        h0 = h0 + self.pos_embed.unsqueeze(0)
        # Substrate setup
        context = self.context_pool(h0, None)
        self.dynamics.set_context(context, mask=None)
        # ODE evolution (fixed n_steps for predictability during probe)
        n_steps = self.config.n_ode_steps if not self.training else int(
            torch.randint(self.config.ode_steps_min, self.config.ode_steps_max + 1, (1,)).item()
        )
        self.dynamics.set_n_steps(n_steps)
        T = float(self.config.integration_time)
        out = euler_solve_halting(
            self.dynamics, h0, (0.0, T), n_steps,
            min_steps=self.config.halting_min_steps,
        )
        if isinstance(out, tuple):
            h = out[0]
        else:
            h = out
        # Readout from target_state position
        logit = self.readout(h[:, 3]).squeeze(-1)
        return logit


def build_dataset(memory_bank_path: Path, dataset_root: Path):
    """Same approach as train_phase_classifier but labels are GRIPPER VALUE
    not phase: y=1 if expert grip<0 (open) at this state, y=0 if grip>0 (close).
    """
    from train_phase_classifier import compute_first_flip_per_episode, SUITE_IDX_TO_NAME

    print(f"[probe] loading memory_bank {memory_bank_path}")
    bank = np.load(memory_bank_path, allow_pickle=True)
    feats = bank["features"]
    suite_idx = bank["suite_idx"]
    ep = bank["ep"]
    t = bank["t"]
    N = len(feats)
    print(f"[probe] bank: N={N}, d={feats.shape[1]}")

    # Look up expert grip[0] per sample by re-reading teacher_chunks for the matching (suite, ep, t)
    grip_label = np.zeros(N, dtype=np.float32)
    per_suite_data = {}
    for s_int, s_name in SUITE_IDX_TO_NAME.items():
        sd = dataset_root / f"libero-{s_name.split('_')[-1]}-expert-v1"
        if not sd.exists():
            continue
        idx_file = np.load(sd / "labels_index.npz")
        sample_idx_arr = idx_file["sample_idx"]
        n_samples_s = int(idx_file["n_samples"])
        chunks_s = np.memmap(sd / "teacher_chunks.dat", dtype=np.float32, mode="r",
                             shape=(n_samples_s, 16, 7))
        per_suite_data[s_int] = (sample_idx_arr, chunks_s)

    print("[probe] building gripper labels (open=1, close=0)...")
    t0 = time.time()
    # For efficiency, build a (suite, ep, t) → s map then look up grip
    suite_ep_t_to_s = {}
    for s_int, (sample_idx_arr, _) in per_suite_data.items():
        for s_local in range(len(sample_idx_arr)):
            ep_i, t_i, _task = sample_idx_arr[s_local]
            suite_ep_t_to_s[(s_int, int(ep_i), int(t_i))] = s_local

    for i in range(N):
        key = (int(suite_idx[i]), int(ep[i]), int(t[i]))
        s_local = suite_ep_t_to_s.get(key)
        if s_local is None:
            grip_label[i] = 1.0  # default to "open"
            continue
        chunks_s = per_suite_data[int(suite_idx[i])][1]
        g = float(chunks_s[s_local, 0, -1])
        grip_label[i] = 1.0 if g < 0 else 0.0
    print(f"[probe] labels built in {time.time()-t0:.1f}s")
    print(f"[probe] open={int(grip_label.sum())} close={int((1-grip_label).sum())}")
    return feats.astype(np.float32), grip_label


def train_head(head, feats, labels, train_idx, val_idx, args, label="head"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = head.to(device)
    print(f"[{label}] params: {sum(p.numel() for p in head.parameters()):,}")
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_feats = torch.from_numpy(feats[train_idx]).to(device)
    train_labels = torch.from_numpy(labels[train_idx]).to(device)
    val_feats = torch.from_numpy(feats[val_idx]).to(device)
    val_labels = torch.from_numpy(labels[val_idx]).to(device)

    n_train = len(train_idx)
    best_val_acc = 0.0
    best_state = None
    for epoch in range(args.epochs):
        head.train()
        perm = torch.randperm(n_train, device=device)
        train_feats_e = train_feats[perm]
        train_labels_e = train_labels[perm]
        loss_sum = 0.0; n_batches = 0
        for start in range(0, n_train, args.batch_size):
            end = min(start + args.batch_size, n_train)
            x = train_feats_e[start:end]
            y = train_labels_e[start:end]
            logits = head(x)
            # Focal BCE: weight false-close (model says close when expert is open) 3x
            base = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
            # Per-sample weight: if label=1 (expert open) but model logit > 0 (says close), upweight 3x
            with torch.no_grad():
                model_says_close = (logits < 0).float()  # logit<0 → P<0.5 → close
                # When expert open (y=1) AND model says close: high penalty
                penalty_mask = y * model_says_close
                weights = 1.0 + 2.0 * penalty_mask  # 1x baseline, 3x for the failure mode
            loss = (base * weights).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += float(loss.detach())
            n_batches += 1
        head.eval()
        with torch.no_grad():
            val_logits = head(val_feats)
            val_preds = (val_logits > 0).float()
            val_acc = (val_preds == val_labels).float().mean().item()
            open_mask = val_labels == 1
            close_mask = val_labels == 0
            acc_open = (val_preds[open_mask] == 1).float().mean().item() if open_mask.any() else 0
            acc_close = (val_preds[close_mask] == 0).float().mean().item() if close_mask.any() else 0
        print(f"[{label}] epoch {epoch:>2}  loss={loss_sum/max(n_batches,1):.4f}  "
              f"val_acc={val_acc:.4f}  acc_open={acc_open:.3f}  acc_close={acc_close:.3f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    if best_state is not None:
        head.load_state_dict(best_state)
    return head, best_val_acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--memory_bank", default="/home/pokazge/datasets/memory_bank_v11.npz")
    p.add_argument("--dataset_root", default="/home/pokazge/datasets")
    p.add_argument("--output_dir", default="/tmp/gripper_probe")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_frac", type=float, default=0.1)
    args = p.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    feats, labels = build_dataset(Path(args.memory_bank), Path(args.dataset_root))
    N = len(feats)
    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    val_n = int(N * args.val_frac)
    val_idx = perm[:val_n]
    train_idx = perm[val_n:]
    print(f"[probe] train: {len(train_idx)}  val: {len(val_idx)}")

    print("\n=== Training MLP gripper head ===")
    mlp = MLPGripperHead(d_in=feats.shape[1], d_hidden=256)
    mlp, mlp_val_acc = train_head(mlp, feats, labels, train_idx, val_idx, args, label="mlp")
    torch.save({"state_dict": mlp.state_dict(), "d_in": feats.shape[1], "d_hidden": 256,
                "val_acc": mlp_val_acc},
               out_dir / "mlp_head.pt")

    print("\n=== Training Substrate gripper head ===")
    sub = SubstrateGripperHead(d_in=feats.shape[1], d=128, K=4)
    sub, sub_val_acc = train_head(sub, feats, labels, train_idx, val_idx, args, label="sub")
    torch.save({"state_dict": sub.state_dict(), "d_in": feats.shape[1], "d": 128, "K": 4,
                "val_acc": sub_val_acc},
               out_dir / "substrate_head.pt")

    print(f"\n=== SUMMARY ===")
    print(f"MLP val_acc:       {mlp_val_acc:.4f}")
    print(f"Substrate val_acc: {sub_val_acc:.4f}")
    print(f"Δ (sub - mlp):     {(sub_val_acc - mlp_val_acc)*100:+.2f}pp")
    if sub_val_acc - mlp_val_acc > 0.03:
        print("GATE PASSED: substrate >3pp better — proceed to v17 full build")
    elif sub_val_acc - mlp_val_acc > 0:
        print("MARGINAL: substrate slightly better but not 3pp — risky to proceed")
    else:
        print("GATE FAILED: substrate not better than MLP — substrate not adding value here either")


if __name__ == "__main__":
    main()
