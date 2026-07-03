"""Diagnostic suite for trained JEPA-LGT substrates.

Tests what the substrate actually learned (independent of GR00T behavior eval):

A. Held-out prediction loss
   - Run substrate forward on triples NOT in training set
   - Compare: trained substrate vs untrained random-init vs naive (z_pred=z_t)
   - Substrate << naive on held-out → learned generalizable dynamics
   - Substrate >> training loss on held-out → overfit

B. Linear probes on h_goal
   - Collect (h_goal[K*d-flat], labels) over held-out turns
   - Train linear probes for: task_id, suite, step_in_episode, frac_done
   - Compare probe accuracy/R² trained vs untrained
   - If trained > untrained, h_goal encodes that info

E. Multi-step rollout coherence
   - For K in {1, 4, 8, 16, 32}: run K-step autoregressive rollout
   - Track per-step prediction error
   - Coherent dynamics → error grows linearly; chaotic → exponential

Usage:
  python diagnostic_substrate.py \\
      --ckpt /tmp/lgt_jepa_big_K4_v3.pt \\
      --ckpt_step8 /tmp/lgt_jepa_big_K4_v3_step8000.pt \\
      --held_out_data /tmp/libero_jepa_held_out_triples.npz \\
      --out_json /tmp/diagnostic_v3.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Spark sm121 SDPA fix (no-op for big variant but inherited from earlier)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

_LA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_LA_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from liquid_goal_tracker_jepa_big import JEPA_LGT_Big  # type: ignore

torch.set_float32_matmul_precision("high")


def load_triples(path):
    d = np.load(path, allow_pickle=True)
    return {
        "z_t": d["z_t"], "chunks": d["chunks"], "z_next": d["z_next"],
        "z_goal": d["z_goal"], "ep_id": d["ep_id"], "suite": d["suite"],
        "task_id": d["task_id"],
    }


def build_episode_index(ep_id, suite):
    keys = [(str(s), int(e)) for s, e in zip(suite, ep_id)]
    eps: Dict[Tuple[str, int], List[int]] = {}
    for i, k in enumerate(keys):
        eps.setdefault(k, []).append(i)
    return [sorted(idxs) for idxs in eps.values()]


def load_substrate(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    sa = ck["args"]
    sub = JEPA_LGT_Big(
        z_vl_dim=ck["z_vl_dim"],
        action_dim=ck["action_dim"],
        horizon=ck["horizon"],
        d=sa["d_substrate"], K=sa["K_belief"],
        tangent_scale=sa["tangent_scale"],
    ).to(device)
    sub.load_state_dict(ck["substrate_state_dict"])
    sub.eval()
    return sub, sa


def make_untrained_like(trained_sub, device):
    """Random-init substrate with same hyperparams as trained one."""
    args = {
        "z_vl_dim": trained_sub.z_vl_dim,
        "action_dim": trained_sub.action_dim,
        "horizon": trained_sub.horizon,
        "d": trained_sub.d, "K": trained_sub.K,
        "tangent_scale": trained_sub.tangent_scale,
    }
    untr = JEPA_LGT_Big(**args).to(device).eval()
    return untr


# -------------- A. Held-out prediction loss --------------

def diag_A_held_out_loss(substrate, data, episodes, device, label):
    """Walk every episode, single-step prediction, accumulate loss."""
    losses_pred = []
    losses_naive = []
    cv_vals = []
    tang_norms = []
    with torch.no_grad():
        for ep_idxs in episodes:
            h_goal = substrate.init_state(1, device)
            for i in ep_idxs:
                z_t = torch.from_numpy(data["z_t"][i]).to(device).unsqueeze(0)
                chunk_t = torch.from_numpy(data["chunks"][i]).to(device).unsqueeze(0)
                z_goal = torch.from_numpy(data["z_goal"][i]).to(device).unsqueeze(0)
                z_next = torch.from_numpy(data["z_next"][i]).to(device).unsqueeze(0)
                h_goal, z_pred, tangent, info = substrate.step(
                    h_goal, z_t, z_goal, chunk_t)
                l_pred = F.smooth_l1_loss(z_pred, z_next, beta=1.0)
                l_naive = F.smooth_l1_loss(z_t, z_next, beta=1.0)
                losses_pred.append(float(l_pred))
                losses_naive.append(float(l_naive))
                cv_vals.append(float(info["metric_cv"]))
                tang_norms.append(float(info["tangent_norm"]))
    out = {
        "label": label,
        "n_turns": len(losses_pred),
        "pred_loss_mean": float(np.mean(losses_pred)),
        "naive_loss_mean": float(np.mean(losses_naive)),
        "ratio_pred_naive": float(np.mean(losses_pred) / max(np.mean(losses_naive), 1e-8)),
        "cv_mean": float(np.mean(cv_vals)),
        "tangent_norm_mean": float(np.mean(tang_norms)),
    }
    return out


# -------------- B. Linear probes on h_goal --------------

def collect_h_goal_labels(substrate, data, episodes, device):
    """For each turn, capture h_goal and labels (task_id, suite, step_in_ep, frac_done)."""
    suites_str = [str(s) for s in data["suite"]]
    task_ids = data["task_id"]
    h_goals = []
    labels = {"task_id": [], "suite_id": [], "step_in_ep": [], "frac_done": []}
    suite_to_int = {}
    with torch.no_grad():
        for ep_idxs in episodes:
            h_goal = substrate.init_state(1, device)
            ep_len = len(ep_idxs)
            for step_in_ep, i in enumerate(ep_idxs):
                z_t = torch.from_numpy(data["z_t"][i]).to(device).unsqueeze(0)
                chunk_t = torch.from_numpy(data["chunks"][i]).to(device).unsqueeze(0)
                z_goal = torch.from_numpy(data["z_goal"][i]).to(device).unsqueeze(0)
                h_goal_new, _, _, _ = substrate.step(h_goal, z_t, z_goal, chunk_t)
                h_goal = h_goal_new
                # Capture h_goal at this turn (flatten K*d)
                h_flat = h_goal[0].reshape(-1).cpu().numpy()
                h_goals.append(h_flat)
                labels["task_id"].append(int(task_ids[i]))
                s = suites_str[i]
                if s not in suite_to_int:
                    suite_to_int[s] = len(suite_to_int)
                labels["suite_id"].append(suite_to_int[s])
                labels["step_in_ep"].append(step_in_ep)
                labels["frac_done"].append(step_in_ep / max(ep_len - 1, 1))
    return np.stack(h_goals), {k: np.array(v) for k, v in labels.items()}, suite_to_int


def _train_linear_clf(X, y, n_classes, device, n_epochs=200, lr=1e-2):
    """Multinomial logistic regression with torch (no sklearn)."""
    Xt = torch.from_numpy(X).float().to(device)
    yt = torch.from_numpy(y).long().to(device)
    W = torch.zeros(X.shape[1], n_classes, device=device, requires_grad=True)
    b = torch.zeros(n_classes, device=device, requires_grad=True)
    opt = torch.optim.AdamW([W, b], lr=lr, weight_decay=1e-3)
    for _ in range(n_epochs):
        logits = Xt @ W + b
        loss = F.cross_entropy(logits, yt)
        opt.zero_grad(); loss.backward(); opt.step()
    return W.detach(), b.detach()


def _train_linear_reg(X, y, device, n_epochs=200, lr=1e-2):
    Xt = torch.from_numpy(X).float().to(device)
    yt = torch.from_numpy(y).float().to(device)
    W = torch.zeros(X.shape[1], 1, device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.AdamW([W, b], lr=lr, weight_decay=1e-3)
    for _ in range(n_epochs):
        pred = (Xt @ W + b).squeeze(-1)
        loss = F.mse_loss(pred, yt)
        opt.zero_grad(); loss.backward(); opt.step()
    return W.detach(), b.detach()


def _split(X, y, frac_test=0.3, seed=0):
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * frac_test)
    te_idx, tr_idx = idx[:n_test], idx[n_test:]
    return X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]


def diag_B_probes(substrate, data, episodes, device, label):
    """Train linear probes on h_goal — torch implementation."""
    H, lab, _ = collect_h_goal_labels(substrate, data, episodes, device)
    out = {"label": label, "h_dim": int(H.shape[1])}

    # Task ID — multinomial
    y = lab["task_id"]
    # Remap task_ids to dense [0..C-1]
    uniq = np.unique(y); id_map = {v: i for i, v in enumerate(uniq)}
    y_dense = np.array([id_map[v] for v in y])
    Xtr, Xte, ytr, yte = _split(H, y_dense)
    W, b = _train_linear_clf(Xtr, ytr, len(uniq), device)
    with torch.no_grad():
        pred = (torch.from_numpy(Xte).float().to(device) @ W + b).argmax(-1).cpu().numpy()
    out["probe_task_id_acc"] = float((pred == yte).mean())
    out["probe_task_id_chance"] = 1.0 / len(uniq)

    # Suite — multinomial
    Xtr, Xte, ytr, yte = _split(H, lab["suite_id"])
    W, b = _train_linear_clf(Xtr, ytr, int(lab["suite_id"].max() + 1), device)
    with torch.no_grad():
        pred = (torch.from_numpy(Xte).float().to(device) @ W + b).argmax(-1).cpu().numpy()
    out["probe_suite_acc"] = float((pred == yte).mean())
    out["probe_suite_chance"] = 1.0 / len(np.unique(lab["suite_id"]))

    # Step in episode — regression (R²)
    for name in ("step_in_ep", "frac_done"):
        Xtr, Xte, ytr, yte = _split(H, lab[name].astype(np.float32))
        W, b = _train_linear_reg(Xtr, ytr, device)
        with torch.no_grad():
            pred = ((torch.from_numpy(Xte).float().to(device) @ W).squeeze(-1) + b
                    ).squeeze(-1).cpu().numpy()
        ss_res = float(((yte - pred) ** 2).sum())
        ss_tot = float(((yte - yte.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
        out[f"probe_{name}_r2"] = r2
    return out


# -------------- E. Multi-step rollout coherence --------------

def diag_E_rollout_coherence(substrate, data, episodes, device, label, Ks=(1, 4, 8, 16, 32)):
    """For each rollout horizon K, measure per-step prediction error."""
    results = {}
    with torch.no_grad():
        for K in Ks:
            long_eps = [e for e in episodes if len(e) >= K + 1]
            if not long_eps:
                results[f"K={K}"] = None
                continue
            per_step_errors = [[] for _ in range(K)]
            n_rollouts = 0
            for ep_idxs in long_eps:
                # Roll forward at every valid start
                for t_start in range(0, len(ep_idxs) - K):
                    h_goal = substrate.init_state(1, device)
                    z_pred = torch.from_numpy(
                        data["z_t"][ep_idxs[t_start]]).to(device).unsqueeze(0)
                    z_goal = torch.from_numpy(
                        data["z_goal"][ep_idxs[t_start]]).to(device).unsqueeze(0)
                    for k in range(K):
                        i = ep_idxs[t_start + k]
                        chunk_t = torch.from_numpy(
                            data["chunks"][i]).to(device).unsqueeze(0)
                        h_goal, z_pred_next, _, _ = substrate.step(
                            h_goal, z_pred, z_goal, chunk_t)
                        z_real_next = torch.from_numpy(
                            data["z_next"][i]).to(device).unsqueeze(0)
                        err = F.smooth_l1_loss(z_pred_next, z_real_next, beta=1.0)
                        per_step_errors[k].append(float(err))
                        z_pred = z_pred_next   # AR
                    n_rollouts += 1
            results[f"K={K}"] = {
                "per_step_mean_err": [float(np.mean(p)) for p in per_step_errors],
                "n_rollouts": n_rollouts,
            }
    return {"label": label, "rollout_errors": results}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Main ckpt to diagnose (e.g. v3 step24000)")
    p.add_argument("--ckpt_aux", default=None,
                   help="Optional second ckpt for comparison (e.g. v3 step8000)")
    p.add_argument("--held_out_data", required=True,
                   help="Held-out triples npz (not in training set)")
    p.add_argument("--out_json", default="/tmp/diagnostic.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[diag] device={device}", flush=True)

    data = load_triples(args.held_out_data)
    episodes = build_episode_index(data["ep_id"], data["suite"])
    print(f"[diag] {len(data['z_t'])} held-out triples across {len(episodes)} episodes",
          flush=True)

    # Load substrates
    print(f"[diag] loading main substrate from {args.ckpt}", flush=True)
    sub_main, sa_main = load_substrate(args.ckpt, device)
    print(f"[diag] main substrate: d={sa_main['d_substrate']}, K={sa_main['K_belief']}",
          flush=True)

    untr = make_untrained_like(sub_main, device)
    print(f"[diag] untrained random-init control substrate built", flush=True)

    subs = {
        "trained_main": sub_main,
        "untrained_control": untr,
    }
    if args.ckpt_aux is not None:
        print(f"[diag] loading aux substrate from {args.ckpt_aux}", flush=True)
        sub_aux, _ = load_substrate(args.ckpt_aux, device)
        subs["trained_aux"] = sub_aux

    results = {"meta": {"ckpt": args.ckpt, "ckpt_aux": args.ckpt_aux,
                          "n_held_out_triples": len(data["z_t"]),
                          "n_held_out_episodes": len(episodes)}}

    for name, sub in subs.items():
        print(f"\n========== diagnosing: {name} ==========", flush=True)
        results[name] = {}
        # A — held-out loss
        print(f"  [A] held-out prediction loss...", flush=True)
        A = diag_A_held_out_loss(sub, data, episodes, device, name)
        results[name]["A_pred_loss"] = A
        print(f"      pred={A['pred_loss_mean']:.5f} naive={A['naive_loss_mean']:.5f} "
              f"ratio={A['ratio_pred_naive']:.3f} cv={A['cv_mean']:.3f} "
              f"tang={A['tangent_norm_mean']:.3f}", flush=True)
        # B — linear probes
        print(f"  [B] linear probes on h_goal...", flush=True)
        B = diag_B_probes(sub, data, episodes, device, name)
        results[name]["B_probes"] = B
        print(f"      task_id={B['probe_task_id_acc']:.3f} "
              f"(chance={B['probe_task_id_chance']:.3f})  "
              f"suite={B['probe_suite_acc']:.3f} "
              f"(chance={B['probe_suite_chance']:.3f})  "
              f"step_R²={B['probe_step_in_ep_r2']:.3f} "
              f"frac_R²={B['probe_frac_done_r2']:.3f}", flush=True)
        # E — rollout coherence
        print(f"  [E] multi-step rollout coherence...", flush=True)
        E = diag_E_rollout_coherence(sub, data, episodes, device, name,
                                       Ks=(1, 4, 8, 16, 32))
        results[name]["E_rollout"] = E
        for K_label, R in E["rollout_errors"].items():
            if R is None:
                continue
            errs = R["per_step_mean_err"]
            print(f"      {K_label}: k0={errs[0]:.5f}  k_last={errs[-1]:.5f}  "
                  f"max={max(errs):.5f}  n={R['n_rollouts']}", flush=True)

    Path(args.out_json).write_text(json.dumps(results, indent=2))
    print(f"\n[diag] saved → {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
