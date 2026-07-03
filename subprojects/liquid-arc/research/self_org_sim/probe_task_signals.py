"""Diagnostic: which (h_goal signal, task-state label) pair is most learnable?

Loads /tmp/traj_libero10_s{10,20,30,40}.pt trajectories collected earlier.
Each record: h_goal_traj [T, K, d], goaldist_traj list, sub_steps int, succ bool,
             bailed bool, sub_id int.

Tests multiple (signal, label) combinations via linear & small-MLP probes.
Trajectory-level train/val split (avoids in-trajectory leakage).

Goal: identify the (signal, label) pair with highest predictive power → tells us
WHAT Liquid's belief state actually encodes about task progress, and WHICH training
objective will most directly capture useful task-state tracking.

Usage on Spark:
  python probe_task_signals.py --traj_files /tmp/traj_libero10_s10.pt,...
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def load_records(traj_files):
    all_records = []
    for fp in traj_files:
        ck = torch.load(fp, map_location="cpu", weights_only=False)
        all_records.extend(ck["records"])
        print(f"  {fp}: {len(ck['records'])} records")
    return all_records


# ============================================================
# SIGNAL EXTRACTORS — each takes h_goal_traj [T, K, d] + index t,
# returns 1-D feature vector
# ============================================================

def sig_h_pool_mean(traj, t):
    """Mean over K positions of h_goal at time t."""
    return traj[t].float().mean(dim=0).numpy()  # [d]


def sig_h_flat(traj, t):
    """Flattened h_goal at time t."""
    return traj[t].float().flatten().numpy()  # [K*d]


def sig_h_delta_window(traj, t, W):
    """h_goal[t] - h_goal[t-W], flattened."""
    return (traj[t].float() - traj[t - W].float()).flatten().numpy()  # [K*d]


def sig_h_velocity(traj, t):
    """L2 norm of (h_goal[t] - h_goal[t-1])."""
    if t == 0:
        return np.zeros(1, dtype=np.float32)
    diff = (traj[t].float() - traj[t - 1].float()).numpy()
    return np.array([np.linalg.norm(diff)], dtype=np.float32)


def sig_per_k(traj, t, k):
    """h_goal[t, k] — single position's vector."""
    return traj[t, k].float().numpy()  # [d]


def sig_h_norm(traj, t):
    """L2 norm of h_goal at time t (per K position, concatenated)."""
    return np.linalg.norm(traj[t].float().numpy(), axis=-1)  # [K]


# ============================================================
# LABEL GENERATORS — each takes (record, t) returns scalar/categorical label
# ============================================================

def label_progress(record, t):
    """Linear progress through THIS sub-task: t / T."""
    T = record["h_goal_traj"].shape[0]
    return float(t / max(T - 1, 1))


def label_chunks_remaining_norm(record, t):
    """(T - t) / T — normalized chunks remaining."""
    T = record["h_goal_traj"].shape[0]
    return float((T - t) / max(T, 1))


def label_success_binary(record, t):
    """Will this sub-task succeed? Constant across t for given record."""
    return float(record["succ"])


def label_success_within_5_chunks(record, t):
    """Will sub-task end successfully within next 5 chunks?"""
    T = record["h_goal_traj"].shape[0]
    return float(record["succ"] and (T - t) <= 5)


def label_late_phase(record, t):
    """Is t in the LAST third of this sub-task?"""
    T = record["h_goal_traj"].shape[0]
    return float(t >= 2 * T // 3)


def label_early_phase(record, t):
    """First 1/3 of the sub-task."""
    T = record["h_goal_traj"].shape[0]
    return float(t < T // 3)


def label_mid_phase(record, t):
    """Middle 1/3."""
    T = record["h_goal_traj"].shape[0]
    return float(T // 3 <= t < 2 * T // 3)


def make_label_success_within_k(k):
    def fn(record, t):
        T = record["h_goal_traj"].shape[0]
        return float(record["succ"] and (T - t) <= k)
    fn.__name__ = f"success_within_{k}"
    return fn


def make_label_failure_within_k(k):
    """Will sub-task END as a FAILURE within k chunks?"""
    def fn(record, t):
        T = record["h_goal_traj"].shape[0]
        return float((not record["succ"]) and (T - t) <= k)
    fn.__name__ = f"failure_within_{k}"
    return fn


def make_label_ending_within_k(k):
    """Will sub-task END (success OR failure) within k chunks?
    Pure task-end-detection without commitment on outcome."""
    def fn(record, t):
        T = record["h_goal_traj"].shape[0]
        return float((T - t) <= k)
    fn.__name__ = f"ending_within_{k}"
    return fn


def label_absolute_chunks_remaining(record, t):
    """Un-normalized chunks remaining (clipped to 30 to bound range)."""
    T = record["h_goal_traj"].shape[0]
    return float(min(T - t, 30))


def label_log_chunks_remaining(record, t):
    """log(1 + chunks remaining) — compresses long tail."""
    T = record["h_goal_traj"].shape[0]
    return float(np.log1p(T - t))


def label_progress_x_succ(record, t):
    """Progress weighted by success: 0 for failed sub-tasks, t/T for successful.
    Tests: can substrate predict 'how far we've genuinely progressed toward success'?"""
    T = record["h_goal_traj"].shape[0]
    return float((t / max(T - 1, 1)) * float(record["succ"]))


# ============================================================
# PROBE
# ============================================================

def build_dataset(records, signal_fn, label_fn, min_t=5, window=None):
    """For each record, sample valid t values and emit (signal_vec, label, traj_id)."""
    X = []
    y = []
    traj_ids = []
    for ri, r in enumerate(records):
        traj = r["h_goal_traj"]
        T = traj.shape[0]
        start = max(min_t, window if window else 0)
        if T <= start:
            continue
        for t in range(start, T):
            try:
                if window is not None:
                    feat = signal_fn(traj, t, window)
                else:
                    feat = signal_fn(traj, t)
                lbl = label_fn(r, t)
            except Exception:
                continue
            X.append(feat)
            y.append(lbl)
            traj_ids.append(ri)
    if not X:
        return None
    return np.stack(X), np.array(y, dtype=np.float32), np.array(traj_ids, dtype=np.int64)


def linear_probe(X_train, y_train, X_val, y_val, lr=1e-2, steps=500,
                 binary=False, device="cpu"):
    """Train tiny linear probe with Adam, return val metrics."""
    Xt = torch.from_numpy(X_train).float().to(device)
    yt = torch.from_numpy(y_train).float().to(device)
    Xv = torch.from_numpy(X_val).float().to(device)
    yv = torch.from_numpy(y_val).float().to(device)
    model = nn.Linear(X_train.shape[1], 1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss() if binary else nn.MSELoss()
    for _ in range(steps):
        opt.zero_grad()
        pred = model(Xt).squeeze(-1)
        loss = loss_fn(pred, yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred_v = model(Xv).squeeze(-1)
        if binary:
            prob = torch.sigmoid(pred_v).cpu().numpy()
            y_np = yv.cpu().numpy()
            # AUC
            order = np.argsort(-prob); ys = y_np[order]
            tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
            tpr = tp / max(ys.sum(), 1); fpr = fp / max((1 - ys).sum(), 1)
            auc = float(np.trapz(tpr, fpr))
            acc = float(((prob > 0.5).astype(np.float32) == y_np).mean())
            return {"AUC": auc, "acc": acc,
                    "base_rate": float(y_np.mean())}
        else:
            pred_np = pred_v.cpu().numpy()
            y_np = yv.cpu().numpy()
            mse = float(((pred_np - y_np) ** 2).mean())
            var = float(y_np.var())
            r2 = 1.0 - mse / max(var, 1e-9)
            return {"MSE": mse, "R2": r2, "var": var}


def mlp_probe(X_train, y_train, X_val, y_val, hidden=64, lr=3e-3, steps=1000,
              binary=False, device="cpu"):
    Xt = torch.from_numpy(X_train).float().to(device)
    yt = torch.from_numpy(y_train).float().to(device)
    Xv = torch.from_numpy(X_val).float().to(device)
    yv = torch.from_numpy(y_val).float().to(device)
    model = nn.Sequential(
        nn.Linear(X_train.shape[1], hidden), nn.SiLU(),
        nn.LayerNorm(hidden),
        nn.Linear(hidden, 1)
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss() if binary else nn.MSELoss()
    bs = min(256, len(Xt))
    rng = np.random.default_rng(0)
    for step in range(steps):
        idx = rng.choice(len(Xt), bs, replace=False)
        opt.zero_grad()
        pred = model(Xt[idx]).squeeze(-1)
        loss = loss_fn(pred, yt[idx])
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred_v = model(Xv).squeeze(-1)
        if binary:
            prob = torch.sigmoid(pred_v).cpu().numpy()
            y_np = yv.cpu().numpy()
            order = np.argsort(-prob); ys = y_np[order]
            tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
            tpr = tp / max(ys.sum(), 1); fpr = fp / max((1 - ys).sum(), 1)
            auc = float(np.trapz(tpr, fpr))
            acc = float(((prob > 0.5).astype(np.float32) == y_np).mean())
            return {"AUC": auc, "acc": acc,
                    "base_rate": float(y_np.mean())}
        else:
            pred_np = pred_v.cpu().numpy()
            y_np = yv.cpu().numpy()
            mse = float(((pred_np - y_np) ** 2).mean())
            var = float(y_np.var())
            r2 = 1.0 - mse / max(var, 1e-9)
            return {"MSE": mse, "R2": r2, "var": var}


def _get_signals_and_labels(K):
    signals = [
        ("h_pool_mean", sig_h_pool_mean, None),
        ("h_flat", sig_h_flat, None),
        ("h_velocity", sig_h_velocity, None),
        ("h_norms_perK", sig_h_norm, None),
    ]
    for w in [1, 3, 5, 10, 15, 20, 30]:
        signals.append((f"h_delta_W{w}", sig_h_delta_window, w))
    for k in range(K):
        signals.append((f"h_K{k}", lambda traj, t, _k=k: sig_per_k(traj, t, _k), None))
    labels = [
        ("progress_t/T",                  label_progress,                    False),
        ("chunks_remaining/T",            label_chunks_remaining_norm,       False),
        ("absolute_chunks_remaining",     label_absolute_chunks_remaining,   False),
        ("log_chunks_remaining",          label_log_chunks_remaining,        False),
        ("progress_x_success",            label_progress_x_succ,             False),
        ("early_phase",                   label_early_phase,                 True),
        ("mid_phase",                     label_mid_phase,                   True),
        ("late_phase",                    label_late_phase,                  True),
        ("success_binary",                label_success_binary,              True),
    ]
    for k in [1, 3, 5, 10, 15, 20]:
        labels.append((f"success_within_{k}",   make_label_success_within_k(k),  True))
    for k in [1, 3, 5, 10]:
        labels.append((f"failure_within_{k}",   make_label_failure_within_k(k),  True))
    for k in [1, 3, 5, 10]:
        labels.append((f"ending_within_{k}",    make_label_ending_within_k(k),   True))
    return signals, labels


def run_sweep(train_records, val_records, K, d, device):
    """Run all (signal × label × probe) combinations on this train/val split.
    Returns list of result dicts."""
    signals, labels = _get_signals_and_labels(K)
    results = []
    for sig_name, sig_fn, sig_window in signals:
        for lbl_name, lbl_fn, is_binary in labels:
            train_ds = build_dataset(train_records, sig_fn, lbl_fn, window=sig_window)
            val_ds = build_dataset(val_records, sig_fn, lbl_fn, window=sig_window)
            if train_ds is None or val_ds is None:
                continue
            X_train, y_train, _ = train_ds
            X_val, y_val, _ = val_ds
            if X_train.shape[1] == 1:
                lin = linear_probe(X_train, y_train, X_val, y_val,
                                    binary=is_binary, device=device)
                results.append({"signal": sig_name, "label": lbl_name,
                                 "probe": "linear", "feat_dim": X_train.shape[1],
                                 **lin})
                continue
            lin = linear_probe(X_train, y_train, X_val, y_val,
                                binary=is_binary, device=device)
            results.append({"signal": sig_name, "label": lbl_name,
                             "probe": "linear", "feat_dim": X_train.shape[1],
                             **lin})
            mlp = mlp_probe(X_train, y_train, X_val, y_val,
                              binary=is_binary, device=device)
            results.append({"signal": sig_name, "label": lbl_name,
                             "probe": "mlp", "feat_dim": X_train.shape[1],
                             **mlp})
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traj_files", required=True,
                   help="Comma-separated paths")
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--split_mode", choices=["trajectory", "task_id"],
                    default="trajectory",
                    help="trajectory: hold out random trajectories. task_id: hold "
                         "out entire sub-task IDs — measures generalization across "
                         "novel tasks (not just novel rollouts of seen tasks).")
    p.add_argument("--out_md", default="/tmp/probe_results.md")
    p.add_argument("--split_seed", type=int, default=42,
                    help="Seed for train/val partition")
    p.add_argument("--leave_one_task_out", action="store_true",
                    help="Run leave-one-task-out cross-validation (task_id split). "
                         "For each unique sub-task ID, hold it out as val, train on rest. "
                         "Reports mean ± std per (signal, label) combination across folds.")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[probe] device={device}")
    records = load_records([s.strip() for s in args.traj_files.split(",")])
    n_total = len(records)
    print(f"[probe] {n_total} trajectories total")
    K = records[0]["h_goal_traj"].shape[1]
    d = records[0]["h_goal_traj"].shape[2]
    print(f"[probe] K={K}, d={d}")

    rng = np.random.default_rng(args.split_seed)
    if args.leave_one_task_out:
        all_sub_ids = sorted({int(r["sub_id"]) for r in records})
        folds = [(tid, [t for t in all_sub_ids if t != tid]) for tid in all_sub_ids]
        print(f"[probe] LOTO: {len(folds)} folds over sub-task IDs {all_sub_ids}")
        # Run sweep per fold, collect per-(signal,label) lists of vals
        from collections import defaultdict
        agg = defaultdict(list)  # (label, signal, probe) → list of metric values
        for fold_idx, (held_out, train_ids_list) in enumerate(folds):
            tr_recs = [r for r in records if int(r["sub_id"]) in train_ids_list]
            va_recs = [r for r in records if int(r["sub_id"]) == held_out]
            fold_results = run_sweep(tr_recs, va_recs, K, d, device)
            for r in fold_results:
                key = (r["label"], r["signal"], r["probe"])
                metric_val = r.get("AUC", r.get("R2", None))
                if metric_val is not None:
                    agg[key].append(metric_val)
            print(f"  fold {fold_idx+1}/{len(folds)} (held out task {held_out}): "
                   f"{len(fold_results)} results")
        # Summarize: mean ± std per (label, signal, probe)
        print("\n=== LOTO Cross-Validation Results (mean ± std across folds) ===")
        md_lines = ["| label | signal | probe | mean_metric | std | n_folds |",
                    "|---|---|---|---|---|---|"]
        rows = []
        for (label, signal, probe), vals in agg.items():
            rows.append({"label": label, "signal": signal, "probe": probe,
                          "mean": float(np.mean(vals)), "std": float(np.std(vals)),
                          "n": len(vals)})
        rows.sort(key=lambda r: (r["label"], -r["mean"]))
        for r in rows:
            line = (f"  {r['label']:>25s}  {r['signal']:>18s}  {r['probe']:>6s}  "
                     f"mean={r['mean']:.3f}  std={r['std']:.3f}  n={r['n']}")
            print(line)
            md_lines.append(f"| {r['label']} | {r['signal']} | {r['probe']} | "
                             f"{r['mean']:.3f} | {r['std']:.3f} | {r['n']} |")
        Path(args.out_md).write_text("\n".join(md_lines))
        print(f"\n[probe] markdown → {args.out_md}")
        return

    if args.split_mode == "trajectory":
        perm = rng.permutation(n_total)
        n_val = int(args.val_frac * n_total)
        val_traj = set(perm[:n_val].tolist())
        train_records = [r for i, r in enumerate(records) if i not in val_traj]
        val_records = [r for i, r in enumerate(records) if i in val_traj]
        print(f"[probe] split=trajectory: {len(train_records)} train traj / "
              f"{len(val_records)} val traj")
    else:  # task_id
        all_sub_ids = sorted({int(r["sub_id"]) for r in records})
        rng.shuffle(all_sub_ids)
        n_val_ids = max(1, int(args.val_frac * len(all_sub_ids)))
        val_ids = set(all_sub_ids[:n_val_ids])
        train_ids = set(all_sub_ids[n_val_ids:])
        train_records = [r for r in records if int(r["sub_id"]) in train_ids]
        val_records = [r for r in records if int(r["sub_id"]) in val_ids]
        print(f"[probe] split=task_id: train ids {sorted(train_ids)} "
              f"({len(train_records)} traj) / val ids {sorted(val_ids)} "
              f"({len(val_records)} traj)")

    results = run_sweep(train_records, val_records, K, d, device)

    # Sort by primary metric per label-type
    def score(r):
        if "AUC" in r:
            return r["AUC"]
        return r.get("R2", -999)

    results.sort(key=lambda r: (r["label"], -score(r)))

    # Print + write markdown
    print("\n=== Results (sorted by label, best first) ===\n")
    md_lines = ["| label | signal | probe | metric | value | feat_dim |",
                "|---|---|---|---|---|---|"]
    for r in results:
        if "AUC" in r:
            metric = "AUC"; val = f"{r['AUC']:.3f}"
            extra = f" (acc={r['acc']:.3f} base={r['base_rate']:.3f})"
        else:
            metric = "R2"; val = f"{r['R2']:.3f}"
            extra = f" (MSE={r['MSE']:.4f})"
        line = f"  {r['label']:>22s}  {r['signal']:>15s}  {r['probe']:>6s}  " \
               f"{metric}={val}{extra}  feat_dim={r['feat_dim']}"
        print(line)
        md_lines.append(f"| {r['label']} | {r['signal']} | {r['probe']} | "
                         f"{metric} | {val} | {r['feat_dim']} |")

    Path(args.out_md).write_text("\n".join(md_lines))
    print(f"\n[probe] markdown table → {args.out_md}")


if __name__ == "__main__":
    main()
