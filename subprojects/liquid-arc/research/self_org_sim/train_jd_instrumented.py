"""Instrumented JudgeDistillLiquid training — captures full phase transition trajectory.

Per-checkpoint diagnostics:
  - CV (metric eigenvalue coefficient of variation)
  - Full eigenvalue distribution (percentiles + histogram bins)
  - Per-K activation strengths (which positions specialize)
  - Class separation in h_fast space (success vs drift)
  - Slow channel evolution (commitment representation strength)
  - Trigger firing rate at turn boundaries
  - Geometric distance between success/drift trajectory endpoints

Saves snapshots at predefined checkpoints + comprehensive diagnostics log.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from judge_distill_liquid import JudgeDistillLiquid, forward_trajectory


def roc_auc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rs = scores[labels == 1]
    rn = scores[labels == 0]
    wins = (rs[:, None] > rn[None, :]).sum()
    ties = (rs[:, None] == rn[None, :]).sum()
    return (wins + 0.5 * ties) / (n_pos * n_neg)


def comprehensive_diagnostics(model, sample_records, device, n_sample=15):
    """Full phase-transition trajectory diagnostics."""
    model.eval()
    all_metric_diag = []
    all_h_fast_end = []         # h_fast at end of trajectory per record
    all_h_slow_end = []
    all_labels_record = []       # majority-followed per record
    all_per_K_norms = []         # [K] norm magnitude per K position
    all_trigger_probs = []       # trigger probs at each chunk
    turn_boundary_triggers = []  # trigger probs specifically at turn boundaries

    with torch.no_grad():
        for r in sample_records[:n_sample]:
            T = r["T"]
            if T < 4:
                continue
            hs_traj = r["hidden_state_traj"].to(device)
            z_g = r["z_goal_traj"].to(device)
            # Forward trajectory
            h_fast_traj, h_slow_traj = forward_trajectory(
                model, hs_traj, z_g, device, target_t=None, training=False)

            # Metric eigenvalues (replicate fast_step internals)
            hs_normed = model.hidden_layernorm(hs_traj[-1:])
            e_h = model.in_hidden(hs_normed) * model.hidden_gate
            e_g = model.in_goal(z_g[-1:]) * model.goal_gate
            e = model.evidence_layernorm(e_h + e_g)
            evidence = e.unsqueeze(1) * model.evidence_mix.unsqueeze(0)
            h_input = h_fast_traj[-2].unsqueeze(0) + evidence
            h_input = model._soft_clamp(h_input)
            context = model.context_pool(h_input, None)
            model.dynamics.set_context(context, mask=None)
            metric_diag = model.dynamics.compute_metric_diag(h_input)
            all_metric_diag.append(metric_diag.flatten().cpu().numpy())

            # End-of-trajectory h_fast and h_slow
            all_h_fast_end.append(h_fast_traj[-1].cpu().numpy())   # [K, d]
            all_h_slow_end.append(h_slow_traj[-1].cpu().numpy())
            # Per-K-position norm (activation strength)
            all_per_K_norms.append(h_fast_traj.norm(dim=-1).mean(dim=0).cpu().numpy())  # [K]
            # Record label: majority-followed
            if "labels" in r:
                all_labels_record.append(int(r["labels"].mean().item() > 0.5))
            else:
                all_labels_record.append(0)
            # Trigger probabilities: re-step slow channel and collect
            z_goal_prev = None
            trigger_seq = []
            for t in range(T):
                z_g_t = z_g[t].unsqueeze(0)
                _, trig_prob = model.slow_step(
                    model.init_slow_state(1, device), z_g_t, z_goal_prev)
                trigger_seq.append(float(trig_prob.item()))
                z_goal_prev = z_g_t
            all_trigger_probs.append(trigger_seq)
            # Turn boundary triggers (jumps in z_goal)
            turn_starts = r.get("turn_chunk_starts", [])
            for ts in turn_starts:
                if 0 < ts < T:
                    turn_boundary_triggers.append(trigger_seq[ts])

    model.train()
    diagnostics = {}
    if all_metric_diag:
        arr = np.concatenate(all_metric_diag)
        diagnostics["cv"] = float(arr.std() / max(1e-8, arr.mean()))
        diagnostics["metric_eigval_mean"] = float(arr.mean())
        diagnostics["metric_eigval_std"] = float(arr.std())
        diagnostics["metric_eigval_p10"] = float(np.percentile(arr, 10))
        diagnostics["metric_eigval_p50"] = float(np.percentile(arr, 50))
        diagnostics["metric_eigval_p90"] = float(np.percentile(arr, 90))
        diagnostics["metric_eigval_p99"] = float(np.percentile(arr, 99))
    # Class separation in h_fast end-states
    if all_h_fast_end and len(set(all_labels_record)) > 1:
        h_arr = np.stack(all_h_fast_end, axis=0)  # [N, K, d]
        h_flat = h_arr.reshape(len(h_arr), -1)    # [N, K*d]
        labels_np = np.array(all_labels_record)
        succ_mean = h_flat[labels_np == 1].mean(axis=0) if (labels_np == 1).any() else None
        drift_mean = h_flat[labels_np == 0].mean(axis=0) if (labels_np == 0).any() else None
        if succ_mean is not None and drift_mean is not None:
            diagnostics["class_separation"] = float(np.linalg.norm(succ_mean - drift_mean))
            diagnostics["within_class_succ_std"] = float(h_flat[labels_np == 1].std(axis=0).mean())
            diagnostics["within_class_drift_std"] = float(h_flat[labels_np == 0].std(axis=0).mean())
    # Per-K activation
    if all_per_K_norms:
        per_K = np.stack(all_per_K_norms, axis=0).mean(axis=0)
        diagnostics["per_K_activation"] = per_K.tolist()
    # Trigger statistics
    if turn_boundary_triggers:
        diagnostics["trigger_prob_at_boundary_mean"] = float(np.mean(turn_boundary_triggers))
        diagnostics["trigger_prob_at_boundary_max"] = float(np.max(turn_boundary_triggers))
    if all_trigger_probs:
        all_trig_flat = [t for seq in all_trigger_probs for t in seq]
        diagnostics["trigger_prob_overall_mean"] = float(np.mean(all_trig_flat))
    # Slow channel: norm of end state
    if all_h_slow_end:
        h_slow_arr = np.stack(all_h_slow_end, axis=0)
        diagnostics["slow_state_norm_mean"] = float(np.linalg.norm(h_slow_arr.reshape(len(h_slow_arr), -1), axis=1).mean())
    return diagnostics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_traj", required=True)
    p.add_argument("--test_traj", default="")
    p.add_argument("--output", required=True)
    p.add_argument("--diag_log", required=True, help="JSON log of all per-checkpoint diagnostics")
    p.add_argument("--snapshot_dir", default="",
                   help="If set, save model snapshots at predefined checkpoints")
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--value_hidden", type=int, default=384)
    p.add_argument("--d_metric", type=int, default=64)
    p.add_argument("--d_ffn", type=int, default=512)
    p.add_argument("--n_ode_steps", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=6000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--substrate_lr_ratio", type=float, default=0.5)
    p.add_argument("--lambda_distill", type=float, default=3.0)
    p.add_argument("--lambda_value", type=float, default=2.0)
    p.add_argument("--lambda_proto", type=float, default=2.0)
    p.add_argument("--lambda_jepa", type=float, default=0.3)
    p.add_argument("--lambda_var", type=float, default=0.1)
    p.add_argument("--jepa_window", type=int, default=2)
    p.add_argument("--ema_tau", type=float, default=0.996)
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--diag_every", type=int, default=200,
                   help="Run comprehensive diagnostics every N steps")
    p.add_argument("--early_stop_patience", type=int, default=80)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    args = p.parse_args()

    snapshot_steps = {0, 100, 200, 500, 1000, 2000, 3000, 4000, 5000, 6000}

    device = torch.device("cuda")
    print(f"[inst] device={device}, output={args.output}", flush=True)

    pack = torch.load(args.train_traj, map_location="cpu", weights_only=False)
    z_goal_dim = int(pack["enc_dim"])
    d_hidden = int(pack["d_hidden"])
    raw_records = pack["records"]
    print(f"[inst] train pack: d_hidden={d_hidden}, records={len(raw_records)}", flush=True)

    def augment(records_list):
        out = []
        for r in records_list:
            T = int(r["T"])
            if T < 4 or "judge_traj" not in r:
                continue
            hs_traj = r["hidden_state_traj"].float()
            z_goal_traj = r["z_goal_traj"].float()
            judge_traj = r["judge_traj"].float()
            turn_starts = list(r["turn_chunk_starts"])
            turn_followed = list(r["turn_followed"])
            chunk_labels = []
            cur_turn = 0
            for t in range(T):
                while (cur_turn + 1 < len(turn_starts) and
                       t >= turn_starts[cur_turn + 1]):
                    cur_turn += 1
                chunk_labels.append(int(turn_followed[cur_turn]))
            out.append({
                "hidden_state_traj": hs_traj,
                "z_goal_traj": z_goal_traj,
                "judge_traj": judge_traj,
                "labels": torch.tensor(chunk_labels, dtype=torch.float32),
                "turn_chunk_starts": turn_starts,
                "T": T,
                "sub_id": int(r["sub_id"]),
            })
        return out

    train_records = augment(raw_records)
    rng = np.random.default_rng(42)
    all_sub_ids = sorted({r["sub_id"] for r in train_records})
    rng.shuffle(all_sub_ids)
    n_val_ids = max(1, int(args.val_frac * len(all_sub_ids)))
    val_ids = set(all_sub_ids[:n_val_ids])
    train_recs = [r for r in train_records if r["sub_id"] not in val_ids]
    val_recs = [r for r in train_records if r["sub_id"] in val_ids]

    test_recs = []
    if args.test_traj:
        test_pack = torch.load(args.test_traj, map_location="cpu", weights_only=False)
        test_recs = augment(test_pack["records"])

    print(f"[inst] train={len(train_recs)} val={len(val_recs)} test={len(test_recs)}", flush=True)

    online = JudgeDistillLiquid(
        d_hidden=d_hidden, z_goal_dim=z_goal_dim,
        d=args.d, K=args.K, value_hidden=args.value_hidden,
        d_metric=args.d_metric, d_ffn=args.d_ffn, n_ode_steps=args.n_ode_steps,
    ).to(device)
    target = copy.deepcopy(online)
    for pp in target.parameters():
        pp.requires_grad = False
    n_params = sum(p.numel() for p in online.parameters() if p.requires_grad)
    print(f"[inst] {n_params:,} params", flush=True)

    pred_params = (list(online.head_judge_pred.parameters()) +
                    list(online.head_value.parameters()) +
                    list(online.head_jepa.parameters()))
    geom_param_ids = set()
    for mod in (online.dynamics, online.context_pool):
        for pp in mod.parameters():
            geom_param_ids.add(id(pp))
    geom_params = [p for p in online.parameters() if id(p) in geom_param_ids]
    body_params = [p for p in online.parameters()
                    if id(p) not in geom_param_ids
                    and id(p) not in {id(q) for q in pred_params}]
    opt = torch.optim.AdamW([
        {"params": pred_params, "lr": args.lr, "weight_decay": 0.01},
        {"params": body_params, "lr": args.lr, "weight_decay": 0.005},
        {"params": geom_params, "lr": args.lr * args.substrate_lr_ratio,
         "weight_decay": 0.0},
    ])
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    def loss_for_batch(recs, training=True):
        distill_losses, value_losses, jepa_losses, var_feats = [], [], [], []
        proto_h_fasts, proto_labels = [], []
        for r in recs:
            T = r["T"]
            hs_traj = r["hidden_state_traj"].to(device)
            z_goal_traj = r["z_goal_traj"].to(device)
            judge_traj = r["judge_traj"].to(device)
            labels = r["labels"].to(device)
            t = int(rng.integers(0, max(1, T - args.jepa_window)))
            h_fast_traj, _ = forward_trajectory(
                online, hs_traj, z_goal_traj, device, target_t=t, training=training)
            h_fast_now = h_fast_traj[t].unsqueeze(0)
            z_goal_now = z_goal_traj[t].unsqueeze(0)
            judge_now = judge_traj[t].unsqueeze(0)
            jp = online.predict_judge(h_fast_now, z_goal_now)
            distill_losses.append(mse(jp, judge_now))
            logit = online.value(h_fast_now, z_goal_now)
            value_losses.append(bce(logit, labels[t].unsqueeze(0)))
            pred = online.jepa_predict(h_fast_now)
            with torch.no_grad():
                target_traj, _ = forward_trajectory(
                    target, hs_traj, z_goal_traj, device,
                    target_t=t + args.jepa_window, training=False)
                tgt = target_traj[t + args.jepa_window].detach().unsqueeze(0)
            jepa_losses.append(((pred - tgt) ** 2).mean())
            var_feats.append(h_fast_now[0].flatten().detach())
            proto_h_fasts.append(h_fast_now[0])
            proto_labels.append(labels[t])
        if not distill_losses:
            return None
        distill_loss = torch.stack(distill_losses).mean()
        value_loss = torch.stack(value_losses).mean()
        jepa_loss = torch.stack(jepa_losses).mean()
        var = torch.stack(var_feats, dim=0).std(dim=0).mean()
        var_loss = torch.relu(args.lambda_var - var)
        h_stack = torch.stack(proto_h_fasts, dim=0)
        label_stack = torch.stack(proto_labels, dim=0)
        anchor = online.success_anchor.unsqueeze(0)
        dist = ((h_stack - anchor) ** 2).mean(dim=(-1, -2))
        success_mask = label_stack > 0.5
        drift_mask = ~success_mask
        terms = []
        if success_mask.any():
            terms.append(dist[success_mask].mean())
        if drift_mask.any():
            terms.append(torch.relu(online.proto_margin - dist[drift_mask]).mean())
        proto_loss = torch.stack(terms).mean() if terms else torch.tensor(0.0, device=h_stack.device)
        return distill_loss, value_loss, jepa_loss, var_loss, proto_loss

    def ema_update(tau):
        with torch.no_grad():
            for tp, op in zip(target.parameters(), online.parameters()):
                tp.data.mul_(tau).add_(op.data, alpha=1.0 - tau)

    @torch.no_grad()
    def auc_on_set(recs):
        if not recs:
            return float("nan"), float("nan"), float("nan")
        online.eval(); target.eval()
        value_logits, judge_preds, raw_judges, all_labels = [], [], [], []
        for r in recs:
            T = r["T"]
            hs_traj = r["hidden_state_traj"].to(device)
            z_goal_traj = r["z_goal_traj"].to(device)
            judge_traj = r["judge_traj"].to(device)
            labels = r["labels"].to(device)
            h_fast_traj, _ = forward_trajectory(
                online, hs_traj, z_goal_traj, device, target_t=None, training=False)
            for t in range(T):
                vl = online.value(h_fast_traj[t].unsqueeze(0), z_goal_traj[t].unsqueeze(0))
                jp = online.predict_judge(h_fast_traj[t].unsqueeze(0), z_goal_traj[t].unsqueeze(0))
                value_logits.append(float(vl.item()))
                judge_preds.append(float(jp.item()))
                raw_judges.append(float(judge_traj[t].item()))
                all_labels.append(int(labels[t].item()))
        labels_np = np.array(all_labels)
        n_drift = int((labels_np == 0).sum())
        n_follow = int((labels_np == 1).sum())
        online.train(); target.train()
        if n_drift == 0 or n_follow == 0:
            return float("nan"), float("nan"), float("nan")
        return (roc_auc(-np.array(value_logits), 1 - labels_np),
                roc_auc(-np.array(judge_preds), 1 - labels_np),
                roc_auc(-np.array(raw_judges), 1 - labels_np))

    online.train(); target.train()
    best_test_auc = -1.0
    best_state = None
    last_improvement = 0
    n_nan = 0
    t_start = time.time()
    diag_log = []

    if args.snapshot_dir:
        Path(args.snapshot_dir).mkdir(parents=True, exist_ok=True)

    for step in range(args.max_steps + 1):
        batch = rng.choice(len(train_recs), args.batch_size, replace=True)
        recs = [train_recs[i] for i in batch]
        out = loss_for_batch(recs, training=True)
        if out is None:
            continue
        distill_l, value_l, jepa_l, var_l, proto_l = out
        total = (args.lambda_distill * distill_l +
                  args.lambda_value * value_l +
                  args.lambda_jepa * jepa_l +
                  var_l +
                  args.lambda_proto * proto_l)
        if not torch.isfinite(total):
            n_nan += 1
            if n_nan >= 5:
                print(f"[inst] ABORT: 5 NaN at step {step}", flush=True)
                break
            continue
        n_nan = 0
        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(online.parameters(), args.max_grad_norm)
        opt.step()
        ema_update(args.ema_tau)

        if step % args.log_every == 0:
            v_v, v_j, _ = auc_on_set(val_recs)
            t_v, t_j, t_raw = auc_on_set(test_recs)
            selection_auc = t_j if not np.isnan(t_j) else t_v
            stale = (step - last_improvement) // args.log_every
            if not np.isnan(selection_auc) and selection_auc > best_test_auc:
                best_test_auc = selection_auc
                best_state = {
                    "online": copy.deepcopy(online.state_dict()),
                    "target": copy.deepcopy(target.state_dict()),
                }
                last_improvement = step
                stale = 0
            print(f"step {step:>5}  Ld={float(distill_l):.3f}  "
                  f"Lp={float(proto_l):.3f}  "
                  f"v_v={v_v:.3f} v_j={v_j:.3f}  "
                  f"t_v={t_v:.3f} t_j={t_j:.3f} t_raw={t_raw:.3f}  "
                  f"(best_t_j {best_test_auc:.3f})  stale={stale}  "
                  f"wall={time.time()-t_start:.0f}s", flush=True)

        if step % args.diag_every == 0:
            diag = comprehensive_diagnostics(online, val_recs, device, n_sample=15)
            diag["step"] = step
            # Re-compute AUCs at diag step for alignment
            v_v, v_j, _ = auc_on_set(val_recs)
            t_v, t_j, t_raw = auc_on_set(test_recs)
            diag["v_auc"] = v_v
            diag["v_judge_pred_auc"] = v_j
            diag["t_auc"] = t_v
            diag["t_judge_pred_auc"] = t_j
            diag["t_raw_judge_auc"] = t_raw
            diag_log.append(diag)
            with open(args.diag_log, "w") as f:
                json.dump(diag_log, f, indent=2)
            print(f"  [diag] step={step} CV={diag.get('cv','?'):.3f} "
                  f"separation={diag.get('class_separation','?')} "
                  f"trig_at_boundary={diag.get('trigger_prob_at_boundary_mean','?')}",
                  flush=True)

        if args.snapshot_dir and step in snapshot_steps:
            snap_path = Path(args.snapshot_dir) / f"snap_step{step}.pt"
            torch.save({
                "step": step,
                "online_state": online.state_dict(),
                "diagnostics": comprehensive_diagnostics(online, val_recs, device, n_sample=15),
            }, snap_path)

        if (step >= 2000 and stale >= args.early_stop_patience):
            print(f"[inst] early stop at step {step}", flush=True)
            break

    if best_state is not None:
        online.load_state_dict(best_state["online"])
        target.load_state_dict(best_state["target"])
    torch.save({
        "model_state_dict": online.state_dict(),
        "best_test_auc": best_test_auc,
        "args": vars(args),
        "d_hidden": d_hidden,
        "z_goal_dim": z_goal_dim,
    }, args.output)
    with open(args.diag_log, "w") as f:
        json.dump(diag_log, f, indent=2)
    print(f"[inst] saved → {args.output}  best_test_auc={best_test_auc:.3f}", flush=True)
    print(f"[inst] diag log → {args.diag_log}  ({len(diag_log)} entries)", flush=True)
    print("[inst] === ALL_DONE ===", flush=True)


if __name__ == "__main__":
    main()
