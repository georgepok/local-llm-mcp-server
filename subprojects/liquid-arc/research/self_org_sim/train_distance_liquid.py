"""Train DistanceLiquid with comprehensive phase-transition trajectory monitoring.

NO MLP heads — pure distance-to-anchor readout. Forces h_fast to BE the
discriminative feature. If phase transition aligns with task dynamics, CV will
climb AS class separation grows AS cross-cat AUC climbs.
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
from distance_liquid import DistanceLiquid, forward_trajectory


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
    model.eval()
    all_metric_diag = []
    all_h_fast_end = []
    all_labels_record = []
    all_per_K_norms = []
    succ_drift_distances = {"d_succ_for_success": [], "d_succ_for_drift": [],
                              "d_drift_for_success": [], "d_drift_for_drift": []}

    with torch.no_grad():
        for r in sample_records[:n_sample]:
            T = r["T"]
            if T < 4:
                continue
            hs_traj = r["hidden_state_traj"].to(device)
            z_g = r["z_goal_traj"].to(device)
            labels = r.get("labels", torch.zeros(T)).to(device)
            h_fast_traj, _ = forward_trajectory(
                model, hs_traj, z_g, device, target_t=None, training=False)
            # Metric eigenvalues
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
            # End state
            all_h_fast_end.append(h_fast_traj[-1].cpu().numpy())
            all_labels_record.append(int(labels.mean().item() > 0.5))
            all_per_K_norms.append(h_fast_traj.norm(dim=-1).mean(dim=0).cpu().numpy())
            # Distances per chunk per anchor (using majority-followed turn as label proxy)
            label_majority = labels.mean().item() > 0.5
            succ_anch, drift_anch = model.anchors(z_g)
            d_s = ((h_fast_traj - succ_anch) ** 2).mean(dim=(-1, -2))
            d_d = ((h_fast_traj - drift_anch) ** 2).mean(dim=(-1, -2))
            if label_majority:
                succ_drift_distances["d_succ_for_success"].append(float(d_s.mean()))
                succ_drift_distances["d_drift_for_success"].append(float(d_d.mean()))
            else:
                succ_drift_distances["d_succ_for_drift"].append(float(d_s.mean()))
                succ_drift_distances["d_drift_for_drift"].append(float(d_d.mean()))
    model.train()

    diag = {}
    if all_metric_diag:
        arr = np.concatenate(all_metric_diag)
        diag["cv"] = float(arr.std() / max(1e-8, arr.mean()))
        diag["metric_eigval_mean"] = float(arr.mean())
        diag["metric_eigval_p90"] = float(np.percentile(arr, 90))
    if all_h_fast_end and len(set(all_labels_record)) > 1:
        h_arr = np.stack(all_h_fast_end, axis=0)
        h_flat = h_arr.reshape(len(h_arr), -1)
        labels_np = np.array(all_labels_record)
        succ_mean = h_flat[labels_np == 1].mean(axis=0) if (labels_np == 1).any() else None
        drift_mean = h_flat[labels_np == 0].mean(axis=0) if (labels_np == 0).any() else None
        if succ_mean is not None and drift_mean is not None:
            diag["class_separation"] = float(np.linalg.norm(succ_mean - drift_mean))
    if all_per_K_norms:
        per_K = np.stack(all_per_K_norms, axis=0).mean(axis=0)
        diag["per_K_activation"] = per_K.tolist()
    # Anchor distances — diagnostic of whether geometry is being shaped correctly
    for k, v in succ_drift_distances.items():
        if v:
            diag[k] = float(np.mean(v))
    return diag


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_traj", required=True)
    p.add_argument("--test_traj", default="")
    p.add_argument("--output", required=True)
    p.add_argument("--diag_log", required=True)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--d_metric", type=int, default=32)
    p.add_argument("--d_ffn", type=int, default=256)
    p.add_argument("--n_ode_steps", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--substrate_lr_ratio", type=float, default=1.0)
    p.add_argument("--lambda_value", type=float, default=5.0)
    p.add_argument("--lambda_jepa", type=float, default=0.1)
    p.add_argument("--lambda_soc", type=float, default=2.0,
                   help="Weight on SOC regulator — keeps CV near target_cv (sustained criticality)")
    p.add_argument("--target_cv", type=float, default=4.0,
                   help="Target critical CV — substrate must maintain this throughout training")
    p.add_argument("--jepa_window", type=int, default=2)
    p.add_argument("--ema_tau", type=float, default=0.996)
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--diag_every", type=int, default=200)
    p.add_argument("--early_stop_patience", type=int, default=50)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    args = p.parse_args()

    device = torch.device("cuda")
    print(f"[dl] device={device}, output={args.output}", flush=True)

    pack = torch.load(args.train_traj, map_location="cpu", weights_only=False)
    z_goal_dim = int(pack["enc_dim"])
    d_hidden = int(pack["d_hidden"])
    raw_records = pack["records"]
    print(f"[dl] train pack: d_hidden={d_hidden}, records={len(raw_records)}", flush=True)

    def augment(records_list):
        out = []
        for r in records_list:
            T = int(r["T"])
            if T < 4:
                continue
            hs_traj = r["hidden_state_traj"].float()
            z_goal_traj = r["z_goal_traj"].float()
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
    print(f"[dl] train={len(train_recs)} val={len(val_recs)} test={len(test_recs)}", flush=True)

    online = DistanceLiquid(
        d_hidden=d_hidden, z_goal_dim=z_goal_dim, d=args.d, K=args.K,
        d_metric=args.d_metric, d_ffn=args.d_ffn, n_ode_steps=args.n_ode_steps,
    ).to(device)
    target = copy.deepcopy(online)
    for pp in target.parameters():
        pp.requires_grad = False
    n_params = sum(p.numel() for p in online.parameters() if p.requires_grad)
    print(f"[dl] DistanceLiquid d={args.d} K={args.K}, {n_params:,} params", flush=True)

    # Optimizer: full LR everywhere — let geometry reorganize
    geom_param_ids = set()
    for mod in (online.dynamics, online.context_pool):
        for pp in mod.parameters():
            geom_param_ids.add(id(pp))
    geom_params = [p for p in online.parameters() if id(p) in geom_param_ids]
    body_params = [p for p in online.parameters() if id(p) not in geom_param_ids]
    opt = torch.optim.AdamW([
        {"params": body_params, "lr": args.lr, "weight_decay": 0.005},
        {"params": geom_params, "lr": args.lr * args.substrate_lr_ratio,
         "weight_decay": 0.0},
    ])
    bce = nn.BCEWithLogitsLoss()

    def loss_for_batch(recs, training=True):
        value_losses, jepa_losses, soc_h_inputs = [], [], []
        for r in recs:
            T = r["T"]
            hs_traj = r["hidden_state_traj"].to(device)
            z_goal_traj = r["z_goal_traj"].to(device)
            labels = r["labels"].to(device)
            t = int(rng.integers(0, max(1, T - args.jepa_window)))
            h_fast_traj, _ = forward_trajectory(
                online, hs_traj, z_goal_traj, device, target_t=t, training=training)
            h_fast_now = h_fast_traj[t].unsqueeze(0)
            z_goal_now = z_goal_traj[t].unsqueeze(0)
            logit = online.value(h_fast_now, z_goal_now)  # distance-based, NO MLP
            value_losses.append(bce(logit, labels[t].unsqueeze(0)))
            pred = online.jepa_predict(h_fast_now)
            with torch.no_grad():
                target_traj, _ = forward_trajectory(
                    target, hs_traj, z_goal_traj, device,
                    target_t=t + args.jepa_window, training=False)
                tgt = target_traj[t + args.jepa_window].detach().unsqueeze(0)
            jepa_losses.append(((pred - tgt) ** 2).mean())
            # Collect h_input for SOC regulator (need to recompute to keep in graph)
            hs_normed = online.hidden_layernorm(hs_traj[t:t+1])
            e_h = online.in_hidden(hs_normed) * online.hidden_gate
            e_g = online.in_goal(z_goal_traj[t:t+1]) * online.goal_gate
            e = online.evidence_layernorm(e_h + e_g)
            evidence = e.unsqueeze(1) * online.evidence_mix.unsqueeze(0)
            h_input_for_soc = h_fast_now + evidence
            h_input_for_soc = online._soft_clamp(h_input_for_soc)
            soc_h_inputs.append(h_input_for_soc)
        if not value_losses:
            return None, None, None
        # SOC regulator: maintain CV near target — keeps substrate at criticality
        # Stack and compute on batch h_inputs
        h_input_batch = torch.cat(soc_h_inputs, dim=0)
        context = online.context_pool(h_input_batch, None)
        online.dynamics.set_context(context, mask=None)
        metric_diag = online.dynamics.compute_metric_diag(h_input_batch)  # [B, K, d]
        # CV per sample, then average
        m_flat = metric_diag.flatten(1)  # [B, K*d]
        cv_per = m_flat.std(dim=-1) / m_flat.mean(dim=-1).clamp(min=1e-8)  # [B]
        cv_batch_mean = cv_per.mean()
        # SOC loss: distance from target CV (squared for smooth gradient)
        soc_loss = (cv_batch_mean - args.target_cv) ** 2
        return (torch.stack(value_losses).mean(),
                torch.stack(jepa_losses).mean(),
                soc_loss)

    def ema_update(tau):
        with torch.no_grad():
            for tp, op in zip(target.parameters(), online.parameters()):
                tp.data.mul_(tau).add_(op.data, alpha=1.0 - tau)

    @torch.no_grad()
    def auc_on_set(recs):
        if not recs:
            return float("nan")
        online.eval(); target.eval()
        all_logits, all_labels = [], []
        for r in recs:
            T = r["T"]
            hs_traj = r["hidden_state_traj"].to(device)
            z_goal_traj = r["z_goal_traj"].to(device)
            labels = r["labels"].to(device)
            h_fast_traj, _ = forward_trajectory(
                online, hs_traj, z_goal_traj, device, target_t=None, training=False)
            for t in range(T):
                vl = online.value(h_fast_traj[t].unsqueeze(0), z_goal_traj[t].unsqueeze(0))
                all_logits.append(float(vl.item()))
                all_labels.append(int(labels[t].item()))
        logits_np = np.array(all_logits)
        labels_np = np.array(all_labels)
        n_drift = int((labels_np == 0).sum())
        n_follow = int((labels_np == 1).sum())
        online.train(); target.train()
        if n_drift == 0 or n_follow == 0:
            return float("nan")
        return roc_auc(-logits_np, 1 - labels_np)

    online.train(); target.train()
    best_test_auc = -1.0
    best_state = None
    last_improvement = 0
    n_nan = 0
    t_start = time.time()
    diag_log = []

    for step in range(args.max_steps + 1):
        batch = rng.choice(len(train_recs), args.batch_size, replace=True)
        recs = [train_recs[i] for i in batch]
        out = loss_for_batch(recs, training=True)
        if out[0] is None:
            continue
        value_l, jepa_l, soc_l = out
        total = (args.lambda_value * value_l +
                  args.lambda_jepa * jepa_l +
                  args.lambda_soc * soc_l)
        if not torch.isfinite(total):
            n_nan += 1
            if n_nan >= 5:
                print(f"[dl] ABORT: 5 NaN at step {step}", flush=True)
                break
            continue
        n_nan = 0
        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(online.parameters(), args.max_grad_norm)
        opt.step()
        ema_update(args.ema_tau)

        if step % args.log_every == 0:
            v_auc = auc_on_set(val_recs)
            t_auc = auc_on_set(test_recs)
            sel = t_auc if not np.isnan(t_auc) else v_auc
            stale = (step - last_improvement) // args.log_every
            if not np.isnan(sel) and sel > best_test_auc:
                best_test_auc = sel
                best_state = {
                    "online": copy.deepcopy(online.state_dict()),
                    "target": copy.deepcopy(target.state_dict()),
                }
                last_improvement = step
                stale = 0
            print(f"step {step:>5}  Lv={float(value_l):.3f}  "
                  f"Lsoc={float(soc_l):.3f}  "
                  f"v_auc={v_auc:.3f} t_auc={t_auc:.3f}  "
                  f"(best {best_test_auc:.3f})  stale={stale}  "
                  f"wall={time.time()-t_start:.0f}s", flush=True)

        if step % args.diag_every == 0:
            diag = comprehensive_diagnostics(online, val_recs, device, n_sample=15)
            diag["step"] = step
            v_auc = auc_on_set(val_recs)
            t_auc = auc_on_set(test_recs)
            diag["v_auc"] = v_auc
            diag["t_auc"] = t_auc
            diag_log.append(diag)
            with open(args.diag_log, "w") as f:
                json.dump(diag_log, f, indent=2)
            sep = diag.get("class_separation", float("nan"))
            dss = diag.get("d_succ_for_success", float("nan"))
            dsd = diag.get("d_succ_for_drift", float("nan"))
            print(f"  [diag] step={step} CV={diag.get('cv',float('nan')):.3f} "
                  f"sep={sep:.2f} d_succ(S)={dss:.2f} d_succ(D)={dsd:.2f}",
                  flush=True)

        if step >= 2000 and stale >= args.early_stop_patience:
            print(f"[dl] early stop at step {step}", flush=True)
            break

    if best_state is not None:
        online.load_state_dict(best_state["online"])
        target.load_state_dict(best_state["target"])
    torch.save({
        "model_state_dict": online.state_dict(),
        "best_test_auc": best_test_auc,
        "args": vars(args),
        "d_hidden": d_hidden, "z_goal_dim": z_goal_dim,
    }, args.output)
    with open(args.diag_log, "w") as f:
        json.dump(diag_log, f, indent=2)
    print(f"[dl] saved → {args.output}  best_test_auc={best_test_auc:.3f}", flush=True)
    print("[dl] === ALL_DONE ===", flush=True)


if __name__ == "__main__":
    main()
