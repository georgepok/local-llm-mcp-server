"""Two-substrate (slow + fast) JEPA training with coupling readout.

Per project_dynamical_goal_object framework:
- SLOW substrate (goal-manifold): reads z_lang + z_goal, leaky integrator with
  trigger-gated update boost. Long effective τ. Represents "what task am I on."
- FAST substrate (state): existing JEPA_LGT_Proprio. Reads z_vl + state + chunk.
  Short τ. Represents "where am I in execution."
- COUPLING readout: predicts h_fast[t+W] from (h_slow[t], h_fast[t], chunks).
  Loss = MSE on h_fast_target. High coupling = goal predicts state evolution =
  legitimate. Low coupling = drift = decoupled flows.

Total loss:
  L = L_jepa (fast self-pred) + L_succ (outcome aux) + λ_coupling * L_coupling

The coupling loss is the "deviation readout" — its inverse is the actual signal.

Usage on Spark:
  python train_substrate_twoflow.py \
    --traj_files /tmp/traj_jepa_libero10_s{10,20,30,40}.pt \
    --substrate_ckpt /tmp/substrate_dynamics_corr.pt \
    --output /tmp/substrate_twoflow.pt
"""
from __future__ import annotations
import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

_LA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_LA_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from liquid_goal_tracker_proprio import JEPA_LGT_Proprio  # type: ignore

torch.set_float32_matmul_precision("high")


def vicreg_var_loss(z, target_std=1.0):
    if z.shape[0] < 2:
        return torch.tensor(0.0, device=z.device)
    var = z.var(dim=0, unbiased=False)
    std = torch.sqrt(var + 1e-6)
    return torch.mean(torch.clamp(target_std - std, min=0.0))


def load_records_with_inputs(traj_files):
    all_records = []
    for fp in traj_files:
        ck = torch.load(fp, map_location="cpu", weights_only=False)
        for r in ck["records"]:
            if "z_vl_traj" in r and "chunk_traj" in r and "z_lang_traj" in r:
                all_records.append(r)
        print(f"  {fp}: {len([r for r in ck['records'] if 'z_vl_traj' in r])} records",
              flush=True)
    return all_records


def forward_two_flow(substrate, record, device, target_t):
    """Forward BOTH slow and fast substrate to target_t with z_lang_prev passed
    to slow_step for delta-detection. Truncated BPTT: gradient only at target_t.
    """
    z_vl = record["z_vl_traj"].float().to(device)
    z_lang = record["z_lang_traj"].float().to(device)
    state8 = record["state8_traj"].float().to(device)
    chunks = record["chunk_traj"].float().to(device)
    z_goal = record["z_goal"].float().to(device).unsqueeze(0)
    T = z_vl.shape[0]

    h_slow = substrate.init_slow_state(1, device)
    h_fast = substrate.init_state(1, device)
    z_lang_prev = None
    h_slow_traj, h_fast_traj = [], []
    end_t = min(target_t + 1, T)
    for t in range(end_t):
        z_t = z_vl[t].unsqueeze(0)
        z_l = z_lang[t].unsqueeze(0)
        s_t = state8[t].unsqueeze(0)
        ch_t = chunks[t].unsqueeze(0)
        if t == target_t:
            h_slow, _ = substrate.slow_step(h_slow, z_l, z_goal, z_lang_prev)
            h_fast, _, _, _ = substrate.step(h_fast, z_t, z_goal, ch_t, s_t,
                                                z_lang_t=z_l)
        else:
            with torch.no_grad():
                h_slow, _ = substrate.slow_step(h_slow, z_l, z_goal, z_lang_prev)
                h_fast, _, _, _ = substrate.step(h_fast, z_t, z_goal, ch_t, s_t,
                                                    z_lang_t=z_l)
            h_slow = h_slow.detach()
            h_fast = h_fast.detach()
        z_lang_prev = z_l.detach()
        h_slow_traj.append(h_slow[0])
        h_fast_traj.append(h_fast[0])
    return torch.stack(h_slow_traj, dim=0), torch.stack(h_fast_traj, dim=0)


@torch.no_grad()
def forward_two_flow_no_grad(substrate, record, device, target_t):
    z_vl = record["z_vl_traj"].float().to(device)
    z_lang = record["z_lang_traj"].float().to(device)
    state8 = record["state8_traj"].float().to(device)
    chunks = record["chunk_traj"].float().to(device)
    z_goal = record["z_goal"].float().to(device).unsqueeze(0)
    T = z_vl.shape[0]
    h_slow = substrate.init_slow_state(1, device)
    h_fast = substrate.init_state(1, device)
    z_lang_prev = None
    h_slow_traj, h_fast_traj = [], []
    end_t = min(target_t + 1, T)
    for t in range(end_t):
        z_t = z_vl[t].unsqueeze(0)
        z_l = z_lang[t].unsqueeze(0)
        s_t = state8[t].unsqueeze(0)
        ch_t = chunks[t].unsqueeze(0)
        h_slow, _ = substrate.slow_step(h_slow, z_l, z_goal, z_lang_prev)
        h_fast, _, _, _ = substrate.step(h_fast, z_t, z_goal, ch_t, s_t,
                                            z_lang_t=z_l)
        z_lang_prev = z_l
        h_slow_traj.append(h_slow[0])
        h_fast_traj.append(h_fast[0])
    return torch.stack(h_slow_traj, dim=0), torch.stack(h_fast_traj, dim=0)


def ema_update(target_substrate, online_substrate, tau: float):
    with torch.no_grad():
        for tp, op in zip(target_substrate.parameters(),
                            online_substrate.parameters()):
            tp.data.mul_(tau).add_(op.data, alpha=1.0 - tau)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traj_files", required=True)
    p.add_argument("--substrate_ckpt", required=True)
    p.add_argument("--output", default="/tmp/substrate_twoflow.pt")
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--min_chunks", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--substrate_lr_ratio", type=float, default=0.05)
    p.add_argument("--ema_tau", type=float, default=0.996)
    p.add_argument("--lambda_var", type=float, default=0.1)
    p.add_argument("--lambda_succ", type=float, default=5.0)
    p.add_argument("--lambda_coupling", type=float, default=2.0,
                   help="Weight on coupling JEPA loss (predict fast from slow+fast)")
    p.add_argument("--pos_weight_succ", type=float, default=2.0)
    p.add_argument("--var_target", type=float, default=1.0)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--split_mode", choices=["trajectory", "task_id"],
                    default="task_id")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--early_stop_patience", type=int, default=20)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--unfreeze_geometric", action="store_true",
                   help="Also unfreeze ContinuousDynamics + ContextPool with slower LR")
    p.add_argument("--geometric_lr_ratio", type=float, default=0.01,
                   help="LR multiplier for geometric params when unfrozen (default 0.01 = 100× slower)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[twoflow] device={device}, output={args.output}", flush=True)

    ck = torch.load(args.substrate_ckpt, map_location=device, weights_only=False)
    sa = ck["args"]
    d_sub = sa.get("d_substrate", 64)
    K_bel = sa.get("K_belief", 4)

    def build_substrate():
        s = JEPA_LGT_Proprio(
            z_vl_dim=ck["z_vl_dim"], action_dim=ck["action_dim"],
            horizon=ck["horizon"], state_dim=ck["state_dim"],
            d=d_sub, K=K_bel, n_tok_per_k=sa.get("n_tok_per_k", 1),
        ).to(device)
        s.load_state_dict(ck["substrate_state_dict"], strict=False)
        s.use_evidence_layernorm = True
        s.h_input_clamp = 50.0
        return s

    online = build_substrate()
    target = copy.deepcopy(online)
    for pp in target.parameters():
        pp.requires_grad = False
    print(f"[twoflow] online + target substrate loaded; K={K_bel} d={d_sub}",
          flush=True)
    print(f"[twoflow] slow channel: K_slow={online.K_slow} d_slow={online.d_slow}, "
          f"alpha={float(online.slow_alpha):.3f} trigger_boost={online.trigger_boost}",
          flush=True)

    # Unfreeze: fast body (in_*, gates, layernorm) + slow branch + all relevant heads.
    # Keep frozen: dynamics, context_pool, all aux heads not used here.
    for pp in online.parameters():
        pp.requires_grad = False
    safe_modules = (
        online.in_z, online.in_goal, online.in_delta,
        online.in_action, online.in_state, online.in_lang,
        online.head_jepa_predictor, online.head_success_predictor,
        online.evidence_layernorm,
        # Slow branch:
        online.slow_in_lang, online.slow_in_goal,
        online.slow_layernorm, online.head_trigger,
        # Coupling readout:
        online.head_coupling_predictor,
    )
    for mod in safe_modules:
        for pp in mod.parameters():
            pp.requires_grad = True
    for name in ("action_gate", "goal_gate", "delta_gate", "state_gate",
                  "lang_gate", "init_belief", "evidence_mix",
                  "slow_init_belief", "slow_alpha"):
        getattr(online, name).requires_grad = True
    geometric_modules = []
    if args.unfreeze_geometric:
        geometric_modules = [online.dynamics, online.context_pool]
        for mod in geometric_modules:
            for pp in mod.parameters():
                pp.requires_grad = True
    n_trainable = sum(pp.numel() for pp in online.parameters() if pp.requires_grad)
    n_geom = sum(pp.numel() for mod in geometric_modules for pp in mod.parameters())
    freeze_state = (f"geometric UNFROZEN ({n_geom:,} geom params, "
                     f"lr×{args.geometric_lr_ratio})"
                     if args.unfreeze_geometric else "dynamics + context_pool FROZEN")
    print(f"[twoflow] online trainable: fast body + slow branch + coupling "
          f"({n_trainable:,} params); {freeze_state}", flush=True)

    records = load_records_with_inputs([s.strip() for s in args.traj_files.split(",")])
    records = [r for r in records
                if r["h_goal_traj"].shape[0] >= args.min_chunks + args.window]
    print(f"[twoflow] {len(records)} records after min_chunks filter", flush=True)

    rng = np.random.default_rng(42)
    if args.split_mode == "task_id":
        all_sub_ids = sorted({int(r["sub_id"]) for r in records})
        rng.shuffle(all_sub_ids)
        n_val_ids = max(1, int(args.val_frac * len(all_sub_ids)))
        val_ids = set(all_sub_ids[:n_val_ids])
        train_ids = set(all_sub_ids[n_val_ids:])
        train_records = [r for r in records if int(r["sub_id"]) in train_ids]
        val_records = [r for r in records if int(r["sub_id"]) in val_ids]
        print(f"[twoflow] task_id split: train={sorted(train_ids)} "
              f"({len(train_records)}) / val={sorted(val_ids)} ({len(val_records)})",
              flush=True)
    else:
        idx = rng.permutation(len(records))
        n_val = max(1, int(args.val_frac * len(records)))
        val_records = [records[i] for i in idx[:n_val]]
        train_records = [records[i] for i in idx[n_val:]]

    pred_params = [p for p in online.head_jepa_predictor.parameters() if p.requires_grad]
    coupling_params = [p for p in online.head_coupling_predictor.parameters()
                         if p.requires_grad]
    geometric_param_ids = set()
    if args.unfreeze_geometric:
        for mod in (online.dynamics, online.context_pool):
            for pp in mod.parameters():
                geometric_param_ids.add(id(pp))
    geometric_params = [p for p in online.parameters()
                          if id(p) in geometric_param_ids and p.requires_grad]
    body_params = [p for n, p in online.named_parameters()
                    if not n.startswith("head_jepa_predictor")
                    and not n.startswith("head_coupling_predictor")
                    and id(p) not in geometric_param_ids
                    and p.requires_grad]
    opt_groups = [
        {"params": pred_params, "lr": args.lr},
        {"params": coupling_params, "lr": args.lr},
        {"params": body_params, "lr": args.lr * args.substrate_lr_ratio},
    ]
    if geometric_params:
        opt_groups.append({"params": geometric_params,
                            "lr": args.lr * args.geometric_lr_ratio})
    opt = torch.optim.AdamW(opt_groups, weight_decay=0.0)

    pos_w_succ = torch.tensor([args.pos_weight_succ], device=device)
    bce_succ = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w_succ)

    def loss_for_records(recs, training=True):
        pred_losses, var_targets, succ_logits, succ_labels = [], [], [], []
        coupling_losses = []
        for r in recs:
            T = r["h_goal_traj"].shape[0]
            if T < args.window + 2:
                continue
            t = int(rng.integers(0, T - args.window))
            if training:
                h_slow_traj, h_fast_traj = forward_two_flow(online, r, device, t)
            else:
                h_slow_traj, h_fast_traj = forward_two_flow_no_grad(online, r, device, t)
            h_slow_now = h_slow_traj[t].unsqueeze(0)
            h_fast_now = h_fast_traj[t].unsqueeze(0)

            # Target: fast trajectory from EMA-tracked target substrate
            _, h_fast_target_traj = forward_two_flow_no_grad(target, r, device,
                                                                 t + args.window)
            h_fast_future_target = h_fast_target_traj[t + args.window].detach().unsqueeze(0)

            chunks_at_t = r["chunk_traj"][t].float().to(device).flatten().unsqueeze(0)

            # JEPA on fast (existing predictor)
            pred = online.jepa_predict_future_h_goal(h_fast_now, chunks_at_t)
            pred_losses.append(((pred - h_fast_future_target) ** 2).mean())

            # Coupling prediction: h_fast_future from (h_slow, chunks) ONLY — NO h_fast.
            # Forces slow to carry predictive info; if slow collapsed, this loss
            # won't beat random baseline.
            coupling_pred = online.coupling_predict(h_slow_now, chunks_at_t, h_fast_now)
            coupling_losses.append(((coupling_pred - h_fast_future_target) ** 2).mean())

            # Success aux from fast h_goal
            succ_logit = online.jepa_predict_success(h_fast_now).squeeze()
            succ_logits.append(succ_logit)
            succ_labels.append(torch.tensor(float(r["succ"]), device=device))
            var_targets.append(h_fast_now[0].flatten().detach())

        if not pred_losses:
            return None, None, None, None
        pred_loss = torch.stack(pred_losses).mean()
        coupling_loss = torch.stack(coupling_losses).mean()
        var_loss = vicreg_var_loss(torch.stack(var_targets, dim=0),
                                      target_std=args.var_target)
        succ_loss = bce_succ(torch.stack(succ_logits), torch.stack(succ_labels))
        return pred_loss, var_loss, succ_loss, coupling_loss

    t_start = time.time()
    best_val = float("inf")
    best_state = None
    last_improvement = 0
    n_skipped_nan = 0
    for step in range(args.max_steps):
        batch_recs = [train_records[int(rng.integers(0, len(train_records)))]
                       for _ in range(args.batch_size)]
        pred_loss, var_loss, succ_loss, coupling_loss = loss_for_records(
            batch_recs, training=True)
        if pred_loss is None:
            continue
        total_loss = (pred_loss + args.lambda_var * var_loss
                       + args.lambda_succ * succ_loss
                       + args.lambda_coupling * coupling_loss)
        if not torch.isfinite(total_loss):
            n_skipped_nan += 1
            if n_skipped_nan >= 5:
                print(f"[twoflow] ABORT: 5 consecutive NaN at step {step}", flush=True)
                break
            continue
        n_skipped_nan = 0
        opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in online.parameters() if p.requires_grad],
            args.max_grad_norm)
        opt.step()
        ema_update(target, online, args.ema_tau)

        if step % args.log_every == 0:
            vp_list, vs_list, vc_list = [], [], []
            for r in val_records[:min(16, len(val_records))]:
                pl, _, sl, cl = loss_for_records([r], training=False)
                if pl is not None:
                    vp_list.append(float(pl.detach()))
                    vs_list.append(float(sl.detach()))
                    vc_list.append(float(cl.detach()))
            vp = float(np.mean(vp_list)) if vp_list else 0.0
            vs = float(np.mean(vs_list)) if vs_list else 0.0
            vc = float(np.mean(vc_list)) if vc_list else 0.0
            # Save by combined: balance pred, succ, coupling
            score = vp + args.lambda_succ * vs + args.lambda_coupling * vc
            if score < best_val:
                best_val = score
                best_state = {
                    "online": {k: v.clone().cpu() for k, v in online.state_dict().items()},
                    "target": {k: v.clone().cpu() for k, v in target.state_dict().items()},
                }
                last_improvement = step
                torch.save({
                    "substrate_state_dict": best_state["online"],
                    "target_substrate_state_dict": best_state["target"],
                    "args": vars(args), "step": step,
                    "z_vl_dim": ck["z_vl_dim"], "action_dim": ck["action_dim"],
                    "horizon": ck["horizon"], "state_dim": ck["state_dim"],
                    "dist_mean": ck["dist_mean"], "dist_std": ck["dist_std"],
                    "sd_mean": ck["sd_mean"], "sd_std": ck["sd_std"],
                    "best_val_score": best_val,
                    "jepa_window": args.window,
                }, args.output)
            stale = (step - last_improvement) // args.log_every
            print(f"step {step:>4}  L_pred={float(pred_loss.detach()):.3f}  "
                  f"L_coup={float(coupling_loss.detach()):.3f}  "
                  f"L_succ={float(succ_loss.detach()):.3f}  "
                  f"vL_pred={vp:.3f}  vL_coup={vc:.3f}  vL_succ={vs:.3f}  "
                  f"score={score:.3f} (best {best_val:.3f})  stale={stale}  "
                  f"wall={time.time()-t_start:.0f}s", flush=True)
            if stale >= args.early_stop_patience:
                print(f"[twoflow] early stop at step {step}", flush=True)
                break

    if best_state is not None:
        online.load_state_dict(best_state["online"])
        target.load_state_dict(best_state["target"])
        print(f"[twoflow] restored best ckpt (score={best_val:.3f})", flush=True)

    torch.save({
        "substrate_state_dict": online.state_dict(),
        "target_substrate_state_dict": target.state_dict(),
        "args": vars(args), "step": args.max_steps,
        "z_vl_dim": ck["z_vl_dim"], "action_dim": ck["action_dim"],
        "horizon": ck["horizon"], "state_dim": ck["state_dim"],
        "dist_mean": ck["dist_mean"], "dist_std": ck["dist_std"],
        "sd_mean": ck["sd_mean"], "sd_std": ck["sd_std"],
        "best_val_score": best_val,
        "jepa_window": args.window,
    }, args.output)
    print(f"\n[twoflow] saved → {args.output}  best_score={best_val:.3f}", flush=True)


if __name__ == "__main__":
    main()
