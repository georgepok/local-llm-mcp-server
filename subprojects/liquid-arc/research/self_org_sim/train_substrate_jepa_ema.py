"""JEPA training with EMA target encoder — the standard stability fix.

Two substrate instances:
- online_substrate (trainable): gradient updates from JEPA loss
- target_substrate (no_grad, EMA-tracking online): provides STABLE target h_goal

Each step:
  1. Forward online substrate to chunk t (truncated BPTT: prev_h detached, only
     substrate body at t is differentiable)
  2. Forward target substrate to chunk t+W (entirely no_grad on target weights)
  3. predictor(h_online[t], chunks_at_t) → predicted h_goal[t+W]
  4. Loss = MSE(pred, h_target[t+W].detach())
  5. opt.step() updates ONLINE substrate + predictor
  6. EMA update: target.params ← tau * target.params + (1-tau) * online.params

EMA tau≈0.99 means target weights drift very slowly even when online updates fast.
This breaks the feedback loop where bad-weight → bad-target → bad-gradient → catastrophic
update that destabilized the BPTT variants.

Usage on Spark:
  python train_substrate_jepa_ema.py \
    --traj_files /tmp/traj_jepa_libero10_s10.pt,... \
    --substrate_ckpt /tmp/substrate_dynamics_corr.pt \
    --output /tmp/substrate_jepa_ema.pt
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
            if "z_vl_traj" in r and "chunk_traj" in r:
                all_records.append(r)
        print(f"  {fp}: loaded {len([r for r in ck['records'] if 'z_vl_traj' in r])} "
              f"with-inputs records", flush=True)
    return all_records


def forward_online_to_t(substrate, record, device, target_t):
    """Forward online substrate to time target_t. h passed in detached between
    steps so grad only flows through substrate body params AT time target_t.
    Returns h_traj [target_t+1, K, d].
    """
    z_vl = record["z_vl_traj"].float().to(device)
    z_lang = record["z_lang_traj"].float().to(device)
    state8 = record["state8_traj"].float().to(device)
    chunks = record["chunk_traj"].float().to(device)
    z_goal = record["z_goal"].float().to(device).unsqueeze(0)
    T = z_vl.shape[0]
    h = substrate.init_state(1, device)
    h_traj = []
    end_t = min(target_t + 1, T)
    for t in range(end_t):
        z_t = z_vl[t].unsqueeze(0)
        z_l = z_lang[t].unsqueeze(0)
        s_t = state8[t].unsqueeze(0)
        ch_t = chunks[t].unsqueeze(0)
        if t == target_t:
            h, _, _, _ = substrate.step(h, z_t, z_goal, ch_t, s_t, z_lang_t=z_l)
        else:
            with torch.no_grad():
                h, _, _, _ = substrate.step(h, z_t, z_goal, ch_t, s_t, z_lang_t=z_l)
            h = h.detach()  # break gradient chain
        h_traj.append(h[0])
    return torch.stack(h_traj, dim=0)


@torch.no_grad()
def forward_target_to_t(target_substrate, record, device, target_t):
    """Forward target substrate (no_grad) to time target_t. Returns h_traj."""
    z_vl = record["z_vl_traj"].float().to(device)
    z_lang = record["z_lang_traj"].float().to(device)
    state8 = record["state8_traj"].float().to(device)
    chunks = record["chunk_traj"].float().to(device)
    z_goal = record["z_goal"].float().to(device).unsqueeze(0)
    T = z_vl.shape[0]
    h = target_substrate.init_state(1, device)
    h_traj = []
    end_t = min(target_t + 1, T)
    for t in range(end_t):
        z_t = z_vl[t].unsqueeze(0)
        z_l = z_lang[t].unsqueeze(0)
        s_t = state8[t].unsqueeze(0)
        ch_t = chunks[t].unsqueeze(0)
        h, _, _, _ = target_substrate.step(h, z_t, z_goal, ch_t, s_t, z_lang_t=z_l)
        h_traj.append(h[0])
    return torch.stack(h_traj, dim=0)


def ema_update(target_substrate, online_substrate, tau: float):
    """target.params ← tau * target.params + (1-tau) * online.params"""
    with torch.no_grad():
        for tp, op in zip(target_substrate.parameters(),
                            online_substrate.parameters()):
            tp.data.mul_(tau).add_(op.data, alpha=1.0 - tau)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traj_files", required=True)
    p.add_argument("--substrate_ckpt", required=True)
    p.add_argument("--output", default="/tmp/substrate_jepa_ema.pt")
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--min_chunks", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--substrate_lr_ratio", type=float, default=0.05,
                   help="LR ratio for substrate body relative to predictor")
    p.add_argument("--ema_tau", type=float, default=0.996,
                   help="EMA decay for target encoder")
    p.add_argument("--lambda_var", type=float, default=0.1)
    p.add_argument("--var_target", type=float, default=1.0)
    p.add_argument("--lambda_succ", type=float, default=1.0,
                   help="Weight on outcome-aware success prediction BCE loss")
    p.add_argument("--pos_weight_succ", type=float, default=2.0,
                   help="BCE positive weight for success head (failure ~25% base rate)")
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--split_mode", choices=["trajectory", "task_id"],
                    default="task_id")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--early_stop_patience", type=int, default=15)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[jepa-ema] device={device}, output={args.output}, W={args.window}, "
          f"tau={args.ema_tau}", flush=True)

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
        return s

    online = build_substrate()
    # Architectural safety: enable evidence LayerNorm + soft-clamp h_input.
    # Without these, body training NaN's after ~160 steps (verified empirically).
    online.use_evidence_layernorm = True
    online.h_input_clamp = 50.0
    target = copy.deepcopy(online)
    for pp in target.parameters():
        pp.requires_grad = False
    print(f"[jepa-ema] online + target substrate loaded; K={K_bel} d={d_sub}",
          flush=True)
    print(f"[jepa-ema] safety: use_evidence_layernorm={online.use_evidence_layernorm} "
          f"h_input_clamp={online.h_input_clamp}", flush=True)

    # Unfreeze ONLY safe body modules (input projections + scalar gates + predictor
    # + success predictor + evidence_layernorm). Dynamics + context_pool frozen.
    for pp in online.parameters():
        pp.requires_grad = False
    safe_body_modules = (
        online.in_z, online.in_goal, online.in_delta,
        online.in_action, online.in_state, online.in_lang,
        online.head_jepa_predictor, online.head_success_predictor,
        online.evidence_layernorm,
    )
    for mod in safe_body_modules:
        for pp in mod.parameters():
            pp.requires_grad = True
    for name in ("action_gate", "goal_gate", "delta_gate", "state_gate",
                  "lang_gate", "init_belief", "evidence_mix"):
        getattr(online, name).requires_grad = True
    n_trainable = sum(pp.numel() for pp in online.parameters() if pp.requires_grad)
    print(f"[jepa-ema] online trainable: in_* + gates + init_belief + evidence_mix "
          f"+ predictor + success_head ({n_trainable:,} params); dynamics + context_pool FROZEN",
          flush=True)

    records = load_records_with_inputs([s.strip() for s in args.traj_files.split(",")])
    records = [r for r in records
                if r["h_goal_traj"].shape[0] >= args.min_chunks + args.window]
    print(f"[jepa-ema] {len(records)} records after min_chunks filter", flush=True)
    if not records:
        return

    rng = np.random.default_rng(42)
    if args.split_mode == "task_id":
        all_sub_ids = sorted({int(r["sub_id"]) for r in records})
        rng.shuffle(all_sub_ids)
        n_val_ids = max(1, int(args.val_frac * len(all_sub_ids)))
        val_ids = set(all_sub_ids[:n_val_ids])
        train_ids = set(all_sub_ids[n_val_ids:])
        train_records = [r for r in records if int(r["sub_id"]) in train_ids]
        val_records = [r for r in records if int(r["sub_id"]) in val_ids]
        print(f"[jepa-ema] task_id split: train={sorted(train_ids)} "
              f"({len(train_records)}) / val={sorted(val_ids)} ({len(val_records)})",
              flush=True)
    else:
        idx = rng.permutation(len(records))
        n_val = max(1, int(args.val_frac * len(records)))
        val_records = [records[i] for i in idx[:n_val]]
        train_records = [records[i] for i in idx[n_val:]]

    pred_params = list(online.head_jepa_predictor.parameters())
    body_params = [p for n, p in online.named_parameters()
                    if not n.startswith("head_jepa_predictor") and p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": pred_params, "lr": args.lr},
        {"params": body_params, "lr": args.lr * args.substrate_lr_ratio},
    ], weight_decay=0.0)

    pos_w_succ = torch.tensor([args.pos_weight_succ], device=device)
    bce_succ = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w_succ)

    def jepa_loss_for_records(recs, training=True):
        pred_losses = []
        var_targets = []
        succ_logits = []
        succ_labels = []
        for r in recs:
            T = r["h_goal_traj"].shape[0]
            if T < args.window + 2:
                continue
            t = int(rng.integers(0, T - args.window))
            # Online: differentiate substrate body at t (truncated)
            if training:
                h_online_traj = forward_online_to_t(online, r, device, t)
            else:
                with torch.no_grad():
                    h_online_traj = forward_online_to_t(online, r, device, t)
            h_now = h_online_traj[t].unsqueeze(0)
            # Target: full no_grad forward via target substrate (stable EMA weights)
            h_target_traj = forward_target_to_t(target, r, device, t + args.window)
            h_future_target = h_target_traj[t + args.window].detach().unsqueeze(0)
            chunks_at_t = r["chunk_traj"][t].float().to(device).flatten().unsqueeze(0)
            pred = online.jepa_predict_future_h_goal(h_now, chunks_at_t)
            pred_losses.append(((pred - h_future_target) ** 2).mean())
            # Outcome-aware: success head from current h_goal
            succ_logit = online.jepa_predict_success(h_now)  # [1]
            succ_logits.append(succ_logit.squeeze())
            succ_labels.append(torch.tensor(float(r["succ"]), device=device))
            var_targets.append(h_now[0].flatten().detach())
        if not pred_losses:
            return None, None, None
        pred_loss = torch.stack(pred_losses).mean()
        var_stack = torch.stack(var_targets, dim=0)
        var_loss = vicreg_var_loss(var_stack, target_std=args.var_target)
        succ_logit_t = torch.stack(succ_logits)
        succ_label_t = torch.stack(succ_labels)
        succ_loss = bce_succ(succ_logit_t, succ_label_t)
        return pred_loss, var_loss, succ_loss

    t_start = time.time()
    best_val = float("inf")
    best_state = None
    last_improvement = 0
    n_skipped_nan = 0
    for step in range(args.max_steps):
        batch_recs = [train_records[int(rng.integers(0, len(train_records)))]
                       for _ in range(args.batch_size)]
        pred_loss, var_loss, succ_loss = jepa_loss_for_records(batch_recs, training=True)
        if pred_loss is None:
            continue
        total_loss = (pred_loss + args.lambda_var * var_loss
                       + args.lambda_succ * succ_loss)
        if not torch.isfinite(total_loss):
            n_skipped_nan += 1
            if n_skipped_nan == 1:
                # FIRST NaN: enable diagnostic on substrate.step + re-run forward
                print(f"[jepa-ema] FIRST NaN at step {step} — diagnosing...",
                      flush=True)
                online._STEP_DEBUG = True
                target._STEP_DEBUG = True
                # Re-run on same batch to capture WHERE NaN appears
                _ = jepa_loss_for_records(batch_recs, training=False)
                online._STEP_DEBUG = False
                target._STEP_DEBUG = False
                # Also report current substrate param max magnitudes
                print(f"  [substrate param magnitudes after step {step-1}]", flush=True)
                for name, p in online.named_parameters():
                    if p.requires_grad and not name.startswith("head_jepa"):
                        absmax = float(p.detach().abs().max())
                        if absmax > 5.0 or not torch.isfinite(p).all():
                            print(f"    {name}: shape={tuple(p.shape)} abs_max={absmax:.2f}",
                                  flush=True)
            if n_skipped_nan >= 5:
                print(f"[jepa-ema] ABORT: 5 consecutive NaN at step {step}",
                      flush=True)
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
            val_pred_losses = []
            val_var_losses = []
            val_succ_losses = []
            for r in val_records[:min(16, len(val_records))]:
                pl, vl, sl = jepa_loss_for_records([r], training=False)
                if pl is not None:
                    val_pred_losses.append(float(pl.detach()))
                    val_var_losses.append(float(vl.detach()))
                    val_succ_losses.append(float(sl.detach()))
            vp = float(np.mean(val_pred_losses)) if val_pred_losses else 0.0
            vv = float(np.mean(val_var_losses)) if val_var_losses else 0.0
            vs = float(np.mean(val_succ_losses)) if val_succ_losses else 0.0
            score = vs  # save by best vL_succ (outcome-aware mode)
            if score < best_val:
                best_val = score
                best_state = {
                    "online": {k: v.clone().cpu() for k, v in online.state_dict().items()},
                    "target": {k: v.clone().cpu() for k, v in target.state_dict().items()},
                }
                last_improvement = step
            stale = (step - last_improvement) // args.log_every
            print(f"step {step:>4}  L_pred={float(pred_loss.detach()):.3f}  "
                  f"L_var={float(var_loss.detach()):.3f}  "
                  f"L_succ={float(succ_loss.detach()):.3f}  "
                  f"vL_pred={vp:.3f}  vL_succ={vs:.3f}  "
                  f"score={score:.3f} (best {best_val:.3f})  stale={stale}  "
                  f"wall={time.time()-t_start:.0f}s", flush=True)
            if stale >= args.early_stop_patience:
                print(f"[jepa-ema] early stop at step {step} "
                      f"(best vL_pred={best_val:.3f} @ {last_improvement})",
                      flush=True)
                break

    if best_state is not None:
        online.load_state_dict(best_state["online"])
        target.load_state_dict(best_state["target"])
        print(f"[jepa-ema] restored best ckpt (vL_pred={best_val:.3f})", flush=True)

    torch.save({
        "substrate_state_dict": online.state_dict(),
        "target_substrate_state_dict": target.state_dict(),
        "args": vars(args), "step": args.max_steps,
        "z_vl_dim": ck["z_vl_dim"], "action_dim": ck["action_dim"],
        "horizon": ck["horizon"], "state_dim": ck["state_dim"],
        "dist_mean": ck["dist_mean"], "dist_std": ck["dist_std"],
        "sd_mean": ck["sd_mean"], "sd_std": ck["sd_std"],
        "best_val_loss": best_val,
        "jepa_window": args.window,
    }, args.output)
    print(f"\n[jepa-ema] saved → {args.output}  best_vL={best_val:.3f}", flush=True)


if __name__ == "__main__":
    main()
