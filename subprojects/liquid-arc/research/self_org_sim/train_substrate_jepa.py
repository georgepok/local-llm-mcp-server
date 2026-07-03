"""JEPA training: substrate body trains end-to-end to predict its OWN future
h_goal from current h_goal + intended action chunk.

Per user spec: identify a HIGH-DIMENSIONAL state-transition derivative that
indicates the meta-condition of the state trajectory toward the task goal.
JEPA: substrate's belief state at t+W is a high-D target; substrate must learn
to make h_goal evolve in a structurally predictable way.

Loss = MSE(predictor(h_goal[t], action_chunk[t]), h_goal[t+W].detach())
       + lambda_var * VICReg variance regularizer (collapse-prevention)

Substrate body UNFROZEN — in_z, in_goal, in_delta, in_action, in_state, in_lang,
gates, evidence_mix, context_pool, dynamics, init_belief — all trainable.
Predictor head also trainable. Other heads (goaldist, gripper_moving, etc.)
frozen to avoid interference.

Requires JEPA-extended trajectory data with substrate INPUTS per chunk
(z_vl, z_lang, state8, chunk, z_goal). Collected via chained_libero_eval_dynamics.py
with --collect_trajectories flag (extended version).

Usage on Spark:
  python train_substrate_jepa.py \
    --traj_files /tmp/traj_jepa_libero10_s10.pt,... \
    --substrate_ckpt /tmp/substrate_dynamics_corr.pt \
    --output /tmp/substrate_jepa.pt
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

_LA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_LA_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from liquid_goal_tracker_proprio import JEPA_LGT_Proprio  # type: ignore

torch.set_float32_matmul_precision("high")


def vicreg_var_loss(z: torch.Tensor, target_std: float = 1.0) -> torch.Tensor:
    """Hinge loss penalizing per-dim std < target. Prevents h_goal collapse.
    z: [B, D] -> scalar loss. Returns 0 when B < 2 (need at least 2 samples for var).
    """
    if z.shape[0] < 2:
        return torch.tensor(0.0, device=z.device)
    var = z.var(dim=0, unbiased=False)  # use biased var to avoid div-by-zero
    std = torch.sqrt(var + 1e-6)
    return torch.mean(torch.clamp(target_std - std, min=0.0))


def load_records_with_inputs(traj_files):
    """Load only records that have the JEPA-extended fields."""
    all_records = []
    skipped = 0
    for fp in traj_files:
        ck = torch.load(fp, map_location="cpu", weights_only=False)
        for r in ck["records"]:
            if "z_vl_traj" in r and "chunk_traj" in r:
                all_records.append(r)
            else:
                skipped += 1
        print(f"  {fp}: loaded {len([r for r in ck['records'] if 'z_vl_traj' in r])} "
              f"with-inputs records", flush=True)
    if skipped:
        print(f"  skipped {skipped} records without JEPA-extended fields", flush=True)
    return all_records


def forward_through_substrate(substrate, record, device, grad_at_t=None, target_t=None):
    """Run substrate.step over chunk sequence. TRUNCATED BPTT:
    - For t < grad_at_t: no_grad (h passed in as detached)
    - For t == grad_at_t: with grad (substrate params can update via this step)
    - For t > grad_at_t until target_t: no_grad (compute target without grad)
    Returns full h_goal_traj [T, K, d] tensor.

    If grad_at_t is None, runs entirely with grad enabled (legacy path).
    """
    z_vl = record["z_vl_traj"].float().to(device)
    z_lang = record["z_lang_traj"].float().to(device)
    state8 = record["state8_traj"].float().to(device)
    chunks = record["chunk_traj"].float().to(device)
    z_goal = record["z_goal"].float().to(device)
    T = z_vl.shape[0]

    h = substrate.init_state(1, device)
    h_traj = []
    z_g_b = z_goal.unsqueeze(0)
    for t in range(T):
        z_t = z_vl[t].unsqueeze(0)
        z_l = z_lang[t].unsqueeze(0)
        s_t = state8[t].unsqueeze(0)
        ch_t = chunks[t].unsqueeze(0)
        if grad_at_t is None or t == grad_at_t:
            h, _, _, _ = substrate.step(h, z_t, z_g_b, ch_t, s_t, z_lang_t=z_l)
        else:
            with torch.no_grad():
                h, _, _, _ = substrate.step(h, z_t, z_g_b, ch_t, s_t, z_lang_t=z_l)
        # Detach between steps for truncated BPTT (no chain through h_{t-1})
        h_traj.append(h[0])
        if grad_at_t is not None and t == grad_at_t:
            # Save with grad
            pass
        # Detach h for next iteration's input — bounds gradient to one step
        if grad_at_t is not None:
            h = h.detach()
        if target_t is not None and t >= target_t:
            break
    return torch.stack(h_traj, dim=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traj_files", required=True)
    p.add_argument("--substrate_ckpt", required=True)
    p.add_argument("--output", default="/tmp/substrate_jepa.pt")
    p.add_argument("--window", type=int, default=5,
                   help="Predict h_goal[t+W] from h_goal[t] + action_chunks[t..t+W-1]")
    p.add_argument("--min_chunks", type=int, default=8,
                   help="Skip records shorter than this many chunks")
    p.add_argument("--max_steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=8,
                   help="Number of records per gradient step")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--split_mode", choices=["trajectory", "task_id"],
                    default="task_id")
    p.add_argument("--lambda_var", type=float, default=1.0,
                   help="VICReg variance regularizer weight (collapse-prevention)")
    p.add_argument("--var_target", type=float, default=1.0,
                   help="Target per-dim std for VICReg (h_goal stds clipped below this)")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--early_stop_patience", type=int, default=10)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--predictor_warmup_steps", type=int, default=200,
                   help="Initial steps with substrate FROZEN, train predictor only.")
    p.add_argument("--substrate_lr_ratio", type=float, default=0.1,
                   help="LR multiplier for substrate body relative to predictor.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[jepa] device={device}, output={args.output}, W={args.window}", flush=True)

    ck = torch.load(args.substrate_ckpt, map_location=device, weights_only=False)
    sa = ck["args"]
    d_sub = sa.get("d_substrate", 64)
    K_bel = sa.get("K_belief", 4)
    substrate = JEPA_LGT_Proprio(
        z_vl_dim=ck["z_vl_dim"], action_dim=ck["action_dim"],
        horizon=ck["horizon"], state_dim=ck["state_dim"],
        d=d_sub, K=K_bel, n_tok_per_k=sa.get("n_tok_per_k", 1),
    ).to(device)
    substrate.load_state_dict(ck["substrate_state_dict"], strict=False)
    print(f"[jepa] substrate loaded; K={K_bel} d={d_sub}", flush=True)

    # STAGE A start: substrate body FROZEN, only predictor head trains.
    # After predictor_warmup_steps, unfreeze substrate body for end-to-end JEPA.
    for pp in substrate.parameters():
        pp.requires_grad = False
    for pp in substrate.head_jepa_predictor.parameters():
        pp.requires_grad = True
    n_trainable = sum(pp.numel() for pp in substrate.parameters() if pp.requires_grad)
    print(f"[jepa] STAGE A (warmup, body frozen): predictor only "
          f"({n_trainable:,} params)", flush=True)

    aux_heads_to_freeze = (
        substrate.head_goaldist, substrate.head_gripper_moving,
        substrate.head_state_delta, substrate.head_virtual_tokens,
        substrate.head_residual, substrate.head_q_bb, substrate.head_k_sub,
        substrate.head_v_sub, substrate.dyn_in_state, substrate.dyn_in_action,
        substrate.dyn_in_intent, substrate.head_dynamics,
        substrate.head_correction, substrate.head_projection,
        substrate.head_zvl_residual,
    )

    def unfreeze_substrate_body():
        """STAGE B: SELECTIVE unfreeze — only input projections + scalar gates +
        init_belief + evidence_mix. Keep context_pool and ContinuousDynamics FROZEN
        because BPTT through them is numerically fragile and dominates instability.
        """
        body_modules = (substrate.in_z, substrate.in_goal, substrate.in_delta,
                          substrate.in_action, substrate.in_state, substrate.in_lang)
        for mod in body_modules:
            for pp in mod.parameters():
                pp.requires_grad = True
        for name in ("action_gate", "goal_gate", "delta_gate", "state_gate",
                      "lang_gate", "init_belief", "evidence_mix"):
            getattr(substrate, name).requires_grad = True
        for mod in aux_heads_to_freeze:
            for pp in mod.parameters():
                pp.requires_grad = False
        # Explicitly KEEP frozen: context_pool, dynamics (high-fragility modules)
        for mod in (substrate.context_pool, substrate.dynamics):
            for pp in mod.parameters():
                pp.requires_grad = False
        n = sum(pp.numel() for pp in substrate.parameters() if pp.requires_grad)
        print(f"[jepa] STAGE B SELECTIVE unfreeze: in_* + gates + init_belief + "
              f"evidence_mix + predictor ({n:,} params; context_pool + dynamics frozen)",
              flush=True)

    records = load_records_with_inputs([s.strip() for s in args.traj_files.split(",")])
    n_total = len(records)
    records = [r for r in records if r["h_goal_traj"].shape[0] >= args.min_chunks
                + args.window]
    print(f"[jepa] {len(records)}/{n_total} records pass min_chunks filter", flush=True)
    if not records:
        print("[jepa] no usable records"); return

    rng = np.random.default_rng(42)
    if args.split_mode == "task_id":
        all_sub_ids = sorted({int(r["sub_id"]) for r in records})
        rng.shuffle(all_sub_ids)
        n_val_ids = max(1, int(args.val_frac * len(all_sub_ids)))
        val_ids = set(all_sub_ids[:n_val_ids])
        train_ids = set(all_sub_ids[n_val_ids:])
        train_records = [r for r in records if int(r["sub_id"]) in train_ids]
        val_records = [r for r in records if int(r["sub_id"]) in val_ids]
        print(f"[jepa] task_id split: train ids {sorted(train_ids)} "
              f"({len(train_records)} traj) / val ids {sorted(val_ids)} "
              f"({len(val_records)} traj)", flush=True)
    else:
        idx = rng.permutation(len(records))
        n_val = max(1, int(args.val_frac * len(records)))
        val_records = [records[i] for i in idx[:n_val]]
        train_records = [records[i] for i in idx[n_val:]]
        print(f"[jepa] traj split: {len(train_records)} train / "
              f"{len(val_records)} val", flush=True)

    # Two param groups: predictor at full LR, substrate body (when unfrozen) at lower LR.
    # weight_decay=0 to avoid drift on frozen params during stage A.
    def build_optimizer():
        pred_params = [p for p in substrate.head_jepa_predictor.parameters()
                        if p.requires_grad]
        body_params = [p for n, p in substrate.named_parameters()
                        if not n.startswith("head_jepa_predictor") and p.requires_grad]
        groups = [{"params": pred_params, "lr": args.lr}]
        if body_params:
            groups.append({"params": body_params,
                             "lr": args.lr * args.substrate_lr_ratio})
        return torch.optim.AdamW(groups, weight_decay=0.0)

    opt = build_optimizer()

    def jepa_loss_for_records(recs, training: bool):
        """Sample one (t, t+W) per record. Truncated BPTT: gradient only through
        substrate.step at time t (target h[t+W] is no_grad → detached automatically).
        """
        pred_losses = []
        var_targets = []
        for r in recs:
            T = r["h_goal_traj"].shape[0]
            if T < args.window + 2:
                continue
            # Sample one t per record
            t = int(rng.integers(0, T - args.window))
            if training:
                h_traj = forward_through_substrate(substrate, r, device,
                                                      grad_at_t=t, target_t=t+args.window)
            else:
                with torch.no_grad():
                    h_traj = forward_through_substrate(substrate, r, device,
                                                          target_t=t+args.window)
            chunks_at_t = r["chunk_traj"][t].float().to(device).flatten().unsqueeze(0)
            h_now = h_traj[t].unsqueeze(0)              # [1, K, d] — has grad at training
            h_future_target = h_traj[t + args.window]   # [K, d] — no grad
            pred = substrate.jepa_predict_future_h_goal(h_now, chunks_at_t)
            target = h_future_target.detach().unsqueeze(0)
            pred_losses.append(((pred - target) ** 2).mean())
            var_targets.append(h_now[0].flatten().detach())  # detach for var reg, free graph
        if not pred_losses:
            return None, None
        pred_loss = torch.stack(pred_losses).mean()
        var_stack = torch.stack(var_targets, dim=0)
        var_loss = vicreg_var_loss(var_stack, target_std=args.var_target)
        return pred_loss, var_loss

    t_start = time.time()
    best_val_loss = float("inf")
    best_state = None
    last_improvement = 0
    stage = "A"
    n_skipped_nan = 0
    for step in range(args.max_steps):
        if step == args.predictor_warmup_steps and stage == "A":
            unfreeze_substrate_body()
            opt = build_optimizer()  # rebuild with the newly-unfrozen body params
            stage = "B"
        batch_recs = [train_records[int(rng.integers(0, len(train_records)))]
                       for _ in range(args.batch_size)]
        pred_loss, var_loss = jepa_loss_for_records(batch_recs, training=True)
        if pred_loss is None:
            continue
        total_loss = pred_loss + args.lambda_var * var_loss
        if not torch.isfinite(total_loss):
            n_skipped_nan += 1
            if n_skipped_nan >= 10:
                print(f"[jepa] ABORT: 10 consecutive NaN losses at step {step}",
                      flush=True)
                break
            continue
        n_skipped_nan = 0
        opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in substrate.parameters() if p.requires_grad],
            args.max_grad_norm)
        opt.step()

        if step % args.log_every == 0:
            # Val
            val_pred_losses, val_var_losses = [], []
            for r in val_records[:min(16, len(val_records))]:
                pl, vl = jepa_loss_for_records([r], training=False)
                if pl is not None:
                    val_pred_losses.append(float(pl))
                    val_var_losses.append(float(vl))
            vp = float(np.mean(val_pred_losses)) if val_pred_losses else 0.0
            vv = float(np.mean(val_var_losses)) if val_var_losses else 0.0
            if vp < best_val_loss:
                best_val_loss = vp
                best_state = {k: v.clone().cpu()
                                for k, v in substrate.state_dict().items()}
                last_improvement = step
            stale = (step - last_improvement) // args.log_every
            print(f"step {step:>4}  L_pred={float(pred_loss):.4f}  "
                  f"L_var={float(var_loss):.4f}  vL_pred={vp:.4f} "
                  f"(best {best_val_loss:.4f}) vL_var={vv:.4f}  "
                  f"stale={stale}  wall={time.time()-t_start:.0f}s", flush=True)
            if stale >= args.early_stop_patience:
                print(f"[jepa] early stop at step {step} "
                      f"(best vL_pred={best_val_loss:.4f} @ {last_improvement})",
                      flush=True)
                break

    if best_state is not None:
        substrate.load_state_dict(best_state)
        print(f"[jepa] restored best ckpt (vL_pred={best_val_loss:.4f})", flush=True)

    torch.save({
        "substrate_state_dict": substrate.state_dict(),
        "args": vars(args), "step": args.max_steps,
        "z_vl_dim": ck["z_vl_dim"], "action_dim": ck["action_dim"],
        "horizon": ck["horizon"], "state_dim": ck["state_dim"],
        "dist_mean": ck["dist_mean"], "dist_std": ck["dist_std"],
        "sd_mean": ck["sd_mean"], "sd_std": ck["sd_std"],
        "best_val_loss": best_val_loss,
        "jepa_window": args.window,
    }, args.output)
    print(f"\n[jepa] saved → {args.output}  best_vL_pred={best_val_loss:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
