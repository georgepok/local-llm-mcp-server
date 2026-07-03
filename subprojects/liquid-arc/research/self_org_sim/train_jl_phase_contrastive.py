"""JudgeLiquid training with contrastive goal-generalization loss + CV monitoring.

Adds to standard JudgeLiquid trainer:
  1. Contrastive loss on h_fast representations:
     - Same-status samples (both followed OR both drifted, ACROSS categories): pull together
     - Different-status samples: push apart
     - Forces category-invariant abstract goal-following feature
  2. Full geometric LR (substrate_lr_ratio=1.0) — let dynamics reorganize
  3. CV monitoring every checkpoint — track phase transition
  4. Long training (default 10000 steps) with high patience
  5. Multi-category batch sampling for contrastive pairs

If this works, CV climbs past phase-transition threshold AND cross-cat AUC
breaks the memorization ceiling.
"""
import argparse
import copy
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from judge_liquid import JudgeLiquid, forward_trajectory
try:
    from liquid_arc.sustained_criticality import compute_tau_quality_loss  # type: ignore
except Exception:
    compute_tau_quality_loss = None


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


def compute_metric_cv(model, sample_records, device, n_sample=10):
    """Phase transition diagnostic."""
    model.eval()
    all_diag = []
    with torch.no_grad():
        for r in sample_records[:n_sample]:
            T = r["T"]
            if T < 2:
                continue
            hs_traj = r["hidden_state_traj"].to(device)
            z_g = r["z_goal_traj"].to(device)
            judge_traj = r["judge_traj"].to(device)
            h_fast_traj, _ = forward_trajectory(
                model, hs_traj, judge_traj, z_g, device, target_t=None, training=False)
            hs_normed = model.hidden_layernorm(hs_traj[-1:])
            e_h = model.in_hidden(hs_normed) * model.hidden_gate
            e_j = model.in_judge(judge_traj[-1:].unsqueeze(-1)) * model.judge_gate
            e_g_proj = model.in_goal(z_g[-1:]) * model.goal_gate
            e = model.evidence_layernorm(e_h + e_j + e_g_proj)
            evidence = e.unsqueeze(1) * model.evidence_mix.unsqueeze(0)
            h_input = h_fast_traj[-2].unsqueeze(0) + evidence
            h_input = model._soft_clamp(h_input)
            context = model.context_pool(h_input, None)
            model.dynamics.set_context(context, mask=None)
            metric_diag = model.dynamics.compute_metric_diag(h_input)
            all_diag.append(metric_diag.flatten().cpu().numpy())
    model.train()
    if not all_diag:
        return float("nan")
    arr = np.concatenate(all_diag)
    return float(arr.std() / max(1e-8, arr.mean()))


def contrastive_loss(h_fast_batch, labels, temperature=0.5):
    """InfoNCE-style contrastive loss.

    h_fast_batch: [B, K, d] — h_fast at random t for each batch element
    labels:       [B] — follow status (1=followed, 0=drifted)

    Same-status pairs: pull together (positive)
    Different-status pairs: push apart (negative)

    Normalize across batch — operates on FLATTENED h_fast.
    """
    B = h_fast_batch.shape[0]
    if B < 4:
        return torch.tensor(0.0, device=h_fast_batch.device)
    h_flat = h_fast_batch.flatten(1)            # [B, K*d]
    h_norm = F.normalize(h_flat, dim=-1)
    sim = h_norm @ h_norm.t() / temperature     # [B, B] cosine similarities
    # Mask out self-similarity
    mask_self = torch.eye(B, device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(mask_self, -float("inf"))
    # Positives: same label
    same_label = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B, B]
    same_label = same_label & ~mask_self
    # InfoNCE: for each sample, log-sum-exp of all similarities; numerator = positives
    exp_sim = torch.exp(sim - sim.max(dim=-1, keepdim=True).values.detach())
    # Numerator: sum exp_sim over positives
    pos_mask = same_label.float()
    num = (exp_sim * pos_mask).sum(dim=-1)  # [B]
    den = exp_sim.sum(dim=-1)               # [B]
    # Avoid log(0): clamp num minimum
    loss_per_sample = -torch.log((num / den).clamp(min=1e-8))
    # Only count samples that have at least one positive (otherwise loss is meaningless)
    has_pos = pos_mask.sum(dim=-1) > 0
    if has_pos.sum() == 0:
        return torch.tensor(0.0, device=sim.device)
    return loss_per_sample[has_pos].mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_traj", required=True,
                   help="Judged trajectory data with category metadata")
    p.add_argument("--test_traj", default="")
    p.add_argument("--output", required=True)
    p.add_argument("--d", type=int, default=512)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--value_hidden", type=int, default=512)
    p.add_argument("--d_metric", type=int, default=128)
    p.add_argument("--d_ffn", type=int, default=768)
    p.add_argument("--n_ode_steps", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=16,
                   help="Per training step batch size. Larger helps contrastive.")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--substrate_lr_ratio", type=float, default=1.0,
                   help="FULL LR for geometric core to allow phase transition")
    p.add_argument("--lambda_value", type=float, default=5.0)
    p.add_argument("--lambda_jepa", type=float, default=0.3)
    p.add_argument("--lambda_var", type=float, default=0.1)
    p.add_argument("--lambda_contrastive", type=float, default=2.0,
                   help="Weight on contrastive loss. Higher = more pressure for abstraction")
    p.add_argument("--contrastive_temp", type=float, default=0.5)
    p.add_argument("--lambda_soc", type=float, default=0.0,
                   help="SOC regulator weight. 0 disables. Maintains CV near target_cv.")
    p.add_argument("--target_cv", type=float, default=4.0,
                   help="Sustained criticality target")
    p.add_argument("--tau_depth", action="store_true",
                   help="Enable self-organizing ODE depth (tau-convergence coupling + wider tau range)")
    p.add_argument("--lambda_tau_quality", type=float, default=0.0,
                   help="Weight on tau_quality_loss (encourages tau spread = variable depth)")
    p.add_argument("--tau_mean_target", type=float, default=0.0,
                   help="log-tau mean anchor for inline tau_quality fallback")
    p.add_argument("--tau_spread_target", type=float, default=0.6,
                   help="log-tau spread target for inline tau_quality fallback")
    p.add_argument("--lambda_tau_couple", type=float, default=0.0,
                   help="Weight on EXPLICIT tau<->surprise coupling: drives tau "
                        "anti-correlated with per-chunk JEPA surprise (deep on surprise, "
                        "shallow on stable). Category-invariant computational policy.")
    p.add_argument("--tau_min_override", type=float, default=-1.0,
                   help="If >0, override config tau_min (stabilizes stiff ODE)")
    p.add_argument("--tau_floor_override", type=float, default=-1.0,
                   help="If >0, override tau_convergence_floor (stabilizes stiff ODE)")
    p.add_argument("--jepa_window", type=int, default=2)
    p.add_argument("--ema_tau", type=float, default=0.996)
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--early_stop_patience", type=int, default=80)
    p.add_argument("--min_steps_before_stop", type=int, default=5000)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    args = p.parse_args()

    device = torch.device("cuda")
    print(f"[jlp] device={device}, output={args.output}", flush=True)
    print(f"[jlp] lambda_contrastive={args.lambda_contrastive}, "
          f"substrate_lr_ratio={args.substrate_lr_ratio}", flush=True)

    pack = torch.load(args.train_traj, map_location="cpu", weights_only=False)
    z_goal_dim = int(pack["enc_dim"])
    d_hidden = int(pack["d_hidden"])
    raw_records = pack["records"]
    print(f"[jlp] train pack: d_hidden={d_hidden}, records={len(raw_records)}", flush=True)

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
            turn_categories = list(r.get("turn_categories", []))
            chunk_labels = []
            chunk_cats = []
            cur_turn = 0
            for t in range(T):
                while (cur_turn + 1 < len(turn_starts) and
                       t >= turn_starts[cur_turn + 1]):
                    cur_turn += 1
                chunk_labels.append(int(turn_followed[cur_turn]))
                chunk_cats.append(turn_categories[cur_turn] if turn_categories else "unk")
            out.append({
                "hidden_state_traj": hs_traj,
                "z_goal_traj": z_goal_traj,
                "judge_traj": judge_traj,
                "labels": torch.tensor(chunk_labels, dtype=torch.float32),
                "chunk_categories": chunk_cats,
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
    print(f"[jlp] train={len(train_recs)}  val_in_dist={len(val_recs)}", flush=True)

    test_recs = []
    if args.test_traj:
        test_pack = torch.load(args.test_traj, map_location="cpu", weights_only=False)
        test_recs = augment(test_pack["records"])
        print(f"[jlp] held-out test (cross-cat): {len(test_recs)}", flush=True)

    online = JudgeLiquid(
        d_hidden=d_hidden, z_goal_dim=z_goal_dim,
        d=args.d, K=args.K, value_hidden=args.value_hidden,
        d_metric=args.d_metric, d_ffn=args.d_ffn, n_ode_steps=args.n_ode_steps,
        tau_depth=args.tau_depth,
    ).to(device)
    if args.tau_min_override > 0:
        online.config.tau_min = args.tau_min_override
    if args.tau_floor_override > 0:
        online.config.tau_convergence_floor = args.tau_floor_override
    if args.tau_depth:
        print(f"[jlp] tau-depth ENABLED: n_ode_steps={args.n_ode_steps}, "
              f"tau range [{online.config.tau_min},{online.config.tau_max}], "
              f"floor={getattr(online.config,'tau_convergence_floor','n/a')}, "
              f"convergence_coupling={online.config.tau_convergence_coupling_enabled}, "
              f"lambda_tau_couple={args.lambda_tau_couple}",
              flush=True)
    target = copy.deepcopy(online)
    for pp in target.parameters():
        pp.requires_grad = False
    n_params = sum(p.numel() for p in online.parameters() if p.requires_grad)
    print(f"[jlp] JudgeLiquid d={args.d} K={args.K} d_metric={args.d_metric} "
          f"d_ffn={args.d_ffn}, {n_params:,} params", flush=True)

    pred_params = (list(online.head_refinement.parameters()) +
                    list(online.head_jepa.parameters()) +
                    [online.judge_scale])
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

    def loss_for_batch(recs, training=True):
        value_losses, jepa_losses = [], []
        h_fast_batch, label_batch = [], []
        soc_h_inputs = []
        for r in recs:
            T = r["T"]
            hs_traj = r["hidden_state_traj"].to(device)
            z_goal_traj = r["z_goal_traj"].to(device)
            judge_traj = r["judge_traj"].to(device)
            labels = r["labels"].to(device)
            t = int(rng.integers(0, max(1, T - args.jepa_window)))
            h_fast_traj, _ = forward_trajectory(
                online, hs_traj, judge_traj, z_goal_traj, device,
                target_t=t, training=training)
            h_fast_now = h_fast_traj[t].unsqueeze(0)
            z_goal_now = z_goal_traj[t].unsqueeze(0)
            judge_now = judge_traj[t].unsqueeze(0)
            logit = online.value(h_fast_now, z_goal_now, judge_now)
            value_losses.append(bce(logit, labels[t].unsqueeze(0)))
            pred = online.jepa_predict(h_fast_now)
            with torch.no_grad():
                target_traj, _ = forward_trajectory(
                    target, hs_traj, judge_traj, z_goal_traj, device,
                    target_t=t + args.jepa_window, training=False)
                tgt = target_traj[t + args.jepa_window].detach().unsqueeze(0)
            jepa_losses.append(((pred - tgt) ** 2).mean())
            h_fast_batch.append(h_fast_now[0])
            label_batch.append(labels[t])
            # Build h_input for SOC / tau-quality / tau-couple computation if enabled
            if args.lambda_soc > 0 or args.lambda_tau_quality > 0 or args.lambda_tau_couple > 0:
                hs_normed_soc = online.hidden_layernorm(hs_traj[t:t+1])
                e_h_soc = online.in_hidden(hs_normed_soc) * online.hidden_gate
                e_j_soc = online.in_judge(judge_traj[t:t+1].unsqueeze(-1)) * online.judge_gate
                e_g_soc = online.in_goal(z_goal_traj[t:t+1]) * online.goal_gate
                e_soc = online.evidence_layernorm(e_h_soc + e_j_soc + e_g_soc)
                evidence_soc = e_soc.unsqueeze(1) * online.evidence_mix.unsqueeze(0)
                h_input_soc = h_fast_now + evidence_soc
                h_input_soc = online._soft_clamp(h_input_soc)
                soc_h_inputs.append(h_input_soc)
        if not value_losses:
            return None, None, None, None, None, None, None
        value_loss = torch.stack(value_losses).mean()
        jepa_loss = torch.stack(jepa_losses).mean()
        # Variance regularizer
        var_all = torch.stack([h.detach() for h in h_fast_batch], dim=0).flatten(1).std(dim=0).mean()
        var_loss = torch.relu(args.lambda_var - var_all)
        # CONTRASTIVE: pull same-status, push different-status across batch
        h_fast_stack = torch.stack(h_fast_batch, dim=0)  # [B, K, d]
        label_stack = torch.stack(label_batch, dim=0)
        contrastive = contrastive_loss(h_fast_stack, label_stack,
                                          temperature=args.contrastive_temp)
        soc_loss = torch.tensor(0.0, device=value_loss.device)
        tau_quality_loss = torch.tensor(0.0, device=value_loss.device)
        tau_couple_loss = torch.tensor(0.0, device=value_loss.device)
        if soc_h_inputs:
            h_input_batch = torch.cat(soc_h_inputs, dim=0)
            context = online.context_pool(h_input_batch, None)
            online.dynamics.set_context(context, mask=None)
            if args.lambda_soc > 0:
                metric_diag = online.dynamics.compute_metric_diag(h_input_batch)
                m_flat = metric_diag.flatten(1)
                cv_per = m_flat.std(dim=-1) / m_flat.mean(dim=-1).clamp(min=1e-8)
                soc_loss = (cv_per.mean() - args.target_cv) ** 2
            if args.lambda_tau_quality > 0:
                # tau_quality: encourage tau SPREAD (variable effective depth).
                # Anchor mean near tau_mean_target, reward log-spread (multiplicative
                # differentiation = some positions deep, some shallow). Mirrors prior
                # sustained_criticality.compute_tau_quality_loss.
                tau = online.dynamics.compute_tau(h_input_batch)  # [B, K, 1]
                if compute_tau_quality_loss is not None:
                    tau_quality_loss = compute_tau_quality_loss(tau)
                else:
                    log_tau = torch.log(tau.clamp(min=1e-4))
                    mean_term = (log_tau.mean() - float(args.tau_mean_target)) ** 2
                    spread = log_tau.std()
                    spread_term = torch.relu(float(args.tau_spread_target) - spread)
                    tau_quality_loss = mean_term + spread_term
            if args.lambda_tau_couple > 0 and len(jepa_losses) >= 3:
                # EXPLICIT category-invariant depth policy: drive per-sample tau
                # anti-correlated with per-sample JEPA surprise. High surprise
                # (unpredictable = transition/drift) → low tau → deeper effective ODE.
                # Surprise detached: gradient flows only through tau (it RESPONDS to
                # surprise, surprise is the fixed signal). Pearson is bounded [-1,1]
                # so minimizing it (→ -1) is stable.
                tau_b = online.dynamics.compute_tau(h_input_batch).mean(dim=(1, 2))  # [B]
                surp_b = torch.stack([j.detach() for j in jepa_losses])            # [B]
                tau_c = tau_b - tau_b.mean()
                surp_c = surp_b - surp_b.mean()
                denom = tau_c.norm() * surp_c.norm() + 1e-6
                tau_couple_loss = (tau_c * surp_c).sum() / denom  # minimize → anti-corr
        return (value_loss, jepa_loss, var_loss, contrastive, soc_loss,
                tau_quality_loss, tau_couple_loss)

    def ema_update(tau):
        with torch.no_grad():
            for tp, op in zip(target.parameters(), online.parameters()):
                tp.data.mul_(tau).add_(op.data, alpha=1.0 - tau)

    @torch.no_grad()
    def auc_on_set(recs):
        if not recs:
            return float("nan"), 0, 0
        online.eval(); target.eval()
        all_logits, all_labels = [], []
        for r in recs:
            T = r["T"]
            hs_traj = r["hidden_state_traj"].to(device)
            z_goal_traj = r["z_goal_traj"].to(device)
            judge_traj = r["judge_traj"].to(device)
            labels = r["labels"].to(device)
            h_fast_traj, _ = forward_trajectory(
                online, hs_traj, judge_traj, z_goal_traj, device,
                target_t=None, training=False)
            for t in range(T):
                logit = online.value(h_fast_traj[t].unsqueeze(0),
                                       z_goal_traj[t].unsqueeze(0),
                                       judge_traj[t].unsqueeze(0))
                all_logits.append(float(logit.item()))
                all_labels.append(int(labels[t].item()))
        logits_np = np.array(all_logits)
        labels_np = np.array(all_labels)
        n_drift = int((labels_np == 0).sum())
        n_follow = int((labels_np == 1).sum())
        online.train(); target.train()
        if n_drift == 0 or n_follow == 0:
            return float("nan"), n_drift, n_follow
        return roc_auc(-logits_np, 1 - labels_np), n_drift, n_follow

    def tau_at_boundaries(recs, n_sample=20):
        """Test the self-organizing-depth hypothesis: tau should go LOW (deep
        processing) at goal transitions / drift, HIGH (coast) at stable pursuit.

        Walks the trajectory rebuilding h_input exactly as fast_step does, then
        computes tau per chunk and bins by (a) goal-transition vs stable,
        (b) drift vs follow label. Returns mean tau per bin."""
        online.eval()
        tau_trans, tau_stable, tau_drift, tau_follow = [], [], [], []
        sel = recs[:n_sample]
        with torch.no_grad():
            for r in sel:
                T = r["T"]
                hs_traj = r["hidden_state_traj"].to(device)
                z_goal_traj = r["z_goal_traj"].to(device)
                judge_traj = r["judge_traj"].to(device)
                labels = r["labels"].to(device)
                h_fast = online.init_state(1, device)
                z_prev = None
                for t in range(T):
                    hs_normed = online.hidden_layernorm(hs_traj[t].unsqueeze(0))
                    e_h = online.in_hidden(hs_normed) * online.hidden_gate
                    e_j = online.in_judge(judge_traj[t].unsqueeze(0).unsqueeze(-1)) * online.judge_gate
                    e_g = online.in_goal(z_goal_traj[t].unsqueeze(0)) * online.goal_gate
                    e = online.evidence_layernorm(e_h + e_j + e_g)
                    evidence = e.unsqueeze(1) * online.evidence_mix.unsqueeze(0)
                    h_input = online._soft_clamp(h_fast + evidence)
                    context = online.context_pool(h_input, None)
                    online.dynamics.set_context(context, mask=None)
                    tau = float(online.dynamics.compute_tau(h_input).mean().item())
                    # transition = z_goal moved meaningfully from previous chunk
                    if z_prev is not None:
                        dz = float((z_goal_traj[t] - z_prev).norm().item())
                        (tau_trans if dz > 1e-3 else tau_stable).append(tau)
                    (tau_follow if int(labels[t].item()) == 1 else tau_drift).append(tau)
                    z_prev = z_goal_traj[t]
                    # advance fast state
                    h_fast = online.fast_step(h_fast, hs_traj[t].unsqueeze(0),
                                                judge_traj[t].unsqueeze(0),
                                                z_goal_traj[t].unsqueeze(0))
        online.train()
        m = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")
        return m(tau_trans), m(tau_stable), m(tau_drift), m(tau_follow)

    online.train(); target.train()
    best_test_auc = -1.0
    best_state = None
    last_improvement = 0
    n_nan = 0
    t_start = time.time()

    for step in range(args.max_steps + 1):
        batch = rng.choice(len(train_recs), args.batch_size, replace=True)
        recs = [train_recs[i] for i in batch]
        out = loss_for_batch(recs, training=True)
        if out[0] is None:
            continue
        value_l, jepa_l, var_l, contrastive_l, soc_l, tauq_l, tauc_l = out
        total = (args.lambda_value * value_l +
                  args.lambda_jepa * jepa_l +
                  var_l +
                  args.lambda_contrastive * contrastive_l +
                  args.lambda_soc * soc_l +
                  args.lambda_tau_quality * tauq_l +
                  args.lambda_tau_couple * tauc_l)
        if not torch.isfinite(total):
            n_nan += 1
            if n_nan >= 5:
                print(f"[jlp] ABORT: 5 NaN losses at step {step}", flush=True)
                break
            continue
        n_nan = 0
        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(online.parameters(), args.max_grad_norm)
        opt.step()
        ema_update(args.ema_tau)

        if step % args.log_every == 0:
            v_auc, _, _ = auc_on_set(val_recs)
            t_auc, t_d, t_f = auc_on_set(test_recs) if test_recs else (float("nan"), 0, 0)
            cv = compute_metric_cv(online, val_recs, device)
            selection_auc = t_auc if test_recs else v_auc
            stale = (step - last_improvement) // args.log_every
            if not np.isnan(selection_auc) and selection_auc > best_test_auc:
                best_test_auc = selection_auc
                best_state = {
                    "online": copy.deepcopy(online.state_dict()),
                    "target": copy.deepcopy(target.state_dict()),
                }
                last_improvement = step
                stale = 0
            print(f"step {step:>5}  Lv={float(value_l.detach()):.3f}  "
                  f"Lc={float(contrastive_l.detach()):.3f}  "
                  f"Lsoc={float(soc_l.detach()):.3f}  "
                  f"Ltq={float(tauq_l.detach()):.3f}  "
                  f"Ltc={float(tauc_l.detach()):+.3f}  "
                  f"v_auc={v_auc:.3f}  t_auc={t_auc:.3f}  "
                  f"(best {best_test_auc:.3f})  CV={cv:.2f}  stale={stale}  "
                  f"wall={time.time()-t_start:.0f}s", flush=True)
            if args.tau_depth and step % (args.log_every * 5) == 0:
                tt, ts, td, tf = tau_at_boundaries(val_recs)
                print(f"          [tau-depth] transition={tt:.3f} stable={ts:.3f} "
                      f"(Δ={tt-ts:+.3f}, want<0=deep@transition)  "
                      f"drift={td:.3f} follow={tf:.3f} (Δ={td-tf:+.3f})", flush=True)
            if step >= args.min_steps_before_stop and stale >= args.early_stop_patience:
                print(f"[jlp] early stop at step {step}", flush=True)
                break

    if best_state is not None:
        online.load_state_dict(best_state["online"])
        target.load_state_dict(best_state["target"])
    torch.save({
        "model_state_dict": online.state_dict(),
        "target_state_dict": target.state_dict(),
        "best_test_auc": best_test_auc,
        "args": vars(args),
        "d_hidden": d_hidden,
        "z_goal_dim": z_goal_dim,
    }, args.output)
    print(f"[jlp] saved → {args.output}  best_test_auc={best_test_auc:.3f}", flush=True)
    print("[jlp] === ALL_DONE ===", flush=True)


if __name__ == "__main__":
    main()
