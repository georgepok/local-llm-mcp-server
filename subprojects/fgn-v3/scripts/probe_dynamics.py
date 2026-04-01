"""Probe ARC Sandwich Loop dynamics — understand the ~65% ceiling.

Diagnostics:
  1. Prediction confidence: softmax entropy, top-1 probability, confidence when right vs wrong
  2. Copy baseline: what fraction of correct predictions just copy from test_input?
  3. Color confusion matrix: what errors does the model make?
  4. Iteration dynamics: how much does the hidden state change across the 8 middle loops?
  5. Per-iteration predictions: does the model get better or worse with more iterations?
  6. Attention pattern analysis: what do the middle attention layers attend to?
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fgn.config import FGNConfig
from fgn.model_arc_sandwich import SandwichARC, create_arc_model
from fgn.tasks.arc import (
    ARCTask, N_COLORS,
    ROLE_INPUT_DEMO, ROLE_OUTPUT_DEMO, ROLE_TEST_INPUT, ROLE_TEST_OUTPUT,
)


def probe_dynamics(model, eval_task, device, n_batches=30, batch_size=4):
    """Investigate prediction dynamics and iteration behavior."""
    model.eval()
    m = model._orig_mod if hasattr(model, '_orig_mod') else model

    if not isinstance(m, SandwichARC):
        print("ERROR: This probe requires SandwichARC model")
        return

    n_iters = m.middle_iters
    print(f"Model: SandwichARC, middle_iters={n_iters}")
    print(f"  bottom_geo: {len(m.bottom_geo)} layers")
    print(f"  middle_attn: {len(m.middle_attn)} layers")
    print(f"  top_geo: {len(m.top_geo)} layers")

    # Accumulators
    # Prediction confidence
    all_correct_probs = []    # top-1 prob when prediction is correct
    all_wrong_probs = []      # top-1 prob when prediction is wrong
    all_correct_entropy = []  # entropy when correct
    all_wrong_entropy = []    # entropy when wrong
    all_pred_entropy = []     # all prediction entropies

    # Copy baseline
    copy_correct = 0   # predictions that match input AND are correct
    copy_total = 0     # predictions that match input
    noncopy_correct = 0
    noncopy_total = 0

    # Color confusion: confusion[true_color][predicted_color] += 1
    confusion = np.zeros((N_COLORS, N_COLORS), dtype=np.int64)

    # Per-task accuracy
    task_accuracies = []

    # Iteration dynamics
    iter_cosine_sims = []      # cosine sim between h after iter k vs iter k-1 (at test_out positions)
    iter_delta_norms = []      # L2 norm of change per iteration
    iter_predictions = []      # [n_iters] list of accuracy at each iteration
    iter_entropies = []        # [n_iters] list of mean entropy at each iteration
    iter_prediction_changes = []  # how many predictions change between consecutive iterations

    # Attention analysis
    attn_role_fracs = defaultdict(list)  # source_role -> fraction of attention from test_out

    with torch.no_grad():
        for batch_i in range(n_batches):
            try:
                _, _, meta = eval_task.generate_batch(batch_size, device=device)
            except RuntimeError:
                continue

            colors = meta["colors"]
            xs = meta["xs"]
            ys = meta["ys"]
            roles = meta["roles"]
            sep_mask = meta["sep_mask"]
            sep_types = meta["sep_types"]
            target_mask = meta["target_mask"]
            target_labels = meta["target_labels"]
            context_mask = meta["context_mask"]
            lengths = meta["lengths"]

            B, N = colors.shape

            # Mask test output colors with input colors
            colors_masked = colors.clone()
            target_input_colors = meta.get("target_input_colors")
            if target_input_colors is not None:
                colors_masked[target_mask] = target_input_colors[target_mask]
            else:
                colors_masked[target_mask] = 10

            # ── Manual forward pass to capture per-iteration states ──

            h = m.embedding(colors_masked, xs, ys, roles, sep_mask, sep_types)
            h = m.embed_drop(h)
            context = m.context_pool(h, context_mask)

            # Bottom geo
            for layer in m.bottom_geo:
                h, kappa, m_cv, t_avg = layer(h, context, mask=None)

            # ── Middle attention: run iteration by iteration ──
            h_pre_middle = h.clone()
            h_prev = h.clone()
            iter_h_states = [h.clone()]  # state before any middle iteration

            batch_iter_preds = []  # [n_iters][B, N]
            batch_iter_entropies = []

            for iter_idx in range(n_iters):
                for layer in m.middle_attn:
                    h = layer(h, mask=None)

                iter_h_states.append(h.clone())

                # Compute predictions at this iteration point
                # Run through top geo + output head
                h_temp = h.clone()
                for layer in m.top_geo:
                    h_temp, _, _, _ = layer(h_temp, context, mask=None)
                h_normed = m.norm(h_temp)
                logits_iter = m.output_head(h_normed)
                probs_iter = F.softmax(logits_iter, dim=-1)
                preds_iter = logits_iter.argmax(dim=-1)
                entropy_iter = -(probs_iter * (probs_iter + 1e-10).log()).sum(dim=-1)

                batch_iter_preds.append(preds_iter)
                batch_iter_entropies.append(entropy_iter)

                # Cosine sim and delta norm vs previous iteration
                # Only at target (test_output) positions
                for b in range(B):
                    tgt_pos = target_mask[b].nonzero(as_tuple=True)[0]
                    if tgt_pos.shape[0] == 0:
                        continue

                    h_cur_tgt = h[b, tgt_pos]       # [n_tgt, d]
                    h_prev_tgt = h_prev[b, tgt_pos]

                    # Cosine similarity
                    cos = F.cosine_similarity(h_cur_tgt, h_prev_tgt, dim=-1).mean().item()
                    delta = (h_cur_tgt - h_prev_tgt).norm(dim=-1).mean().item()

                    while len(iter_cosine_sims) <= iter_idx:
                        iter_cosine_sims.append([])
                        iter_delta_norms.append([])
                    iter_cosine_sims[iter_idx].append(cos)
                    iter_delta_norms[iter_idx].append(delta)

                h_prev = h.clone()

            # ── Final predictions (from last iteration, standard path) ──
            # Use the last iter's logits
            final_logits = logits_iter
            final_probs = probs_iter
            final_preds = preds_iter
            final_entropy = entropy_iter

            # ── Per-item analysis ──
            for b in range(B):
                tgt = target_labels[b]
                valid = tgt != -100
                n_valid = valid.sum().item()
                if n_valid == 0:
                    continue

                preds_b = final_preds[b][valid]
                labels_b = tgt[valid]
                probs_b = final_probs[b][valid]   # [n_valid, 10]
                entropy_b = final_entropy[b][valid]

                correct_mask = (preds_b == labels_b)
                item_acc = correct_mask.float().mean().item()
                task_accuracies.append(item_acc)

                # Top-1 probability
                top1_probs = probs_b.max(dim=-1).values
                if correct_mask.sum() > 0:
                    all_correct_probs.extend(top1_probs[correct_mask].cpu().tolist())
                    all_correct_entropy.extend(entropy_b[correct_mask].cpu().tolist())
                if (~correct_mask).sum() > 0:
                    all_wrong_probs.extend(top1_probs[~correct_mask].cpu().tolist())
                    all_wrong_entropy.extend(entropy_b[~correct_mask].cpu().tolist())
                all_pred_entropy.extend(entropy_b.cpu().tolist())

                # Copy baseline
                input_colors_b = None
                if target_input_colors is not None:
                    input_colors_b = target_input_colors[b][valid]

                if input_colors_b is not None:
                    is_copy = (labels_b == input_colors_b)  # ground truth matches input
                    copy_positions = is_copy.sum().item()
                    noncopy_positions = (~is_copy).sum().item()

                    copy_correct += (preds_b[is_copy] == labels_b[is_copy]).sum().item() if copy_positions > 0 else 0
                    copy_total += copy_positions
                    noncopy_correct += (preds_b[~is_copy] == labels_b[~is_copy]).sum().item() if noncopy_positions > 0 else 0
                    noncopy_total += noncopy_positions

                # Confusion matrix
                for i in range(n_valid):
                    true_c = labels_b[i].item()
                    pred_c = preds_b[i].item()
                    if 0 <= true_c < N_COLORS and 0 <= pred_c < N_COLORS:
                        confusion[true_c][pred_c] += 1

                # Per-iteration accuracy and entropy
                for iter_idx in range(n_iters):
                    preds_it = batch_iter_preds[iter_idx][b][valid]
                    entropy_it = batch_iter_entropies[iter_idx][b][valid]
                    acc_it = (preds_it == labels_b).float().mean().item()
                    ent_it = entropy_it.mean().item()

                    while len(iter_predictions) <= iter_idx:
                        iter_predictions.append([])
                        iter_entropies.append([])
                    iter_predictions[iter_idx].append(acc_it)
                    iter_entropies[iter_idx].append(ent_it)

                # Prediction changes between iterations
                for iter_idx in range(1, n_iters):
                    prev_preds = batch_iter_preds[iter_idx - 1][b][valid]
                    curr_preds = batch_iter_preds[iter_idx][b][valid]
                    n_changed = (prev_preds != curr_preds).sum().item()
                    frac_changed = n_changed / max(n_valid, 1)
                    while len(iter_prediction_changes) <= iter_idx:
                        iter_prediction_changes.append([])
                    iter_prediction_changes[iter_idx].append(frac_changed)

            # ── Attention pattern analysis (last iteration) ──
            # Extract attention weights from the middle attention layers
            # We need to manually compute attention to get the weights
            h_for_attn = iter_h_states[-2]  # state before last iteration
            for layer in m.middle_attn:
                attn_mod = layer.attention
                h_normed = layer.norm_attn(h_for_attn)

                # Compute Q, K manually
                Q = attn_mod.W_q(h_normed)
                K = attn_mod.W_k(h_normed)

                nh = attn_mod.n_heads
                d_h = Q.shape[-1] // nh

                Q = Q.view(B, N, nh, d_h).transpose(1, 2)  # [B, nh, N, d_h]
                K = K.view(B, N, nh, d_h).transpose(1, 2)

                scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_h ** 0.5)
                attn_weights = F.softmax(scores, dim=-1)  # [B, nh, N, N]

                # Average over heads
                attn_avg = attn_weights.mean(dim=1)  # [B, N, N]

                for b in range(B):
                    tgt_pos = ((roles[b] == ROLE_TEST_OUTPUT) & (~sep_mask[b])).nonzero(as_tuple=True)[0]
                    if tgt_pos.shape[0] == 0:
                        continue

                    # Attention from test_output to each role
                    attn_from_tgt = attn_avg[b, tgt_pos]  # [n_tgt, N]

                    for role_id, role_name in [(0, "demo_in"), (1, "demo_out"),
                                               (2, "test_in"), (3, "test_out")]:
                        role_pos = ((roles[b] == role_id) & (~sep_mask[b]))
                        if role_pos.sum() > 0:
                            frac = attn_from_tgt[:, role_pos].sum(dim=-1).mean().item()
                            attn_role_fracs[role_name].append(frac)

                    sep_pos = sep_mask[b]
                    if sep_pos.sum() > 0:
                        frac = attn_from_tgt[:, sep_pos].sum(dim=-1).mean().item()
                        attn_role_fracs["separator"].append(frac)

                # Only probe first middle attn layer
                break

            if (batch_i + 1) % 10 == 0:
                print(f"  Probed {batch_i + 1}/{n_batches} batches...")

    # ── Report ──────────────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print(f"  ARC Sandwich Loop — Dynamics Probe")
    print(f"{'='*70}")

    # 1. Overall accuracy
    accs = task_accuracies
    print(f"\n  Overall: {len(accs)} tasks, mean acc={np.mean(accs):.4f}, "
          f"median={np.median(accs):.4f}")
    bins = [0, 0.25, 0.5, 0.75, 0.9, 1.0, 1.01]
    hist, _ = np.histogram(accs, bins=bins)
    labels = ["0-25%", "25-50%", "50-75%", "75-90%", "90-99%", "100%"]
    for label, count in zip(labels, hist):
        bar = "#" * (count * 40 // max(len(accs), 1))
        print(f"    {label:>7s}: {count:4d} ({100*count/len(accs):5.1f}%) {bar}")

    # 2. Prediction confidence
    print(f"\n  Prediction confidence:")
    if all_correct_probs:
        print(f"    Correct predictions (n={len(all_correct_probs)}):")
        print(f"      top-1 prob: mean={np.mean(all_correct_probs):.4f}, "
              f"median={np.median(all_correct_probs):.4f}")
        print(f"      entropy:    mean={np.mean(all_correct_entropy):.4f}")
    if all_wrong_probs:
        print(f"    Wrong predictions (n={len(all_wrong_probs)}):")
        print(f"      top-1 prob: mean={np.mean(all_wrong_probs):.4f}, "
              f"median={np.median(all_wrong_probs):.4f}")
        print(f"      entropy:    mean={np.mean(all_wrong_entropy):.4f}")
    if all_pred_entropy:
        print(f"    Overall entropy: mean={np.mean(all_pred_entropy):.4f}, "
              f"median={np.median(all_pred_entropy):.4f}")

    # 3. Copy baseline
    print(f"\n  Copy baseline (does ground truth = test_input color?):")
    print(f"    Copy positions:    {copy_total:6d} ({100*copy_total/max(copy_total+noncopy_total,1):.1f}%)")
    print(f"    Non-copy positions: {noncopy_total:6d} ({100*noncopy_total/max(copy_total+noncopy_total,1):.1f}%)")
    if copy_total > 0:
        print(f"    Copy accuracy:     {copy_correct/copy_total:.4f} "
              f"({copy_correct}/{copy_total})")
    if noncopy_total > 0:
        print(f"    Non-copy accuracy: {noncopy_correct/noncopy_total:.4f} "
              f"({noncopy_correct}/{noncopy_total})")
    print(f"    → If model just copied input: {copy_total/max(copy_total+noncopy_total,1):.4f} accuracy")

    # 4. Color confusion matrix
    print(f"\n  Color confusion matrix (rows=true, cols=predicted):")
    print(f"    {'':>3s}", end="")
    for c in range(N_COLORS):
        print(f" {c:>5d}", end="")
    print(f"  {'acc':>6s}")
    for true_c in range(N_COLORS):
        row_total = confusion[true_c].sum()
        if row_total == 0:
            continue
        row_acc = confusion[true_c][true_c] / row_total
        print(f"    {true_c:>3d}", end="")
        for pred_c in range(N_COLORS):
            count = confusion[true_c][pred_c]
            if count == 0:
                print(f"   {'·':>3s}", end="")
            else:
                pct = 100 * count / row_total
                print(f" {pct:5.1f}", end="")
        print(f"  {row_acc:6.3f}")

    # 5. Iteration dynamics
    print(f"\n  Iteration dynamics ({n_iters} iterations of {len(m.middle_attn)} attn layers):")
    print(f"    {'Iter':>4s}  {'cos_sim':>8s}  {'Δ_norm':>8s}  {'accuracy':>8s}  {'entropy':>8s}  {'%changed':>8s}")
    for i in range(n_iters):
        cos = np.mean(iter_cosine_sims[i]) if i < len(iter_cosine_sims) and iter_cosine_sims[i] else 0
        delta = np.mean(iter_delta_norms[i]) if i < len(iter_delta_norms) and iter_delta_norms[i] else 0
        acc = np.mean(iter_predictions[i]) if i < len(iter_predictions) and iter_predictions[i] else 0
        ent = np.mean(iter_entropies[i]) if i < len(iter_entropies) and iter_entropies[i] else 0
        changed = np.mean(iter_prediction_changes[i]) if i < len(iter_prediction_changes) and iter_prediction_changes[i] else 0
        print(f"    {i+1:>4d}  {cos:8.5f}  {delta:8.2f}  {acc:8.4f}  {ent:8.4f}  {100*changed:7.2f}%")

    # 6. Attention patterns
    print(f"\n  Middle attention: test_output attends to (head-averaged):")
    total_frac = 0
    for role_name in ["demo_in", "demo_out", "test_in", "test_out", "separator"]:
        vals = attn_role_fracs.get(role_name, [])
        if vals:
            mean_frac = np.mean(vals)
            total_frac += mean_frac
            bar = "#" * int(mean_frac * 50)
            print(f"    {role_name:>10s}: {mean_frac:.4f} {bar}")
    if total_frac > 0:
        print(f"    {'total':>10s}: {total_frac:.4f}")

    # 7. Key insights
    print(f"\n{'='*70}")
    print(f"  Key findings:")
    if all_wrong_probs and all_correct_probs:
        print(f"    - Model confidence: correct={np.mean(all_correct_probs):.3f}, wrong={np.mean(all_wrong_probs):.3f}")
        if np.mean(all_wrong_probs) > 0.5:
            print(f"      → Model is CONFIDENTLY WRONG (high prob on wrong predictions)")
        else:
            print(f"      → Model is uncertain on errors (low prob)")
    if copy_total > 0 and noncopy_total > 0:
        copy_acc = copy_correct / copy_total
        noncopy_acc = noncopy_correct / noncopy_total
        copy_frac = copy_total / (copy_total + noncopy_total)
        print(f"    - Copy baseline: {copy_frac:.1%} of cells match input, "
              f"copy_acc={copy_acc:.3f}, transform_acc={noncopy_acc:.3f}")
        if copy_acc > noncopy_acc + 0.1:
            print(f"      → Model much better at copying ({copy_acc:.3f}) than transforming ({noncopy_acc:.3f})")
    if iter_predictions and len(iter_predictions) >= 2:
        first_acc = np.mean(iter_predictions[0])
        last_acc = np.mean(iter_predictions[-1])
        print(f"    - Iteration effect: iter1={first_acc:.4f} → iter{n_iters}={last_acc:.4f} "
              f"(Δ={last_acc-first_acc:+.4f})")
        if abs(last_acc - first_acc) < 0.01:
            print(f"      → Iterations have NEGLIGIBLE effect on accuracy")
        elif last_acc < first_acc:
            print(f"      → Later iterations HURT accuracy!")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Probe ARC Sandwich Loop dynamics")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/arc")
    parser.add_argument("--n_batches", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    config = FGNConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_arc_model(config, device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    step = ckpt.get("step", "?")
    best_acc = ckpt.get("best_eval_acc", "?")
    print(f"Loaded checkpoint from step {step} (best_eval_acc={best_acc})")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    eval_task = ARCTask(
        seq_len=config.max_seq_len,
        data_dir=args.data_dir,
        split="eval",
        augment=False,
    )

    probe_dynamics(model, eval_task, device,
                   n_batches=args.n_batches,
                   batch_size=args.batch_size)


if __name__ == "__main__":
    main()
