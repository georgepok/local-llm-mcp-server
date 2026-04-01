"""Dream-powered Test-Time Training V2 — VQ Encoder + AR Decoder.

V2 changes over V1:
1. VQEncoder replaces ConceptEncoder — encodes to discrete z_q
2. ARDecoder replaces DreamDecoder — generates crisp integer grids
3. W_o explicitly verified in melt set (was already present in V1)

At test time:
1. Encode demos -> z_q (quantized)
2. Clone ODE
3. Generate crisp dreams from z_q + random grids via AR decoder
4. TTT the cloned ODE on dreams
5. Return sculpted ODE
"""

import random
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

# Import from liquid-arc
_LIQUID_ARC_ROOT = str(Path(__file__).resolve().parent.parent.parent / "liquid-arc")
if _LIQUID_ARC_ROOT not in sys.path:
    sys.path.insert(0, _LIQUID_ARC_ROOT)

from liquid_arc.model import LiquidARCModel
from liquid_arc.tasks.procedural import build_sequence

from .config import WakeSleepConfig
from .vq_encoder import VQEncoder
from .ar_decoder import ARDecoder
from .model import forward_with_external_context
from .wake_sleep import collate_sequences, extract_grid_pairs


def dream_ttt(
    base_model: LiquidARCModel,
    encoder: VQEncoder,
    decoder: ARDecoder,
    z_to_context: nn.Module,
    task: dict,
    config: WakeSleepConfig,
    device: torch.device,
    verbose: bool = False,
) -> LiquidARCModel:
    """Dream-powered Test-Time Training V2.

    1. Encode demos -> z_q (quantized via VQ)
    2. Clone ODE
    3. Generate crisp dreams from z_q + random grids via AR decoder
    4. TTT the cloned ODE on dreams for ws_dream_ttt_steps steps
    5. Return sculpted ODE

    Args:
        base_model: pre-trained LiquidARCModel (not modified)
        encoder: trained VQEncoder
        decoder: trained ARDecoder
        z_to_context: trained Linear(z_dim, d_model) projection
        task: ARC task dict with "train" (demos) and "test" keys
        config: WakeSleepConfig
        device: torch device
        verbose: print per-step loss

    Returns:
        Adapted (sculpted) LiquidARCModel
    """
    # 1. Extract concept from demo pairs (quantized)
    demo_pairs = extract_grid_pairs(task, device)
    with torch.no_grad():
        _, z_q, _, _ = encoder(demo_pairs)  # z_q [1, L, z_dim]
        # Mean-pool spatial tokens to match z_to_context input dim [1, z_dim]
        z_context = z_to_context(z_q.mean(dim=1))  # [1, d_model]

    # 2. Clone via state_dict (deepcopy fails on torch.compiled models)
    model_clone = LiquidARCModel(base_model.config)
    # Load state dict from unwrapped model
    src_state = base_model.state_dict()
    # Strip _orig_mod prefix from compiled model keys
    clean_state = {}
    for k, v in src_state.items():
        clean_k = k.replace("dynamics._orig_mod.", "dynamics.")
        clean_state[clean_k] = v.clone()
    model_clone.load_state_dict(clean_state, strict=False)
    model_clone.to(device)
    model_clone.train()

    # Freeze everything
    for p in model_clone.parameters():
        p.requires_grad = False

    # Melt MetricNet + W_o + TauNet/GateNet (full WHERE+WHEN+WHAT plasticity)
    melt_modules = [
        model_clone.dynamics.metric_net_linear1,
        model_clone.dynamics.metric_net_linear2,
        model_clone.dynamics.W_o,  # WHAT transformation — explicitly mandated for V2
    ]
    if config.channel_gate_enabled:
        melt_modules += [
            model_clone.dynamics.gate_net_linear1,
            model_clone.dynamics.gate_net_linear2,
        ]
    else:
        melt_modules += [
            model_clone.dynamics.tau_net_linear1,
            model_clone.dynamics.tau_net_linear2,
        ]

    n_unfrozen = 0
    for m in melt_modules:
        for p in m.parameters():
            p.requires_grad_(True)
            n_unfrozen += p.numel()

    opt = torch.optim.Adam(
        [p for m in melt_modules for p in m.parameters()],
        lr=config.ws_dream_ttt_lr,
    )

    # 3. Dream-TTT loop
    ce_history = []
    for step in range(config.ws_dream_ttt_steps):
        # Generate random grid
        H = random.randint(config.ws_dream_grid_min, config.ws_dream_grid_max)
        W = random.randint(config.ws_dream_grid_min, config.ws_dream_grid_max)
        rand_grid = torch.randint(0, 10, (1, H, W), device=device)

        # AR Decoder generates crisp integer targets (no .clamp needed)
        with torch.no_grad():
            dream_target = decoder.dream(z_q, rand_grid, temperature=0.0)

        # Use another random grid as test
        H2 = random.randint(config.ws_dream_grid_min, config.ws_dream_grid_max)
        W2 = random.randint(config.ws_dream_grid_min, config.ws_dream_grid_max)
        test_in = torch.randint(0, 10, (1, H2, W2), device=device)
        with torch.no_grad():
            test_out = decoder.dream(z_q, test_in, temperature=0.0)

        # Serialize dream as ODE input
        demo = (rand_grid[0].cpu().tolist(), dream_target[0].cpu().tolist())
        seq = build_sequence([demo], test_in[0].cpu().tolist(), test_out[0].cpu().tolist())
        batch = collate_sequences([seq], device, config.max_seq_len)

        # ODE forward
        opt.zero_grad()
        result = forward_with_external_context(
            model_clone,
            z_context=z_context,
            colors=batch["colors"],
            xs=batch["xs"],
            ys=batch["ys"],
            roles=batch["roles"],
            sep_mask=batch["sep_mask"],
            sep_types=batch["sep_types"],
            target_mask=batch["target_mask"],
            target_labels=batch["target_labels"],
            context_mask=batch["context_mask"],
            target_input_colors=batch["target_input_colors"],
            grid_ids=batch["grid_ids"],
            n_steps=config.n_ode_steps,
        )

        # Use xform_loss (CE on transform cells only) like existing TTT V2
        xf = result.get("xform_loss", result["ce_loss"])
        loss = xf + config.curvature_lambda * result["avg_kappa"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for m in melt_modules for p in m.parameters()], 1.0)
        opt.step()

        ce_val = xf.item()
        ce_history.append(ce_val)

        if verbose and step % 10 == 0:
            print(f"    Dream-TTT step {step}: xf_ce={ce_val:.4f}, "
                  f"|k|={result['avg_kappa'].item():.4f}")

    if verbose:
        print(f"    Dream-TTT done: {len(ce_history)} steps, "
              f"CE {ce_history[0]:.4f} -> {ce_history[-1]:.4f}, "
              f"{n_unfrozen} params unfrozen")

    return model_clone


def evaluate_dream_ttt(
    base_model: LiquidARCModel,
    encoder: VQEncoder,
    decoder: ARDecoder,
    z_to_context: nn.Module,
    data_dir: str,
    config: WakeSleepConfig,
    device: torch.device,
    n_tasks: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[float, float]:
    """Full Dream-TTT V2 evaluation on ARC eval tasks.

    Returns:
        (cell_accuracy, transform_accuracy) aggregated over all test pairs
    """
    # Import ARC data utilities
    FGN_ROOT = str(Path(__file__).resolve().parent.parent.parent / "fgn-v3")
    if FGN_ROOT not in sys.path:
        sys.path.insert(0, FGN_ROOT)
    from fgn.tasks.arc import load_arc_tasks, build_sequence as arc_build_sequence, pad_single_to_batch

    all_tasks = load_arc_tasks(data_dir)
    eval_tasks = all_tasks.get("eval", [])
    if not eval_tasks:
        print("  WARNING: No eval tasks found")
        return 0.0, 0.0

    if n_tasks is not None:
        eval_tasks = eval_tasks[:n_tasks]

    total_cell = 0.0
    total_xform = 0.0
    count = 0
    skipped = 0

    base_model.eval()
    encoder.eval()
    decoder.eval()

    import time
    t0 = time.time()

    for i, task in enumerate(eval_tasks):
        task_id = task.get("task_id", f"task_{i}")

        # Adapt via Dream-TTT
        try:
            adapted = dream_ttt(
                base_model, encoder, decoder, z_to_context,
                task, config, device, verbose=verbose,
            )
        except Exception as e:
            skipped += 1
            print(f"  [{i+1}/{len(eval_tasks)}] {task_id}: SKIPPED ({e})")
            continue

        adapted.eval()

        # Evaluate on all test pairs
        for test_idx in range(len(task["test"])):
            seq = arc_build_sequence(task, d4_idx=0, color_perm=None,
                                     test_idx=test_idx, max_seq_len=config.max_seq_len)
            if seq is None:
                skipped += 1
                continue

            meta = pad_single_to_batch(seq, config.max_seq_len, device)

            # Get z_context for this task
            demo_pairs = extract_grid_pairs(task, device)
            with torch.no_grad():
                _, z_q, _, _ = encoder(demo_pairs)  # z_q [1, L, z_dim]
                z_ctx = z_to_context(z_q.mean(dim=1))  # mean-pool -> [1, d_model]

                result = forward_with_external_context(
                    adapted,
                    z_context=z_ctx,
                    colors=meta["colors"],
                    xs=meta["xs"],
                    ys=meta["ys"],
                    roles=meta["roles"],
                    sep_mask=meta["sep_mask"],
                    sep_types=meta["sep_types"],
                    target_mask=meta["target_mask"],
                    target_labels=meta["target_labels"],
                    context_mask=meta["context_mask"],
                    target_input_colors=meta.get("target_input_colors"),
                    grid_ids=meta.get("grid_ids"),
                    n_steps=config.n_ode_steps,
                )

            cell_acc = result.get("cell_accuracy", torch.tensor(0.0))
            if isinstance(cell_acc, torch.Tensor):
                cell_acc = cell_acc.item()
            xform_acc = result.get("transform_accuracy", torch.tensor(0.0))
            if isinstance(xform_acc, torch.Tensor):
                xform_acc = xform_acc.item()

            total_cell += cell_acc
            total_xform += xform_acc
            count += 1

        del adapted

        if verbose and (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(eval_tasks)}] {elapsed:.1f}s, "
                  f"cell={total_cell/max(count,1):.4f}, "
                  f"xform={total_xform/max(count,1):.4f}")

    elapsed = time.time() - t0
    cell_acc = total_cell / max(count, 1)
    xform_acc = total_xform / max(count, 1)

    print(f"  Dream-TTT eval: {len(eval_tasks)} tasks ({skipped} skipped), {elapsed:.1f}s")
    print(f"  Cell acc: {cell_acc:.4f}, Xform acc: {xform_acc:.4f}")

    return cell_acc, xform_acc
