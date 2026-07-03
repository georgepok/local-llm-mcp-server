"""ODE-convergence diagnostic for LiquidARC.

Foundational question: is a trained LiquidARC model actually approximating a
continuous-time ODE, or is it encoding "exactly N specific Euler iterations"
as its computation?

Test: load a trained checkpoint, run inference at multiple n_steps values
(holding integration_time fixed). Two diagnostics:

1. Output stability — does cell/transform accuracy stay roughly constant
   as n_steps grows beyond the trained value? An ODE-like model should
   improve or plateau (more steps = more accurate integration). A pure
   "weight-tied N-step recurrent network" should peak at the trained N
   and degrade at other values.

2. Hidden-state convergence — does ||h_final(n) - h_final(2n)|| /
   ||h_final(2n)|| shrink as n grows? An ODE has a convergent trajectory:
   the residual should decay roughly linearly in dt (or faster for higher-
   order solvers). A non-ODE model has no such structure — the residual
   stays bounded away from zero.

Usage:
    python scripts/eval_ode_convergence.py \
        --checkpoint output_30m/checkpoints/best.pt \
        --data_dir /home/pokazge/arc-data \
        --n_steps_list 4,8,16,32,64,128 \
        --n_batches 8

If output stabilises and the residual decays as n grows: the ODE framing
is empirically defensible. If output peaks at the trained n and degrades
elsewhere: the model is a weight-tied N-step recurrent network that just
happens to use Euler-form updates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liquid_arc.config import LiquidARCConfig
from liquid_arc.model import create_model

FGN_ROOT = str(Path(__file__).resolve().parent.parent.parent / "fgn-v3")
sys.path.insert(0, FGN_ROOT)
from fgn.tasks.arc import ARCTask


@torch.no_grad()
def run_eval_at_n_steps(model, eval_task, device, n_batches, batch_size,
                        n_steps: int, capture_h: bool = True):
    """Run inference at the given n_steps. Returns dict with cell accuracy,
    transform accuracy, mean CE, and stacked h_final samples for stability."""
    model.eval()
    total_correct = 0
    total_cells = 0
    total_xform_correct = 0
    total_xform = 0
    total_ce = 0.0
    n_valid = 0
    h_samples = []        # list of [B, N_target_pos, d] (mean-pooled to [B, d])

    for i in range(n_batches):
        _, _, meta = eval_task.generate_batch(batch_size, device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                 enabled=(device.type == "cuda")):
            result = model(
                colors=meta["colors"], xs=meta["xs"], ys=meta["ys"],
                roles=meta["roles"], sep_mask=meta["sep_mask"],
                sep_types=meta["sep_types"], target_mask=meta["target_mask"],
                target_labels=meta["target_labels"],
                context_mask=meta["context_mask"],
                grid_ids=meta.get("grid_ids"),
                lengths=meta.get("lengths"),
                target_input_colors=meta.get("target_input_colors"),
                n_steps=n_steps,                  # <-- the override
            )
        cell_acc = result.get("cell_accuracy", torch.tensor(0.0))
        cell_acc = cell_acc.item() if isinstance(cell_acc, torch.Tensor) else cell_acc
        xform_acc = result.get("transform_accuracy", torch.tensor(0.0))
        xform_acc = xform_acc.item() if isinstance(xform_acc, torch.Tensor) else xform_acc
        ce = result.get("loss_ce", result.get("ce_loss", torch.tensor(0.0)))
        ce = ce.item() if isinstance(ce, torch.Tensor) else ce

        # Cell counts approximate via accuracy * batch_size (eval loop uses
        # mean — for a stability test, batch-level mean is enough)
        total_correct += cell_acc * batch_size
        total_cells += batch_size
        total_xform_correct += xform_acc * batch_size
        total_xform += batch_size
        total_ce += ce
        n_valid += 1

        if capture_h:
            h = result["h_final"].detach().float()  # [B, N, d]
            # Pool over only target-mask positions (those that matter for output)
            tm = meta["target_mask"].unsqueeze(-1)
            h_pooled = (h * tm).sum(dim=1) / tm.sum(dim=1).clamp(min=1)  # [B, d]
            h_samples.append(h_pooled.cpu())

    return {
        "cell_acc": total_correct / max(total_cells, 1),
        "xform_acc": total_xform_correct / max(total_xform, 1),
        "ce": total_ce / max(n_valid, 1),
        "h_pooled": torch.cat(h_samples, dim=0) if h_samples else None,  # [B*n_batches, d]
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_dir", type=str, default="/home/pokazge/arc-data")
    p.add_argument("--n_steps_list", type=str, default="4,8,16,32,64,128")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_batches", type=int, default=8)
    p.add_argument("--out", type=str, default="ode_convergence.json")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_steps_list = [int(x) for x in args.n_steps_list.split(",")]
    print(f"Device: {device}")
    print(f"n_steps_list: {n_steps_list}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    if isinstance(config, dict):
        config = LiquidARCConfig(**config)
    print(f"Trained config: d_model={config.d_model}, "
          f"n_ode_steps={config.n_ode_steps}, "
          f"integration_time={getattr(config, 'integration_time', 1.0)}, "
          f"tau=[{config.tau_min}, {config.tau_max}], "
          f"routing={config.routing_mode}")

    model = create_model(config, device)
    sd = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    # Strip torch.compile's _orig_mod. prefix (canonical fix per project memory)
    sd = {k.replace("._orig_mod.", "."): v for k, v in sd.items()}
    # Schema drift: older checkpoints have metric_net_linear2 → renamed to
    # metric_net_linear2_diag after low-rank metric was added. Map it through.
    remapped = {}
    for k, v in sd.items():
        if "metric_net_linear2." in k and "metric_net_linear2_diag" not in k:
            k = k.replace("metric_net_linear2.", "metric_net_linear2_diag.")
        remapped[k] = v
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing:
        print(f"  [load] {len(missing)} missing keys (kept at init): "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  [load] {len(unexpected)} unexpected keys (ignored): "
              f"{unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    step = ckpt.get("step", "?")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded ckpt step={step}, params={n_params:,}\n")

    eval_task = ARCTask(
        seq_len=config.max_seq_len,
        data_dir=args.data_dir,
        split="eval",
        augment=False,
    )

    # Use the SAME random seed/data per n_steps run so accuracy diff is purely
    # solver-driven, not data-driven.
    base_seed = args.seed
    results = {}
    for n in n_steps_list:
        torch.manual_seed(base_seed)
        if hasattr(eval_task, "rng"):
            try:
                import random
                eval_task.rng = random.Random(base_seed)
            except Exception:
                pass
        r = run_eval_at_n_steps(model, eval_task, device,
                                args.n_batches, args.batch_size,
                                n_steps=n, capture_h=True)
        # h_pooled tensor not JSON-serialisable — keep separately for diff
        h_pooled = r.pop("h_pooled")
        r["n_steps"] = n
        results[n] = (r, h_pooled)
        print(f"  n_steps={n:4d}  cell_acc={r['cell_acc']*100:6.2f}%  "
              f"xform_acc={r['xform_acc']*100:6.2f}%  ce={r['ce']:.4f}")

    # Hidden-state convergence: ||h(n) - h(2n)|| / ||h(2n)||
    print("\nHidden-state stability (lower = more ODE-like):")
    print(f"  {'n_steps':>10}  {'rel_residual':>15}  {'cos_sim_with_max':>18}")
    h_max_n = max(n_steps_list)
    _, h_max = results[h_max_n]
    norm_max = h_max.norm(dim=-1).mean().item()
    for n in n_steps_list:
        r, h = results[n]
        # Relative residual vs the highest-n_steps reference (treated as "ODE limit")
        diff = (h - h_max).norm(dim=-1).mean().item()
        rel = diff / max(norm_max, 1e-8)
        # Cosine similarity per example, averaged
        cos = torch.nn.functional.cosine_similarity(h, h_max, dim=-1).mean().item()
        print(f"  {n:>10d}  {rel:>15.6f}  {cos:>18.6f}")
        r["rel_residual_vs_max"] = rel
        r["cos_sim_vs_max"] = cos

    # Pairwise n-vs-2n diagnostic — does the residual decay as n grows?
    print("\nPairwise stability (||h(n) - h(2n)|| / ||h(2n)||) — should "
          "decay as n grows for ODE-like behaviour:")
    for i, n in enumerate(n_steps_list[:-1]):
        n2 = n_steps_list[i + 1]
        _, h_n = results[n]
        _, h_2n = results[n2]
        norm_2n = h_2n.norm(dim=-1).mean().item()
        diff = (h_n - h_2n).norm(dim=-1).mean().item()
        rel = diff / max(norm_2n, 1e-8)
        print(f"  n={n:4d} → 2n={n2:4d}: rel_residual = {rel:.6f}")

    # Save JSON summary
    json_results = {n: r for n, (r, _) in results.items()}
    json_results["meta"] = {
        "trained_n_ode_steps": config.n_ode_steps,
        "integration_time": getattr(config, "integration_time", 1.0),
        "tau_min": config.tau_min, "tau_max": config.tau_max,
        "routing_mode": config.routing_mode,
        "ckpt_step": step,
        "n_steps_list": n_steps_list,
        "checkpoint": args.checkpoint,
    }
    with open(args.out, "w") as f:
        json.dump(json_results, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, torch.Tensor) else str(o))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
