"""Post-hoc CV (phase transition diagnostic) on JudgeLiquid checkpoint."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from judge_liquid import JudgeLiquid, forward_trajectory


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True, help="judged trajectory data")
    p.add_argument("--n_samples", type=int, default=30)
    args = p.parse_args()

    device = torch.device("cuda")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sa = ck.get("args", {})
    model = JudgeLiquid(
        d_hidden=ck["d_hidden"], z_goal_dim=ck["z_goal_dim"],
        d=sa.get("d", 64), K=sa.get("K", 4),
        value_hidden=sa.get("value_hidden", 192),
        d_metric=sa.get("d_metric", 16),
        d_ffn=sa.get("d_ffn", 128),
        n_ode_steps=sa.get("n_ode_steps", 3),
    ).to(device).eval()
    model.load_state_dict(ck["model_state_dict"], strict=False)
    print(f"[cv] loaded ckpt d={sa.get('d')} K={sa.get('K')} d_metric={sa.get('d_metric')}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[cv] {n_params:,} params, best_test_auc={ck.get('best_test_auc','?')}")

    pack = torch.load(args.data, map_location="cpu", weights_only=False)
    records = pack["records"][:args.n_samples]

    all_diag = []
    with torch.no_grad():
        for r in records:
            T = int(r["T"])
            if T < 2 or "judge_traj" not in r:
                continue
            hs_traj = r["hidden_state_traj"].float().to(device)
            z_g = r["z_goal_traj"].float().to(device)
            judge_traj = r["judge_traj"].float().to(device)
            h_fast_traj, _ = forward_trajectory(
                model, hs_traj, judge_traj, z_g, device, target_t=None, training=False)
            # Replicate fast_step internals to set dynamics context, then read metric
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

    arr = np.concatenate(all_diag)
    cv = float(arr.std() / max(1e-8, arr.mean()))
    print(f"\n[cv] Metric eigenvalue CV: {cv:.3f}")
    print(f"[cv]   distribution: min={arr.min():.4f} max={arr.max():.4f} "
          f"mean={arr.mean():.4f} std={arr.std():.4f}")
    print(f"[cv]   n_eigenvalues sampled: {len(arr)}")
    print(f"\n[cv] Phase transition reference:")
    print(f"     pre-transition (flat geometry):  CV < 1")
    print(f"     mid-transition:                  CV 1-5")
    print(f"     post-transition (rich curvature): CV > 6 (robotics target)")


if __name__ == "__main__":
    main()
