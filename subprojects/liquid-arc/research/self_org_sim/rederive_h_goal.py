"""Re-derive h_goal trajectories using a NEW substrate body (e.g. JEPA-trained).
Reads saved JEPA-extended trajectory data (z_vl, z_lang, state8, chunk inputs),
forwards them through the new substrate (no GR00T needed), saves new h_goal_traj
in same format as input — so probe scripts can compare predictive power.

Usage:
  python rederive_h_goal.py \
    --in_files /tmp/traj_jepa_libero10_s10.pt,... \
    --substrate_ckpt /tmp/substrate_jepa_ema.pt \
    --out_dir /tmp/traj_jepa_rederived
"""
from __future__ import annotations
import argparse
import sys
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_files", required=True)
    p.add_argument("--substrate_ckpt", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--use_evidence_layernorm", action="store_true",
                   help="Enable LayerNorm at inference (matches JEPA-trained substrate)")
    p.add_argument("--h_input_clamp", type=float, default=50.0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[rederive] device={device}", flush=True)

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
    substrate.use_evidence_layernorm = args.use_evidence_layernorm
    substrate.h_input_clamp = args.h_input_clamp
    substrate.eval()
    print(f"[rederive] substrate loaded; K={K_bel} d={d_sub} "
          f"LayerNorm={args.use_evidence_layernorm} clamp={args.h_input_clamp}",
          flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    for fp in [x.strip() for x in args.in_files.split(",") if x.strip()]:
        ck_in = torch.load(fp, map_location="cpu", weights_only=False)
        records = ck_in["records"]
        new_records = []
        n_skip = 0
        for r in records:
            if "z_vl_traj" not in r:
                n_skip += 1
                continue
            z_vl = r["z_vl_traj"].float().to(device)
            z_lang = r["z_lang_traj"].float().to(device)
            state8 = r["state8_traj"].float().to(device)
            chunks = r["chunk_traj"].float().to(device)
            z_goal = r["z_goal"].float().to(device).unsqueeze(0)
            T = z_vl.shape[0]
            h = substrate.init_state(1, device)
            h_traj_new = []
            with torch.no_grad():
                for t in range(T):
                    h, _, _, _ = substrate.step(
                        h, z_vl[t].unsqueeze(0), z_goal,
                        chunks[t].unsqueeze(0), state8[t].unsqueeze(0),
                        z_lang_t=z_lang[t].unsqueeze(0),
                    )
                    h_traj_new.append(h[0].cpu())
            new_h_traj = torch.stack(h_traj_new, dim=0).to(torch.float16)
            new_r = dict(r)
            new_r["h_goal_traj"] = new_h_traj
            new_r["_rederive_substrate"] = str(Path(args.substrate_ckpt).name)
            new_records.append(new_r)
        out_fp = out_dir / Path(fp).name
        torch.save({"records": new_records,
                      "substrate_ckpt": args.substrate_ckpt,
                      "use_evidence_layernorm": args.use_evidence_layernorm},
                     out_fp)
        print(f"  {fp}: {len(new_records)} records re-derived "
              f"(skipped {n_skip}) → {out_fp}", flush=True)

    print(f"[rederive] done → {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
