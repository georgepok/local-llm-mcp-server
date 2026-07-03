"""Convert text traces (z_t_traj, z_lang_traj, z_goal, succ) into robotics-shape
records consumable by train_substrate_twoflow.py without modifying the trainer.

Steps:
  1. Load /home/pokazge/data/text_traces_gsm8k.pt (dict with 'records', 'enc_dim')
  2. For each record: rename z_t_traj→z_vl_traj, add zero state8_traj/chunk_traj,
     add dummy h_goal_traj for shape check.
  3. Save as /home/pokazge/data/text_traces_gsm8k_rb.pt with {"records": [...]}.
  4. Build fresh text-substrate starter ckpt with z_vl_dim=enc_dim and save to
     /home/pokazge/checkpoints/substrate_text_starter.pt.
"""
import argparse
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from liquid_goal_tracker_proprio import JEPA_LGT_Proprio


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text_traj", required=True)
    p.add_argument("--out_traj", required=True)
    p.add_argument("--out_starter", required=True)
    p.add_argument("--d_substrate", type=int, default=64)
    p.add_argument("--K_belief", type=int, default=4)
    p.add_argument("--action_dim", type=int, default=7)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--state_dim", type=int, default=8)
    args = p.parse_args()

    text_pack = torch.load(args.text_traj, map_location="cpu", weights_only=False)
    text_records = text_pack["records"]
    enc_dim = int(text_pack["enc_dim"])
    print(f"[conv] loaded {len(text_records)} text records, enc_dim={enc_dim}", flush=True)

    K = args.K_belief
    d = args.d_substrate

    converted = []
    for r in text_records:
        T = int(r["T"])
        rec = {
            "z_vl_traj":   r["z_t_traj"].float(),                 # [T, enc_dim]
            "z_lang_traj": r["z_lang_traj"].float(),              # [T, enc_dim]
            "z_goal":      r["z_goal"].float(),                   # [enc_dim]
            "state8_traj": torch.zeros(T, args.state_dim),
            "chunk_traj":  torch.zeros(T, args.horizon, args.action_dim),
            "h_goal_traj": torch.zeros(T, K, d),                  # dummy for shape check
            "succ":        int(r["succ"]),
            "sub_id":      int(r["sub_id"]),
        }
        converted.append(rec)
    print(f"[conv] converted; example T={converted[0]['z_vl_traj'].shape[0]}, "
          f"z_vl_dim={converted[0]['z_vl_traj'].shape[1]}", flush=True)

    # Save robotics-format file
    Path(args.out_traj).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"records": converted}, args.out_traj)
    print(f"[conv] saved trajectory file → {args.out_traj}", flush=True)

    # Build text-substrate starter
    sub = JEPA_LGT_Proprio(
        z_vl_dim=enc_dim, action_dim=args.action_dim,
        horizon=args.horizon, state_dim=args.state_dim,
        d=d, K=K, n_tok_per_k=1,
    )
    sub.use_evidence_layernorm = True
    sub.h_input_clamp = 50.0
    starter_args = {
        "d_substrate": d, "K_belief": K, "n_tok_per_k": 1,
    }
    Path(args.out_starter).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "substrate_state_dict": sub.state_dict(),
        "args": starter_args,
        "z_vl_dim": enc_dim,
        "action_dim": args.action_dim,
        "horizon": args.horizon,
        "state_dim": args.state_dim,
        "dist_mean": 0.0, "dist_std": 1.0,
        "sd_mean": 0.0, "sd_std": 1.0,
    }, args.out_starter)
    print(f"[conv] saved text-substrate starter → {args.out_starter}", flush=True)
    print(f"[conv] === ALL_DONE ===", flush=True)


if __name__ == "__main__":
    main()
