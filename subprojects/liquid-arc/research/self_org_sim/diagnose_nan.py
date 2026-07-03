"""Diagnose source of NaN in text MT substrate val computation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from liquid_goal_tracker_proprio import JEPA_LGT_Proprio
from train_substrate_twoflow import load_records_with_inputs, forward_two_flow_no_grad


def main():
    device = torch.device("cuda")
    ck = torch.load("/home/pokazge/checkpoints/substrate_twoflow_text_mt.pt",
                    map_location="cpu", weights_only=False)
    model = JEPA_LGT_Proprio(
        z_vl_dim=ck["z_vl_dim"], action_dim=ck["action_dim"],
        horizon=ck["horizon"], state_dim=ck["state_dim"],
        d=64, K=4, n_tok_per_k=1,
    )
    model.load_state_dict(ck["substrate_state_dict"], strict=False)
    model.use_evidence_layernorm = True
    model.h_input_clamp = 50.0
    model = model.to(device).eval()

    n_total = sum(1 for _ in model.parameters())
    n_nan = sum(int(torch.isnan(p).any().item()) for p in model.parameters())
    print(f"params with NaN: {n_nan}/{n_total}")

    records = load_records_with_inputs(["/home/pokazge/data/text_traces_mt_rb.pt"])
    n_nan_recs = 0
    nan_ids = []
    for r in records:
        T = r["z_vl_traj"].shape[0]
        # Check input data for NaN/Inf
        if (torch.isnan(r["z_vl_traj"]).any() or torch.isnan(r["z_lang_traj"]).any()
            or torch.isnan(r["z_goal"]).any()):
            print(f"  INPUT NaN: sub_id={r['sub_id']}")
            continue
        h_slow, h_fast = forward_two_flow_no_grad(model, r, device, T - 1)
        if torch.isnan(h_slow).any() or torch.isnan(h_fast).any():
            n_nan_recs += 1
            nan_ids.append(r["sub_id"])
            if n_nan_recs <= 5:
                for t in range(T):
                    if torch.isnan(h_fast[t]).any() or torch.isnan(h_slow[t]).any():
                        print(f"  sub_id={r['sub_id']} T={T} → first NaN at t={t}")
                        break
    print(f"records producing NaN: {n_nan_recs}/{len(records)}")
    print(f"first 10 nan sub_ids: {nan_ids[:10]}")


if __name__ == "__main__":
    main()
