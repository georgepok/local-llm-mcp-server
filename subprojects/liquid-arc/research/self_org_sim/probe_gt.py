"""Diagnostic probe for SubstrateGoalTracker server.

Simulates one episode by feeding the server a sequence of OBSERVATIONS from
the expert dataset, prints per-call GT diagnostics (P(open), override count,
metric_cv). Tells us if GT's belief evolves sensibly across an episode.
"""
import sys
import pickle
import zmq
import numpy as np
from pathlib import Path

SUITE_DIR = Path("/home/pokazge/datasets/libero-10-expert-v1")

def main():
    print(f"[probe] loading expert episode 0 from {SUITE_DIR}")
    idx = np.load(SUITE_DIR / "index.npz")
    starts = idx["episode_starts"]
    lengths = idx["episode_lengths"]
    n_total = int(idx["n_total"])
    img_size = int(idx["img_size"])
    labels = np.load(SUITE_DIR / "labels_index.npz")
    sample_idx = labels["sample_idx"]
    n_samples = int(labels["n_samples"])
    chunks = np.memmap(SUITE_DIR / "teacher_chunks.dat", dtype=np.float32, mode="r",
                       shape=(n_samples, 16, 7))
    imgs = np.memmap(SUITE_DIR / "imgs.dat", dtype=np.uint8, mode="r",
                     shape=(n_total, img_size, img_size, 3))
    wrists = np.memmap(SUITE_DIR / "wrists.dat", dtype=np.uint8, mode="r",
                       shape=(n_total, img_size, img_size, 3))
    states = np.memmap(SUITE_DIR / "states.dat", dtype=np.float32, mode="r",
                       shape=(n_total, 8))

    # Episode 0 samples in order
    ep_mask = sample_idx[:, 0] == 0
    ep_samples = np.where(ep_mask)[0]
    order = np.argsort(sample_idx[ep_samples, 1])
    ep_samples_sorted = ep_samples[order]
    # Subsample at stride 8 (matches exec_horizon)
    turn_samples = ep_samples_sorted[::8][:30]
    print(f"[probe] episode 0 has {len(ep_samples_sorted)} samples, using {len(turn_samples)} turns")

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 30000)
    sock.connect("tcp://localhost:7777")

    # Reset episode
    sock.send(pickle.dumps({"cmd": "episode_reset"}))
    resp = pickle.loads(sock.recv())
    print(f"[probe] episode_reset: {resp}")

    print(f"\n{'turn':>5} {'t_in_ep':>7} {'expert_grip[0]':>14} {'gt_p_open':>10} {'gt_overridden':>14}  gripper_logit_summary")
    for turn, s_local in enumerate(turn_samples):
        ep, t, _ = sample_idx[s_local]
        global_idx = int(starts[ep]) + int(t)
        expert_grip = float(chunks[s_local, 0, -1])

        sock.send(pickle.dumps({
            "cmd": "predict_chunk",
            "img_raw": np.array(imgs[global_idx]),
            "wrist_raw": np.array(wrists[global_idx]),
            "state8": np.array(states[global_idx]),
            "n_steps": 10,
        }))
        resp = pickle.loads(sock.recv())
        if not resp.get("ok"):
            print(f"  ERROR: {resp.get('error')}"); break

        chunk = resp["chunk"]
        out_grip0 = chunk[0, -1]
        print(f"{turn:>5} {int(t):>7} {expert_grip:>+14.3f}    "
              f"{resp.get('gt_p_open_mean', 0):>+8.3f}    {resp.get('gt_overridden', 0):>10}  "
              f"out_grip[0]={out_grip0:+.3f}  cv={resp.get('gt_metric_cv', 0):.3f}")

    print("\n--- summary ---")
    print("expert_grip starts at -1 (open) and flips to +1 (close) around t=43-59")
    print("gt_p_open should be HIGH (~1.0) early then DROP toward 0 at grip transition")
    print("gt_overridden should be NONZERO when model says close but GT says open")

if __name__ == "__main__":
    main()
