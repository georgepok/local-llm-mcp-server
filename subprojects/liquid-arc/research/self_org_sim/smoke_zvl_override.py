"""Smoke test for GR00T's z_vl override op.

Verifies:
1. get_action_with_state(obs) → chunk_A, z_vl
2. get_action_with_zvl_override(obs, zvl_residual=zeros[2048]) → chunk_B
3. chunk_A ≈ chunk_B (identity residual must not change behavior)
4. get_action_with_zvl_override(obs, zvl_residual=ones×0.1) → chunk_C
5. chunk_C ≠ chunk_A (non-zero residual must change behavior)
"""
import sys
from pathlib import Path
import pickle
import numpy as np
import zmq

CALVIN_ROOT = Path("/home/pokazge/calvin")
SELF_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CALVIN_ROOT / "calvin_env"))
sys.path.insert(0, str(SELF_DIR))

from rollout_calvin_zeroshot import (  # type: ignore
    calvin_obs_to_groot, make_env, load_episode_state,
)

env = make_env()
init = load_episode_state(553636)  # validation episode 0 start
env.reset(robot_obs=init["robot_obs"], scene_obs=init["scene_obs"])
settle = np.zeros(7, dtype=np.float32); settle[-1] = 1.0
obs, _, _, _ = env.step(settle)

groot_obs, _ = calvin_obs_to_groot(obs, "lift the red block from the table")

ctx = zmq.Context.instance()
sock = ctx.socket(zmq.REQ)
sock.setsockopt(zmq.RCVTIMEO, 60000)
sock.connect("tcp://localhost:5559")

# Sample N chunks per condition to characterize noise vs override-induced change
def sample_chunks(op_name, kwargs_extra, N):
    chunks = []
    for _ in range(N):
        req = {"op": op_name, "obs": groot_obs}
        req.update(kwargs_extra)
        sock.send(pickle.dumps(req))
        resp = pickle.loads(sock.recv())
        chunks.append(resp["chunk"])
    return np.stack(chunks)  # [N, 16, 7]

N = 5
print(f"Sampling {N} chunks per condition...")

# Baseline: normal get_action_with_state (stochastic from random noise init)
baseline = sample_chunks("get_action_with_state", {}, N)
print(f"baseline      : mean(chunk) [first3 dim] = {baseline.mean(axis=0)[0, :3]}")

# Get a z_vl for sizing
sock.send(pickle.dumps({"op": "get_action_with_state", "obs": groot_obs}))
z_vl = pickle.loads(sock.recv())["z_vl"]
print(f"z_vl mean={z_vl.mean():.4f} std={z_vl.std():.4f} norm={np.linalg.norm(z_vl):.2f}")

zero_residual = np.zeros_like(z_vl)
small_residual = np.ones_like(z_vl) * 0.5  # 0.5×sqrt(2048)≈22.6 norm
big_residual = np.ones_like(z_vl) * 5.0    # ~226 norm — should dominate

zeros = sample_chunks("get_action_with_zvl_override", {"zvl_residual": zero_residual}, N)
small = sample_chunks("get_action_with_zvl_override", {"zvl_residual": small_residual}, N)
big = sample_chunks("get_action_with_zvl_override", {"zvl_residual": big_residual}, N)

# Per-condition mean (stochastic noise averages out across N samples)
print()
print("Mean chunk[0, :3] across samples (averaging out flow-matching noise):")
print(f"  baseline     : {baseline.mean(axis=0)[0, :3]}")
print(f"  zero_override: {zeros.mean(axis=0)[0, :3]}")
print(f"  small_resid  : {small.mean(axis=0)[0, :3]}")
print(f"  big_resid    : {big.mean(axis=0)[0, :3]}")

# Across-condition difference (signal from override)
within = baseline.std(axis=0).mean()
zero_diff = np.abs(baseline.mean(axis=0) - zeros.mean(axis=0)).mean()
small_diff = np.abs(baseline.mean(axis=0) - small.mean(axis=0)).mean()
big_diff = np.abs(baseline.mean(axis=0) - big.mean(axis=0)).mean()

print()
print(f"within-condition std (noise floor):   {within:.4f}")
print(f"zero override - baseline diff:        {zero_diff:.4f}  (should be < noise floor)")
print(f"small (0.5) override - baseline diff: {small_diff:.4f}  (should exceed noise floor)")
print(f"big (5.0) override - baseline diff:   {big_diff:.4f}    (should clearly dominate)")

print()
print("VERDICTS:")
print(f"  Zero override matches baseline (within noise): {'PASS' if zero_diff < 2 * within else 'FAIL'}")
print(f"  Big override clearly shifts output:            {'PASS' if big_diff > 3 * within else 'FAIL'}")
print(f"  Override magnitude monotonically increases diff: {'PASS' if zero_diff < small_diff < big_diff else 'FAIL'}")
