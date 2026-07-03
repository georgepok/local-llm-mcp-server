"""Smoke test: instantiate CALVIN env, step a few zero actions, verify obs structure.

Per prior session: env_cfg needs scene_cfg.data_path -> URDF dir, use_egl=True,
delete tactile camera (uses deprecated np.float). Env returns:
  obs["rgb_obs"]["rgb_static"]:   [200,200,3] uint8
  obs["rgb_obs"]["rgb_gripper"]:  [84,84,3] uint8
  obs["depth_obs"]:               dict of depth maps
  obs["robot_obs"]:               [15,] float (eef xyz/rpy + gripper width + arm joints)
  obs["scene_obs"]:               [24,] float (object positions)
Action is 7-dim [dx, dy, dz, droll, dpitch, dyaw, gripper] in relative_action mode.
"""
import sys
from pathlib import Path
import numpy as np

CALVIN_ROOT = Path("/home/pokazge/calvin")
DATASET = CALVIN_ROOT / "dataset" / "calvin_debug_dataset"

sys.path.insert(0, str(CALVIN_ROOT / "calvin_env"))

import hydra
from omegaconf import OmegaConf

print(f"[smoke] dataset: {DATASET} exists={DATASET.exists()}")

# Find hydra config in the dataset
hydra_cfg = DATASET / "validation" / ".hydra" / "merged_config.yaml"
if not hydra_cfg.exists():
    hydra_cfg = DATASET / "training" / ".hydra" / "merged_config.yaml"
print(f"[smoke] hydra_cfg: {hydra_cfg} exists={hydra_cfg.exists()}")

cfg = OmegaConf.load(hydra_cfg)
env_cfg = cfg.env

# data_path must point to URDF dir (where calvin_table_D/ lives)
env_cfg["scene_cfg"]["data_path"] = str(CALVIN_ROOT / "calvin_env" / "data")
env_cfg["use_egl"] = True   # GB10 hardware GL (no X server)
env_cfg["show_gui"] = False
# Drop tactile camera (np.float deprecated)
if "cameras" in env_cfg and "tactile" in env_cfg["cameras"]:
    del env_cfg["cameras"]["tactile"]
    print("[smoke] disabled tactile camera")

env = hydra.utils.instantiate(env_cfg)
print(f"[smoke] env instantiated: {type(env).__name__}")

obs = env.reset()
print(f"[smoke] reset OK; obs type={type(obs).__name__}")
if isinstance(obs, dict):
    for k, v in obs.items():
        if hasattr(v, "shape"):
            print(f"  {k}: shape={v.shape} dtype={v.dtype}")
        elif isinstance(v, dict):
            print(f"  {k}: dict")
            for kk, vv in v.items():
                if hasattr(vv, "shape"):
                    print(f"    {k}/{kk}: shape={vv.shape} dtype={vv.dtype}")

# 5 zero-action env steps
for i in range(5):
    action = np.zeros(7, dtype=np.float32)
    action[-1] = 1.0  # gripper open
    result = env.step(action)
    if isinstance(result, tuple) and len(result) >= 1:
        o = result[0]
        if isinstance(o, dict) and "robot_obs" in o:
            rob = o["robot_obs"]
            print(f"  step {i}: robot_obs[0:3] (eef xyz)={rob[:3]}, gripper={rob[6]:.3f}")

print("[smoke] OK — env is functional")
