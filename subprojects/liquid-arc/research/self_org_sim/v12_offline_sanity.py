"""Quick offline check: does v12 ckpt predict close to its training teacher chunks?

Loads a few training samples from libero-10-expert-v1, runs v12 predict_chunk,
compares to teacher_chunk via per-step MSE. If MSE is small, training is
working — current eval failure is undertraining/distribution-shift, not bug.
"""
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from distill_groot_v12 import V12Policy
from liquid_arc_substrate_libero import make_v12_config


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "/tmp/distill_v12/step_007000.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = make_v12_config()
    for k, v in ckpt["config"].items():
        if hasattr(config, k):
            setattr(config, k, v)
    model = V12Policy(config, action_horizon=16, action_dim=7, state_dim=8,
                     head_d=256, head_layers=4, head_heads=4,
                     use_goal_img=False).to(device)
    sd = ckpt["policy"]
    own = model.state_dict()
    for k, v in sd.items():
        kk = k.replace("_orig_mod.", "")
        if kk in own and own[kk].shape == v.shape:
            own[kk].copy_(v)
    model.eval()

    # Load 10 training samples
    suite_dir = Path("/home/pokazge/datasets/libero-10-expert-v1")
    idx = np.load(suite_dir / "index.npz")
    n_total = int(idx["n_total"])
    img_size = int(idx["img_size"])
    imgs = np.memmap(suite_dir / "imgs.dat", dtype=np.uint8, mode="r",
                     shape=(n_total, img_size, img_size, 3))
    wrists = np.memmap(suite_dir / "wrists.dat", dtype=np.uint8, mode="r",
                       shape=(n_total, img_size, img_size, 3))
    states = np.memmap(suite_dir / "states.dat", dtype=np.float32, mode="r",
                       shape=(n_total, 8))
    lbl = np.load(suite_dir / "labels_index.npz")
    sample_idx = lbl["sample_idx"]
    n_samples = int(lbl["n_samples"])
    chunks = np.memmap(suite_dir / "teacher_chunks.dat", dtype=np.float32, mode="r",
                       shape=(n_samples, 16, 7))

    n_test = 5
    from PIL import Image
    print(f"Testing v12 ckpt={ckpt_path} on {n_test} training samples")
    for s in range(n_test):
        ep_i, t, _ = sample_idx[s * 100]
        global_idx = int(idx["episode_starts"][int(ep_i)]) + int(t)
        img_raw = np.array(imgs[global_idx])
        wri_raw = np.array(wrists[global_idx])
        st = np.array(states[global_idx])
        target_chunk = np.array(chunks[s * 100])

        # Resize to 224 and run model
        img224 = np.array(Image.fromarray(img_raw).resize((224, 224)), dtype=np.uint8)
        wri224 = np.array(Image.fromarray(wri_raw).resize((224, 224)), dtype=np.uint8)
        img_t = torch.from_numpy(img224).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        wri_t = torch.from_numpy(wri224).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        st_t = torch.from_numpy(st).to(device).float().unsqueeze(0)

        with torch.no_grad():
            out = model.encode(img_t, wri_t, st_t)
            cond = out["cond"]
            # Sample chunk via flow integration
            x = torch.randn(1, 16, 7, device=device)
            n_steps = 10
            dt = 1.0 / n_steps
            for i in range(n_steps):
                tt = torch.full((1,), i * dt, device=device)
                v = model.velocity(x, tt, cond)
                x = x + dt * v
            chunk_pred = x[0].cpu().numpy()
            # Gripper from sigmoid head
            gripper_logits = model.gripper_logits(cond)
            gripper_bin = (torch.sigmoid(gripper_logits[0]) > 0.5).float() * 2 - 1
            chunk_pred[:, -1] = gripper_bin.cpu().numpy()

        # MSE breakdown
        mse_xyz = np.mean((chunk_pred[:, :3] - target_chunk[:, :3]) ** 2)
        mse_rpy = np.mean((chunk_pred[:, 3:6] - target_chunk[:, 3:6]) ** 2)
        mse_grip = np.mean((chunk_pred[:, -1] - target_chunk[:, -1]) ** 2)
        total_mse = np.mean((chunk_pred - target_chunk) ** 2)
        print(f"sample {s}: total_mse={total_mse:.4f}  xyz={mse_xyz:.4f}  rpy={mse_rpy:.4f}  grip={mse_grip:.4f}")
        print(f"  target chunk[0]: {target_chunk[0]}")
        print(f"  pred   chunk[0]: {chunk_pred[0]}")
        print(f"  target gripper sequence: {target_chunk[:, -1]}")
        print(f"  pred   gripper sequence: {chunk_pred[:, -1]}")

if __name__ == "__main__":
    main()
