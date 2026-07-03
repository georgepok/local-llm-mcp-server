"""Compare v12 predictions across training checkpoints on a fixed batch
to see if predictions are converging toward good control or stuck.
"""
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from distill_groot_v12 import V12Policy
from liquid_arc_substrate_libero import make_v12_config


def load(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = make_v12_config()
    for k, v in ckpt["config"].items():
        if hasattr(config, k):
            setattr(config, k, v)
    model = V12Policy(config, action_horizon=16, action_dim=7, state_dim=8,
                     head_d=256, head_layers=4, head_heads=4, use_goal_img=False).to(device)
    sd = ckpt["policy"]
    own = model.state_dict()
    for k, v in sd.items():
        kk = k.replace("_orig_mod.", "")
        if kk in own and own[kk].shape == v.shape:
            own[kk].copy_(v)
    model.eval()
    return model, ckpt.get("step", 0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load 10 fixed training samples
    from PIL import Image
    suite_dir = Path("/home/pokazge/datasets/libero-10-expert-v1")
    idx = np.load(suite_dir / "index.npz")
    n_total = int(idx["n_total"]); img_size = int(idx["img_size"])
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

    test_idxs = np.linspace(0, n_samples - 1, 10, dtype=int)
    batch_img, batch_wri, batch_st, batch_target = [], [], [], []
    for s in test_idxs:
        ep_i, t, _ = sample_idx[s]
        g = int(idx["episode_starts"][int(ep_i)]) + int(t)
        i224 = np.array(Image.fromarray(np.array(imgs[g])).resize((224, 224)), dtype=np.uint8)
        w224 = np.array(Image.fromarray(np.array(wrists[g])).resize((224, 224)), dtype=np.uint8)
        batch_img.append(i224); batch_wri.append(w224)
        batch_st.append(np.array(states[g])); batch_target.append(np.array(chunks[s]))
    batch_img = torch.from_numpy(np.stack(batch_img)).to(device).float().permute(0, 3, 1, 2) / 255.0
    batch_wri = torch.from_numpy(np.stack(batch_wri)).to(device).float().permute(0, 3, 1, 2) / 255.0
    batch_st = torch.from_numpy(np.stack(batch_st)).to(device).float()
    batch_target = np.stack(batch_target)

    @torch.no_grad()
    def predict(model):
        torch.manual_seed(42)
        out = model.encode(batch_img, batch_wri, batch_st)
        cond = out["cond"]
        x = torch.randn(10, 16, 7, device=device)
        n_steps = 10
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t_val = torch.full((10,), i * dt, device=device)
            v = model.velocity(x, t_val, cond)
            x = x + dt * v
        return x.cpu().numpy()

    print("\nstep | total_mse | xyz_mse | grip_mean | first_norm | xyz_corr_v_target")
    for ckpt_name in ["step_001000.pt", "step_002000.pt", "step_004000.pt",
                       "step_006000.pt", "step_008000.pt"]:
        ckpt_path = Path("/tmp/distill_v12") / ckpt_name
        model, step = load(ckpt_path, device)
        pred = predict(model)
        total_mse = float(np.mean((pred - batch_target) ** 2))
        xyz_mse = float(np.mean((pred[:, :, :3] - batch_target[:, :, :3]) ** 2))
        grip_mean = float(pred[:, :, -1].mean())
        first_norm = float(np.mean(np.linalg.norm(pred[:, 0, :6], axis=-1)))
        # Pearson correlation between pred xyz flat and target xyz flat
        p_flat = pred[:, :, :3].flatten()
        t_flat = batch_target[:, :, :3].flatten()
        corr = float(np.corrcoef(p_flat, t_flat)[0, 1])
        print(f"  {step:5d} | {total_mse:.4f}    | {xyz_mse:.4f}  | {grip_mean:+.3f}    | "
              f"{first_norm:.3f}      | {corr:.3f}")


if __name__ == "__main__":
    main()
