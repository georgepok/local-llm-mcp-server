"""BC train LiquidGoalTracker through GR00T's REAL action head (flow-matching loss).

Per turn:
  obs = build_groot_obs(img, wrist, state, lang)
  processed = policy.processor([{type=EPISODE_STEP, content=vla_step_data}])
  inputs = policy.collate_fn([processed])["inputs"]
  inputs = _rec_to_dtype(inputs, bf16); move to device
  inputs["action"] = expert_chunk_padded ; inputs["action_mask"] = mask
  bb_inputs = BatchFeature(inputs); ah_inputs = BatchFeature(inputs)
  with no_grad: bb_out = policy.model.backbone(bb_inputs)
  z = bb_out.backbone_features.mean(1).float()
  h_goal, residual, _ = substrate.step(h_goal, z)
  bb_out.backbone_features = bb_out.backbone_features + residual.to(bf16).unsqueeze(1)
  loss = policy.model.action_head.forward(bb_out, ah_inputs).loss  # flow-matching MSE
  loss.backward()

GR00T frozen; only substrate trains.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

_LA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_LA_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from liquid_goal_tracker import LiquidGoalTracker  # type: ignore

torch.set_float32_matmul_precision("high")

DATASET_ROOT = Path("/home/pokazge/datasets")


def build_groot_obs(img_256, wrist_256, st8, task):
    state_slots = {"x": (0, 1), "y": (1, 2), "z": (2, 3),
                   "roll": (3, 4), "pitch": (4, 5), "yaw": (5, 6),
                   "gripper": (6, 8)}
    obs = {"video": {}, "state": {}, "language": {}}
    obs["video"]["image"] = img_256[None, None, ...]
    obs["video"]["wrist_image"] = wrist_256[None, None, ...]
    for k, (lo, hi) in state_slots.items():
        obs["state"][k] = st8[lo:hi].astype(np.float32)[None, None, :]
    obs["language"]["annotation.human.action.task_description"] = [[task]]
    return obs


def make_inputs(policy, obs_dict, device, bf16):
    """Return tensor dict ready for backbone + action_head (matches `collate_fn(...)["inputs"]`).

    Floats cast to bf16; everything moved to device.
    """
    from gr00t.data.types import MessageType  # type: ignore
    from gr00t.policy.gr00t_policy import _rec_to_dtype  # type: ignore
    unb = policy._unbatch_observation(obs_dict)
    processed = []
    for o in unb:
        vla = policy._to_vla_step_data(o)
        messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla}]
        processed.append(policy.processor(messages))
    collated = policy.collate_fn(processed)
    inputs = collated["inputs"]
    inputs = _rec_to_dtype(inputs, dtype=bf16)
    moved = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    return moved


def load_suite_episodes(suite_name, max_episodes, stride, action_horizon=16):
    suite_short = suite_name.replace("libero_", "")
    sd = DATASET_ROOT / f"libero-{suite_short}-expert-v1"
    if not sd.exists():
        return []
    idx = np.load(sd / "index.npz")
    starts = idx["episode_starts"]; lengths = idx["episode_lengths"]
    task_indices = idx["task_indices"]
    success = idx.get("success_per_episode", np.ones(len(lengths), dtype=bool))
    n_total = int(idx["n_total"]); img_size = int(idx["img_size"])
    imgs = np.memmap(sd / "imgs.dat", dtype=np.uint8, mode="r",
                     shape=(n_total, img_size, img_size, 3))
    wrists = np.memmap(sd / "wrists.dat", dtype=np.uint8, mode="r",
                       shape=(n_total, img_size, img_size, 3))
    states = np.memmap(sd / "states.dat", dtype=np.float32, mode="r",
                       shape=(n_total, 8))
    task_lang = {}
    lang_path = sd / "task_languages.json"
    if lang_path.exists():
        import json as _json
        task_lang = _json.loads(lang_path.read_text())

    labels = np.load(sd / "labels_index.npz")
    sample_idx = labels["sample_idx"]
    n_samples = int(labels["n_samples"])
    chunks_mm = np.memmap(sd / "teacher_chunks.dat", dtype=np.float32, mode="r",
                          shape=(n_samples, action_horizon, 7))
    lookup = {(int(s[0]), int(s[1])): i for i, s in enumerate(sample_idx)}

    succ_eps = [i for i in range(len(lengths)) if bool(success[i])][:max_episodes]
    episodes = []
    for ep in succ_eps:
        ep_start = int(starts[ep]); ep_len = int(lengths[ep])
        task_id = int(task_indices[ep])
        lang = (task_lang.get(str(task_id)) or task_lang.get(task_id) or "do the task")
        turns = []
        for t in range(0, ep_len - action_horizon, stride):
            key = (ep, t)
            if key not in lookup:
                continue
            gi = ep_start + t
            img = np.array(imgs[gi])[::-1, ::-1].copy()
            wri = np.array(wrists[gi])[::-1, ::-1].copy()
            st = np.array(states[gi])
            chunk = np.array(chunks_mm[lookup[key]])
            turns.append({"img": img, "wri": wri, "st": st, "lang": lang,
                          "expert_chunk": chunk})
        if len(turns) >= 2:
            episodes.append(turns)
    return episodes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suites", default="libero_10,libero_spatial,libero_object,libero_goal")
    p.add_argument("--max_eps_per_suite", type=int, default=20)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--teacher_path",
                   default="/home/pokazge/Isaac-GR00T/checkpoints/GR00T-N1.7-LIBERO/libero_10")
    p.add_argument("--output", default="/tmp/lgt_libero_bc.pt")
    p.add_argument("--batch_episodes", type=int, default=1)
    p.add_argument("--max_turns_per_ep", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--ckpt_every", type=int, default=100)
    p.add_argument("--d_substrate", type=int, default=64)
    p.add_argument("--K_belief", type=int, default=4)
    p.add_argument("--out_scale", type=float, default=0.2)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[bc] device={device}, output={args.output}", flush=True)

    print(f"[bc] loading GR00T from {args.teacher_path}", flush=True)
    from gr00t.data.embodiment_tags import EmbodimentTag  # type: ignore
    from gr00t.policy.gr00t_policy import Gr00tPolicy  # type: ignore
    from transformers.feature_extraction_utils import BatchFeature  # type: ignore
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.LIBERO_PANDA,
        model_path=str(args.teacher_path),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    for pp in policy.model.parameters():
        pp.requires_grad = False
    policy.model.eval()

    model = policy.model
    backbone_module = model.backbone
    action_head = model.action_head
    ah_action_dim = action_head.action_dim
    ah_action_horizon = action_head.action_horizon
    bf16 = torch.bfloat16
    print(f"[bc] action_head.action_dim={ah_action_dim} action_horizon={ah_action_horizon}",
          flush=True)

    print(f"[bc] loading expert episodes...", flush=True)
    all_eps = []
    for s in [s.strip() for s in args.suites.split(",") if s.strip()]:
        eps = load_suite_episodes(s, args.max_eps_per_suite, args.stride)
        print(f"  {s}: {len(eps)} episodes", flush=True)
        all_eps.extend(eps)
    print(f"[bc] total {len(all_eps)} episodes", flush=True)
    assert len(all_eps) > 0, "no expert episodes found"

    substrate = LiquidGoalTracker(
        z_vl_dim=2048, d=args.d_substrate, K=args.K_belief, out_scale=args.out_scale,
    ).to(device)
    n_params = sum(p.numel() for p in substrate.parameters())
    print(f"[bc] substrate params: {n_params:,}, out_scale={args.out_scale}", flush=True)
    opt = torch.optim.AdamW(substrate.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def add_action(inputs, expert_chunk_np):
        ec = expert_chunk_np.astype(np.float32)
        full_action = np.zeros((1, ah_action_horizon, ah_action_dim), dtype=np.float32)
        full_action[0, :ec.shape[0], :ec.shape[1]] = ec[:ah_action_horizon]
        full_mask = np.zeros((1, ah_action_horizon, ah_action_dim), dtype=np.float32)
        full_mask[0, :ec.shape[0], :ec.shape[1]] = 1.0
        inputs["action"] = torch.from_numpy(full_action).to(device=device, dtype=bf16)
        inputs["action_mask"] = torch.from_numpy(full_mask).to(device=device, dtype=bf16)
        return inputs

    t_start = time.time()
    rolling = {"loss": [], "res_norm": [], "cv": []}
    last_turns_for_null = []
    for step in range(args.max_steps):
        ep_idxs = np.random.choice(len(all_eps), args.batch_episodes, replace=False)
        ep_losses = []
        ep_res_norms = []
        ep_cvs = []
        for ei in ep_idxs:
            ep = all_eps[ei]
            T = min(len(ep), args.max_turns_per_ep)
            if len(ep) > T:
                t0 = np.random.randint(0, len(ep) - T + 1)
                turns = ep[t0:t0 + T]
            else:
                turns = ep
            h_goal = substrate.init_state(1, device)
            ep_loss = torch.tensor(0.0, device=device)
            for turn in turns:
                obs_dict = build_groot_obs(turn["img"], turn["wri"], turn["st"], turn["lang"])
                inputs = make_inputs(policy, obs_dict, device, bf16)
                inputs = add_action(inputs, turn["expert_chunk"])
                bb_inputs = BatchFeature(data=inputs)
                ah_inputs = BatchFeature(data=inputs)

                with torch.no_grad():
                    bb_out = backbone_module(bb_inputs)
                bb_feats = bb_out["backbone_features"]  # [1, seq, 2048] bf16
                z_vl_pooled = bb_feats.mean(dim=1).to(torch.float32)
                h_goal, residual, info = substrate.step(h_goal, z_vl_pooled)
                residual_bf = residual.to(bb_feats.dtype)
                bb_out["backbone_features"] = bb_feats + residual_bf.unsqueeze(1)

                out = action_head.forward(bb_out, ah_inputs)
                loss_t = out["loss"] if isinstance(out, dict) else out.loss
                ep_loss = ep_loss + loss_t
                ep_res_norms.append(float(residual.detach().norm(dim=-1).mean()))
                cv_val = info["metric_cv"]
                if isinstance(cv_val, torch.Tensor):
                    cv_val = cv_val.detach()
                ep_cvs.append(float(cv_val))
            avg_loss = ep_loss / len(turns)
            ep_losses.append(avg_loss)
            last_turns_for_null = turns
        batch_loss = torch.stack(ep_losses).mean()
        opt.zero_grad()
        batch_loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(substrate.parameters(), args.max_grad_norm)
        opt.step()

        rolling["loss"].append(float(batch_loss))
        rolling["res_norm"].append(float(np.mean(ep_res_norms)))
        rolling["cv"].append(float(np.mean(ep_cvs)))
        for k in rolling:
            rolling[k] = rolling[k][-50:]

        if step % args.log_every == 0:
            with torch.no_grad():
                null_loss = 0.0
                null_count = 0
                last_turns = last_turns_for_null[: min(4, len(last_turns_for_null))]
                for turn in last_turns:
                    obs_dict = build_groot_obs(turn["img"], turn["wri"], turn["st"], turn["lang"])
                    inputs = make_inputs(policy, obs_dict, device, bf16)
                    inputs = add_action(inputs, turn["expert_chunk"])
                    bi = BatchFeature(data=inputs)
                    ai = BatchFeature(data=inputs)
                    bbo = backbone_module(bi)
                    nl = action_head.forward(bbo, ai)
                    null_loss += float(nl["loss"] if isinstance(nl, dict) else nl.loss)
                    null_count += 1
                null_loss = null_loss / max(null_count, 1)
            wall = time.time() - t_start
            print(f"step {step:>4}  bc_loss={np.mean(rolling['loss']):.5f}  "
                  f"null_loss={null_loss:.5f}  "
                  f"res_norm={np.mean(rolling['res_norm']):.3f}  "
                  f"cv={np.mean(rolling['cv']):.3f}  wall={wall:.0f}s", flush=True)

        if args.ckpt_every > 0 and step > 0 and step % args.ckpt_every == 0:
            torch.save({
                "substrate_state_dict": substrate.state_dict(),
                "args": vars(args), "z_vl_dim": 2048, "step": step,
            }, args.output.replace(".pt", f"_step{step}.pt"))

    torch.save({
        "substrate_state_dict": substrate.state_dict(),
        "args": vars(args), "z_vl_dim": 2048, "step": args.max_steps,
    }, args.output)
    print(f"\n[bc] saved → {args.output}", flush=True)


if __name__ == "__main__":
    main()
