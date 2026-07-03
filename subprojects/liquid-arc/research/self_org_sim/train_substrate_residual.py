"""BC train substrate's broadcast-residual head through frozen GR00T action head.

Per turn:
  1. backbone(no_grad) → bb_features [B, seq, 2048]
  2. substrate.step(...) → residual [B, 2048] (per-call broadcast residual)
  3. bb_features_modified = bb_features + residual.unsqueeze(1) (broadcast over seq dim)
  4. action_head.forward(modified_bb_out, ah_inputs) → flow-matching loss vs expert chunk
  5. Backprop through head_residual only (substrate core + other heads frozen)

This is the FORCED-contribution mechanism (variant #9 BC, +12.9pp L=3 / +7.3pp L=5 hard).
Compared to vl-token append (optional, within-noise), residual rewrites every token's
content so action head's cross-attention is forced to use modified bb_features.
"""
from __future__ import annotations
import argparse
import sys
import time
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


def load_suite_episodes(suite_name, max_episodes, stride, action_horizon=16,
                          episode_offset=0):
    suite_short = suite_name.replace("libero_", "")
    sd = DATASET_ROOT / f"libero-{suite_short}-expert-v1"
    if not sd.exists():
        return [], None
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

    all_succ = [i for i in range(len(lengths)) if bool(success[i])]
    by_task: dict = {}
    for ep in all_succ:
        tid = int(task_indices[ep])
        by_task.setdefault(tid, []).append(ep)
    per_task = max(1, max_episodes // max(1, len(by_task)))
    succ_eps = []
    for tid in sorted(by_task.keys()):
        succ_eps.extend(by_task[tid][episode_offset:episode_offset + per_task])
    succ_eps = succ_eps[:max_episodes]

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
    return episodes, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--substrate_ckpt", required=True,
                   help="Proprio substrate ckpt (head_residual trained, all else frozen)")
    p.add_argument("--triples_data", default="/tmp/libero_jepa_mod_train.npz",
                   help="For substrate's z_goal lookup per task")
    p.add_argument("--output", default="/tmp/substrate_residual.pt")
    p.add_argument("--teacher_path",
                   default="/home/pokazge/Isaac-GR00T/checkpoints/GR00T-N1.7-LIBERO/libero_10")
    p.add_argument("--suites", default="libero_10")
    p.add_argument("--max_eps_per_suite", type=int, default=50)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--max_turns_per_ep", type=int, default=12)
    p.add_argument("--batch_episodes", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--ckpt_every", type=int, default=500)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--res_scale_train", type=float, default=1.0,
                   help="Scale on residual during training (1.0 = full)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[res-tok] device={device}, output={args.output}", flush=True)

    print(f"[res-tok] loading GR00T from {args.teacher_path}", flush=True)
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

    print(f"[res-tok] loading proprio substrate from {args.substrate_ckpt}", flush=True)
    ck = torch.load(args.substrate_ckpt, map_location=device, weights_only=False)
    sa = ck["args"]
    substrate = JEPA_LGT_Proprio(
        z_vl_dim=ck["z_vl_dim"], action_dim=ck["action_dim"],
        horizon=ck["horizon"], state_dim=ck["state_dim"],
        d=sa.get("d_substrate", 64), K=sa.get("K_belief", 4),
        n_tok_per_k=sa.get("n_tok_per_k", 1),
    ).to(device)
    try:
        substrate.load_state_dict(ck["substrate_state_dict"], strict=False)
        print(f"[res-tok] loaded substrate state_dict (strict=False)", flush=True)
    except Exception as e:
        print(f"[res-tok] WARN loading substrate: {e}", flush=True)

    for pp in substrate.parameters():
        pp.requires_grad = False
    for pp in substrate.head_residual.parameters():
        pp.requires_grad = True
    # JEPA-VL: also unfreeze the new in_lang linear + lang_gate (added later, so they're
    # randomly initialized in the loaded proprio ckpt — must train for signal to develop).
    for pp in substrate.in_lang.parameters():
        pp.requires_grad = True
    substrate.lang_gate.requires_grad = True
    # CROSS-ATTN: unfreeze new cross-attention heads (q_bb, k_sub, v_sub).
    # These produce per-token residual that bypasses the broadcast bottleneck.
    for mod in (substrate.head_q_bb, substrate.head_k_sub, substrate.head_v_sub):
        for pp in mod.parameters():
            pp.requires_grad = True
    n_trainable = sum(pp.numel() for pp in substrate.parameters() if pp.requires_grad)
    print(f"[res-tok] training head_residual + in_lang + lang_gate + xattn heads "
          f"({n_trainable:,} params)", flush=True)

    triples = np.load(args.triples_data, allow_pickle=True)
    z_goal_by_task = {}
    for i in range(len(triples["task_id"])):
        tid = int(triples["task_id"][i])
        if tid not in z_goal_by_task:
            z_goal_by_task[tid] = triples["z_goal"][i].copy()
    print(f"[res-tok] {len(z_goal_by_task)} z_goal entries cached", flush=True)

    print(f"[res-tok] loading expert episodes...", flush=True)
    all_eps = []
    for s in [x.strip() for x in args.suites.split(",") if x.strip()]:
        eps, _ = load_suite_episodes(s, args.max_eps_per_suite, args.stride)
        print(f"  {s}: {len(eps)} episodes", flush=True)
        for ep in eps:
            for turn in ep:
                turn["suite"] = s
        all_eps.extend(eps)
    print(f"[res-tok] total {len(all_eps)} episodes", flush=True)
    if not all_eps:
        print("[res-tok] no eps"); return

    z_goal_default = next(iter(z_goal_by_task.values()))

    opt_params = [pp for pp in substrate.parameters() if pp.requires_grad]
    opt = torch.optim.AdamW(opt_params, lr=args.lr, weight_decay=args.weight_decay)

    t_start = time.time()
    rolling = {"loss": [], "res_norm": []}
    for step in range(args.max_steps):
        ep_idxs = np.random.choice(len(all_eps), args.batch_episodes, replace=False)
        ep_losses = []
        ep_res_norms = []
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
                obs_dict = build_groot_obs(turn["img"], turn["wri"], turn["st"],
                                              turn["lang"])
                inputs = make_inputs(policy, obs_dict, device, bf16)
                ec = turn["expert_chunk"].astype(np.float32)
                full_action = np.zeros((1, ah_action_horizon, ah_action_dim),
                                          dtype=np.float32)
                full_action[0, :ec.shape[0], :ec.shape[1]] = ec[:ah_action_horizon]
                full_mask = np.zeros((1, ah_action_horizon, ah_action_dim),
                                        dtype=np.float32)
                full_mask[0, :ec.shape[0], :ec.shape[1]] = 1.0
                inputs["action"] = torch.from_numpy(full_action).to(device=device, dtype=bf16)
                inputs["action_mask"] = torch.from_numpy(full_mask).to(device=device, dtype=bf16)
                bb_inputs = BatchFeature(data=inputs)
                ah_inputs = BatchFeature(data=inputs)
                with torch.no_grad():
                    bb_out = backbone_module(bb_inputs)
                bb_feats = bb_out["backbone_features"]  # [1, seq, 2048] bf16

                z_t_pooled = bb_feats.mean(dim=1).to(torch.float32)
                # JEPA-VL: pool LANGUAGE tokens only via image_mask
                z_lang_pooled = z_t_pooled.clone()
                img_mask = getattr(bb_out, "image_mask", None)
                attn_mask = getattr(bb_out, "backbone_attention_mask", None)
                if img_mask is not None and attn_mask is not None:
                    lang_mask = ((img_mask == 0) & (attn_mask == 1))[0].float()  # [seq]
                    n_lang = float(lang_mask.sum().item())
                    if n_lang > 0:
                        z_lang_pooled = ((bb_feats[0].float() *
                                            lang_mask.unsqueeze(-1)).sum(dim=0)
                                          / n_lang).unsqueeze(0)
                z_goal_t = torch.from_numpy(z_goal_default).to(device).unsqueeze(0)
                state8_t = torch.from_numpy(turn["st"].astype(np.float32)).to(device).unsqueeze(0)
                chunk_t = torch.from_numpy(ec).to(device).unsqueeze(0).float()
                h_goal, _, aux, _ = substrate.step(
                    h_goal, z_t_pooled, z_goal_t, chunk_t, state8_t,
                    z_lang_t=z_lang_pooled,
                    bb_features=bb_feats)  # CROSS-ATTN: pass full sequence
                # PER-TOKEN residual from cross-attention (preserves seq structure)
                per_tok = aux["per_token_residual"]  # [1, seq, 2048] fp32
                if per_tok is None:
                    # Fallback to broadcast residual (shouldn't happen if bb_features passed)
                    residual = aux["residual"]
                    res_bf = residual.to(bb_feats.dtype).unsqueeze(1) * args.res_scale_train
                    bb_feats_mod = bb_feats + res_bf
                else:
                    res_bf = per_tok.to(bb_feats.dtype) * args.res_scale_train
                    bb_feats_mod = bb_feats + res_bf  # per-token addition
                bb_out["backbone_features"] = bb_feats_mod

                out = action_head.forward(bb_out, ah_inputs)
                loss_t = out["loss"] if isinstance(out, dict) else out.loss
                ep_loss = ep_loss + loss_t
                # Log per-token residual norm (avg over seq positions) if cross-attn
                if per_tok is not None:
                    ep_res_norms.append(float(per_tok.detach().norm(dim=-1).mean()))
                else:
                    ep_res_norms.append(float(aux["residual"].detach().norm(dim=-1).mean()))
            avg_loss = ep_loss / len(turns)
            ep_losses.append(avg_loss)
        batch_loss = torch.stack(ep_losses).mean()
        opt.zero_grad()
        batch_loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(opt_params, args.max_grad_norm)
        opt.step()

        rolling["loss"].append(float(batch_loss))
        rolling["res_norm"].append(float(np.mean(ep_res_norms)))
        for k in rolling: rolling[k] = rolling[k][-50:]
        if step % args.log_every == 0:
            print(f"step {step:>4}  loss={np.mean(rolling['loss']):.5f}  "
                  f"res_norm={np.mean(rolling['res_norm']):.3f}  "
                  f"wall={time.time()-t_start:.0f}s", flush=True)
        if args.ckpt_every > 0 and step > 0 and step % args.ckpt_every == 0:
            torch.save({
                "substrate_state_dict": substrate.state_dict(),
                "args": vars(args), "step": step,
                "z_vl_dim": ck["z_vl_dim"], "action_dim": ck["action_dim"],
                "horizon": ck["horizon"], "state_dim": ck["state_dim"],
                "dist_mean": ck["dist_mean"], "dist_std": ck["dist_std"],
                "sd_mean": ck["sd_mean"], "sd_std": ck["sd_std"],
            }, args.output.replace(".pt", f"_step{step}.pt"))

    torch.save({
        "substrate_state_dict": substrate.state_dict(),
        "args": vars(args), "step": args.max_steps,
        "z_vl_dim": ck["z_vl_dim"], "action_dim": ck["action_dim"],
        "horizon": ck["horizon"], "state_dim": ck["state_dim"],
        "dist_mean": ck["dist_mean"], "dist_std": ck["dist_std"],
        "sd_mean": ck["sd_mean"], "sd_std": ck["sd_std"],
    }, args.output)
    print(f"\n[res-tok] saved → {args.output}", flush=True)


if __name__ == "__main__":
    main()
