"""Server with v10-DEMO motor + SubstrateGoalTracker as persistent goal observer.

Per-episode lifecycle on server:
  init / episode_reset → h_goal = goal_tracker.init_state()
  predict_and_retrieve → v10 predicts chunk; goal_tracker.step() updates h_goal
                          and outputs gripper override; apply override; return chunk

The goal tracker's h_goal state is MAINTAINED across calls within an episode.
On episode_reset it's reinitialized from the learned prior.

Compatible with the existing rollout client (same RPC interface).
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import zmq

_LA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_LA_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from distill_groot_flow import LiquidFlowPolicy  # type: ignore
from retrieval import RetrievalBank  # type: ignore
from substrate_goal_tracker import SubstrateGoalTracker  # type: ignore

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

torch.set_float32_matmul_precision("high")


class GoalTrackerServer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[gt_server] device={self.device}")

        # === Load v10-DEMO motor module ===
        print(f"[gt_server] loading motor ckpt {args.student_ckpt}")
        ckpt = torch.load(args.student_ckpt, map_location=self.device, weights_only=False)
        sa = ckpt["args"]
        self.sa = sa
        self.target_size = sa["img_size"]
        halt_mode = "learned" if sa["policy"] == "liquid_halt" else "none"
        self.model = LiquidFlowPolicy(
            state_dim=8, action_dim=7, action_horizon=sa["action_horizon"],
            d=sa["d"], d_vis=sa["d"], img_size=sa["img_size"],
            k_max=sa["k"], halt_mode=halt_mode,
            min_steps=sa["halting_min_steps"],
            n_tasks=sa["n_tasks"], d_task=sa["d_task"],
            head_d=sa["head_d"], head_layers=sa["head_layers"],
            head_heads=sa["head_heads"],
            n_task_heads=sa.get("n_task_heads", 0),
            z_groot_dim=sa.get("z_groot_dim", 0),
            gated_mixture=sa.get("gated_mixture", False),
            z_channel_dims=sa.get("z_channel_dims", None),
            query_bank=sa.get("use_query_bank", False),
        ).to(self.device)
        sd = ckpt["policy"]
        own = self.model.state_dict()
        loaded = 0
        for k, v in sd.items():
            kk = k.replace("_orig_mod.", "")
            if kk in own and own[kk].shape == v.shape:
                own[kk].copy_(v); loaded += 1
        print(f"[gt_server] motor loaded {loaded}/{len(own)} tensors")
        self.model.eval()
        for pp in self.model.parameters():
            pp.requires_grad = False

        # === Load SubstrateGoalTracker ===
        print(f"[gt_server] loading goal tracker {args.goal_tracker}")
        gt_ckpt = torch.load(args.goal_tracker, map_location=self.device, weights_only=False)
        d_obs = gt_ckpt.get("d_obs", sa["d"])
        K_belief = gt_ckpt.get("K_belief", 8)
        d_substrate = gt_ckpt.get("d_substrate", 128)
        self.gt = SubstrateGoalTracker(
            d_obs=d_obs, d_state=8, d_chunk=16 * 7,
            d=d_substrate, K=K_belief, action_horizon=16,
        ).to(self.device)
        self.gt.load_state_dict(gt_ckpt["state_dict"])
        self.gt.eval()
        for pp in self.gt.parameters():
            pp.requires_grad = False
        print(f"[gt_server] goal tracker ready: K={K_belief}, d={d_substrate}")

        # Optional retrieval (kept for compat with eval wrapper that calls set_retrieval_filter)
        self.retrieval_bank = None
        if args.memory_bank:
            self.retrieval_bank = RetrievalBank(
                bank_path=args.memory_bank, device=self.device,
                top_k=args.retrieve_top_k, alpha_base=args.retrieve_alpha,
                adaptive_alpha=False,
                filter_suite=args.retrieve_filter_suite_default or None,
                success_only=False,
                softmax_temperature=8.0,
            )

        # Image preprocessing
        self._mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(self.device)
        self._std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(self.device)

        # === Per-episode state ===
        self.h_goal = None
        self.prev_chunk = None
        self.threshold = args.gt_threshold
        self.gt_apply_to_positions = args.gt_apply_positions

        print(f"[gt_server] ready (threshold={self.threshold}, "
              f"apply_positions={self.gt_apply_to_positions})")

    def _img_to_tensor(self, img_uint8):
        x = torch.from_numpy(img_uint8).to(self.device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        if x.shape[-1] != self.target_size:
            x = F.interpolate(x, size=(self.target_size, self.target_size),
                              mode="bilinear", align_corners=False)
        x = (x - self._mean) / self._std
        return x

    @torch.no_grad()
    def _v10_sample_chunk(self, img_t, wri_t, st_t, n_steps=10):
        chunk = self.model.sample(
            img_t, wri_t, st_t, task_id=None, n_steps=n_steps,
            z_groot=None, z_bank=None, delta_bank=None, goal_img=None,
        )
        return chunk[0].cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def _v10_cond(self, img_t, wri_t, st_t):
        cond, _ = self.model.encoder(img_t, wri_t, st_t)
        return cond  # [B, d_obs]

    def cmd_init(self, kwargs):
        self.h_goal = self.gt.init_state(1, self.device)
        self.prev_chunk = None
        return {"ok": True, "info": {
            "d": int(self.sa["d"]), "version": "gt",
            "img_size": int(self.target_size),
            "action_horizon": int(self.sa["action_horizon"]),
        }}

    def cmd_episode_reset(self, kwargs):
        # Critical: reset goal-belief state at episode boundary
        self.h_goal = self.gt.init_state(1, self.device)
        self.prev_chunk = None
        return {"ok": True}

    @torch.no_grad()
    def cmd_predict_chunk(self, kwargs):
        img_raw = kwargs["img_raw"]
        wrist_raw = kwargs["wrist_raw"]
        state8 = np.asarray(kwargs["state8"], dtype=np.float32)

        img_t = self._img_to_tensor(img_raw)
        wri_t = self._img_to_tensor(wrist_raw)
        st_t = torch.from_numpy(state8).to(self.device).unsqueeze(0)

        # 1. v10 motor predicts chunk (the "GR00T transformer shadow" role)
        chunk_np = self._v10_sample_chunk(img_t, wri_t, st_t,
                                          n_steps=int(kwargs.get("n_steps", 10)))
        chunk_t = torch.from_numpy(chunk_np).to(self.device).unsqueeze(0)  # [1, 16, 7]

        # 2. Compute obs features from v10 encoder
        cond = self._v10_cond(img_t, wri_t, st_t)  # [1, d_obs]

        # 3. Goal tracker updates persistent h_goal and outputs gripper override
        if self.h_goal is None:
            self.h_goal = self.gt.init_state(1, self.device)
        h_goal_new, gripper_logits, info = self.gt.step(
            self.h_goal, cond, st_t, chunk_t,
            prev_groot_chunk=self.prev_chunk,
        )
        self.h_goal = h_goal_new                  # persist for next call
        self.prev_chunk = chunk_t                 # for next call's chunk_diff

        # 4. Apply gripper override: when goal-belief says "open" but model says "close"
        sub_p_open = torch.sigmoid(gripper_logits)[0].cpu().numpy()  # [16]
        chunk_out = chunk_np.copy()
        overridden = 0
        n_apply = min(self.gt_apply_to_positions, chunk_out.shape[0])
        for k in range(n_apply):
            if sub_p_open[k] > self.threshold and chunk_out[k, -1] > 0.1:
                chunk_out[k, -1] = -1.0
                overridden += 1

        return {
            "ok": True, "chunk": chunk_out,
            "gt_overridden": overridden,
            "gt_p_open_mean": float(sub_p_open.mean()),
            "gt_metric_cv": float(info["metric_cv"]),
        }

    @torch.no_grad()
    def cmd_predict_and_retrieve(self, kwargs):
        # Goal-tracked prediction (no retrieval blending — keep it pure)
        out = self.cmd_predict_chunk(kwargs)
        out["blended"] = False
        return out

    def cmd_adapt_v8(self, kwargs):
        return {"ok": True, "loss": 0.0, "note": "no-op for gt server"}

    def cmd_adapt_demo(self, kwargs):
        return {"ok": True, "loss": 0.0, "note": "no-op for gt server"}

    def cmd_set_retrieval_filter(self, kwargs):
        # No-op (retrieval not used in this server but client may call it)
        return {"ok": True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student_ckpt", required=True, type=str,
                   help="v10-DEMO motor checkpoint")
    p.add_argument("--goal_tracker", required=True, type=str,
                   help="trained SubstrateGoalTracker .pt")
    p.add_argument("--port", type=int, default=7777)
    p.add_argument("--memory_bank", default="", type=str)
    p.add_argument("--retrieve_top_k", type=int, default=3)
    p.add_argument("--retrieve_alpha", type=float, default=0.0)
    p.add_argument("--retrieve_filter_suite_default", default="libero_10", type=str)
    p.add_argument("--gt_threshold", type=float, default=0.5,
                   help="P(open) threshold above which goal tracker overrides gripper")
    p.add_argument("--gt_apply_positions", type=int, default=8,
                   help="Number of chunk positions to apply override to (default=exec_horizon)")
    args = p.parse_args()

    server = GoalTrackerServer(args)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://*:{args.port}")
    print(f"[gt_server] listening on tcp://*:{args.port}")

    cmd_table = {
        "init": server.cmd_init,
        "episode_reset": server.cmd_episode_reset,
        "predict_chunk": server.cmd_predict_chunk,
        "predict_and_retrieve": server.cmd_predict_and_retrieve,
        "adapt_v8": server.cmd_adapt_v8,
        "adapt_demo": server.cmd_adapt_demo,
        "set_retrieval_filter": server.cmd_set_retrieval_filter,
    }

    while True:
        try:
            msg = pickle.loads(sock.recv())
        except Exception as e:
            print(f"[gt_server] recv error: {e}"); continue
        cmd = msg.get("cmd")
        kwargs = {k: v for k, v in msg.items() if k != "cmd"}
        t0 = time.perf_counter()
        if cmd in cmd_table:
            try:
                resp = cmd_table[cmd](kwargs)
            except Exception as e:
                import traceback
                resp = {"ok": False, "error": f"{e}\n{traceback.format_exc()}"}
        else:
            resp = {"ok": False, "error": f"unknown cmd: {cmd}"}
        resp["server_ms"] = (time.perf_counter() - t0) * 1000
        sock.send(pickle.dumps(resp))


if __name__ == "__main__":
    main()
