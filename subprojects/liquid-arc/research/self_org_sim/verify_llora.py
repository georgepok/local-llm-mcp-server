"""Verify the Liquid->LoRA integration is doing what it claims. Checks, on the 1.5B (fast):
  (1) GRAD FLOW: after a REINFORCE backward, gradient norm on controller / coeff_head / LoRA-A /
      LoRA-B — any ~0 means that component isn't training (broken loop).
  (2) DELTA MAGNITUDE: |LoRA delta| / |layer output| — ~0 means the adapter is effectively OFF.
  (3) CONDITIONING: per-goal coeffs for 4 different goals — if ~identical the LoRA is NOT
      goal-specific (degenerate), which would explain no generalization.
"""
import sys
from pathlib import Path
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from train_steer_controller import SteerController, encode_goal
from train_liquid_lora import DynLoRA, run_traj_lora, build_plans
from train_steer_commit import COMMIT_GOALS
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


class A:  # arg holder
    n_steps = 4; n_turns = 4; advance_thr = 0.62; think = False
    max_new_tokens = 35; temperature = 0.8; min_len = 8; lambda_flu = 1.0; ref_flu = -1.3; beta_lora = 0.01


def main():
    device = torch.device("cuda")
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True); torch.set_float32_matmul_precision("high")
    gm = "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(gm, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(gm, dtype=torch.float16, trust_remote_code=True,
                                                   low_cpu_mem_usage=True, device_map={"": 0}).eval()
    for pp in model.parameters():
        pp.requires_grad = False
    enc_tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    enc_model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5").to(device).eval()
    ctrl = SteerController(d_llm=model.config.hidden_size, d=128, K=4, use_slow=True, n_inject=1).to(device)
    layers = [8, 14, 20, 26]
    lora = DynLoRA(model, layers, "o_proj", ctrl_state_dim=4 * 128, r=8, K=8).to(device)
    lora.register()
    opt = torch.optim.AdamW(list(ctrl.parameters()) + list(lora.parameters()), lr=2e-4)
    plans = build_plans(model, tok, enc_tok, enc_model, COMMIT_GOALS[:6], 4, device)
    goals = [g for g in COMMIT_GOALS[:6] if g in plans]
    rng = np.random.default_rng(0)

    print("\n=== (1) train 24 steps, then check GRAD FLOW on the last step ===", flush=True)
    gnorm = {"controller": 0.0, "coeff_head": 0.0, "lora.A": 0.0, "lora.B": 0.0}
    for step in range(1, 25):
        opt.zero_grad()
        g = goals[rng.integers(len(goals))]; _, wemb = plans[g]; zG = encode_goal(g, enc_tok, enc_model, device)
        prog, logps, flu, min_n, mag = run_traj_lora(model, tok, ctrl, lora, enc_tok, enc_model, g, wemb, zG, device, A, grad=True)
        if not logps:
            continue
        R = prog * (1.0 if min_n >= A.min_len else 0.0) - A.lambda_flu * max(0.0, A.ref_flu - flu)
        adv = R - 0.4
        loss = -adv * torch.stack(logps).sum() + (A.beta_lora * mag if mag is not None else 0.0)
        (loss * 256.0).backward()
        for pp in list(ctrl.parameters()) + list(lora.parameters()):
            if pp.grad is not None:
                pp.grad /= 256.0
        if step == 24:
            gnorm["controller"] = float(torch.norm(torch.stack([p.grad.norm() for p in ctrl.parameters() if p.grad is not None])))
            gnorm["coeff_head"] = float(torch.norm(torch.stack([p.grad.norm() for p in lora.coeff_head.parameters() if p.grad is not None])))
            gnorm["lora.A"] = float(torch.norm(torch.stack([p.grad.norm() for p in lora.A if p.grad is not None])) if any(p.grad is not None for p in lora.A) else torch.tensor(0.0))
            gnorm["lora.B"] = float(torch.norm(torch.stack([p.grad.norm() for p in lora.B if p.grad is not None])) if any(p.grad is not None for p in lora.B) else torch.tensor(0.0))
        torch.nn.utils.clip_grad_norm_(list(ctrl.parameters()) + list(lora.parameters()), 1.0)
        opt.step()
    for k, v in gnorm.items():
        flag = " <-- ZERO! not training" if v < 1e-9 else ""
        print(f"   grad-norm {k:12s} = {v:.3e}{flag}", flush=True)
    print(f"   LoRA-B max abs = {max(float(b.abs().max()) for b in lora.B):.4f}  (was 0 at init; >0 means it learned)", flush=True)

    print("\n=== (2)+(3) DELTA MAGNITUDE + CONDITIONING across 4 goals (first waypoint) ===", flush=True)
    ctrl.eval(); lora.eval(); lora.debug = True
    coeff_vecs = []
    for g in goals[:4]:
        _, wemb = plans[g]; zG = encode_goal(g, enc_tok, enc_model, device)
        ctrl.reset_episode(1, device); ctrl.slow_step(zG.unsqueeze(0))
        h = ctrl.dyn_state(wemb[0].unsqueeze(0)); lora.set_state(h)
        lora.active = True
        msgs = [{"role": "user", "content": f"I want to {g}. Let's begin."}]
        chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        with torch.no_grad():
            model(tok(chat, return_tensors="pt").input_ids.to(device))
        lora.active = False
        c = lora.coeffs.detach().flatten().cpu().numpy(); coeff_vecs.append(c)
        print(f"   goal[{g[:34]:34s}] delta/out per-layer={[round(x,3) for x in lora.last_rel]}  "
              f"coeffs[:6]={np.round(c[:6],3)}", flush=True)
    cv = np.stack(coeff_vecs)
    # pairwise cosine between goals' coeff vectors: ~1.0 = identical (degenerate), <0.9 = conditioned
    import itertools
    cos = [float(np.dot(cv[i], cv[j]) / (np.linalg.norm(cv[i]) * np.linalg.norm(cv[j]) + 1e-9))
           for i, j in itertools.combinations(range(len(cv)), 2)]
    print(f"\n   pairwise coeff cosine across goals: mean={np.mean(cos):.3f} (1.0=degenerate/constant, "
          f"<0.9=genuinely goal-conditioned)", flush=True)
    print("[verify] === ALL_DONE ===", flush=True)


if __name__ == "__main__":
    main()
