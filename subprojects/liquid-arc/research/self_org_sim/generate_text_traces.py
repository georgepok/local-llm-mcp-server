"""Generate text reasoning traces for two-flow substrate validation on text-only tasks.

For each GSM8K problem:
  1. Query small reasoning LLM (Qwen2.5-Math-1.5B-Instruct) to solve step-by-step
  2. Split generation into chunks (~30 tokens each)
  3. Encode each chunk with bge-small-en-v1.5 (CLS-pooled, normalized)
  4. Compute success label: does last number in generation match GT?
  5. Save record per problem with same shape as LIBERO traj:
     - z_t_traj         [T, 384]  per-chunk embeddings ← analog of GR00T z_vl
     - z_goal           [384]     problem statement embedding ← analog of z_goal
     - z_lang_traj      [T, 384]  same as z_t (text-only) ← analog of z_lang
     - chunk_traj       [T, H, A] zero-filled (no robotics chunks) — kept for shape compat
     - h_goal_traj      [T, K, d] random-init substrate output (filled lazily at train time)
     - state8_traj      [T, 8]    zero-filled (no robotics state)
     - succ             0 or 1
     - sub_id           problem id

Adapted from collect_traj_jepa_extended.py (robotics version).
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def extract_final_number(text: str):
    """Extract the last numeric value mentioned in text (handles 1,234 and $1.23)."""
    matches = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if not matches:
        return None
    last = matches[-1].replace(",", "")
    try:
        return float(last)
    except ValueError:
        return None


def chunk_generation(token_ids, chunk_size=32):
    """Split a token-id list into chunks of `chunk_size`. Returns list of slices."""
    chunks = []
    for i in range(0, len(token_ids), chunk_size):
        chunks.append(token_ids[i:i + chunk_size])
    return chunks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen_model", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    p.add_argument("--enc_model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--n_problems", type=int, default=200)
    p.add_argument("--max_new_tokens", type=int, default=384)
    p.add_argument("--chunk_size", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    p.add_argument("--start_idx", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gen] device={device}, gen={args.gen_model}, enc={args.enc_model}", flush=True)

    # === Load encoder (bge-small) ===
    print("[gen] loading encoder...", flush=True)
    enc_tok = AutoTokenizer.from_pretrained(args.enc_model)
    enc_model = AutoModel.from_pretrained(args.enc_model).to(device).eval()
    enc_dim = enc_model.config.hidden_size
    print(f"[gen] encoder dim={enc_dim}", flush=True)

    # === Load generator (Qwen) ===
    print("[gen] loading generator...", flush=True)
    gen_tok = AutoTokenizer.from_pretrained(args.gen_model, trust_remote_code=True)
    gen_model = AutoModelForCausalLM.from_pretrained(
        args.gen_model, torch_dtype=torch.float16, trust_remote_code=True,
    ).to(device).eval()
    if gen_tok.pad_token_id is None:
        gen_tok.pad_token_id = gen_tok.eos_token_id
    print(f"[gen] generator loaded ({sum(p.numel() for p in gen_model.parameters())/1e9:.2f}B params)",
          flush=True)

    # === Load GSM8K ===
    print("[gen] loading GSM8K...", flush=True)
    ds = load_dataset("openai/gsm8k", "main", split="train")
    problems = list(ds)[args.start_idx:args.start_idx + args.n_problems]
    print(f"[gen] {len(problems)} problems", flush=True)

    @torch.no_grad()
    def encode_text(text):
        toks = enc_tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        out = enc_model(**toks)
        cls = out.last_hidden_state[:, 0]  # CLS for bge
        return torch.nn.functional.normalize(cls, dim=-1).squeeze(0).cpu().numpy()

    @torch.no_grad()
    def encode_tokens_in_context(prompt_ids, generated_ids_so_far):
        """Encode the running text (prompt + so-far) as one vector. Returns [enc_dim]."""
        full_ids = torch.cat([prompt_ids, generated_ids_so_far], dim=-1)
        text = gen_tok.decode(full_ids[0], skip_special_tokens=True)
        return encode_text(text)

    records = []
    n_succ = 0
    t_start = time.time()
    for i, ex in enumerate(problems):
        q = ex["question"]
        # GT answer is at the end after ####
        gt_str = ex["answer"].split("####")[-1].strip().replace(",", "")
        try:
            gt = float(gt_str)
        except ValueError:
            print(f"[gen] skip {i}: bad GT '{gt_str}'", flush=True)
            continue

        sys_msg = "Solve the math problem step-by-step. End with: 'Final Answer: <number>'."
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": q},
        ]
        chat = gen_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = gen_tok(chat, return_tensors="pt").input_ids.to(device)

        gen_out = gen_model.generate(
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=True, temperature=args.temperature, top_p=0.95,
            pad_token_id=gen_tok.pad_token_id,
        )
        new_tokens = gen_out[0, prompt_ids.shape[1]:]
        gen_text = gen_tok.decode(new_tokens, skip_special_tokens=True)
        ans = extract_final_number(gen_text)
        succ = int(ans is not None and abs(ans - gt) < 1e-3)
        n_succ += succ

        # Encode goal once
        z_goal = encode_text(f"Problem: {q}")

        # Encode each chunk as the running text so far (prompt + new tokens up to chunk end)
        chunks = chunk_generation(new_tokens.tolist(), chunk_size=args.chunk_size)
        z_t_list, z_lang_list = [], []
        running = torch.empty(0, dtype=new_tokens.dtype, device=device)
        for ch in chunks:
            ch_tensor = torch.tensor(ch, device=device, dtype=new_tokens.dtype)
            running = torch.cat([running, ch_tensor])
            z = encode_tokens_in_context(prompt_ids, running.unsqueeze(0))
            z_t_list.append(z)
            z_lang_list.append(z)  # for text we set z_lang == z_t (no separate lang stream)

        z_t_traj = np.stack(z_t_list, axis=0)  # [T, enc_dim]
        z_lang_traj = np.stack(z_lang_list, axis=0)
        T = z_t_traj.shape[0]

        records.append({
            "z_t_traj": torch.from_numpy(z_t_traj).float(),
            "z_lang_traj": torch.from_numpy(z_lang_traj).float(),
            "z_goal": torch.from_numpy(z_goal).float(),
            "T": T,
            "succ": succ,
            "sub_id": int(i + args.start_idx),
            "gt": float(gt),
            "ans": float(ans) if ans is not None else None,
            "prompt": q,
            "generation": gen_text,
        })

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t_start
            print(f"[gen] [{i+1}/{len(problems)}]  succ={n_succ}/{i+1} = "
                  f"{100*n_succ/(i+1):.0f}%  elapsed={elapsed:.0f}s  "
                  f"eta={elapsed*(len(problems)-i-1)/(i+1):.0f}s", flush=True)

    print(f"[gen] DONE. succ={n_succ}/{len(records)} = {100*n_succ/max(1,len(records)):.0f}%", flush=True)
    print(f"[gen] saving to {args.output}", flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "records": records,
        "enc_dim": enc_dim,
        "gen_model": args.gen_model,
        "enc_model": args.enc_model,
        "chunk_size": args.chunk_size,
    }, args.output)
    print(f"[gen] === ALL_DONE === saved {len(records)} records", flush=True)


if __name__ == "__main__":
    main()
