"""Multi-turn agentic trace generation with DUAL encoders for two-flow text validation.

For each GSM8K problem, run a 5-turn agentic protocol:
  T1 user: "Solve this step by step." (initial)
  T2 model: attempts solution
  T3 user: "Verify each step. If you find an error, fix it."
  T4 model: verifies / revises
  T5 user: "Give your final answer as 'Final Answer: <number>'."
  T6 model: commits

Chunk per 24 tokens across the ENTIRE multi-turn assistant generation. Per-chunk
encoding by TWO independent encoders:
  z_t (fast)    = bge-small-en-v1.5    [384]  retrieval-trained
  z_lang (slow) = all-MiniLM-L6-v2     [384]  similarity-trained on diverse pairs

Different models, different training objectives → genuinely independent latent
streams. This is the proper text analog of GR00T's (z_vl from vision branch,
z_lang from language branch) two-encoder split.

Success label: final number matches GT.
"""
import argparse
import re
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


def extract_final_number(text: str):
    matches = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if not matches:
        return None
    last = matches[-1].replace(",", "")
    try:
        return float(last)
    except ValueError:
        return None


def chunk_token_ids(token_ids, chunk_size=24):
    chunks = []
    for i in range(0, len(token_ids), chunk_size):
        chunks.append(token_ids[i:i + chunk_size])
    return chunks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen_model", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    p.add_argument("--enc_fast", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--enc_slow", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--n_problems", type=int, default=250)
    p.add_argument("--max_new_tokens", type=int, default=256,
                   help="Per-turn cap on model generation")
    p.add_argument("--chunk_size", type=int, default=24)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gen] device={device}", flush=True)
    print(f"[gen] gen={args.gen_model}", flush=True)
    print(f"[gen] fast enc={args.enc_fast}", flush=True)
    print(f"[gen] slow enc={args.enc_slow}", flush=True)

    # === Load encoders ===
    print("[gen] loading fast encoder...", flush=True)
    fast_tok = AutoTokenizer.from_pretrained(args.enc_fast)
    fast_model = AutoModel.from_pretrained(args.enc_fast).to(device).eval()
    fast_dim = fast_model.config.hidden_size

    print("[gen] loading slow encoder...", flush=True)
    slow_tok = AutoTokenizer.from_pretrained(args.enc_slow)
    slow_model = AutoModel.from_pretrained(args.enc_slow).to(device).eval()
    slow_dim = slow_model.config.hidden_size

    assert fast_dim == slow_dim, f"encoder dims must match: fast={fast_dim}, slow={slow_dim}"
    print(f"[gen] encoder dim={fast_dim}", flush=True)

    # === Load generator ===
    print("[gen] loading generator...", flush=True)
    gen_tok = AutoTokenizer.from_pretrained(args.gen_model, trust_remote_code=True)
    gen_model = AutoModelForCausalLM.from_pretrained(
        args.gen_model, dtype=torch.float16, trust_remote_code=True,
    ).to(device).eval()
    if gen_tok.pad_token_id is None:
        gen_tok.pad_token_id = gen_tok.eos_token_id

    # === Load GSM8K ===
    print("[gen] loading GSM8K...", flush=True)
    ds = load_dataset("openai/gsm8k", "main", split="train")
    problems = list(ds)[:args.n_problems]
    print(f"[gen] {len(problems)} problems", flush=True)

    @torch.no_grad()
    def encode(text, which="fast"):
        tok = fast_tok if which == "fast" else slow_tok
        mod = fast_model if which == "fast" else slow_model
        toks = tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        out = mod(**toks)
        if which == "fast":
            # bge-small: CLS-pool, then normalize
            v = out.last_hidden_state[:, 0]
        else:
            # MiniLM: mean-pool over attention mask
            mask = toks["attention_mask"].unsqueeze(-1).float()
            v = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        v = torch.nn.functional.normalize(v, dim=-1)
        return v.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def encode_running_text(prompt_ids, generated_ids_so_far, which):
        full_ids = torch.cat([prompt_ids, generated_ids_so_far], dim=-1)
        text = gen_tok.decode(full_ids[0], skip_special_tokens=True)
        return encode(text, which)

    # 5-turn user prompts to drive multi-turn protocol
    user_turn_2 = "Verify each step of your solution carefully. If you find any error, fix it and continue."
    user_turn_3 = "Now give your final answer in exactly this format on its own line: 'Final Answer: <number>'."

    records = []
    n_succ = 0
    t_start = time.time()
    for i, ex in enumerate(problems):
        q = ex["question"]
        gt_str = ex["answer"].split("####")[-1].strip().replace(",", "")
        try:
            gt = float(gt_str)
        except ValueError:
            continue

        sys_msg = "You are a careful math problem solver. Work step-by-step and verify each step."
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": q + " Solve step by step, showing all work."},
        ]

        all_new_tokens_per_turn = []
        all_assistant_text = []
        for turn_idx in range(3):
            chat = gen_tok.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)
            prompt_ids = gen_tok(chat, return_tensors="pt").input_ids.to(device)
            gen_out = gen_model.generate(
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=True, temperature=args.temperature, top_p=0.95,
                pad_token_id=gen_tok.pad_token_id,
            )
            new_tokens = gen_out[0, prompt_ids.shape[1]:]
            new_text = gen_tok.decode(new_tokens, skip_special_tokens=True)
            all_new_tokens_per_turn.append(new_tokens)
            all_assistant_text.append(new_text)
            messages.append({"role": "assistant", "content": new_text})
            if turn_idx == 0:
                messages.append({"role": "user", "content": user_turn_2})
            elif turn_idx == 1:
                messages.append({"role": "user", "content": user_turn_3})

        # Final answer extraction from last turn
        final_text = all_assistant_text[-1]
        ans = extract_final_number(final_text)
        # Fallback: check entire concatenated text
        if ans is None:
            ans = extract_final_number(" ".join(all_assistant_text))
        succ = int(ans is not None and abs(ans - gt) < 1e-3)
        n_succ += succ

        # === Encode: concatenate all assistant generations, chunk, encode each chunk ===
        all_tokens_concat = torch.cat(all_new_tokens_per_turn, dim=0)
        # Build the full text we'll incrementally encode
        # Use just the assistant text concatenation for encoding context (excludes user turns)
        full_assistant_text = " ".join(all_assistant_text)
        chunks = chunk_token_ids(all_tokens_concat.tolist(), args.chunk_size)

        z_goal_fast = encode(f"Problem: {q}", "fast")
        z_goal_slow = encode(f"Problem: {q}", "slow")

        z_t_list, z_lang_list = [], []
        running = ""
        # Iterate through chunks, building running text by appending decoded chunk text
        for ch in chunks:
            ch_text = gen_tok.decode(ch, skip_special_tokens=True)
            running = running + ch_text
            z_t_list.append(encode(running, "fast"))
            z_lang_list.append(encode(running, "slow"))

        if not z_t_list:
            continue
        z_t_traj = np.stack(z_t_list, axis=0)
        z_lang_traj = np.stack(z_lang_list, axis=0)
        T = z_t_traj.shape[0]

        records.append({
            "z_t_traj":    torch.from_numpy(z_t_traj).float(),
            "z_lang_traj": torch.from_numpy(z_lang_traj).float(),
            "z_goal":      torch.from_numpy(z_goal_fast).float(),
            "z_goal_slow": torch.from_numpy(z_goal_slow).float(),
            "T": T,
            "succ": succ,
            "sub_id": int(i),
            "gt": float(gt),
            "ans": float(ans) if ans is not None else None,
            "prompt": q,
            "assistant_turns": all_assistant_text,
        })

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t_start
            print(f"[gen] [{i+1}/{len(problems)}]  succ={n_succ}/{i+1} = "
                  f"{100*n_succ/(i+1):.0f}%  T_avg={np.mean([r['T'] for r in records]):.1f}  "
                  f"elapsed={elapsed:.0f}s  "
                  f"eta={elapsed*(len(problems)-i-1)/(i+1):.0f}s", flush=True)

    avg_T = float(np.mean([r["T"] for r in records])) if records else 0
    print(f"[gen] DONE. succ={n_succ}/{len(records)}  T_avg={avg_T:.1f}", flush=True)
    print(f"[gen] saving to {args.output}", flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "records": records,
        "enc_dim": fast_dim,
        "enc_fast": args.enc_fast,
        "enc_slow": args.enc_slow,
        "gen_model": args.gen_model,
        "chunk_size": args.chunk_size,
        "n_turns": 3,
    }, args.output)
    print(f"[gen] === ALL_DONE === saved {len(records)} records, total {sum(r['T'] for r in records)} chunk-events",
          flush=True)


if __name__ == "__main__":
    main()
