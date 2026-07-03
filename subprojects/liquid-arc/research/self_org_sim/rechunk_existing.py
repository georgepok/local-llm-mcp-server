"""Re-encode existing multi-turn text traces at smaller chunk size to get longer T.

Original generation stored assistant_turns text per problem. Re-tokenize, re-chunk
at smaller size, re-encode with both fast and slow encoders. Output robotics-format.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


@torch.no_grad()
def encode(text, tok, model, device, pool="cls"):
    toks = tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    out = model(**toks)
    if pool == "cls":
        v = out.last_hidden_state[:, 0]
    else:
        mask = toks["attention_mask"].unsqueeze(-1).float()
        v = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1.0)
    return torch.nn.functional.normalize(v, dim=-1).squeeze(0).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--chunk_size", type=int, default=12)
    p.add_argument("--enc_fast", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--enc_slow", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--gen_model", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    args = p.parse_args()

    device = torch.device("cuda")
    print(f"[rechunk] chunk_size={args.chunk_size}", flush=True)

    fast_tok = AutoTokenizer.from_pretrained(args.enc_fast)
    fast_mod = AutoModel.from_pretrained(args.enc_fast).to(device).eval()
    slow_tok = AutoTokenizer.from_pretrained(args.enc_slow)
    slow_mod = AutoModel.from_pretrained(args.enc_slow).to(device).eval()
    gen_tok = AutoTokenizer.from_pretrained(args.gen_model, trust_remote_code=True)

    src = torch.load(args.src, map_location="cpu", weights_only=False)
    out_records = []
    Ts = []
    for ri, r in enumerate(src["records"]):
        # Concatenate assistant turns, retokenize with gen_tok, rechunk
        full_text = " ".join(r.get("assistant_turns", []))
        if not full_text.strip():
            continue
        all_ids = gen_tok(full_text, return_tensors="pt").input_ids[0].tolist()
        chunks = []
        for i in range(0, len(all_ids), args.chunk_size):
            chunks.append(all_ids[i:i + args.chunk_size])
        if not chunks:
            continue

        z_goal_fast = encode(f"Problem: {r['prompt']}", fast_tok, fast_mod, device, "cls")
        z_goal_slow = encode(f"Problem: {r['prompt']}", slow_tok, slow_mod, device, "mean")

        z_t_list, z_lang_list = [], []
        running = ""
        any_nan = False
        for ch in chunks:
            ch_text = gen_tok.decode(ch, skip_special_tokens=True)
            running = running + ch_text
            zf = encode(running, fast_tok, fast_mod, device, "cls")
            zs = encode(running, slow_tok, slow_mod, device, "mean")
            if np.isnan(zf).any() or np.isnan(zs).any():
                any_nan = True
                break
            z_t_list.append(zf)
            z_lang_list.append(zs)
        if any_nan or not z_t_list:
            continue

        z_t_traj = np.stack(z_t_list, axis=0)
        z_lang_traj = np.stack(z_lang_list, axis=0)
        T = z_t_traj.shape[0]
        Ts.append(T)
        out_records.append({
            "z_vl_traj":   torch.from_numpy(z_t_traj).float(),
            "z_lang_traj": torch.from_numpy(z_lang_traj).float(),
            "z_goal":      torch.from_numpy(z_goal_fast).float(),
            "state8_traj": torch.zeros(T, 8),
            "chunk_traj":  torch.zeros(T, 16, 7),
            "h_goal_traj": torch.zeros(T, 4, 64),
            "succ":        int(r["succ"]),
            "sub_id":      int(r["sub_id"]),
        })
        if (ri + 1) % 25 == 0:
            print(f"  [{ri+1}]  T_avg={np.mean(Ts):.1f}", flush=True)

    print(f"[rechunk] kept {len(out_records)} records. T mean={np.mean(Ts):.1f} "
          f"min={np.min(Ts)} max={np.max(Ts)}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"records": out_records}, args.out)
    print(f"[rechunk] saved → {args.out}", flush=True)
    print("[rechunk] === ALL_DONE ===", flush=True)


if __name__ == "__main__":
    main()
