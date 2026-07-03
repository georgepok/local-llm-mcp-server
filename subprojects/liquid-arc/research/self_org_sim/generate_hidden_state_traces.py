"""Generate multi-turn conversations with TRANSFORMER HIDDEN STATES extracted.

Liquid will receive these hidden states as input (not text encoder embeddings).
The transformer's internal activation IS the model's belief about its current
generation trajectory under the current goal — much richer signal than encoded
output text.

Per chunk: extract hidden_state from a chosen layer of the generator transformer,
pool over the chunk's tokens (mean over chunk tokens). Save as the substrate
input trajectory.

Output record:
  hidden_state_traj: [T, d_transformer]  per-chunk LLM hidden states
  z_goal:            [d_encoder]          encoded current goal text (for value head)
  z_goal_traj:       [T, d_encoder]       per-chunk goal (jumps at turn boundary)
  turn_followed, turn_chunk_starts, turn_categories, turn_instructions
"""
import argparse
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

# Reuse the diverse-goal makers
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_diverse_goals import (
    CATEGORY_MAKERS, TOPICS, PHRASING_TEMPLATES, _phrase, make_goal, chunk_token_ids,
)


@torch.no_grad()
def encode_text(text, tok, model, device):
    toks = tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    out = model(**toks)
    v = out.last_hidden_state[:, 0]
    return torch.nn.functional.normalize(v, dim=-1).squeeze(0).cpu().numpy()


@torch.no_grad()
def gen_with_hidden_states(gen_model, gen_tok, messages, max_new_tokens,
                              temperature, layer_idx, device):
    """Generate continuation and return (new_token_ids, per_token_hidden_states_at_layer).

    Hidden states are recorded ONLY for the newly-generated tokens (not the prompt).
    Returns:
      new_ids: [N_new] generated token ids
      hidden:  [N_new, d_transformer] per-new-token hidden state at layer_idx
    """
    chat = gen_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = gen_tok(chat, return_tensors="pt").input_ids.to(device)
    # Generate WITH hidden state output for each step
    gen_out = gen_model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True, temperature=temperature, top_p=0.95,
        pad_token_id=gen_tok.pad_token_id,
        return_dict_in_generate=True,
        output_hidden_states=True,
    )
    new_ids = gen_out.sequences[0, prompt_ids.shape[1]:]
    # hidden_states is a tuple of length n_steps (one per generated token).
    # Each element is a tuple of (n_layers + 1) tensors. Each tensor [B, T_step, d_hidden].
    # For step 0 (first generated token): T_step = 1 (the new token at position prompt_len)
    # For step k: T_step = 1 (autoregressive — KV cache, only new token's hidden state)
    # Note: huggingface returns ALL tokens hidden at step 0 (prompt) and just NEW at later steps
    hidden_per_step = []
    for step_idx, hs_tuple in enumerate(gen_out.hidden_states):
        # hs_tuple is (n_layers + 1,) each [B, T_step, d_hidden]
        layer_h = hs_tuple[layer_idx]  # [B, T_step, d]
        if step_idx == 0:
            # Step 0 also has the prompt tokens; we want just the LAST one (first new token)
            last_h = layer_h[0, -1]  # [d]
        else:
            # Step k>=1 has just the single new token
            last_h = layer_h[0, -1]  # [d]
        hidden_per_step.append(last_h.cpu())
    # Length should equal len(new_ids)
    hidden = torch.stack(hidden_per_step, dim=0)  # [N_new, d_hidden]
    # Handle mismatch (early stopping at EOS)
    if hidden.shape[0] > new_ids.shape[0]:
        hidden = hidden[:new_ids.shape[0]]
    elif hidden.shape[0] < new_ids.shape[0]:
        new_ids = new_ids[:hidden.shape[0]]
    return new_ids.cpu(), hidden


def pool_chunk_hidden(hidden_per_token, chunk_indices):
    """Mean-pool hidden states across a chunk's token positions."""
    if len(chunk_indices) == 0:
        return None
    return hidden_per_token[chunk_indices].mean(dim=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--enc_model", default="BAAI/bge-small-en-v1.5",
                   help="Encoder for goal text (used by value head)")
    p.add_argument("--n_conversations", type=int, default=400)
    p.add_argument("--n_goals", type=int, default=5)
    p.add_argument("--max_new_tokens", type=int, default=96)
    p.add_argument("--chunk_size", type=int, default=12)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--layer_idx", type=int, default=14,
                   help="Which transformer layer's hidden state to extract (Qwen2.5-1.5B has 28 layers + embedding)")
    p.add_argument("--exclude_category", default="")
    p.add_argument("--only_category", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    rng = random.Random(args.seed)
    device = torch.device("cuda")
    print(f"[hs] device={device}, layer_idx={args.layer_idx}", flush=True)

    # Encoder for goal text (separate from generator)
    enc_tok = AutoTokenizer.from_pretrained(args.enc_model)
    enc_model = AutoModel.from_pretrained(args.enc_model).to(device).eval()
    enc_dim = enc_model.config.hidden_size

    # Generator
    gen_tok = AutoTokenizer.from_pretrained(args.gen_model, trust_remote_code=True)
    gen_model = AutoModelForCausalLM.from_pretrained(
        args.gen_model, dtype=torch.float16, trust_remote_code=True,
        output_hidden_states=True,
    ).to(device).eval()
    if gen_tok.pad_token_id is None:
        gen_tok.pad_token_id = gen_tok.eos_token_id
    n_layers = gen_model.config.num_hidden_layers
    d_hidden = gen_model.config.hidden_size
    print(f"[hs] gen layers={n_layers}, d_hidden={d_hidden}", flush=True)
    print(f"[hs] extracting layer {args.layer_idx} of {n_layers}", flush=True)
    if args.layer_idx > n_layers:
        raise ValueError(f"layer_idx {args.layer_idx} > num_layers {n_layers}")

    records = []
    cat_follow = {c: [0, 0] for c in CATEGORY_MAKERS}
    t_start = time.time()

    for ci in range(args.n_conversations):
        messages = [{"role": "system",
                     "content": "Follow the user's instruction precisely. Address the most recent instruction."}]
        # Category selection
        if args.only_category:
            cats_for_conv = [args.only_category] * args.n_goals
        elif args.exclude_category:
            avail = [c for c in CATEGORY_MAKERS if c != args.exclude_category]
            cats_for_conv = [rng.choice(avail) for _ in range(args.n_goals)]
        else:
            cats_for_conv = [None] * args.n_goals
        goals = [make_goal(rng, category=cats_for_conv[i]) for i in range(args.n_goals)]

        all_concat_tokens = []
        all_concat_hidden = []           # accumulate hidden per generated token
        turn_outputs = []
        turn_followed = []
        turn_instructions = []
        turn_categories = []
        turn_goal_types = []
        turn_chunk_start_indices = []
        had_failure = False

        for ti, (cat, gtype, instr, check_fn, _tmpl) in enumerate(goals):
            messages.append({"role": "user", "content": instr})
            turn_instructions.append(instr)
            turn_categories.append(cat)
            turn_goal_types.append(gtype)

            try:
                new_ids, hidden = gen_with_hidden_states(
                    gen_model, gen_tok, messages,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    layer_idx=args.layer_idx, device=device,
                )
            except Exception as e:
                print(f"[hs] gen error sub={ci} turn={ti}: {e}", flush=True)
                had_failure = True
                break

            new_text = gen_tok.decode(new_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": new_text})
            turn_outputs.append(new_text)
            followed = bool(check_fn(new_text))
            turn_followed.append(followed)
            cat_follow[cat][0] += int(followed)
            cat_follow[cat][1] += 1

            current_chunk_start = len(all_concat_tokens) // args.chunk_size
            turn_chunk_start_indices.append(current_chunk_start)
            all_concat_tokens.extend(new_ids.tolist())
            all_concat_hidden.append(hidden)

        if had_failure or not all_concat_tokens:
            continue

        all_hidden = torch.cat(all_concat_hidden, dim=0)  # [N_total_new, d_hidden]
        # Chunk into chunk_size-token chunks; pool hidden per chunk
        chunks = chunk_token_ids(all_concat_tokens, args.chunk_size)
        T = len(chunks)
        hidden_per_chunk = []
        idx = 0
        for ch in chunks:
            end_idx = idx + len(ch)
            chunk_hidden = all_hidden[idx:end_idx].float().mean(dim=0)  # [d_hidden]
            if torch.isnan(chunk_hidden).any():
                hidden_per_chunk = []
                break
            hidden_per_chunk.append(chunk_hidden)
            idx = end_idx
        if not hidden_per_chunk:
            continue
        hidden_traj = torch.stack(hidden_per_chunk, dim=0)  # [T, d_hidden]

        # Chunk → turn map
        chunk_to_turn = []
        cur_turn = 0
        for ti in range(T):
            while (cur_turn + 1 < args.n_goals and
                   ti >= turn_chunk_start_indices[cur_turn + 1]):
                cur_turn += 1
            chunk_to_turn.append(cur_turn)

        # Encode goals (for value head's z_goal input)
        per_turn_z_goal = [encode_text(instr, enc_tok, enc_model, device)
                            for instr in turn_instructions]
        z_goal_traj = np.stack([per_turn_z_goal[chunk_to_turn[t]] for t in range(T)], axis=0)

        records.append({
            "hidden_state_traj": hidden_traj.float(),                         # [T, d_hidden]
            "z_goal_traj":       torch.from_numpy(z_goal_traj).float(),       # [T, enc_dim]
            "z_goal":            torch.from_numpy(per_turn_z_goal[0]).float(),# [enc_dim]
            "T": T,
            "succ": int(sum(turn_followed) >= args.n_goals // 2 + 1),
            "sub_id": ci,
            "turn_followed": turn_followed,
            "turn_chunk_starts": turn_chunk_start_indices,
            "turn_instructions": turn_instructions,
            "turn_categories": turn_categories,
            "turn_goal_types": turn_goal_types,
            "turn_outputs": turn_outputs,
        })

        if (ci + 1) % 10 == 0 or ci == 0:
            elapsed = time.time() - t_start
            n = len(records)
            avg_T = float(np.mean([r["T"] for r in records])) if records else 0
            cat_str = " ".join(f"{c[:4]}={cat_follow[c][0]}/{cat_follow[c][1]}"
                                  for c in CATEGORY_MAKERS if cat_follow[c][1] > 0)
            print(f"[hs] [{ci+1}/{args.n_conversations}]  recs={n}  T_avg={avg_T:.1f}  "
                  f"{cat_str}  elapsed={elapsed:.0f}s  "
                  f"eta={elapsed*(args.n_conversations-ci-1)/(ci+1):.0f}s", flush=True)

    print(f"[hs] DONE. records={len(records)}", flush=True)
    for c in CATEGORY_MAKERS:
        f, t = cat_follow[c]
        if t > 0:
            print(f"  {c}: {f}/{t} = {100*f/t:.0f}% followed", flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "records": records,
        "enc_dim": enc_dim,
        "d_hidden": d_hidden,
        "n_layers": n_layers,
        "layer_idx": args.layer_idx,
        "enc_model": args.enc_model,
        "gen_model": args.gen_model,
        "chunk_size": args.chunk_size,
        "n_goals": args.n_goals,
        "categories": list(CATEGORY_MAKERS.keys()),
    }, args.output)
    print(f"[hs] === ALL_DONE === saved {len(records)} records", flush=True)


if __name__ == "__main__":
    main()
