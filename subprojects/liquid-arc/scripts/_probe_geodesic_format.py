"""One-shot probe: what does Qwen3-4B actually emit for a geodesic prompt?

Loads model, runs ONE generation each (natural + adversarial) for a small
graph, prints the raw decoded output verbatim. Use this to fix the parser
before relaunching the full sweep.
"""

import sys, os, re, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.bench_geodesic import (
    make_graph, render_natural, render_adversarial, PROMPT_TEMPLATE,
    parse_response, score,
)
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/workspace/models/qwen3-4b"

print("loading...")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
llm = AutoModelForCausalLM.from_pretrained(
    MODEL, device_map='cuda', torch_dtype=torch.bfloat16, trust_remote_code=True)
llm.eval()
print("loaded")

g = make_graph(n=10, extra_edges=4, seed=0)
assert g is not None
rng_n = random.Random(0)
rng_a = random.Random(1)

for name, edges in [("natural", render_natural(g, rng_n)),
                    ("adversarial", render_adversarial(g, rng_a))]:
    prompt = PROMPT_TEMPLATE.format(edges=edges, s=g.s, t=g.t)
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    print(f"PROMPT:\n{prompt}")
    print(f"\nTRUTH: optimal {g.optimal_path}={g.optimal_cost}  "
          f"decoy {g.decoy_path}={g.decoy_cost}")

    # Try with explicit thinking-disabled chat template
    msgs = [{"role": "user", "content": prompt}]
    try:
        full = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        full = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(full, return_tensors='pt', truncation=True, max_length=8192).to('cuda')
    n = inp['input_ids'].shape[1]
    print(f"\nprompt_tokens={n}")

    for max_new in [200, 600]:
        with torch.no_grad():
            out = llm.generate(**inp, max_new_tokens=max_new, do_sample=False,
                               repetition_penalty=1.05)
        raw = tok.decode(out[0][n:], skip_special_tokens=False)
        print(f"\n--- max_new={max_new} | RAW (with special tokens) ---")
        print(repr(raw[:1500]))
        print("--- end raw ---")
        # Try the parser
        clean = tok.decode(out[0][n:], skip_special_tokens=True)
        m = re.search(r'</think>\s*(.*)', clean, flags=re.DOTALL)
        post_think = m.group(1).strip() if m and len(m.group(1).strip()) > 5 else clean
        nodes, cost = parse_response(post_think)
        print(f"parsed_path={nodes} parsed_cost={cost}")
        sc = score(g, post_think)
        print(f"score_label={sc['label']} true_cost={sc['true_cost']}")
        if max_new == 200 and nodes:
            break  # parsed at small budget, don't need bigger
