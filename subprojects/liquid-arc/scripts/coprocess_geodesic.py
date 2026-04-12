"""coprocess_geodesic.py — failing LLM + LiquidARC → working solver.

The LLM in direct mode (no CoT) scores ~22% on the geodesic bench. The
hand-coded ODE scores 100% but doesn't involve the LLM. This script puts
them together in two modes:

  text:    ODE solves; result injected as text hint into LLM prompt.
           Frozen LLM, no training. Establishes the upper-bound that the
           combined system CAN succeed.

  learned: ODE solves; result encoded by a small trained encoder into K
           soft-prompt tokens prepended to the LLM input embeddings.
           Frozen LLM, encoder trained end-to-end with CE on the answer.
           Demonstrates the bridge IS learnable.

Both evaluated on the same 240-graph bench as bench_geodesic.py.

Run (text-mode quick check):
  python3 scripts/coprocess_geodesic.py --mode text \
      --n_graphs 30 --sizes 10,14,18,22 --cycle_densities 0.4,0.8

Run (learned bridge — trains then evals):
  python3 scripts/coprocess_geodesic.py --mode learned \
      --train_steps 3000 --n_graphs 30 --sizes 10,14,18,22 \
      --cycle_densities 0.4,0.8
"""

import argparse, json, random, re, sys, os, time, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

# Make sibling scripts importable
_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS)
sys.path.insert(0, os.path.dirname(_THIS))

from bench_geodesic import (
    make_graph, render_natural, score, Graph, PROMPT_DIRECT,
)

PROMPT_NO_EDGES = """The shortest-path query has been pre-encoded for you. \
Output exactly one line in the form:
PATH: {s} -> ... -> {t}  COST: <integer>"""

from geo_solver import solve_min_plus, build_matrices

from transformers import AutoModelForCausalLM, AutoTokenizer


# ─────────────────────────────────────────────────────────────────────
# Shared LLM helpers
# ─────────────────────────────────────────────────────────────────────

def load_llm(model_path):
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    llm = AutoModelForCausalLM.from_pretrained(
        model_path, device_map='cuda', torch_dtype=torch.bfloat16,
        trust_remote_code=True)
    llm.eval()
    for p in llm.parameters():
        p.requires_grad_(False)
    return llm, tok


def chat_format(tok, user_content):
    msgs = [{"role": "user", "content": user_content}]
    try:
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)


def make_target(g, path, cost):
    return f"PATH: {' -> '.join(path)}  COST: {cost}"


def gen_and_score(llm, tok, prompt_str, prefix_embeds, g, max_new=80):
    """Run generation. If prefix_embeds is given (learned mode), prepend
    the soft tokens to the input embeddings."""
    full = chat_format(tok, prompt_str)
    inp = tok(full, return_tensors='pt').to('cuda')
    input_ids = inp['input_ids']
    attn = inp['attention_mask']

    if prefix_embeds is not None:
        # Build inputs_embeds = [prefix_embeds, embed(input_ids)]
        emb_layer = llm.get_input_embeddings()
        tok_embeds = emb_layer(input_ids).to(prefix_embeds.dtype)
        inputs_embeds = torch.cat([prefix_embeds, tok_embeds], dim=1)
        prefix_attn = torch.ones(
            (1, prefix_embeds.shape[1]), dtype=attn.dtype, device=attn.device)
        attn_full = torch.cat([prefix_attn, attn], dim=1)
        with torch.no_grad():
            out = llm.generate(
                inputs_embeds=inputs_embeds, attention_mask=attn_full,
                max_new_tokens=max_new, do_sample=False,
                pad_token_id=tok.pad_token_id)
        # When using inputs_embeds, generate returns ONLY new tokens
        gen_ids = out[0]
    else:
        with torch.no_grad():
            out = llm.generate(
                **inp, max_new_tokens=max_new, do_sample=False,
                pad_token_id=tok.pad_token_id)
        gen_ids = out[0][input_ids.shape[1]:]

    txt = tok.decode(gen_ids, skip_special_tokens=True)
    txt = re.sub(r'</?think>', '', txt).strip()
    return score(g, txt), txt


# ─────────────────────────────────────────────────────────────────────
# MODE A — text-grounded
# ─────────────────────────────────────────────────────────────────────

def text_hint_prompt(g, edges_text, hint):
    base = PROMPT_DIRECT.format(edges=edges_text, s=g.s, t=g.t)
    if hint is None:
        return base
    return (f"An external geometric solver suggests the answer is:\n"
            f"  {hint}\n\nVerify this is consistent with the listed edges, "
            f"then output exactly that one line.\n\n{base}")


def run_text_mode(llm, tok, graphs):
    results = []
    rng = random.Random(0)
    for gi, g in enumerate(graphs):
        edges_text = render_natural(g, random.Random(1000 + gi))
        path, cost, _ = solve_min_plus(g)
        hint = make_target(g, path, cost) if path else None
        prompt = text_hint_prompt(g, edges_text, hint)
        s, _ = gen_and_score(llm, tok, prompt, prefix_embeds=None, g=g)
        results.append(s)
    return results


# ─────────────────────────────────────────────────────────────────────
# MODE B — learned soft-prompt bridge
# ─────────────────────────────────────────────────────────────────────

class GraphEncoder(nn.Module):
    """Encodes (W, s, t, ODE_solution) → K soft prompt tokens.

    The ODE provides node-level distance-to-target values. The encoder
    learns to embed the path and key graph features into the LLM's
    embedding space as a small soft prefix.
    """
    def __init__(self, d_llm: int, d_hidden: int = 256, n_soft: int = 16,
                 max_nodes: int = 32):
        super().__init__()
        self.d_llm = d_llm
        self.n_soft = n_soft
        self.max_nodes = max_nodes
        # Per-node feature: [is_source, is_target, on_path, dist_to_target_norm,
        #                    degree_norm, position_in_path_norm]
        self.feat_dim = 6
        self.node_proj = nn.Linear(self.feat_dim, d_hidden)
        # Small transformer over node sequence
        layer = nn.TransformerEncoderLayer(
            d_model=d_hidden, nhead=4, dim_feedforward=d_hidden * 2,
            batch_first=True, dropout=0.0, activation='gelu')
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        # Soft prompt tokens: queries that pool over node sequence
        self.soft_queries = nn.Parameter(torch.randn(n_soft, d_hidden) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_hidden, num_heads=4, batch_first=True, dropout=0.0)
        self.out_proj = nn.Linear(d_hidden, d_llm)
        # Initialize out_proj small so initial soft tokens don't disrupt LLM
        nn.init.normal_(self.out_proj.weight, std=0.01)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, W: torch.Tensor, s_idx: int, t_idx: int,
                path_idx: list, dist_to_t: torch.Tensor):
        """W: [N,N] adjacency (float, weighted)
           s_idx, t_idx: int
           path_idx: list of node indices in shortest path
           dist_to_t: [N] true distance from each node to target (for richness)
        Returns: [1, n_soft, d_llm] soft prompt embeddings.
        """
        N = W.shape[0]
        device = W.device
        feats = torch.zeros(N, self.feat_dim, device=device)
        feats[s_idx, 0] = 1.0
        feats[t_idx, 1] = 1.0
        path_pos = {ni: i for i, ni in enumerate(path_idx)}
        for ni, pos in path_pos.items():
            feats[ni, 2] = 1.0
            feats[ni, 5] = pos / max(1, len(path_idx) - 1)
        max_dist = dist_to_t[torch.isfinite(dist_to_t)].max().clamp(min=1.0)
        feats[:, 3] = torch.where(
            torch.isfinite(dist_to_t), dist_to_t / max_dist,
            torch.ones_like(dist_to_t))
        deg = (W > 0).float().sum(dim=1)
        feats[:, 4] = deg / deg.max().clamp(min=1.0)

        h = self.node_proj(feats)         # [N, d_hidden]
        h = self.transformer(h.unsqueeze(0)).squeeze(0)  # [N, d_hidden]
        q = self.soft_queries.unsqueeze(0)  # [1, n_soft, d_hidden]
        kv = h.unsqueeze(0)                 # [1, N, d_hidden]
        soft, _ = self.cross_attn(q, kv, kv)  # [1, n_soft, d_hidden]
        soft = self.out_proj(soft)            # [1, n_soft, d_llm]
        return soft


def features_for_graph(g: Graph):
    """Returns (W [N,N] float, s_idx, t_idx, path_idx list, dist_to_t [N])
    using the ODE solver + Dijkstra ground truth (for dist field richness).
    """
    nodes, idx, W_np, _, D_init = build_matrices(g)
    N = len(nodes)
    s_idx, t_idx = idx[g.s], idx[g.t]
    path_lab, cost, _ = solve_min_plus(g)
    if path_lab is None:
        path_idx = []
    else:
        path_idx = [idx[lab] for lab in path_lab]
    # Dist-to-target via min-plus closure (already computed inside solver)
    # Recompute the all-pairs matrix here for the dist field
    D = D_init.copy()
    for _ in range(int(math.ceil(math.log2(max(2, N))))):
        D = (D[:, :, None] + D[None, :, :]).min(axis=1)
    dist_to_t = D[:, t_idx]
    W_t = torch.from_numpy(W_np).float().cuda()
    dist_to_t_t = torch.from_numpy(dist_to_t).float().cuda()
    return W_t, s_idx, t_idx, path_idx, dist_to_t_t, path_lab, cost


def loss_for_example(llm, tok, encoder, g, edges_text, no_edge_text=False):
    """Teacher-forced CE on the target answer string."""
    W_t, s_idx, t_idx, path_idx, dist_to_t_t, path_lab, cost = features_for_graph(g)
    if path_lab is None:
        return None
    target_str = make_target(g, path_lab, cost)

    soft = encoder(W_t, s_idx, t_idx, path_idx, dist_to_t_t)  # [1,K,d_llm]
    soft = soft.to(torch.bfloat16)

    if no_edge_text:
        user = PROMPT_NO_EDGES.format(s=g.s, t=g.t)
    else:
        user = PROMPT_DIRECT.format(edges=edges_text, s=g.s, t=g.t)
    prompt_str = chat_format(tok, user)
    prompt_ids = tok(prompt_str, return_tensors='pt').input_ids.cuda()
    target_ids = tok(target_str + tok.eos_token,
                     return_tensors='pt', add_special_tokens=False).input_ids.cuda()

    emb_layer = llm.get_input_embeddings()
    prompt_emb = emb_layer(prompt_ids).to(torch.bfloat16)
    target_emb = emb_layer(target_ids).to(torch.bfloat16)
    inputs_embeds = torch.cat([soft, prompt_emb, target_emb], dim=1)

    n_prefix = soft.shape[1] + prompt_emb.shape[1]
    n_target = target_ids.shape[1]
    attn = torch.ones(
        (1, inputs_embeds.shape[1]), dtype=torch.long, device='cuda')

    out = llm(inputs_embeds=inputs_embeds, attention_mask=attn,
              use_cache=False)
    logits = out.logits  # [1, T, V]
    # Predict target_ids[t] from position n_prefix + t - 1
    pred_logits = logits[:, n_prefix - 1: n_prefix - 1 + n_target, :]
    loss = F.cross_entropy(
        pred_logits.reshape(-1, pred_logits.shape[-1]).float(),
        target_ids.reshape(-1))
    return loss


def train_encoder(llm, tok, encoder, args):
    optim = torch.optim.AdamW(encoder.parameters(), lr=args.lr)
    encoder.train()
    log_path = os.path.join(os.path.dirname(args.out), 'coprocess_train.log')
    log = open(log_path, 'w')
    losses = []
    seed = 100_000  # disjoint from eval seeds
    t0 = time.time()
    step = 0
    while step < args.train_steps:
        n = random.choice([8, 10, 12, 14, 16, 18, 20])
        density = random.choice([0.3, 0.5, 0.7, 1.0])
        extra = max(1, int(round(density * n)))
        g = make_graph(n, extra, seed=seed)
        seed += 1
        if g is None:
            continue
        edges_text = render_natural(g, random.Random(seed))
        loss = loss_for_example(llm, tok, encoder, g, edges_text,
                                no_edge_text=args.no_edge_text)
        if loss is None:
            continue
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optim.step()
        losses.append(loss.item())
        step += 1
        if step % args.log_every == 0:
            recent = losses[-args.log_every:]
            msg = (f"[step {step:>5}/{args.train_steps}] "
                   f"loss={np.mean(recent):.4f} "
                   f"min={min(recent):.4f} "
                   f"t={(time.time()-t0):.0f}s")
            print(msg)
            log.write(msg + '\n')
            log.flush()
        if step % args.save_every == 0:
            ckpt_path = os.path.join(os.path.dirname(args.out),
                                     f'coprocess_encoder_step{step}.pt')
            torch.save(encoder.state_dict(), ckpt_path)
    log.close()
    final_ckpt = os.path.join(os.path.dirname(args.out),
                              'coprocess_encoder_final.pt')
    torch.save(encoder.state_dict(), final_ckpt)
    return final_ckpt


def run_learned_mode(llm, tok, encoder, graphs, no_edge_text=False):
    encoder.eval()
    results = []
    for gi, g in enumerate(graphs):
        edges_text = render_natural(g, random.Random(1000 + gi))
        W_t, s_idx, t_idx, path_idx, dist_to_t_t, path_lab, cost = features_for_graph(g)
        if path_lab is None:
            results.append({'label': 'UNPARSED', 'path': None,
                            'true_cost': None, 'claimed_cost': None})
            continue
        with torch.no_grad():
            soft = encoder(W_t, s_idx, t_idx, path_idx, dist_to_t_t)
            soft = soft.to(torch.bfloat16)
        if no_edge_text:
            user = PROMPT_NO_EDGES.format(s=g.s, t=g.t)
        else:
            user = PROMPT_DIRECT.format(edges=edges_text, s=g.s, t=g.t)
        prompt_str = chat_format(tok, user)
        s, _ = gen_and_score(llm, tok, prompt_str, prefix_embeds=soft, g=g)
        results.append(s)
    return results


# ─────────────────────────────────────────────────────────────────────
# Eval driver
# ─────────────────────────────────────────────────────────────────────

def collect_graphs(sizes, densities, n_per_cell, seed_start=0):
    """Same seeds as bench_geodesic.py for apples-to-apples."""
    cells = []
    seed = seed_start
    for n in sizes:
        for density in densities:
            extra = max(1, int(round(density * n)))
            graphs = []
            attempts = 0
            while len(graphs) < n_per_cell and attempts < n_per_cell * 10:
                g = make_graph(n, extra, seed=seed)
                seed += 1
                attempts += 1
                if g is not None:
                    graphs.append(g)
            cells.append({'n': n, 'extra_edges': extra, 'graphs': graphs})
    return cells


def tally_results(results):
    from collections import Counter
    c = Counter(r['label'] for r in results)
    tot = max(1, len(results))
    return {k: c.get(k, 0) / tot for k in
            ['CORRECT', 'DECOY', 'SUBOPTIMAL', 'HALLUCINATED', 'REVISIT', 'UNPARSED']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['text', 'learned'], required=True)
    p.add_argument('--model', default='/workspace/models/qwen3-4b')
    p.add_argument('--n_graphs', type=int, default=30)
    p.add_argument('--sizes', default='10,14,18,22')
    p.add_argument('--cycle_densities', default='0.4,0.8')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--train_steps', type=int, default=3000)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--n_soft', type=int, default=16)
    p.add_argument('--log_every', type=int, default=20)
    p.add_argument('--save_every', type=int, default=500)
    p.add_argument('--encoder_ckpt', default=None,
                   help='if provided in learned mode, skip training and use this checkpoint')
    p.add_argument('--no_edge_text', action='store_true',
                   help='remove edge list from prompt; soft tokens carry all info')
    p.add_argument('--out', default='/workspace/liquid-arc/coprocess_eval.json')
    args = p.parse_args()

    sizes = [int(x) for x in args.sizes.split(',')]
    densities = [float(x) for x in args.cycle_densities.split(',')]

    print("=" * 70)
    print(f"COPROCESS GEODESIC — mode={args.mode}")
    print("=" * 70)

    print("loading LLM...")
    llm, tok = load_llm(args.model)
    print("loaded")

    cells_data = collect_graphs(sizes, densities, args.n_graphs, args.seed)
    print(f"  cells: {len(cells_data)}, total graphs: "
          f"{sum(len(c['graphs']) for c in cells_data)}")

    encoder = None
    if args.mode == 'learned':
        d_llm = llm.get_input_embeddings().embedding_dim
        encoder = GraphEncoder(d_llm=d_llm, n_soft=args.n_soft).cuda()
        print(f"  encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
        if args.encoder_ckpt and os.path.exists(args.encoder_ckpt):
            print(f"  loading encoder from {args.encoder_ckpt}")
            encoder.load_state_dict(torch.load(args.encoder_ckpt))
        else:
            print(f"  training encoder for {args.train_steps} steps...")
            ckpt = train_encoder(llm, tok, encoder, args)
            print(f"  trained, saved {ckpt}")
            encoder.load_state_dict(torch.load(ckpt))

    out_cells = []
    grand_correct, grand_total = 0, 0
    for ci, cell in enumerate(cells_data):
        graphs = cell['graphs']
        print(f"\n--- cell {ci+1}/{len(cells_data)} n={cell['n']} extra={cell['extra_edges']} ({len(graphs)} g) ---")
        t0 = time.time()
        if args.mode == 'text':
            results = run_text_mode(llm, tok, graphs)
        else:
            results = run_learned_mode(llm, tok, encoder, graphs,
                                       no_edge_text=args.no_edge_text)
        dt = time.time() - t0
        tally = tally_results(results)
        out_cells.append({'n': cell['n'], 'extra_edges': cell['extra_edges'],
                          'n_graphs': len(graphs), 'tally': tally,
                          'mean_latency_ms': dt / len(graphs) * 1000})
        from collections import Counter
        grand_correct += Counter(r['label'] for r in results).get('CORRECT', 0)
        grand_total += len(results)
        print(f"  CORRECT={tally['CORRECT']:>5.0%}  "
              f"SUBOPT={tally['SUBOPTIMAL']:>5.0%}  "
              f"HALLUC={tally['HALLUCINATED']:>4.0%}  "
              f"UNPARSED={tally['UNPARSED']:>4.0%}  "
              f"|  {dt/len(graphs)*1000:>5.0f} ms/g")

    with open(args.out, 'w') as f:
        json.dump({'mode': args.mode, 'cells': out_cells, 'config': vars(args)},
                  f, indent=2)
    print("\n" + "=" * 70)
    print(f"OVERALL ({args.mode}): {grand_correct}/{grand_total} = "
          f"{grand_correct / max(1, grand_total):.0%}")
    print("=" * 70)


if __name__ == '__main__':
    main()
