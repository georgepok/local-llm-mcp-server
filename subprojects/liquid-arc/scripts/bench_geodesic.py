"""Geodesic benchmark: shortest path on cyclic weighted graphs.

Hypothesis: flat attention at 4B scale hallucinates path structure from
textual ordering. The adversarial-vs-natural ordering gap measures how
much the model is *projecting* a sequential prior onto the latent graph
geometry rather than actually traversing it.

Per graph, two textual conditions over IDENTICAL edges:
  natural:     edges in BFS order from source (no ordering hint)
  adversarial: edges of a DECOY (sub-optimal) path placed first and
               consecutively, optimal-path edges scattered late

Scoring per response:
  CORRECT          path equals one of the true shortest paths
  SUBOPTIMAL       path is valid but heavier than optimum
  DECOY            path equals the planted decoy (the failure mode)
  HALLUCINATED     path uses an edge that doesn't exist
  UNPARSED         couldn't extract a path

Predicted signature: as graph size and cycle count grow, DECOY rate on
adversarial condition grows faster than SUBOPTIMAL rate on natural —
the model is following text order, not graph structure.

Run (Spark):
  python3 scripts/bench_geodesic.py --model /workspace/models/qwen3-4b \
    --n_graphs 30 --sizes 10,14,18 --out bench_geodesic.json

Run (offline generator inspection only):
  python3 scripts/bench_geodesic.py --dry_run --n_graphs 5 --sizes 10
"""

import argparse, heapq, json, random, re, time
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────────────────────────────
# Graph generation
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Graph:
    n: int
    nodes: list                     # ['A','B',...]
    edges: list                     # [(u, v, w)] undirected, u<v lexicographically
    s: str
    t: str
    optimal_path: list              # node sequence
    optimal_cost: int
    decoy_path: list                # planted sub-optimal path
    decoy_cost: int
    n_cycles_independent: int       # |E| - |V| + 1 (cyclomatic complexity)


def _node_label(i: int) -> str:
    # Single letters A-Z, then AA, AB, ... (we stay <= 26 here)
    if i < 26:
        return chr(ord('A') + i)
    raise ValueError("graphs >26 nodes need multi-char labels; not used here")


def _shortest_path(adj: dict, s: str, t: str):
    """Dijkstra. Returns (cost, path)."""
    pq = [(0, s, [s])]
    seen = {}
    while pq:
        c, u, path = heapq.heappop(pq)
        if u in seen and seen[u] <= c:
            continue
        seen[u] = c
        if u == t:
            return c, path
        for v, w in adj[u]:
            if v not in seen or seen[v] > c + w:
                heapq.heappush(pq, (c + w, v, path + [v]))
    return None, None


def _all_simple_paths(adj: dict, s: str, t: str, max_len: int):
    """Yield (cost, path) for every simple s→t path up to max_len nodes."""
    stack = [(s, [s], set([s]), 0)]
    while stack:
        u, path, visited, cost = stack.pop()
        if u == t:
            yield cost, path
            continue
        if len(path) >= max_len:
            continue
        for v, w in adj[u]:
            if v not in visited:
                stack.append((v, path + [v], visited | {v}, cost + w))


def make_graph(n: int, extra_edges: int, seed: int) -> Optional[Graph]:
    """Build a connected weighted graph with ≥1 cycle and a clear decoy.

    Strategy:
      1. Random spanning tree (guarantees connectivity).
      2. Add `extra_edges` chords (creates cycles).
      3. Pick s, t = furthest pair on the tree (long backbone).
      4. Find true shortest path with Dijkstra.
      5. Decoy = shortest path's longer alternative — must differ in ≥2 edges
         and cost strictly more.
    """
    rng = random.Random(seed)
    nodes = [_node_label(i) for i in range(n)]
    edges = []
    edge_set = set()

    # Spanning tree via random parent assignment
    order = nodes[:]
    rng.shuffle(order)
    for i in range(1, n):
        parent = order[rng.randint(0, i - 1)]
        child = order[i]
        u, v = sorted([parent, child])
        w = rng.randint(1, 9)
        edges.append((u, v, w))
        edge_set.add((u, v))

    # Add chord edges (these create cycles)
    tries = 0
    added = 0
    while added < extra_edges and tries < extra_edges * 10:
        u, v = rng.sample(nodes, 2)
        u, v = sorted([u, v])
        if (u, v) in edge_set:
            tries += 1
            continue
        w = rng.randint(1, 9)
        edges.append((u, v, w))
        edge_set.add((u, v))
        added += 1
        tries += 1

    # Build adjacency
    adj = {x: [] for x in nodes}
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))

    # Pick s,t with a long-ish optimal path (≥3 hops) so a decoy can exist
    best = None
    for _ in range(20):
        s, t = rng.sample(nodes, 2)
        cost, path = _shortest_path(adj, s, t)
        if path is not None and cost is not None and len(path) >= 4:
            best = (s, t, cost, path)
            break
    if best is None:
        return None
    s, t, opt_cost, opt_path = best  # opt_cost: int  (guarded above)

    # Find a decoy: enumerate simple paths up to length n, pick one that
    # is strictly heavier and shares < half the edges with the optimum.
    opt_edges = set(tuple(sorted([opt_path[i], opt_path[i + 1]])) for i in range(len(opt_path) - 1))
    candidates = []
    for c, p in _all_simple_paths(adj, s, t, max_len=min(n, len(opt_path) + 4)):
        if c <= opt_cost:
            continue
        p_edges = set(tuple(sorted([p[i], p[i + 1]])) for i in range(len(p) - 1))
        overlap = len(p_edges & opt_edges) / max(1, len(opt_edges))
        if overlap < 0.5 and len(p) >= 3:
            candidates.append((c, p, overlap))
        if len(candidates) > 200:
            break
    if not candidates:
        return None
    # Prefer decoys that are *close* in cost (more deceptive) but distinct
    candidates.sort(key=lambda x: (x[2], x[0] - opt_cost))
    decoy_cost, decoy_path, _ = candidates[0]

    return Graph(
        n=n, nodes=nodes, edges=edges, s=s, t=t,
        optimal_path=opt_path, optimal_cost=opt_cost,
        decoy_path=decoy_path, decoy_cost=decoy_cost,
        n_cycles_independent=len(edges) - n + 1,
    )


# ─────────────────────────────────────────────────────────────────────
# Textual presentation
# ─────────────────────────────────────────────────────────────────────

def _format_edges(edges, order):
    return ", ".join(f"{u}-{v}:{w}" for (u, v, w) in [edges[i] for i in order])


def render_natural(g: Graph, rng: random.Random) -> str:
    """Edges listed in BFS order from s — no ordering signal toward any path."""
    adj_idx = {x: [] for x in g.nodes}
    for i, (u, v, w) in enumerate(g.edges):
        adj_idx[u].append(i)
        adj_idx[v].append(i)
    visited_nodes = {g.s}
    visited_edges = []
    seen_edges = set()
    queue = [g.s]
    while queue:
        u = queue.pop(0)
        eis = adj_idx[u][:]
        rng.shuffle(eis)
        for ei in eis:
            if ei in seen_edges:
                continue
            seen_edges.add(ei)
            visited_edges.append(ei)
            a, b, _ = g.edges[ei]
            other = b if a == u else a
            if other not in visited_nodes:
                visited_nodes.add(other)
                queue.append(other)
    # Append any disconnected leftovers (shouldn't happen — graph is connected)
    for i in range(len(g.edges)):
        if i not in seen_edges:
            visited_edges.append(i)
    return _format_edges(g.edges, visited_edges)


def render_adversarial(g: Graph, rng: random.Random) -> str:
    """Decoy path edges first and consecutive; optimal path edges last and scattered.

    This is the projection trap: a model that follows textual order will
    compose the decoy path before it ever encounters the optimal edges.
    """
    edge_idx = {tuple(sorted([u, v])): i for i, (u, v, w) in enumerate(g.edges)}
    decoy_eis = [edge_idx[tuple(sorted([g.decoy_path[i], g.decoy_path[i + 1]]))]
                 for i in range(len(g.decoy_path) - 1)]
    opt_eis = [edge_idx[tuple(sorted([g.optimal_path[i], g.optimal_path[i + 1]]))]
               for i in range(len(g.optimal_path) - 1)]

    used = set(decoy_eis) | set(opt_eis)
    other_eis = [i for i in range(len(g.edges)) if i not in used]
    rng.shuffle(other_eis)

    # Layout: [decoy in order] [other edges interleaved with optimal edges scattered to the back half]
    half = len(other_eis) // 2
    front_filler, back_filler = other_eis[:half], other_eis[half:]
    # Scatter optimal edges into back half so they don't appear consecutively
    back = back_filler[:]
    if opt_eis:
        rng.shuffle(opt_eis)
        positions = sorted(rng.sample(range(len(back) + len(opt_eis)),
                                      len(opt_eis)))
        for pos, ei in zip(positions, opt_eis):
            back.insert(pos, ei)

    order = decoy_eis + front_filler + back
    return _format_edges(g.edges, order)


PROMPT_COT = """You are given an undirected weighted graph. Each edge \
is written as A-B:W meaning nodes A and B are connected with weight W.

Edges:
{edges}

Find the shortest path from {s} to {t}. Use only edges listed above; do not \
invent edges. You may reason step by step, but your FINAL line MUST be \
exactly:
PATH: {s} -> ... -> {t}  COST: <integer>"""

PROMPT_DIRECT = """You are given an undirected weighted graph. Each edge \
is written as A-B:W meaning nodes A and B are connected with weight W.

Edges:
{edges}

Find the shortest path from {s} to {t}. Use only edges listed above; do not \
invent edges. Do NOT show any reasoning or working. Reply with one single \
line and nothing else:
PATH: {s} -> ... -> {t}  COST: <integer>"""


# ─────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────

# Strict format: "PATH: A -> B ... COST: N"
PATH_RE = re.compile(
    r"PATH[:\s]+([A-Z](?:\s*(?:->|→|-)\s*[A-Z])+)\s*[,;\s]*COST[:\s]+(-?\d+)",
    re.IGNORECASE)
# Loose chain: any A -> B -> C ... sequence
CHAIN_RE = re.compile(r"\b([A-Z](?:\s*(?:->|→)\s*[A-Z]){1,})")
# Cost mentions
COST_RE = re.compile(
    r"(?:COST|TOTAL|WEIGHT|DISTANCE|cost|total|weight|distance)[\s:=]+(-?\d+)")


def parse_response(text: str, s: str, t: str):
    """Extract (path, claimed_cost). Tolerant of CoT preamble.

    Strategy:
      1. If 'PATH: ... COST: N' present, take the LAST such match.
      2. Else find all 'A -> B -> ...' chains; pick the last one whose
         endpoints are (s, t). Pair with the last COST mention near it.
    """
    # 1. Strict format, take last match (after all reasoning)
    matches = list(PATH_RE.finditer(text))
    if matches:
        m = matches[-1]
        nodes = re.split(r"\s*(?:->|→|-)\s*", m.group(1).strip())
        nodes = [n.strip() for n in nodes if n.strip()]
        return nodes, int(m.group(2))

    # 2. Loose chain fallback. Find chains and pick the last s→t chain.
    chains = []
    for cm in CHAIN_RE.finditer(text):
        nodes = re.split(r"\s*(?:->|→)\s*", cm.group(1).strip())
        nodes = [n.strip() for n in nodes if n.strip()]
        if len(nodes) >= 2 and nodes[0] == s and nodes[-1] == t:
            chains.append((cm.start(), cm.end(), nodes))
    if chains:
        path_start, path_end, nodes = chains[-1]
        # Cost: nearest COST mention after the path
        cost = None
        tail = text[path_end:path_end + 200]
        cm = COST_RE.search(tail)
        if cm:
            cost = int(cm.group(1))
        return nodes, cost

    return None, None


def score(g: Graph, response: str) -> dict:
    nodes, claimed_cost = parse_response(response, g.s, g.t)
    out = {"label": "UNPARSED", "claimed_cost": claimed_cost, "true_cost": None,
           "path": nodes}
    if not nodes or len(nodes) < 2 or nodes[0] != g.s or nodes[-1] != g.t:
        return out

    edge_w = {tuple(sorted([u, v])): w for (u, v, w) in g.edges}
    total = 0
    for a, b in zip(nodes, nodes[1:]):
        key = tuple(sorted([a, b]))
        if key not in edge_w:
            out["label"] = "HALLUCINATED"
            return out
        total += edge_w[key]
    if len(set(nodes)) != len(nodes):
        # walks with revisits are technically valid in undirected graphs but
        # not interesting — flag separately
        out["label"] = "REVISIT"
        out["true_cost"] = total
        return out

    out["true_cost"] = total
    if total == g.optimal_cost:
        out["label"] = "CORRECT"
    elif nodes == g.decoy_path:
        out["label"] = "DECOY"
    else:
        out["label"] = "SUBOPTIMAL"
    return out


# ─────────────────────────────────────────────────────────────────────
# LLM driver (only loaded when actually running)
# ─────────────────────────────────────────────────────────────────────

def load_llm(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    llm = AutoModelForCausalLM.from_pretrained(
        model_path, device_map='cuda', torch_dtype=torch.bfloat16,
        trust_remote_code=True)
    llm.eval()
    return llm, tok


def generate(llm, tok, prompt: str, max_new: int = 200) -> str:
    import torch
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
    with torch.no_grad():
        out = llm.generate(**inp, max_new_tokens=max_new, do_sample=False,
                           temperature=1.0, repetition_penalty=1.05)
    txt = tok.decode(out[0][n:], skip_special_tokens=True)
    m = re.search(r'</think>\s*(.*)', txt, flags=re.DOTALL)
    if m and len(m.group(1).strip()) > 5:
        txt = m.group(1).strip()
    return re.sub(r'</?think>', '', txt).strip()


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='/workspace/models/qwen3-4b')
    p.add_argument('--n_graphs', type=int, default=20,
                   help='graphs per (size, cycle_density) cell')
    p.add_argument('--sizes', default='10,14,18',
                   help='comma-sep node counts')
    p.add_argument('--cycle_densities', default='0.4,0.8',
                   help='extra_edges = ceil(density * n)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='bench_geodesic.json')
    p.add_argument('--mode', choices=['cot', 'direct'], default='cot',
                   help='cot: allow reasoning before final line; direct: force one-line answer')
    p.add_argument('--max_new', type=int, default=1500)
    p.add_argument('--dry_run', action='store_true',
                   help='generate + print prompts without LLM calls')
    args = p.parse_args()
    PROMPT = PROMPT_DIRECT if args.mode == 'direct' else PROMPT_COT

    sizes = [int(x) for x in args.sizes.split(',')]
    densities = [float(x) for x in args.cycle_densities.split(',')]

    print("=" * 70)
    print("GEODESIC BENCHMARK — adversarial vs natural ordering on cyclic graphs")
    print("=" * 70)
    print(f"  sizes={sizes}  densities={densities}  n_graphs/cell={args.n_graphs}")
    print(f"  total graphs={len(sizes) * len(densities) * args.n_graphs}")

    llm = tok = None
    if not args.dry_run:
        print(f"  loading {args.model} ...")
        t0 = time.time()
        llm, tok = load_llm(args.model)
        print(f"  loaded in {time.time() - t0:.1f}s")

    cells = []
    seed_counter = args.seed
    for n in sizes:
        for density in densities:
            extra = max(1, int(round(density * n)))
            graphs = []
            attempts = 0
            while len(graphs) < args.n_graphs and attempts < args.n_graphs * 10:
                g = make_graph(n, extra, seed=seed_counter)
                seed_counter += 1
                attempts += 1
                if g is not None:
                    graphs.append(g)
            print(f"\n--- cell n={n} extra_edges={extra} ({len(graphs)} graphs) ---")

            results = {'natural': [], 'adversarial': []}
            for gi, g in enumerate(graphs):
                rng_n = random.Random(1000 + gi)
                rng_a = random.Random(2000 + gi)
                edges_nat = render_natural(g, rng_n)
                edges_adv = render_adversarial(g, rng_a)
                prompt_nat = PROMPT.format(edges=edges_nat, s=g.s, t=g.t)
                prompt_adv = PROMPT.format(edges=edges_adv, s=g.s, t=g.t)

                if args.dry_run:
                    if gi == 0:
                        print(f"\n  [example natural]\n{prompt_nat}")
                        print(f"\n  [example adversarial]\n{prompt_adv}")
                        print(f"\n  truth: optimal={g.optimal_path}({g.optimal_cost})  "
                              f"decoy={g.decoy_path}({g.decoy_cost})")
                    continue

                resp_nat = generate(llm, tok, prompt_nat, max_new=args.max_new)
                resp_adv = generate(llm, tok, prompt_adv, max_new=args.max_new)
                s_nat = score(g, resp_nat)
                s_adv = score(g, resp_adv)
                results['natural'].append(s_nat)
                results['adversarial'].append(s_adv)
                print(f"  g{gi:02d} n={n} cyc={g.n_cycles_independent} "
                      f"opt={g.optimal_cost} decoy={g.decoy_cost}  "
                      f"nat={s_nat['label']:<11} adv={s_adv['label']:<11}")

            if args.dry_run:
                continue

            def tally(rs):
                from collections import Counter
                c = Counter(r['label'] for r in rs)
                tot = max(1, len(rs))
                return {k: c.get(k, 0) / tot for k in
                        ['CORRECT', 'DECOY', 'SUBOPTIMAL', 'HALLUCINATED', 'REVISIT', 'UNPARSED']}

            nat_t = tally(results['natural'])
            adv_t = tally(results['adversarial'])
            cell = {
                'n': n, 'extra_edges': extra, 'n_graphs': len(graphs),
                'natural': nat_t, 'adversarial': adv_t,
                'gap_correct': nat_t['CORRECT'] - adv_t['CORRECT'],
                'gap_decoy': adv_t['DECOY'] - nat_t['DECOY'],
            }
            cells.append(cell)
            print(f"  → natural CORRECT={nat_t['CORRECT']:.0%}  "
                  f"adversarial CORRECT={adv_t['CORRECT']:.0%}  "
                  f"adversarial DECOY={adv_t['DECOY']:.0%}  "
                  f"Δcorrect={cell['gap_correct']:+.0%}")

    if not args.dry_run:
        with open(args.out, 'w') as f:
            json.dump({'cells': cells, 'config': vars(args)}, f, indent=2)
        print(f"\nSaved → {args.out}")

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"{'n':>3} {'cyc':>4} | nat-CORRECT  adv-CORRECT  adv-DECOY  Δcorrect")
        for c in cells:
            print(f"{c['n']:>3} {c['extra_edges']:>4} | "
                  f"{c['natural']['CORRECT']:>10.0%}  "
                  f"{c['adversarial']['CORRECT']:>11.0%}  "
                  f"{c['adversarial']['DECOY']:>9.0%}  "
                  f"{c['gap_correct']:>+8.0%}")


if __name__ == '__main__':
    main()
