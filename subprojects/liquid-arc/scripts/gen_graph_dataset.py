"""Synthetic graph dataset generator for Phase 3 training.

Spec: GRAPH_REASONING_ENGINE_SPEC.md lines 304-318.

Generates ~25K examples across four task families:
  - 10K causal chains (Task A: root cause queries)
  -  5K parallel chains (Task B: connection-check queries)
  -  5K analogy pairs  (Task C: isomorphism check)
  -  5K scoped logic    (Task D: implication check)

Optional (5th): 10K Freebase/Wikidata fragments — not implemented here; the
spec marks them optional and they require external data. The core 25K is
sufficient to train the four output heads.

Output: sharded JSONL files under output_dir/
    causal_chains.jsonl
    parallel_chains.jsonl
    analogy_pairs.jsonl
    scoped_logic.jsonl

Each line is a JSON object with:
    task:          'root_cause' | 'connection_check' | 'analogy' | 'implication_check'
    nodes:         [{id, type (int), role (int)}, ...]
    edges:         [{src, dst, type (int), scope (int, optional)}]
    query:         task-specific dict
    target:        task-specific ground truth
    (for analogy): graph_a, graph_b instead of single graph
"""

import argparse
import json
import os
import random
import sys
from typing import List, Dict, Any

import networkx as nx


# ─────────────────────────────────────────────────────────────────────
# Type vocabularies (deliberately small so clustering is strong)
# ─────────────────────────────────────────────────────────────────────

# Node types encoded as integers for TypeEmbed (n_node_types=32 in embed).
# We use a handful of categorical slots to keep clusters clean.
NODE_TYPES = {
    'event': 0,
    'consequence': 1,
    'state': 2,
    'cause': 3,
    'role': 4,        # scoped-logic: a scope-defining node
    'credential': 5,
    'requirement': 6,
    'prerequisite': 7,
    'trigger': 8,
    'step': 9,
    'outcome': 10,
}

ROLES = {
    'root': 0,
    'intermediate': 1,
    'terminal': 2,
    'scope': 3,
    'query': 4,
    'other': 5,
}

EDGE_TYPES = {
    'causes': 0,
    'requires': 1,
    'precedes': 2,
    'enables': 3,
}


# ─────────────────────────────────────────────────────────────────────
# Task A — causal chain root cause
# ─────────────────────────────────────────────────────────────────────

def gen_causal_chain(rng: random.Random, min_nodes: int = 3,
                     max_nodes: int = 10) -> Dict[str, Any]:
    """Generate a random DAG causal chain, pick a terminal, query its root.

    Structure:
      - Random DAG, typically a tree with 0-2 shortcut edges.
      - Types cycled among {event, consequence, state}.
      - Query: root_cause of a randomly picked terminal (leaf) node.
    """
    n = rng.randint(min_nodes, max_nodes)
    # Random DAG via topological order
    order = list(range(n))
    edges = []
    # Spanning tree (each node i > 0 connects to a random earlier node)
    for i in range(1, n):
        parent = rng.randint(0, i - 1)
        edges.append((parent, i))
    # Add a few shortcut/bypass edges to create branching (0-2 extra)
    n_extra = rng.randint(0, max(1, n // 4))
    tries = 0
    while len(edges) < n - 1 + n_extra and tries < 10:
        u = rng.randint(0, n - 2)
        v = rng.randint(u + 1, n - 1)
        if (u, v) not in edges:
            edges.append((u, v))
        tries += 1

    # Build nx graph to compute roles
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)

    roots = [v for v in g.nodes if g.in_degree(v) == 0]
    leaves = [v for v in g.nodes if g.out_degree(v) == 0]
    if not leaves:
        # Shouldn't happen with our construction, but guard
        return None
    target = rng.choice(leaves)

    # Pick a single root that is an ancestor of target — that's the "true" root_cause.
    # If multiple roots have a path to target, choose the one with the longest path.
    best_root, best_path = None, []
    for r in roots:
        try:
            path = nx.shortest_path(g, r, target)
            if len(path) > len(best_path):
                best_root, best_path = r, path
        except nx.NetworkXNoPath:
            continue
    if best_root is None:
        return None

    # Assign types cyclically from {event, consequence, state}
    type_cycle = [NODE_TYPES['event'], NODE_TYPES['consequence'], NODE_TYPES['state']]
    nodes = []
    for i in range(n):
        role = ROLES['root'] if i in roots and i == best_root else (
               ROLES['terminal'] if i == target else ROLES['intermediate'])
        nodes.append({'id': i, 'type': type_cycle[i % 3], 'role': role})

    edges_out = [{'src': u, 'dst': v, 'type': EDGE_TYPES['causes']} for u, v in edges]

    return {
        'task': 'root_cause',
        'nodes': nodes,
        'edges': edges_out,
        'query': {'type': 'root_cause', 'target': target},
        'target': best_root,
    }


# ─────────────────────────────────────────────────────────────────────
# Task B — parallel chain connection check
# ─────────────────────────────────────────────────────────────────────

def gen_parallel_chains(rng: random.Random,
                        n_chains_range=(2, 4),
                        chain_len_range=(3, 5)) -> Dict[str, Any]:
    """Two or more disjoint chains, interleaved.

    Query: are two randomly-picked nodes connected?
      - If both picked from same chain → connected
      - If picked from different chains → not connected
    Balanced: ~50% positive, ~50% negative.
    """
    n_chains = rng.randint(*n_chains_range)
    chains = []
    node_id = 0
    for c in range(n_chains):
        length = rng.randint(*chain_len_range)
        chain = list(range(node_id, node_id + length))
        chains.append(chain)
        node_id += length
    N = node_id

    # Edges only within chains
    edges = []
    for chain in chains:
        for a, b in zip(chain, chain[1:]):
            edges.append((a, b))

    # Pick query — bias toward 50/50 positive/negative
    if rng.random() < 0.5 and n_chains >= 2:
        # negative: two nodes from different chains
        ca, cb = rng.sample(range(n_chains), 2)
        src = rng.choice(chains[ca])
        dst = rng.choice(chains[cb])
        connected = False
    else:
        # positive: two nodes from same chain
        c = rng.randrange(n_chains)
        src, dst = rng.sample(chains[c], 2)
        connected = True

    # TYPE-DISJOINT CHAINS: partition the type vocabulary across chains so
    # no two chains ever share a type. This guarantees chain-specific initial
    # embeddings and maximizes separation in the ODE attractors — essential
    # for the connection head to learn same-chain vs different-chain.
    all_types = list(NODE_TYPES.values())
    rng.shuffle(all_types)
    per_chain_pool = len(all_types) // max(1, len(chains))
    chain_type_seq = []
    for ci in range(len(chains)):
        start = ci * per_chain_pool
        pool = all_types[start:start + per_chain_pool]
        if not pool:
            # Fallback if we have more chains than types — reuse but shuffled
            pool = all_types[:]
            rng.shuffle(pool)
        chain_type_seq.append([rng.choice(pool) for _ in range(len(chains[ci]))])
    nodes = []
    chain_of = {}
    for ci, chain in enumerate(chains):
        for pos, nid in enumerate(chain):
            chain_of[nid] = ci
    for i in range(N):
        ci = chain_of[i]
        pos_in_chain = chains[ci].index(i)
        ntype = chain_type_seq[ci][pos_in_chain]
        if pos_in_chain == 0:
            role = ROLES['root']
        elif pos_in_chain == len(chains[ci]) - 1:
            role = ROLES['terminal']
        else:
            role = ROLES['intermediate']
        if i == src or i == dst:
            role = ROLES['query']
        nodes.append({'id': i, 'type': ntype, 'role': role,
                      '_chain_id': ci})

    edges_out = [{'src': u, 'dst': v, 'type': EDGE_TYPES['causes']} for u, v in edges]

    return {
        'task': 'connection_check',
        'nodes': nodes,
        'edges': edges_out,
        'query': {'type': 'connection_check', 'src': src, 'dst': dst},
        'target': bool(connected),
    }


# ─────────────────────────────────────────────────────────────────────
# Task C — structural analogy (isomorphism)
# ─────────────────────────────────────────────────────────────────────

def _random_tree_topology(rng: random.Random, n: int):
    """Spanning tree as sequence of (parent, child) edges in topo order."""
    edges = []
    for i in range(1, n):
        parent = rng.randint(0, i - 1)
        edges.append((parent, i))
    return edges


def gen_analogy_pair(rng: random.Random,
                     min_nodes: int = 4,
                     max_nodes: int = 8) -> Dict[str, Any]:
    """Generate a pair of graphs, 50/50 isomorphic/not.

    - Positive: same topology, different node type labels.
    - Negative: different topology (different edge set / node count).
    """
    n = rng.randint(min_nodes, max_nodes)

    if rng.random() < 0.5:
        # Positive: same topology, different labels
        edges = _random_tree_topology(rng, n)
        label_a = [rng.randint(0, 10) for _ in range(n)]
        label_b = [rng.randint(0, 10) for _ in range(n)]
        nodes_a = [{'id': i, 'type': label_a[i], 'role': ROLES['other']}
                   for i in range(n)]
        nodes_b = [{'id': i, 'type': label_b[i], 'role': ROLES['other']}
                   for i in range(n)]
        edges_a = [{'src': u, 'dst': v, 'type': EDGE_TYPES['causes']} for u, v in edges]
        edges_b = [{'src': u, 'dst': v, 'type': EDGE_TYPES['causes']} for u, v in edges]
        return {
            'task': 'analogy',
            'graph_a': {'nodes': nodes_a, 'edges': edges_a},
            'graph_b': {'nodes': nodes_b, 'edges': edges_b},
            'query': {'type': 'analogy'},
            'target': True,
        }
    else:
        # Negative: different topology
        edges_a = _random_tree_topology(rng, n)
        # Generate B with different node count OR different edge set
        if rng.random() < 0.5:
            n_b = rng.randint(max(min_nodes, n - 2), min(max_nodes, n + 2))
            while n_b == n:
                n_b = rng.randint(min_nodes, max_nodes)
            edges_b = _random_tree_topology(rng, n_b)
        else:
            n_b = n
            tries = 0
            while tries < 20:
                edges_b = _random_tree_topology(rng, n_b)
                if sorted(edges_b) != sorted(edges_a):
                    break
                tries += 1
        nodes_a = [{'id': i, 'type': rng.randint(0, 10),
                    'role': ROLES['other']} for i in range(n)]
        nodes_b = [{'id': i, 'type': rng.randint(0, 10),
                    'role': ROLES['other']} for i in range(n_b)]
        edges_a_out = [{'src': u, 'dst': v, 'type': EDGE_TYPES['causes']} for u, v in edges_a]
        edges_b_out = [{'src': u, 'dst': v, 'type': EDGE_TYPES['causes']} for u, v in edges_b]
        return {
            'task': 'analogy',
            'graph_a': {'nodes': nodes_a, 'edges': edges_a_out},
            'graph_b': {'nodes': nodes_b, 'edges': edges_b_out},
            'query': {'type': 'analogy'},
            'target': False,
        }


# ─────────────────────────────────────────────────────────────────────
# Task D — scoped logic
# ─────────────────────────────────────────────────────────────────────

def gen_scoped_logic(rng: random.Random,
                     min_nodes: int = 7,
                     max_nodes: int = 14) -> Dict[str, Any]:
    """Diverse scoped-logic dependency graphs.

    Generator randomizes:
      - Number of scopes (2 or 3)
      - Backbone chain length (2-4 nodes before scope-gated branch point)
      - Number of scope-gated branches per scope (1-3)
      - Which node is premise (any backbone or branch node)
      - Which node is conclusion (any reachable or unreachable node)
      - Positive vs negative example (50/50)
      - Node type labels drawn from the extended vocabulary

    This replaces the previous fixed 8-node topology which produced a head
    that memorized positions rather than learning scope-gated reachability.
    """
    n_scopes = rng.choice([2, 3])
    backbone_len = rng.randint(2, 4)
    branches_per_scope = rng.randint(1, 3)

    # ── Layout node indices ────────────────────────────────────────
    nodes = []
    edges = []
    node_id = 0

    # Scope nodes 0..n_scopes-1
    scope_ids = list(range(n_scopes))
    for s in scope_ids:
        nodes.append({'id': s, 'type': NODE_TYPES['role'], 'role': ROLES['scope']})
    node_id = n_scopes

    # Backbone chain (shared) — each scope connects to backbone[0]
    backbone = list(range(node_id, node_id + backbone_len))
    node_id += backbone_len
    backbone_types = [NODE_TYPES['credential'], NODE_TYPES['requirement']]
    for i, bid in enumerate(backbone):
        nodes.append({
            'id': bid,
            'type': backbone_types[i % len(backbone_types)],
            'role': ROLES['intermediate'],
        })
    # scope -> backbone[0]
    for s in scope_ids:
        edges.append((s, backbone[0], EDGE_TYPES['requires'], None))
    # backbone chain
    for a, b in zip(backbone, backbone[1:]):
        edges.append((a, b, EDGE_TYPES['requires'], None))

    # Fork point: end of backbone
    fork = backbone[-1]

    # Per-scope branch trees
    scope_branch_roots: Dict[int, list] = {s: [] for s in scope_ids}
    for s in scope_ids:
        for _ in range(branches_per_scope):
            bid = node_id
            node_id += 1
            nodes.append({
                'id': bid,
                'type': NODE_TYPES['requirement'],
                'role': ROLES['intermediate'],
            })
            # scope-gated edge from fork to branch root
            edges.append((fork, bid, EDGE_TYPES['requires'], s))
            scope_branch_roots[s].append(bid)
            # Each branch may have a follow-up prereq (50% chance)
            if rng.random() < 0.6 and node_id < max_nodes - 1:
                pid = node_id
                node_id += 1
                nodes.append({
                    'id': pid,
                    'type': NODE_TYPES['prerequisite'],
                    'role': ROLES['terminal'],
                })
                edges.append((bid, pid, EDGE_TYPES['requires'], s))
                scope_branch_roots[s].append(pid)

    # ── Query sampling ─────────────────────────────────────────────
    # Choose a query scope
    use_scope = rng.choice(scope_ids)

    # Build scope-filtered adjacency to compute ground-truth reachability
    filt_adj: Dict[int, list] = {n['id']: [] for n in nodes}
    for u, v, _et, sc in edges:
        if sc is None or sc == use_scope:
            filt_adj[u].append(v)

    def reachable(src, dst):
        if src == dst:
            return True
        seen = {src}
        stack = [src]
        while stack:
            u = stack.pop()
            for w in filt_adj.get(u, []):
                if w == dst:
                    return True
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        return False

    all_nodes = [n['id'] for n in nodes if n['id'] not in scope_ids]

    # 50/50 positive / negative
    want_valid = rng.random() < 0.5
    premise = None
    conclusion = None
    tries = 0
    while tries < 50:
        p = rng.choice(all_nodes)
        c = rng.choice(all_nodes)
        if p == c:
            tries += 1
            continue
        is_reach = reachable(p, c)
        if want_valid and is_reach:
            premise, conclusion = p, c
            break
        if (not want_valid) and (not is_reach):
            premise, conclusion = p, c
            break
        tries += 1
    if premise is None:
        # Couldn't find matching example; fall back to any pair
        premise = rng.choice(all_nodes)
        conclusion = rng.choice([x for x in all_nodes if x != premise])
        want_valid = reachable(premise, conclusion)

    edges_out = []
    for u, v, et, sc in edges:
        e = {'src': u, 'dst': v, 'type': et}
        if sc is not None:
            e['scope'] = sc
        edges_out.append(e)

    return {
        'task': 'implication_check',
        'nodes': nodes,
        'edges': edges_out,
        'query': {
            'type': 'implication_check',
            'premise': premise,
            'conclusion': conclusion,
            'context_scope': use_scope,
        },
        'target': bool(want_valid),
    }


# ─────────────────────────────────────────────────────────────────────
# Dataset writer
# ─────────────────────────────────────────────────────────────────────

def write_jsonl(path: str, records: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='/workspace/liquid-arc/data/graph_engine')
    p.add_argument('--n_causal', type=int, default=10000)
    p.add_argument('--n_parallel', type=int, default=5000)
    p.add_argument('--n_analogy', type=int, default=5000)
    p.add_argument('--n_scoped', type=int, default=5000)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)

    print(f"generating dataset → {args.out_dir}")

    # Causal chains
    print(f"  causal chains: {args.n_causal}")
    recs = []
    tries = 0
    while len(recs) < args.n_causal and tries < args.n_causal * 2:
        r = gen_causal_chain(rng)
        tries += 1
        if r is not None:
            recs.append(r)
    write_jsonl(os.path.join(args.out_dir, 'causal_chains.jsonl'), recs)
    print(f"    wrote {len(recs)} records")

    # Parallel chains
    print(f"  parallel chains: {args.n_parallel}")
    recs = [gen_parallel_chains(rng) for _ in range(args.n_parallel)]
    write_jsonl(os.path.join(args.out_dir, 'parallel_chains.jsonl'), recs)
    print(f"    wrote {len(recs)} records")

    # Analogy pairs
    print(f"  analogy pairs: {args.n_analogy}")
    recs = [gen_analogy_pair(rng) for _ in range(args.n_analogy)]
    write_jsonl(os.path.join(args.out_dir, 'analogy_pairs.jsonl'), recs)
    print(f"    wrote {len(recs)} records")

    # Scoped logic
    print(f"  scoped logic: {args.n_scoped}")
    recs = [gen_scoped_logic(rng) for _ in range(args.n_scoped)]
    write_jsonl(os.path.join(args.out_dir, 'scoped_logic.jsonl'), recs)
    print(f"    wrote {len(recs)} records")

    total = args.n_causal + args.n_parallel + args.n_analogy + args.n_scoped
    print(f"\n  total: {total} examples across 4 task families")


if __name__ == '__main__':
    main()
