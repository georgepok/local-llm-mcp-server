import json, os, random, time, sys
sys.path.insert(0, "/workspace/liquid-arc")

# ---- 1) Monkey-patch KnowledgeGraphDB with the candidate function ----
from liquid_arc.graph_rag.decoupled import graph_db as gdb_module
from liquid_arc.graph_rag.decoupled.graph_db import KnowledgeGraphDB

CANDIDATE_CODE = """
import networkx as nx

def query_temporal(self, query_nodes, k):
    if not query_nodes:
        return []
    
    # Compute mean last_seen of query nodes
    query_last_seen = [self.G.nodes[n]["last_seen"] for n in query_nodes if "last_seen" in self.G.nodes[n]]
    if not query_last_seen:
        return []
    q_mean = sum(query_last_seen) / len(query_last_seen)
    
    # Find candidate nodes (not in query_nodes)
    candidates = [n for n in self.G.nodes() if n not in query_nodes and "last_seen" in self.G.nodes[n]]
    if not candidates:
        return []
    
    # Compute temporal scores
    temporal_scores = []
    max_delta = 0.0
    for n in candidates:
        delta = abs(self.G.nodes[n]["last_seen"] - q_mean)
        max_delta = max(max_delta, delta)
    
    for n in candidates:
        delta = abs(self.G.nodes[n]["last_seen"] - q_mean)
        temporal_score = 1.0 - (delta / max_delta) if max_delta > 0 else 1.0
        temporal_scores.append((n, temporal_score))
    
    # Compute causal scores using shortest path on undirected graph
    G_undirected = self.G.to_undirected()
    causal_scores = {}
    for n in candidates:
        min_dist = float('inf')
        for q in query_nodes:
            if nx.has_path(G_undirected, q, n):
                dist = nx.shortest_path_length(G_undirected, q, n)
                min_dist = min(min_dist, dist)
        if min_dist == float('inf'):
            causal_scores[n] = 0.0
        else:
            causal_scores[n] = 1.0 / (1.0 + min_dist)
    
    # Combine scores
    results = []
    for n, temporal_score in temporal_scores:
        causal_score = causal_scores[n]
        combined_score = 0.5 * temporal_score + 0.5 * causal_score
        results.append({
            "id": n,
            "type": self.G.nodes[n]["type"],
            "role": self.G.nodes[n]["role"],
            "temporal_score": temporal_score,
            "causal_score": causal_score,
            "combined_score": combined_score,
            "source": "temporal"
        })
    
    # Sort by combined_score descending and return top-k
    results.sort(key=lambda x: x["combined_score"], reverse=True)
    return results[:k]

"""

# Exec candidate into a scratch namespace inheriting the module's imports.
_ns = dict(gdb_module.__dict__)
exec(CANDIDATE_CODE, _ns)

# Patch the function onto the class as a new method.
assert "query_temporal" in _ns, "candidate did not define query_temporal"
KnowledgeGraphDB.query_temporal = _ns["query_temporal"]

# ---- 2) Build test graph ----
# 30 nodes across 3 causal chains, with timestamps spread over a week.
random.seed(42)
db = KnowledgeGraphDB("/tmp/stage1_db.json")
db.clear()
NOW = 1_700_000_000.0
HOUR = 3600.0
frag_nodes, frag_edges = [], []

# Chain A: events separated by 1 hour, starting NOW
for i in range(10):
    frag_nodes.append({"id": f"A_{i}", "type": "event", "role":
                       "root" if i==0 else ("terminal" if i==9 else "intermediate")})
    if i > 0:
        frag_edges.append({"src": f"A_{i-1}", "dst": f"A_{i}", "type": "causes"})

# Chain B: events separated by 1 day, starting NOW - 3 days
for i in range(10):
    frag_nodes.append({"id": f"B_{i}", "type": "state", "role":
                       "root" if i==0 else ("terminal" if i==9 else "intermediate")})
    if i > 0:
        frag_edges.append({"src": f"B_{i-1}", "dst": f"B_{i}", "type": "precedes"})

# Chain C: events separated by 6 hours, starting NOW - 1 day
for i in range(10):
    frag_nodes.append({"id": f"C_{i}", "type": "consequence", "role":
                       "root" if i==0 else ("terminal" if i==9 else "intermediate")})
    if i > 0:
        frag_edges.append({"src": f"C_{i-1}", "dst": f"C_{i}", "type": "enables"})

db.add_fragment({"nodes": frag_nodes, "edges": frag_edges}, autosave=False)

# Rewrite last_seen timestamps to our controlled values
for i in range(10):
    db.G.nodes[f"A_{i}"]["last_seen"] = NOW - (9 - i) * HOUR              # 0..9h ago
    db.G.nodes[f"B_{i}"]["last_seen"] = NOW - 3 * 24 * HOUR - (9 - i) * 24 * HOUR  # 3-12 days ago
    db.G.nodes[f"C_{i}"]["last_seen"] = NOW - 24 * HOUR - (9 - i) * 6 * HOUR       # 1-3.25 days ago

# ---- 3) Shape check ----
result_shape = {"shape_ok": False, "ordering_correct": 0.0,
                 "beats_baseline": 0.0, "no_regression": 0.0,
                 "errors": []}

try:
    out = db.query_temporal(["A_0"], k=5)
    assert isinstance(out, list), "query_temporal must return list"
    assert len(out) <= 5, "must respect k"
    if out:
        first = out[0]
        assert isinstance(first, dict), "each item must be dict"
        required = {"id", "type", "role", "temporal_score",
                     "causal_score", "combined_score", "source"}
        missing = required - set(first.keys())
        assert not missing, f"missing keys: {missing}"
        assert first["source"] == "temporal", "source must be 'temporal'"
        # Check sorted by combined_score descending
        scores = [r["combined_score"] for r in out]
        assert scores == sorted(scores, reverse=True), "must sort by combined_score desc"
    result_shape["shape_ok"] = True
except Exception as e:
    result_shape["errors"].append(f"shape: {e}")

# ---- 4) Ordering correctness ----
# For query [A_0], expected ordering (top-5) should include close-in-chain
# or temporally-close nodes. A_1..A_4 should dominate (causal near + temporal close).
try:
    out = db.query_temporal(["A_0"], k=5)
    returned_ids = [r["id"] for r in out]
    # Expected: at least 3 of top-5 should be in chain A (most temporally close
    # AND causally closest).
    chain_A_in_top5 = sum(1 for nid in returned_ids if nid.startswith("A_"))
    result_shape["ordering_correct"] = min(1.0, chain_A_in_top5 / 3)
except Exception as e:
    result_shape["errors"].append(f"ordering: {e}")

# ---- 5) Beats recency-only baseline ----
# A time-sensitive query: query for A_5 should find temporally-close nodes.
# The baseline sorts only by |last_seen - query.last_seen|, ignoring chain.
# Temporal mode (with causal component) should do BETTER on chain membership.
def recency_baseline(query_nodes, k):
    q_ls = [db.G.nodes[q]["last_seen"] for q in query_nodes if q in db.G]
    if not q_ls:
        return []
    q_mean = sum(q_ls) / len(q_ls)
    scored = []
    for nid, data in db.G.nodes(data=True):
        if nid in query_nodes:
            continue
        d = abs(data["last_seen"] - q_mean)
        scored.append((nid, -d))
    scored.sort(key=lambda x: -x[1])
    return [nid for nid, _ in scored[:k]]

try:
    # Evaluate how many top-5 results are in the SAME CHAIN as the query.
    # Temporal mode should pick its chain; recency might pick cross-chain
    # nodes that happen to have close timestamps.
    wins = 0
    total = 0
    for seed in ("A_5", "B_5", "C_5", "A_0", "C_0"):
        t_out = db.query_temporal([seed], k=5)
        t_ids = [r["id"] for r in t_out]
        r_ids = recency_baseline([seed], k=5)
        chain = seed[0]
        t_hits = sum(1 for n in t_ids if n.startswith(chain))
        r_hits = sum(1 for n in r_ids if n.startswith(chain))
        total += 1
        if t_hits > r_hits:
            wins += 1
        elif t_hits == r_hits:
            wins += 0.5
    result_shape["beats_baseline"] = wins / total
except Exception as e:
    result_shape["errors"].append(f"baseline: {e}")

# ---- 6) No regression on existing KnowledgeGraphDB methods ----
try:
    # Sanity: existing graph-DB methods still function after monkey-patch.
    reach = db.get_reachable("A_0", max_hops=10)
    assert isinstance(reach, set), "get_reachable must return set"
    assert "A_1" in reach and "A_5" in reach, (
        f"get_reachable regressed — expected A_1 and A_5 in reach, got {sorted(reach)[:10]}")
    sub = db.extract_subgraph(["A_0", "A_1", "A_2"], max_nodes=10)
    assert isinstance(sub, dict) and "nodes" in sub and "edges" in sub
    chain = db.trace_causal_chain("A_9")
    assert chain.get("root") == "A_0", (
        f"trace_causal_chain regressed — expected root A_0, got {chain.get('root')}")
    result_shape["no_regression"] = 1.0
except Exception as e:
    result_shape["errors"].append(f"regression: {e}")

# ---- 7) Compute reward ----
reward = (0.25 * (1.0 if result_shape["shape_ok"] else 0.0)
          + 0.25 * result_shape["ordering_correct"]
          + 0.30 * result_shape["beats_baseline"]
          + 0.20 * result_shape["no_regression"])

result = {**result_shape, "reward": reward}
print("RESULT_JSON: " + json.dumps(result))
