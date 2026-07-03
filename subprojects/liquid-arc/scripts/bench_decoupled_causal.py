"""Experiment 2 — causal chain BFS vs ODE.

Plant 50 known causal chains (length 3-8) across 500 docs. Test 50
root-cause queries. Compare:
  A — Decoupled graph-DB BFS                 (no ODE)
  B — Monolithic full-ODE via GraphEngine    (analyze_graph on the whole graph)
  C — ODE on extracted neighborhood subgraph

Pass criteria:
  A matches ground truth on ≥95% of queries.
  A is ≥10× faster than B.
  C doesn't add accuracy over A for pure causal queries.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from liquid_arc.graph_engine_inference import GraphEngine
from liquid_arc.graph_rag.decoupled.graph_db import KnowledgeGraphDB


DOMAINS = ["finance", "hospital", "cyber", "supply", "rnd"]


def plant_chains(n_chains: int = 50,
                 length_range=(3, 8),
                 seed: int = 17) -> List[Dict[str, Any]]:
    """Return a list of causal chains. Each chain: list of nodes from root
    to terminal. Node IDs are unique across chains."""
    rng = random.Random(seed)
    chains = []
    for ci in range(n_chains):
        dom = DOMAINS[ci % len(DOMAINS)]
        L = rng.randint(*length_range)
        nodes = [f"{dom}_c{ci:03d}_n{i:02d}" for i in range(L)]
        chains.append({
            "chain_id": ci, "domain": dom,
            "nodes": nodes,
            "root": nodes[0], "terminal": nodes[-1],
            "length": L,
        })
    return chains


def interleave_into_docs(chains: List[Dict[str, Any]],
                         n_docs: int = 500) -> List[Dict[str, Any]]:
    """Each doc contains one or two edges from a chain, plus filler nodes.
    Returns fragments keyed to doc index so they can be ingested in any order.
    """
    rng = random.Random(23)
    docs: List[Dict[str, Any]] = []
    # Distribute chain edges across docs
    edges_per_chain: List[List[Dict[str, Any]]] = []
    for c in chains:
        edges = []
        for i in range(len(c["nodes"]) - 1):
            edges.append({
                "src": c["nodes"][i], "dst": c["nodes"][i + 1],
                "type": "causes",
                "chain_id": c["chain_id"],
            })
        edges_per_chain.append(edges)
    # Flatten and shuffle
    all_edges = [(c_i, e) for c_i, edges in enumerate(edges_per_chain)
                 for e in edges]
    rng.shuffle(all_edges)
    # Split into docs
    per_doc = max(1, len(all_edges) // n_docs)
    for di in range(n_docs):
        start = di * per_doc
        end = start + per_doc + (1 if di < len(all_edges) % n_docs else 0)
        edges_slice = all_edges[start:end]
        # Build fragment
        nodes: Dict[str, Dict[str, Any]] = {}
        edges: List[Dict[str, Any]] = []
        for c_i, e in edges_slice:
            for nid in (e["src"], e["dst"]):
                if nid not in nodes:
                    # Role: root if no predecessor, terminal if no successor
                    is_root = nid == chains[c_i]["root"]
                    is_term = nid == chains[c_i]["terminal"]
                    role = "root" if is_root else (
                        "terminal" if is_term else "intermediate")
                    nodes[nid] = {
                        "id": nid,
                        "type": "cause" if is_root else (
                            "consequence" if is_term else "event"),
                        "role": role,
                    }
            edges.append({
                "src": e["src"], "dst": e["dst"], "type": "causes"})
        docs.append({
            "doc_id": di,
            "fragment": {"nodes": list(nodes.values()), "edges": edges},
        })
    return docs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out_json", required=True)
    p.add_argument("--n_chains", type=int, default=50)
    p.add_argument("--n_docs", type=int, default=500)
    p.add_argument("--db_path", default="/tmp/decoupled_causal_db.json")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if os.path.exists(args.db_path):
        os.remove(args.db_path)

    chains = plant_chains(n_chains=args.n_chains)
    docs = interleave_into_docs(chains, n_docs=args.n_docs)

    db = KnowledgeGraphDB(args.db_path)
    t_in = time.time()
    for d in docs:
        db.add_fragment(d["fragment"], autosave=False)
    db._save()
    ingest_s = time.time() - t_in
    stats = db.stats(compute_communities=True)
    print(f"[exp2] ingested {len(docs)} docs ({len(chains)} chains) "
          f"in {ingest_s:.1f}s: {stats['n_nodes']} nodes / "
          f"{stats['n_edges']} edges / {stats['n_communities']} communities",
          flush=True)

    engine = GraphEngine(args.checkpoint, device=args.device,
                         corrections_log=None)

    # Build full-graph JSON for condition B once (re-used per query).
    full_nodes = [
        {"id": n, "type": data["type"], "role": data["role"]}
        for n, data in db.G.nodes(data=True)
    ]
    full_edges = [
        {"src": u, "dst": v, "type": data.get("type", "related_to")}
        for u, v, data in db.G.edges(data=True)
    ]
    full_graph_json = json.dumps({"nodes": full_nodes, "edges": full_edges})
    print(f"[exp2] full graph JSON size {len(full_graph_json)/1024:.1f}KB",
          flush=True)

    per_query = []
    for c in chains:
        target = c["terminal"]
        # Condition A: graph-DB BFS
        t0 = time.time()
        bfs = db.trace_causal_chain(target)
        a_ms = (time.time() - t0) * 1000
        a_root = bfs["root"]
        a_correct = a_root == c["root"]

        # Condition B: monolithic full-graph ODE analyze
        t0 = time.time()
        try:
            raw = engine.analyze_graph(
                full_graph_json,
                json.dumps({"type": "root_cause", "target": target}))
            b_result = json.loads(raw)
            b_root = b_result.get("root_cause")
        except Exception as exc:
            b_root = f"error:{exc}"
        b_ms = (time.time() - t0) * 1000
        b_correct = b_root == c["root"]

        # Condition C: ODE on extracted neighborhood
        t0 = time.time()
        hood = db.get_neighbors([target], hops=c["length"] + 1,
                                 direction="backward") | {target}
        sub = db.extract_subgraph(hood, max_nodes=200)
        try:
            raw = engine.analyze_graph(
                json.dumps(sub),
                json.dumps({"type": "root_cause", "target": target}))
            c_result = json.loads(raw)
            c_root = c_result.get("root_cause")
        except Exception as exc:
            c_root = f"error:{exc}"
        c_ms = (time.time() - t0) * 1000
        c_correct = c_root == c["root"]

        per_query.append({
            "chain_id": c["chain_id"],
            "length": c["length"],
            "target": target,
            "expected_root": c["root"],
            "A_root": a_root, "A_correct": a_correct, "A_ms": a_ms,
            "B_root": b_root, "B_correct": b_correct, "B_ms": b_ms,
            "C_root": c_root, "C_correct": c_correct, "C_ms": c_ms,
        })
        print(f"  chain{c['chain_id']:03d} L={c['length']} "
              f"A={'✓' if a_correct else '✗'}/{a_ms:5.1f}ms "
              f"B={'✓' if b_correct else '✗'}/{b_ms:6.1f}ms "
              f"C={'✓' if c_correct else '✗'}/{c_ms:6.1f}ms",
              flush=True)

    def rate(key): return sum(1 for r in per_query if r[key]) / len(per_query)
    def mean_ms(key): return sum(r[key] for r in per_query) / len(per_query)
    agg = {
        "A_accuracy": rate("A_correct"),
        "B_accuracy": rate("B_correct"),
        "C_accuracy": rate("C_correct"),
        "A_mean_ms": mean_ms("A_ms"),
        "B_mean_ms": mean_ms("B_ms"),
        "C_mean_ms": mean_ms("C_ms"),
        "A_speedup_vs_B": mean_ms("B_ms") / max(1e-6, mean_ms("A_ms")),
    }
    gates = {
        "A_accuracy_ge_0_95": agg["A_accuracy"] >= 0.95,
        "A_at_least_10x_faster_than_B": agg["A_speedup_vs_B"] >= 10.0,
    }
    summary = {
        "n_chains": args.n_chains,
        "n_docs": args.n_docs,
        "graph_stats": stats,
        "aggregate": agg,
        "gates": gates,
        "overall_pass": all(gates.values()),
        "per_query": per_query,
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== EXP 2 — CAUSAL BFS vs ODE ===", flush=True)
    print(f"  accuracy  A (BFS)={agg['A_accuracy']:.2f}  "
          f"B (full ODE)={agg['B_accuracy']:.2f}  "
          f"C (subgraph ODE)={agg['C_accuracy']:.2f}", flush=True)
    print(f"  latency   A={agg['A_mean_ms']:.2f}ms  "
          f"B={agg['B_mean_ms']:.1f}ms  "
          f"C={agg['C_mean_ms']:.1f}ms", flush=True)
    print(f"  A vs B speedup: {agg['A_speedup_vs_B']:.1f}×", flush=True)
    print(f"  overall: {'PASS' if summary['overall_pass'] else 'FAIL'}",
          flush=True)
    print(f"  wrote {out}", flush=True)
    sys.exit(0 if summary["overall_pass"] else 1)


if __name__ == "__main__":
    main()
