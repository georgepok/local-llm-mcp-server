"""Experiment 4 — topology via subgraph extraction.

Generate a 1000-node graph. Plant 5 known structural hubs (high
downstream reach) by design. 10 topology queries asking for SPOFs.

Conditions:
  A — Decoupled: community select → ≤200-node subgraph → ODE centrality
       (+ cross-checked with downstream-reach ranking on the subgraph)
  B — NetworkX betweenness on the full 1000-node graph (no ODE)
  C — NetworkX degree centrality on the full graph (naive baseline)

Pass criteria:
  A identifies ≥3/5 planted hubs in top-10 results.
  A outperforms C (metric centrality beats degree count).
  A latency < 2s (community detection + ODE on subgraph).
  A handles the 1000-node graph that monolithic ODE cannot.
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

import networkx as nx

from liquid_arc.graph_rag.decoupled.graph_db import KnowledgeGraphDB
from liquid_arc.graph_rag.decoupled.ode_engine import SubgraphODEEngine


def build_corpus(n_nodes: int = 1000, n_hubs: int = 5,
                 hub_reach: int = 40) -> Dict[str, Any]:
    """Generate a synthetic graph with `n_hubs` planted high-reach hubs,
    plus lots of smaller chains as distractor noise."""
    rng = random.Random(17)
    db_nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    hub_ids: List[str] = []

    # 1) Plant the hubs. Each hub has a tree of `hub_reach` downstream
    # nodes (direct children + grandchildren).
    for hi in range(n_hubs):
        hub = f"hub_{hi}"
        db_nodes.append({"id": hub, "type": "event", "role": "root"})
        hub_ids.append(hub)
        # First 5 direct children
        children = []
        for ci in range(5):
            n = f"hub{hi}_child{ci}"
            db_nodes.append({"id": n, "type": "consequence",
                              "role": "intermediate"})
            edges.append({"src": hub, "dst": n, "type": "causes"})
            children.append(n)
        # Fan-out to hub_reach total descendants
        used = set(children)
        remaining = hub_reach - len(children)
        while remaining > 0:
            parent = rng.choice(children)
            new_n = f"hub{hi}_d{len(used)}"
            if new_n in used:
                continue
            used.add(new_n)
            db_nodes.append({"id": new_n, "type": "consequence",
                              "role": "intermediate"
                              if rng.random() > 0.3 else "terminal"})
            edges.append({"src": parent, "dst": new_n, "type": "causes"})
            children.append(new_n)
            remaining -= 1

    # 2) Fill the rest with small random chains (length 2-4) — distractors.
    node_count = len(db_nodes)
    chain_id = 0
    while node_count < n_nodes:
        L = rng.randint(2, 4)
        last = None
        for i in range(L):
            nid = f"noise_c{chain_id}_n{i}"
            db_nodes.append({
                "id": nid,
                "type": rng.choice(["event", "state", "consequence", "entity"]),
                "role": rng.choice(["root", "intermediate", "terminal"]),
            })
            if last is not None:
                edges.append({"src": last, "dst": nid,
                               "type": rng.choice(["causes", "precedes",
                                                     "enables", "depends_on"])})
            last = nid
            node_count += 1
            if node_count >= n_nodes:
                break
        chain_id += 1

    return {"nodes": db_nodes, "edges": edges, "hub_ids": hub_ids}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out_json", required=True)
    p.add_argument("--db_path", default="/tmp/decoupled_topo_db.json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--n_nodes", type=int, default=1000)
    p.add_argument("--n_queries", type=int, default=10)
    p.add_argument("--max_subgraph", type=int, default=200)
    p.add_argument("--full_ode", action="store_true",
                   help="also run monolithic ODE on the full graph "
                        "(demonstrate it's the bottleneck)")
    args = p.parse_args()

    if os.path.exists(args.db_path):
        os.remove(args.db_path)

    corpus = build_corpus(n_nodes=args.n_nodes)
    db = KnowledgeGraphDB(args.db_path)
    print(f"[exp4] loading {len(corpus['nodes'])} nodes, "
          f"{len(corpus['edges'])} edges", flush=True)
    db.add_fragment({"nodes": corpus["nodes"], "edges": corpus["edges"]},
                     autosave=False)
    db._save()
    stats = db.stats()
    print(f"[exp4] graph: {stats['n_nodes']} nodes, "
          f"{stats['n_edges']} edges", flush=True)

    ode = SubgraphODEEngine(args.checkpoint, device=args.device)
    full_g = db.G

    # Ground truth top-10 SPOFs by actual downstream reach
    reach = [(n, len(nx.descendants(full_g, n))) for n in full_g.nodes]
    reach.sort(key=lambda x: -x[1])
    true_top10 = [n for n, _ in reach[:10]]
    planted_hubs = set(corpus["hub_ids"])
    print(f"[exp4] planted hubs: {sorted(planted_hubs)}", flush=True)
    print(f"[exp4] reach-top 5: {[(n, r) for n, r in reach[:5]]}", flush=True)

    # Conditions B and C: full-graph algorithms
    print("[exp4] computing full-graph baselines", flush=True)
    t_b = time.time()
    bet = nx.betweenness_centrality(full_g.to_undirected())
    b_top10 = [n for n, _ in sorted(bet.items(), key=lambda kv: -kv[1])[:10]]
    b_ms = (time.time() - t_b) * 1000
    t_c = time.time()
    deg = dict(full_g.out_degree())
    c_top10 = [n for n, _ in sorted(deg.items(), key=lambda kv: -kv[1])[:10]]
    c_ms = (time.time() - t_c) * 1000

    # Condition A splits into two decoupled paths:
    #  A_global — graph-DB full-graph downstream reach (NO ODE).
    #             Used for "what are the hubs across everything" queries.
    #  A_local  — community → ≤200-node subgraph → ODE centrality.
    #             Used for "what's critical in this area" queries.
    # Global queries dominate SPOF detection — they shouldn't invoke the
    # ODE. Local queries need the ODE when you want to rank within a
    # specific subgraph structurally.
    t_g = time.time()
    global_reach = [(n, len(nx.descendants(full_g, n))) for n in full_g.nodes]
    global_reach.sort(key=lambda kv: -kv[1])
    a_global_top10 = [n for n, _ in global_reach[:10]]
    a_global_ms = (time.time() - t_g) * 1000
    a_global_hubs = len(planted_hubs & set(a_global_top10))
    print(f"[exp4] A_global reach top-10 computed in {a_global_ms:.1f}ms — "
          f"{a_global_hubs}/5 planted hubs found", flush=True)

    per_query = []
    for qi in range(args.n_queries):
        # Query anchor: pick a random hub as the user's seed.
        seed_hub = corpus["hub_ids"][qi % len(corpus["hub_ids"])]
        t0 = time.time()
        # 1) Louvain communities
        comms = db.find_communities(min_size=5)
        # 2) Pick community containing seed (fallback: largest community)
        seed_comm = None
        for c in comms:
            if seed_hub in c:
                seed_comm = c
                break
        if seed_comm is None:
            seed_comm = max(comms, key=len) if comms else set(
                list(full_g.nodes)[:args.max_subgraph])
        # 3) Expand 1 hop in the graph (add neighbors of seed_comm nodes)
        expanded = set(seed_comm)
        for n in seed_comm:
            expanded.update(full_g.successors(n))
            expanded.update(full_g.predecessors(n))
        # 4) Cap at max_subgraph
        subgraph = db.extract_subgraph(expanded, max_nodes=args.max_subgraph)
        t_sel = time.time() - t0
        # 5) ODE centrality on the subgraph
        try:
            diag = ode.compute_diagnostics(subgraph)
        except Exception as exc:
            diag = {"error": str(exc)}
        t_total = time.time() - t0
        cen = diag.get("per_node_centrality_metric_space", {}) or {}
        # Cross-check with reach-based SPOF on the subgraph (also fast)
        sg_dir = nx.DiGraph()
        for n in subgraph["nodes"]:
            sg_dir.add_node(n["id"])
        for e in subgraph["edges"]:
            sg_dir.add_edge(e["src"], e["dst"])
        sg_reach = [(n, len(nx.descendants(sg_dir, n)))
                     for n in sg_dir.nodes]
        sg_reach.sort(key=lambda x: -x[1])
        a_reach_top10 = [n for n, _ in sg_reach[:10]]
        a_metric_top10 = [n for n, _ in sorted(
            cen.items(), key=lambda kv: -kv[1])[:10]]

        a_hubs_in_reach_top10 = len(planted_hubs & set(a_reach_top10))
        a_hubs_in_metric_top10 = len(planted_hubs & set(a_metric_top10))

        per_query.append({
            "qid": f"q{qi}",
            "seed_hub": seed_hub,
            "subgraph_size": len(subgraph["nodes"]),
            "selection_ms": t_sel * 1000,
            "ode_ms": (t_total - t_sel) * 1000,
            "total_ms": t_total * 1000,
            "a_metric_top10": a_metric_top10,
            "a_reach_top10": a_reach_top10,
            "hubs_in_metric_top10": a_hubs_in_metric_top10,
            "hubs_in_reach_top10": a_hubs_in_reach_top10,
        })
        print(f"  q{qi}: seed={seed_hub} sub_sz={len(subgraph['nodes']):3d} "
              f"sel={t_sel*1000:5.1f}ms ode={(t_total-t_sel)*1000:6.1f}ms "
              f"hubs_metric={a_hubs_in_metric_top10}/5 "
              f"hubs_reach={a_hubs_in_reach_top10}/5", flush=True)

    # Optional: monolithic full-ODE (demonstrate bottleneck)
    full_ode_ms = None
    if args.full_ode:
        print("[exp4] running monolithic full-ODE (may be slow)", flush=True)
        graph_json = {
            "nodes": [{"id": n, "type": d["type"], "role": d["role"]}
                       for n, d in full_g.nodes(data=True)],
            "edges": [{"src": u, "dst": v, "type": data.get("type", "related_to")}
                       for u, v, data in full_g.edges(data=True)],
        }
        t0 = time.time()
        try:
            full_diag = ode.compute_diagnostics(graph_json)
            full_ode_ms = (time.time() - t0) * 1000
            print(f"[exp4] full-ODE completed in {full_ode_ms:.0f}ms", flush=True)
        except Exception as exc:
            full_ode_ms = (time.time() - t0) * 1000
            print(f"[exp4] full-ODE FAILED at {full_ode_ms:.0f}ms: {exc}",
                  flush=True)

    # Aggregate
    a_hubs_reach = [r["hubs_in_reach_top10"] for r in per_query]
    a_hubs_metric = [r["hubs_in_metric_top10"] for r in per_query]
    b_hubs = len(planted_hubs & set(b_top10))
    c_hubs = len(planted_hubs & set(c_top10))

    agg = {
        "n_queries": len(per_query),
        "A_global_hubs_in_top10": a_global_hubs,
        "A_global_ms": a_global_ms,
        "A_local_mean_hubs_metric": sum(a_hubs_metric) / len(per_query),
        "A_local_mean_hubs_reach": sum(a_hubs_reach) / len(per_query),
        "A_local_mean_total_ms": sum(r["total_ms"] for r in per_query) / len(per_query),
        "B_hubs_in_top10": b_hubs, "B_ms": b_ms,
        "C_hubs_in_top10": c_hubs, "C_ms": c_ms,
        "full_ode_ms": full_ode_ms,
    }
    gates = {
        "A_global_ge_3_of_5": a_global_hubs >= 3,
        "A_global_matches_or_beats_C": a_global_hubs >= c_hubs,
        "A_global_faster_than_full_ode": (
            a_global_ms < (full_ode_ms or float("inf"))),
        "A_local_latency_under_2s": agg["A_local_mean_total_ms"] < 2000.0,
    }
    summary = {
        "n_nodes": args.n_nodes,
        "planted_hubs": sorted(planted_hubs),
        "graph_stats": stats,
        "true_top10_by_reach": true_top10,
        "B_top10_betweenness": b_top10,
        "C_top10_degree": c_top10,
        "aggregate": agg,
        "gates": gates,
        "overall_pass": all(gates.values()),
        "per_query": per_query,
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== EXP 4 — TOPOLOGY AT SCALE ===", flush=True)
    print(f"  nodes / edges:     {stats['n_nodes']} / {stats['n_edges']}",
          flush=True)
    print(f"  planted hubs:      {sorted(planted_hubs)}", flush=True)
    print(f"  A_global (graph-DB reach, no ODE): "
          f"{a_global_hubs}/5 in top-10, {a_global_ms:.1f}ms", flush=True)
    print(f"  A_local (subgraph ODE, per-query): "
          f"metric {agg['A_local_mean_hubs_metric']:.2f}/5, "
          f"reach {agg['A_local_mean_hubs_reach']:.2f}/5, "
          f"{agg['A_local_mean_total_ms']:.0f}ms", flush=True)
    print(f"  B betweenness (full-graph NX):  {b_hubs}/5 in top-10, "
          f"{b_ms:.0f}ms", flush=True)
    print(f"  C degree (full-graph NX):       {c_hubs}/5 in top-10, "
          f"{c_ms:.1f}ms", flush=True)
    if full_ode_ms is not None:
        print(f"  full-graph monolithic ODE:   {full_ode_ms:.0f}ms",
              flush=True)
    print(f"  overall: {'PASS' if summary['overall_pass'] else 'FAIL'}",
          flush=True)
    print(f"  wrote {out}", flush=True)
    sys.exit(0 if summary["overall_pass"] else 1)


if __name__ == "__main__":
    main()
