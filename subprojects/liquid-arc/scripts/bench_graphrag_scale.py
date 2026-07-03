"""Benchmark 2 — scale stress test.

Composes all 3 Phase 2 variants (50 interactions each, different entity
names, same topology) into ONE navigator state: 150 interactions, ~370
nodes, ~290 edges. For each query, we measure:

  - Navigator's structural recall when the state contains 3× more
    (mostly irrelevant) structure than in Phase 2 (distractor noise)
  - Hierarchical subgraph selection effectiveness:
      * direct: analyze_graph on the full 370-node state
      * hierarchical: community-select a <=200-node subgraph first

This demonstrates whether the architecture degrades gracefully under
scale and whether the hierarchical layer buys us time savings on large
graphs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from liquid_arc.graph_engine_inference import GraphEngine
from liquid_arc.graph_rag.hierarchical import HierarchicalGraphRAG
from liquid_arc.navigator import GeometricNavigator
from liquid_arc.navigator_patterns import PatternLibrary
from liquid_arc.navigator_state import GeometricState


def _stems(nid: str) -> set:
    return {t[:4] for t in (nid or "").lower().split("_") if t}


def _mentions(answer: str, node_id: str) -> bool:
    import re
    words = re.split(r"[^a-zA-Z0-9]+", (answer or "").lower())
    ans_stems = {w[:4] for w in words if w}
    node_stems = {s for s in _stems(node_id) if len(s) > 2}
    if not node_stems:
        return False
    return len(node_stems & ans_stems) >= max(1, len(node_stems) // 2)


def structural_hit(retrieved_node_ids: List[str],
                   expected_node_ids: List[str]) -> Dict[str, Any]:
    if not expected_node_ids:
        return {"hits": 0, "total": 0, "score": None}
    hits = 0
    present = set(retrieved_node_ids)
    for eid in expected_node_ids:
        if eid in present:
            hits += 1
    return {"hits": hits, "total": len(expected_node_ids),
            "score": hits / len(expected_node_ids)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--sessions_dir", required=True)
    p.add_argument("--out_json", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--state_path", default="/tmp/bench2_state.json")
    p.add_argument("--pattern_path", default="/tmp/bench2_patterns.json")
    p.add_argument("--max_subgraph", type=int, default=200)
    args = p.parse_args()

    for path in (args.state_path, args.pattern_path):
        if os.path.exists(path):
            os.remove(path)

    engine = GraphEngine(args.checkpoint, device=args.device,
                         corrections_log=None)
    state = GeometricState(args.state_path, engine, max_nodes=2048)
    state.reset()
    patterns = PatternLibrary(args.pattern_path)
    patterns.reset()
    nav = GeometricNavigator(
        engine=engine, state=state, extractor=None,
        pattern_library=patterns)

    # ── Compose all 3 variants into a single state ───────────────
    sessions_dir = Path(args.sessions_dir)
    all_variants = []
    for vid in (0, 1, 2):
        with open(sessions_dir / f"variant_{vid}.json") as f:
            all_variants.append(json.load(f))

    t0 = time.time()
    total_interactions = 0
    for v in all_variants:
        for it in v["interactions"]:
            nav.process_interaction(it["text"], pre_extracted=it["fragment"])
            total_interactions += 1
    ingest_s = time.time() - t0
    print(f"[bench2] ingested {total_interactions} interactions from "
          f"3 variants in {ingest_s:.1f}s. nodes={len(state.nodes)} "
          f"edges={len(state.edges)}", flush=True)

    hier = HierarchicalGraphRAG(navigator=nav,
                                 max_subgraph_nodes=args.max_subgraph)

    # ── For each query, measure structural retrieval under scale ──
    # We use each variant's queries but expect variant-specific nodes.
    per_query = []
    for v in all_variants:
        for q in v["queries"]:
            expected = q.get("expected_answer_node_ids", []) or []
            anchors = q.get("anchor_nodes", []) or []

            t_d = time.time()
            direct_result = nav.process_interaction(
                q["text"], pre_extracted={
                    "nodes": [{"id": a, "type": state.nodes[a]["type"],
                               "role": state.nodes[a]["role"]}
                              for a in anchors if a in state.nodes],
                    "edges": []})
            direct_context = [n["id"] for n in
                               direct_result.get("context_nodes", [])]
            direct_s = time.time() - t_d

            t_h = time.time()
            sub = hier.select_subgraph(anchors or [])
            sub_s = time.time() - t_h

            direct_score = structural_hit(direct_context, expected)
            # For hierarchical: how many expected nodes are in the selected
            # subgraph? This is a proxy for "does the subgraph contain
            # what's needed to answer"
            hier_score = structural_hit(sub["nodes"], expected)

            per_query.append({
                "qid": q["qid"],
                "variant_id": v["variant_id"],
                "type": q["type"],
                "n_expected": len(expected),
                "direct_context_size": len(direct_context),
                "direct_structural_score": direct_score["score"],
                "direct_elapsed_s": direct_s,
                "subgraph_size": len(sub["nodes"]),
                "subgraph_capped": sub["coverage_stats"]["capped"],
                "communities_selected": sub["communities_selected"],
                "hierarchical_structural_score": hier_score["score"],
                "hierarchical_elapsed_s": sub_s,
            })
            print(f"  [v{v['variant_id']} {q['qid']:30s} {q['type']:14s}] "
                  f"direct_top10={len(direct_context):3d} "
                  f"subgraph={len(sub['nodes']):3d}/{len(state.nodes)} "
                  f"direct_rec={(direct_score['score'] or 0):.2f} "
                  f"sub_rec={(hier_score['score'] or 0):.2f} "
                  f"d_t={direct_s*1000:5.0f}ms h_t={sub_s*1000:5.0f}ms",
                  flush=True)

    # ── Aggregate by query type ───────────────────────────────────
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for r in per_query:
        by_type.setdefault(r["type"], []).append(r)
    agg = {}
    for t, rows in by_type.items():
        agg[t] = {
            "n": len(rows),
            "direct_mean_score": sum(
                (r["direct_structural_score"] or 0) for r in rows) / max(1, len(rows)),
            "hier_mean_score": sum(
                (r["hierarchical_structural_score"] or 0) for r in rows) / max(1, len(rows)),
            "direct_mean_ms": sum(
                r["direct_elapsed_s"] for r in rows) / max(1, len(rows)) * 1000,
            "hier_mean_ms": sum(
                r["hierarchical_elapsed_s"] for r in rows) / max(1, len(rows)) * 1000,
            "mean_subgraph_size": sum(
                r["subgraph_size"] for r in rows) / max(1, len(rows)),
            "pct_capped": sum(
                1 for r in rows if r["subgraph_capped"]) / max(1, len(rows)),
        }
    summary = {
        "n_interactions": total_interactions,
        "ingest_s": ingest_s,
        "n_nodes": len(state.nodes),
        "n_edges": len(state.edges),
        "n_patterns": len(patterns.patterns),
        "max_subgraph": args.max_subgraph,
        "per_type": agg,
        "per_query": per_query,
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== BENCHMARK 2 — SCALE STRESS ===", flush=True)
    print(f"  interactions:      {total_interactions}", flush=True)
    print(f"  nodes / edges:     {len(state.nodes)} / {len(state.edges)}",
          flush=True)
    print(f"  patterns:          {len(patterns.patterns)}", flush=True)
    for t, s in agg.items():
        print(f"  {t:14s} n={s['n']} direct={s['direct_mean_score']:.2f} "
              f"hier={s['hier_mean_score']:.2f} "
              f"sub_size={s['mean_subgraph_size']:.0f} "
              f"direct={s['direct_mean_ms']:.0f}ms "
              f"hier={s['hier_mean_ms']:.0f}ms", flush=True)
    print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
