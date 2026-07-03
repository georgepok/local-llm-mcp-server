"""Experiment 3 — scope filtering at scale.

5 scopes × 100 scoped edges across 500 docs. 30 scoped queries.
Measures precision@5 and latency for graph-DB scope filtering.

Pass criteria:
  A (graph-DB scope filter): precision@5 ≥ 0.95, latency < 50ms.
  B (vector-only, no scope awareness): precision@5 < 0.50.
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

from liquid_arc.graph_rag.chunker import Chunker
from liquid_arc.graph_rag.decoupled.graph_db import KnowledgeGraphDB
from liquid_arc.graph_rag.decoupled.orchestrator import DecoupledGraphRAG, _ShimState
from liquid_arc.graph_rag.entity_resolver import EntityResolver
from liquid_arc.graph_rag.vector_db import VectorDB


SCOPES = ["production", "staging", "finance_team", "sales_team",
          "engineering_team"]
TOPICS = ["deployment_approval", "data_access", "expense_approval",
           "change_approval", "code_review", "security_review",
           "audit_logging", "incident_response", "vendor_contracts",
           "password_rotation"]
SCOPE_PROXIES = {
    "production": ["an SRE rotation", "the live-service tier",
                   "a customer-facing deploy"],
    "staging": ["a QA environment", "a pre-production rollout",
                 "a pre-release sandbox"],
    "finance_team": ["the accounting function",
                       "the controller organization",
                       "a fiscal reporting workflow"],
    "sales_team": ["a quota-carrying team",
                    "a customer-facing commercial workflow",
                    "a pipeline owner"],
    "engineering_team": ["a software development team",
                           "an internal platform group",
                           "a backend engineering workflow"],
}


def gen_docs(n_per_scope_topic: int = 10) -> List[Dict[str, Any]]:
    """10 topics × 5 scopes × n_per_scope_topic docs. Each doc has the
    topic+scope+authority fragment. At n=10 that's 500 docs."""
    rng = random.Random(42)
    docs: List[Dict[str, Any]] = []
    for scope in SCOPES:
        for topic in TOPICS:
            authority = f"{scope}_{topic}_authority"
            for i in range(n_per_scope_topic):
                filler = rng.choice([
                    "The review committee met last Tuesday.",
                    "The runbook was refreshed this quarter.",
                    "The escalation path was clarified.",
                    "Documentation was updated by the policy team.",
                ])
                text = (f"[Policy — scope: {scope.replace('_', ' ')}] "
                        f"{topic.replace('_', ' ').capitalize()} is governed "
                        f"by a matrix. For {scope.replace('_', ' ')}, the "
                        f"authorized rule is {authority}. {filler}")
                fragment = {
                    "nodes": [
                        {"id": scope, "type": "role", "role": "scope"},
                        {"id": topic, "type": "requirement",
                         "role": "intermediate"},
                        {"id": authority, "type": "credential",
                         "role": "terminal"},
                    ],
                    "edges": [
                        {"src": topic, "dst": authority,
                         "type": "requires", "scope": scope},
                    ],
                }
                docs.append({
                    "doc_id": f"{scope}_{topic}_{i:02d}",
                    "scope": scope, "topic": topic, "authority": authority,
                    "text": text, "fragment": fragment,
                })
    return docs


def gen_queries(n_per_scope: int = 6) -> List[Dict[str, Any]]:
    qs = []
    rng = random.Random(7)
    for scope in SCOPES:
        proxies = SCOPE_PROXIES[scope]
        for i in range(n_per_scope):
            topic = TOPICS[i % len(TOPICS)]
            proxy = proxies[i % len(proxies)]
            qs.append({
                "qid": f"q_{scope}_{topic}",
                "scope": scope, "topic": topic,
                "text": (f"For {proxy}, what is the authorized rule for "
                         f"{topic.replace('_', ' ')}?"),
            })
    return qs


class CannedExtractor:
    def __init__(self, docs_by_id):
        self.docs_by_id = docs_by_id
        self.current_doc = None

    def extract(self, text: str) -> Dict[str, Any]:
        return (self.current_doc or {}).get(
            "fragment", {"nodes": [], "edges": []})


class QueryExtractor:
    """For scope queries we seed the query with the topic node."""

    def __init__(self, topic: str, scope: str):
        self.topic = topic
        self.scope = scope

    def extract(self, text: str) -> Dict[str, Any]:
        return {
            "nodes": [{"id": self.topic, "type": "requirement",
                        "role": "intermediate"}],
            "edges": [],
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="unused here")
    p.add_argument("--out_json", required=True)
    p.add_argument("--db_path", default="/tmp/decoupled_scope_db.json")
    p.add_argument("--n_per_scope_topic", type=int, default=10)
    p.add_argument("--n_per_scope_query", type=int, default=6)
    args = p.parse_args()
    if os.path.exists(args.db_path):
        os.remove(args.db_path)

    docs = gen_docs(n_per_scope_topic=args.n_per_scope_topic)
    queries = gen_queries(n_per_scope=args.n_per_scope_query)

    db = KnowledgeGraphDB(args.db_path)
    vdb = VectorDB(dim=1024)

    # Build a minimal extractor holder — we'll switch its "current_doc"
    # per ingest call.
    doc_map = {d["doc_id"]: d for d in docs}
    canned = CannedExtractor(doc_map)
    # Use a strict entity resolver — the default 0.6 stem-Jaccard is too
    # aggressive for IDs like "finance_team_X_authority" vs
    # "engineering_team_X_authority" (Jaccard=0.67) which are distinct
    # entities by design in this benchmark.
    strict_resolver = EntityResolver(
        _ShimState(db), stem_jaccard_threshold=0.85)
    rag = DecoupledGraphRAG(
        graph_db=db, ode_engine=None, vector_db=vdb,
        extractor=canned, pattern_library=None,
        chunker=Chunker(target_tokens=200, overlap_tokens=40),
        resolver=strict_resolver,
    )

    print(f"[exp3] ingesting {len(docs)} policy docs", flush=True)
    t0 = time.time()
    for d in docs:
        canned.current_doc = d
        rag.ingest(d["text"], metadata={
            "scope": d["scope"], "topic": d["topic"],
            "doc_id": d["doc_id"]})
    ingest_s = time.time() - t0
    stats = db.stats(compute_communities=False)
    print(f"[exp3] ingest: {ingest_s:.1f}s, nodes={stats['n_nodes']} "
          f"edges={stats['n_edges']}", flush=True)

    print(f"[exp3] running {len(queries)} scoped queries", flush=True)
    per_query = []
    for q in queries:
        qext = QueryExtractor(q["topic"], q["scope"])

        # Condition A: DecoupledGraphRAG with scope filter
        t0 = time.time()
        result_a = rag.query(q["text"], scope=q["scope"], k_vector=5,
                              _extractor_override=qext)
        a_ms = (time.time() - t0) * 1000
        a_top5 = result_a["chunks"][:5]
        a_in_scope = sum(
            1 for c in a_top5
            if (c.get("metadata") or {}).get("scope") == q["scope"]
            or (c.get("metadata") or {}).get("doc_metadata", {}).get("scope")
                 == q["scope"]
        )
        a_precision = a_in_scope / max(1, len(a_top5))

        # Condition B: vector-only, no scope awareness
        t0 = time.time()
        b_top5 = vdb.query(q["text"], k=5)
        b_ms = (time.time() - t0) * 1000
        b_in_scope = sum(
            1 for c in b_top5
            if (c.get("metadata") or {}).get("scope") == q["scope"]
        )
        b_precision = b_in_scope / max(1, len(b_top5))

        per_query.append({
            "qid": q["qid"], "scope": q["scope"], "topic": q["topic"],
            "A_precision": a_precision, "A_ms": a_ms,
            "B_precision": b_precision, "B_ms": b_ms,
            "A_top5_scopes": [
                (c.get("metadata") or {}).get("scope")
                or (c.get("metadata") or {}).get(
                    "doc_metadata", {}).get("scope")
                for c in a_top5],
            "B_top5_scopes": [
                (c.get("metadata") or {}).get("scope") for c in b_top5],
        })
        print(f"  [{q['qid']:45s}] A={a_precision:.2f}/{a_ms:5.1f}ms "
              f"B={b_precision:.2f}/{b_ms:5.1f}ms", flush=True)

    a_mean_p = sum(r["A_precision"] for r in per_query) / len(per_query)
    a_mean_ms = sum(r["A_ms"] for r in per_query) / len(per_query)
    b_mean_p = sum(r["B_precision"] for r in per_query) / len(per_query)
    b_mean_ms = sum(r["B_ms"] for r in per_query) / len(per_query)
    gates = {
        "A_precision_ge_0_95": a_mean_p >= 0.95,
        "A_ms_lt_50": a_mean_ms < 50.0,
        "B_precision_lt_0_50": b_mean_p < 0.50,
    }
    summary = {
        "n_docs": len(docs),
        "n_queries": len(queries),
        "graph_stats": stats,
        "A_mean_precision": a_mean_p,
        "A_mean_ms": a_mean_ms,
        "B_mean_precision": b_mean_p,
        "B_mean_ms": b_mean_ms,
        "gates": gates,
        "overall_pass": all(gates.values()),
        "per_query": per_query,
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== EXP 3 — SCOPE FILTER AT SCALE ===", flush=True)
    print(f"  docs:           {len(docs)}", flush=True)
    print(f"  queries:        {len(queries)}", flush=True)
    print(f"  graph:          {stats['n_nodes']} nodes / {stats['n_edges']} edges",
          flush=True)
    print(f"  A precision@5:  {a_mean_p:.3f}  latency {a_mean_ms:.1f}ms",
          flush=True)
    print(f"  B precision@5:  {b_mean_p:.3f}  latency {b_mean_ms:.1f}ms",
          flush=True)
    print(f"  overall: {'PASS' if summary['overall_pass'] else 'FAIL'}",
          flush=True)
    print(f"  wrote {out}", flush=True)
    sys.exit(0 if summary["overall_pass"] else 1)


if __name__ == "__main__":
    main()
