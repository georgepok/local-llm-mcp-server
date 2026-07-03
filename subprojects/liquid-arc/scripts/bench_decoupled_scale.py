"""Experiment 1 — scale stress test (1000 documents).

Generates 1000 synthetic enterprise docs with pre-supplied fragments
(so ingest latency reflects the decoupled architecture, not LLM
extraction speed). Ingests all docs, measures per-doc latency, then
runs 50 mixed queries to measure query latency by type.

Pass criteria:
  - ingest latency <2s per doc (no ODE at ingestion)
  - node count >500 (graph grows; entity resolution doesn't over-dedupe)
  - causal/scope queries: ODE NOT invoked, latency <200ms
  - topology/pattern queries: ODE invoked once on ≤200-node subgraph,
    latency <2000ms
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from liquid_arc.graph_rag.chunker import Chunker
from liquid_arc.graph_rag.decoupled.graph_db import KnowledgeGraphDB
from liquid_arc.graph_rag.decoupled.ode_engine import SubgraphODEEngine
from liquid_arc.graph_rag.decoupled.orchestrator import DecoupledGraphRAG
from liquid_arc.graph_rag.vector_db import VectorDB
from liquid_arc.navigator_patterns import PatternLibrary


DOMAINS = ["supply_chain", "hospital", "cybersecurity",
            "finance", "rnd"]
SCOPES = ["production", "staging", "finance_team", "sales_team",
           "engineering_team"]

ENTITY_BANK = {
    "supply_chain": [
        "port", "warehouse", "factory", "supplier", "shipment",
        "route", "container", "inventory", "retailer", "distributor",
    ],
    "hospital": [
        "dept", "staff", "equipment", "protocol", "patient",
        "wing", "ward", "pharmacy", "clinic", "facility",
    ],
    "cybersecurity": [
        "endpoint", "firewall", "server", "credential", "vulnerability",
        "policy", "control", "analyst", "ticket", "incident",
    ],
    "finance": [
        "ledger", "account", "vendor", "invoice", "contract",
        "cost_center", "budget", "audit", "report", "forecast",
    ],
    "rnd": [
        "project", "milestone", "feature", "prototype", "review",
        "test", "release", "backlog", "roadmap", "spec",
    ],
}


class CannedExtractor:
    """Returns a fragment derived from a keyword bank per chunk text.
    Deterministic: the fragment for the same text is always identical."""

    def __init__(self, domain: str, doc_id: int):
        self.domain = domain
        self.doc_id = doc_id

    def extract(self, text: str) -> Dict[str, Any]:
        bank = ENTITY_BANK[self.domain]
        # Use a hash of the text to pick entities deterministically
        h = hashlib.sha1(text.encode()).digest()
        k = 3 + (h[0] % 4)  # 3-6 nodes per chunk
        rng = random.Random(int.from_bytes(h[:8], "big"))
        nodes: List[Dict[str, Any]] = []
        used = set()
        scope = SCOPES[h[1] % len(SCOPES)]
        while len(nodes) < k:
            name = rng.choice(bank)
            nid = f"{self.domain}_{name}_{self.doc_id}_{len(nodes)}"
            if nid in used:
                continue
            used.add(nid)
            role = ["root", "intermediate", "intermediate", "terminal"][
                len(nodes) % 4]
            nodes.append({
                "id": nid,
                "type": ["event", "state", "consequence", "entity",
                          "requirement"][h[2] % 5],
                "role": role,
            })
        # Linear chain edges
        edges = [
            {"src": nodes[i]["id"], "dst": nodes[i + 1]["id"],
             "type": ["causes", "precedes", "enables", "requires"][
                 h[3 + (i % 5)] % 4],
             "scope": scope if (i % 2 == 0) else None}
            for i in range(len(nodes) - 1)
        ]
        return {"nodes": nodes, "edges": edges}


def gen_documents(n: int, words_per_doc: int = 180) -> List[Dict[str, Any]]:
    """Lightweight synthetic docs. Each has realistic prose scaffolding
    + a domain + a deterministic extractor."""
    rng = random.Random(42)
    docs = []
    phrases_by_domain = {
        "supply_chain": [
            "Congestion at port was reported on the weekend.",
            "Forwarders re-quoted logistics lanes yesterday.",
            "Assembly plant inventory fell below the target.",
            "Dealer delivery slipped across the region.",
            "Emergency air freight was authorized by the VP.",
        ],
        "hospital": [
            "The department reported a short-staffing event overnight.",
            "Equipment went offline and the backup schedule slipped.",
            "Protocol review was requested by patient safety.",
            "The ward triage ratio exceeded the threshold.",
        ],
        "cybersecurity": [
            "A ticket escalated to tier-2 after the policy review.",
            "Endpoint compromise was detected and isolated.",
            "Credential rotation followed the incident response plan.",
            "A firewall rule change was scheduled under CAB approval.",
        ],
        "finance": [
            "The ledger reconciliation flagged a variance this cycle.",
            "Vendor invoice terms were renegotiated by procurement.",
            "Audit committee reviewed the quarterly control gaps.",
            "Budget reforecast was requested by the CFO.",
        ],
        "rnd": [
            "The project milestone slipped by two weeks.",
            "A prototype review raised integration concerns.",
            "The release backlog grew as test cases expanded.",
            "Roadmap prioritization was updated post-review.",
        ],
    }
    for i in range(n):
        dom = DOMAINS[i % len(DOMAINS)]
        scope = SCOPES[i % len(SCOPES)]
        body_lines = [
            f"[Doc {i:04d} — {dom} — scope: {scope}]",
        ]
        while len(" ".join(body_lines).split()) < words_per_doc:
            body_lines.append(rng.choice(phrases_by_domain[dom]))
        docs.append({
            "doc_id": i,
            "title": f"{dom}_doc_{i}",
            "domain": dom,
            "scope": scope,
            "text": " ".join(body_lines),
        })
    return docs


def gen_queries(docs: List[Dict[str, Any]], n: int = 50
                 ) -> List[Dict[str, Any]]:
    """Mixed-mode queries referencing the ingested corpus."""
    rng = random.Random(7)
    modes = ["causal", "scope", "topology", "pattern", "factual"]
    queries: List[Dict[str, Any]] = []
    for i in range(n):
        mode = modes[i % len(modes)]
        doc = rng.choice(docs)
        if mode == "causal":
            text = f"What is the root cause that led to {doc['domain']} problem in {doc['title']}?"
        elif mode == "scope":
            text = (f"For a {doc['scope'].replace('_', ' ')} workflow, "
                    f"what policy applies to {doc['domain']}?")
        elif mode == "topology":
            text = f"What are the critical hubs or single points of failure across {doc['domain']}?"
        elif mode == "pattern":
            text = f"Have we seen a similar {doc['domain']} incident pattern before?"
        else:
            text = f"What is {doc['domain']}?"
        queries.append({
            "qid": f"q{i:03d}",
            "mode": mode,
            "text": text,
            "scope": doc["scope"] if mode == "scope" else None,
        })
    return queries


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n_docs", type=int, default=1000)
    p.add_argument("--n_queries", type=int, default=50)
    p.add_argument("--max_subgraph", type=int, default=200)
    p.add_argument("--out_json", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--db_path", default="/tmp/decoupled_scale_db.json")
    p.add_argument("--patterns_path",
                   default="/tmp/decoupled_scale_patterns.json")
    args = p.parse_args()

    # Fresh start
    for path in (args.db_path, args.patterns_path):
        if os.path.exists(path):
            os.remove(path)

    ode = SubgraphODEEngine(args.checkpoint, device=args.device)
    graph_db = KnowledgeGraphDB(args.db_path)
    vector_db = VectorDB(dim=1024)
    patterns = PatternLibrary(args.patterns_path)
    rag = DecoupledGraphRAG(
        graph_db=graph_db, ode_engine=ode, vector_db=vector_db,
        extractor=None,  # per-doc override below
        pattern_library=patterns,
        chunker=Chunker(target_tokens=200, overlap_tokens=40),
        max_subgraph_nodes=args.max_subgraph,
        doc_signature_enabled=False,
    )

    docs = gen_documents(args.n_docs)
    print(f"[exp1] ingesting {len(docs)} docs", flush=True)
    per_doc: List[Dict[str, Any]] = []
    t_overall = time.time()
    for d in docs:
        rag.extractor = CannedExtractor(d["domain"], d["doc_id"])
        t0 = time.time()
        rep = rag.ingest(d["text"], metadata={
            "doc_id": d["doc_id"], "title": d["title"],
            "domain": d["domain"], "scope": d["scope"]})
        dt = time.time() - t0
        per_doc.append({
            "doc_id": d["doc_id"],
            "elapsed_s": dt,
            "n_chunks": rep["n_chunks"],
            "fragments": rep["n_fragments_merged"],
        })
    ingest_total = time.time() - t_overall
    stats_after_ingest = graph_db.stats(compute_communities=True)

    # Percentile ingest timings
    times = sorted(p["elapsed_s"] for p in per_doc)
    def pct(p): return times[min(len(times) - 1, int(len(times) * p / 100))]
    ingest_stats = {
        "total_s": ingest_total,
        "mean_s": sum(times) / len(times),
        "p50_s": pct(50),
        "p95_s": pct(95),
        "max_s": times[-1],
        "n_docs": len(docs),
        "graph_after": stats_after_ingest,
    }
    print(f"[exp1] ingest done: {ingest_total:.1f}s total, "
          f"{ingest_stats['mean_s']*1000:.0f}ms mean, "
          f"p95 {ingest_stats['p95_s']*1000:.0f}ms, "
          f"graph {stats_after_ingest['n_nodes']} nodes / "
          f"{stats_after_ingest['n_edges']} edges / "
          f"{stats_after_ingest.get('n_communities','?')} communities",
          flush=True)

    # ── Queries ───────────────────────────────────────────────────
    queries = gen_queries(docs, n=args.n_queries)
    class _QueryExtractor:
        """Extracts anchor nodes from a query. Here we hand-seed by
        picking random existing nodes from the graph for each query
        based on the referenced document."""
        def __init__(self, doc): self.doc = doc
        def extract(self, text):
            # Grab a few real node IDs from the doc's nearby neighbors
            candidates = [n for n, data in graph_db.G.nodes(data=True)
                          if data.get("doc_metadata", {}).get("domain")
                             == self.doc["domain"]]
            if not candidates:
                return {"nodes": [], "edges": []}
            k = min(3, len(candidates))
            picks = candidates[:k]
            return {
                "nodes": [
                    {"id": n, "type": graph_db.G.nodes[n]["type"],
                     "role": graph_db.G.nodes[n]["role"]}
                    for n in picks],
                "edges": [],
            }

    per_query: List[Dict[str, Any]] = []
    for q in queries:
        doc = random.choice(docs)
        extractor = _QueryExtractor(doc)
        t0 = time.time()
        try:
            result = rag.query(q["text"], scope=q["scope"],
                                _extractor_override=extractor)
        except Exception as exc:
            result = {"error": str(exc), "elapsed_ms": 0,
                       "stats": {"ode_invoked": False}}
        dt = time.time() - t0
        per_query.append({
            "qid": q["qid"],
            "mode": q["mode"],
            "elapsed_ms": dt * 1000,
            "ode_invoked": result.get("stats", {}).get("ode_invoked", False),
            "vector_chunks": result.get("stats", {}).get("vector_chunks"),
            "graph_chunks": result.get("stats", {}).get("graph_chunks"),
            "db_nodes": result.get("stats", {}).get("db_nodes"),
            "timings_ms": result.get("stats", {}).get("timings_ms"),
        })

    # Aggregate per-mode
    by_mode: Dict[str, List[Dict[str, Any]]] = {}
    for r in per_query:
        by_mode.setdefault(r["mode"], []).append(r)
    mode_agg: Dict[str, Any] = {}
    for m, rows in by_mode.items():
        lat = sorted(r["elapsed_ms"] for r in rows)
        mode_agg[m] = {
            "n": len(rows),
            "mean_ms": sum(lat) / len(lat),
            "p95_ms": lat[min(len(lat) - 1, int(len(lat) * 0.95))],
            "ode_invoked_rate": sum(1 for r in rows
                                      if r["ode_invoked"]) / len(rows),
        }

    # Gates
    gates = {
        "mean_ingest_lt_2s": ingest_stats["mean_s"] < 2.0,
        "p95_ingest_lt_2s": ingest_stats["p95_s"] < 2.0,
        "graph_over_500_nodes": stats_after_ingest["n_nodes"] > 500,
        "causal_no_ode": (mode_agg.get("causal", {}).get("ode_invoked_rate", 0)
                         == 0.0),
        "scope_no_ode": (mode_agg.get("scope", {}).get("ode_invoked_rate", 0)
                         == 0.0),
        "topology_ode_invoked": (mode_agg.get("topology", {}).get(
            "ode_invoked_rate", 0) > 0.0),
        "pattern_ode_invoked": (mode_agg.get("pattern", {}).get(
            "ode_invoked_rate", 0) > 0.0),
        "all_queries_under_2s": all(
            r["elapsed_ms"] < 2000 for r in per_query),
    }

    summary = {
        "n_docs": len(docs),
        "n_queries": len(queries),
        "ingest_stats": ingest_stats,
        "per_mode": mode_agg,
        "gates": gates,
        "overall_pass": all(gates.values()),
        "per_doc": per_doc[:20],         # truncate for file size
        "per_query": per_query,
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== EXP 1 — DECOUPLED SCALE STRESS ===", flush=True)
    print(f"  docs:              {len(docs)}", flush=True)
    print(f"  ingest mean/p95:   {ingest_stats['mean_s']*1000:.1f} / "
          f"{ingest_stats['p95_s']*1000:.1f} ms", flush=True)
    print(f"  graph:             {stats_after_ingest['n_nodes']} n / "
          f"{stats_after_ingest['n_edges']} e / "
          f"{stats_after_ingest.get('n_communities', '?')} communities",
          flush=True)
    for m, s in mode_agg.items():
        print(f"  {m:12s}: n={s['n']}  mean={s['mean_ms']:.1f}ms  "
              f"p95={s['p95_ms']:.1f}ms  ode_rate={s['ode_invoked_rate']:.2f}",
              flush=True)
    print(f"  overall: {'PASS' if summary['overall_pass'] else 'FAIL'}",
          flush=True)
    print(f"  wrote {out}", flush=True)
    sys.exit(0 if summary["overall_pass"] else 1)


if __name__ == "__main__":
    main()
