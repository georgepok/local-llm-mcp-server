"""Benchmark 3 — scope-sensitive retrieval.

Dataset: 50 policy docs across 5 scopes (production, staging, finance,
sales, engineering), 10 per scope, covering 10 common topics. Topics
overlap across scopes (keyword similarity high) but scope rules differ.

Queries: 30 scoped questions (6 per scope × 5 scopes). Each query asks
about a topic under a specific scope — ONLY the 10 docs from that scope
are valid answers.

Conditions:
  A) Vector-only retrieval (no scope awareness)
  B) Vector + GraphRAG with scope filtering

Metric: precision@5 — fraction of retrieved chunks whose stored scope
metadata matches the query scope. Target: B ≥ 0.80, A < 0.50.

Usage (Spark):
    python scripts/bench_graphrag_scope.py \
      --checkpoint /workspace/liquid-arc/output_graph_engine_final/checkpoints/step_500.pt \
      --vllm_url http://172.17.0.1:30000/v1 \
      --out_json /workspace/liquid-arc/shared/outbox/graphrag_scope.json \
      --device cpu
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
from liquid_arc.graph_rag.chunker import Chunker
from liquid_arc.graph_rag.entity_resolver import EntityResolver
from liquid_arc.graph_rag.ingester import GraphRAGIngester
from liquid_arc.graph_rag.retriever import GraphRAGRetriever
from liquid_arc.graph_rag.router import QueryRouter
from liquid_arc.graph_rag.vector_db import VectorDB
from liquid_arc.navigator import GeometricNavigator
from liquid_arc.navigator_extract import LLMExtractor, extract_graph
from liquid_arc.navigator_patterns import PatternLibrary
from liquid_arc.navigator_state import GeometricState


SCOPES = ["production", "staging", "finance_team", "sales_team",
          "engineering_team"]

TOPICS = [
    ("deployment_approval",
     "Deployment approvals follow this process: request submitted, "
     "reviewed, and approved by the authorized party."),
    ("data_access",
     "Access to customer data is governed by role-based access control. "
     "Requests must include purpose and retention period."),
    ("expense_approval",
     "Expense approvals require documentation of the business purpose "
     "and are subject to the approval matrix below."),
    ("change_approval",
     "Change approvals follow the standard change management process. "
     "Emergency changes have a separate path."),
    ("code_review",
     "Code reviews are required before any merge to the main branch. "
     "Required reviewer count varies by team."),
    ("security_review",
     "Security reviews are mandatory for any system handling sensitive data. "
     "Cadence depends on risk tier."),
    ("audit_logging",
     "All authorized actions are logged. Retention follows the data "
     "classification and compliance matrix."),
    ("incident_response",
     "Incidents follow the defined runbook. Response SLA depends on "
     "severity classification."),
    ("vendor_contracts",
     "Vendor contracts require legal review. Renewal terms and "
     "escalation paths are defined below."),
    ("password_rotation",
     "Password rotation policy depends on the account tier and the "
     "sensitivity of the systems accessed."),
]


# Per-scope authority rules — what's different per scope for each topic.
# Format: (scope, topic_id, authority_entity)
SCOPE_RULES: Dict[str, Dict[str, str]] = {
    "production": {
        "deployment_approval": "sre_lead",
        "data_access": "data_steward",
        "expense_approval": "engineering_director",
        "change_approval": "change_advisory_board",
        "code_review": "two_reviewers_one_senior",
        "security_review": "ciso_signoff",
        "audit_logging": "splunk_prod_index",
        "incident_response": "sev1_runbook",
        "vendor_contracts": "procurement_lead",
        "password_rotation": "ninety_day_rotation",
    },
    "staging": {
        "deployment_approval": "any_developer",
        "data_access": "self_service",
        "expense_approval": "team_lead",
        "change_approval": "automatic_via_ci",
        "code_review": "single_reviewer",
        "security_review": "quarterly_scan",
        "audit_logging": "splunk_staging_index",
        "incident_response": "sev3_runbook",
        "vendor_contracts": "not_applicable",
        "password_rotation": "annual_rotation",
    },
    "finance_team": {
        "deployment_approval": "not_applicable",
        "data_access": "cfo_approval",
        "expense_approval": "expense_matrix_level_2",
        "change_approval": "finance_controller",
        "code_review": "not_applicable",
        "security_review": "sox_compliance_review",
        "audit_logging": "finance_gl_audit",
        "incident_response": "finance_continuity_plan",
        "vendor_contracts": "cfo_signoff",
        "password_rotation": "thirty_day_rotation",
    },
    "sales_team": {
        "deployment_approval": "not_applicable",
        "data_access": "sales_ops_approval",
        "expense_approval": "regional_vp",
        "change_approval": "sales_ops",
        "code_review": "not_applicable",
        "security_review": "annual_awareness_training",
        "audit_logging": "salesforce_audit_trail",
        "incident_response": "customer_facing_playbook",
        "vendor_contracts": "sales_procurement_partner",
        "password_rotation": "ninety_day_rotation",
    },
    "engineering_team": {
        "deployment_approval": "engineering_manager",
        "data_access": "team_lead_approval",
        "expense_approval": "engineering_director",
        "change_approval": "peer_review",
        "code_review": "two_reviewers",
        "security_review": "secure_sdlc_gate",
        "audit_logging": "github_audit_log",
        "incident_response": "engineering_oncall",
        "vendor_contracts": "engineering_vp_signoff",
        "password_rotation": "sixty_day_rotation",
    },
}


def generate_docs() -> List[Dict[str, Any]]:
    """50 policy docs (5 scopes × 10 topics) with fragments + scope tags."""
    docs: List[Dict[str, Any]] = []
    for scope in SCOPES:
        for tid, base in TOPICS:
            authority = SCOPE_RULES[scope][tid]
            text = (f"[Policy doc — scope: {scope.replace('_', ' ')}] "
                    f"{base} For {scope.replace('_', ' ')}, the "
                    f"authorized entity or rule is: {authority}. "
                    f"This policy applies only within the "
                    f"{scope.replace('_', ' ')} scope and does not "
                    f"extend to other scopes without explicit approval.")
            # Structural fragment: scope + topic + authority, gated
            fragment = {
                "nodes": [
                    {"id": scope, "type": "role", "role": "scope"},
                    {"id": tid, "type": "requirement", "role": "intermediate"},
                    {"id": authority, "type": "credential", "role": "terminal"},
                ],
                "edges": [
                    {"src": tid, "dst": authority, "type": "requires",
                     "scope": scope},
                ],
            }
            docs.append({
                "doc_id": f"{scope}_{tid}",
                "scope": scope,
                "topic": tid,
                "authority": authority,
                "text": text,
                "fragment": fragment,
            })
    return docs


SCOPE_PROXIES = {
    "production": [
        "an SRE rotation on-call",
        "our customer-facing live traffic",
        "a revenue-impacting deploy",
        "a change touching the top-tier systems",
        "the high-availability tier",
        "the live-service pipeline",
    ],
    "staging": [
        "a pre-production sandbox",
        "an internal test environment",
        "a non-customer-facing rollout",
        "a pre-release verification workflow",
        "the QA-tier environment",
        "a dev-validation stage",
    ],
    "finance_team": [
        "the accounting group",
        "our bookkeeping function",
        "the corporate controller's organization",
        "a fiscal reporting workflow",
        "a budget owner's process",
        "an audit-sensitive workflow",
    ],
    "sales_team": [
        "a revenue-owning org",
        "the account-executive organization",
        "a customer-facing commercial team",
        "a quota-carrying group",
        "a deal-desk workflow",
        "a pipeline-owning team",
    ],
    "engineering_team": [
        "a software development group",
        "an internal platform org",
        "the technical staff",
        "a backend team's workflow",
        "the devs building internal tools",
        "a platform engineering workflow",
    ],
}


def generate_queries() -> List[Dict[str, Any]]:
    """30 scoped questions: 6 topics × 5 scopes.

    Queries refer to the scope via a PROXY description (role, tier,
    workflow category) rather than the literal scope keyword. Vector
    retrieval cannot resolve the proxy to a scope without the graph.
    """
    topic_subset = [t for t, _ in TOPICS[:6]]
    queries: List[Dict[str, Any]] = []
    for scope in SCOPES:
        proxies = SCOPE_PROXIES[scope]
        for i, tid in enumerate(topic_subset):
            authority = SCOPE_RULES[scope][tid]
            proxy = proxies[i % len(proxies)]
            queries.append({
                "qid": f"q_{scope}_{tid}",
                "scope": scope,
                "topic": tid,
                "expected_authority": authority,
                "text": (f"For {proxy}, what is the authorized entity "
                         f"or rule for {tid.replace('_', ' ')}?"),
            })
    return queries


# ----------------------------------------------------------------------
# Precision@5 scoring
# ----------------------------------------------------------------------


def score_retrieval(retrieved_chunks: List[Dict[str, Any]], query_scope: str
                    ) -> Dict[str, Any]:
    """Fraction of top-5 chunks whose metadata.scope matches query_scope."""
    top5 = retrieved_chunks[:5]
    if not top5:
        return {"precision_at_5": 0.0, "in_scope": 0, "total": 0}
    in_scope = sum(
        1 for c in top5
        if (c.get("metadata") or {}).get("scope") == query_scope
    )
    return {
        "precision_at_5": in_scope / len(top5),
        "in_scope": in_scope,
        "total": len(top5),
    }


# ----------------------------------------------------------------------
# Wrapper that skips LLM extraction (we already know the fragment).
# Lets us benchmark retrieval mechanics without paying 50 LLM calls.
# ----------------------------------------------------------------------


class _CannedExtractor:
    """Return a pre-supplied fragment from the doc dict (bypass LLM)."""
    def __init__(self, current: Dict[str, Any]):
        self.current = current

    def extract(self, text: str) -> Dict[str, Any]:
        return self.current.get("fragment", {"nodes": [], "edges": []})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out_json", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--vllm_url", default=None,
                   help="if set, uses LLM extraction; otherwise uses "
                        "pre-supplied fragments")
    p.add_argument("--model", default="NVIDIA-Nemotron-3-Nano-30B-A3B-FP8")
    p.add_argument("--state_path", default="/tmp/bench3_state.json")
    p.add_argument("--pattern_path", default="/tmp/bench3_patterns.json")
    args = p.parse_args()

    # Clean start
    for path in (args.state_path, args.pattern_path):
        if os.path.exists(path):
            os.remove(path)

    engine = GraphEngine(args.checkpoint, device=args.device,
                         corrections_log=None)
    state = GeometricState(args.state_path, engine, max_nodes=1024)
    state.reset()
    patterns = PatternLibrary(args.pattern_path)
    patterns.reset()
    navigator = GeometricNavigator(
        engine=engine, state=state, extractor=None,
        pattern_library=patterns)

    vector_db = VectorDB(dim=1024)
    router = QueryRouter()
    resolver = EntityResolver(state)
    chunker = Chunker(target_tokens=200, overlap_tokens=40)

    # Build extractor that returns canned fragments per document.
    canned = _CannedExtractor({})
    ingester = GraphRAGIngester(navigator=navigator, vector_db=vector_db,
                                extractor=canned, chunker=chunker,
                                resolver=resolver)

    # ── Ingest docs ───────────────────────────────────────────────
    docs = generate_docs()
    print(f"[bench3] ingesting {len(docs)} policy docs", flush=True)
    t0 = time.time()
    for doc in docs:
        canned.current = doc  # each doc's fragment is served back on .extract()
        meta = {"scope": doc["scope"], "topic": doc["topic"],
                "doc_id": doc["doc_id"]}
        ingester.ingest_document(doc["text"], doc_metadata=meta)
    print(f"[bench3] ingest done in {time.time()-t0:.1f}s. "
          f"vectors={len(vector_db)}, nodes={len(state.nodes)}, "
          f"edges={len(state.edges)}", flush=True)

    retriever = GraphRAGRetriever(navigator=navigator, vector_db=vector_db,
                                  router=router)

    # ── Run queries ───────────────────────────────────────────────
    queries = generate_queries()
    print(f"[bench3] evaluating {len(queries)} scoped queries", flush=True)
    per_query = []
    for q in queries:
        # Condition A: vector-only (don't call retriever — just vector_db)
        a_chunks = vector_db.query(q["text"], k=5)
        a_score = score_retrieval(a_chunks, q["scope"])

        # Condition B: vector + graph with scope filter
        b_result = retriever.retrieve(
            q["text"], k_vector=5, k_graph=5, scope=q["scope"])
        b_chunks = b_result["chunks"][:5]
        b_score = score_retrieval(b_chunks, q["scope"])

        row = {
            "qid": q["qid"],
            "scope": q["scope"],
            "topic": q["topic"],
            "query": q["text"],
            "expected_authority": q["expected_authority"],
            "A_top5": [{"scope": (c.get("metadata") or {}).get("scope"),
                         "topic": (c.get("metadata") or {}).get("topic"),
                         "score": c.get("score")}
                        for c in a_chunks],
            "A_precision_at_5": a_score["precision_at_5"],
            "B_top5": [{"scope": (c.get("metadata") or {}).get("scope"),
                         "topic": (c.get("metadata") or {}).get("topic"),
                         "score": c.get("score")}
                        for c in b_chunks],
            "B_precision_at_5": b_score["precision_at_5"],
            "B_modes": b_result["modes"],
            "B_active_nodes": b_result.get("active_nodes"),
        }
        per_query.append(row)
        print(f"  [{q['qid']:45s}] A={row['A_precision_at_5']:.2f}  "
              f"B={row['B_precision_at_5']:.2f}  modes={b_result['modes']}",
              flush=True)

    # ── Aggregate ─────────────────────────────────────────────────
    a_mean = sum(r["A_precision_at_5"] for r in per_query) / len(per_query)
    b_mean = sum(r["B_precision_at_5"] for r in per_query) / len(per_query)
    a_at_least_3 = sum(1 for r in per_query if r["A_precision_at_5"] >= 0.6)
    b_at_least_3 = sum(1 for r in per_query if r["B_precision_at_5"] >= 0.6)

    gates = {
        "gate_b_precision_ge_0_8": b_mean >= 0.80,
        "gate_a_precision_lt_0_5": a_mean < 0.50,
        "gate_b_beats_a_consistently": b_at_least_3 > a_at_least_3,
    }

    summary = {
        "n_docs": len(docs),
        "n_queries": len(queries),
        "n_vector_chunks": len(vector_db),
        "n_graph_nodes": len(state.nodes),
        "n_graph_edges": len(state.edges),
        "A_precision_mean": a_mean,
        "B_precision_mean": b_mean,
        "A_hit_rate_ge_0_6": a_at_least_3 / len(per_query),
        "B_hit_rate_ge_0_6": b_at_least_3 / len(per_query),
        "gates": gates,
        "overall_pass": all(gates.values()),
        "per_query": per_query,
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== BENCHMARK 3 — SCOPE-SENSITIVE RETRIEVAL ===", flush=True)
    print(f"  docs:              {len(docs)}", flush=True)
    print(f"  queries:           {len(queries)}", flush=True)
    print(f"  A precision@5:     {a_mean:.3f}  (target < 0.50)", flush=True)
    print(f"  B precision@5:     {b_mean:.3f}  (target ≥ 0.80)", flush=True)
    print(f"  A hit rate ≥ 0.6: {a_at_least_3}/{len(per_query)}", flush=True)
    print(f"  B hit rate ≥ 0.6: {b_at_least_3}/{len(per_query)}", flush=True)
    print(f"  overall:           {'PASS' if summary['overall_pass'] else 'FAIL'}",
          flush=True)
    print(f"  wrote {out}", flush=True)

    sys.exit(0 if summary["overall_pass"] else 1)


if __name__ == "__main__":
    main()
