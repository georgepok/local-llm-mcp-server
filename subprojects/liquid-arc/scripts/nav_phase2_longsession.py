"""Phase 2 evaluation — long-session cross-domain queries.

NAVIGATOR_CONTINUATION_SPEC §Phase 2 Design.

For each variant (there are 3):
  - Ingest the 50-interaction session into a fresh navigator state.
  - Answer 10 cross-domain queries under 5 conditions:
      A) Plain LLM, last 5 interactions as context
      B) LLM + Navigator (uses accumulated h_state + pattern library)
      C) LLM + Full history (oracle upper bound — all 50 interactions)
      D) LLM + Navigator minus pattern library (ablation)
      E) LLM + Networkx (full accumulated graph, no metric)

  - Score with LLM-as-judge on a 4-tier rubric.

Outputs per-variant + aggregate JSON + checkpoint diagnostics on the
h_state after 50 interactions.

Usage (on Spark):
    python scripts/nav_phase2_longsession.py \\
        --checkpoint /workspace/liquid-arc/output_graph_engine_final/checkpoints/step_500.pt \\
        --sessions_dir /workspace/liquid-arc/data/navigator/phase2 \\
        --vllm_url http://172.17.0.1:30000/v1 \\
        --model NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \\
        --out_dir /workspace/liquid-arc/shared/outbox/phase2/ \\
        --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx

from liquid_arc.graph_engine_inference import GraphEngine
from liquid_arc.navigator import GeometricNavigator
from liquid_arc.navigator_extract import LLMExtractor
from liquid_arc.navigator_patterns import PatternLibrary
from liquid_arc.navigator_state import GeometricState


VERDICTS = ("CORRECT", "PARTIAL", "WRONG", "REFUSED")


# ----------------------------------------------------------------------
# Structural scoring — grades the answer against node IDs that SHOULD be
# named, rather than the LLM-as-judge surface match against a hand-written
# expected_answer_text. This fixes the evaluator bias that treated
# "unpatched_vpn" and "cybersecurity lateral target" as unrelated strings.
# ----------------------------------------------------------------------


def _node_id_stems(node_id: str) -> set:
    """4-char stems of the underscore-separated tokens in a node ID."""
    return {t[:4] for t in node_id.lower().split("_") if t}


def _answer_tokens(text: str) -> set:
    """Rough word set from the answer — split on non-alphanumeric, stem."""
    import re
    words = re.split(r"[^a-zA-Z0-9]+", (text or "").lower())
    return {w[:4] for w in words if w}


def _mentions_node(answer_text: str, node_id: str) -> bool:
    """True if at least one underscore-separated token stem of the node
    ID appears in the answer. Tokens of length ≤ 2 are dropped as
    uninformative."""
    ans_stems = _answer_tokens(answer_text)
    node_stems = {s for s in _node_id_stems(node_id) if len(s) > 2}
    if not node_stems:
        return False
    # Require at least one meaningful stem overlap
    hits = node_stems & ans_stems
    # Count only when the overlap covers a meaningful share of the node id
    return len(hits) >= max(1, len(node_stems) // 2)


def structural_score(answer_text: str,
                     expected_node_ids: List[str],
                     ) -> Dict[str, Any]:
    """Fraction of expected node IDs that the answer mentions.

    Matching is token-stem-based: "unpatched_vpn" matches "unpatched VPN",
    "vpn exposure", "the unpatched VPN portal". Returns a {score, hits,
    misses, total} dict.
    """
    if not expected_node_ids:
        return {"score": None, "hits": [], "misses": [], "total": 0}
    hits: List[str] = []
    misses: List[str] = []
    for nid in expected_node_ids:
        (hits if _mentions_node(answer_text, nid) else misses).append(nid)
    return {
        "score": len(hits) / max(1, len(expected_node_ids)),
        "hits": hits,
        "misses": misses,
        "total": len(expected_node_ids),
    }


ANSWER_PROMPT = """\
You are answering a question about an ongoing conversation.

{context_section}
Current question:
{question}

Respond in 1-2 sentences. If you don't have enough information to answer, \
say so explicitly."""


JUDGE_PROMPT = """\
You are evaluating whether an AI answer correctly addresses a question.

Question: {question}

Expected answer: {expected}

Required capability: {requires}

AI's answer: {answer}

Score with ONE of:
- CORRECT: answer contains the key information from expected answer
- PARTIAL: answer is on the right track but misses key details
- WRONG: answer is incorrect or doesn't address the question
- REFUSED: answer says it doesn't have enough information

Output ONLY one of: CORRECT, PARTIAL, WRONG, REFUSED"""


# ----------------------------------------------------------------------
# Context formatters
# ----------------------------------------------------------------------


def _format_interactions(interactions: List[Dict[str, Any]]) -> str:
    if not interactions:
        return "(no prior context)"
    return "\n".join(
        f"[turn {it.get('turn', '?')}] {it['text']}"
        for it in interactions
    )


def _format_text_segments(segments: List[Dict[str, Any]]) -> str:
    if not segments:
        return "(no relevant historical text)"
    return "\n".join(
        f"[turn {s.get('interaction_index', '?')}] {s['text']}"
        for s in segments
    )


def _networkx_hint(state: GeometricState, query_anchor_ids: List[str]) -> str:
    """For condition E — use only graph algorithms on accumulated graph."""
    if not state.nodes:
        return "(graph is empty)"
    g = nx.DiGraph()
    for nid, meta in state.nodes.items():
        g.add_node(nid, type=meta["type"], role=meta["role"])
    for e in state.edges:
        if e["src"] in state.nodes and e["dst"] in state.nodes:
            g.add_edge(e["src"], e["dst"], type=e["type"])

    lines = [f"Accumulated graph: {g.number_of_nodes()} nodes, "
             f"{g.number_of_edges()} edges."]

    # Centrality (betweenness) — top-5 hubs
    try:
        centrality = nx.betweenness_centrality(g)
        top = sorted(centrality.items(), key=lambda x: -x[1])[:5]
        if top:
            lines.append("Top-5 centrality nodes: "
                         + ", ".join(f"{nid} ({v:.3f})" for nid, v in top))
    except Exception:
        pass

    # For each anchor: neighbors + ancestors
    for a in query_anchor_ids:
        if a in g:
            preds = list(g.predecessors(a))[:5]
            succs = list(g.successors(a))[:5]
            ancestors: List[str] = []
            try:
                anc = nx.ancestors(g, a)
                ancestors = list(anc)[:5]
            except Exception:
                pass
            lines.append(f"Around `{a}`: predecessors={preds}, "
                         f"successors={succs}, ancestors={ancestors}")
    return "\n".join(lines)


def _dedupe_interaction_window(interactions: List[Dict[str, Any]],
                                window: int = 5) -> List[Dict[str, Any]]:
    """Return the last `window` interactions from a 50-interaction list."""
    return interactions[-window:]


# ----------------------------------------------------------------------
# LLM wrappers
# ----------------------------------------------------------------------


def _llm_answer(llm: LLMExtractor, question: str,
                context_section: str) -> str:
    prompt = ANSWER_PROMPT.format(context_section=context_section,
                                  question=question)
    return llm.generate(prompt).strip()


def _llm_judge(llm: LLMExtractor, question: str, expected: str,
               requires: str, answer: str) -> str:
    prompt = JUDGE_PROMPT.format(
        question=question, expected=expected,
        requires=requires, answer=answer)
    raw = llm.generate(prompt).strip().upper()
    for v in VERDICTS:
        if v in raw:
            return v
    return "WRONG"


# ----------------------------------------------------------------------
# Condition runners
# ----------------------------------------------------------------------


def _condition_A(llm: LLMExtractor, query: Dict[str, Any],
                 last5: List[Dict[str, Any]]) -> str:
    ctx = ("Recent interactions:\n" + _format_interactions(last5) + "\n\n")
    return _llm_answer(llm, query["text"], ctx)


def _condition_B(llm: LLMExtractor, navigator: GeometricNavigator,
                 query: Dict[str, Any],
                 last5: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Navigator processes the query text to pick anchors, then synthesizes
    # a hint + historical text from the accumulated state.
    anchors = query.get("anchor_nodes") or []
    # If query names specific anchors, treat them as the fragment.
    if anchors:
        anchor_fragment = {
            "nodes": [{"id": a, "type": navigator.state.nodes[a]["type"],
                       "role": navigator.state.nodes[a]["role"]}
                      for a in anchors if a in navigator.state.nodes],
            "edges": [],
        }
        nav_result = navigator.process_interaction(
            query["text"], pre_extracted=anchor_fragment)
    else:
        # No anchors named — ask the navigator for the top-k global hubs
        nav_result = navigator.process_interaction(
            query["text"], pre_extracted={"nodes": [], "edges": []})

    hint_text = nav_result.get("rendered_hint", "")
    text_segs = nav_result.get("relevant_text_segments", [])
    ctx_lines = [
        "Recent interactions:",
        _format_interactions(last5),
    ]
    if hint_text:
        ctx_lines.append("\nStructural analysis (from accumulated state):")
        ctx_lines.append(hint_text)
    if text_segs:
        ctx_lines.append("\nRelevant historical context (retrieved by "
                         "the navigator):")
        ctx_lines.append(_format_text_segments(text_segs))
    ctx = "\n".join(ctx_lines) + "\n\n"
    answer = _llm_answer(llm, query["text"], ctx)
    return {"answer": answer, "hint": hint_text,
            "text_segments": text_segs,
            "nav_analysis": nav_result.get("analysis"),
            "nav_pattern": nav_result.get("pattern_match")}


def _condition_C(llm: LLMExtractor, query: Dict[str, Any],
                 all_interactions: List[Dict[str, Any]]) -> str:
    ctx = ("Complete interaction history:\n"
           + _format_interactions(all_interactions) + "\n\n")
    return _llm_answer(llm, query["text"], ctx)


def _condition_D(llm: LLMExtractor, navigator_no_patterns: GeometricNavigator,
                 query: Dict[str, Any],
                 last5: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Identical to B but with pattern_library disabled (passed None).
    return _condition_B(llm, navigator_no_patterns, query, last5)


def _condition_E(llm: LLMExtractor, state: GeometricState,
                 query: Dict[str, Any],
                 last5: List[Dict[str, Any]]) -> str:
    anchors = query.get("anchor_nodes") or []
    hint = _networkx_hint(state, anchors)
    ctx = ("Recent interactions:\n" + _format_interactions(last5)
           + "\n\nGraph algorithm summary (networkx):\n" + hint + "\n\n")
    return _llm_answer(llm, query["text"], ctx)


# ----------------------------------------------------------------------
# Variant runner
# ----------------------------------------------------------------------


def run_variant(variant: Dict[str, Any], *, engine: GraphEngine,
                llm: LLMExtractor, judge_llm: LLMExtractor,
                work_dir: Path) -> Dict[str, Any]:
    vid = variant["variant_id"]
    interactions = variant["interactions"]
    queries = variant["queries"]
    print(f"\n[variant {vid}] starting — {len(interactions)} interactions, "
          f"{len(queries)} queries", flush=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Build a fresh navigator for conditions B & D, and a shared state to
    # also back condition E.
    state_b = GeometricState(str(work_dir / f"state_v{vid}_B.json"),
                             engine, max_nodes=1024)
    state_b.reset()
    patterns = PatternLibrary(str(work_dir / f"patterns_v{vid}.json"))
    patterns.reset()
    navigator_b = GeometricNavigator(
        engine=engine, state=state_b, extractor=None,
        pattern_library=patterns, pattern_threshold=0.85)

    state_d = GeometricState(str(work_dir / f"state_v{vid}_D.json"),
                             engine, max_nodes=1024)
    state_d.reset()
    navigator_d = GeometricNavigator(
        engine=engine, state=state_d, extractor=None,
        pattern_library=None)

    # Ingest all 50 interactions into both navigators
    t0 = time.time()
    for it in interactions:
        navigator_b.process_interaction(it["text"], pre_extracted=it["fragment"])
        navigator_d.process_interaction(it["text"], pre_extracted=it["fragment"])
    print(f"[variant {vid}] ingested {len(interactions)} interactions in "
          f"{time.time()-t0:.1f}s — B nodes={len(state_b.nodes)}, "
          f"patterns={len(patterns.patterns)}", flush=True)

    # Post-ingestion diagnostics on the full state_b
    diag = {}
    try:
        diag = json.loads(engine.get_graph_diagnostics(
            json.dumps(state_b.to_graph_dict())))
        diag["n_clusters"] = len(state_b.clusters)
        diag["n_patterns"] = len(patterns.patterns)
    except Exception as exc:
        print(f"[variant {vid}] diagnostics failed: {exc}", flush=True)

    # Run each query under 5 conditions
    last5 = _dedupe_interaction_window(interactions, 5)
    per_query = []
    for i, q in enumerate(queries):
        t_q = time.time()
        row: Dict[str, Any] = {
            "qid": q["qid"], "type": q["type"],
            "question": q["text"],
            "expected": q["expected_answer_text"],
            "requires": q["requires"],
        }

        # A, C, E give raw strings. B, D return dicts (with metadata).
        ans_a = _condition_A(llm, q, last5)
        res_b = _condition_B(llm, navigator_b, q, last5)
        ans_c = _condition_C(llm, q, interactions)
        res_d = _condition_D(llm, navigator_d, q, last5)
        ans_e = _condition_E(llm, state_b, q, last5)

        row["A_answer"] = ans_a
        row["B_answer"] = res_b["answer"]
        row["B_hint"] = res_b["hint"]
        row["B_text_segments"] = [
            {"turn": s.get("interaction_index"),
             "text": s["text"][:200]} for s in res_b["text_segments"]]
        row["B_analysis"] = res_b["nav_analysis"]
        row["B_pattern_match"] = res_b.get("nav_pattern")
        row["C_answer"] = ans_c
        row["D_answer"] = res_d["answer"]
        row["E_answer"] = ans_e

        # Judge with a separate temperature=0 call AND structural scorer.
        # The structural scorer grades against `expected_answer_node_ids`:
        # fraction of the structurally-correct node IDs the answer names,
        # robust to surface wording differences.
        expected_ids = q.get("expected_answer_node_ids", []) or []
        for cond, ans in (("A", ans_a), ("B", res_b["answer"]),
                          ("C", ans_c), ("D", res_d["answer"]),
                          ("E", ans_e)):
            row[f"{cond}_verdict"] = _llm_judge(
                judge_llm, q["text"], q["expected_answer_text"],
                q["requires"], ans)
            sc = structural_score(ans, expected_ids)
            row[f"{cond}_structural"] = sc["score"]
            row[f"{cond}_structural_hits"] = sc["hits"]
            row[f"{cond}_structural_total"] = sc["total"]

        row["elapsed_s"] = time.time() - t_q
        per_query.append(row)
        print(f"  [{i+1:02d}/{len(queries)} {q['qid']:26s} {q['type']:14s}] "
              f"A={row['A_verdict']:8s} B={row['B_verdict']:8s} "
              f"C={row['C_verdict']:8s} D={row['D_verdict']:8s} "
              f"E={row['E_verdict']:8s} {row['elapsed_s']:.1f}s", flush=True)

    return {
        "variant_id": vid,
        "diagnostics": diag,
        "per_query": per_query,
    }


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------


def _score(verdict: str) -> float:
    return {"CORRECT": 1.0, "PARTIAL": 0.5,
            "WRONG": 0.0, "REFUSED": 0.0}.get(verdict, 0.0)


def aggregate(variant_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_queries: List[Dict[str, Any]] = []
    for v in variant_results:
        for r in v["per_query"]:
            all_queries.append({**r, "variant_id": v["variant_id"]})

    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for r in all_queries:
        by_type.setdefault(r["type"], []).append(r)

    per_type_summary: Dict[str, Any] = {}
    for t, rows in by_type.items():
        s: Dict[str, Any] = {"n": len(rows)}
        for c in "ABCDE":
            s[f"{c}_correct"] = sum(1 for r in rows if r[f"{c}_verdict"] == "CORRECT")
            s[f"{c}_partial"] = sum(1 for r in rows if r[f"{c}_verdict"] == "PARTIAL")
            s[f"{c}_wrong"] = sum(1 for r in rows if r[f"{c}_verdict"] == "WRONG")
            s[f"{c}_refused"] = sum(1 for r in rows if r[f"{c}_verdict"] == "REFUSED")
            s[f"{c}_score"] = sum(_score(r[f"{c}_verdict"]) for r in rows) / max(1, len(rows))
            # Structural: mean fraction of expected-answer node IDs named.
            struc = [r.get(f"{c}_structural") for r in rows
                     if r.get(f"{c}_structural") is not None]
            s[f"{c}_structural_mean"] = (
                sum(struc) / len(struc) if struc else None)
        per_type_summary[t] = s

    # Success criteria
    def win_count(rows, cond_w, cond_lose):
        """Count cases where cond_w is CORRECT and cond_lose is WRONG/REFUSED."""
        return sum(
            1 for r in rows
            if r[f"{cond_w}_verdict"] == "CORRECT"
               and r[f"{cond_lose}_verdict"] in ("WRONG", "REFUSED")
        )

    recall_rows = by_type.get("recall", [])
    analogy_rows = by_type.get("analogy", [])
    topology_rows = by_type.get("topology", [])

    # Regressions: B WRONG where A CORRECT
    regressions = sum(
        1 for r in all_queries
        if r["B_verdict"] == "WRONG" and r["A_verdict"] == "CORRECT"
    )

    # B ≥ C: cases where B is CORRECT but C wasn't
    b_beats_c = sum(
        1 for r in all_queries
        if r["B_verdict"] == "CORRECT"
           and r["C_verdict"] in ("WRONG", "REFUSED", "PARTIAL")
    )

    gates = {
        "gate_1_B_beats_A_recall": win_count(recall_rows, "B", "A") >= 2 * len(variant_results),
        "gate_2_B_beats_A_analogy": win_count(analogy_rows, "B", "A") >= 2 * len(variant_results),
        "gate_3_B_beats_A_topology": win_count(topology_rows, "B", "A") >= 1 * len(variant_results),
        "gate_4_B_ge_C_3plus": b_beats_c >= 3 * len(variant_results),
        "gate_5_B_beats_E_analogy": win_count(analogy_rows, "B", "E") >= 2 * len(variant_results),
        "gate_6_B_beats_D_analogy": win_count(analogy_rows, "B", "D") >= 1 * len(variant_results),
        "gate_7_zero_regressions": regressions == 0,
    }

    # Consistency across variants (gate 8)
    per_variant_b_correct: Dict[int, int] = {}
    for v in variant_results:
        per_variant_b_correct[v["variant_id"]] = sum(
            1 for r in v["per_query"] if r["B_verdict"] == "CORRECT")
    b_range = (max(per_variant_b_correct.values())
               - min(per_variant_b_correct.values())
               if per_variant_b_correct else 0)
    gates["gate_8_variant_consistency"] = b_range <= 2

    return {
        "n_variants": len(variant_results),
        "n_queries_total": len(all_queries),
        "per_type": per_type_summary,
        "regressions_B_vs_A": regressions,
        "B_beats_C": b_beats_c,
        "per_variant_B_correct": per_variant_b_correct,
        "gates": gates,
        "overall_pass": all(gates.values()),
    }


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--sessions_dir", required=True,
                   help="directory containing variant_{N}.json files")
    p.add_argument("--vllm_url", default="http://localhost:30000/v1")
    p.add_argument("--model", default="NVIDIA-Nemotron-3-Nano-30B-A3B-FP8")
    p.add_argument("--judge_vllm_url", default=None,
                   help="use a different endpoint for the judge "
                        "(defaults to --vllm_url)")
    p.add_argument("--judge_model", default=None,
                   help="if set, use this model for LLM-as-judge")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--variants", default="0,1,2",
                   help="comma-separated variant IDs to run")
    p.add_argument("--max_answer_tokens", type=int, default=200)
    p.add_argument("--max_judge_tokens", type=int, default=20)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions_dir = Path(args.sessions_dir)

    print(f"[phase2] loading graph engine from {args.checkpoint}", flush=True)
    engine = GraphEngine(args.checkpoint, device=args.device,
                         corrections_log=None)

    llm = LLMExtractor(base_url=args.vllm_url, model=args.model,
                       max_tokens=args.max_answer_tokens, temperature=0.1)
    judge_url = args.judge_vllm_url or args.vllm_url
    judge_model = args.judge_model or args.model
    judge_llm = LLMExtractor(base_url=judge_url, model=judge_model,
                             max_tokens=args.max_judge_tokens, temperature=0.0)

    want_variants = [int(x) for x in args.variants.split(",") if x.strip()]
    variant_results = []
    for vid in want_variants:
        variant_path = sessions_dir / f"variant_{vid}.json"
        if not variant_path.exists():
            print(f"[phase2] missing {variant_path}, skipping", flush=True)
            continue
        with open(variant_path) as f:
            variant = json.load(f)
        res = run_variant(variant, engine=engine, llm=llm, judge_llm=judge_llm,
                          work_dir=out_dir / f"variant_{vid}")
        per_path = out_dir / f"variant_{vid}.json"
        with open(per_path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[phase2] wrote {per_path}", flush=True)
        variant_results.append(res)

    agg = aggregate(variant_results)
    agg_path = out_dir / "aggregate.json"
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\n=== PHASE 2 SUMMARY ===", flush=True)
    print(json.dumps(agg, indent=2), flush=True)
    print(f"\nwrote {agg_path}", flush=True)

    sys.exit(0 if agg["overall_pass"] else 1)


if __name__ == "__main__":
    main()
