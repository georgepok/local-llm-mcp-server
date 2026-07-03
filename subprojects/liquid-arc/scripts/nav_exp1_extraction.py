"""Navigator Experiment 1 — structure extraction validation.

Feeds the 30-passage test set through the LLM extractor and measures:
  - Node F1       (target > 0.85)
  - Edge F1       (target > 0.75)
  - Type accuracy over matched nodes
  - Role accuracy over matched nodes
  - Edge-type accuracy over matched edges

Usage (on Spark, fgn-train or any container with access to vLLM):
    python -m liquid_arc.scripts.nav_exp1_extraction \\
        --testset /workspace/liquid-arc/data/navigator/extraction_testset.jsonl \\
        --vllm_url http://localhost:30000/v1 \\
        --model NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \\
        --out_json /workspace/liquid-arc/shared/outbox/nav_exp1_extraction.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from liquid_arc.navigator_extract import LLMExtractor, extract_graph


def _load_testset(path: str) -> List[Dict[str, Any]]:
    cases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def _edge_key(e: Dict[str, Any]) -> Tuple[str, str]:
    return (e["src"], e["dst"])


def _typed_edge_key(e: Dict[str, Any]) -> Tuple[str, str, str]:
    return (e["src"], e["dst"], e.get("type", "related_to"))


def _stems(nid: str) -> set:
    """Underscore-tokenize → 4-char stems. Collapses plurals ('server' /
    'servers' → 'serv'), tenses ('interruption' / 'interrupted' → 'inte'),
    and simple morphological variants."""
    return set(t[:4] for t in nid.lower().split("_") if t)


def _match_pairs(expected_ids: List[str], predicted_ids: List[str]
                 ) -> Dict[str, str]:
    """Greedy 1-1 matching on stem-set Jaccard ≥ 0.5."""
    scores = []
    for e in expected_ids:
        for p in predicted_ids:
            ta, tb = _stems(e), _stems(p)
            if not ta or not tb:
                continue
            j = len(ta & tb) / len(ta | tb)
            if j >= 0.5:
                scores.append((j, e, p))
    scores.sort(key=lambda x: -x[0])
    exp_taken, pred_taken = set(), set()
    mapping: Dict[str, str] = {}
    for _, e, p in scores:
        if e in exp_taken or p in pred_taken:
            continue
        mapping[e] = p
        exp_taken.add(e)
        pred_taken.add(p)
    return mapping


def _score_case(expected: Dict[str, Any],
                predicted: Dict[str, Any]) -> Dict[str, float]:
    exp_nodes = {n["id"]: n for n in expected["nodes"]}
    pred_nodes = {n["id"]: n for n in predicted["nodes"]}

    # Fuzzy id-to-id mapping via token overlap.
    id_map = _match_pairs(list(exp_nodes), list(pred_nodes))
    tp_nodes = set(id_map.keys())                # matched expected ids
    n_precision = len(tp_nodes) / max(1, len(pred_nodes))
    n_recall = len(tp_nodes) / max(1, len(exp_nodes))
    n_f1 = (2 * n_precision * n_recall / (n_precision + n_recall)
            if (n_precision + n_recall) > 0 else 0.0)

    type_hits = sum(1 for e_id in tp_nodes
                    if pred_nodes[id_map[e_id]]["type"] == exp_nodes[e_id]["type"])
    role_hits = sum(1 for e_id in tp_nodes
                    if pred_nodes[id_map[e_id]]["role"] == exp_nodes[e_id]["role"])
    type_acc = type_hits / max(1, len(tp_nodes))
    role_acc = role_hits / max(1, len(tp_nodes))

    # Edges: map expected (src,dst) → (pred_src,pred_dst) via id_map, then
    # check membership in predicted edge set.
    pred_edge_set = {_edge_key(e) for e in predicted["edges"]}
    pred_typed_set = {_typed_edge_key(e) for e in predicted["edges"]}

    exp_edges_mapped = []
    exp_typed_mapped = []
    skipped = 0
    for e in expected["edges"]:
        if e["src"] not in id_map or e["dst"] not in id_map:
            skipped += 1
            continue
        exp_edges_mapped.append((id_map[e["src"]], id_map[e["dst"]]))
        exp_typed_mapped.append(
            (id_map[e["src"]], id_map[e["dst"]], e.get("type", "related_to")))

    exp_edge_set = set(exp_edges_mapped)
    tp_edges = exp_edge_set & pred_edge_set
    e_precision = len(tp_edges) / max(1, len(pred_edge_set))
    e_recall = len(tp_edges) / max(1, len(expected["edges"]))
    e_f1 = (2 * e_precision * e_recall / (e_precision + e_recall)
            if (e_precision + e_recall) > 0 else 0.0)

    exp_typed = set(exp_typed_mapped)
    tp_typed = exp_typed & pred_typed_set
    edge_type_acc = len(tp_typed) / max(1, len(tp_edges)) if tp_edges else 0.0

    return {
        "node_precision": n_precision, "node_recall": n_recall, "node_f1": n_f1,
        "edge_precision": e_precision, "edge_recall": e_recall, "edge_f1": e_f1,
        "type_accuracy": type_acc, "role_accuracy": role_acc,
        "edge_type_accuracy": edge_type_acc,
        "n_exp_nodes": len(exp_nodes), "n_pred_nodes": len(pred_nodes),
        "n_exp_edges": len(expected["edges"]),
        "n_pred_edges": len(predicted["edges"]),
        "n_edges_unmapped": skipped,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--testset", required=True)
    p.add_argument("--vllm_url", default="http://localhost:30000/v1")
    p.add_argument("--model", default="NVIDIA-Nemotron-3-Nano-30B-A3B-FP8")
    p.add_argument("--out_json", required=True)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--max_tokens", type=int, default=800)
    p.add_argument("--limit", type=int, default=0,
                   help="only run the first N cases (0 = all)")
    args = p.parse_args()

    cases = _load_testset(args.testset)
    if args.limit > 0:
        cases = cases[:args.limit]
    print(f"[exp1] testset: {len(cases)} cases from {args.testset}", flush=True)

    extractor = LLMExtractor(base_url=args.vllm_url, model=args.model,
                             temperature=args.temperature,
                             max_tokens=args.max_tokens)

    per_case = []
    parse_failures = 0
    t0 = time.time()
    for i, case in enumerate(cases):
        t_case = time.time()
        try:
            predicted = extract_graph(case["text"], extractor)
        except Exception as exc:
            predicted = None
            print(f"  [{case['id']}] extraction error: {exc}", flush=True)
        if predicted is None:
            parse_failures += 1
            predicted = {"nodes": [], "edges": []}
        scores = _score_case(case["expected"], predicted)
        scores["id"] = case["id"]
        scores["category"] = case["category"]
        scores["predicted"] = predicted
        scores["expected"] = case["expected"]
        scores["latency_s"] = time.time() - t_case
        per_case.append(scores)
        print(f"  [{i+1:02d}/{len(cases)} {case['id']:12s} {case['category']:16s}] "
              f"node_f1={scores['node_f1']:.2f} edge_f1={scores['edge_f1']:.2f} "
              f"type_acc={scores['type_accuracy']:.2f} role_acc={scores['role_accuracy']:.2f} "
              f"{scores['latency_s']:.1f}s", flush=True)

    total = time.time() - t0
    agg_keys = ["node_precision", "node_recall", "node_f1",
                "edge_precision", "edge_recall", "edge_f1",
                "type_accuracy", "role_accuracy", "edge_type_accuracy"]
    agg = {k: sum(c[k] for c in per_case) / max(1, len(per_case)) for k in agg_keys}

    by_cat: Dict[str, List[Dict]] = {}
    for c in per_case:
        by_cat.setdefault(c["category"], []).append(c)
    cat_agg = {}
    for cat, lst in by_cat.items():
        cat_agg[cat] = {
            k: sum(c[k] for c in lst) / max(1, len(lst)) for k in agg_keys
        }
        cat_agg[cat]["n"] = len(lst)

    node_f1 = agg["node_f1"]
    edge_f1 = agg["edge_f1"]
    gate_node = node_f1 >= 0.85
    gate_edge = edge_f1 >= 0.75
    overall_pass = gate_node and gate_edge

    summary = {
        "testset": args.testset,
        "vllm_url": args.vllm_url,
        "model": args.model,
        "n_cases": len(cases),
        "parse_failures": parse_failures,
        "total_s": total,
        "aggregate": agg,
        "per_category": cat_agg,
        "gates": {
            "node_f1_target": 0.85, "node_f1_actual": node_f1, "pass": gate_node,
        },
        "edge_gate": {
            "edge_f1_target": 0.75, "edge_f1_actual": edge_f1, "pass": gate_edge,
        },
        "overall_pass": overall_pass,
        "per_case": per_case,
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== EXPERIMENT 1 — SUMMARY ===", flush=True)
    print(f"  cases:            {len(cases)}", flush=True)
    print(f"  parse failures:   {parse_failures}", flush=True)
    print(f"  node F1:          {node_f1:.3f}  (target 0.85, "
          f"{'PASS' if gate_node else 'FAIL'})", flush=True)
    print(f"  edge F1:          {edge_f1:.3f}  (target 0.75, "
          f"{'PASS' if gate_edge else 'FAIL'})", flush=True)
    print(f"  type accuracy:    {agg['type_accuracy']:.3f}", flush=True)
    print(f"  role accuracy:    {agg['role_accuracy']:.3f}", flush=True)
    print(f"  edge-type acc:    {agg['edge_type_accuracy']:.3f}", flush=True)
    print(f"  overall:          {'PASS' if overall_pass else 'FAIL'}", flush=True)
    print(f"  wrote {out}", flush=True)

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
