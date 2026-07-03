"""Navigator Experiment 3 — metric-based context vs recency-based context.

After accumulating the 20 supply-chain interactions, each of 10 test
queries has:
  - a query node (anchor)
  - a hand-labeled 'relevant_ids' set (nodes a human says should come back)
  - an expected_old_turn (interaction where the relevant nodes live)

For each query we compare:
  A) Metric retrieval:  state.query_relevant(anchor, k=10)
  B) Recency retrieval: the last k=10 nodes added to state

Pass criterion: on at least 5 of 10 queries, metric finds ≥2 relevant
nodes that recency misses.

Usage:
    python -m liquid_arc.scripts.nav_exp3_context \\
        --checkpoint /workspace/liquid-arc/output_graph_engine_final/checkpoints/step_500.pt \\
        --interactions /workspace/liquid-arc/data/navigator/supply_chain_interactions.jsonl \\
        --state_path /workspace/liquid-arc/navigator_state_exp3.json \\
        --out_json /workspace/liquid-arc/shared/outbox/nav_exp3_context.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from liquid_arc.graph_engine_inference import GraphEngine
from liquid_arc.navigator_state import GeometricState


# Each query: {anchor, relevant_ids, expected_old_turn, notes}
# anchor is a node id that already exists in state after accumulation.
# relevant_ids is a list of nodes that the metric should consider 'related'
# — chosen by hand to be semantically close to the anchor but far away in
# insertion order.
TEST_QUERIES: List[Dict[str, Any]] = [
    {
        "qid": "q1_semiconductor",
        "anchor": "semiconductor_shortage",
        "relevant_ids": ["taiwan_typhoon", "fab_pause", "electronics_assembly_delay",
                         "component_price_rise", "device_retail_up"],
        "expected_old_turn": 5,
    },
    {
        "qid": "q2_shanghai",
        "anchor": "shanghai_port",
        "relevant_ids": ["port_congestion", "shipment_delay", "la_hub",
                         "hub_backlog", "phoenix_warehouse", "retailer_restock_delay"],
        "expected_old_turn": 1,
    },
    {
        "qid": "q3_stock_out",
        "anchor": "stock_out",
        "relevant_ids": ["low_inventory", "popular_skus", "lost_revenue",
                         "complaint_spike", "retailer_restock_delay"],
        "expected_old_turn": 3,
    },
    {
        "qid": "q4_trucker",
        "anchor": "trucker_strike",
        "relevant_ids": ["freight_halt", "hamburg_munich_route",
                         "auto_parts_delay", "assembly_plant"],
        "expected_old_turn": 8,
    },
    {
        "qid": "q5_bird_flu",
        "anchor": "bird_flu",
        "relevant_ids": ["poultry_cull", "egg_supply_drop", "egg_wholesale_up",
                         "bakery_cost_rise", "restaurant_cost_rise"],
        "expected_old_turn": 12,
    },
    {
        "qid": "q6_vehicle_delivery",
        "anchor": "vehicle_delivery_drop",
        "relevant_ids": ["assembly_plant", "dealer_inventory_low",
                         "emergency_air_freight", "logistics_cost_up"],
        "expected_old_turn": 10,
    },
    {
        "qid": "q7_bread_price",
        "anchor": "bread_price_up",
        "relevant_ids": ["bakery_cost_rise", "egg_wholesale_up",
                         "pastry_price_up", "egg_supply_drop"],
        "expected_old_turn": 14,
    },
    {
        "qid": "q8_rotterdam",
        "anchor": "rotterdam_port",
        "relevant_ids": ["container_shortage", "eu_export_slowdown",
                         "north_america", "wine_shortage", "cheese_shortage"],
        "expected_old_turn": 15,
    },
    {
        "qid": "q9_promo_cut",
        "anchor": "promo_cut",
        "relevant_ids": ["component_price_rise", "device_retail_up",
                         "semiconductor_shortage"],
        "expected_old_turn": 7,
    },
    {
        "qid": "q10_dealer_inventory",
        "anchor": "dealer_inventory_low",
        "relevant_ids": ["vehicle_delivery_drop", "assembly_plant",
                         "emergency_air_freight", "logistics_cost_up"],
        "expected_old_turn": 10,
    },
]


def _recency_topk(state: GeometricState, anchor: str, k: int = 10) -> List[str]:
    items = [(nid, meta["last_seen"]) for nid, meta in state.nodes.items()
             if nid != anchor]
    items.sort(key=lambda x: x[1], reverse=True)
    return [nid for nid, _ in items[:k]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--interactions", required=True)
    p.add_argument("--state_path", required=True)
    p.add_argument("--out_json", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--k", type=int, default=10)
    args = p.parse_args()

    engine = GraphEngine(args.checkpoint, device=args.device,
                         corrections_log=None)

    # Fresh state
    if os.path.exists(args.state_path):
        os.remove(args.state_path)
    state = GeometricState(args.state_path, engine, max_nodes=512)

    interactions = []
    with open(args.interactions) as f:
        for line in f:
            line = line.strip()
            if line:
                interactions.append(json.loads(line))
    for item in interactions:
        state.merge_fragment(item["fragment"])

    per_query = []
    wins = 0
    for q in TEST_QUERIES:
        anchor = q["anchor"]
        relevant = set(q["relevant_ids"])
        if anchor not in state.nodes:
            print(f"  [{q['qid']}] anchor '{anchor}' missing, skipping", flush=True)
            continue

        metric_hits = state.query_relevant([anchor], k=args.k)
        metric_ids = [r["id"] for r in metric_hits]
        recency_ids = _recency_topk(state, anchor, k=args.k)

        metric_relevant = relevant & set(metric_ids)
        recency_relevant = relevant & set(recency_ids)
        metric_extra = metric_relevant - recency_relevant

        win = len(metric_extra) >= 2
        if win:
            wins += 1

        per_query.append({
            "qid": q["qid"],
            "anchor": anchor,
            "expected_old_turn": q["expected_old_turn"],
            "relevant_total": len(relevant),
            "metric_topk": metric_ids,
            "recency_topk": recency_ids,
            "metric_hits": sorted(metric_relevant),
            "recency_hits": sorted(recency_relevant),
            "metric_extra": sorted(metric_extra),
            "metric_recall": len(metric_relevant) / max(1, len(relevant)),
            "recency_recall": len(recency_relevant) / max(1, len(relevant)),
            "win": win,
        })
        print(f"  [{q['qid']:20s} anchor={anchor:25s}] "
              f"metric_recall={per_query[-1]['metric_recall']:.2f} "
              f"recency_recall={per_query[-1]['recency_recall']:.2f} "
              f"extra={len(metric_extra)} {'WIN' if win else '---'}", flush=True)

    n_queries = len(per_query)
    overall_pass = wins >= 5
    summary = {
        "checkpoint": args.checkpoint,
        "n_queries": n_queries,
        "wins": wins,
        "pass_target": 5,
        "per_query": per_query,
        "mean_metric_recall": (
            sum(q["metric_recall"] for q in per_query) / max(1, n_queries)),
        "mean_recency_recall": (
            sum(q["recency_recall"] for q in per_query) / max(1, n_queries)),
        "overall_pass": overall_pass,
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== EXPERIMENT 3 — SUMMARY ===", flush=True)
    print(f"  queries run:          {n_queries}", flush=True)
    print(f"  wins (metric > recency by ≥2): {wins}/{n_queries}", flush=True)
    print(f"  metric mean recall:   {summary['mean_metric_recall']:.3f}",
          flush=True)
    print(f"  recency mean recall:  {summary['mean_recency_recall']:.3f}",
          flush=True)
    print(f"  overall:              {'PASS' if overall_pass else 'FAIL'}", flush=True)
    print(f"  wrote {out}", flush=True)
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
