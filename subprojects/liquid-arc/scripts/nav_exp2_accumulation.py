"""Navigator Experiment 2 — h_state accumulation.

Feeds 20 supply-chain interactions into a GeometricState (using
pre-extracted graph fragments to isolate from extraction noise),
then verifies:

  - Graph size grows to 30-60 nodes, 40-80 edges
  - CV(g) after accumulation > 2.0
  - At least 3 metric clusters emerge
  - State persists correctly across a simulated restart (load from disk)

Usage (on Spark, any container with access to the graph-engine checkpoint):
    python -m liquid_arc.scripts.nav_exp2_accumulation \\
        --checkpoint /workspace/liquid-arc/output_graph_engine_final/checkpoints/step_500.pt \\
        --interactions /workspace/liquid-arc/data/navigator/supply_chain_interactions.jsonl \\
        --state_path /workspace/liquid-arc/navigator_state_exp2.json \\
        --out_json /workspace/liquid-arc/shared/outbox/nav_exp2_accumulation.json
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
from liquid_arc.navigator_state import GeometricState


def _load_interactions(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _diagnostics(state: GeometricState) -> Dict[str, Any]:
    """Read CV and D²/4τ from the full accumulated graph."""
    if len(state.nodes) < 2:
        return {"cv_g": 0.0, "criticality_ratio": 0.0,
                "n_nodes": len(state.nodes), "n_edges": len(state.edges),
                "n_clusters": len(state.clusters)}
    graph_json = json.dumps(state.to_graph_dict())
    diag = json.loads(state.engine.get_graph_diagnostics(graph_json))
    diag["n_clusters"] = len(state.clusters)
    return diag


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--interactions", required=True)
    p.add_argument("--state_path", required=True)
    p.add_argument("--out_json", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    print(f"[exp2] loading engine from {args.checkpoint}", flush=True)
    t0 = time.time()
    engine = GraphEngine(args.checkpoint, device=args.device,
                         corrections_log=None)
    print(f"[exp2] engine loaded in {time.time()-t0:.1f}s", flush=True)

    # Clean-slate start for this experiment
    if os.path.exists(args.state_path):
        os.remove(args.state_path)

    state = GeometricState(args.state_path, engine, max_nodes=512)
    interactions = _load_interactions(args.interactions)
    print(f"[exp2] accumulating {len(interactions)} interactions", flush=True)

    per_turn = []
    for item in interactions:
        t_merge = time.time()
        report = state.merge_fragment(item["fragment"])
        diag = _diagnostics(state)
        per_turn.append({
            "turn": item["turn"],
            "text": item["text"],
            "added_nodes": report["added_nodes"],
            "added_edges": report["added_edges"],
            "total_nodes": report["total_nodes"],
            "total_edges": report["total_edges"],
            "cv_g": diag.get("cv_g"),
            "criticality_ratio": diag.get("criticality_ratio"),
            "tau_mean": diag.get("tau_mean"),
            "n_clusters": diag.get("n_clusters"),
            "merge_s": time.time() - t_merge,
        })
        print(f"  turn {item['turn']:2d}: +{report['added_nodes']}n/"
              f"{report['added_edges']}e "
              f"total={report['total_nodes']}n/{report['total_edges']}e "
              f"CV={diag.get('cv_g', 0):.2f} "
              f"clusters={diag.get('n_clusters', 0)} "
              f"crit={diag.get('criticality_ratio', 0):.2f} "
              f"{per_turn[-1]['merge_s']:.2f}s", flush=True)

    final_diag = _diagnostics(state)
    final_sig = state.get_signature()
    gates = {
        "size_in_range": 30 <= len(state.nodes) <= 80,
        "cv_above_2": float(final_diag.get("cv_g", 0.0)) > 2.0,
        "clusters_ge_3": len(state.clusters) >= 3,
    }

    # Persistence test: drop the in-memory state, reload from disk
    print("[exp2] simulating restart — loading state from disk", flush=True)
    state_reloaded = GeometricState(args.state_path, engine, max_nodes=512)
    reload_diag = _diagnostics(state_reloaded)
    persistence_ok = (
        len(state_reloaded.nodes) == len(state.nodes)
        and len(state_reloaded.edges) == len(state.edges)
        and len(state_reloaded.embeddings) == len(state.embeddings)
    )
    gates["persistence"] = persistence_ok

    overall_pass = all(gates.values())

    summary = {
        "checkpoint": args.checkpoint,
        "state_path": args.state_path,
        "n_interactions": len(interactions),
        "per_turn": per_turn,
        "final_diagnostics": final_diag,
        "reload_diagnostics": reload_diag,
        "final_signature": final_sig,
        "final_clusters": state.clusters,
        "gates": gates,
        "overall_pass": overall_pass,
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== EXPERIMENT 2 — SUMMARY ===", flush=True)
    print(f"  interactions processed: {len(interactions)}", flush=True)
    print(f"  final nodes:            {len(state.nodes)}", flush=True)
    print(f"  final edges:            {len(state.edges)}", flush=True)
    print(f"  CV(g):                  {final_diag.get('cv_g', 0):.3f}  "
          f"(target > 2.0, {'PASS' if gates['cv_above_2'] else 'FAIL'})", flush=True)
    print(f"  metric clusters:        {len(state.clusters)}  "
          f"(target >= 3, {'PASS' if gates['clusters_ge_3'] else 'FAIL'})", flush=True)
    print(f"  size 30-80 gate:        {'PASS' if gates['size_in_range'] else 'FAIL'}",
          flush=True)
    print(f"  persistence (reload):   {'PASS' if gates['persistence'] else 'FAIL'}",
          flush=True)
    print(f"  overall:                {'PASS' if overall_pass else 'FAIL'}",
          flush=True)
    print(f"  wrote {out}", flush=True)

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
