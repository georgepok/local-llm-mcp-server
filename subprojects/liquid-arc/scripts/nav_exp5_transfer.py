"""Navigator Experiment 5 — cross-domain structural pattern transfer.

Phase 1: Navigator processes the 20 supply-chain interactions, building
         a pattern library in 'cascade_failure' domain.
Phase 2: Present ecology problems with isomorphic structure (predator
         removal → cascade biodiversity collapse). The pattern library
         should match these to supply-chain patterns because the metric
         signature captures topology, not content.

Metrics:
  - Max pattern cosine similarity on each ecology problem (target > 0.80)
  - LLM answer correctness with vs without the pattern hint

Pass criterion: max cosine > 0.80 on all ecology problems.

Usage:
    python -m liquid_arc.scripts.nav_exp5_transfer \\
        --checkpoint /workspace/liquid-arc/output_graph_engine_final/checkpoints/step_500.pt \\
        --supply_chain /workspace/liquid-arc/data/navigator/supply_chain_interactions.jsonl \\
        --ecology /workspace/liquid-arc/data/navigator/ecology_isomorphic.jsonl \\
        --vllm_url http://localhost:30000/v1 \\
        --out_json /workspace/liquid-arc/shared/outbox/nav_exp5_transfer.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from liquid_arc.graph_engine_inference import GraphEngine
from liquid_arc.navigator import GeometricNavigator
from liquid_arc.navigator_patterns import PatternLibrary, _cosine
from liquid_arc.navigator_state import GeometricState


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--supply_chain", required=True)
    p.add_argument("--ecology", required=True)
    p.add_argument("--out_json", required=True)
    p.add_argument("--state_path", default="/tmp/nav_exp5_state.json")
    p.add_argument("--pattern_path", default="/tmp/nav_exp5_patterns.json")
    p.add_argument("--vllm_url", default="http://localhost:30000/v1")
    p.add_argument("--model", default="NVIDIA-Nemotron-3-Nano-30B-A3B-FP8")
    p.add_argument("--device", default="cuda")
    p.add_argument("--cosine_threshold", type=float, default=0.80)
    args = p.parse_args()

    for path in (args.state_path, args.pattern_path):
        if os.path.exists(path):
            os.remove(path)

    engine = GraphEngine(args.checkpoint, device=args.device,
                         corrections_log=None)
    state = GeometricState(args.state_path, engine, max_nodes=512)
    patterns = PatternLibrary(args.pattern_path)
    navigator = GeometricNavigator(
        engine=engine, state=state, extractor=None,
        pattern_library=patterns, pattern_threshold=0.0,  # always record
    )

    # ── Phase 1: accumulate supply-chain pattern library ──────────
    interactions = _load_jsonl(args.supply_chain)
    print(f"[exp5] phase 1 — ingesting {len(interactions)} supply-chain "
          f"interactions", flush=True)
    for item in interactions:
        navigator.process_interaction(item["text"],
                                      pre_extracted=item["fragment"])
    print(f"[exp5] phase 1 done. nodes={len(state.nodes)} "
          f"patterns={len(patterns.patterns)}", flush=True)

    if not patterns.patterns:
        print("[exp5] FAIL: no patterns recorded after phase 1", flush=True)
        sys.exit(1)

    # ── Phase 2: isomorphic ecology problems — preserve supply-chain
    #    pattern library but start a fresh state so the only thing
    #    available for matching is structure, not the literal nodes.
    eco_cases = _load_jsonl(args.ecology)
    print(f"[exp5] phase 2 — {len(eco_cases)} isomorphic ecology problems",
          flush=True)

    per_case = []
    for case in eco_cases:
        # Fresh scratch state per ecology case so signatures reflect ONLY
        # the ecology fragment, not residual supply-chain structure.
        scratch_path = args.state_path + f".{case['qid']}.scratch"
        if os.path.exists(scratch_path):
            os.remove(scratch_path)
        scratch_state = GeometricState(scratch_path, engine, max_nodes=64)
        scratch_nav = GeometricNavigator(
            engine=engine, state=scratch_state, extractor=None,
            pattern_library=patterns,
            pattern_threshold=args.cosine_threshold,
        )
        result = scratch_nav.process_interaction(
            case["text"], pre_extracted=case["fragment"])

        current_sig = scratch_state.get_signature()
        # Compute max cosine vs every stored pattern
        max_sim = 0.0
        best_label = None
        if current_sig is not None:
            for pat in patterns.patterns:
                s = _cosine(current_sig, pat["signature"])
                if s > max_sim:
                    max_sim = s
                    best_label = pat.get("label")

        per_case.append({
            "qid": case["qid"],
            "question": case["question"],
            "expected": case["answer_key"],
            "max_cosine": max_sim,
            "best_label": best_label,
            "match_above_threshold": max_sim >= args.cosine_threshold,
            "navigator_analysis": result.get("analysis"),
            "navigator_query": result.get("query"),
        })
        print(f"  {case['qid']:30s} max_cos={max_sim:.3f} "
              f"label={best_label} "
              f"{'MATCH' if max_sim >= args.cosine_threshold else 'no-match'}",
              flush=True)
        os.remove(scratch_path)

    n_match = sum(1 for c in per_case if c["match_above_threshold"])
    overall_pass = n_match == len(eco_cases)

    summary = {
        "supply_chain_path": args.supply_chain,
        "ecology_path": args.ecology,
        "cosine_threshold": args.cosine_threshold,
        "n_patterns_built": len(patterns.patterns),
        "n_ecology_cases": len(eco_cases),
        "n_matched": n_match,
        "per_case": per_case,
        "overall_pass": overall_pass,
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== EXPERIMENT 5 — SUMMARY ===", flush=True)
    print(f"  patterns built:         {len(patterns.patterns)}", flush=True)
    print(f"  ecology cases:          {len(eco_cases)}", flush=True)
    print(f"  matched above {args.cosine_threshold:.2f}: "
          f"{n_match}/{len(eco_cases)}", flush=True)
    print(f"  overall:                {'PASS' if overall_pass else 'FAIL'}",
          flush=True)
    print(f"  wrote {out}", flush=True)
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
