"""Re-score existing Phase 2 outputs with the structural scorer.

Purpose: validate the claim that the navigator produces structurally-
correct answers that the LLM-as-judge under-credits. Reads cached
variant_{N}.json files from a previous run, applies structural_score()
from nav_phase2_longsession.py against the (now-derived) expected_answer
_node_ids from gen_nav_session.py, and emits a side-by-side comparison.

No new LLM calls — only deterministic text scoring.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from nav_phase2_longsession import aggregate, structural_score

sys.path.insert(0, str(Path(__file__).parent))
from gen_nav_session import build_variant  # noqa: E402


def rescore(per_variant_paths: List[Path]) -> Dict[str, Any]:
    out_variants: List[Dict[str, Any]] = []
    for path in per_variant_paths:
        with open(path) as f:
            cached = json.load(f)
        vid = cached["variant_id"]
        # Rebuild the variant to get expected_answer_node_ids keyed by qid
        fresh = build_variant(vid)
        qid_to_expected = {
            q["qid"]: q.get("expected_answer_node_ids", []) or []
            for q in fresh["queries"]
        }
        qid_to_note = {q["qid"]: q.get("ground_truth_note", "")
                        for q in fresh["queries"]}
        for row in cached["per_query"]:
            expected = qid_to_expected.get(row["qid"], [])
            row["expected_answer_node_ids"] = expected
            row["ground_truth_note"] = qid_to_note.get(row["qid"], "")
            for cond in "ABCDE":
                ans = row.get(f"{cond}_answer", "")
                sc = structural_score(ans, expected)
                row[f"{cond}_structural"] = sc["score"]
                row[f"{cond}_structural_hits"] = sc["hits"]
                row[f"{cond}_structural_total"] = sc["total"]
        out_variants.append(cached)
    return out_variants


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", required=True,
                   help="directory containing variant_{0,1,2}.json")
    p.add_argument("--out_json", required=True)
    args = p.parse_args()
    in_dir = Path(args.in_dir)
    per_variant_paths = sorted(in_dir.glob("variant_*.json"))
    if not per_variant_paths:
        print(f"no variant_*.json under {in_dir}", flush=True)
        sys.exit(1)
    variants = rescore(per_variant_paths)
    agg = aggregate(variants)

    # Re-write each variant with structural scores, plus aggregate.
    for v, path in zip(variants, per_variant_paths):
        out_path = Path(args.out_json).parent / f"{path.stem}_rescored.json"
        with open(out_path, "w") as f:
            json.dump(v, f, indent=2)
    with open(args.out_json, "w") as f:
        json.dump(agg, f, indent=2)

    # Pretty table
    print("\n=== STRUCTURAL vs LLM-AS-JUDGE SCORES (3 variants, 30 queries) ===",
          flush=True)
    print(f"{'type':<16} {'n':<3} "
          f"{'A L/S':<10} {'B L/S':<10} {'C L/S':<10} "
          f"{'D L/S':<10} {'E L/S':<10}", flush=True)
    for t, s in agg["per_type"].items():
        line = f"{t:<16} {s['n']:<3} "
        for c in "ABCDE":
            ll = s.get(f"{c}_score")
            ss = s.get(f"{c}_structural_mean")
            ss_str = f"{ss:.2f}" if ss is not None else "-"
            line += f"{ll:.2f}/{ss_str:<5} "
        print(line, flush=True)
    print(f"\nwrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
