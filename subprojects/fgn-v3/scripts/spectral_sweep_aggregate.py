"""Aggregate the rank+accuracy sweep results."""
import json
import sys
from collections import defaultdict


def main(path):
    rows = [json.loads(l) for l in open(path)]
    by_run = defaultdict(list)
    for r in rows:
        by_run[r["run_label"]].append(r)

    print(f"{'Scenario':<22} {'FinalRank':>10} {'Anisotropy':>11} "
          f"{'NonUnreachAcc':>14} {'OverallAcc':>11} {'Degen':>7}")
    print("-" * 85)

    for label, recs in by_run.items():
        steps = [r for r in recs if "eff_rank" in r]
        summs = [r for r in recs if r.get("depth_type") == "summary"]
        if steps:
            steps.sort(key=lambda r: r["depth"])
            final = steps[-1]
        else:
            final = {}
        rank = final.get("eff_rank", 0)
        aniso = final.get("anisotropy_mean_cos", 0)
        if summs:
            s = summs[0]
            print(f"{label:<22} {rank:>10.1f} {aniso:>+11.3f} "
                  f"{s.get('acc_non_unreach', 0):>14.3f} "
                  f"{s.get('acc_total', 0):>11.3f} "
                  f"{s.get('degeneracy', 0):>7.3f}")
        else:
            print(f"{label:<22} {rank:>10.1f} {aniso:>+11.3f}  (no accuracy)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/spectral_sweep.jsonl")
