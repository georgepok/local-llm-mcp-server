"""Aggregate spectral_analysis.jsonl into readable tables."""
import json
import sys
from collections import defaultdict


def main(path):
    rows = [json.loads(l) for l in open(path)]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model_type"], r["run_label"])].append(r)
    # Sort by (model_type, step if parseable)
    def key(kv):
        mt, lab = kv[0]
        step = 0
        if "step" in lab:
            try:
                step = int(lab.split("step")[-1])
            except Exception:
                step = 99999
        elif "final" in lab:
            step = 10000
        return (mt, step)
    sorted_groups = sorted(groups.items(), key=key)

    print("=" * 100)
    print("EFFECTIVE RANK across depth (rows=run, cols=layer/ode_step)")
    print("=" * 100)
    for (mt, lab), recs in sorted_groups:
        recs.sort(key=lambda r: r["depth"])
        ranks = " ".join(f"{r['eff_rank']:>6.1f}" for r in recs)
        print(f"{lab:<22} {ranks}")

    print()
    print("=" * 100)
    print("ANISOTROPY (mean pairwise cos) across depth")
    print("=" * 100)
    for (mt, lab), recs in sorted_groups:
        recs.sort(key=lambda r: r["depth"])
        aniso = " ".join(f"{r['anisotropy_mean_cos']:>+.2f}" for r in recs)
        print(f"{lab:<22} {aniso}")

    print()
    print("=" * 100)
    print("SUMMARY: input rank, output rank, compression ratio")
    print("=" * 100)
    print(f"{'Run':<22} {'Input_rank':>10} {'Output_rank':>12} {'Compression':>12}")
    for (mt, lab), recs in sorted_groups:
        recs.sort(key=lambda r: r["depth"])
        in_r = recs[0]["eff_rank"]
        out_r = recs[-1]["eff_rank"]
        print(f"{lab:<22} {in_r:>10.1f} {out_r:>12.1f} {in_r/out_r:>11.1f}x")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/spectral.jsonl")
