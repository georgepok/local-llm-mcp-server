"""MISSION probe — does the seeded native goal-field PERSIST across turns, or erode?

Mission = temporal stability of the field. Seed z_G from the FIRST turn only, then measure, at each
LATER turn t, whether the native basin still tracks the goal:
  engagement_t : cos(h_pos_t, z_G) > cos(h_drift_t, z_G)   (still pulling toward goal vs drift)
  identity_t   : cos(h_pos_t, z_G) > cos(h_pos'_t, z_G)     (still THIS goal vs another goal)
If accuracy holds flat across t, the native field is temporally stable on its own (mission ~ native).
If it degrades with t, the seed erodes -> active holding (the Liquid's slow dynamics) is required.
Held-out goals only; native basin only (no learning). CPU, cached reps.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/home/pokazge/checkpoints/induction_repr_30b_big.pt")
    ap.add_argument("--n_heldout_goals", type=int, default=40)
    args = ap.parse_args()
    data = torch.load(args.cache, weights_only=False, map_location="cpu")
    goals = sorted(data.keys())
    rng = np.random.default_rng(7)
    ho = sorted(rng.choice(goals, size=min(args.n_heldout_goals, len(goals)), replace=False).tolist())

    def nat(h, z): return float((F.normalize(h, dim=-1) * z).sum(-1))

    # seed z_G from TURN 0 goal-serving state only (earliest seed)
    zG = {g: F.normalize(data[g][0][0], dim=0) for g in ho}
    maxT = max(len(data[g]) for g in ho)
    print(f"[mission] held-out goals={len(ho)}  seed z_G = turn-0 only", flush=True)
    print(f"{'turn':>4} {'engagement_t':>13} {'identity_t':>11} {'n':>4}", flush=True)
    for t in range(1, maxT):                                          # later turns (seed was turn 0)
        eng, idn, n = [], [], 0
        for g in ho:
            if t >= len(data[g]):
                continue
            z = zG[g]; hp = data[g][t][0]; hd = data[g][t][1]
            eng.append(nat(hp, z) > nat(hd, z))
            others = [x for x in ho if x != g and t < len(data[x])]
            if others:
                o = others[rng.integers(len(others))]
                hpo = data[o][rng.integers(len(data[o]))][0]
                idn.append(nat(hp, z) > nat(hpo, z))
            n += 1
        print(f"{t:>4} {np.mean(eng):>13.3f} {np.mean(idn):>11.3f} {n:>4}", flush=True)
    print("[mission] flat across t = native field temporally stable; declining = Liquid holding needed", flush=True)
    print("[ind-mission] === ALL_DONE ===", flush=True)


if __name__ == "__main__":
    main()
