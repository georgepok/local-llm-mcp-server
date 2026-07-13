import os, statistics as st, random
os.environ['ME_MODE']='none'
import famC   # FROZEN

# PART 6 O7 (GEN0_M3_NEGATIVE) — the definitive construction test: among seeds whose INITIAL pool is M3-NEGATIVE at
# t0, does operational reconstruction (M3) arise MORE under real selection (C1) than under shuffled(C2)/no-sel(C6)?
# If yes -> genuine construction (selection builds M3 where absent at t0). If not -> no construction, M4 = standing variation.
THR=0.02
def m3_of(pops, s):
    return max((famC.leave_one_out(famC.express(p, s*71+i*13+5), seedv=s*17+3)[0] for i,p in enumerate(pops)), default=0.0)
def inc(v): return st.mean(1.0 if x>THR else 0.0 for x in v) if v else 0.0
def boot(d,B=4000):
    if not d: return 0.0,0.0,0.0
    r=random.Random(0); ms=sorted(st.mean(r.choice(d) for _ in range(len(d))) for _ in range(B)); return round(st.mean(d),4), round(ms[int(.025*B)],4), round(ms[int(.975*B)],4)

if __name__=='__main__':
    SEEDS=list(range(96))
    print("=== PART-6 O7: GEN0_M3_NEGATIVE — does M3 arise under selection where ABSENT at t0? (96 seeds) ===", flush=True)
    t0neg=[]
    for s in SEEDS:
        _, init_pops = famC.evolve(s,'C1')
        if m3_of(init_pops,s)<=THR: t0neg.append(s)          # keep only t0-M3-NEGATIVE seeds
    print(f"  t0-M3-negative seeds: {len(t0neg)}/96", flush=True)
    fin={c:{s:m3_of(famC.evolve(s,c)[0], s) for s in t0neg} for c in ['C1','C2','C6']}
    for c in ['C1','C2','C6']:
        v=list(fin[c].values()); print(f"  among t0-negative: {c} final M3 incidence = {inc(v):.3f} ({sum(1 for x in v if x>THR)}/{len(v)}) mean={st.mean(v):.4f}", flush=True)
    for b in ['C2','C6']:
        d=[fin['C1'][s]-fin[b][s] for s in t0neg]; m,lo,hi=boot(d)
        print(f"  O7 construction test C1-{b} (t0-negative seeds): {m:+.4f} 95%CI[{lo:+.4f},{hi:+.4f}] excl0={lo>0}", flush=True)
    print("  => if C1>>C2/C6 among t0-negative (CI excl 0) -> CONSTRUCTION; else -> M4 is standing-variation retention only.", flush=True)
    print("=== FAMC_O7_DONE ===", flush=True)
