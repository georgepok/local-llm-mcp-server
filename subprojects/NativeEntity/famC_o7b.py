import os, statistics as st, random
os.environ['ME_MODE']='none'
import famC
THR=0.02
def m3_of(pops, s):
    return max((famC.leave_one_out(famC.express(p, s*71+i*13+5), seedv=s*17+3)[0] for i,p in enumerate(pops)), default=0.0)
def inc(v): return st.mean(1.0 if x>THR else 0.0 for x in v) if v else 0.0
def boot(d,B=4000):
    if not d: return 0.0,0.0,0.0
    r=random.Random(0); ms=sorted(st.mean(r.choice(d) for _ in range(len(d))) for _ in range(B)); return round(st.mean(d),4), round(ms[int(.025*B)],4), round(ms[int(.975*B)],4)
if __name__=='__main__':
    SEEDS=list(range(96))
    t0neg=[s for s in SEEDS if m3_of(famC.evolve(s,'C1')[1], s)<=THR]
    print(f"=== O7 MODE test: is t0-negative construction MUTATION-driven or ECOLOGICAL? ({len(t0neg)} t0-neg seeds) ===", flush=True)
    c1_mut=[m3_of(famC.evolve(s,'C1')[0], s) for s in t0neg]                 # C1 with mutation (MUT=0.06)
    save=famC.CFG.MUT; famC.CFG.MUT=0.0
    c1_nomut=[m3_of(famC.evolve(s,'C1')[0], s) for s in t0neg]               # C1 selection, NO mutation
    famC.CFG.MUT=save
    print(f"  C1 MUT=0.06 : M3 incidence {inc(c1_mut):.3f} ({sum(1 for x in c1_mut if x>THR)}/{len(t0neg)})", flush=True)
    print(f"  C1 MUT=0    : M3 incidence {inc(c1_nomut):.3f} ({sum(1 for x in c1_nomut if x>THR)}/{len(t0neg)})", flush=True)
    m,lo,hi=boot([a-b for a,b in zip(c1_mut,c1_nomut)])
    print(f"  paired (mut - nomut) among t0-neg = {m:+.4f} 95%CI[{lo:+.4f},{hi:+.4f}] excl0={lo>0}", flush=True)
    print("  => mut>>nomut (CI excl 0) -> MUTATION-driven (translational) construction; mut~=nomut -> ECOLOGICAL (selection+composition, no topology mutation).", flush=True)
    print("=== FAMC_O7B_DONE ===", flush=True)
