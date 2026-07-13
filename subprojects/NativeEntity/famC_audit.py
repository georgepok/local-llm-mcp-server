import os, statistics as st, random, math
os.environ['ME_MODE']='none'
import famC   # FROZEN — read-only import, not modified

# FAMC_M4_ORIGIN_AND_CAUSALITY_AUDIT_V1 (Parts 1/3/6/8) — is the held-out-confirmed M4 effect CONSTRUCTION or
# selection over hidden STANDING VARIATION? Decisive tests, famC frozen:
#  Part 3  t0 negative certificate: run the M3 assay on the DEVELOPED gen-0 pool (express -> leave_one_out). If
#          M3 present at t0 -> standing-variation, NOT construction.
#  Part 6  O1 (MUT=0.06, real selection+mutation) vs O2 (MUT=0, real selection NO mutation). If M3 needs mutation
#          (O1 >> O2) -> constructed; if O2 ~= O1 -> standing variation enriched by selection.
#  Part 8  classify each successful C1 seed: STANDING (t0 M3>0) vs CONSTRUCTED (t0 M3=0, final M3>0).
THR=0.02
def m3_of(pops, s):
    return max((famC.leave_one_out(famC.express(p, s*71+i*13+5), seedv=s*17+3)[0] for i,p in enumerate(pops)), default=0.0)
def inc(v): return st.mean(1.0 if x>THR else 0.0 for x in v)
def boot(d,B=4000):
    r=random.Random(0); ms=sorted(st.mean(r.choice(d) for _ in range(len(d))) for _ in range(B)); return round(st.mean(d),4), round(ms[int(.025*B)],4), round(ms[int(.975*B)],4)

if __name__=='__main__':
    SEEDS=list(range(48))
    print("=== FAMC_M4 ORIGIN AUDIT — construction vs standing variation (discovery seeds 0-47) ===", flush=True)
    t0=[]; fin=[]; rows=[]
    for s in SEEDS:
        pops, init_pops = famC.evolve(s,'C1')
        a=m3_of(init_pops,s); b=m3_of(pops,s); t0.append(a); fin.append(b); rows.append((s,round(a,4),round(b,4)))
    print(f"  PART 3 t0-certificate: gen-0 M3 incidence(C1) = {inc(t0):.3f} ({sum(1 for x in t0 if x>THR)}/48) | final M3 incidence = {inc(fin):.3f} ({sum(1 for x in fin if x>THR)}/48)", flush=True)
    succ=[(s,a,b) for (s,a,b) in zip(SEEDS,t0,fin) if b>THR]
    standing=[x for x in succ if x[1]>THR]; constructed=[x for x in succ if x[1]<=THR]
    print(f"  PART 8 origin of successful C1 seeds (n={len(succ)}): STANDING(t0 M3>0)={len(standing)}  CONSTRUCTED(t0 M3=0, final>0)={len(constructed)}", flush=True)
    # PART 6: O2 = real selection, NO mutation (CFG.MUT=0)
    save=famC.CFG.MUT; famC.CFG.MUT=0.0
    o2=[m3_of(famC.evolve(s,'C1')[0], s) for s in SEEDS]; famC.CFG.MUT=save
    print(f"  PART 6 O1(MUT={save}) final M3 incidence = {inc(fin):.3f} | O2(MUT=0, selection-no-mutation) M3 incidence = {inc(o2):.3f}", flush=True)
    m,lo,hi=boot([a-b for a,b in zip(fin,o2)])
    print(f"  PART 6 paired O1-O2 continuous-M3 diff = {m:+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]  excl0={lo>0}", flush=True)
    print(f"  INTERPRETATION: if t0 M3~0 AND CONSTRUCTED>>STANDING AND O1>>O2 -> M3 is CONSTRUCTED by mutation (not standing variation).", flush=True)
    print(f"                  if t0 M3>0 OR O2~=O1 -> STANDING-VARIATION enrichment (NOT endogenous construction).", flush=True)
    print(f"  raw (seed, t0_M3, final_M3) first 12: {rows[:12]}", flush=True)
    print("=== FAMC_AUDIT_DONE ===", flush=True)
