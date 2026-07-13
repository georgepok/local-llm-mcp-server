import os, random
os.environ['ME_MODE']='none'
import numpy as np
from initfair import best_fair, FAM
from worldlaws import CLAUDE, random_law
from worldlaws2 import STRONG

# DEFINITIVE fair re-score under VALIDATED protocol (universal leak + primordial-soup inits, GS->Case D confirmed).
# Re-scores meta-gen-1 (CLAUDE 24, orig near-0 init was confounded) + meta-gen-2 (STRONG 12) + 30 random, all fairly.
# Answers Part-12 enrichment: does LLM world-law synthesis enrich for dissipative self-maintaining organization (Case C/D)
# over random, and does ANY law reach the Case-D milestone under a fair, GS-validated protocol?
if __name__=='__main__':
    print("=== DEFINITIVE FAIR RE-SCORE (validated protocol; GS->D confirmed) ===", flush=True)
    def tally(name, laws):
        cases={}; cd=[]; cc=[]
        for nm,law in laws:
            r=best_fair(law); cases[r['case']]=cases.get(r['case'],0)+1
            if r['case']=='D': cd.append(nm)
            if r['case']=='C': cc.append((nm,round(r['inflow_dep'],2),round(r['org_caus'],2)))
            if r['case'] in ('C','D'): print(f"  [{name}] {nm:24s} {r['case']} loc={r['localization']:.2f} infdep={r['inflow_dep']:.2f} orgC={r['org_caus']:.2f} repC={r['repair_c']:.2f} fam={r['fam']}", flush=True)
        n=len(laws); cdcc=cases.get('C',0)+cases.get('D',0)
        print(f"  >>> {name}: n={n} cases={cases} Case-D={len(cd)} Case-C+D={cdcc} ({100*cdcc/n:.0f}%) D_hits={cd}", flush=True)
        return len(cd), cdcc, n
    llm=list(CLAUDE.items())+list(STRONG.items())
    rnd=[(f'random_{s}', random_law(2000+s)) for s in range(30)]
    print(f"  -- LLM-synthesized (Claude {len(CLAUDE)} + strong {len(STRONG)} = {len(llm)}) --", flush=True)
    ld,lcc,ln=tally('LLM', llm)
    print(f"  -- RANDOM (30) --", flush=True)
    rd,rcc,rn=tally('RAND', rnd)
    print(f"=== ENRICHMENT (fair): LLM Case-C+D={lcc}/{ln} ({100*lcc/ln:.0f}%) Case-D={ld} | RAND Case-C+D={rcc}/{rn} ({100*rcc/rn:.0f}%) Case-D={rd} ===", flush=True)
    print("=== MILESTONE: any Case-D beyond GS? / Part-12: does LLM synthesis enrich dissipative organization over random? ===", flush=True)
    print("=== FAIRALL_DONE ===", flush=True)
