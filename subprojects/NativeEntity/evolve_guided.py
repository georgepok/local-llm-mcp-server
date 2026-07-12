import os, json, random
os.environ['ME_MODE']='none'
import numpy as np
from evolve import evolve
from lineage import compile_gen, parse_genome, qwen_gen
from gendev_e3 import make_steps_e3, obs_e3, eval_e3
from gendev import D_ACT, DICT, ALPHABET

# PART 6 — LLM-guided variation. LLM proposes descendants from a parent + its RAW trajectory (no diagnosis,
# no correct-answer label); the ENVIRONMENT still selects. Conditions: pure (no LLM) / qwen / qwen_neutral
# (C8: consequences stripped from the trajectory -> if it works equally, LLM isn't using env feedback = Failure E).
GENP=open('/home/pokazge/NativeEntity/gen0_e3_prompt.txt').read()

def raw_traj(genome, n=2, seed=0):
    net,msg=compile_gen(genome)
    if net is None: return "(this organism did not run stably)"
    rng=random.Random(seed); L=[]
    for ep in range(n):
        net.reset(); steps,rev=make_steps_e3(rng); L.append(f"episode (reward reverses ~trial {rev}):")
        for st in steps:
            out=net.step(obs_e3(st['ev'],st['sym']))
            if not np.all(np.isfinite(out)): L.append("  (went unstable)"); break
            osym=ALPHABET[int(np.argmax(DICT@out[D_ACT:]))]
            if st['ev']=='CHOICE': L.append(f"  CHOICE shown={ALPHABET[st['sym']]}")
            elif 'REWARD' in st['ev']: L.append(f"  REWARD cons={st['cons']:+.0f}")
            elif st['probe']: L.append(f"  PROBE output={osym}")
            net.update(cons=st['cons'])
    return "\n".join(L)

def make_qwen_guided(neutral=False, temp=0.7):
    def gf(parent, pfit, nid):
        traj=raw_traj(parent)
        if neutral: traj="\n".join(l for l in traj.split("\n") if 'cons' not in l)     # C8 neutral control
        prompt=(GENP+f"\n\n=== AN EXISTING ORGANISM AND HOW IT BEHAVED IN THE ENVIRONMENT ===\nGenome:\n```json\n"
                +json.dumps(parent)+f"\n```\nIts scalar fitness was {pfit:.3f}. Raw behavior:\n{traj}\n\n"
                "Propose ONE MODIFIED descendant genome that might survive better (change coefficients, traces, terms, or structure). Output ONE genome JSON in a ```json block.")
        try:
            g=parse_genome(qwen_gen(prompt, temp=temp, mx=1100))
            if g and isinstance(g,dict) and 'hidden_dim' in g and 'plastic' in g:
                g['_id']=nid; g['_parents']=[parent.get('_id')]; return g
        except Exception: pass
        return None
    return gf

if __name__=='__main__':
    N=int(os.environ.get('EV_N','40')); G=int(os.environ.get('EV_G','18')); sd=int(os.environ.get('EV_SEED','0'))
    conds=os.environ.get('EV_GUIDE','pure,qwen,qwen_neutral').split(',')
    out={}
    for cond in conds:
        gf=make_qwen_guided(neutral=(cond=='qwen_neutral')) if cond in ('qwen','qwen_neutral') else None
        print(f"### GUIDED CONDITION {cond} (N={N} gens={G} seed={sd})", flush=True)
        h,bg=evolve('pure', N=N, gens=G, seed=sd, guided_fn=gf, guided_frac=0.15)   # env-selection always; LLM only varies
        base=eval_e3(bg,neps=80) if bg else {'probe_acc':0.0}
        abl=eval_e3(bg,neps=80,ablate_cons=True) if bg else {'probe_acc':0.0}
        out[cond]={'best':max(x['best'] for x in h),'probe':base['probe_acc'],'rewdep':round(base['probe_acc']-abl['probe_acc'],3),'dep_trace':[x['dep'] for x in h]}
        print(f"  {cond}: best_fit={out[cond]['best']:.3f} | probe={base['probe_acc']:.2f} REWARD-DEP={out[cond]['rewdep']:+.2f} | dep_trace={[x['dep'] for x in h][::max(1,G//8)]}", flush=True)
    json.dump(out, open('/home/pokazge/NativeEntity/evolve_guided_results.json','w'))
    print("=== GUIDED SUMMARY (does Qwen-guided variation ACCELERATE construction vs pure? neutral=C8 control) ===", flush=True)
    for c,r in out.items(): print(f"  {c:13s}: best={r['best']:.3f} reward-dep={r['rewdep']:+.2f}", flush=True)
