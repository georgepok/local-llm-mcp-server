import os, json, random, copy
os.environ['ME_MODE']='none'
import numpy as np
from evolve import sample_genome, mutate, fitness
from lineage import compile_gen, parse_genome, qwen_gen
from gendev_e3 import make_steps_e3, obs_e3
from gendev import D_ACT, DICT, ALPHABET

# Q2 (Parts 6-8): does STRICT LOCAL-edit guidance make Qwen useful vs its disruptive full-rewrite mode?
# For evolved E3 parents: pure-mutation vs QWEN_LOCAL (1 atomic edit) vs QWEN_FULL. Metrics: edit distance,
# stability, delta-fitness, P(child>parent), mechanism retention. Part 8: QWEN_LOCAL real vs shuffled trajectory.
GENP=open('/home/pokazge/NativeEntity/gen0_e3_prompt.txt').read()

def gdist(a,b):
    d=0.0
    for k in ('hidden_dim','n_mod'): d+= 1.0*(a.get(k)!=b.get(k))
    da,db=a.get('dynamics',{}),b.get('dynamics',{})
    for k in ('gain','leak','activation'): d+= 1.0*(da.get(k)!=db.get(k))
    pa=a.get('plastic',{}).get('output',{}); pb=b.get('plastic',{}).get('output',{})
    d+= 2.0*abs(len(pa.get('terms',[]))-len(pb.get('terms',[]))) + 2.0*abs(len(pa.get('traces',[]))-len(pb.get('traces',[])))
    for ta,tb in zip(pa.get('terms',[]),pb.get('terms',[])):
        d+= 1.5*(ta.get('factors')!=tb.get('factors')) + 1.0*(abs(float(ta.get('coef',0))-float(tb.get('coef',0)))>0.02+0.02*abs(float(ta.get('coef',0))))
    return round(d,1)

def raw_traj(genome, seed=0, n=2):
    net,msg=compile_gen(genome)
    if net is None: return "(did not run)"
    rng=random.Random(seed); L=[]
    for ep in range(n):
        net.reset(); steps,rev=make_steps_e3(rng); L.append(f"episode (reversal ~trial {rev}):")
        for st in steps:
            out=net.step(obs_e3(st['ev'],st['sym']))
            if not np.all(np.isfinite(out)): break
            osym=ALPHABET[int(np.argmax(DICT@out[D_ACT:]))]
            if st['ev']=='CHOICE': L.append(f"  CHOICE shown={ALPHABET[st['sym']]}")
            elif 'REWARD' in st['ev']: L.append(f"  REWARD cons={st['cons']:+.0f}")
            elif st['probe']: L.append(f"  PROBE output={osym}")
            net.update(cons=st['cons'])
    return "\n".join(L)

LOCAL_RULES=("Make EXACTLY ONE atomic edit, chosen from: change ONE coefficient by at most 10%; add ONE term; "
 "remove ONE term; change ONE trace decay; change ONE modulatory input; change ONE connection. Keep EVERYTHING "
 "else identical to the parent. Output the FULL genome JSON (parent with only that one edit) in a ```json block.")

def qwen_child(parent, mode, traj_mode='real'):
    if mode=='full':
        tr=raw_traj(parent); instr="Propose ONE MODIFIED descendant genome that might survive better (any changes). Output ONE genome JSON in a ```json block."
    else:
        tr=raw_traj(parent) if traj_mode=='real' else ("\n".join(l for l in raw_traj(parent).split("\n") if 'cons' not in l) if traj_mode=='shuffled' else "(no trajectory provided)")
        instr=LOCAL_RULES
    prompt=GENP+f"\n\n=== PARENT ORGANISM ===\n```json\n{json.dumps(parent)}\n```\nIts fitness={fitness(parent,0)[0]:.3f}. Raw behavior:\n{tr}\n\n{instr}"
    try:
        g=parse_genome(qwen_gen(prompt, temp=0.6, mx=700))
        if g and isinstance(g,dict) and 'hidden_dim' in g and 'plastic' in g: return g
    except Exception: pass
    return None

def stats(children, parents):
    valid=[(c,p) for c,p in zip(children,parents) if c is not None]
    if not valid: return {'n':0}
    dists=[gdist(p,c) for c,p in valid]
    stab=[1 if compile_gen(c)[0] is not None else 0 for c,p in valid]
    df=[]; up=[]
    for c,p in valid:
        fc=fitness(c,0)[0]; fp=fitness(p,0)[0]; df.append(fc-fp); up.append(1 if fc>fp else 0)
    return {'n':len(valid),'parserate':round(len(valid)/len(children),2),'dist':round(float(np.median(dists)),1),
            'stable':round(float(np.mean(stab)),2),'dfit':round(float(np.mean(df)),3),'P_better':round(float(np.mean(up)),2)}

if __name__=='__main__':
    g=random.Random(0)
    NP=int(os.environ.get('Q2_NP','10'))          # parents = random genomes with accumulated mutations (partial structure)
    parents=[]
    for i in range(NP):
        p=sample_genome(g, i)
        for _ in range(g.randint(2,6)): p=mutate(p,g,1000+i)   # give parents some accumulated structure
        if compile_gen(p)[0] is not None: parents.append(p)
    print(f"=== Q2 LOCALITY: {len(parents)} evolved parents; pure-mut vs QWEN_LOCAL vs QWEN_FULL ===", flush=True)
    pure=[mutate(p,g,9000+i) for i,p in enumerate(parents)]
    print(f"  PURE_MUTATION   : {stats(pure,parents)}", flush=True)
    qloc=[qwen_child(p,'local','real') for p in parents]
    print(f"  QWEN_LOCAL(real): {stats(qloc,parents)}", flush=True)
    qful=[qwen_child(p,'full') for p in parents]
    print(f"  QWEN_FULL       : {stats(qful,parents)}", flush=True)
    # Part 8 counterfactual: QWEN_LOCAL real vs shuffled trajectory
    qsh=[qwen_child(p,'local','shuffled') for p in parents]
    print(f"  QWEN_LOCAL(shuf): {stats(qsh,parents)}", flush=True)
    print("=== DECISION: Case D (QWEN_LOCAL P_better>pure & >QWEN_FULL) / E (<=pure -> close) ; real>shuffled? ===", flush=True)
    print("=== Q2_DONE ===", flush=True)
