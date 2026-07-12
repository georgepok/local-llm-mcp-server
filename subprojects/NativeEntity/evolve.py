import os, json, random, copy
os.environ['ME_MODE']='none'
import numpy as np
from gendev_e3 import eval_e3, HANDBUILT_E3
from lineage import compile_gen

# POPULATIONAL_DEVELOPMENTAL_SYNTHESIS_V1 — population evolution over the generic DSL on E3.
# PRIMARY Q: can environmental SELECTION + mutation + crossover + inheritance ASSEMBLE a delayed-credit
# mechanism from partial genomes, WITHOUT any single synthesizer deriving it? Strongest control = SHUFFLED_FITNESS.
# No named primitives; the hand-built Claude rule is a hidden feasibility reference only.
np.seterr(over='ignore', invalid='ignore')

TERM_VARS=['pre','post','obsemb','pred','cons','w','mod0']          # + trace names
TRACE_VARS=['pre','post','obsemb','pred','cons','mod0']
EVDIMS=[0,1,2,3,4,5]

def _gs(rng,which):
    if which=='recurrent':
        return {'gen':rng.choice(['dense','sparse']),'seed':rng.randint(0,99999),'scale':round(rng.uniform(0.4,1.1),2),
                'spectral_radius':round(rng.uniform(0.6,1.1),2),'diag':round(rng.uniform(0.0,0.8),2),'sparsity':round(rng.uniform(0.1,0.4),2)}
    return {'gen':'dense','seed':rng.randint(0,99999),'scale':round(rng.uniform(0.05,1.0),2)}

def sample_genome(rng, gid=0):
    nmod=rng.choice([0,0,1,2])
    dims=[[rng.choice(EVDIMS) for _ in range(rng.randint(1,2))] for _ in range(nmod)]
    # traces: sometimes present (a building block for delayed credit)
    traces=[]
    for _ in range(rng.choice([0,0,1,1,2])):
        traces.append({'name':f"t{len(traces)}",'decay':round(rng.uniform(0.2,0.9),2),
                       'factors':[rng.choice(TRACE_VARS) for _ in range(rng.randint(1,2))]})
    tnames=[t['name'] for t in traces]
    pool=TERM_VARS+tnames
    terms=[]
    for _ in range(rng.randint(1,3)):
        terms.append({'coef':round(rng.uniform(0.05,1.0),3),'sign':rng.choice([1,-1]),
                      'factors':[rng.choice(pool) for _ in range(rng.randint(1,3))]})
    g={'hidden_dim':rng.choice([16,24,32,48,64]),'n_mod':nmod,
       'weights':{'recurrent':_gs(rng,'recurrent'),'input':_gs(rng,'i'),'output':_gs(rng,'o'),
                  'mod_input':{'gen':'select','dims':dims,'scale':1.0}},
       'dynamics':{'activation':rng.choice(['tanh','relu']),'gain':round(rng.uniform(0.8,1.8),2),
                   'leak':round(rng.uniform(0.05,0.4),2),'noise':round(rng.uniform(0,0.03),3)},
       'mod_activation':rng.choice(['relu','sigmoid','tanh']),
       'plastic':{'output':{'traces':traces,'terms':terms,'clip':round(rng.uniform(1.0,6.0),1)}},
       'init_state':{'gen':'zeros'},'_id':gid,'_parents':[]}
    return g

def _tnames(g): return [t['name'] for t in g['plastic'].get('output',{}).get('traces',[])]
def mutate(g, rng, nid):
    g=copy.deepcopy(g); g['_parents']=[g.get('_id')]; g['_id']=nid
    out=g['plastic']['output']; kind=rng.choice(
        ['coef','decay','leak','sr','diag','gain','addterm','delterm','changefactor','sign',
         'addtrace','deltrace','tracefactor','moddims','hidden','addmod','activation','clip','readscale'])
    try:
        if kind=='coef' and out['terms']:
            t=rng.choice(out['terms']); t['coef']=round(max(0.01,t['coef']*rng.uniform(0.5,2)),3)
        elif kind=='decay' and out['traces']:
            t=rng.choice(out['traces']); t['decay']=round(min(0.98,max(0.05,t['decay']+rng.uniform(-0.2,0.2))),2)
        elif kind=='leak': g['dynamics']['leak']=round(min(0.6,max(0.02,g['dynamics']['leak']+rng.uniform(-0.1,0.1))),2)
        elif kind=='sr': g['weights']['recurrent']['spectral_radius']=round(min(1.4,max(0.5,g['weights']['recurrent'].get('spectral_radius',0.9)+rng.uniform(-0.2,0.2))),2)
        elif kind=='diag': g['weights']['recurrent']['diag']=round(min(0.9,max(0.0,g['weights']['recurrent'].get('diag',0.0)+rng.uniform(-0.2,0.2))),2)
        elif kind=='gain': g['dynamics']['gain']=round(min(2.5,max(0.5,g['dynamics']['gain']+rng.uniform(-0.3,0.3))),2)
        elif kind=='addterm':
            out['terms'].append({'coef':round(rng.uniform(0.05,0.8),3),'sign':rng.choice([1,-1]),'factors':[rng.choice(TERM_VARS+_tnames(g)) for _ in range(rng.randint(1,3))]})
        elif kind=='delterm' and len(out['terms'])>1: out['terms'].pop(rng.randrange(len(out['terms'])))
        elif kind=='changefactor' and out['terms']:
            t=rng.choice(out['terms']); t['factors'][rng.randrange(len(t['factors']))]=rng.choice(TERM_VARS+_tnames(g))
        elif kind=='sign' and out['terms']: t=rng.choice(out['terms']); t['sign']=-t.get('sign',1)
        elif kind=='addtrace':
            nm=f"t{len(out['traces'])}{rng.randint(0,9)}"; out['traces'].append({'name':nm,'decay':round(rng.uniform(0.2,0.9),2),'factors':[rng.choice(TRACE_VARS) for _ in range(rng.randint(1,2))]})
        elif kind=='deltrace' and out['traces']:
            rm=out['traces'].pop(rng.randrange(len(out['traces'])))['name']
            for t in out['terms']: t['factors']=[f if f!=rm else rng.choice(TERM_VARS) for f in t['factors']]
        elif kind=='tracefactor' and out['traces']:
            t=rng.choice(out['traces']); t['factors'][rng.randrange(len(t['factors']))]=rng.choice(TRACE_VARS)
        elif kind=='moddims' and g['n_mod']>0:
            g['weights']['mod_input']['dims']=[[rng.choice(EVDIMS) for _ in range(rng.randint(1,2))] for _ in range(g['n_mod'])]
        elif kind=='hidden': g['hidden_dim']=rng.choice([16,24,32,48,64])
        elif kind=='addmod': g['n_mod']=min(3,g['n_mod']+1); g['weights']['mod_input']['dims']=[[rng.choice(EVDIMS)] for _ in range(g['n_mod'])]
        elif kind=='activation': g['dynamics']['activation']=rng.choice(['tanh','relu'])
        elif kind=='clip': out['clip']=round(min(8.0,max(0.5,out['clip']*rng.uniform(0.5,2))),1)
        elif kind=='readscale': g['weights']['output']['scale']=round(max(0.01,g['weights']['output']['scale']*rng.uniform(0.4,2)),3)
    except Exception: pass
    return g

def crossover(a, b, rng, nid):
    c=copy.deepcopy(a); c['_parents']=[a.get('_id'),b.get('_id')]; c['_id']=nid
    block=rng.choice(['traces','terms','dynamics','recurrent'])
    try:
        if block=='traces': c['plastic']['output']['traces']=copy.deepcopy(b['plastic']['output']['traces'])
        elif block=='terms': c['plastic']['output']['terms']=copy.deepcopy(b['plastic']['output']['terms'])
        elif block=='dynamics': c['dynamics']=copy.deepcopy(b['dynamics'])
        elif block=='recurrent': c['weights']['recurrent']=copy.deepcopy(b['weights']['recurrent'])
    except Exception: pass
    # repair: ensure term trace-factors reference existing traces
    tn=set(_tnames(c))
    for t in c['plastic']['output']['terms']:
        t['factors']=[f if (f in TERM_VARS or f in tn) else rng.choice(TERM_VARS) for f in t['factors']]
    return c

def mech_signature(g):
    """structural presence of delayed-credit building blocks (fast proxy; causal M1-M7 ablation is Part 8)."""
    out=g.get('plastic',{}).get('output',{}); trs=out.get('traces',[]); terms=out.get('terms',[])
    has_trace_obs=any('obsemb' in t['factors'] for t in trs)                       # M1 proxy: choice-time trace of symbol
    tn=set(t['name'] for t in trs)
    has_cons_term=any('cons' in t['factors'] for t in terms)                        # M3/M4 proxy: consequence modulates update
    has_cons_x_trace=any(('cons' in t['factors']) and any(f in tn for f in t['factors']) for t in terms)  # M3xM2: reward x trace
    has_decay=any(('w' in t['factors']) and t.get('sign',1)<0 for t in terms)       # M6 proxy: forgetting/overwrite
    return (has_trace_obs,has_cons_term,has_cons_x_trace,has_decay)

def fitness(g, gen):
    net,msg=compile_gen(g)
    if net is None: return -1.0, {'valid':False}
    r=eval_e3(g, neps=30, seed=1000+gen)
    ra=eval_e3(g, neps=30, seed=1000+gen, ablate_cons=True)     # SAME worlds w/o reward -> isolates reward-DEPENDENCE
    rh=eval_e3(g, neps=20, held=True, seed=5000+gen)
    dep=max(0.0, r['probe_acc']-ra['probe_acc'])               # causal reward-dependence (kills reward-independent shortcut)
    out=g['plastic']['output']; comp=len(out.get('terms',[]))+len(out.get('traces',[]))+g['hidden_dim']/48.0
    score=0.15*r['probe_acc']+0.2*r['post_rev']+0.15*rh['probe_acc']+0.7*dep-0.002*comp   # fitness REQUIRES reward-use
    return round(score,4), {'valid':True,'probe':r['probe_acc'],'post':r['post_rev'],'held':rh['probe_acc'],'dep':round(dep,3),'sig':mech_signature(g)}

def evolve(condition, N=64, gens=40, seed=0, elite_frac=0.1, seed_handbuilt=False, guided_fn=None, guided_frac=0.25):
    rng=random.Random(seed); gid=[0]
    def newid(): gid[0]+=1; return gid[0]
    pop=[sample_genome(rng, newid()) for _ in range(N)]
    if seed_handbuilt:                          # hidden feasibility control: does selection PRESERVE+PROPAGATE a working rule?
        hb=copy.deepcopy(HANDBUILT_E3); hb['_id']=newid(); hb['_parents']=['HANDBUILT']; pop[0]=hb
    hist=[]; best_genome=None; best_fit=-9
    for gen in range(gens):
        scored=[(fitness(g,gen),g) for g in pop]
        real=[s[0][0] for s in scored]                              # REAL fitness (always tracked)
        # selection signal depends on condition
        if condition=='shuffled':
            sel=real[:]; rng.shuffle(sel)
        elif condition=='random':
            sel=[rng.random() for _ in real]
        else:
            sel=real[:]
        order=sorted(range(N), key=lambda i:-sel[i])
        ne=max(1,int(N*elite_frac))
        # diversity: cap elites per mechanism-signature family
        elites=[]; seen={}
        for i in order:
            sig=scored[i][0][1].get('sig'); c=seen.get(sig,0)
            if c<max(1,ne//3): elites.append(pop[i]); seen[sig]=c+1
            if len(elites)>=ne: break
        if not elites: elites=[pop[order[0]]]
        # novelty protection: also keep the most component-rich distinct signatures (guards building blocks
        # that are neutral-in-isolation from being erased before they can recombine)
        rich=sorted(range(N), key=lambda i:-sum(scored[i][0][1].get('sig',(0,0,0,0))))
        nkept={mech_signature(e) for e in elites}
        for i in rich:
            if len(elites)>=ne+max(2,N//12): break
            if scored[i][0][1].get('valid') and scored[i][0][1]['sig'] not in nkept:
                elites.append(pop[i]); nkept.add(scored[i][0][1]['sig'])
        def tourni():
            cands=rng.sample(range(N),min(4,N)); return max(cands,key=lambda i:sel[i])
        offspring=[]; gcount=0
        while len(offspring)<N-len(elites):
            if guided_fn is not None and rng.random()<guided_frac and gcount<max(2,int((N-len(elites))*guided_frac)):
                pi=tourni(); kid=guided_fn(copy.deepcopy(pop[pi]), real[pi], newid()); gcount+=1     # LLM proposes; env still selects
                if kid is not None: offspring.append(kid); continue
            if rng.random()<0.4:
                offspring.append(crossover(pop[tourni()],pop[tourni()],rng,newid()))
            else:
                offspring.append(mutate(pop[tourni()],rng,newid()))
        pop=elites+offspring
        # report on REAL fitness + mechanism-component frequencies
        valid=[s for s in scored if s[0][1].get('valid')]
        sigs=[s[0][1]['sig'] for s in valid]
        freq=[round(np.mean([sg[k] for sg in sigs]),2) if sigs else 0 for k in range(4)]
        best=max(real); med=float(np.median(real))
        bi=int(np.argmax(real))
        if real[bi]>best_fit: best_fit=real[bi]; best_genome=copy.deepcopy(pop[bi])
        dep=scored[bi][0][1].get('dep',0.0)                    # per-gen best genome's CAUSAL reward-dependence
        hist.append({'gen':gen,'best':round(best,3),'median':round(med,3),'valid':len(valid),
                     'dep':round(dep,3),'mech_M1_M3_M3xM2_M6':freq})
        if gen%5==0 or gen==gens-1:
            print(f"  [{condition}] gen{gen:2d}: best={best:.3f} median={med:.3f} valid={len(valid)}/{N} mech(trace_obs,cons,cons*trace,decay)={freq}", flush=True)
    return hist, best_genome

if __name__=='__main__':
    N=int(os.environ.get('EV_N','64')); G=int(os.environ.get('EV_G','40'))
    conds=os.environ.get('EV_CONDS','pure,shuffled,random').split(',')
    from gendev_e3 import eval_e3
    seeds=[int(s) for s in os.environ.get('EV_SEEDS','0').split(',')]
    allh={}; bestg={}
    for sd in seeds:
        for c in conds:
            key=f"{c}_s{sd}"
            print(f"### CONDITION {c} seed={sd} (N={N}, gens={G})", flush=True)
            allh[key],bestg[key]=evolve('pure' if c=='feasibility' else c, N=N, gens=G, seed=sd, seed_handbuilt=(c=='feasibility'))
    json.dump({'hist':allh,'best_genomes':bestg}, open('/home/pokazge/NativeEntity/evolve_results.json','w'))
    print("=== CAUSAL CHECK: best genome reward-dependence per condition/seed + gradual-assembly trace ===", flush=True)
    for key,g in bestg.items():
        if g is None: continue
        base=eval_e3(g, neps=80); abl=eval_e3(g, neps=80, ablate_cons=True)
        deptraj=[h['dep'] for h in allh[key]]
        print(f"  {key:14s}: best_fit={max(h['best'] for h in allh[key]):.3f} | probe normal={base['probe_acc']:.2f} cons-ABL={abl['probe_acc']:.2f} REWARD-DEP={base['probe_acc']-abl['probe_acc']:+.2f} | dep_trace={deptraj[::max(1,len(deptraj)//8)]}", flush=True)
