import os, re, json, random
os.environ['ME_MODE']='none'
import numpy as np
from gendev import (GenNet, gen_rollout, eval_gen, D_OUT, D_ACT, D_SYM, decode, DICT,
                    ALPHABET, EVENTS, EV_IDX, ACTIONS, D_IN, obs_vec, make_episode, score_rollout)

# PRIMITIVE_HIDDEN_LINEAGE_SYNTHESIS_V1 — Parts 2/3/6/7 infra.
# compile+validate arbitrary synthesized generic genomes; extract LABEL-FREE environmental evidence;
# run the lineage loop with ancestry / restart / shuffled-consequence conditions.

# ---------- compile + validate ----------
def compile_gen(genome):
    try:
        if not isinstance(genome,dict): return None,'not a dict'
        H=int(genome.get('hidden_dim',0))
        if not (2<=H<=128): return None,f'hidden {H}'
        for k in ('recurrent','input','output'):
            if k not in genome.get('weights',{}): return None,f'missing weights.{k}'
        net=GenNet(genome)
        rng=np.random.RandomState(0)
        for _ in range(30):
            out=net.step(rng.randn(D_IN).astype(np.float32))
            if not np.all(np.isfinite(out)) or np.max(np.abs(net.h))>50: return None,'unstable'
            net.update(cons=0.0)
        net.reset()
        return net,'ok'
    except Exception as e:
        return None,f'build_err:{type(e).__name__}:{e}'

# ---------- random generic genome (control + shuffled-evidence source + evo seed) ----------
def rand_gen_genome(seed):
    g=random.Random(seed)
    VARS=['pre','post','w','obsemb','pred','mod0','cons','act']
    def term():
        k=g.randint(1,3); return {'coef':round(g.uniform(0.02,0.6),3),'sign':g.choice([1,-1]),
                                  'factors':[g.choice(VARS) for _ in range(k)]}
    nmod=g.choice([0,1,2])
    dims=[[g.randrange(len(EVENTS)) for _ in range(g.randint(1,2))] for _ in range(nmod)]
    plastic={}
    for m in (['output'] if g.random()<0.6 else ['output','recurrent']):
        plastic[m]={'terms':[term() for _ in range(g.randint(1,3))],'clip':6.0}
        if g.random()<0.4: plastic[m]['traces']=[{'name':'e','decay':round(g.uniform(0.3,0.9),2),'factors':[g.choice(['pre','post','obsemb']),g.choice(['pre','post','mod0'])]}]
    return {'hidden_dim':g.choice([32,48,64]),'n_mod':nmod,
            'weights':{'recurrent':{'gen':g.choice(['dense','sparse']),'seed':g.randint(0,9999),'scale':round(g.uniform(0.5,1.1),2),'spectral_radius':round(g.uniform(0.8,1.1),2),'sparsity':0.2},
                       'input':{'gen':'dense','seed':g.randint(0,9999),'scale':round(g.uniform(0.4,0.9),2)},
                       'output':{'gen':'dense','seed':g.randint(0,9999),'scale':round(g.uniform(0.02,0.2),3)},
                       'mod_input':{'gen':'select','dims':dims,'scale':1.0}},
            'dynamics':{'activation':'tanh','gain':round(g.uniform(0.9,1.3),2),'leak':round(g.uniform(0.1,0.3),2),'noise':0.0},
            'mod_activation':g.choice(['relu','sigmoid']),'plastic':plastic,'init_state':{'gen':'zeros'}}

# ---------- LABEL-FREE environmental evidence ----------
def run_and_evidence(genome, seed=0, neps=80):
    net,msg=compile_gen(genome)
    if net is None: return {'ok':False,'msg':msg}, f"GENOME FAILED TO COMPILE/STABILIZE: {msg}"
    rng=random.Random(seed); per=[]; trajs=[]; dW=[]; modact={e:[] for e in EVENTS}
    for ep in range(neps):
        net.reset(); held=(ep%3==0); steps=make_episode(rng, held=held)
        traj=[]; Wprev=net.Wout.copy()
        acts=[]; preds=[]
        for st in steps:
            o=obs_vec(st['ev'],st['sym'],st['dis']); out=net.step(o)
            if not np.all(np.isfinite(out)): break
            a=ACTIONS[int(np.argmax(out[:D_ACT]))]; sp=decode(out[D_ACT:])
            acts.append(a); preds.append(sp)
            if net.nmod: modact[st['ev']].append(float(net.mods[0]))
            traj.append((st['ev'], st['sym'] if st['sym'] else '-', a, sp, round(float(np.linalg.norm(net.h)),2)))
            net.update(cons=0.0)
            dW.append(float(np.linalg.norm(net.Wout-Wprev))); Wprev=net.Wout.copy()
        while len(acts)<len(steps): acts.append('HOLD'); preds.append(None)
        r=score_rollout(steps,acts,preds); per.append(r)
        if ep<3: trajs.append(traj)
    third=max(1,neps//3); av=lambda k: round(float(np.mean([d[k] for d in per[-third:]])),3)
    metrics={'ok':True,'survival':av('survival'),'probe':av('probe'),'false_resist':av('false_resist'),
             'valid_update':av('valid_update'),'invalid_reject':av('invalid_reject')}
    # ---- format evidence text (NO labels, NO diagnosis) ----
    L=[f"ORGANISM BEHAVIOR OVER {neps} PROCEDURAL EPISODES (mixed familiar + never-before-seen symbols):"]
    L.append(f"terminal survival rate = {metrics['survival']:.2f}  (organism dies after a few wrong PROBE answers)")
    L.append(f"mean |change in output weights| per step = {round(float(np.mean(dW)) if dW else 0,4)}")
    if net.nmod:
        L.append("modulatory-neuron mean activity by event: "+", ".join(f"{e}={round(float(np.mean(v)),2) if v else 0}" for e,v in modact.items()))
    L.append("sample trajectories (event, shown_symbol, organism_action, organism_output_symbol, state_norm):")
    for ti,tr in enumerate(trajs):
        L.append(f"  episode {ti}:")
        for step in tr: L.append(f"    {step[0]:11s} shown={step[1]:8s} act={step[2]:7s} out={step[3]:8s} |h|={step[4]}")
    return metrics, "\n".join(L)

# ---------- prompts ----------
def gen0_prompt():
    return open('/home/pokazge/NativeEntity/gen0_prompt.txt').read()

def parse_genome(text):
    bs=re.findall(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if not bs: bs=re.findall(r'(\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})', text, re.DOTALL)
    for b in bs[::-1]:
        try:
            g=json.loads(b)
            if isinstance(g,dict) and 'hidden_dim' in g and 'plastic' in g: return g
        except: pass
    return None

# ---------- Qwen server synthesis ----------
import urllib.request
def qwen_gen(prompt, mx=1500, temp=0.0):
    body=json.dumps({'messages':[{'role':'user','content':prompt}],'max_new':mx,'temp':temp}).encode()
    req=urllib.request.Request('http://localhost:8765/gen',data=body,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=1200) as r: return json.loads(r.read())['text']

def intergen_prompt(evidence_text, prev_genome=None, history=None):
    P=gen0_prompt()+"\n\n=== FEEDBACK FROM YOUR PREVIOUS ORGANISM IN THE ENVIRONMENT ===\n"+evidence_text
    if prev_genome is not None:
        P+="\n\nYOUR PREVIOUS GENOME (revise it to survive better; you may change structure and update equations):\n```json\n"+json.dumps(prev_genome)+"\n```"
    if history: P+="\n\nSURVIVAL over your lineage so far (oldest→newest): "+history
    P+="\n\nOutput ONE improved genome as a JSON object in a ```json block, then (a) predicted behavior (b) failure modes (c) which environmental signal drives each plastic term."
    return P

def run_lineage(condition, ngen=10, seed0=1, temp=0.0, tag=''):
    # condition: 'ancestry' | 'restart' | 'shuffled'
    print(f"### LINEAGE [{condition}] {tag} ngen={ngen} temp={temp}", flush=True)
    prompt=gen0_prompt(); prev=None; hist=[]; best=-1; results=[]
    for g in range(ngen):
        txt=qwen_gen(prompt, temp=temp); genome=parse_genome(txt)
        if genome is None:
            print(f"  gen{g}: PARSE_FAIL", flush=True); results.append({'gen':g,'ok':False});
            prompt=intergen_prompt("Your previous output could not be parsed as a valid genome JSON.", prev, ", ".join(f"{h:.2f}" for h in hist)); continue
        metrics,ev=run_and_evidence(genome, seed=100+g)
        surv=metrics.get('survival',0.0) if metrics.get('ok') else 0.0
        hist.append(surv); best=max(best,surv); results.append({'gen':g,'ok':metrics.get('ok'),'survival':surv,'probe':metrics.get('probe')})
        print(f"  gen{g}: survival={surv:.2f} probe={metrics.get('probe',0)} best={best:.2f} {'' if metrics.get('ok') else metrics.get('msg')}", flush=True)
        # build evidence for NEXT generation according to condition
        if condition=='shuffled':
            ev_next,_=None,None
            rmet,rev=run_and_evidence(rand_gen_genome(9999+g*7), seed=100+g)   # evidence from an UNRELATED random genome
            ev_use=rev
        else:
            ev_use=ev
        prevg = genome if condition in ('ancestry','shuffled') else None       # restart: no previous genome
        prompt=intergen_prompt(ev_use, prevg, ", ".join(f"{h:.2f}" for h in hist))
        prev=genome
    print(f"### DONE [{condition}] best_survival={best:.2f} trajectory={[round(r.get('survival',0),2) for r in results]}", flush=True)
    return results

def qwen_gen0(n_greedy=5, n_sampled=5):
    print("### PART 2 — Qwen3.6-27B gen-0 population (generic DSL, no primitive names)", flush=True)
    survs=[]
    for i in range(n_greedy+n_sampled):
        temp=0.0 if i<n_greedy else 0.7
        try: txt=qwen_gen(gen0_prompt(), temp=temp)
        except Exception as e: print(f"  q{i}: GEN_ERR {e}", flush=True); continue
        g=parse_genome(txt)
        if g is None: print(f"  q{i}(t{temp}): PARSE_FAIL", flush=True); continue
        m,_=run_and_evidence(g, seed=7)
        s=m.get('survival',0) if m.get('ok') else 0.0
        survs.append(s); print(f"  q{i}(t{temp}): survival={s:.2f} probe={m.get('probe',0)} compile={'ok' if m.get('ok') else m.get('msg')}", flush=True)
    if survs: print(f"  >> Qwen gen-0 survival: median={np.median(survs):.2f} best={max(survs):.2f} n_valid={len(survs)}", flush=True)
    return survs

if __name__=='__main__':
    stage=os.environ.get('LIN_STAGE','test')
    if stage=='test':
        txt=qwen_gen(gen0_prompt(), temp=0.0); g=parse_genome(txt)
        print("=== QWEN GEN-0 TEST parse:", 'OK' if g else 'FAIL', flush=True)
        if g:
            m,ev=run_and_evidence(g, seed=7)
            print("=== metrics:", m, flush=True); print(json.dumps(g)[:600], flush=True)
    elif stage=='gen0':
        qwen_gen0()
    elif stage=='lineage':
        allr={}
        for cond in ['ancestry','restart','shuffled']:
            allr[cond]=run_lineage(cond, ngen=int(os.environ.get('LIN_NGEN','10')), temp=float(os.environ.get('LIN_TEMP','0.5')), tag=cond)
        json.dump(allr, open('/home/pokazge/NativeEntity/lineage_results.json','w'), indent=1)
        print("=== LINEAGE SUMMARY (best survival per condition) ===", flush=True)
        for c,rs in allr.items(): print(f"  {c:9s}: trajectory={[round(r.get('survival',0),2) for r in rs]} best={max([r.get('survival',0) for r in rs]):.2f}", flush=True)
