import os, json, math, random
import numpy as np
from micro_entity import MicroNet, compile_and_check, ACTIONS, D_IN, D_OUT, ALPHABET

# ============================================================================
# Cross-environment quantitative eval: E2 (delayed conditional transform) and
# E3 (contextual reversal). Reuses the micro_entity compiler (now with real
# oja/hebbian/reward_hebb plasticity). Faithful to spec_e2.txt / spec_e3.txt.
# ============================================================================
np.seterr(over='ignore', invalid='ignore')
N_SYM = 16
SEED = int(os.environ.get('ME_SEED','0'))
TRAIN_SYM = list(range(12)); HELD_SYM = list(range(12,16))

# ---- cyclic symbol embeddings: successor is a fixed rotation in embedding space ----
def _cyclic_emb(freqs=(1,2,3,5)):
    E=np.zeros((N_SYM,8),np.float32)
    for i in range(N_SYM):
        v=[]
        for f in freqs:
            th=2*math.pi*f*i/N_SYM; v+=[math.cos(th),math.sin(th)]
        E[i]=v
    return (E/np.linalg.norm(E,axis=1,keepdims=True)).astype(np.float32)
EMB=_cyclic_emb()
_rg=np.random.RandomState(999)
CODE={'A':_rg.randn(8).astype(np.float32),'B':_rg.randn(8).astype(np.float32)}
CODE={k:v/np.linalg.norm(v) for k,v in CODE.items()}

def _succ(i): return (i+1)%N_SYM
def _pred(i): return (i-1)%N_SYM

# ============================== E2 ==============================
E2_EV=['COMMIT','FILLER','TRANSFORM_A','TRANSFORM_B','PROBE','NULL']
E2_IX={e:i for i,e in enumerate(E2_EV)}
def _obs_e2(ev, v8):
    o=np.zeros(14,np.float32); o[E2_IX[ev]]=1.0; o[6:14]=v8; return o

def make_ep_e2(rng, held=False, ncyc=3, nfill=(1,3)):
    syms=HELD_SYM if held else TRAIN_SYM
    K=rng.choice(syms); steps=[{'ev':'COMMIT','v8':EMB[K],'sidx':K,'ans':None}]
    for _ in range(ncyc):
        for _ in range(rng.randint(*nfill)):
            d=rng.choice(syms); steps.append({'ev':'FILLER','v8':EMB[d],'sidx':d,'ans':None})
        t=rng.choice(['A','B']); steps.append({'ev':'TRANSFORM_'+t,'v8':CODE[t],'sidx':None,'ans':None})
        ans=_succ(K) if t=='A' else _pred(K)
        steps.append({'ev':'PROBE','v8':np.zeros(8,np.float32),'sidx':None,'ans':ans})
    return steps,K

def rollout_e2(net, steps, rng=None):
    preds=[]; hs=[]; unstable=False
    for st in steps:
        out=net.step(_obs_e2(st['ev'],st['v8']), rng)
        if not np.all(np.isfinite(out)): unstable=True; break
        preds.append(int(np.argmax(out[len(ACTIONS):]))); hs.append(net.h.copy())
        if net.plastic:
            bind_ok = st['ev'] in ('COMMIT','FILLER')      # build a general state->symbol un-embedding decoder
            net.plastic_update(event=st['ev'], sym_idx=st['sidx'], reward=0.0, bind_ok=bind_ok)
    while len(preds)<len(steps): preds.append(-1); hs.append(np.zeros(net.H,np.float32))
    return preds, unstable

def score_e2(steps, preds, vmax=3):
    V=vmax; alive=True; corr=[];
    for st,pr in zip(steps,preds):
        if st['ev']=='PROBE':
            ok=int(pr==st['ans']); corr.append(ok)
            if alive and not ok:
                V-=1
                if V<=0: alive=False
    pa=float(np.mean(corr)) if corr else 0.0
    return {'probe':pa,'survival':int(alive)}

# ============================== E3 ==============================
E3_EV=['CHOICE_PROMPT','REWARD_POS','REWARD_NEG','PROBE']
E3_IX={e:i for i,e in enumerate(E3_EV)}
def _obs_e3(ev, sym8, last_r, last_a):
    o=np.zeros(14,np.float32); o[E3_IX[ev]]=1.0; o[4:12]=sym8; o[12]=last_r; o[13]=last_a; return o

def make_plan_e3(rng, held=False, ntrial=14, rev=7):
    syms=HELD_SYM if held else TRAIN_SYM
    A,B=rng.sample(syms,2); rewarded=A; plan=[]
    for t in range(ntrial):
        if t==rev: rewarded=B if rewarded==A else A
        shown=A if t%2==0 else B
        plan.append({'shown':shown,'rewarded':rewarded,'A':A,'B':B,'probe':(t%3==2)})
    return plan, rev

def rollout_e3(net, plan, rng=None):
    last_r=0.0; last_a=0.0; rec=[]; unstable=False
    for tr in plan:
        # CHOICE
        out=net.step(_obs_e3('CHOICE_PROMPT',EMB[tr['shown']],last_r,last_a), rng)
        if not np.all(np.isfinite(out)): unstable=True; break
        act=int(np.argmax(out[:len(ACTIONS)]))     # 0=choose-A(bet shown IS rewarded), 1=choose-B(bet NOT)
        accept=(act==0); is_rew=(tr['shown']==tr['rewarded'])
        correct=int(accept==is_rew); reward=1.0 if correct else -1.0
        rprobe=int(np.argmax(out[len(ACTIONS):]))
        rec.append({'correct':correct,'rewarded':tr['rewarded'],'rprobe':rprobe,'probe':tr['probe']})
        if net.plastic:
            net.plastic_update(event='CHOICE_PROMPT', sym_idx=tr['shown'], reward=reward,
                               bind_ok=(accept and correct))   # bind shown when confirmed rewarded
        # REWARD feedback
        ev='REWARD_POS' if reward>0 else 'REWARD_NEG'
        out2=net.step(_obs_e3(ev,EMB[tr['shown']],reward,float(act)), rng)
        if not np.all(np.isfinite(out2)): unstable=True; break
        if net.plastic:
            net.plastic_update(event=ev, sym_idx=tr['shown'], reward=reward,
                               bind_ok=(is_rew and reward>0))
        last_r=reward; last_a=float(act)
    return rec, unstable

def score_e3(rec, rev):
    if not rec: return {'pre':0.0,'post':0.0,'delay':99,'probe':0.0,'overall':0.0}
    pre=[r['correct'] for i,r in enumerate(rec) if rev-4<=i<rev]
    post=[r['correct'] for i,r in enumerate(rec) if i>=len(rec)-4]
    # adaptation delay: trials after reversal until 3 consecutive correct
    delay=99; streak=0
    for i in range(rev,len(rec)):
        streak=streak+1 if rec[i]['correct'] else 0
        if streak>=3: delay=i-rev-2; break
    pr=[int(r['rprobe']==r['rewarded']) for r in rec if r['probe']]
    return {'pre':round(float(np.mean(pre)) if pre else 0,3),'post':round(float(np.mean(post)) if post else 0,3),
            'delay':delay,'probe':round(float(np.mean(pr)) if pr else 0,3),
            'overall':round(float(np.mean([r['correct'] for r in rec])),3)}

# ============================== eval harness ==============================
def eval_e2(genome, neps=120, held=False, seed=SEED, wnoise=0.0, hpert=0.0):
    net,msg=compile_and_check(genome)
    if net is None: return None
    if wnoise>0:
        wr=np.random.RandomState(seed+7); net.Wrec=net.Wrec+wnoise*wr.randn(*net.Wrec.shape).astype(np.float32); net.Wout=net.Wout+wnoise*wr.randn(*net.Wout.shape).astype(np.float32)
    rng=random.Random(seed); prng=np.random.RandomState(seed+3) if hpert>0 else None; per=[]
    for _ in range(neps):
        net.reset('reset'); steps,K=make_ep_e2(rng, held=held)
        preds,unst=rollout_e2(net,steps, rng=prng)
        per.append(score_e2(steps,preds))
    third=max(1,neps//3)
    av=lambda L,k: round(float(np.mean([d[k] for d in L])),3)
    return {'probe':av(per[-third:],'probe'),'survival':av(per[-third:],'survival'),'early_probe':av(per[:third],'probe')}

def eval_e3(genome, neps=120, held=False, seed=SEED, wnoise=0.0):
    net,msg=compile_and_check(genome)
    if net is None: return None
    if wnoise>0:
        wr=np.random.RandomState(seed+7); net.Wrec=net.Wrec+wnoise*wr.randn(*net.Wrec.shape).astype(np.float32); net.Wout=net.Wout+wnoise*wr.randn(*net.Wout.shape).astype(np.float32)
    rng=random.Random(seed); per=[]
    for _ in range(neps):
        net.reset('reset'); plan,rev=make_plan_e3(rng, held=held)
        rec,unst=rollout_e3(net,plan)
        per.append(score_e3(rec,rev))
    av=lambda k: round(float(np.mean([d[k] for d in per])),3)
    return {'pre':av('pre'),'post':av('post'),'delay':round(float(np.median([d['delay'] for d in per])),1),'probe':av('probe'),'overall':av('overall')}

# ---- random-genome baseline (equal DSL) ----
def rand_genome(seed, env):
    g=random.Random(seed); rule=g.choice(['none','hebbian','oja','reward_hebb','input_bind']); plastic=rule!='none'
    return {'family':g.choice(['reservoir','gru','ctrnn','vanilla_rnn']),'input_dim':14,'hidden_dim':g.choice([32,48,64,96]),'output_dim':21,'slow_hidden':0,
      'weights':{'recurrent':{'gen':g.choice(['dense','sparse']),'seed':g.randint(0,99999),'scale':round(g.uniform(0.5,1.1),2),'spectral_radius':round(g.uniform(0.7,1.2),2),'sparsity':round(g.uniform(0.1,0.4),2)},
                 'input':{'gen':'dense','seed':g.randint(0,99999),'scale':round(g.uniform(0.4,1.0),2)},
                 'readout':{'gen':'dense','seed':g.randint(0,99999),'scale':round(g.uniform(0.05,0.4),2)}},
      'dynamics':{'activation':'tanh','gain':round(g.uniform(0.8,1.3),2),'leak':round(g.uniform(0.05,0.4),2),'tau':round(g.uniform(1,4),1),'noise':0.0},
      'plasticity':{'enabled':plastic,'targets':[g.choice(['readout','recurrent'])],'rule':rule if plastic else 'hebbian','input_bind':rule=='input_bind','lr':round(g.uniform(0.01,0.2),3),'decay':round(g.uniform(0,0.02),3),'eligibility':round(g.uniform(0,0.8),2),'reward_mod':rule=='reward_hebb'},
      'init_state':{'gen':'zeros'}}

def _fixed(g):
    import copy; g=copy.deepcopy(g); g['plasticity']['enabled']=False; return g

def run(env):
    genomes=json.load(open(f'{env.lower()}_genomes.json'))
    evalf=eval_e2 if env=='E2' else eval_e3
    print(f"=== PART 4 QUANTITATIVE — {env} ===", flush=True)
    if env=='E2':
        print("  chance probe=1/16=0.06 ; ANSWER=transform(K) is NEVER observed", flush=True)
        print(f"  {'genome':22s} {'rule':11s} | {'train':>6s} {'held':>6s} {'surv':>5s} {'wnoise':>6s} {'hpert':>6s}", flush=True)
    else:
        print("  chance choice=0.50 ; must adapt to mid-episode reward reversal", flush=True)
        print(f"  {'genome':22s} {'rule':11s} | {'pre':>5s} {'post':>5s} {'delay':>5s} {'probe':>5s} {'overall':>7s} {'held_ov':>7s}", flush=True)
    for item in genomes:
        g=item['genome']; tag=item['tag']; rule=g.get('plasticity',{}).get('rule') if g.get('plasticity',{}).get('enabled') else 'FIXED'
        if env=='E2':
            tr=evalf(g); hd=evalf(g,held=True); wn=evalf(g,wnoise=0.05); hp=evalf(g,hpert=0.5)
            if tr is None: print(f"  {tag:22s} COMPILE-FAIL", flush=True); continue
            print(f"  {tag:22s} {str(rule):11s} | {tr['probe']:6.2f} {hd['probe']:6.2f} {tr['survival']:5.2f} {wn['probe']:6.2f} {hp['probe']:6.2f}", flush=True)
        else:
            tr=evalf(g); hd=evalf(g,held=True)
            if tr is None: print(f"  {tag:22s} COMPILE-FAIL", flush=True); continue
            print(f"  {tag:22s} {str(rule):11s} | {tr['pre']:5.2f} {tr['post']:5.2f} {tr['delay']:5.1f} {tr['probe']:5.2f} {tr['overall']:7.2f} {hd['overall']:7.2f}", flush=True)
    # baselines: random-genome distribution (equal DSL budget) + fixed variants
    print("  -- baselines --", flush=True)
    rands=[rand_genome(s, env) for s in range(30)]
    scored=[]
    for g in rands:
        r=evalf(g)
        if r is not None: scored.append(r['probe'] if env=='E2' else r['overall'])
    if scored:
        scored=sorted(scored)
        key='probe' if env=='E2' else 'overall'
        print(f"  random-genome ({len(scored)}/30 valid) {key}: median={np.median(scored):.2f} best={max(scored):.2f} p90={np.percentile(scored,90):.2f}", flush=True)
    # fixed variants of the plastic Claude genomes (isolate plasticity contribution)
    for item in genomes:
        g=item['genome']
        if g.get('plasticity',{}).get('enabled'):
            r=evalf(_fixed(g));
            if r is not None:
                k=r['probe'] if env=='E2' else r['overall']
                print(f"  fixed({item['tag']}) {('probe' if env=='E2' else 'overall')}={k:.2f}", flush=True)
    print(f"=== {env}_DONE ===", flush=True)

def _mut(g,seed):
    import copy; r=random.Random(seed); g=copy.deepcopy(g); p=g['plasticity']; rec=g['weights']['recurrent']
    ch=r.choice(['lr','sr','leak','rule','hidden','target','elig','readscale'])
    if ch=='lr': p['lr']=round(max(0.005,p.get('lr',0.05)*r.uniform(0.5,2)),3)
    elif ch=='sr': rec['spectral_radius']=round(min(1.4,max(0.6,rec.get('spectral_radius',0.9)+r.uniform(-0.2,0.2))),2)
    elif ch=='leak': g['dynamics']['leak']=round(min(0.6,max(0.05,g['dynamics']['leak']+r.uniform(-0.1,0.1))),2)
    elif ch=='rule': rr=r.choice(['hebbian','oja','reward_hebb','input_bind']); p['rule']=rr; p['input_bind']=rr=='input_bind'; p['reward_mod']=rr=='reward_hebb'; p['enabled']=True
    elif ch=='hidden': g['hidden_dim']=r.choice([32,48,64,96])
    elif ch=='target': p['targets']=[r.choice(['readout','recurrent'])]
    elif ch=='elig': p['eligibility']=round(r.uniform(0,0.9),2)
    elif ch=='readscale': g['weights']['readout']['scale']=round(max(0.01,g['weights']['readout']['scale']*r.uniform(0.4,2)),3)
    return g

def search(env):
    evalf=eval_e2 if env=='E2' else eval_e3
    def fit(g):
        r=evalf(g, neps=60); return (r['probe'] if env=='E2' else r['overall']) if r else 0.0
    thr=0.5 if env=='E2' else 0.70
    print(f"=== PART 3 CROSS-ENV SEARCH — {env} (fitness={'train probe' if env=='E2' else 'overall'}; chance={'0.06' if env=='E2' else '0.50'}) ===",flush=True)
    best=0.0; evals=0; first=None
    for s in range(80):
        f=fit(rand_genome(s,env)); evals+=1
        if f>best: best=f
        if first is None and f>=thr: first=evals
    print(f"  RANDOM search: {evals} evals | best={best:.2f} | evals-to-threshold({thr})={first}",flush=True)
    pop=[rand_genome(1000+s,env) for s in range(12)]; scored=[(fit(g),g) for g in pop]; evals+=12
    for gen in range(6):
        scored.sort(key=lambda x:-x[0]); elite=[g for _,g in scored[:4]]
        for i in range(8):
            c=_mut(random.Random(gen*100+i).choice(elite), gen*1000+i); scored.append((fit(c),c)); evals+=1
        scored=sorted(scored,key=lambda x:-x[0])[:12]
    print(f"  EVO search: {evals} total evals | best={scored[0][0]:.2f}",flush=True)
    print(f"  (gradient-trained ceiling: E2 train=1.00/held=0.00 ; E3 held post_rev=0.72 — OUTSIDE the DSL search space)",flush=True)
    print(f"=== {env}_SEARCH_DONE ===",flush=True)

if __name__=='__main__':
    if os.environ.get('ME2_MODE')=='search': search(os.environ.get('ME2_ENV','E2'))
    else: run(os.environ.get('ME2_ENV','E2'))
