import os, json, math, random
import numpy as np

# ============================================================================
# SYNTHESIZED_MICRO_ENTITY_V1  (SEPARATE hypothesis from register-workspace probe)
# Question: can an LLM COMPILE a semantic environment description into an executable
# neural dynamical system, via a compact "neural genome" DSL?  NOT "train a model on
# generated examples" — the LLM emits a genome; a deterministic compiler instantiates a
# small recurrent net; we evaluate zero-shot + local-plasticity (no backprop for synthesized).
# Compare: A=hand FSM, B=gradient-RNN, C=LLM fixed genome, D=LLM plastic genome, E=random genome.
# Claim ceiling: "LLM synthesized a compact recurrent system whose dynamics+local plasticity
# support viability" — NOT entity/selfhood/workspace/criticality.
# ============================================================================

MODE = os.environ.get('ME_MODE', 'smoke')
SEED = int(os.environ.get('ME_SEED', '0'))

# ---- symbol alphabet (held-out split) + fixed deterministic embeddings ----
D_SYM = 8
ALPHABET = ['alpha','bravo','charlie','delta','echo','foxtrot','golf','hotel',
            'india','juliet','kilo','lima','mike','november','oscar','papa']  # 16
HELDOUT = set(ALPHABET[12:])           # 4 held-out symbols
TRAINSYM = ALPHABET[:12]
DISTRACT = ['zulu','yankee','xray','whiskey','victor','uniform']  # filler-only pool
def _emb_table(names, seed):
    g = np.random.RandomState(seed)
    return {n: g.randn(D_SYM).astype(np.float32) for n in names}
SYM_EMB = _emb_table(ALPHABET, 12345)
DIS_EMB = _emb_table(DISTRACT, 54321)

EVENTS = ['COMMIT','FILLER','FALSE','VALID_REL','INVALID_REL','PROBE']
EV_IDX = {e:i for i,e in enumerate(EVENTS)}
ACTIONS = ['HOLD','UPDATE','REJECT','QUERY','RESPOND']
ACT_IDX = {a:i for i,a in enumerate(ACTIONS)}
D_IN = len(EVENTS) + D_SYM               # obs = event onehot + symbol embedding
D_OUT = len(ACTIONS) + len(ALPHABET)     # action logits + symbol readout logits

def obs_vec(event, sym_name, distractor=False):
    v = np.zeros(D_IN, np.float32)
    v[EV_IDX[event]] = 1.0
    if sym_name is not None:
        v[len(EVENTS):] = (DIS_EMB[sym_name] if distractor else SYM_EMB[sym_name])
    return v

# ---- world: observation-based viability episode ----
def make_episode(rng, held=False, phrase_shift=False, nfill=None):
    syms = list(HELDOUT) if held else TRAINSYM
    seq = rng.sample(syms, 3); C0, X1, X2 = seq[0], seq[1], seq[2]
    def W(cur):
        w = rng.choice(syms)
        while w == cur: w = rng.choice(syms)
        return w
    steps = []  # (event, sym_name, distractor, true_commitment_AFTER, expected_action, is_probe)
    cur = C0
    def add(ev, sym, dis, act):
        nonlocal cur
        steps.append({'ev':ev,'sym':sym,'dis':dis,'true':cur,'act':act})
    add('COMMIT', C0, False, 'UPDATE')
    def fillers(k):
        for _ in range(k): add('FILLER', rng.choice(DISTRACT), True, 'HOLD')
    fillers(nfill or rng.randint(1,3)); add('FALSE', W(cur), False, 'REJECT'); add('PROBE', None, False, 'RESPOND')
    fillers(nfill or rng.randint(1,3)); cur = X1; add('VALID_REL', X1, False, 'UPDATE'); add('PROBE', None, False, 'RESPOND')
    fillers(nfill or rng.randint(1,3)); add('INVALID_REL', W(cur), False, 'REJECT'); add('PROBE', None, False, 'RESPOND')
    fillers(nfill or rng.randint(1,3)); add('FALSE', W(cur), False, 'REJECT'); add('PROBE', None, False, 'RESPOND')
    fillers(nfill or rng.randint(1,3)); cur = X2; add('VALID_REL', X2, False, 'UPDATE'); add('PROBE', None, False, 'RESPOND')
    return steps

def score_rollout(steps, actions, symbol_preds, vmax=3):
    # actions: list of action-name per step; symbol_preds: list of predicted symbol-name (or None) per step
    m = {'valid_update':[0,0],'false_resist':[0,0],'invalid_reject':[0,0],'probe':[0,0],'post_release':[0,0]}
    V = vmax; alive = True; last_rel = None
    for i,st in enumerate(steps):
        ev = st['ev']; a = actions[i]; sp = symbol_preds[i]; true = st['true']
        if ev == 'VALID_REL':
            ok = int(a == 'UPDATE'); m['valid_update'][0]+=ok; m['valid_update'][1]+=1; last_rel=true
        elif ev == 'FALSE':
            ok = int(a in ('REJECT','HOLD')); m['false_resist'][0]+=ok; m['false_resist'][1]+=1
        elif ev == 'INVALID_REL':
            ok = int(a in ('REJECT','HOLD')); m['invalid_reject'][0]+=ok; m['invalid_reject'][1]+=1
        elif ev == 'PROBE':
            ok = int(sp == true); m['probe'][0]+=ok; m['probe'][1]+=1
            if last_rel is not None: m['post_release'][0]+=int(sp==true); m['post_release'][1]+=1
            if alive and not ok:
                V -= 1
                if V <= 0: alive = False
    r = {k:(v[0]/v[1] if v[1] else 0.0) for k,v in m.items()}
    r['survival'] = int(alive)
    return r

# ---- neural genome compiler ----
FAMILIES = {'vanilla_rnn','gru','ctrnn','reservoir'}
LIMITS = {'hidden_max':128,'spectral_max':1.6}

def validate(g):
    try:
        if g['family'] not in FAMILIES: return False,f"family {g.get('family')}"
        H = int(g['hidden_dim'])
        if not (2 <= H <= LIMITS['hidden_max']): return False,f"hidden {H}"
        if int(g['input_dim']) != D_IN: return False,f"input_dim must be {D_IN}"
        if int(g['output_dim']) != D_OUT: return False,f"output_dim must be {D_OUT}"
        dyn = g['dynamics']
        if not (0.0 <= dyn.get('leak',0.2) <= 1.0): return False,"leak range"
        if dyn.get('gain',1.0) > 3.0: return False,"gain too high"
        pl = g.get('plasticity',{})
        if pl.get('enabled') and pl.get('rule') not in ('hebbian','reward_hebb','oja','input_bind'): return False,"plastic rule"
        return True,"ok"
    except Exception as e:
        return False,f"malformed: {e}"

def _gen_matrix(spec, rows, cols, default_scale=0.5):
    seed = int(spec.get('seed',0)); g = np.random.RandomState(seed); scale = float(spec.get('scale',default_scale))
    typ = spec.get('gen','dense')
    if typ == 'lowrank':
        r = int(spec.get('rank',4)); U = g.randn(rows,r); Vv = g.randn(r,cols); M = (U@Vv)/math.sqrt(r)
    elif typ == 'sparse':
        M = g.randn(rows,cols); mask = (g.rand(rows,cols) < float(spec.get('sparsity',0.2))).astype(np.float32); M = M*mask
    else:
        M = g.randn(rows,cols)
    M = M.astype(np.float32) * scale
    return M

def _apply_recurrent_structure(M, spec):
    # spectral radius normalization + diagonal stability + E/I sign structure
    sr = spec.get('spectral_radius')
    if sr is not None:
        ev = np.max(np.abs(np.linalg.eigvals(M))) + 1e-8
        M = M * (float(sr)/ev)
    diag = spec.get('diag')
    if diag is not None:
        np.fill_diagonal(M, M.diagonal() + float(diag))
    ei = spec.get('ei')
    if ei is not None:
        exc = int(M.shape[1]*float(ei.get('exc_frac',0.8)))
        signs = np.ones(M.shape[1], np.float32); signs[exc:] = -1.0
        M = np.abs(M)*signs[None,:]
    return M.astype(np.float32)

class MicroNet:
    def __init__(self, g):
        self.g = g; H = int(g['hidden_dim']); self.H = H
        w = g['weights']
        self.Wrec = _apply_recurrent_structure(_gen_matrix(w['recurrent'],H,H,0.9), w['recurrent'])
        self.Win  = _gen_matrix(w['input'], H, D_IN, 0.5)
        self.Wout = _gen_matrix(w['readout'], D_OUT, H, 0.3)
        self.bout = np.zeros(D_OUT, np.float32)
        d = g['dynamics']
        self.act = d.get('activation','tanh'); self.gain=float(d.get('gain',1.0))
        self.leak=float(d.get('leak',0.2)); self.tau=float(d.get('tau',1.0)); self.noise=float(d.get('noise',0.0))
        self.family = g['family']
        pl = g.get('plasticity',{})
        self.plastic = bool(pl.get('enabled',False)); self.pl = pl
        self.elig = np.zeros_like(self.Wout)
        self._init_state(g.get('init_state',{}))
    def _init_state(self, spec):
        if spec.get('gen') == 'seed':
            self.h = np.random.RandomState(int(spec.get('seed',0))).randn(self.H).astype(np.float32)*0.1
        else:
            self.h = np.zeros(self.H, np.float32)
    def reset(self, mode='init'):
        if mode=='reset': self.h = np.zeros(self.H, np.float32)
        else: self._init_state(self.g.get('init_state',{}))
    def _nl(self, x):
        return np.tanh(x) if self.act=='tanh' else np.maximum(0,x)
    def step(self, obs, rng=None):
        pre = self.gain*(self.Wrec@self.h) + self.Win@obs
        u = self._nl(pre)
        if self.family=='ctrnn':
            self.h = self.h + (1.0/max(self.tau,1e-3))*(-self.h + u)
        else:
            self.h = (1-self.leak)*self.h + self.leak*u
        if self.noise>0 and rng is not None: self.h = self.h + self.noise*rng.randn(self.H).astype(np.float32)
        self.h = np.clip(self.h, -10, 10)
        out = self.Wout@self.h + self.bout
        return out
    def plastic_update(self, event=None, sym_idx=None, reward=0.0):
        # local, no-backprop plasticity. input_bind: at ADOPTION events (COMMIT/VALID_REL) the symbol is IN
        # the observation (visible to all systems) -> self-supervised delta rule binding maintained state ->
        # observed symbol in the symbol-readout. NOT a probe-time answer label. Optional reward modulation.
        if not self.plastic: return
        pl=self.pl; lr=float(pl.get('lr',0.02)); dec=float(pl.get('decay',0.0))
        if pl.get('input_bind') and event in ('COMMIT','VALID_REL') and sym_idx is not None:
            row=len(ACTIONS)+sym_idx
            pred=self.Wout@self.h
            target=pred.copy(); target[len(ACTIONS):]=-0.2; target[row]=1.5   # push this symbol up, others down
            err=target-pred
            self.Wout += lr*np.outer(err, self.h)
        if dec>0: self.Wout -= dec*self.Wout

def rollout(net, steps, rng=None, perturb_at=None, perturb_scale=0.0):
    actions=[]; symbol_preds=[]; hs=[]; unstable=False
    for i,st in enumerate(steps):
        o = obs_vec(st['ev'], st['sym'], st['dis'])
        if perturb_at is not None and i==perturb_at and rng is not None:
            net.h = net.h + perturb_scale*rng.randn(net.H).astype(np.float32)
        out = net.step(o, rng)
        if not np.all(np.isfinite(out)): unstable=True; break
        a = ACTIONS[int(np.argmax(out[:len(ACTIONS)]))]
        sp = ALPHABET[int(np.argmax(out[len(ACTIONS):]))]
        actions.append(a); symbol_preds.append(sp); hs.append(net.h.copy())
        if net.plastic:
            sidx = ALPHABET.index(st['sym']) if (st['sym'] in SYM_EMB and not st['dis']) else None
            net.plastic_update(event=st['ev'], sym_idx=sidx)
    while len(actions)<len(steps):                 # unstable rollout broke early -> pad as failed steps
        actions.append('HOLD'); symbol_preds.append(None); hs.append(np.zeros(net.H,np.float32))
    return actions, symbol_preds, hs, unstable

def compile_and_check(g):
    ok,reason = validate(g)
    if not ok: return None, f"INVALID: {reason}"
    try:
        net = MicroNet(g)
        # stability probe: random drive for 30 steps, reject if explodes/NaN
        rng = np.random.RandomState(SEED)
        for _ in range(30):
            out = net.step(rng.randn(D_IN).astype(np.float32))
            if not np.all(np.isfinite(out)) or np.max(np.abs(net.h))>50: return None,"UNSTABLE"
        net.reset()
        return net, "ok"
    except Exception as e:
        return None, f"BUILD_ERR: {e}"

# ---- baseline A: hand-designed FSM (optimal reference) ----
class FSM:
    def __init__(self): self.cur=None
    def run(self, steps):
        actions=[]; preds=[]; self.cur=None
        for st in steps:
            ev=st['ev']
            if ev=='COMMIT': self.cur=st['sym']; actions.append('UPDATE'); preds.append(self.cur)
            elif ev=='VALID_REL': self.cur=st['sym']; actions.append('UPDATE'); preds.append(self.cur)
            elif ev in ('FALSE','INVALID_REL'): actions.append('REJECT'); preds.append(self.cur)
            elif ev=='FILLER': actions.append('HOLD'); preds.append(self.cur)
            elif ev=='PROBE': actions.append('RESPOND'); preds.append(self.cur)
        return actions, preds

def eval_system(runner, neps=60, held=False, seed=SEED, **kw):
    rng=random.Random(seed); agg={}; hist={}
    for _ in range(neps):
        steps=make_episode(rng, held=held)
        actions,preds=runner(steps)
        r=score_rollout(steps,actions,preds)
        for k,v in r.items(): agg[k]=agg.get(k,0)+v
        for p in preds: hist[p]=hist.get(p,0)+1
    n=neps; return {k:v/n for k,v in agg.items()}, hist

# ---- random genome sampler (baseline E) ----
def random_genome(seed):
    g=random.Random(seed)
    fam=g.choice(list(FAMILIES)); H=g.choice([16,32,48,64])
    return {'family':fam,'input_dim':D_IN,'hidden_dim':H,'output_dim':D_OUT,'slow_hidden':0,
        'weights':{'recurrent':{'gen':g.choice(['dense','lowrank','sparse']),'seed':g.randint(0,9999),'scale':round(g.uniform(0.3,1.0),2),'spectral_radius':round(g.uniform(0.5,1.3),2)},
                   'input':{'gen':'dense','seed':g.randint(0,9999),'scale':round(g.uniform(0.3,0.8),2)},
                   'readout':{'gen':'dense','seed':g.randint(0,9999),'scale':round(g.uniform(0.2,0.6),2)}},
        'dynamics':{'activation':'tanh','gain':round(g.uniform(0.7,1.3),2),'leak':round(g.uniform(0.1,0.5),2),'tau':1.0,'noise':0.0},
        'plasticity':{'enabled':g.random()<0.5,'targets':['readout'],'rule':'reward_hebb','lr':0.01,'reward_mod':True},
        'init_state':{'gen':'zeros'}}

HAND_GENOME = {  # sanity: reservoir (memory) + input-binding plastic readout — does the DSL express a viable solution?
  'family':'reservoir','input_dim':D_IN,'hidden_dim':64,'output_dim':D_OUT,'slow_hidden':0,
  'weights':{'recurrent':{'gen':'sparse','seed':7,'scale':1.0,'sparsity':0.2,'spectral_radius':0.95},
             'input':{'gen':'dense','seed':8,'scale':0.6},
             'readout':{'gen':'dense','seed':9,'scale':0.05}},
  'dynamics':{'activation':'tanh','gain':1.0,'leak':0.2,'tau':1.0,'noise':0.0},
  'plasticity':{'enabled':True,'targets':['readout'],'rule':'input_bind','input_bind':True,'lr':0.05,'decay':0.0,'reward_mod':False},
  'init_state':{'gen':'zeros'}}

def eval_genome(genome, neps=150, online=True, held=False, seed=SEED, weight_noise=0.0, hidden_perturb=0.0):
    net,msg = compile_and_check(genome)
    if net is None: return None, msg
    if not online: net.plastic=False
    if weight_noise>0:
        wr=np.random.RandomState(seed+7)
        net.Wrec = net.Wrec + weight_noise*wr.randn(*net.Wrec.shape).astype(np.float32)
        net.Wout = net.Wout + weight_noise*wr.randn(*net.Wout.shape).astype(np.float32)
    rng=random.Random(seed); per=[]; hist={}; prng=np.random.RandomState(seed+3)
    for ei in range(neps):
        net.reset('reset')                    # hidden resets each episode; readout PERSISTS if online (accumulates decoder)
        steps=make_episode(rng, held=held)
        pa = (len(steps)//2) if hidden_perturb>0 else None
        a,p,hs,unst=rollout(net,steps, rng=prng, perturb_at=pa, perturb_scale=hidden_perturb)
        r=score_rollout(steps,a,p); per.append(r)
        for x in p: hist[x]=hist.get(x,0)+1
    def avg(lst):
        n=len(lst); return {k:round(sum(d[k] for d in lst)/n,3) for k in lst[0]}
    third=max(1,neps//3)
    return {'early':avg(per[:third]),'late':avg(per[-third:]),'hist':dict(sorted(hist.items(),key=lambda x:-x[1])[:6])}, "ok"

def state_metrics(genome, neps=60, seed=SEED):
    net,msg=compile_and_check(genome)
    if net is None: return {}
    net.plastic=False                          # intrinsic reservoir memory (no readout adaptation)
    rng=random.Random(seed); Hp=[];Yc=[];Hf=[];Yf=[]; norms=[]
    for _ in range(neps):
        net.reset('reset'); steps=make_episode(rng); a,p,hs,unst=rollout(net,steps)
        for i,st in enumerate(steps):
            norms.append(float(np.linalg.norm(hs[i])))
            if st['ev']=='PROBE': Hp.append(hs[i]); Yc.append(ALPHABET.index(st['true']))
            elif st['ev']=='FILLER' and st['sym'] in DISTRACT: Hf.append(hs[i]); Yf.append(DISTRACT.index(st['sym']))
    def pa(H,Y,k):
        if len(Y)<20: return 0.0
        X=np.stack(H).astype(np.float32);y=np.array(Y);n=len(y);idx=np.arange(n);np.random.RandomState(0).shuffle(idx)
        tr=idx[:int(n*.7)];te=idx[int(n*.7):];mu=X[tr].mean(0);sd=X[tr].std(0)+1e-6
        Xtr=np.c_[(X[tr]-mu)/sd,np.ones((len(tr),1),np.float32)];Xte=np.c_[(X[te]-mu)/sd,np.ones((len(te),1),np.float32)]
        Yt=np.zeros((len(tr),k),np.float32);Yt[np.arange(len(tr)),y[tr]]=1
        W=np.linalg.solve(Xtr.T@Xtr+np.eye(Xtr.shape[1],dtype=np.float32),Xtr.T@Yt)
        return round(float(((Xte@W).argmax(1)==y[te]).mean()),3)
    return {'commit_decode':pa(Hp,Yc,len(ALPHABET)),'filler_decode':pa(Hf,Yf,len(DISTRACT)),'state_norm':round(float(np.mean(norms)),2)}

def full_eval(genome, tag, neps=150):
    net,msg=compile_and_check(genome)
    if net is None: print(f"  {tag:18s} COMPILE-FAIL: {msg}", flush=True); return None
    zs,_=eval_genome(genome,online=False,neps=neps)
    on,_=eval_genome(genome,online=True,neps=neps)
    onh,_=eval_genome(genome,online=True,neps=neps,held=True)
    wn,_=eval_genome(genome,online=True,neps=neps,weight_noise=0.05)
    hp,_=eval_genome(genome,online=True,neps=neps,hidden_perturb=1.0)
    sm=state_metrics(genome)
    def g(d,k): return d['late'].get(k,0) if d else 0
    zsp = zs['early']['probe'] if zs else 0
    print(f"  {tag:18s}| zshot_probe={zsp:.2f} | online probe={g(on,'probe'):.2f}/surv={g(on,'survival'):.2f} | HELD probe={g(onh,'probe'):.2f}/surv={g(onh,'survival'):.2f} | wnoise surv={g(wn,'survival'):.2f} | hpert surv={g(hp,'survival'):.2f} | commit_dec={sm.get('commit_decode',0)} filler_dec={sm.get('filler_decode',0)}", flush=True)
    return {'tag':tag,'zs':zsp,'online':on['late'] if on else {},'held':onh['late'] if onh else {},'wnoise':wn['late'] if wn else {},'hpert':hp['late'] if hp else {},'state':sm}

def ablate():
    # PART 2: component ablations of the best Claude-plastic genome. Which piece(s) cause the result?
    import json, copy, random as R
    base=json.load(open('claude_robust.json'))[0]['genome']
    V={}
    V['D-full']=base
    g=copy.deepcopy(base); g['plasticity']['enabled']=False; V['D-no-plasticity']=g
    g=copy.deepcopy(base); rr=R.Random(1); g['plasticity']['lr']=round(rr.uniform(0.001,0.2),3); g['plasticity']['decay']=round(rr.uniform(0,0.1),3); V['D-random-plasticity']=g
    g=copy.deepcopy(base); g['weights']['recurrent']['spectral_radius']=0.1; g['dynamics']['leak']=0.9; V['D-no-reservoir']=g
    g=copy.deepcopy(base); g['plasticity']['rule']='hebbian'; g['plasticity']['input_bind']=False; V['D-no-binding']=g
    g=copy.deepcopy(base); g['weights']['recurrent']['spectral_radius']=0.05; g['dynamics']['leak']=0.95; V['D-readout-only']=g
    g=copy.deepcopy(base); g['weights']['recurrent']['seed']=8888; g['weights']['input']['seed']=8889; g['weights']['readout']['seed']=8890; V['D-random-structure']=g
    V['D-example-template']={"family":"reservoir","input_dim":D_IN,"hidden_dim":64,"output_dim":D_OUT,"slow_hidden":0,"weights":{"recurrent":{"gen":"sparse","seed":7,"scale":1.0,"sparsity":0.15,"spectral_radius":0.9},"input":{"gen":"dense","seed":8,"scale":0.6},"readout":{"gen":"dense","seed":9,"scale":0.05}},"dynamics":{"activation":"tanh","gain":1.0,"leak":0.2,"tau":1.0,"noise":0.0},"plasticity":{"enabled":True,"targets":["readout"],"rule":"input_bind","input_bind":True,"lr":0.05,"decay":0.003,"reward_mod":False},"init_state":{"gen":"zeros"}}
    V['D-minimal-binding']={"family":"reservoir","input_dim":D_IN,"hidden_dim":8,"output_dim":D_OUT,"slow_hidden":0,"weights":{"recurrent":{"gen":"dense","seed":3,"scale":1.0,"spectral_radius":0.9},"input":{"gen":"dense","seed":4,"scale":0.6},"readout":{"gen":"dense","seed":5,"scale":0.02}},"dynamics":{"activation":"tanh","gain":1.0,"leak":0.2,"tau":1.0,"noise":0.0},"plasticity":{"enabled":True,"targets":["readout"],"rule":"input_bind","input_bind":True,"lr":0.08,"decay":0.003,"reward_mod":False},"init_state":{"gen":"zeros"}}
    print("=== PART 2 COMPONENT ABLATIONS (base = Claude D-plastic) ===", flush=True)
    print("  KEY: D-no-reservoir/D-no-binding should FAIL if both components matter; D-minimal-binding==D-full => compact binding algorithm", flush=True)
    for tag,g in V.items(): full_eval(g, tag)
    print("=== MICRO_ENTITY_ABLATE_DONE ===", flush=True)

def evalfile():
    import json
    path=os.environ.get('ME_GENOMES','genomes.json'); genomes=json.load(open(path))
    print(f"=== MICRO_ENTITY_V1 FULL EVAL | {len(genomes)} synthesized genomes ({path}) ===", flush=True)
    fsm=FSM(); r,_=eval_system(fsm.run,60); rh,_=eval_system(fsm.run,60,held=True)
    print(f"  {'A FSM (ceiling)':18s}| train probe={r['probe']:.2f}/surv={r['survival']:.2f} | HELD probe={rh['probe']:.2f}/surv={rh['survival']:.2f}  [grad-RNN B: train 0.80/0.87, HELD 0.00/0.00]", flush=True)
    for entry in genomes: full_eval(entry['genome'], entry['tag'])
    full_eval(random_genome(2000), 'E random')
    print("=== DECISION TREE: C=plastic online>>fixed&random but below grad-RNN | D=plastic matches/beats grad-RNN on TRANSFER (grad-RNN HELD=0.0) / perturbation ===", flush=True)
    print("=== MICRO_ENTITY_EVALFILE_DONE ===", flush=True)

def handtest():
    print("=== MICRO_ENTITY handtest: can the DSL express a viable solution? (reservoir + input-bind plastic readout) ===", flush=True)
    ok,reason=validate(HAND_GENOME); print("validate:",ok,reason, flush=True)
    zs,_=eval_genome(HAND_GENOME, online=False); print("ZERO-SHOT (readout fixed):", {'early':zs['early']} if zs else _, flush=True)
    on,_=eval_genome(HAND_GENOME, online=True)
    if on:
        print("ONLINE-PLASTIC early:", on['early'], flush=True)
        print("ONLINE-PLASTIC late :", on['late'], flush=True)
        print("  pred-hist(top):", on['hist'], flush=True)
    onh,_=eval_genome(HAND_GENOME, online=True, held=True)
    if onh: print("ONLINE-PLASTIC late (HELD-OUT symbols):", onh['late'], flush=True)
    print("=== INTERPRET: if ONLINE late probe/survival >> early >> zero-shot => reservoir+local-plastic DSL is viable => LLM synthesis is meaningful ===", flush=True)
    print("=== MICRO_ENTITY_HANDTEST_DONE ===", flush=True)

def train_rnn(neps=600, hidden=48, seed=0):
    # Baseline B: gradient-trained small GRU (CPU), supervised on correct actions + current commitment (labels
    # allowed for the TRAINED reference; synthesized genomes get NO backprop). The "achievable" upper bound.
    import torch, torch.nn as nn, torch.nn.functional as F
    torch.set_num_threads(2); torch.manual_seed(seed)
    rnn=nn.GRUCell(D_IN,hidden); ha=nn.Linear(hidden,len(ACTIONS)); hs=nn.Linear(hidden,len(ALPHABET))
    opt=torch.optim.Adam(list(rnn.parameters())+list(ha.parameters())+list(hs.parameters()),lr=2e-3)
    rng=random.Random(seed)
    for ep in range(neps):
        steps=make_episode(rng); h=torch.zeros(1,hidden); loss=0.0
        for st in steps:
            o=torch.tensor(obs_vec(st['ev'],st['sym'],st['dis'])).unsqueeze(0); h=rnn(o,h)
            loss=loss+F.cross_entropy(ha(h),torch.tensor([ACT_IDX[st['act']]]))
            s_t=ALPHABET.index(st['true']) if st['true'] else 0
            loss=loss+F.cross_entropy(hs(h),torch.tensor([s_t]))
        opt.zero_grad(); loss.backward(); opt.step()
    def runner(steps):
        with torch.no_grad():
            h=torch.zeros(1,hidden); acts=[]; preds=[]
            for st in steps:
                o=torch.tensor(obs_vec(st['ev'],st['sym'],st['dis'])).unsqueeze(0); h=rnn(o,h)
                acts.append(ACTIONS[int(ha(h).argmax())]); preds.append(ALPHABET[int(hs(h).argmax())])
            return acts,preds
    return runner

def baselines():
    print("=== MICRO_ENTITY baselines: A=FSM  B=gradient-RNN  E=random-genome ===", flush=True)
    fsm=FSM(); r,_=eval_system(fsm.run,neps=60); print("A FSM        :", {k:round(v,3) for k,v in r.items()}, flush=True)
    rh,_=eval_system(fsm.run,neps=60,held=True); print("A FSM  (held):", {k:round(v,3) for k,v in rh.items()}, flush=True)
    print("training gradient-RNN (CPU) ...", flush=True)
    run=train_rnn(neps=600); r,_=eval_system(run,neps=60); print("B grad-RNN   :", {k:round(v,3) for k,v in r.items()}, flush=True)
    rh,_=eval_system(run,neps=60,held=True); print("B grad-RNN(held):", {k:round(v,3) for k,v in rh.items()}, flush=True)
    print("=== MICRO_ENTITY_BASELINES_DONE ===", flush=True)

def smoke():
    print("=== MICRO_ENTITY smoke: validate world + compiler + baselines ===", flush=True)
    print(f"D_IN={D_IN} D_OUT={D_OUT} (5 actions + {len(ALPHABET)} symbols) | trainsym={len(TRAINSYM)} heldout={len(HELDOUT)}", flush=True)
    # A: FSM (should be ~perfect)
    fsm=FSM(); r,_=eval_system(fsm.run, neps=60)
    print("A  FSM        :", {k:round(v,3) for k,v in r.items()}, flush=True)
    # constant predictor control (always HOLD + always predict a fixed symbol)
    fixed=ALPHABET[0]
    def const(steps): return ['HOLD']*len(steps), [fixed]*len(steps)
    rc,_=eval_system(const, neps=60)
    print("   const-pred :", {k:round(v,3) for k,v in rc.items()}, flush=True)
    # E: random genomes (should be ~chance)
    goods=0; results=[]
    for s in range(8):
        g=random_genome(1000+s); net,msg=compile_and_check(g)
        if net is None: results.append((s,msg)); continue
        goods+=1
        def run(steps, net=net):
            net.reset(); a,p,_,unst=rollout(net,steps); return a,p
        r,_=eval_system(run, neps=40)
        results.append((s, {k:round(v,2) for k,v in r.items()}))
    print(f"E  random-genome ({goods}/8 compiled):", flush=True)
    for s,res in results: print(f"     seed{s}: {res}", flush=True)
    print("=== VALID IF: FSM~1.0 on all, const-pred fails probe/update, random-genome ~chance ===", flush=True)
    print("=== MICRO_ENTITY_SMOKE_DONE ===", flush=True)

if MODE=='smoke': smoke()
elif MODE=='handtest': handtest()
elif MODE=='baselines': baselines()
elif MODE=='evalfile': evalfile()
elif MODE=='ablate': ablate()
