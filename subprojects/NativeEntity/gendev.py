import os, random
os.environ['ME_MODE']='none'
import numpy as np
from micro_entity import (ALPHABET, DISTRACT, SYM_EMB, EVENTS, EV_IDX, ACTIONS, D_IN,
                          obs_vec, make_episode, score_rollout, _gen_matrix, _apply_recurrent_structure)

# ============================================================================
# PRIMITIVE_HIDDEN_LINEAGE_SYNTHESIS_V1 — PART 1: generic developmental DSL.
# NO named plasticity primitives. Genome specifies matrices + modulatory units +
# per-matrix decayed traces + GENERIC local update equations over low-level terms:
#   trace  = decay*trace + PROD(factors)
#   deltaW = SUM_k coef_k * PROD(factors_k)            (then clip / optional row-normalize)
# local vars: pre[C], post[R], w[R,C], mod_k, cons, act, obsemb (current observation
# embedding placed on the embedding-output rows), pred (readout's own output), traces.
# OUTPUT = [action logits][D_SYM embedding]; symbol = nearest dictionary embedding
# (correlation decode) => held-out generalization is intrinsic, NO fixed per-symbol decoder.
# The ONLY thing given is the raw observed embedding; bind/recall/gate must be DISCOVERED.
# ============================================================================
np.seterr(over='ignore', invalid='ignore')
SEED=int(os.environ.get('ME_SEED','0'))
D_SYM=SYM_EMB[ALPHABET[0]].shape[0]
D_ACT=len(ACTIONS)
D_OUT=D_ACT+D_SYM
DICT=np.stack([SYM_EMB[s] for s in ALPHABET]).astype(np.float32)     # [16, D_SYM]
def decode(emb): return ALPHABET[int(np.argmax(DICT@emb))]

def _matrix(spec, R, C, default=0.5):
    gen=spec.get('gen','dense')
    if gen=='select':
        M=np.zeros((R,C),np.float32)
        for r,dims in enumerate(spec.get('dims',[])):
            for d in dims:
                if 0<=d<C: M[r,d]=float(spec.get('scale',1.0))
        return M
    if gen=='explicit': return np.array(spec['matrix'],np.float32)
    return _gen_matrix(spec,R,C,default)

class GenNet:
    def __init__(self, g):
        self.g=g; H=int(g['hidden_dim']); self.H=H; w=g['weights']
        self.Wrec=_apply_recurrent_structure(_matrix(w['recurrent'],H,H,0.9), w['recurrent'])
        self.Win =_matrix(w['input'], H, D_IN, 0.5)
        self.Wout=_matrix(w['output'], D_OUT, H, 0.3)
        self.nmod=int(g.get('n_mod',0))
        self.Wmod=_matrix(w['mod_input'], self.nmod, D_IN, 1.0) if self.nmod else np.zeros((0,D_IN),np.float32)
        d=g['dynamics']; self.act=d.get('activation','tanh'); self.gain=float(d.get('gain',1.0))
        self.leak=float(d.get('leak',0.2)); self.noise=float(d.get('noise',0.0))
        self.modact=g.get('mod_activation','relu')
        self.plastic=g.get('plastic',{})
        self.mats={'recurrent':'Wrec','input':'Win','output':'Wout'}
        self.traces={}
        for mname,spec in self.plastic.items():
            W=getattr(self,self.mats[mname])
            for tr in spec.get('traces',[]): self.traces[(mname,tr['name'])]=np.zeros_like(W)
        self._init_state(g.get('init_state',{})); self.mods=np.zeros(max(self.nmod,1),np.float32)
    def _init_state(self,spec):
        self.h=(np.random.RandomState(int(spec.get('seed',0))).randn(self.H).astype(np.float32)*0.1
                if spec.get('gen')=='seed' else np.zeros(self.H,np.float32))
    def reset(self):
        self.h=np.zeros(self.H,np.float32)
        for k in self.traces: self.traces[k][:]=0
        if self.g.get('init_state',{}).get('gen')=='seed': self._init_state(self.g['init_state'])
    def _nl(self,x,kind):
        if kind=='tanh': return np.tanh(x)
        if kind=='sigmoid': return 1.0/(1.0+np.exp(-np.clip(x,-30,30)))
        return np.maximum(0,x)
    def step(self, obs, rng=None):
        self.h_prev=self.h.copy()
        pre=self.gain*(self.Wrec@self.h)+self.Win@obs
        u=self._nl(pre,self.act)
        self.h=(1-self.leak)*self.h+self.leak*u
        if self.noise>0 and rng is not None: self.h=self.h+self.noise*rng.randn(self.H).astype(np.float32)
        self.h=np.clip(self.h,-10,10)
        if self.nmod: self.mods=self._nl(self.Wmod@obs, self.modact)
        self.pred=self.Wout@self.h                              # readout's own output (local post-side signal)
        self.obsemb=np.zeros(D_OUT,np.float32); self.obsemb[D_ACT:]=obs[len(EVENTS):]   # observed embedding on output rows
        self.out=self.pred
        self.obs=obs
        return self.out
    def _resolve(self, f, pre, post, W, cons, act):
        if f=='pre':   return pre[None,:]
        if f=='post':  return post[:,None]
        if f=='w':     return W
        if f=='cons':  return float(cons)
        if f=='act':   return float(act)
        if f=='one':   return 1.0
        if f=='obsemb':return self.obsemb[:,None]              # current observation (embedding) on output rows
        if f=='pred':  return self.pred[:,None]
        if f.startswith('mod'):
            k=int(f[3:]); return float(self.mods[k]) if k<self.nmod else 0.0
        return None
    def update(self, cons=0.0, act=0.0):
        for mname,spec in self.plastic.items():
            attr=self.mats[mname]; W=getattr(self,attr)
            if mname=='output': pre,post=self.h, self.out
            elif mname=='recurrent': pre,post=self.h_prev, self.h
            else: pre,post=self.obs, self.h
            trs={}
            for tr in spec.get('traces',[]):
                key=(mname,tr['name']); acc=1.0
                for f in tr['factors']:
                    v=self._resolve(f,pre,post,W,cons,act)
                    if v is None: v=self.traces.get((mname,f),0.0)
                    acc=acc*v
                self.traces[key]=float(tr.get('decay',0.5))*self.traces[key]+np.broadcast_to(acc,W.shape).astype(np.float32)
                trs[tr['name']]=self.traces[key]
            delta=np.zeros_like(W)
            for term in spec.get('terms',[]):
                acc=float(term.get('coef',0.0))*float(term.get('sign',1))
                for f in term['factors']:
                    v=self._resolve(f,pre,post,W,cons,act)
                    if v is None: v=trs.get(f, self.traces.get((mname,f),0.0))
                    acc=acc*v
                delta=delta+np.broadcast_to(acc,W.shape)
            W+=delta
            c=float(spec.get('clip',6.0)); np.clip(W,-c,c,out=W)
            if spec.get('normalize'):
                nrm=np.linalg.norm(W,axis=1,keepdims=True)+1e-6; W/=(nrm/spec['normalize']).clip(1.0,None)

def gen_rollout(net, steps, rng=None):
    actions=[]; preds=[]; unstable=False
    for st in steps:
        o=obs_vec(st['ev'], st['sym'], st['dis'])
        out=net.step(o, rng)
        if not np.all(np.isfinite(out)): unstable=True; break
        actions.append(ACTIONS[int(np.argmax(out[:D_ACT]))])
        preds.append(decode(out[D_ACT:]))
        net.update(cons=0.0)                                    # E1: no ground-truth reward exists
    while len(actions)<len(steps): actions.append('HOLD'); preds.append(None)
    return actions, preds, unstable

def eval_gen(genome, neps=150, held=False, seed=SEED, plastic_on=True):
    net=GenNet(genome)
    if not plastic_on: net.plastic={}
    rng=random.Random(seed); per=[]
    for _ in range(neps):
        net.reset(); steps=make_episode(rng, held=held)
        a,p,unst=gen_rollout(net,steps)
        per.append(score_rollout(steps,a,p))
    third=max(1,neps//3); av=lambda L,k: round(float(np.mean([d[k] for d in L])),3)
    return {'probe':av(per[-third:],'probe'),'survival':av(per[-third:],'survival'),'false_resist':av(per[-third:],'false_resist')}

# ---- HAND-BUILT GENERIC BINDING GENOME (researcher validation; NEVER shown to synthesizers) ----
# mod-gated DELTA/LMS rule from generic signals: at adoption events drive the embedding-output rows
# toward the OBSERVED embedding and away from own prediction -> deltaW = lr*mod0*(obsemb-pred)(x)h,
# plus gated recency wipe. Composed only from {mod0,obsemb,pred,pre,w}. No named primitive.
HANDBUILT_BIND = {
  'hidden_dim':64, 'n_mod':1,
  'weights':{
    'recurrent':{'gen':'sparse','seed':7,'scale':1.0,'sparsity':0.2,'spectral_radius':0.97},
    'input':{'gen':'dense','seed':8,'scale':0.6},
    'output':{'gen':'dense','seed':9,'scale':0.02},
    'mod_input':{'gen':'select','dims':[[EV_IDX['COMMIT'],EV_IDX['VALID_REL']]],'scale':1.0}},
  'dynamics':{'activation':'tanh','gain':1.0,'leak':0.15,'noise':0.0}, 'mod_activation':'relu',
  'plastic':{'output':{'terms':[{'coef':0.6,'sign':-1,'factors':['mod0','w']},
                                 {'coef':0.5,'factors':['mod0','obsemb','pre']},
                                 {'coef':0.5,'sign':-1,'factors':['mod0','pred','pre']}], 'clip':6.0}},
  'init_state':{'gen':'zeros'}}

if __name__=='__main__':
    print("=== PART 1 (embedding-output): generic-DSL EXPRESSIVENESS VALIDATION ===", flush=True)
    print(f"  OUTPUT=[{D_ACT} action][{D_SYM} embedding]; symbol=nearest dictionary; only raw observed embedding given", flush=True)
    on =eval_gen(HANDBUILT_BIND, plastic_on=True)
    onh=eval_gen(HANDBUILT_BIND, held=True, plastic_on=True)
    off=eval_gen(HANDBUILT_BIND, plastic_on=False)
    print(f"  hand-built GENERIC delta-rule binding : online probe={on['probe']:.2f} surv={on['survival']:.2f} | HELD probe={onh['probe']:.2f} surv={onh['survival']:.2f}", flush=True)
    print(f"  plasticity OFF (control)              : probe={off['probe']:.2f} surv={off['survival']:.2f}", flush=True)
    print(f"  INTERPRET: plastic>>off AND HELD works => generic DSL EXPRESSES binding without naming it, held-out intrinsic", flush=True)
    print("=== GENDEV_VALIDATION_DONE ===", flush=True)
