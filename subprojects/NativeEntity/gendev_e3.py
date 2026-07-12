import os, random
os.environ['ME_MODE']='none'
import numpy as np
from gendev import GenNet, decode, D_ACT, D_OUT, DICT
from micro_entity import ALPHABET, SYM_EMB

# PRIMITIVE_HIDDEN_LINEAGE_SYNTHESIS_V1 — E3 (delayed reversal credit) in the GENERIC DSL.
# Reward-modulated eligibility (reward_hebb / eligibility_trace) is NOT exposed as a primitive;
# the DSL only gives {cons scalar consequence, traces, pre/post/obsemb/w/mod}. The organism must
# COMPOSE a delayed-credit rule. Expressiveness validation: a hand-built generic reward-eligibility
# rule must track the rewarded symbol AND re-adapt after the mid-episode reversal.
np.seterr(over='ignore', invalid='ignore')
TRAIN=list(range(12)); HELD=list(range(12,16))
EMB=np.stack([SYM_EMB[s] for s in ALPHABET]).astype(np.float32)
E3EV={'CHOICE':0,'REWARD_POS':1,'REWARD_NEG':2,'PROBE':5}   # obs dims; symbol emb at 6:14 (reuses GenNet obsemb)
def obs_e3(ev, sym):
    o=np.zeros(14,np.float32); o[E3EV[ev]]=1.0
    if sym is not None: o[6:14]=EMB[sym]
    return o

def make_steps_e3(rng, held=False, ntrial=20, rev=10):
    pool=HELD if held else TRAIN
    A,B=rng.sample(pool,2); rewarded=A; steps=[]
    for t in range(ntrial):
        if t==rev: rewarded = B if rewarded==A else A
        c = A if t%2==0 else B
        steps.append({'ev':'CHOICE','sym':c,'cons':0.0,'probe':False,'rew':rewarded,'t':t})
        cons = 1.0 if c==rewarded else -1.0
        # reward arrives with the symbol ABSENT -> a trace of the just-chosen symbol is REQUIRED (genuine delayed credit)
        steps.append({'ev':'REWARD_POS' if cons>0 else 'REWARD_NEG','sym':None,'cons':cons,'probe':False,'rew':rewarded,'t':t})
        if t%2==1: steps.append({'ev':'PROBE','sym':None,'cons':0.0,'probe':True,'rew':rewarded,'t':t})
    return steps, rev

def rollout_e3(net, steps):
    probes=[]; unstable=False
    for st in steps:
        out=net.step(obs_e3(st['ev'], st['sym']))
        if not np.all(np.isfinite(out)): unstable=True; break
        if st['probe']: probes.append((int(np.argmax(DICT@out[D_ACT:])), st['rew'], st['t']))
        net.update(cons=st['cons'])
    return probes, unstable

def eval_e3(genome, neps=150, held=False, seed=0, plastic_on=True):
    net=GenNet(genome)
    if not plastic_on: net.plastic={}
    rng=random.Random(seed); acc=[]; pre=[]; post=[]
    for ep in range(neps):
        net.reset(); steps,rev=make_steps_e3(rng, held=held)
        probes,unst=rollout_e3(net,steps)
        for p,r,t in probes:
            ok=int(p==r); acc.append(ok)
            (pre if t<rev else post).append(ok)
    av=lambda L: round(float(np.mean(L)),3) if L else 0.0
    return {'probe_acc':av(acc),'pre_rev':av(pre),'post_rev':av(post)}

# ---- HAND-BUILT GENERIC REWARD-ELIGIBILITY RULE (validation; NEVER shown to synthesizers) ----
# eligibility trace = decay*e + obsemb(x)carrier   (accumulates shown-symbol x key)
# deltaW = lr*cons*e   (reward-modulated: bind rewarded symbol, unbind unrewarded; reversal flips sign)
#         - decay*w    (slow forgetting). Composed only from {cons,obsemb,pre,w,traces}. No named primitive.
HANDBUILT_E3 = {
  'hidden_dim':64, 'n_mod':1,
  'weights':{
    'recurrent':{'gen':'dense','seed':7,'scale':1.0,'sparsity':0.2,'spectral_radius':0.9,'diag':0.7},
    'input':{'gen':'dense','seed':8,'scale':0.3},
    'output':{'gen':'dense','seed':9,'scale':0.02},
    'mod_input':{'gen':'select','dims':[[1,2]],'scale':1.0}},
  'dynamics':{'activation':'tanh','gain':1.6,'leak':0.2,'noise':0.0}, 'mod_activation':'relu',   # bistable stable carrier
  'plastic':{'output':{
     'traces':[{'name':'elig','decay':0.6,'factors':['obsemb','pre']}],   # holds just-chosen symbol x carrier across the delay
     'terms':[{'coef':1.2,'factors':['cons','elig']},                      # reward-modulated credit transfer
              {'coef':0.12,'sign':-1,'factors':['w']}], 'clip':4.0}},       # fast forgetting -> re-adapts on reversal
  'init_state':{'gen':'zeros'}}

if __name__=='__main__':
    print("=== E3 generic-DSL EXPRESSIVENESS VALIDATION (delayed reversal credit, no named primitive) ===", flush=True)
    on =eval_e3(HANDBUILT_E3, plastic_on=True)
    onh=eval_e3(HANDBUILT_E3, held=True, plastic_on=True)
    off=eval_e3(HANDBUILT_E3, plastic_on=False)
    print(f"  hand-built GENERIC reward-eligibility : probe_acc={on['probe_acc']:.2f} pre_rev={on['pre_rev']:.2f} post_rev={on['post_rev']:.2f}", flush=True)
    print(f"  HELD-out symbol pairs                 : probe_acc={onh['probe_acc']:.2f} pre_rev={onh['pre_rev']:.2f} post_rev={onh['post_rev']:.2f}", flush=True)
    print(f"  plasticity OFF (control)              : probe_acc={off['probe_acc']:.2f} (chance-of-2 ~0.5 if it fixates, decode-of-16 ~0.06)", flush=True)
    print(f"  INTERPRET: pre_rev AND post_rev both high => tracks reward AND re-adapts after reversal => E3 constructible generically", flush=True)
    print("=== GENDEV_E3_DONE ===", flush=True)
