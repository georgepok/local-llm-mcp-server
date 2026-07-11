import os, random
import numpy as np, torch, torch.nn as nn
os.environ['ME_MODE']='none'   # suppress micro_entity smoke on import
from e2e3 import EMB, make_ep_e2, _obs_e2, make_plan_e3, _obs_e3, TRAIN_SYM, HELD_SYM
torch.manual_seed(0); random.seed(0); np.random.seed(0)
DEV='cpu'
# Solvability ceiling: can a GRADIENT-trained RNN solve E2/E3? If yes, the local-plasticity genomes' failure
# is a mechanism gap (local rules can't learn unobserved transforms / reversal), not task-impossibility.

class GRUNet(nn.Module):
    def __init__(self,h=96):
        super().__init__(); self.g=nn.GRU(14,h,batch_first=True); self.o=nn.Linear(h,21)
    def forward(self,x): y,_=self.g(x); return self.o(y)

def pad(seqs, dim):
    ml=max(len(s) for s in seqs); B=len(seqs)
    X=np.zeros((B,ml,dim),np.float32)
    for i,s in enumerate(seqs): X[i,:len(s)]=s
    return X, ml

# ---------------- E2 ----------------
def e2_batch(bs, held, rng):
    obss=[]; tgs=[]
    for _ in range(bs):
        steps,K=make_ep_e2(rng, held=held)
        obss.append(np.stack([_obs_e2(s['ev'],s['v8']) for s in steps]))
        tgs.append(np.array([s['ans'] if s['ev']=='PROBE' else -1 for s in steps],np.int64))
    ml=max(len(o) for o in obss); B=bs
    X=np.zeros((B,ml,14),np.float32); T=np.full((B,ml),-1,np.int64)
    for i in range(B): X[i,:len(obss[i])]=obss[i]; T[i,:len(tgs[i])]=tgs[i]
    return torch.tensor(X),torch.tensor(T)

def train_e2():
    net=GRUNet().to(DEV); opt=torch.optim.Adam(net.parameters(),2e-3); lf=nn.CrossEntropyLoss(ignore_index=-1)
    rng=random.Random(1)
    for it in range(1800):
        X,T=e2_batch(64,False,rng); out=net(X)[...,5:21].reshape(-1,16)
        loss=lf(out,T.reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
    def ev(held):
        r=random.Random(7); acc=[]
        for _ in range(40):
            X,T=e2_batch(32,held,r); p=net(X)[...,5:21].argmax(-1); m=T>=0
            acc.append(float(((p==T)&m).sum().item()/max(1,m.sum().item())))
        return float(np.mean(acc))
    print(f"  TRAINED-GRU E2 (BPTT, 1800 steps): train_probe={ev(False):.2f}  HELD_probe={ev(True):.2f}  (chance 0.06)", flush=True)

# ---------------- E3 ----------------
def e3_seq(held, rng):
    plan,rev=make_plan_e3(rng, held=held); obs=[]; tsym=[]; tact=[]; lr=0.0; la=0.0
    for tr in plan:
        a=rng.randint(0,1); accept=(a==0); is_rew=(tr['shown']==tr['rewarded']); correct=int(accept==is_rew); reward=1.0 if correct else -1.0
        obs.append(_obs_e3('CHOICE_PROMPT',EMB[tr['shown']],lr,la)); tsym.append(tr['rewarded']); tact.append(0 if is_rew else 1)
        ev='REWARD_POS' if reward>0 else 'REWARD_NEG'
        obs.append(_obs_e3(ev,EMB[tr['shown']],reward,float(a))); tsym.append(tr['rewarded']); tact.append(-1)
        lr=reward; la=float(a)
    return np.stack(obs),np.array(tsym,np.int64),np.array(tact,np.int64),rev

def e3_batch(bs, held, rng):
    S=[];TS=[];TA=[]
    for _ in range(bs):
        o,ts,ta,rev=e3_seq(held,rng); S.append(o);TS.append(ts);TA.append(ta)
    ml=max(len(s) for s in S)
    X=np.zeros((bs,ml,14),np.float32); Ts=np.full((bs,ml),-1,np.int64); Ta=np.full((bs,ml),-1,np.int64)
    for i in range(bs): X[i,:len(S[i])]=S[i]; Ts[i,:len(TS[i])]=TS[i]; Ta[i,:len(TA[i])]=TA[i]
    return torch.tensor(X),torch.tensor(Ts),torch.tensor(Ta)

def train_e3():
    net=GRUNet().to(DEV); opt=torch.optim.Adam(net.parameters(),2e-3); lf=nn.CrossEntropyLoss(ignore_index=-1)
    rng=random.Random(2)
    for it in range(1800):
        X,Ts,Ta=e3_batch(64,False,rng); out=net(X)
        lsym=lf(out[...,5:21].reshape(-1,16),Ts.reshape(-1)); lact=lf(out[...,:5].reshape(-1,5),Ta.reshape(-1))
        loss=lsym+lact; opt.zero_grad(); loss.backward(); opt.step()
    def ev(held):
        r=random.Random(8); post=[]; pr=[]
        for _ in range(60):
            o,ts,ta,rev=e3_seq(held,r); X=torch.tensor(o[None]); out=net(X)[0]
            act=out[:,:5].argmax(-1).numpy(); sym=out[:,5:21].argmax(-1).numpy()
            # choice steps are even indices; post-reversal = last 4 choice trials
            ch=[i for i in range(len(ta)) if ta[i]>=0]; postch=ch[-4:]
            post.append(float(np.mean([int(act[i]==ta[i]) for i in postch])))
            pr.append(float(np.mean([int(sym[i]==ts[i]) for i in range(len(ts))])))
        return float(np.mean(post)),float(np.mean(pr))
    p0,r0=ev(False); p1,r1=ev(True)
    print(f"  TRAINED-GRU E3 (BPTT, 1800 steps): train post_rev_acc={p0:.2f}/sym_track={r0:.2f}  HELD post_rev_acc={p1:.2f}/sym_track={r1:.2f}  (chance 0.50/0.06)", flush=True)

if __name__=='__main__':
    print("=== SOLVABILITY CEILING: gradient-trained GRU (establishes tasks ARE solvable) ===", flush=True)
    train_e2(); train_e3()
    print("=== TRAINBASE_DONE ===", flush=True)
