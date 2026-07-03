import sys; sys.path.insert(0,'/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn.functional as F, numpy as np
from train_steer_controller import SteerController
torch.manual_seed(0)
data=torch.load('/home/pokazge/checkpoints/mission_seqs.pt',weights_only=False,map_location='cpu')
dev=torch.device('cpu')
n=len(data); tr=data[:int(n*0.75)]; te=data[int(n*0.75):]
ctrl=SteerController(d_llm=2048,z_goal_dim=384,d=128,K=8,use_slow=True).to(dev)
opt=torch.optim.Adam(ctrl.parameters(),lr=1e-3,weight_decay=1e-5)
def rollout(m):                                            # LIQUID integrates the reasoning stream (BPTT across turns)
    ctrl.reset_episode(1,dev); ctrl.slow_step(m['seq'][0].unsqueeze(0))   # seed slow w/ first reasoning (NOT z_true)
    z=F.normalize(m['z'],dim=0); ests=[]
    for t in range(m['seq'].shape[0]):
        h=ctrl.dyn_state(m['seq'][t].unsqueeze(0))         # integrate this turn's reasoning
        ctrl.h_c=h                                          # re-attach belief across turns -> learn to HOLD
        ests.append(F.normalize(ctrl.g_head(h.flatten(1)).squeeze(0),dim=0))
    return torch.stack(ests),z,m['trunc']
for ep in range(400):
    opt.zero_grad(); loss=0.0
    for m in tr:
        ests,z,tm=rollout(m)
        coss=(ests*z).sum(-1); w=torch.where(tm,2.5,1.0)   # weight TRUNCATED turns (the hard hold) higher
        loss=loss+((1-coss)*w).mean()
    (loss/len(tr)).backward(); torch.nn.utils.clip_grad_norm_(ctrl.parameters(),1.0); opt.step()
    if ep%80==0:
        with torch.no_grad():
            h=[];r=[]
            for m in te:
                e,z,tm=rollout(m); c=(e*z).sum(-1); rr=(m['seq']*z).sum(-1)
                h.append(float(c[tm].mean())); r.append(float(rr[tm].mean()))
            print('ep %d  held-out TRUNCATED turns: LIQUID-reconstruct=%.3f  raw-LLM=%.3f'%(ep,np.mean(h),np.mean(r)),flush=True)
with torch.no_grad():
    H=[];R=[];Hf=[];Rf=[]
    for m in te:
        e,z,tm=rollout(m); c=(e*z).sum(-1); rr=(m['seq']*z).sum(-1)
        H.append(float(c[tm].mean()));R.append(float(rr[tm].mean()))
        Hf.append(float(c[~tm].mean()));Rf.append(float(rr[~tm].mean()))
print('\n[holder] HELD-OUT MISSIONS:')
print('  full-context turns : LIQUID=%.3f  raw-LLM=%.3f'%(np.mean(Hf),np.mean(Rf)))
print('  TRUNCATED turns    : LIQUID=%.3f  raw-LLM=%.3f   (LIQUID holds the mission the LLM lost: %s)'%(np.mean(H),np.mean(R),'YES' if np.mean(H)>np.mean(R)+0.03 else 'no'))
torch.save({'controller':ctrl.state_dict()},'/home/pokazge/checkpoints/mission_holder.pt')
print('[holder] === ALL_DONE ===')
