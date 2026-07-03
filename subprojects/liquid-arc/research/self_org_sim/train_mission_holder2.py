# Holder v2 = reconstruction (holds, validated) + InfoNCE contrastive (sharpens which-mission
# discrimination -> fixes the 23% exact-retrieval). Same data/arch, client-side CPU.
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn.functional as F, numpy as np
from train_steer_controller import SteerController
torch.manual_seed(0)
data = torch.load('/home/pokazge/checkpoints/mission_seqs.pt', weights_only=False, map_location='cpu')
dev = torch.device('cpu'); n = len(data)
tr = data[:int(n * 0.75)]; te = data[int(n * 0.75):]
Ztr = torch.stack([F.normalize(m['z'], dim=0) for m in tr])        # [ntr,384] in-batch negatives
bank = torch.stack([F.normalize(m['z'], dim=0) for m in data])     # full 64 bank for retrieval eval
# distinct-cluster ids: merge missions with mutual cos>0.9 (near-duplicate templated variants)
S = bank @ bank.T; cid = list(range(n))
for i in range(n):
    for j in range(i):
        if S[i, j] > 0.9: cid[i] = cid[j]; break
cid = torch.tensor(cid)
LR, EP, TAU, LREC, LCON = 1e-3, 400, 0.1, 1.0, 0.5
ctrl = SteerController(d_llm=2048, z_goal_dim=384, d=128, K=8, use_slow=True).to(dev)
opt = torch.optim.Adam(ctrl.parameters(), lr=LR, weight_decay=1e-5)

def rollout(m):
    ctrl.reset_episode(1, dev); ctrl.slow_step(m['seq'][0].unsqueeze(0))
    z = F.normalize(m['z'], dim=0); ests = []
    for t in range(m['seq'].shape[0]):
        h = ctrl.dyn_state(m['seq'][t].unsqueeze(0)); ctrl.h_c = h
        ests.append(F.normalize(ctrl.g_head(h.flatten(1)).squeeze(0), dim=0))
    return torch.stack(ests), z, m['trunc']

for ep in range(EP):
    opt.zero_grad(); H = []; W = []
    rec = 0.0
    for i, m in enumerate(tr):
        e, z, tm = rollout(m)
        w = torch.where(tm, 2.5, 1.0)
        rec = rec + ((1 - (e * z).sum(-1)) * w).mean()
        H.append(e); W.append(tm)
    H = torch.stack(H)                                            # [ntr,T,384]
    con = 0.0
    for t in range(H.shape[1]):
        logits = (H[:, t, :] @ Ztr.T) / TAU                       # [ntr,ntr]
        wt = 2.5 if bool(W[0][t]) else 1.0
        con = con + wt * F.cross_entropy(logits, torch.arange(len(tr)))
    loss = LREC * rec / len(tr) + LCON * con / H.shape[1]
    loss.backward(); torch.nn.utils.clip_grad_norm_(ctrl.parameters(), 1.0); opt.step()
    if ep % 80 == 0:
        with torch.no_grad():
            h = []; r = []; hit = 0; tot = 0; hitC = 0
            base = list(range(int(n * 0.75), n))
            for k, m in enumerate(te):
                gi = base[k]; e, z, tm = rollout(m); c = (e * z).sum(-1); rr = (m['seq'] * z).sum(-1)
                h.append(float(c[tm].mean())); r.append(float(rr[tm].mean()))
                for t in range(e.shape[0]):
                    if bool(tm[t]):
                        j = int((bank @ e[t]).argmax()); tot += 1
                        hit += (j == gi); hitC += (cid[j] == cid[gi])
            print('ep %d  LIQUID=%.3f raw=%.3f  retr top1=%.0f%% distinct=%.0f%%' %
                  (ep, np.mean(h), np.mean(r), 100 * hit / tot, 100 * hitC / tot), flush=True)
with torch.no_grad():
    H = []; R = []; hit = 0; tot = 0; hitC = 0; base = list(range(int(n * 0.75), n))
    for k, m in enumerate(te):
        gi = base[k]; e, z, tm = rollout(m); c = (e * z).sum(-1); rr = (m['seq'] * z).sum(-1)
        H.append(float(c[tm].mean())); R.append(float(rr[tm].mean()))
        for t in range(e.shape[0]):
            if bool(tm[t]):
                j = int((bank @ e[t]).argmax()); tot += 1; hit += (j == gi); hitC += (cid[j] == cid[gi])
print('\n[holder2] HELD-OUT TRUNCATED: LIQUID=%.3f raw-LLM=%.3f' % (np.mean(H), np.mean(R)))
print('[holder2] retrieval exact top1=%.0f%%  distinct-cluster top1=%.0f%%  (%d clusters in %d-bank)' %
      (100 * hit / tot, 100 * hitC / tot, len(set(cid.tolist())), n))
torch.save({'controller': ctrl.state_dict()}, '/home/pokazge/checkpoints/mission_holder2.pt')
print('[holder2] === ALL_DONE ===')
