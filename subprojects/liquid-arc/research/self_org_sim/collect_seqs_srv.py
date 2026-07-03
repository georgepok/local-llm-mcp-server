import sys; sys.path.insert(0,'/home/pokazge/liquid-arc/research/self_org_sim')
import torch, numpy as np, mclient
from task_goals import task_goals
DIST=['Who won the World Cup in 2018?','What is a good recipe for dinner tonight?','Tell me a fun fact about octopuses.','How does Wi-Fi work?','What is the capital of New Zealand?']
goals=task_goals(); data=[]; rng=np.random.default_rng(0)
N=int(sys.argv[1]) if len(sys.argv)>1 else 64
for gi,g in enumerate(goals[:N]):
    z=mclient.encode(g)
    hist=[{'role':'user','content':'Help me, step by step and in order, with this task: '+g}]
    seq=[]; trunc=[]
    prompts=[('full','What is the first concrete step?'),('full','Done. What should we do next?'),
             ('trunc',DIST[rng.integers(5)]),('trunc',DIST[rng.integers(5)]),
             ('trunc','What should I focus on now?'),('trunc','And what is the final step to finish?')]
    for mode,u in prompts:
        ctx = hist if mode=='full' else hist[-2:]
        r=mclient.gen(ctx+[{'role':'user','content':u}],42)
        ms=mclient.gen(ctx+[{'role':'user','content':u},{'role':'assistant','content':r},
                {'role':'user','content':'In ONE short line, restate the overall task and the next step.'}],36)
        seq.append(mclient.encode(ms)); trunc.append(mode=='trunc')
        hist+=[{'role':'user','content':u},{'role':'assistant','content':r}]
    data.append({'g':g,'z':z,'seq':torch.stack(seq),'trunc':torch.tensor(trunc)})
    if gi%8==0: print('mission',gi,'cos(seq->true):',[round(float((s*z).sum()),2) for s in seq],flush=True)
torch.save(data,'/home/pokazge/checkpoints/mission_seqs.pt'); print('saved',len(data),'=== ALL_DONE ===')
