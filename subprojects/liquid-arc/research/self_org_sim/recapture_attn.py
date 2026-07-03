# Re-capture the ROUTING (attention-pattern features: entropy/recency/peak/recent-mass per full-attn
# layer) for the EMPOWER trajectories, reusing EMPOWER values. This is HOW the model selected
# features (the routing dynamics), NOT the selected features. 'aseq' = [turns, 16, 4].
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, json, urllib.request
U = 'http://127.0.0.1:8765'
def post(path, **kw):
    r = urllib.request.Request(U + path, data=json.dumps(kw).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())
def routing(msgs): return torch.tensor(post('/manifold_attn', messages=msgs)['h'])   # [16, 4]
obj = torch.load('/home/pokazge/checkpoints/objective_value_seqs.pt', weights_only=False, map_location='cpu')
data = obj['data']
print('re-capturing ROUTING for %d missions ...' % len(data), flush=True)
for gi, m in enumerate(data):
    g = m['g']; texts = m['texts']; trunc = m['trunc']
    hist = [{'role': 'user', 'content': 'Help me, step by step and in order, with this task: ' + g}]
    aseq = []
    for t, (u, r) in enumerate(texts):
        ctx = hist if not bool(trunc[t]) else hist[-2:]
        full_ctx = ctx + [{'role': 'user', 'content': u}, {'role': 'assistant', 'content': r}]
        aseq.append(routing(full_ctx))                            # [16, 4] routing across full-attn layers
        hist += [{'role': 'user', 'content': u}, {'role': 'assistant', 'content': r}]
    m['aseq'] = torch.stack(aseq)                                  # [turns, 16, 4]
    if gi % 8 == 0: print('mission', gi, 'aseq', tuple(m['aseq'].shape), flush=True)
torch.save({'objective': obj['objective'], 'data': data}, '/home/pokazge/checkpoints/objective_value_attn.pt')
print('saved', len(data), 'missions with routing === ALL_DONE ===')
