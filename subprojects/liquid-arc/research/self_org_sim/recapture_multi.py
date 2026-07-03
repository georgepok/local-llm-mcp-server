# Re-capture the MULTI-LAYER depth-trajectory for the EMPOWER trajectories (texts saved), reusing the
# EMPOWER values. Fast: forwards only, no re-gen/re-judge. Reconstructs contexts EXACTLY as
# collect_objective_value did. Output: each mission gets 'mseq' [turns, n_layers, d_m].
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, json, urllib.request
U = 'http://127.0.0.1:8765'
def post(path, **kw):
    r = urllib.request.Request(U + path, data=json.dumps(kw).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())
def manifold_multi(msgs): return torch.tensor(post('/manifold_multi', messages=msgs)['h'])   # [n_layers, d_m]
obj = torch.load('/home/pokazge/checkpoints/objective_value_seqs.pt', weights_only=False, map_location='cpu')
data = obj['data']
print('re-capturing depth-trajectory for %d missions ...' % len(data), flush=True)
for gi, m in enumerate(data):
    g = m['g']; texts = m['texts']; trunc = m['trunc']
    hist = [{'role': 'user', 'content': 'Help me, step by step and in order, with this task: ' + g}]
    mseq = []
    for t, (u, r) in enumerate(texts):
        ctx = hist if not bool(trunc[t]) else hist[-2:]                     # SAME truncation logic as collect
        full_ctx = ctx + [{'role': 'user', 'content': u}, {'role': 'assistant', 'content': r}]
        mseq.append(manifold_multi(full_ctx))                              # [n_layers, d_m]
        hist += [{'role': 'user', 'content': u}, {'role': 'assistant', 'content': r}]
    m['mseq'] = torch.stack(mseq)                                          # [turns, n_layers, d_m]
    if gi % 8 == 0: print('mission', gi, 'mseq', tuple(m['mseq'].shape), flush=True)
torch.save({'objective': obj['objective'], 'data': data}, '/home/pokazge/checkpoints/objective_value_multi.pt')
print('saved', len(data), 'missions with multi-layer depth-trajectory === ALL_DONE ===')
