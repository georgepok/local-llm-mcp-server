# Re-capture the FLOW (inter-layer residual-stream deltas) for the EMPOWER trajectories, reusing the
# EMPOWER values. The flow = what each block of layers ADDS (raw deltas, magnitude kept) = genuine
# dynamics, not a normalized depth-silhouette. Output: 'fseq' [turns, n_deltas, d_m].
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, json, urllib.request
U = 'http://127.0.0.1:8765'
def post(path, **kw):
    r = urllib.request.Request(U + path, data=json.dumps(kw).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())
def flow(msgs): return torch.tensor(post('/manifold_flow', messages=msgs)['h'])   # [n_deltas, d_m]
obj = torch.load('/home/pokazge/checkpoints/objective_value_seqs.pt', weights_only=False, map_location='cpu')
data = obj['data']
print('re-capturing FLOW (inter-layer deltas) for %d missions ...' % len(data), flush=True)
for gi, m in enumerate(data):
    g = m['g']; texts = m['texts']; trunc = m['trunc']
    hist = [{'role': 'user', 'content': 'Help me, step by step and in order, with this task: ' + g}]
    fseq = []
    for t, (u, r) in enumerate(texts):
        ctx = hist if not bool(trunc[t]) else hist[-2:]
        full_ctx = ctx + [{'role': 'user', 'content': u}, {'role': 'assistant', 'content': r}]
        fseq.append(flow(full_ctx))
        hist += [{'role': 'user', 'content': u}, {'role': 'assistant', 'content': r}]
    m['fseq'] = torch.stack(fseq)                                    # [turns, n_deltas, d_m]
    if gi % 8 == 0: print('mission', gi, 'fseq', tuple(m['fseq'].shape), 'mean|delta|', round(float(m['fseq'].norm(dim=-1).mean()), 1), flush=True)
torch.save({'objective': obj['objective'], 'data': data}, '/home/pokazge/checkpoints/objective_value_flow.pt')
print('saved', len(data), 'missions with FLOW === ALL_DONE ===')
