# Re-capture the FULL-CONTEXT token hiddens (context+response, recent window) for the EMPOWER
# trajectories, reusing EMPOWER values. The belief will ATTEND over ALL these positions (whole-context
# info + goal-feature extraction) -> should clear the 0.899 snapshot ceiling. 'cseq' = list of [<=160, d_m].
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, json, urllib.request
U = 'http://127.0.0.1:8765'
def post(path, **kw):
    r = urllib.request.Request(U + path, data=json.dumps(kw).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())
def ctx_tokens(msgs): return torch.tensor(post('/manifold_ctx', messages=msgs)['h'])   # [<=160, d_m]
obj = torch.load('/home/pokazge/checkpoints/objective_value_seqs.pt', weights_only=False, map_location='cpu')
data = obj['data']
print('re-capturing FULL-CONTEXT tokens for %d missions ...' % len(data), flush=True)
for gi, m in enumerate(data):
    g = m['g']; texts = m['texts']; trunc = m['trunc']
    hist = [{'role': 'user', 'content': 'Help me, step by step and in order, with this task: ' + g}]
    cseq = []
    for t, (u, r) in enumerate(texts):
        ctx = hist if not bool(trunc[t]) else hist[-2:]
        full_ctx = ctx + [{'role': 'user', 'content': u}, {'role': 'assistant', 'content': r}]
        cseq.append(ctx_tokens(full_ctx))                          # [<=160, d_m] all context+response positions
        hist += [{'role': 'user', 'content': u}, {'role': 'assistant', 'content': r}]
    m['cseq'] = cseq
    if gi % 8 == 0: print('mission', gi, 'ctx lens', [t.shape[0] for t in cseq], flush=True)
torch.save({'objective': obj['objective'], 'data': data}, '/home/pokazge/checkpoints/objective_value_ctx.pt')
print('saved', len(data), 'missions with full-context tokens === ALL_DONE ===')
