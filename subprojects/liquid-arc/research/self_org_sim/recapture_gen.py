# Re-capture the GENERATION TRAJECTORY (per-response-token hidden flow) for the EMPOWER trajectories,
# reusing the EMPOWER values. This is the dimension the snapshot-variants (single/multi/flow) all
# discarded: HOW the representation moves WHILE the answer is produced. Ragged (variable response
# length) -> 'gseq' = list of [n_resp_t, d_m] per turn.
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, json, urllib.request
U = 'http://127.0.0.1:8765'
def post(path, **kw):
    r = urllib.request.Request(U + path, data=json.dumps(kw).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())
def gen_traj(context, response): return torch.tensor(post('/manifold_gen', context=context, response=response)['h'])  # [n_resp, d_m]
obj = torch.load('/home/pokazge/checkpoints/objective_value_seqs.pt', weights_only=False, map_location='cpu')
data = obj['data']
print('re-capturing GENERATION TRAJECTORY for %d missions ...' % len(data), flush=True)
for gi, m in enumerate(data):
    g = m['g']; texts = m['texts']; trunc = m['trunc']
    hist = [{'role': 'user', 'content': 'Help me, step by step and in order, with this task: ' + g}]
    gseq = []
    for t, (u, r) in enumerate(texts):
        ctx = hist if not bool(trunc[t]) else hist[-2:]
        context = ctx + [{'role': 'user', 'content': u}]            # context up to (but not incl.) the response
        gseq.append(gen_traj(context, r))                          # [n_resp_t, d_m] the generation path for response r
        hist += [{'role': 'user', 'content': u}, {'role': 'assistant', 'content': r}]
    m['gseq'] = gseq                                                # list of 6 ragged tensors
    if gi % 8 == 0: print('mission', gi, 'resp lens', [t.shape[0] for t in gseq], flush=True)
torch.save({'objective': obj['objective'], 'data': data}, '/home/pokazge/checkpoints/objective_value_gen.pt')
print('saved', len(data), 'missions with generation trajectory === ALL_DONE ===')
