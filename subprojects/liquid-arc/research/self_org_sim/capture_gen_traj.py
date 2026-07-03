# Capture the DEVELOPMENTAL STREAM — how the answer FORMS, token by token (generation trajectory at the read
# layer), the becoming the endpoint snapshot integrated out. NOT the product (last-token hidden); the PROCESS.
# Per turn -> [n_response_tokens, d_m]. This is the substrate the Liquid co-develops with (coupling objective,
# self-supervised, NO value labels). Reuses EMPOWER missions (texts/g/trunc/val/fid kept for bookkeeping only).
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, json, urllib.request
U = 'http://127.0.0.1:8765'
def post(path, **kw):
    r = urllib.request.Request(U + path, data=json.dumps(kw).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())
def gen_traj(ctx, resp): return torch.tensor(post('/manifold_gen', context=ctx, response=resp)['h'])   # [n_resp, d_m]
obj = torch.load('/home/pokazge/checkpoints/objective_value_attn.pt', weights_only=False, map_location='cpu')
data = obj['data']
print('capturing GENERATION developmental trajectories for %d missions ...' % len(data), flush=True)
for gi, m in enumerate(data):
    g = m['g']; texts = m['texts']; trunc = m['trunc']
    hist = [{'role': 'user', 'content': 'Help me, step by step and in order, with this task: ' + g}]
    gen = []
    for t, (u, r) in enumerate(texts):
        ctx = hist if not bool(trunc[t]) else hist[-2:]
        gen.append(gen_traj(ctx + [{'role': 'user', 'content': u}], r))    # [n_resp_tokens, d_m] = how THIS answer develops
        hist += [{'role': 'user', 'content': u}, {'role': 'assistant', 'content': r}]
    m['gen'] = gen                                                          # list of variable-length [n_t, d_m]
    if gi % 8 == 0: print('mission', gi, 'turns', len(gen), 'tok-lens', [t.shape[0] for t in gen], flush=True)
torch.save({'objective': obj['objective'], 'data': data}, '/home/pokazge/checkpoints/objective_value_gen.pt')
print('saved generation developmental stream === ALL_DONE ===')
