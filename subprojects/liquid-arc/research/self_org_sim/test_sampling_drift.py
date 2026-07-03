# Confirm (1) the new sampling param works (coherent, not NaN) and (2) it fixes the greedy LOOP — self-feed the bike
# tutorial seed (which looped verbatim at greedy) with temperature; it should now DRIFT (autocorr decays, text evolves).
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn.functional as F, json, urllib.request
U = 'http://127.0.0.1:8765'
def post(path, **kw):
    r = urllib.request.Request(U + path, data=json.dumps(kw).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())
def gen(msgs, temp): return post('/gen', messages=msgs, max_new=60, temp=temp)['text']
def gist(ctx, r): return torch.tensor(post('/manifold_gen', context=ctx, response=r)['h']).mean(0)
print('SAMPLING SANITY (temp=0.8):', gen([{'role': 'user', 'content': 'Say one vivid sentence about the sea.'}], 0.8)[:90].replace(chr(10), ' '), flush=True)
seed = 'write a tutorial on how to change a bike tire'
for temp in (0.7, 1.0):
    hist = [{'role': 'user', 'content': seed}]; gs = []; texts = []
    for step in range(10):
        r = gen(hist, temp); gs.append(gist(hist[:], r)); texts.append(r)
        hist = hist + [{'role': 'assistant', 'content': r}, {'role': 'user', 'content': r}]
    G = torch.stack(gs); Gn = F.normalize(G - G.mean(0), dim=-1)
    print('\ntemp=%.1f  drift cos: %s' % (temp, '  '.join('k=%d %+.3f' % (k, float((Gn[:-k] * Gn[k:]).sum(-1).mean())) for k in (1, 3, 6))), flush=True)
    print('  step0: %s' % texts[0][:85].replace(chr(10), ' '))
    print('  step9: %s' % texts[9][:85].replace(chr(10), ' '))
print('=== ALL_DONE ===')
