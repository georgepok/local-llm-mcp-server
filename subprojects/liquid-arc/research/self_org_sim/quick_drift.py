# Verify the SELF-FEEDING substrate actually DRIFTS (greedy self-feed often loops -> dead trajectory). Seed the LLM,
# feed its own output back as the next prompt (full history), 18 steps. Measure gist drift: centered cos(gist[t],gist[t+k])
# should DECAY with k if it drifts. Print sample text at step 0 / 9 / 17 to see it qualitatively. If it loops (autocorr
# stays high / text repeats), we enable sampling. This is the substrate for Liquid-as-context-compressor.
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn.functional as F, json, urllib.request
U = 'http://127.0.0.1:8765'
def post(path, **kw):
    r = urllib.request.Request(U + path, data=json.dumps(kw).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())
def gen(msgs): return post('/gen', messages=msgs)['text']
def gist(ctx, r): return torch.tensor(post('/manifold_gen', context=ctx, response=r)['h']).mean(0)
src = torch.load('/home/pokazge/checkpoints/objective_value_gen.pt', weights_only=False, map_location='cpu')
seeds = [src['data'][i]['g'] for i in (0, 15, 30, 45)]
for si, seed in enumerate(seeds):
    hist = [{'role': 'user', 'content': seed}]; gs = []; texts = []
    for step in range(18):
        r = gen(hist); gs.append(gist(hist[:], r)); texts.append(r)
        hist = hist + [{'role': 'assistant', 'content': r}, {'role': 'user', 'content': r}]   # self-feed: output -> next prompt
    G = torch.stack(gs); Gn = F.normalize(G - G.mean(0), dim=-1)
    print('\n=== seed %d: %s' % (si, seed[:60]), flush=True)
    print('   drift cos[t,t+k]: ' + '  '.join('k=%d %+.3f' % (k, float((Gn[:-k] * Gn[k:]).sum(-1).mean())) for k in (1, 3, 6, 12)))
    print('   step0 : %s' % texts[0][:90].replace(chr(10), ' '))
    print('   step9 : %s' % texts[9][:90].replace(chr(10), ' '))
    print('   step17: %s' % texts[17][:90].replace(chr(10), ' '))
print('\n=== ALL_DONE ===')
