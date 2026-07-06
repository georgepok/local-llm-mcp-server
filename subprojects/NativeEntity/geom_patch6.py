import io
PATH = '/home/pokazge/NativeEntity/world_pop.py'
FN = r'''def geom_meta():
    # (2) Where does rule-application become readable+generalizable? Layer sweep of cross-principle
    # action-readout transfer (train 2 principles -> test held principle), linear vs nonlinear(meta).
    LAYERS = [int(x) for x in os.environ.get('GEO_LAYERS', '8,16,24,32,40,48,56,60').split(',')]
    HELDPRIN = os.environ.get('GEO_HELD_PRIN', 'THRESHOLD')
    M = int(os.environ.get('GEO_M', '12'))
    ALLPRIN = ['MATCH', 'LOOKUP', 'THRESHOLD']
    SURFACES = ['ledger', 'vault', 'archive', 'pipeline', 'roster', 'gauge']
    print('=== GEOM_META | held=%s layers=%s ===' % (HELDPRIN, LAYERS), flush=True)
    rng = random.Random(SEED)

    def gen_lex(prin, surf):
        order = LATENTS[:]; rng.shuffle(order); labs = ['KEEP', 'REJECT', 'DEFER', 'ASK']; lp = labs[:]; rng.shuffle(lp)
        mp = dict(zip(order, lp)); anchor = order[rng.randrange(4)]; q = LATENTS[rng.randrange(4)]
        spec = ('ASSESSMENT for the %s. Items in order: %s. Designated item: %s. Mapping: %s.'
                % (surf, ', '.join(order), anchor, ', '.join('%s=%s' % (t, mp[t]) for t in order)))
        if prin == 'LOOKUP':
            rule = 'Rule: the ruling is the mapping value listed for the query item.'; c = mp[q]
        elif prin == 'MATCH':
            rule = 'Rule: the ruling is KEEP if the query item is the designated item, otherwise REJECT.'
            c = 'KEEP' if q == anchor else 'REJECT'
        else:
            rule = ('Rule: the ruling is KEEP if the query item is at or before the designated item '
                    'in the order, otherwise REJECT.')
            c = 'KEEP' if order.index(q) <= order.index(anchor) else 'REJECT'
        return spec + ' ' + rule + ' QUERY item: %s. %s' % (q, ASK_INSTR), ACTIONS.index(c)

    worlds = []
    for prin in ALLPRIN:
        for surf in SURFACES:
            for _ in range(M):
                p, a = gen_lex(prin, surf); worlds.append({'held': (prin == HELDPRIN), 'prompt': p, 'a': a})

    @torch.no_grad()
    def getfeats(prompt):
        ids = tok(H.tmpl([{'role': 'user', 'content': prompt}]), return_tensors='pt').input_ids.to(dev)
        hs = model(ids, output_hidden_states=True).hidden_states
        d = {}
        for L in LAYERS:
            h = hs[L][0].float()
            d['L%02d_mean' % L] = h.mean(0).to(torch.float16).cpu()
            d['L%02d_last' % L] = h[-1].to(torch.float16).cpu()
        return d

    print('GEOMM collecting %d ...' % len(worlds), flush=True)
    for i, w in enumerate(worlds):
        w['f'] = getfeats(w['prompt'])
        if (i + 1) % 72 == 0: print('  %d/%d' % (i + 1, len(worlds)), flush=True)

    y_act = torch.tensor([w['a'] for w in worlds]); held = torch.tensor([w['held'] for w in worlds]); tr = ~held
    import collections as _cl
    _hb = _cl.Counter([worlds[i]['a'] for i in range(len(worlds)) if worlds[i]['held']])
    base = max(_hb.values()) / float(sum(_hb.values()))
    print('GEOMM held-principle base-rate = %.3f' % base, flush=True)

    def readout(Xtr, ytr, Xte, yte, mlp=False, epochs=500):
        mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
        Xtr, Xte = ((Xtr - mu) / sd).to(dev), ((Xte - mu) / sd).to(dev); ytr, yte = ytr.to(dev), yte.to(dev)
        d = Xtr.shape[1]
        net = (nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 6))
               if mlp else nn.Linear(d, 6)).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3); net.train()
        for e in range(epochs): opt.zero_grad(); F.cross_entropy(net(Xtr), ytr).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            return float((net(Xte).argmax(1) == yte).float().mean())

    hi = [i for i in range(len(worlds)) if worlds[i]['held']]; random.Random(SEED).shuffle(hi); half = len(hi) // 2
    print('GEOMM  feature       in-principle  cross-prin(lin)  cross-prin(mlp)   (base=%.3f)' % base, flush=True)
    for v in sorted(worlds[0]['f'].keys()):
        Z = torch.stack([w['f'][v].float() for w in worlds])
        inpr = readout(Z[hi[:half]], y_act[hi[:half]], Z[hi[half:]], y_act[hi[half:]])
        xl = readout(Z[tr], y_act[tr], Z[held], y_act[held], mlp=False)
        xm = readout(Z[tr], y_act[tr], Z[held], y_act[held], mlp=True)
        print('GEOMM  %-11s   %.3f         %.3f            %.3f' % (v, inpr, xl, xm), flush=True)
    print('=== GEOM_META_DONE ===', flush=True)


'''
src = io.open(PATH, encoding='utf-8').read()
if 'def geom_meta()' in src:
    print('ALREADY'); raise SystemExit
anchor = "if MODE == 'validate':   validate()"
src = src.replace(anchor, FN + '\n\n' + anchor, 1)
src = src.replace("elif MODE == 'geom_novel': geom_novel()",
                  "elif MODE == 'geom_novel': geom_novel()\nelif MODE == 'geom_meta': geom_meta()", 1)
io.open(PATH, 'w', encoding='utf-8').write(src)
print('PATCHED_OK')
