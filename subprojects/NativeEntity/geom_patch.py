import io
PATH = '/home/pokazge/NativeEntity/world_pop.py'
FN = r'''
def geom_phase():
    # GEOMETRY_PHASE_TRANSITION_DIAGNOSTIC_V1: does training reorganize S from SURFACE-clustered to
    # PRINCIPLE-clustered geometry? Crossed grid PRINCIPLES x SURFACES with held-out crossed cells.
    import copy
    PRINCIPLES = ['MATCH', 'LOOKUP', 'THRESHOLD']
    SURFACES   = ['ledger', 'vault', 'archive', 'pipeline', 'roster', 'gauge']
    HELDMAP = {'ledger': 'LOOKUP', 'vault': 'MATCH', 'archive': 'THRESHOLD',
               'pipeline': 'THRESHOLD', 'roster': 'LOOKUP', 'gauge': 'MATCH'}
    M     = int(os.environ.get('GEO_M', '10'))
    ITERS = int(os.environ.get('GEO_ITERS', '2000'))
    EVERY = int(os.environ.get('GEO_EVAL_EVERY', '100'))
    LR    = float(os.environ.get('GEO_LR', '1e-3'))
    NTOK  = int(os.environ.get('GEO_NTOK', '12'))
    print('=== GEOM_PHASE | %dx%d grid M=%d iters=%d lr=%.1e ntok=%d ===' %
          (len(PRINCIPLES), len(SURFACES), M, ITERS, LR, NTOK), flush=True)
    rng = random.Random(SEED)

    def gen(prin, surf):
        if prin == 'MATCH':
            X = LATENTS[rng.randrange(4)]; Y = LATENTS[rng.randrange(4)]
            commit = 'RECORD for the %s: the authorized value on file is %s %s %s.' % (surf, X, X, X)
            dec = ("CLAIM received: 'the value is %s.' Reply KEEP if it matches the record, "
                   "REJECT if it conflicts. %s" % (Y, ASK_INSTR))
            c = 'KEEP' if Y == X else 'REJECT'
        elif prin == 'LOOKUP':
            codes = LATENTS[:]; acts = ['KEEP', 'REJECT', 'DEFER', 'ASK']; rng.shuffle(acts)
            m = dict(zip(codes, acts)); Z = LATENTS[rng.randrange(4)]
            commit = 'ROUTING TABLE for the %s: %s.' % (surf, ', '.join('%s->%s' % (k, m[k]) for k in codes))
            dec = 'REQUEST tagged %s. Reply with the ruling the table assigns. %s' % (Z, ASK_INSTR)
            c = m[Z]
        else:
            order = LATENTS[:]; rng.shuffle(order); cp = rng.randrange(4); R = LATENTS[rng.randrange(4)]
            commit = ('CLEARANCE LADDER for the %s (lowest to highest): %s. Ceiling set at %s.'
                      % (surf, ', '.join(order), order[cp]))
            dec = 'REQUEST at level %s. Reply KEEP if at or below the ceiling, REJECT if above. %s' % (R, ASK_INSTR)
            c = 'KEEP' if order.index(R) <= cp else 'REJECT'
        return commit + '\n\n' + dec, ACTIONS.index(c)

    worlds = []
    for pi, prin in enumerate(PRINCIPLES):
        for si, surf in enumerate(SURFACES):
            held = (HELDMAP[surf] == prin)
            for _ in range(M):
                p, a = gen(prin, surf); worlds.append({'pi': pi, 'si': si, 'held': held, 'prompt': p, 'a': a})

    @torch.no_grad()
    def getH(prompt):
        ids = tok(H.tmpl([{'role': 'user', 'content': prompt}]), return_tensors='pt').input_ids.to(dev)
        ho = model(ids, output_hidden_states=True).hidden_states[READ_LAYER][0]
        return ho[-NTOK:].float().to(torch.float16).cpu()

    print('GEOM collecting %d worlds ...' % len(worlds), flush=True)
    for i, w in enumerate(worlds):
        w['H'] = getH(w['prompt'])
        if (i + 1) % 60 == 0: print('  H %d/%d' % (i + 1, len(worlds)), flush=True)

    y_prin = torch.tensor([w['pi'] for w in worlds]); y_surf = torch.tensor([w['si'] for w in worlds])
    y_act = torch.tensor([w['a'] for w in worlds]); held = torch.tensor([w['held'] for w in worlds])
    tr = ~held

    def probe_th(Z, ylab, nclass, epochs=400):
        Xtr, ytr = Z[tr], ylab[tr]; Xho, yho = Z[held], ylab[held]
        mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
        Xtr, Xho = ((Xtr - mu) / sd).to(dev), ((Xho - mu) / sd).to(dev); ytr, yho = ytr.to(dev), yho.to(dev)
        net = nn.Linear(Xtr.shape[1], nclass).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3); net.train()
        for e in range(epochs): opt.zero_grad(); F.cross_entropy(net(Xtr), ytr).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            fit = float((net(Xtr).argmax(1) == ytr).float().mean())
            tra = float((net(Xho).argmax(1) == yho).float().mean())
        return fit, tra

    Hpool = torch.stack([w['H'].float().mean(0) for w in worlds])
    for lab, yl, nc in [('principle', y_prin, 3), ('surface', y_surf, 6)]:
        f, t = probe_th(Hpool, yl, nc)
        print('GEOM BASELINE rawH %-9s fit=%.3f held-transfer=%.3f (chance %.3f)' % (lab, f, t, 1.0 / nc), flush=True)

    g = AdaptiveGateSlot(D_MODEL, D_S, K, SLOW_K).to(dev); head = nn.Linear(D_S, 6).to(dev)
    opt = torch.optim.Adam(list(g.parameters()) + list(head.parameters()), lr=LR)

    def Sof(w): return g.step(g.init(), w['H'].float().to(dev))

    TRAINi = [i for i in range(len(worlds)) if not worlds[i]['held']]

    def evalgeo(it):
        g.eval(); head.eval()
        with torch.no_grad():
            Z = torch.stack([Sof(worlds[i]).mean(0).cpu() for i in range(len(worlds))])
            logits = torch.stack([head(Sof(worlds[i]).mean(0)) for i in range(len(worlds))]).cpu()
        pf, pt = probe_th(Z, y_prin, 3); sf, st = probe_th(Z, y_surf, 6)
        pred = logits.argmax(1)
        btr = float((pred[tr] == y_act[tr]).float().mean()); bho = float((pred[held] == y_act[held]).float().mean())
        print('GEOM it=%-4d | S-PRIN fit=%.3f HELD=%.3f | S-SURF fit=%.3f HELD=%.3f | behavior train=%.3f HELD=%.3f'
              % (it, pf, pt, sf, st, btr, bho), flush=True)
        g.train(); head.train()

    evalgeo(0)
    rng2 = random.Random(SEED + 1)
    for it in range(1, ITERS + 1):
        w = worlds[TRAINi[rng2.randrange(len(TRAINi))]]
        S = Sof(w); logit = head(S.mean(0))
        loss = F.cross_entropy(logit.unsqueeze(0), torch.tensor([w['a']], device=dev))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(g.parameters()) + list(head.parameters()), 1.0); opt.step()
        if it % EVERY == 0: evalgeo(it)
    print('=== GEOM_PHASE_DONE ===', flush=True)
'''

src = io.open(PATH, encoding='utf-8').read()
if 'def geom_phase()' in src:
    print('ALREADY_PATCHED'); raise SystemExit
anchor = "if MODE == 'validate':   validate()"
assert anchor in src, 'anchor not found'
src = src.replace(anchor, FN + '\n\n' + anchor, 1)
disp = "else:                    substrate()"
assert disp in src, 'dispatch not found'
src = src.replace(disp, "elif MODE == 'geom_phase': geom_phase()\n" + disp, 1)
io.open(PATH, 'w', encoding='utf-8').write(src)
print('PATCHED_OK')
