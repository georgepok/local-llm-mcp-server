import io
PATH = '/home/pokazge/NativeEntity/world_pop.py'
FN = r'''def geom_novel():
    # NOVEL_PRINCIPLE_HOLDOUT_V1: train g+head on 2 principles (ALL surfaces), hold out 1 ENTIRE principle.
    # Locate the wall: Qwen-gen ceiling / rawH in-principle & cross-principle / substrate behavior + S-principle.
    ALLPRIN = ['MATCH', 'LOOKUP', 'THRESHOLD']
    SURFACES = ['ledger', 'vault', 'archive', 'pipeline', 'roster', 'gauge']
    HELDPRIN = os.environ.get('GEO_HELD_PRIN', 'THRESHOLD')
    M = int(os.environ.get('GEO_M', '12')); ITERS = int(os.environ.get('GEO_ITERS', '3000'))
    EVERY = int(os.environ.get('GEO_EVAL_EVERY', '300')); LR = float(os.environ.get('GEO_LR', '1e-4'))
    NTOK = int(os.environ.get('GEO_NTOK', '12'))
    print('=== GEOM_NOVEL | held-principle=%s train=%s M=%d ===' %
          (HELDPRIN, [p for p in ALLPRIN if p != HELDPRIN], M), flush=True)
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
    for pi, prin in enumerate(ALLPRIN):
        for surf in SURFACES:
            for _ in range(M):
                p, a = gen_lex(prin, surf); worlds.append({'pi': pi, 'held': (prin == HELDPRIN), 'prompt': p, 'a': a})

    @torch.no_grad()
    def getH(prompt):
        ids = tok(H.tmpl([{'role': 'user', 'content': prompt}]), return_tensors='pt').input_ids.to(dev)
        return model(ids, output_hidden_states=True).hidden_states[READ_LAYER][0][-NTOK:].float().to(torch.float16).cpu()

    print('GEOMN collecting %d ...' % len(worlds), flush=True)
    for i, w in enumerate(worlds):
        w['H'] = getH(w['prompt'])
        if (i + 1) % 72 == 0: print('  H %d/%d' % (i + 1, len(worlds)), flush=True)

    y_act = torch.tensor([w['a'] for w in worlds]); y_prin = torch.tensor([w['pi'] for w in worlds])
    held = torch.tensor([w['held'] for w in worlds]); tr = ~held

    @torch.no_grad()
    def qwen_gen(prompt):
        ids = tok(H.tmpl([{'role': 'user', 'content': prompt}]), return_tensors='pt').input_ids.to(dev)
        out = model.generate(ids, max_new_tokens=6, do_sample=False, pad_token_id=tok.eos_token_id)
        txt = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).upper()
        return next((j for j, a in enumerate(ACTIONS) if a in txt), -1)

    heldw = [w for w in worlds if w['held']]
    qc = sum(int(qwen_gen(w['prompt']) == w['a']) for w in heldw[:24])
    print('GEOMN QWEN-GEN ceiling on held %s = %.3f (n=24, in-window)' % (HELDPRIN, qc / 24.0), flush=True)

    Hpool = torch.stack([w['H'].float().mean(0) for w in worlds])

    def readout(Xtr, ytr, Xte, yte, nc=6, epochs=500):
        mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
        Xtr, Xte = ((Xtr - mu) / sd).to(dev), ((Xte - mu) / sd).to(dev); ytr, yte = ytr.to(dev), yte.to(dev)
        net = nn.Linear(Xtr.shape[1], nc).to(dev); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
        net.train()
        for e in range(epochs): opt.zero_grad(); F.cross_entropy(net(Xtr), ytr).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            return float((net(Xte).argmax(1) == yte).float().mean())

    hi = [i for i in range(len(worlds)) if worlds[i]['held']]
    random.Random(SEED).shuffle(hi); half = len(hi) // 2
    acc_inpr = readout(Hpool[hi[:half]], y_act[hi[:half]], Hpool[hi[half:]], y_act[hi[half:]])
    acc_xpr = readout(Hpool[tr], y_act[tr], Hpool[held], y_act[held])
    print('GEOMN rawH action: in-principle(%s) ceiling=%.3f | cross-principle(train->held) transfer=%.3f'
          % (HELDPRIN, acc_inpr, acc_xpr), flush=True)

    g = AdaptiveGateSlot(D_MODEL, D_S, K, SLOW_K).to(dev); head = nn.Linear(D_S, 6).to(dev)
    opt = torch.optim.Adam(list(g.parameters()) + list(head.parameters()), lr=LR)

    def Sof(w): return g.step(g.init(), w['H'].float().to(dev))

    TRAINi = [i for i in range(len(worlds)) if not worlds[i]['held']]

    def evaln(it):
        g.eval(); head.eval()
        with torch.no_grad():
            Z = torch.stack([Sof(worlds[i]).mean(0).cpu() for i in range(len(worlds))])
            logits = torch.stack([head(Sof(worlds[i]).mean(0)) for i in range(len(worlds))]).cpu()
        pred = logits.argmax(1)
        btr = float((pred[tr] == y_act[tr]).float().mean()); bho = float((pred[held] == y_act[held]).float().mean())
        idx = list(range(len(worlds))); random.Random(SEED).shuffle(idx); h = len(idx) // 2
        pp = readout(Z[idx[:h]], y_prin[idx[:h]], Z[idx[h:]], y_prin[idx[h:]], nc=3, epochs=400)
        print('GEOMN it=%-4d | behavior train=%.3f HELD-PRIN=%.3f | S 3way-principle-probe(incl held)=%.3f'
              % (it, btr, bho, pp), flush=True)
        g.train(); head.train()

    evaln(0)
    rng2 = random.Random(SEED + 1)
    for it in range(1, ITERS + 1):
        w = worlds[TRAINi[rng2.randrange(len(TRAINi))]]
        S = Sof(w); logit = head(S.mean(0))
        loss = F.cross_entropy(logit.unsqueeze(0), torch.tensor([w['a']], device=dev))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(g.parameters()) + list(head.parameters()), 1.0); opt.step()
        if it % EVERY == 0: evaln(it)
    print('=== GEOM_NOVEL_DONE ===', flush=True)


'''
src = io.open(PATH, encoding='utf-8').read()
if 'def geom_novel()' in src:
    print('ALREADY'); raise SystemExit
anchor = "if MODE == 'validate':   validate()"
src = src.replace(anchor, FN + '\n\n' + anchor, 1)
src = src.replace("elif MODE == 'geom_phase': geom_phase()",
                  "elif MODE == 'geom_phase': geom_phase()\nelif MODE == 'geom_novel': geom_novel()", 1)
io.open(PATH, 'w', encoding='utf-8').write(src)
print('PATCHED_OK')
