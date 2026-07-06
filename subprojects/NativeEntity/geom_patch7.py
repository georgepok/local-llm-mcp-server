import io
PATH = '/home/pokazge/NativeEntity/world_pop.py'
FN = r'''def geom_generic():
    # GENERIC-DEPTH substrate: g reads the LLM LAYER STACK [n_layers, D] (per-layer pooled hidden) and
    # its cross-attention LEARNS which depth to use -- NO hand-picked layer, architecture-agnostic.
    # Head-to-head vs fixed-layer-32 baseline on the novel-principle holdout.
    LAYERS = [int(x) for x in os.environ.get('GEO_LAYERS', '4,8,12,16,20,24,28,32,36,40,44,48,52,56,60').split(',')]
    BASEL = int(os.environ.get('GEO_BASE_LAYER', str(READ_LAYER)))
    HELDPRIN = os.environ.get('GEO_HELD_PRIN', 'LOOKUP')
    M = int(os.environ.get('GEO_M', '12')); ITERS = int(os.environ.get('GEO_ITERS', '2500'))
    EVERY = int(os.environ.get('GEO_EVAL_EVERY', '500')); LR = float(os.environ.get('GEO_LR', '1e-4'))
    ALLPRIN = ['MATCH', 'LOOKUP', 'THRESHOLD']
    SURFACES = ['ledger', 'vault', 'archive', 'pipeline', 'roster', 'gauge']
    print('=== GEOM_GENERIC | held=%s stack=%d layers base=L%d ===' % (HELDPRIN, len(LAYERS), BASEL), flush=True)
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
    def getstack(prompt):
        ids = tok(H.tmpl([{'role': 'user', 'content': prompt}]), return_tensors='pt').input_ids.to(dev)
        hs = model(ids, output_hidden_states=True).hidden_states
        st = torch.stack([hs[L][0].float().mean(0) for L in LAYERS])   # [n_layers, D]
        return st.to(torch.float16).cpu()

    print('GEOMG collecting %d ...' % len(worlds), flush=True)
    for i, w in enumerate(worlds):
        w['stack'] = getstack(w['prompt'])
        if (i + 1) % 72 == 0: print('  %d/%d' % (i + 1, len(worlds)), flush=True)
    li = LAYERS.index(BASEL) if BASEL in LAYERS else min(range(len(LAYERS)), key=lambda k: abs(LAYERS[k] - BASEL))

    y_act = torch.tensor([w['a'] for w in worlds]); held = torch.tensor([w['held'] for w in worlds]); tr = ~held
    import collections as _cl
    _hb = _cl.Counter([worlds[i]['a'] for i in range(len(worlds)) if worlds[i]['held']])
    base = max(_hb.values()) / float(sum(_hb.values()))
    print('GEOMG held base-rate = %.3f' % base, flush=True)

    def run(tag, stack_input):
        g = AdaptiveGateSlot(D_MODEL, D_S, K, SLOW_K).to(dev); head = nn.Linear(D_S, 6).to(dev)
        opt = torch.optim.Adam(list(g.parameters()) + list(head.parameters()), lr=LR)

        def Sof(i):
            x = worlds[i]['stack'].float().to(dev)          # [n_layers, D]
            xin = x if stack_input else x[li:li + 1]        # generic: full stack ; baseline: single layer
            return g.step(g.init(), xin)

        TRAINi = [i for i in range(len(worlds)) if not worlds[i]['held']]
        rng2 = random.Random(SEED + 1)

        def ev(it):
            g.eval(); head.eval()
            with torch.no_grad():
                logits = torch.stack([head(Sof(i).mean(0)) for i in range(len(worlds))]).cpu()
            pred = logits.argmax(1)
            btr = float((pred[tr] == y_act[tr]).float().mean()); bho = float((pred[held] == y_act[held]).float().mean())
            print('GEOMG [%s] it=%-4d behavior train=%.3f HELD-PRIN=%.3f (base %.3f)' % (tag, it, btr, bho, base), flush=True)
            g.train(); head.train(); return bho

        ev(0)
        for it in range(1, ITERS + 1):
            i = TRAINi[rng2.randrange(len(TRAINi))]
            S = Sof(i); loss = F.cross_entropy(head(S.mean(0)).unsqueeze(0), torch.tensor([worlds[i]['a']], device=dev))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(g.parameters()) + list(head.parameters()), 1.0); opt.step()
            if it % EVERY == 0: ev(it)
        return ev(ITERS)

    b_base = run('fixed-L%d' % BASEL, stack_input=False)
    b_gen = run('generic-stack', stack_input=True)
    print('GEOMG SUMMARY held=%s: fixed-L%d HELD=%.3f | generic-stack HELD=%.3f | base=%.3f'
          % (HELDPRIN, BASEL, b_base, b_gen, base), flush=True)
    print('=== GEOM_GENERIC_DONE ===', flush=True)


'''
src = io.open(PATH, encoding='utf-8').read()
if 'def geom_generic()' in src:
    print('ALREADY'); raise SystemExit
anchor = "if MODE == 'validate':   validate()"
src = src.replace(anchor, FN + '\n\n' + anchor, 1)
src = src.replace("elif MODE == 'geom_meta': geom_meta()",
                  "elif MODE == 'geom_meta': geom_meta()\nelif MODE == 'geom_generic': geom_generic()", 1)
io.open(PATH, 'w', encoding='utf-8').write(src)
print('PATCHED_OK')
