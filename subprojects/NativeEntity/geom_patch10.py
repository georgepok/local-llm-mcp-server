import io
PATH = '/home/pokazge/NativeEntity/world_pop.py'
FN = r'''def gen_both():
    # CONVERGENCE: ONE generic stack-reading substrate trained on INTERLEAVED memory + inference worlds.
    # Memory = carry explicit action out-of-window; Inference = crossed rules, hold out LOOKUP principle.
    LAYERS = [int(x) for x in os.environ.get('GEO_LAYERS', '4,8,12,16,20,24,28,32,36,40,44,48,52,56,60').split(',')]
    HELDPRIN = os.environ.get('GEO_HELD_PRIN', 'LOOKUP')
    DISTS = [int(x) for x in os.environ.get('GEO_DISTS', '0,4,8').split(',')]
    M_INF = int(os.environ.get('GEO_M', '8')); NW_MEM = int(os.environ.get('GEO_NW', '16'))
    ITERS = int(os.environ.get('GEO_ITERS', '3000')); EVERY = int(os.environ.get('GEO_EVAL_EVERY', '500'))
    LR = float(os.environ.get('GEO_LR', '1e-4'))
    PRINCIPLES = ['MATCH', 'LOOKUP', 'THRESHOLD']
    SURFACES = ['ledger', 'vault', 'archive', 'pipeline', 'roster', 'gauge']
    print('=== GEN_BOTH | interleaved memory+inference | held-principle=%s dists=%s ===' % (HELDPRIN, DISTS), flush=True)
    rng = random.Random(SEED)
    FILL_U, FILL_A = 'Routine status check, no action needed.', 'Acknowledged. Standing by.'

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

    @torch.no_grad()
    def turn_stack(hist):
        ids = tok(H.tmpl(hist[-WINDOW:]), return_tensors='pt').input_ids.to(dev)
        hs = model(ids, output_hidden_states=True).hidden_states
        return torch.stack([hs[L][0][-1].float() for L in LAYERS]).to(torch.float16).cpu()

    worlds = []
    print('GENB building worlds ...', flush=True)
    for pi, prin in enumerate(PRINCIPLES):
        for surf in SURFACES:
            for _ in range(M_INF):
                p, a = gen_lex(prin, surf)
                worlds.append({'kind': 'inf', 'heldprin': (prin == HELDPRIN), 'act': a,
                               'stacks': [turn_stack([{'role': 'user', 'content': p}])]})
    for wi in range(NW_MEM):
        for d in DISTS:
            act = ACTIONS[rng.randrange(6)]
            commit = 'STANDING RULING for this session: the authorized action is %s. Retain it.' % act
            hist = [{'role': 'user', 'content': commit}, {'role': 'assistant', 'content': 'Acknowledged.'}]
            stacks = [turn_stack(hist)]
            for _ in range(d):
                hist += [{'role': 'user', 'content': FILL_U}, {'role': 'assistant', 'content': FILL_A}]
                stacks.append(turn_stack(hist))
            hist += [{'role': 'user', 'content': 'What is the standing authorized action for this session? %s' % ASK_INSTR}]
            stacks.append(turn_stack(hist))
            worlds.append({'kind': 'mem', 'd': d, 'act': ACTIONS.index(act), 'stacks': stacks})

    r = random.Random(SEED)
    for w in worlds:
        w['test'] = (r.random() < 0.3)                       # 30% held-out for eval
    # trainable = inf(train-principle, train-split) + mem(train-split)
    TRAINi = [i for i, w in enumerate(worlds)
              if not w['test'] and not (w['kind'] == 'inf' and w['heldprin'])]

    g = AdaptiveGateSlot(D_MODEL, D_S, K, SLOW_K).to(dev); head = nn.Linear(D_S, 6).to(dev)
    opt = torch.optim.Adam(list(g.parameters()) + list(head.parameters()), lr=LR)

    def Sof(w):
        S = g.init()
        for st in w['stacks']:
            S = g.step(S, st.float().to(dev))
        return S

    y = torch.tensor([w['act'] for w in worlds])

    def ev(it):
        g.eval(); head.eval()
        with torch.no_grad():
            pred = torch.tensor([int(head(Sof(worlds[i]).mean(0)).argmax()) for i in range(len(worlds))])
        def acc(sel):
            sel = [i for i in range(len(worlds)) if sel(worlds[i])]
            return (float(sum(int(pred[i] == y[i]) for i in sel) / max(1, len(sel))), len(sel))
        inf_fit, _ = acc(lambda w: w['kind'] == 'inf' and not w['heldprin'] and w['test'])
        inf_tr, _ = acc(lambda w: w['kind'] == 'inf' and w['heldprin'])
        memd = {d: acc(lambda w, d=d: w['kind'] == 'mem' and w['d'] == d and w['test'])[0] for d in DISTS}
        print('GENB it=%-4d | INFER train-prin-fit=%.3f held-LOOKUP=%.3f | MEM %s'
              % (it, inf_fit, inf_tr, ' '.join('d%d:%.2f' % (d, memd[d]) for d in DISTS)), flush=True)
        g.train(); head.train()

    ev(0)
    rng2 = random.Random(SEED + 1)
    for it in range(1, ITERS + 1):
        i = TRAINi[rng2.randrange(len(TRAINi))]
        S = Sof(worlds[i]); loss = F.cross_entropy(head(S.mean(0)).unsqueeze(0), torch.tensor([worlds[i]['act']], device=dev))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(g.parameters()) + list(head.parameters()), 1.0); opt.step()
        if it % EVERY == 0: ev(it)
    print('=== GEN_BOTH_DONE ===', flush=True)


'''
src = io.open(PATH, encoding='utf-8').read()
if 'def gen_both()' in src:
    print('ALREADY'); raise SystemExit
anchor = "if MODE == 'validate':   validate()"
src = src.replace(anchor, FN + '\n\n' + anchor, 1)
src = src.replace("elif MODE == 'gen_memory': gen_memory()",
                  "elif MODE == 'gen_memory': gen_memory()\nelif MODE == 'gen_both': gen_both()", 1)
io.open(PATH, 'w', encoding='utf-8').write(src)
print('PATCHED_OK')
