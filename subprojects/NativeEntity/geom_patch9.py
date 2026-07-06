import io
PATH = '/home/pokazge/NativeEntity/world_pop.py'
FN = r'''def gen_memory():
    # GENERIC-STACK MEMORY: does a substrate reading the LLM layer STACK RETAIN out-of-window carry?
    # Pure-carry task: commit states an explicit ACTION; recall it at distance d (commit leaves window).
    # 3 arms: generic-stack g / fixed-layer g / LLM-alone readout. Behavior vs d.
    LAYERS = [int(x) for x in os.environ.get('GEO_LAYERS', '4,8,12,16,20,24,28,32,36,40,44,48,52,56,60').split(',')]
    BASEL = int(os.environ.get('GEO_BASE_LAYER', str(READ_LAYER)))
    DISTS = [int(x) for x in os.environ.get('GEO_DISTS', '0,2,4,8').split(',')]
    NW = int(os.environ.get('GEO_NW', '20')); ITERS = int(os.environ.get('GEO_ITERS', '2000'))
    EVERY = int(os.environ.get('GEO_EVAL_EVERY', '500')); LR = float(os.environ.get('GEO_LR', '1e-4'))
    print('=== GEN_MEMORY | carry-action across d | stack vs fixed-L' + str(BASEL) + ' vs LLM-alone ===', flush=True)
    rng = random.Random(SEED)
    li = LAYERS.index(BASEL) if BASEL in LAYERS else min(range(len(LAYERS)), key=lambda k: abs(LAYERS[k] - BASEL))
    FILL_U, FILL_A = 'Routine status check, no action needed.', 'Acknowledged. Standing by.'

    @torch.no_grad()
    def turn_stack(hist):
        ids = tok(H.tmpl(hist[-WINDOW:]), return_tensors='pt').input_ids.to(dev)
        hs = model(ids, output_hidden_states=True).hidden_states
        return torch.stack([hs[L][0][-1].float() for L in LAYERS]).to(torch.float16).cpu()

    samples = []
    print('GENM building %d worlds ...' % (NW * len(DISTS)), flush=True)
    for wi in range(NW):
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
            samples.append({'stacks': stacks, 'act': ACTIONS.index(act), 'd': d})
    y = torch.tensor([s['act'] for s in samples]); ds = torch.tensor([s['d'] for s in samples])
    idx = list(range(len(samples))); random.Random(SEED).shuffle(idx)
    ntr = int(0.7 * len(idx)); TR = set(idx[:ntr]); trm = torch.tensor([i in TR for i in range(len(samples))])

    def readout(Xtr, ytr, Xte, yte, epochs=500):
        mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
        Xtr, Xte = ((Xtr - mu) / sd).to(dev), ((Xte - mu) / sd).to(dev); ytr, yte = ytr.to(dev), yte.to(dev)
        net = nn.Linear(Xtr.shape[1], 6).to(dev); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
        net.train()
        for e in range(epochs): opt.zero_grad(); F.cross_entropy(net(Xtr), ytr).backward(); opt.step()
        net.eval()
        with torch.no_grad(): return float((net(Xte).argmax(1) == yte).float().mean())

    Dec = torch.stack([s['stacks'][-1][li].float() for s in samples])
    llm = {}
    for d in DISTS:
        m = (ds == d)
        llm[d] = readout(Dec[m & trm], y[m & trm], Dec[m & ~trm], y[m & ~trm])

    def run(tag, use_stack):
        g = AdaptiveGateSlot(D_MODEL, D_S, K, SLOW_K).to(dev); head = nn.Linear(D_S, 6).to(dev)
        opt = torch.optim.Adam(list(g.parameters()) + list(head.parameters()), lr=LR)

        def Sof(s):
            S = g.init()
            for st in s['stacks']:
                x = st.float().to(dev)
                S = g.step(S, x if use_stack else x[li:li + 1])
            return S

        TRAINi = [i for i in range(len(samples)) if trm[i]]
        rng2 = random.Random(SEED + 1)

        def ev(it):
            g.eval(); head.eval()
            with torch.no_grad():
                pred = torch.tensor([int(head(Sof(samples[i]).mean(0)).argmax()) for i in range(len(samples))])
            per = ' '.join('%d:%.2f' % (d, float((pred[(ds == d) & ~trm] == y[(ds == d) & ~trm]).float().mean())) for d in DISTS)
            print('GENM [%s] it=%-4d test-by-d %s' % (tag, it, per), flush=True)
            g.train(); head.train()
            return {d: float((pred[(ds == d) & ~trm] == y[(ds == d) & ~trm]).float().mean()) for d in DISTS}

        ev(0)
        for it in range(1, ITERS + 1):
            i = TRAINi[rng2.randrange(len(TRAINi))]
            S = Sof(samples[i])
            loss = F.cross_entropy(head(S.mean(0)).unsqueeze(0), torch.tensor([samples[i]['act']], device=dev))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(g.parameters()) + list(head.parameters()), 1.0); opt.step()
            if it % EVERY == 0: ev(it)
        return ev(ITERS)

    r_fix = run('fixed', use_stack=False)
    r_gen = run('generic', use_stack=True)
    print('GENM SUMMARY test-acc by distance (commit leaves window ~d>2):', flush=True)
    print('GENM   arm        ' + '  '.join('d=%d' % d for d in DISTS), flush=True)
    print('GENM   LLM-alone   ' + '  '.join('%.2f' % llm[d] for d in DISTS), flush=True)
    print('GENM   fixed-L' + str(BASEL) + '   ' + '  '.join('%.2f' % r_fix[d] for d in DISTS), flush=True)
    print('GENM   generic     ' + '  '.join('%.2f' % r_gen[d] for d in DISTS), flush=True)
    print('=== GEN_MEMORY_DONE ===', flush=True)


'''
src = io.open(PATH, encoding='utf-8').read()
if 'def gen_memory()' in src:
    print('ALREADY'); raise SystemExit
anchor = "if MODE == 'validate':   validate()"
src = src.replace(anchor, FN + '\n\n' + anchor, 1)
src = src.replace("elif MODE == 'geom_generic': geom_generic()",
                  "elif MODE == 'geom_generic': geom_generic()\nelif MODE == 'gen_memory': gen_memory()", 1)
io.open(PATH, 'w', encoding='utf-8').write(src)
print('PATCHED_OK')
