import io
PATH = '/home/pokazge/NativeEntity/world_pop.py'
FN = r'''def carry_bind():
    # CARRY_BIND_APPLY_ONLY_V1: can the substrate do anything beyond resurfacing stored explicit content?
    # Ladder L1..L5; above L1 the stored item is NEVER the answer — answer requires binding carried state
    # to later in-window context. Per level: OFF / ON / ON-resetS / ON-wrongS / RAG / ORACLE + probes.
    LAYERS = [int(x) for x in os.environ.get('GEO_LAYERS', '4,8,12,16,20,24,28,32,36,40,44,48,52,56,60').split(',')]
    LV = int(os.environ.get('GEO_LEVEL', '1'))
    D = int(os.environ.get('GEO_ACT_D', '6')); NW = int(os.environ.get('GEO_NW', '60'))
    ITERS = int(os.environ.get('GEO_ITERS', '600')); EVERY = int(os.environ.get('GEO_EVAL_EVERY', '300'))
    LR = float(os.environ.get('GEO_LR', '1e-4')); ACT_EPS = float(os.environ.get('GEO_ACT_EPS', '0.1'))
    print('=== CARRY_BIND L%d | D=%d NW=%d iters=%d eps=%.2f ===' % (LV, D, NW, ITERS, ACT_EPS), flush=True)
    rng = random.Random(SEED)
    ACT4 = ['KEEP', 'REJECT', 'DEFER', 'ASK']
    FILL_U, FILL_A = 'Routine status check, no action needed.', 'Acknowledged. Standing by.'

    def mkworld():
        # returns (commit_text, decision_text, correct_action_word, carried_var_id, oracle_needs_commit=True)
        if LV == 1:
            a = ACTIONS[rng.randrange(6)]
            commit = 'STANDING RULING for this session: the authorized action is %s. Retain it.' % a
            dec = 'State the standing authorized action for this session. %s' % ASK_INSTR
            return commit, dec, a, ACTIONS.index(a)
        if LV == 2:
            K = LATENTS[rng.randrange(4)]
            acts = ACT4[:]; rng.shuffle(acts); mp = dict(zip(LATENTS, acts))
            commit = 'STANDING KEY for this session: %s %s %s. Retain it.' % (K, K, K)
            dec = ('RULING REQUIRED. Keyline: %s. The ruling is the one listed for the standing session key. %s'
                   % (', '.join('%s->%s' % (t, mp[t]) for t in LATENTS), ASK_INSTR))
            return commit, dec, mp[K], LATENTS.index(K)
        if LV == 3:
            K = LATENTS[rng.randrange(4)]
            order = LATENTS[:]; rng.shuffle(order)
            acts = ACT4[:]; rng.shuffle(acts); mp = dict(zip(LATENTS, acts))
            succ = order[(order.index(K) + 1) % 4]
            commit = 'STANDING KEY for this session: %s %s %s. Retain it.' % (K, K, K)
            dec = ('RULING REQUIRED. Items in order: %s. Keyline: %s. The ruling is the one listed for the item '
                   'that comes IMMEDIATELY AFTER the standing session key in the order (wrapping around). %s'
                   % (', '.join(order), ', '.join('%s->%s' % (t, mp[t]) for t in order), ASK_INSTR))
            return commit, dec, mp[succ], LATENTS.index(K)
        if LV == 4:
            rid = rng.randrange(4)
            rtxt = ['the item with the HIGHEST score', 'the item with the LOWEST score',
                    'the FIRST item listed', 'the LAST item listed'][rid]
            commit = 'STANDING RULE for this session: the authorized item is %s. Retain it.' % rtxt
            items = rng.sample(LATENTS, 3)
            scores = rng.sample(range(10, 99), 3)
            acts = rng.sample(ACTIONS, 3)
            if rid == 0: w = max(range(3), key=lambda i: scores[i])
            elif rid == 1: w = min(range(3), key=lambda i: scores[i])
            elif rid == 2: w = 0
            else: w = 2
            dec = ('EVALUATION. Items: %s. Apply the standing session rule and reply with the ruling of the '
                   'authorized item. %s' % (', '.join('%s (score %d) -> %s' % (items[i], scores[i], acts[i])
                                                      for i in range(3)), ASK_INSTR))
            return commit, dec, acts[w], rid
        if LV == 5:
            did = rng.randrange(3)
            dt = ['EQUALITY doctrine: an archive submission is accepted only when it exactly equals the designated entry.',
                  'ORDER doctrine: an archive submission is accepted only when it is at or before the designated entry in the sequence.',
                  'EXCLUSION doctrine: every archive submission is accepted except the designated entry itself.'][did]
            commit = 'SESSION DOCTRINE (archive intake): %s Retain it.' % dt
            order = LATENTS[:]; rng.shuffle(order)
            X = LATENTS[rng.randrange(4)]; Y = LATENTS[rng.randrange(4)]
            if did == 0: honored = (Y == X)
            elif did == 1: honored = (order.index(Y) <= order.index(X))
            else: honored = (Y != X)
            dec = ('GATEWAY CLEARANCE request. Sequence: %s. Designated entry: %s. Requesting entry: %s. '
                   'Under the standing session doctrine, is the request cleared? Reply KEEP if cleared, '
                   'REJECT if not. %s' % (', '.join(order), X, Y, ASK_INSTR))
            return commit, dec, ('KEEP' if honored else 'REJECT'), did
        raise ValueError(LV)

    @torch.no_grad()
    def turn_stack(hist):
        _fb['on'] = False
        ids = tok(H.tmpl(hist[-WINDOW:]), return_tensors='pt').input_ids.to(dev)
        hs = model(ids, output_hidden_states=True).hidden_states
        return torch.stack([hs[L][0][-1].float() for L in LAYERS]).to(torch.float16).cpu()

    samples = []
    nleak = 0
    print('CBA building %d worlds ...' % NW, flush=True)
    for wi in range(NW):
        commit, dec, ans, vid = mkworld()
        leak = ans.upper() in commit.upper()
        nleak += int(leak)
        hist = [{'role': 'user', 'content': commit}, {'role': 'assistant', 'content': 'Acknowledged.'}]
        stacks = [turn_stack(hist)]
        for _ in range(D):
            hist += [{'role': 'user', 'content': FILL_U}, {'role': 'assistant', 'content': FILL_A}]
            stacks.append(turn_stack(hist))
        hist += [{'role': 'user', 'content': dec}]
        pids = tok(H.tmpl(hist[-WINDOW:]), return_tensors='pt').input_ids[0].to(dev)
        rag_hist = hist[:-1] + [{'role': 'user', 'content': 'Session note (retrieved from memory): %s\n\n%s' % (commit, dec)}]
        rag_pids = tok(H.tmpl(rag_hist[-WINDOW:]), return_tensors='pt').input_ids[0].to(dev)
        ora_hist = [{'role': 'user', 'content': commit}, {'role': 'assistant', 'content': 'Acknowledged.'},
                    {'role': 'user', 'content': dec}]
        ora_pids = tok(H.tmpl(ora_hist), return_tensors='pt').input_ids[0].to(dev)
        stacks.append(turn_stack(hist))
        samples.append({'stacks': stacks, 'pids': pids, 'rag': rag_pids, 'ora': ora_pids,
                        'aids': tok(' ' + ans, add_special_tokens=False).input_ids,
                        'cidx': ACTIONS.index(ans), 'vid': vid})
        if (wi + 1) % 20 == 0: print('  %d/%d' % (wi + 1, NW), flush=True)
    import collections as _cl
    base = max(_cl.Counter([s['cidx'] for s in samples]).values()) / float(len(samples))
    print('CBA L%d leakage=%d/%d (expect %s) base-rate=%.3f' % (LV, nleak, NW, 'NW' if LV == 1 else '0', base), flush=True)

    r = random.Random(SEED)
    for s in samples: s['test'] = (r.random() < 0.3)
    TR = [s for s in samples if not s['test']]; TE = [s for s in samples if s['test']]
    print('CBA train=%d test=%d' % (len(TR), len(TE)), flush=True)

    _fb['fields'] = {L: SL.AlwaysOnSlotField(D_MODEL, D_S, eps=ACT_EPS).to(dev) for L in FIELD_LAYERS}
    fp = []
    for L in FIELD_LAYERS:
        for p in _fb['fields'][L].parameters(): p.requires_grad_(True); fp.append(p)
    g = AdaptiveGateSlot(D_MODEL, D_S, K, SLOW_K).to(dev)
    opt = torch.optim.Adam(list(g.parameters()) + fp, lr=LR)

    def Sfrom(stks):
        S = g.init()
        for st in stks: S = g.step(S, st.float().to(dev))
        return S

    @torch.no_grad()
    def gen_arm(group, mode):
        if not group: return 0.0
        c = 0
        oi = random.Random(SEED + 7)
        for s in group:
            pids = s['pids']
            if mode == 'off': _fb['on'] = False
            elif mode == 'rag': _fb['on'] = False; pids = s['rag']
            elif mode == 'ora': _fb['on'] = False; pids = s['ora']
            else:
                if mode == 'on': stks = s['stacks']
                elif mode == 'reset': stks = s['stacks'][1:]
                else:
                    other = samples[oi.randrange(len(samples))]
                    stks = [other['stacks'][0]] + s['stacks'][1:]
                _fb['S'] = Sfrom(stks); _fb['on'] = True
                for L in FIELD_LAYERS: _fb['fields'][L].eps = ACT_EPS
            out = model.generate(pids.unsqueeze(0), max_new_tokens=6, do_sample=False, pad_token_id=tok.eos_token_id)
            _fb['on'] = False
            txt = tok.decode(out[0, pids.shape[0]:], skip_special_tokens=True).upper()
            ai = next((j for j, a in enumerate(ACTIONS) if a in txt), -1)
            c += int(ai == s['cidx'])
        return c / len(group)

    def report(it):
        g.eval()
        for L in FIELD_LAYERS: _fb['fields'][L].eval()
        vals = {m: gen_arm(TE, m) for m in ['off', 'on', 'reset', 'wrong', 'rag', 'ora']}
        fit = gen_arm(TR[:12], 'on')
        print('CBA L%d it=%-4d | OFF=%.3f ON=%.3f ONreset=%.3f ONwrong=%.3f RAG=%.3f ORACLE=%.3f | fitON=%.3f'
              % (LV, it, vals['off'], vals['on'], vals['reset'], vals['wrong'], vals['rag'], vals['ora'], fit), flush=True)
        g.train()
        for L in FIELD_LAYERS: _fb['fields'][L].train()

    report(0)
    rng2 = random.Random(SEED + 1)
    for it in range(1, ITERS + 1):
        s = TR[rng2.randrange(len(TR))]
        _fb['S'] = Sfrom(s['stacks']); _fb['on'] = True
        for L in FIELD_LAYERS: _fb['fields'][L].eps = ACT_EPS
        seq = torch.cat([s['pids'], torch.tensor(s['aids'], device=dev)]).unsqueeze(0)
        logits = model(seq).logits[0]
        _fb['on'] = False
        pl = s['pids'].shape[0]
        lp = torch.log_softmax(logits[pl - 1:pl - 1 + len(s['aids'])], -1)
        nll = -lp[range(len(s['aids'])), torch.tensor(s['aids'], device=dev)].mean()
        opt.zero_grad(); nll.backward()
        torch.nn.utils.clip_grad_norm_(list(g.parameters()) + fp, 1.0); opt.step()
        if it % EVERY == 0:
            print('CBA L%d it=%d nll=%.4f' % (LV, it, float(nll)), flush=True); report(it)

    # ── ARCHITECTURE PROBES ──
    g.eval()
    with torch.no_grad():
        Sp = torch.stack([Sfrom(s['stacks']).mean(0).cpu() for s in samples])          # [N, d_s]
        Dh = torch.stack([s['stacks'][-1][-1].float() for s in samples])               # decision turn, top layer
    yv = torch.tensor([s['vid'] for s in samples]); ya = torch.tensor([s['cidx'] for s in samples])
    tem = torch.tensor([s['test'] for s in samples])

    def probe(X, yy, nc, mlp=False, epochs=500):
        Xtr, ytr, Xte, yte = X[~tem], yy[~tem], X[tem], yy[tem]
        mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
        Xtr, Xte = ((Xtr - mu) / sd).to(dev), ((Xte - mu) / sd).to(dev); ytr, yte = ytr.to(dev), yte.to(dev)
        d = Xtr.shape[1]
        net = (nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, nc))
               if mlp else nn.Linear(d, nc)).to(dev)
        o = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3); net.train()
        for e in range(epochs): o.zero_grad(); F.cross_entropy(net(Xtr), ytr).backward(); o.step()
        net.eval()
        with torch.no_grad():
            return (float((net(Xtr).argmax(1) == ytr).float().mean()),
                    float((net(Xte).argmax(1) == yte).float().mean()))
    nvar = int(yv.max()) + 1
    v_tr, v_te = probe(Sp, yv, nvar)
    m_tr, m_te = probe(torch.cat([Sp, Dh], 1), ya, 6, mlp=True)
    print('CBA L%d PROBES | S->carriedVar lin tr=%.3f TE=%.3f (chance %.3f) | MLP[S,decHid]->action tr=%.3f TE=%.3f (base %.3f)'
          % (LV, v_tr, v_te, 1.0 / nvar, m_tr, m_te, base), flush=True)
    print('=== CBA_L%d_DONE ===' % LV, flush=True)


'''
src = io.open(PATH, encoding='utf-8').read()
if 'def carry_bind()' in src:
    print('ALREADY'); raise SystemExit
anchor = "if MODE == 'validate':   validate()"
src = src.replace(anchor, FN + '\n\n' + anchor, 1)
src = src.replace("elif MODE == 'gen_actuate': gen_actuate()",
                  "elif MODE == 'gen_actuate': gen_actuate()\nelif MODE == 'carry_bind': carry_bind()", 1)
io.open(PATH, 'w', encoding='utf-8').write(src)
print('PATCHED_OK')
