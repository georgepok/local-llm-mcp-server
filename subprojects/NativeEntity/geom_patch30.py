import io
PATH = '/home/pokazge/NativeEntity/world_pop.py'
FN = r'''def carry_kv15():
    # ABSTRACT-BINDING test: scale the relational task to NKEYS keys x NACTS actions (NACTS! tables) so
    # held-out worlds have NOVEL tables -> memorization/interpolation CANNOT generalize; only a real
    # key-in-S x table-in-prompt LOOKUP can. Field (SL.AlwaysOnSlotField) + accC/accW content control.
    #   accC>>accW on NOVEL tables -> ABSTRACT relational binding (rung-3). accC~=accW -> small-space only.
    LAYERS = [int(x) for x in os.environ.get('GEO_LAYERS', '4,8,12,16,20,24,28,32,36,40,44,48,52,56,60').split(',')]
    D = int(os.environ.get('GEO_ACT_D', '6')); NW = int(os.environ.get('GEO_NW', '80'))
    ITERS = int(os.environ.get('GEO_ITERS', '1000')); EVERY = int(os.environ.get('GEO_EVAL_EVERY', '500'))
    LRg = float(os.environ.get('GEO_LR', '1e-4')); FLR = float(os.environ.get('GEO_FLR', '1e-4'))
    EPSF = float(os.environ.get('GEO_FIELD_EPS', '0.1'))
    NKEYS = int(os.environ.get('GEO_NKEYS', '8')); NACTS = int(os.environ.get('GEO_NACTS', '8'))
    KEYPOOL = ['FOXTROT', 'KILO', 'NOVEMBER', 'SIERRA', 'TANGO', 'ZULU', 'ALPHA', 'DELTA', 'ROMEO', 'VICTOR', 'BRAVO', 'ECHO']
    ACTPOOL = ['KEEP', 'REJECT', 'DEFER', 'ASK', 'PURGE', 'FLAG', 'HOLD', 'DROP', 'ROUTE', 'MERGE']
    KEYS = KEYPOOL[:NKEYS]; ACTS = ACTPOOL[:NACTS]
    print('=== CARRY_KV15(ABSTRACT-BINDING, %d keys x %d acts, %d! tables) eps=%.2f ===' % (NKEYS, NACTS, NACTS, EPSF), flush=True)
    print('CKV15 act first-tokens: %s' % {a: tok(' ' + a, add_special_tokens=False).input_ids[0] for a in ACTS}, flush=True)
    rng = random.Random(SEED)
    FILL_U, FILL_A = 'Routine status check, no action needed.', 'Acknowledged. Standing by.'
    _fb['fields'] = {L: SL.AlwaysOnSlotField(D_MODEL, D_S, eps=EPSF).to(dev) for L in FIELD_LAYERS}
    _fb['on'] = False
    field_params = [p for L in FIELD_LAYERS for p in _fb['fields'][L].parameters()]

    def mkworld():
        keys = KEYS[:]; acts = ACTS[:]; rng.shuffle(acts); mp = dict(zip(keys, acts))    # novel table per world
        Kk = keys[rng.randrange(NKEYS)]
        table = ', '.join('%s->%s' % (t, mp[t]) for t in keys)
        return ('STANDING KEY for this session: %s %s %s. Retain it.' % (Kk, Kk, Kk),
                'RULING REQUIRED. Keyline: %s. The ruling is the one listed for the standing session key. %s'
                % (table, ASK_INSTR), mp[Kk])

    @torch.no_grad()
    def turn_stack(hist):
        _fb['on'] = False
        ids = tok(H.tmpl(hist[-WINDOW:]), return_tensors='pt').input_ids.to(dev)
        hs = model(ids, output_hidden_states=True).hidden_states
        return torch.stack([hs[L][0][-1].float() for L in LAYERS]).to(torch.float16).cpu()

    samples = []
    print('CKV15 building %d worlds ...' % NW, flush=True)
    for wi in range(NW):
        commit, dec, ans = mkworld()
        hist = [{'role': 'user', 'content': commit}, {'role': 'assistant', 'content': 'Acknowledged.'}]
        stacks = [turn_stack(hist)]
        for _ in range(D):
            hist += [{'role': 'user', 'content': FILL_U}, {'role': 'assistant', 'content': FILL_A}]
            stacks.append(turn_stack(hist))
        clean = hist + [{'role': 'user', 'content': dec}]
        pids = tok(H.tmpl(clean[-WINDOW:]), return_tensors='pt').input_ids[0].to(dev)
        rag_hist = hist + [{'role': 'user', 'content': 'Session note (retrieved from memory): %s\n\n%s' % (commit, dec)}]
        rag_pids = tok(H.tmpl(rag_hist[-WINDOW:]), return_tensors='pt').input_ids[0].to(dev)
        stacks.append(turn_stack(clean))
        samples.append({'stacks': stacks, 'pids': pids, 'rag': rag_pids,
                        'aids': tok(' ' + ans, add_special_tokens=False).input_ids, 'ans': ans})
        if (wi + 1) % 20 == 0: print('  %d/%d' % (wi + 1, NW), flush=True)
    r = random.Random(SEED)
    for s in samples: s['test'] = (r.random() < 0.3)
    TR = [s for s in samples if not s['test']]; TE = [s for s in samples if s['test']]
    import collections as _cl
    base = max(_cl.Counter([s['ans'] for s in samples]).values()) / len(samples)
    print('CKV15 train=%d test=%d base=%.3f (chance=%.3f)' % (len(TR), len(TE), base, 1.0 / NACTS), flush=True)

    g = AdaptiveGateSlot(D_MODEL, D_S, K, SLOW_K).to(dev)
    opt = torch.optim.Adam([{'params': g.parameters(), 'lr': LRg}, {'params': field_params, 'lr': FLR}])
    def Sfrom(stks):
        S = g.init()
        for st in stks: S = g.step(S, st.float().to(dev))
        return S
    @torch.no_grad()
    def gen(pids, S):
        if S is not None: _fb['S'] = S; _fb['on'] = True
        out = model.generate(pids.unsqueeze(0), max_new_tokens=6, do_sample=False, pad_token_id=tok.eos_token_id)
        _fb['on'] = False
        return tok.decode(out[0, pids.shape[0]:], skip_special_tokens=True).upper()
    @torch.no_grad()
    def preds_argmax(group, wrong=False):
        out = []; oi = random.Random(SEED + 3)
        for s in group:
            stks = samples[oi.randrange(len(samples))]['stacks'] if wrong else s['stacks']
            _fb['S'] = Sfrom(stks); _fb['on'] = True
            lg = model(s['pids'].unsqueeze(0)).logits[0]; _fb['on'] = False
            out.append(int(lg[-1].argmax()))
        return out
    @torch.no_grad()
    def greedy_acc(group, mode):
        c = 0
        for s in group:
            if mode == 'off': _fb['on'] = False; txt = gen(s['pids'], None)
            elif mode == 'rag': _fb['on'] = False; txt = gen(s['rag'], None)
            else: txt = gen(s['pids'], Sfrom(s['stacks']))
            c += int(s['ans'] in txt)
        return c / len(group)
    def report(it):
        g.eval(); [f.eval() for f in _fb['fields'].values()]
        tgt = [s['aids'][0] for s in TE]; pc = preds_argmax(TE); pw = preds_argmax(TE, wrong=True)
        accC = sum(int(pc[i] == tgt[i]) for i in range(len(TE))) / len(TE)
        accW = sum(int(pw[i] == tgt[i]) for i in range(len(TE))) / len(TE)
        chg = sum(int(pc[i] != pw[i]) for i in range(len(TE))) / len(TE)
        gON = greedy_acc(TE, 'on'); gRAG = greedy_acc(TE, 'rag'); fit = greedy_acc(TR[:12], 'on')
        print('CKV15 it=%-4d | accC=%.3f accW=%.3f DELTA=%.3f uniq=%d chg_wrongS=%.3f | greedy ON=%.3f RAG=%.3f fitON=%.3f'
              % (it, accC, accW, accC - accW, len(set(pc)), chg, gON, gRAG, fit), flush=True)
        g.train(); [f.train() for f in _fb['fields'].values()]

    report(0)
    rng2 = random.Random(SEED + 1)
    for it in range(1, ITERS + 1):
        s = TR[rng2.randrange(len(TR))]
        _fb['S'] = Sfrom(s['stacks']); _fb['on'] = True
        seq = torch.cat([s['pids'], torch.tensor(s['aids'], device=dev)]).unsqueeze(0)
        logits = model(seq).logits[0]; _fb['on'] = False
        pl = s['pids'].shape[0]
        lp = torch.log_softmax(logits[pl - 1:pl - 1 + len(s['aids'])], -1)
        nll = -lp[range(len(s['aids'])), torch.tensor(s['aids'], device=dev)].mean()
        opt.zero_grad(); nll.backward()
        torch.nn.utils.clip_grad_norm_(list(g.parameters()) + field_params, 1.0); opt.step()
        if it % EVERY == 0:
            print('CKV15 it=%d nll=%.4f' % (it, float(nll)), flush=True); report(it)
    print('=== CKV15_DONE ===', flush=True)


'''
src = io.open(PATH, encoding='utf-8').read()
if 'def carry_kv15()' in src:
    print('ALREADY'); raise SystemExit
anchor = "if MODE == 'validate':   validate()"
src = src.replace(anchor, FN + '\n\n' + anchor, 1)
src = src.replace("elif MODE == 'carry_kv14': carry_kv14()",
                  "elif MODE == 'carry_kv14': carry_kv14()\nelif MODE == 'carry_kv15': carry_kv15()", 1)
io.open(PATH, 'w', encoding='utf-8').write(src)
print('PATCHED_OK')
