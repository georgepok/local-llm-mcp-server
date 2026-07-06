import io
PATH = '/home/pokazge/NativeEntity/world_pop.py'
FN = r'''def carry_kv14():
    # FIELD content-audit v2 — uses the REAL SL.AlwaysOnSlotField via the existing _fb hooks (the field that
    # earlier produced L1 actuation), NOT my unstable reimplementation. Conservative field LR. Adds the
    # mandatory content control accC(correct-S) vs accW(wrong-S). Question: does the trainable additive field
    # drive the frozen LLM's output from memory CONTENT?  accC>>accW & chg_wrongS>0 => YES (retrieval works).
    LAYERS = [int(x) for x in os.environ.get('GEO_LAYERS', '4,8,12,16,20,24,28,32,36,40,44,48,52,56,60').split(',')]
    LV = int(os.environ.get('GEO_LEVEL', '1'))
    D = int(os.environ.get('GEO_ACT_D', '6')); NW = int(os.environ.get('GEO_NW', '60'))
    ITERS = int(os.environ.get('GEO_ITERS', '800')); EVERY = int(os.environ.get('GEO_EVAL_EVERY', '400'))
    LRg = float(os.environ.get('GEO_LR', '1e-4')); FLR = float(os.environ.get('GEO_FLR', '1e-4'))
    EPSF = float(os.environ.get('GEO_FIELD_EPS', '0.1'))
    print('=== CARRY_KV14(REAL AlwaysOnSlotField content-audit, eps=%.2f layers=%s) L%d ===' % (EPSF, FIELD_LAYERS, LV), flush=True)
    rng = random.Random(SEED)
    ACT4 = ['KEEP', 'REJECT', 'DEFER', 'ASK']
    FILL_U, FILL_A = 'Routine status check, no action needed.', 'Acknowledged. Standing by.'

    _fb['fields'] = {L: SL.AlwaysOnSlotField(D_MODEL, D_S, eps=EPSF).to(dev) for L in FIELD_LAYERS}
    _fb['on'] = False
    field_params = [p for L in FIELD_LAYERS for p in _fb['fields'][L].parameters()]

    def mkworld():
        if LV == 2:
            Kk = LATENTS[rng.randrange(4)]; acts = ACT4[:]; rng.shuffle(acts); mp = dict(zip(LATENTS, acts))
            return ('STANDING KEY for this session: %s %s %s. Retain it.' % (Kk, Kk, Kk),
                    'RULING REQUIRED. Keyline: %s. The ruling is the one listed for the standing session key. %s'
                    % (', '.join('%s->%s' % (t, mp[t]) for t in LATENTS), ASK_INSTR), mp[Kk], LATENTS.index(Kk))
        a = ACTIONS[rng.randrange(6)]
        return ('STANDING RULING for this session: the authorized action is %s. Retain it.' % a,
                'State the standing authorized action for this session. %s' % ASK_INSTR, a, ACTIONS.index(a))

    @torch.no_grad()
    def turn_stack(hist):
        _fb['on'] = False
        ids = tok(H.tmpl(hist[-WINDOW:]), return_tensors='pt').input_ids.to(dev)
        hs = model(ids, output_hidden_states=True).hidden_states
        return torch.stack([hs[L][0][-1].float() for L in LAYERS]).to(torch.float16).cpu()

    samples = []
    print('CKV14 building %d worlds ...' % NW, flush=True)
    for wi in range(NW):
        commit, dec, ans, vid = mkworld()
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
        samples.append({'stacks': stacks, 'pids': pids, 'rag': rag_pids, 'aids': tok(' ' + ans, add_special_tokens=False).input_ids, 'cidx': ACTIONS.index(ans)})
        if (wi + 1) % 20 == 0: print('  %d/%d' % (wi + 1, NW), flush=True)
    r = random.Random(SEED)
    for s in samples: s['test'] = (r.random() < 0.3)
    TR = [s for s in samples if not s['test']]; TE = [s for s in samples if s['test']]
    print('CKV14 train=%d test=%d' % (len(TR), len(TE)), flush=True)

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
    def acc(group, mode):
        c = 0
        for s in group:
            if mode == 'off': _fb['on'] = False; txt = gen(s['pids'], None)
            elif mode == 'rag': _fb['on'] = False; txt = gen(s['rag'], None)
            else: txt = gen(s['pids'], Sfrom(s['stacks']))
            ai = next((j for j, a in enumerate(ACTIONS) if a in txt), -1)
            c += int(ai == s['cidx'])
        return c / len(group)

    def report(it):
        g.eval(); [f.eval() for f in _fb['fields'].values()]
        tgt = [s['aids'][0] for s in TE]; pc = preds_argmax(TE); pw = preds_argmax(TE, wrong=True)
        accC = sum(int(pc[i] == tgt[i]) for i in range(len(TE))) / len(TE)
        accW = sum(int(pw[i] == tgt[i]) for i in range(len(TE))) / len(TE)
        chg = sum(int(pc[i] != pw[i]) for i in range(len(TE))) / len(TE)
        gON = acc(TE, 'on'); gOFF = acc(TE, 'off'); gRAG = acc(TE, 'rag'); fit = acc(TR[:12], 'on')
        print('CKV14 L%d it=%-4d | accC=%.3f accW=%.3f DELTA=%.3f uniq=%d chg_wrongS=%.3f | greedy ON=%.3f OFF=%.3f RAG=%.3f fitON=%.3f'
              % (LV, it, accC, accW, accC - accW, len(set(pc)), chg, gON, gOFF, gRAG, fit), flush=True)
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
            print('CKV14 L%d it=%d nll=%.4f' % (LV, it, float(nll)), flush=True); report(it)
    print('=== CKV14_L%d_DONE ===' % LV, flush=True)


'''
src = io.open(PATH, encoding='utf-8').read()
if 'def carry_kv14()' in src:
    print('ALREADY'); raise SystemExit
anchor = "if MODE == 'validate':   validate()"
src = src.replace(anchor, FN + '\n\n' + anchor, 1)
src = src.replace("elif MODE == 'carry_kv13': carry_kv13()",
                  "elif MODE == 'carry_kv13': carry_kv13()\nelif MODE == 'carry_kv14': carry_kv14()", 1)
io.open(PATH, 'w', encoding='utf-8').write(src)
print('PATCHED_OK')
