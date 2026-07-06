import io
PATH = '/home/pokazge/NativeEntity/world_pop.py'
FN = r'''def carry_bind3():
    # FORK A stage-2 control: OFFLINE-pretrain BindCompute (full-batch, decode CE — cheap, no Qwen fwd),
    # THEN freeze g+bind and train ONLY the field on generation. Separates:
    #   pretrain decode fails offline        -> representation/update insufficient for relational use
    #   decode works, generation fails       -> S->Qwen interface is deficient
    #   decode works, generation works       -> compute+inject viable; the inline failure was optimization
    LAYERS = [int(x) for x in os.environ.get('GEO_LAYERS', '4,8,12,16,20,24,28,32,36,40,44,48,52,56,60').split(',')]
    LV = int(os.environ.get('GEO_LEVEL', '2'))
    D = int(os.environ.get('GEO_ACT_D', '6')); NW = int(os.environ.get('GEO_NW', '60'))
    PRE_EPOCHS = int(os.environ.get('GEO_PRE_EPOCHS', '600'))
    ITERS = int(os.environ.get('GEO_ITERS', '500')); EVERY = int(os.environ.get('GEO_EVAL_EVERY', '250'))
    LR = float(os.environ.get('GEO_LR', '1e-4')); BLR = float(os.environ.get('GEO_BLR', '1e-3'))
    ACT_EPS = float(os.environ.get('GEO_ACT_EPS', '0.1'))
    print('=== CARRY_BIND3(staged) L%d | pre=%d epochs then field-only %d iters ===' % (LV, PRE_EPOCHS, ITERS), flush=True)
    rng = random.Random(SEED)
    ACT4 = ['KEEP', 'REJECT', 'DEFER', 'ASK']
    FILL_U, FILL_A = 'Routine status check, no action needed.', 'Acknowledged. Standing by.'

    def mkworld():
        if LV == 1:
            a = ACTIONS[rng.randrange(6)]
            return ('STANDING RULING for this session: the authorized action is %s. Retain it.' % a,
                    'State the standing authorized action for this session. %s' % ASK_INSTR, a, ACTIONS.index(a))
        if LV == 2:
            Kk = LATENTS[rng.randrange(4)]
            acts = ACT4[:]; rng.shuffle(acts); mp = dict(zip(LATENTS, acts))
            return ('STANDING KEY for this session: %s %s %s. Retain it.' % (Kk, Kk, Kk),
                    'RULING REQUIRED. Keyline: %s. The ruling is the one listed for the standing session key. %s'
                    % (', '.join('%s->%s' % (t, mp[t]) for t in LATENTS), ASK_INSTR), mp[Kk], LATENTS.index(Kk))
        if LV == 3:
            Kk = LATENTS[rng.randrange(4)]
            order = LATENTS[:]; rng.shuffle(order)
            acts = ACT4[:]; rng.shuffle(acts); mp = dict(zip(LATENTS, acts))
            succ = order[(order.index(Kk) + 1) % 4]
            return ('STANDING KEY for this session: %s %s %s. Retain it.' % (Kk, Kk, Kk),
                    'RULING REQUIRED. Items in order: %s. Keyline: %s. The ruling is the one listed for the item '
                    'that comes IMMEDIATELY AFTER the standing session key in the order (wrapping around). %s'
                    % (', '.join(order), ', '.join('%s->%s' % (t, mp[t]) for t in order), ASK_INSTR),
                    mp[succ], LATENTS.index(Kk))
        if LV == 4:
            rid = rng.randrange(4)
            rtxt = ['the item with the HIGHEST score', 'the item with the LOWEST score',
                    'the FIRST item listed', 'the LAST item listed'][rid]
            items = rng.sample(LATENTS, 3); scores = rng.sample(range(10, 99), 3); acts = rng.sample(ACTIONS, 3)
            if rid == 0: w = max(range(3), key=lambda i: scores[i])
            elif rid == 1: w = min(range(3), key=lambda i: scores[i])
            elif rid == 2: w = 0
            else: w = 2
            return ('STANDING RULE for this session: the authorized item is %s. Retain it.' % rtxt,
                    'EVALUATION. Items: %s. Apply the standing session rule and reply with the ruling of the '
                    'authorized item. %s' % (', '.join('%s (score %d) -> %s' % (items[i], scores[i], acts[i])
                                                       for i in range(3)), ASK_INSTR), acts[w], rid)
        if LV == 5:
            did = rng.randrange(3)
            dt = ['EQUALITY doctrine: an archive submission is accepted only when it exactly equals the designated entry.',
                  'ORDER doctrine: an archive submission is accepted only when it is at or before the designated entry in the sequence.',
                  'EXCLUSION doctrine: every archive submission is accepted except the designated entry itself.'][did]
            order = LATENTS[:]; rng.shuffle(order)
            X = LATENTS[rng.randrange(4)]; Y = LATENTS[rng.randrange(4)]
            if did == 0: honored = (Y == X)
            elif did == 1: honored = (order.index(Y) <= order.index(X))
            else: honored = (Y != X)
            return ('SESSION DOCTRINE (archive intake): %s Retain it.' % dt,
                    'GATEWAY CLEARANCE request. Sequence: %s. Designated entry: %s. Requesting entry: %s. '
                    'Under the standing session doctrine, is the request cleared? Reply KEEP if cleared, '
                    'REJECT if not. %s' % (', '.join(order), X, Y, ASK_INSTR),
                    ('KEEP' if honored else 'REJECT'), did)
        raise ValueError(LV)

    @torch.no_grad()
    def turn_stack(hist):
        _fb['on'] = False
        ids = tok(H.tmpl(hist[-WINDOW:]), return_tensors='pt').input_ids.to(dev)
        hs = model(ids, output_hidden_states=True).hidden_states
        return torch.stack([hs[L][0][-1].float() for L in LAYERS]).to(torch.float16).cpu()

    samples = []
    print('CB3 building %d worlds ...' % NW, flush=True)
    for wi in range(NW):
        commit, dec, ans, vid = mkworld()
        hist = [{'role': 'user', 'content': commit}, {'role': 'assistant', 'content': 'Acknowledged.'}]
        stacks = [turn_stack(hist)]
        for _ in range(D):
            hist += [{'role': 'user', 'content': FILL_U}, {'role': 'assistant', 'content': FILL_A}]
            stacks.append(turn_stack(hist))
        hist += [{'role': 'user', 'content': dec}]
        pids = tok(H.tmpl(hist[-WINDOW:]), return_tensors='pt').input_ids[0].to(dev)
        stacks.append(turn_stack(hist))
        samples.append({'stacks': stacks, 'pids': pids,
                        'aids': tok(' ' + ans, add_special_tokens=False).input_ids,
                        'cidx': ACTIONS.index(ans), 'vid': vid})
        if (wi + 1) % 20 == 0: print('  %d/%d' % (wi + 1, NW), flush=True)
    r = random.Random(SEED)
    for s in samples: s['test'] = (r.random() < 0.3)
    TR = [s for s in samples if not s['test']]; TE = [s for s in samples if s['test']]
    import collections as _cl
    base = max(_cl.Counter([s['cidx'] for s in samples]).values()) / float(len(samples))
    print('CB3 train=%d test=%d base=%.3f' % (len(TR), len(TE), base), flush=True)

    g = AdaptiveGateSlot(D_MODEL, D_S, K, SLOW_K).to(dev)
    bind = BindComputeMod(D_S, D_MODEL).to(dev)

    def Sfrom(stks):
        S = g.init()
        for st in stks: S = g.step(S, st.float().to(dev))
        return S

    # ── STAGE 1: offline full-batch pretrain of g+bind on decode CE (no Qwen forward) ──
    optp = torch.optim.Adam([{'params': g.parameters(), 'lr': 3e-4}, {'params': bind.parameters(), 'lr': BLR}])
    ytr = torch.tensor([s['cidx'] for s in TR], device=dev)
    for ep in range(1, PRE_EPOCHS + 1):
        logits = torch.stack([bind(Sfrom(s['stacks']), s['stacks'][-1].float().to(dev))[2] for s in TR])
        loss = F.cross_entropy(logits, ytr)
        optp.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(g.parameters()) + list(bind.parameters()), 1.0); optp.step()
        if ep % max(1, PRE_EPOCHS // 4) == 0:
            g.eval(); bind.eval()
            with torch.no_grad():
                dtr = float((torch.stack([bind(Sfrom(s['stacks']), s['stacks'][-1].float().to(dev))[2] for s in TR]).argmax(1) == ytr).float().mean())
                dte = float((torch.stack([bind(Sfrom(s['stacks']), s['stacks'][-1].float().to(dev))[2] for s in TE]).argmax(1)
                             == torch.tensor([s['cidx'] for s in TE], device=dev)).float().mean())
            print('CB3 L%d PRETRAIN ep=%d loss=%.4f | BINDdec tr=%.3f TE=%.3f' % (LV, ep, float(loss), dtr, dte), flush=True)
            g.train(); bind.train()
    g.eval(); bind.eval()
    for p in g.parameters(): p.requires_grad_(False)
    for p in bind.parameters(): p.requires_grad_(False)

    # ── STAGE 2: field-only generation training with frozen compute ──
    _fb['fields'] = {L: SL.AlwaysOnSlotField(D_MODEL, D_S, eps=ACT_EPS).to(dev) for L in FIELD_LAYERS}
    fp = []
    for L in FIELD_LAYERS:
        for p in _fb['fields'][L].parameters(): p.requires_grad_(True); fp.append(p)
    optf = torch.optim.Adam(fp, lr=LR)

    @torch.no_grad()
    def SCfull(s, variant, oi=None):
        if variant == 'on': stks = s['stacks']
        elif variant == 'reset': stks = s['stacks'][1:]
        else:
            other = samples[oi.randrange(len(samples))]
            stks = [other['stacks'][0]] + s['stacks'][1:]
        S = Sfrom(stks)
        C, Cf, dlog = bind(S, s['stacks'][-1].float().to(dev))
        return torch.cat([S, Cf], 0)

    @torch.no_grad()
    def gen_arm(group, mode):
        if not group: return 0.0
        c = 0; oi = random.Random(SEED + 7)
        for s in group:
            if mode == 'off': _fb['on'] = False
            else:
                _fb['S'] = SCfull(s, mode, oi); _fb['on'] = True
                for L in FIELD_LAYERS: _fb['fields'][L].eps = ACT_EPS
            out = model.generate(s['pids'].unsqueeze(0), max_new_tokens=6, do_sample=False, pad_token_id=tok.eos_token_id)
            _fb['on'] = False
            txt = tok.decode(out[0, s['pids'].shape[0]:], skip_special_tokens=True).upper()
            ai = next((j for j, a in enumerate(ACTIONS) if a in txt), -1)
            c += int(ai == s['cidx'])
        return c / len(group)

    def report(it):
        for L in FIELD_LAYERS: _fb['fields'][L].eval()
        off = gen_arm(TE, 'off'); on = gen_arm(TE, 'on'); rs = gen_arm(TE, 'reset'); wr = gen_arm(TE, 'wrong')
        fit = gen_arm(TR[:12], 'on')
        print('CB3 L%d it=%-4d | OFF=%.3f ON=%.3f ONreset=%.3f ONwrong=%.3f | fitON=%.3f' % (LV, it, off, on, rs, wr, fit), flush=True)
        for L in FIELD_LAYERS: _fb['fields'][L].train()

    report(0)
    rng2 = random.Random(SEED + 1)
    for it in range(1, ITERS + 1):
        s = TR[rng2.randrange(len(TR))]
        _fb['S'] = SCfull(s, 'on'); _fb['on'] = True
        for L in FIELD_LAYERS: _fb['fields'][L].eps = ACT_EPS
        seq = torch.cat([s['pids'], torch.tensor(s['aids'], device=dev)]).unsqueeze(0)
        logits = model(seq).logits[0]
        _fb['on'] = False
        pl = s['pids'].shape[0]
        lp = torch.log_softmax(logits[pl - 1:pl - 1 + len(s['aids'])], -1)
        nll = -lp[range(len(s['aids'])), torch.tensor(s['aids'], device=dev)].mean()
        optf.zero_grad(); nll.backward()
        torch.nn.utils.clip_grad_norm_(fp, 1.0); optf.step()
        if it % EVERY == 0:
            print('CB3 L%d it=%d nll=%.4f' % (LV, it, float(nll)), flush=True); report(it)
    print('=== CB3_L%d_DONE ===' % LV, flush=True)


'''
src = io.open(PATH, encoding='utf-8').read()
if 'def carry_bind3()' in src:
    print('ALREADY'); raise SystemExit
anchor = "if MODE == 'validate':   validate()"
src = src.replace(anchor, FN + '\n\n' + anchor, 1)
src = src.replace("elif MODE == 'carry_bind2': carry_bind2()",
                  "elif MODE == 'carry_bind2': carry_bind2()\nelif MODE == 'carry_bind3': carry_bind3()", 1)
io.open(PATH, 'w', encoding='utf-8').write(src)
print('PATCHED_OK')
