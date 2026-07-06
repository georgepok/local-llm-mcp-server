import io
PATH = '/home/pokazge/NativeEntity/world_pop.py'
FN = r'''class BindComputeMod(nn.Module):
    # Fork A: BindCompute(S, H_current) -> computed latent C_t. Learned compute slots cross-attend
    # over [proj(S); proj(H_stack)] with weight-tied recurrent refinement (manual attn, no SDPA).
    # Outputs: C (latent), field-write vectors Cf (injected into Qwen via the slot field), decode logits.
    def __init__(s, d_s, d_model, d_c=512, n_c=8, steps=3):
        super().__init__()
        s.C0 = nn.Parameter(torch.randn(n_c, d_c) * 0.02)
        s.ps = nn.Linear(d_s, d_c); s.ph = nn.Linear(d_model, d_c)
        s.wq = nn.Linear(d_c, d_c); s.wk = nn.Linear(d_c, d_c); s.wv = nn.Linear(d_c, d_c)
        s.ffn = nn.Sequential(nn.Linear(d_c, 2 * d_c), nn.GELU(), nn.Linear(2 * d_c, d_c))
        s.ln1 = nn.LayerNorm(d_c); s.ln2 = nn.LayerNorm(d_c)
        s.out_field = nn.Linear(d_c, d_s)
        s.out_dec = nn.Linear(d_c, 6)
        s.steps = steps; s.scale = d_c ** -0.5

    def forward(s, S, Hstack):
        KV = torch.cat([s.ps(S), s.ph(Hstack)], 0)
        k = s.wk(KV); v = s.wv(KV)
        C = s.C0
        for _ in range(s.steps):
            q = s.wq(C)
            A = torch.softmax(q @ k.T * s.scale, dim=-1)
            C = s.ln1(C + A @ v)
            C = s.ln2(C + s.ffn(C))
        return C, s.out_field(C), s.out_dec(C.mean(0))


def carry_bind2():
    # FORK A: CARRY_BIND_APPLY_ONLY_V1 re-run WITH BindCompute. Same worlds/arms as carry_bind;
    # field kv = concat[S, BindCompute(S, decision-stack)]; joint loss = gen-NLL + aux CE on decode.
    LAYERS = [int(x) for x in os.environ.get('GEO_LAYERS', '4,8,12,16,20,24,28,32,36,40,44,48,52,56,60').split(',')]
    LV = int(os.environ.get('GEO_LEVEL', '1'))
    D = int(os.environ.get('GEO_ACT_D', '6')); NW = int(os.environ.get('GEO_NW', '60'))
    ITERS = int(os.environ.get('GEO_ITERS', '800')); EVERY = int(os.environ.get('GEO_EVAL_EVERY', '400'))
    LR = float(os.environ.get('GEO_LR', '1e-4')); BLR = float(os.environ.get('GEO_BLR', '3e-4'))
    ACT_EPS = float(os.environ.get('GEO_ACT_EPS', '0.1')); AUXW = float(os.environ.get('GEO_AUXW', '1.0'))
    print('=== CARRY_BIND2(BindCompute) L%d | D=%d NW=%d iters=%d aux=%.1f ===' % (LV, D, NW, ITERS, AUXW), flush=True)
    rng = random.Random(SEED)
    ACT4 = ['KEEP', 'REJECT', 'DEFER', 'ASK']
    FILL_U, FILL_A = 'Routine status check, no action needed.', 'Acknowledged. Standing by.'

    def mkworld():
        if LV == 1:
            a = ACTIONS[rng.randrange(6)]
            commit = 'STANDING RULING for this session: the authorized action is %s. Retain it.' % a
            dec = 'State the standing authorized action for this session. %s' % ASK_INSTR
            return commit, dec, a, ACTIONS.index(a)
        if LV == 2:
            Kk = LATENTS[rng.randrange(4)]
            acts = ACT4[:]; rng.shuffle(acts); mp = dict(zip(LATENTS, acts))
            commit = 'STANDING KEY for this session: %s %s %s. Retain it.' % (Kk, Kk, Kk)
            dec = ('RULING REQUIRED. Keyline: %s. The ruling is the one listed for the standing session key. %s'
                   % (', '.join('%s->%s' % (t, mp[t]) for t in LATENTS), ASK_INSTR))
            return commit, dec, mp[Kk], LATENTS.index(Kk)
        if LV == 3:
            Kk = LATENTS[rng.randrange(4)]
            order = LATENTS[:]; rng.shuffle(order)
            acts = ACT4[:]; rng.shuffle(acts); mp = dict(zip(LATENTS, acts))
            succ = order[(order.index(Kk) + 1) % 4]
            commit = 'STANDING KEY for this session: %s %s %s. Retain it.' % (Kk, Kk, Kk)
            dec = ('RULING REQUIRED. Items in order: %s. Keyline: %s. The ruling is the one listed for the item '
                   'that comes IMMEDIATELY AFTER the standing session key in the order (wrapping around). %s'
                   % (', '.join(order), ', '.join('%s->%s' % (t, mp[t]) for t in order), ASK_INSTR))
            return commit, dec, mp[succ], LATENTS.index(Kk)
        if LV == 4:
            rid = rng.randrange(4)
            rtxt = ['the item with the HIGHEST score', 'the item with the LOWEST score',
                    'the FIRST item listed', 'the LAST item listed'][rid]
            commit = 'STANDING RULE for this session: the authorized item is %s. Retain it.' % rtxt
            items = rng.sample(LATENTS, 3); scores = rng.sample(range(10, 99), 3); acts = rng.sample(ACTIONS, 3)
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

    samples = []; nleak = 0
    print('CB2 building %d worlds ...' % NW, flush=True)
    for wi in range(NW):
        commit, dec, ans, vid = mkworld()
        nleak += int(ans.upper() in commit.upper())
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
    print('CB2 L%d leakage=%d/%d (expect %s) base-rate=%.3f' % (LV, nleak, NW, 'NW' if LV == 1 else '0', base), flush=True)
    r = random.Random(SEED)
    for s in samples: s['test'] = (r.random() < 0.3)
    TR = [s for s in samples if not s['test']]; TE = [s for s in samples if s['test']]
    print('CB2 train=%d test=%d' % (len(TR), len(TE)), flush=True)

    _fb['fields'] = {L: SL.AlwaysOnSlotField(D_MODEL, D_S, eps=ACT_EPS).to(dev) for L in FIELD_LAYERS}
    fp = []
    for L in FIELD_LAYERS:
        for p in _fb['fields'][L].parameters(): p.requires_grad_(True); fp.append(p)
    g = AdaptiveGateSlot(D_MODEL, D_S, K, SLOW_K).to(dev)
    bind = BindComputeMod(D_S, D_MODEL).to(dev)
    opt = torch.optim.Adam([{'params': list(g.parameters()) + fp, 'lr': LR},
                            {'params': list(bind.parameters()), 'lr': BLR}])

    def Sfrom(stks):
        S = g.init()
        for st in stks: S = g.step(S, st.float().to(dev))
        return S

    def SC(s, variant, oi=None):
        if variant == 'on': stks = s['stacks']
        elif variant == 'reset': stks = s['stacks'][1:]
        else:
            other = samples[oi.randrange(len(samples))]
            stks = [other['stacks'][0]] + s['stacks'][1:]
        S = Sfrom(stks)
        C, Cf, dlog = bind(S, s['stacks'][-1].float().to(dev))
        return torch.cat([S, Cf], 0), dlog

    @torch.no_grad()
    def gen_arm(group, mode):
        if not group: return 0.0
        c = 0; oi = random.Random(SEED + 7)
        for s in group:
            pids = s['pids']
            if mode == 'off': _fb['on'] = False
            elif mode == 'rag': _fb['on'] = False; pids = s['rag']
            elif mode == 'ora': _fb['on'] = False; pids = s['ora']
            else:
                Sfull, _ = SC(s, mode, oi)
                _fb['S'] = Sfull; _fb['on'] = True
                for L in FIELD_LAYERS: _fb['fields'][L].eps = ACT_EPS
            out = model.generate(pids.unsqueeze(0), max_new_tokens=6, do_sample=False, pad_token_id=tok.eos_token_id)
            _fb['on'] = False
            txt = tok.decode(out[0, pids.shape[0]:], skip_special_tokens=True).upper()
            ai = next((j for j, a in enumerate(ACTIONS) if a in txt), -1)
            c += int(ai == s['cidx'])
        return c / len(group)

    @torch.no_grad()
    def dec_acc(group):
        if not group: return 0.0
        c = 0
        for s in group:
            _, dlog = SC(s, 'on')
            c += int(int(dlog.argmax()) == s['cidx'])
        return c / len(group)

    def report(it):
        g.eval(); bind.eval()
        for L in FIELD_LAYERS: _fb['fields'][L].eval()
        vals = {m: gen_arm(TE, m) for m in ['off', 'on', 'reset', 'wrong', 'rag', 'ora']}
        fit = gen_arm(TR[:12], 'on'); dtr = dec_acc(TR[:24]); dte = dec_acc(TE)
        print('CB2 L%d it=%-4d | OFF=%.3f ON=%.3f ONreset=%.3f ONwrong=%.3f RAG=%.3f ORACLE=%.3f | fitON=%.3f | BINDdec tr=%.3f TE=%.3f'
              % (LV, it, vals['off'], vals['on'], vals['reset'], vals['wrong'], vals['rag'], vals['ora'], fit, dtr, dte), flush=True)
        g.train(); bind.train()
        for L in FIELD_LAYERS: _fb['fields'][L].train()

    report(0)
    rng2 = random.Random(SEED + 1)
    for it in range(1, ITERS + 1):
        s = TR[rng2.randrange(len(TR))]
        Sfull, dlog = SC(s, 'on')
        _fb['S'] = Sfull; _fb['on'] = True
        for L in FIELD_LAYERS: _fb['fields'][L].eps = ACT_EPS
        seq = torch.cat([s['pids'], torch.tensor(s['aids'], device=dev)]).unsqueeze(0)
        logits = model(seq).logits[0]
        _fb['on'] = False
        pl = s['pids'].shape[0]
        lp = torch.log_softmax(logits[pl - 1:pl - 1 + len(s['aids'])], -1)
        nll = -lp[range(len(s['aids'])), torch.tensor(s['aids'], device=dev)].mean()
        aux = F.cross_entropy(dlog.unsqueeze(0), torch.tensor([s['cidx']], device=dev))
        loss = nll + AUXW * aux
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(g.parameters()) + fp + list(bind.parameters()), 1.0); opt.step()
        if it % EVERY == 0:
            print('CB2 L%d it=%d nll=%.4f aux=%.4f' % (LV, it, float(nll), float(aux)), flush=True); report(it)

    g.eval(); bind.eval()
    with torch.no_grad():
        Sp = torch.stack([Sfrom(s['stacks']).mean(0).cpu() for s in samples])
    yv = torch.tensor([s['vid'] for s in samples]); tem = torch.tensor([s['test'] for s in samples])
    Xtr, ytr, Xte, yte = Sp[~tem], yv[~tem], Sp[tem], yv[tem]
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xte = ((Xtr - mu) / sd).to(dev), ((Xte - mu) / sd).to(dev); ytr, yte = ytr.to(dev), yte.to(dev)
    nvar = int(yv.max()) + 1
    net = nn.Linear(Xtr.shape[1], nvar).to(dev)
    o = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
    for e in range(500): o.zero_grad(); F.cross_entropy(net(Xtr), ytr).backward(); o.step()
    net.eval()
    with torch.no_grad():
        pv = float((net(Xte).argmax(1) == yte).float().mean())
    print('CB2 L%d PROBES | S->carriedVar lin TE=%.3f (chance %.3f)' % (LV, pv, 1.0 / nvar), flush=True)
    print('=== CB2_L%d_DONE ===' % LV, flush=True)


'''
src = io.open(PATH, encoding='utf-8').read()
if 'def carry_bind2()' in src:
    print('ALREADY'); raise SystemExit
anchor = "if MODE == 'validate':   validate()"
src = src.replace(anchor, FN + '\n\n' + anchor, 1)
src = src.replace("elif MODE == 'carry_bind': carry_bind()",
                  "elif MODE == 'carry_bind': carry_bind()\nelif MODE == 'carry_bind2': carry_bind2()", 1)
io.open(PATH, 'w', encoding='utf-8').write(src)
print('PATCHED_OK')
