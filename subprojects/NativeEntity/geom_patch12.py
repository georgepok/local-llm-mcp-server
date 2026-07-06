import io
PATH = '/home/pokazge/NativeEntity/world_pop.py'
FN = r'''def gen_actuate():
    # (b) GENERIC-STACK READ + field ACTUATE: build S from the LLM layer stack (learned depth), then the
    # field surfaces it into Qwen's GENERATION out-of-window. Trained end-to-end. field-ON vs field-OFF.
    LAYERS = [int(x) for x in os.environ.get('GEO_LAYERS', '4,8,12,16,20,24,28,32,36,40,44,48,52,56,60').split(',')]
    D = int(os.environ.get('GEO_ACT_D', '8')); NW = int(os.environ.get('GEO_NW', '28'))
    ITERS = int(os.environ.get('GEO_ITERS', '500')); EVERY = int(os.environ.get('GEO_EVAL_EVERY', '100'))
    LR = float(os.environ.get('GEO_LR', '1e-4')); ACT_EPS = float(os.environ.get('GEO_ACT_EPS', '0.1'))
    print('=== GEN_ACTUATE | generic-stack READ + field ACTUATE | carry-action D=%d eps=%.2f ===' % (D, ACT_EPS), flush=True)
    rng = random.Random(SEED)
    _fb['fields'] = {L: SL.AlwaysOnSlotField(D_MODEL, D_S, eps=ACT_EPS).to(dev) for L in FIELD_LAYERS}
    fp = []
    for L in FIELD_LAYERS:
        for p in _fb['fields'][L].parameters(): p.requires_grad_(True); fp.append(p)
    g = AdaptiveGateSlot(D_MODEL, D_S, K, SLOW_K).to(dev)
    FILL_U, FILL_A = 'Routine status check, no action needed.', 'Acknowledged. Standing by.'

    @torch.no_grad()
    def turn_stack(hist):
        _fb['on'] = False
        ids = tok(H.tmpl(hist[-WINDOW:]), return_tensors='pt').input_ids.to(dev)
        hs = model(ids, output_hidden_states=True).hidden_states
        return torch.stack([hs[L][0][-1].float() for L in LAYERS]).to(torch.float16).cpu()

    samples = []
    print('GENA building %d worlds ...' % NW, flush=True)
    for wi in range(NW):
        act = ACTIONS[rng.randrange(6)]
        commit = 'STANDING RULING for this session: the authorized action is %s. Retain it.' % act
        hist = [{'role': 'user', 'content': commit}, {'role': 'assistant', 'content': 'Acknowledged.'}]
        stacks = [turn_stack(hist)]
        for _ in range(D):
            hist += [{'role': 'user', 'content': FILL_U}, {'role': 'assistant', 'content': FILL_A}]
            stacks.append(turn_stack(hist))
        hist += [{'role': 'user', 'content': 'State the standing authorized action for this session in one word. %s' % ASK_INSTR}]
        pids = tok(H.tmpl(hist[-WINDOW:]), return_tensors='pt').input_ids[0].to(dev)
        aids = tok(' ' + act, add_special_tokens=False).input_ids
        samples.append({'stacks': stacks, 'pids': pids, 'aids': aids, 'cidx': ACTIONS.index(act)})
    r = random.Random(SEED)
    for s in samples: s['test'] = (r.random() < 0.3)
    TR = [s for s in samples if not s['test']]; TE = [s for s in samples if s['test']]
    print('GENA train=%d test=%d' % (len(TR), len(TE)), flush=True)

    def Sof(s):
        S = g.init()
        for st in s['stacks']: S = g.step(S, st.float().to(dev))
        return S

    opt = torch.optim.Adam(list(g.parameters()) + fp, lr=LR)

    @torch.no_grad()
    def gen_acc(group, field_on):
        if not group: return 0.0
        c = 0
        for s in group:
            if field_on:
                _fb['S'] = Sof(s); _fb['on'] = True
                for L in FIELD_LAYERS: _fb['fields'][L].eps = ACT_EPS
            else:
                _fb['on'] = False
            out = model.generate(s['pids'].unsqueeze(0), max_new_tokens=6, do_sample=False, pad_token_id=tok.eos_token_id)
            _fb['on'] = False
            txt = tok.decode(out[0, s['pids'].shape[0]:], skip_special_tokens=True).upper()
            ai = next((j for j, a in enumerate(ACTIONS) if a in txt), -1)
            c += int(ai == s['cidx'])
        return c / len(group)

    def report(it):
        g.eval()
        for L in FIELD_LAYERS: _fb['fields'][L].eval()
        off = gen_acc(TE, False); on = gen_acc(TE, True)
        print('GENA it=%-4d | field-OFF(LLM alone)=%.3f  field-ON(generic-stack)=%.3f' % (it, off, on), flush=True)
        g.train()
        for L in FIELD_LAYERS: _fb['fields'][L].train()

    report(0)
    rng2 = random.Random(SEED + 1)
    for it in range(1, ITERS + 1):
        s = TR[rng2.randrange(len(TR))]
        _fb['S'] = Sof(s); _fb['on'] = True
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
            print('GENA it=%d nll=%.4f' % (it, float(nll)), flush=True); report(it)
    print('=== GEN_ACTUATE_DONE ===', flush=True)


'''
src = io.open(PATH, encoding='utf-8').read()
if 'def gen_actuate()' in src:
    print('ALREADY'); raise SystemExit
anchor = "if MODE == 'validate':   validate()"
src = src.replace(anchor, FN + '\n\n' + anchor, 1)
src = src.replace("elif MODE == 'gen_both2': gen_both2()",
                  "elif MODE == 'gen_both2': gen_both2()\nelif MODE == 'gen_actuate': gen_actuate()", 1)
io.open(PATH, 'w', encoding='utf-8').write(src)
print('PATCHED_OK')
