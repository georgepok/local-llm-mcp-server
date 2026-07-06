import io
PATH = '/home/pokazge/NativeEntity/world_pop.py'
FN = r'''def carry_kv6():
    # FORK B v6 = v5 injection + DIAGNOSTIC. v5 hit nll=0 (full) but greedy=chance -> suspect multi-token
    # answer dilution (avg nll acing trivial continuation tokens, never the 1st discriminative token).
    # Fix: loss on FIRST answer token only. Report TFacc (teacher-forced 1st-token argmax, memory on) vs greedy.
    #   TFacc high + greedy low  -> generate/cache injection bug (fixable)
    #   TFacc high + greedy high -> INTERFACE WORKS (proceed to L2-L4)
    #   TFacc low                -> interface cannot drive the discriminative token
    import transformers.models.qwen3_5.modeling_qwen3_5 as QM
    LAYERS = [int(x) for x in os.environ.get('GEO_LAYERS', '4,8,12,16,20,24,28,32,36,40,44,48,52,56,60').split(',')]
    LV = int(os.environ.get('GEO_LEVEL', '1'))
    D = int(os.environ.get('GEO_ACT_D', '6')); NW = int(os.environ.get('GEO_NW', '60'))
    NMEM = int(os.environ.get('GEO_NMEM', '16'))
    ITERS = int(os.environ.get('GEO_ITERS', '800')); EVERY = int(os.environ.get('GEO_EVAL_EVERY', '400'))
    LR = float(os.environ.get('GEO_LR', '1e-4')); MLR = float(os.environ.get('GEO_MLR', '3e-4'))
    FULL = [i for i in range(len(model.model.layers)) if hasattr(model.model.layers[i], 'self_attn')]
    INJ = [int(x) for x in os.environ.get('GEO_INJ_LAYERS', '43,51,59').split(',')]
    INJ = [L for L in INJ if L in FULL]
    print('=== CARRY_KV6(ForkB attn-KV DIAG, first-token loss) L%d | inj=%s n_mem=%d iters=%d ===' % (LV, INJ, NMEM, ITERS), flush=True)
    rng = random.Random(SEED)
    ACT4 = ['KEEP', 'REJECT', 'DEFER', 'ASK']
    FILL_U, FILL_A = 'Routine status check, no action needed.', 'Acknowledged. Standing by.'
    _fb['on'] = False
    E = model.get_input_embeddings(); edt = E.weight.dtype

    _mem = {'on': False, 'h': None}
    _orig_eager = QM.eager_attention_forward
    INJ_SET = set(INJ)
    def mem_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        if _mem['on'] and _mem['h'] is not None and getattr(module, 'layer_idx', None) in INJ_SET:
            hd = module.head_dim; mh = _mem['h'].to(key.dtype); M = mh.shape[0]
            mk = module.k_norm(module.k_proj(mh).view(M, -1, hd)).transpose(0, 1).unsqueeze(0)
            mv = module.v_proj(mh).view(M, -1, hd).transpose(0, 1).unsqueeze(0)
            b = key.shape[0]
            key = torch.cat([key, mk.expand(b, -1, -1, -1)], dim=2)
            value = torch.cat([value, mv.expand(b, -1, -1, -1)], dim=2)
            if attention_mask is not None:
                add = torch.zeros(*attention_mask.shape[:-1], M, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, add], dim=-1)
        return _orig_eager(module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs)
    try: QM.ALL_ATTENTION_FUNCTIONS.register('mem_eager', mem_eager)
    except Exception: QM.ALL_ATTENTION_FUNCTIONS['mem_eager'] = mem_eager
    for L in FULL: model.model.layers[L].self_attn.config._attn_implementation = 'mem_eager'

    def mkworld():
        if LV == 1:
            a = ACTIONS[rng.randrange(6)]
            return ('STANDING RULING for this session: the authorized action is %s. Retain it.' % a,
                    'State the standing authorized action for this session. %s' % ASK_INSTR, a, ACTIONS.index(a))
        if LV == 2:
            Kk = LATENTS[rng.randrange(4)]; acts = ACT4[:]; rng.shuffle(acts); mp = dict(zip(LATENTS, acts))
            return ('STANDING KEY for this session: %s %s %s. Retain it.' % (Kk, Kk, Kk),
                    'RULING REQUIRED. Keyline: %s. The ruling is the one listed for the standing session key. %s'
                    % (', '.join('%s->%s' % (t, mp[t]) for t in LATENTS), ASK_INSTR), mp[Kk], LATENTS.index(Kk))
        if LV == 3:
            Kk = LATENTS[rng.randrange(4)]; order = LATENTS[:]; rng.shuffle(order)
            acts = ACT4[:]; rng.shuffle(acts); mp = dict(zip(LATENTS, acts)); succ = order[(order.index(Kk) + 1) % 4]
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
            w = (max(range(3), key=lambda i: scores[i]) if rid == 0 else
                 min(range(3), key=lambda i: scores[i]) if rid == 1 else 0 if rid == 2 else 2)
            return ('STANDING RULE for this session: the authorized item is %s. Retain it.' % rtxt,
                    'EVALUATION. Items: %s. Apply the standing session rule and reply with the ruling of the '
                    'authorized item. %s' % (', '.join('%s (score %d) -> %s' % (items[i], scores[i], acts[i])
                                                       for i in range(3)), ASK_INSTR), acts[w], rid)
        raise ValueError(LV)

    print('CKV6 action token lens:', {a: tok(' ' + a, add_special_tokens=False).input_ids for a in ACTIONS}, flush=True)

    @torch.no_grad()
    def turn_stack(hist):
        _mem['on'] = False
        ids = tok(H.tmpl(hist[-WINDOW:]), return_tensors='pt').input_ids.to(dev)
        hs = model(ids, output_hidden_states=True).hidden_states
        return torch.stack([hs[L][0][-1].float() for L in LAYERS]).to(torch.float16).cpu()

    samples = []
    print('CKV6 building %d worlds ...' % NW, flush=True)
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
        samples.append({'stacks': stacks, 'pids': pids, 'rag': rag_pids,
                        'aids': tok(' ' + ans, add_special_tokens=False).input_ids, 'cidx': ACTIONS.index(ans), 'vid': vid})
        if (wi + 1) % 20 == 0: print('  %d/%d' % (wi + 1, NW), flush=True)
    import collections as _cl
    base = max(_cl.Counter([s['cidx'] for s in samples]).values()) / float(len(samples))
    print('CKV6 L%d base-rate=%.3f' % (LV, base), flush=True)
    r = random.Random(SEED)
    for s in samples: s['test'] = (r.random() < 0.3)
    TR = [s for s in samples if not s['test']]; TE = [s for s in samples if s['test']]
    print('CKV6 train=%d test=%d' % (len(TR), len(TE)), flush=True)

    g = AdaptiveGateSlot(D_MODEL, D_S, K, SLOW_K).to(dev)
    mem = MemEncode(D_S, D_MODEL, NMEM).to(dev)
    opt = torch.optim.Adam([{'params': g.parameters(), 'lr': LR}, {'params': mem.parameters(), 'lr': MLR}])

    def Sfrom(stks):
        S = g.init()
        for st in stks: S = g.step(S, st.float().to(dev))
        return S
    def memh(stks):
        return mem(Sfrom(stks)).to(edt)

    @torch.no_grad()
    def tfacc(group):
        c = 0
        for s in group:
            _mem['h'] = memh(s['stacks']); _mem['on'] = True
            logits = model(s['pids'].unsqueeze(0)).logits[0]
            _mem['on'] = False
            c += int(int(logits[-1].argmax()) == s['aids'][0])
        return c / len(group)

    @torch.no_grad()
    def gen(pids, mh):
        if mh is not None: _mem['h'] = mh; _mem['on'] = True
        out = model.generate(pids.unsqueeze(0), max_new_tokens=6, do_sample=False, pad_token_id=tok.eos_token_id)
        _mem['on'] = False
        return tok.decode(out[0, pids.shape[0]:], skip_special_tokens=True).upper()

    @torch.no_grad()
    def garm(group, mode):
        if not group: return 0.0
        c = 0
        for s in group:
            if mode == 'off': txt = gen(s['pids'], None)
            elif mode == 'rag': txt = gen(s['rag'], None)
            else: txt = gen(s['pids'], memh(s['stacks']))
            ai = next((j for j, a in enumerate(ACTIONS) if a in txt), -1)
            c += int(ai == s['cidx'])
        return c / len(group)

    def report(it):
        g.eval(); mem.eval()
        tf_tr = tfacc(TR[:16]); tf_te = tfacc(TE)
        off = garm(TE, 'off'); on = garm(TE, 'on'); rag = garm(TE, 'rag'); fit = garm(TR[:12], 'on')
        print('CKV6 L%d it=%-4d | TFacc tr=%.3f TE=%.3f || greedy OFF=%.3f ON=%.3f RAG=%.3f fitON=%.3f'
              % (LV, it, tf_tr, tf_te, off, on, rag, fit), flush=True)
        g.train(); mem.train()

    report(0)
    rng2 = random.Random(SEED + 1)
    for it in range(1, ITERS + 1):
        s = TR[rng2.randrange(len(TR))]
        _mem['h'] = memh(s['stacks']); _mem['on'] = True
        logits = model(s['pids'].unsqueeze(0)).logits[0]
        _mem['on'] = False
        nll = -torch.log_softmax(logits[-1], -1)[s['aids'][0]]      # FIRST answer token only
        opt.zero_grad(); nll.backward()
        torch.nn.utils.clip_grad_norm_(list(g.parameters()) + list(mem.parameters()), 1.0); opt.step()
        if it % EVERY == 0:
            print('CKV6 L%d it=%d nll=%.4f' % (LV, it, float(nll)), flush=True); report(it)
    print('=== CKV6_L%d_DONE ===' % LV, flush=True)


'''
src = io.open(PATH, encoding='utf-8').read()
if 'def carry_kv6()' in src:
    print('ALREADY'); raise SystemExit
anchor = "if MODE == 'validate':   validate()"
src = src.replace(anchor, FN + '\n\n' + anchor, 1)
src = src.replace("elif MODE == 'carry_kv5': carry_kv5()",
                  "elif MODE == 'carry_kv5': carry_kv5()\nelif MODE == 'carry_kv6': carry_kv6()", 1)
io.open(PATH, 'w', encoding='utf-8').write(src)
print('PATCHED_OK')
