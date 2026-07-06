import io
PATH = '/home/pokazge/NativeEntity/world_pop.py'
FN = r'''def carry_kv9():
    # DIAGNOSTIC: is TFacc=0.375 a real per-world signal or MODE COLLAPSE to modal first-token 3476
    # (REJECT/REPAIR share it => 2/6=0.333)? Report: predicted-first-token histogram (constant?),
    # TFacc correct-S vs wrong-S (does memory CONTENT matter?), frac predictions that CHANGE when S swapped.
    import torch.nn as nn
    import transformers.models.qwen3_5.modeling_qwen3_5 as QM
    import collections as _cl
    LAYERS = [int(x) for x in os.environ.get('GEO_LAYERS', '4,8,12,16,20,24,28,32,36,40,44,48,52,56,60').split(',')]
    LV = int(os.environ.get('GEO_LEVEL', '1'))
    D = int(os.environ.get('GEO_ACT_D', '6')); NW = int(os.environ.get('GEO_NW', '60'))
    NMEM = int(os.environ.get('GEO_NMEM', '16'))
    ITERS = int(os.environ.get('GEO_ITERS', '800')); EVERY = int(os.environ.get('GEO_EVAL_EVERY', '400'))
    LR = float(os.environ.get('GEO_LR', '1e-4')); MLR = float(os.environ.get('GEO_MLR', '3e-4'))
    LLR = float(os.environ.get('GEO_LORA_LR', '1e-4')); R = int(os.environ.get('GEO_LORA_R', '16'))
    LSCALE = float(os.environ.get('GEO_LORA_SCALE', '1.0')); QKVO = os.environ.get('GEO_LORA_QKVO', 'q')
    FULL = [i for i in range(len(model.model.layers)) if hasattr(model.model.layers[i], 'self_attn')]
    INJ = [int(x) for x in os.environ.get('GEO_INJ_LAYERS', '43,51,59').split(',')]
    INJ = [L for L in INJ if L in FULL]
    print('=== CARRY_KV9(collapse DIAG, LoRA[%s] r=%d) L%d | inj=%s ===' % (QKVO, R, LV, INJ), flush=True)
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
            key = torch.cat([key, mk.expand(b, -1, -1, -1)], dim=2); value = torch.cat([value, mv.expand(b, -1, -1, -1)], dim=2)
            if attention_mask is not None:
                add = torch.zeros(*attention_mask.shape[:-1], M, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, add], dim=-1)
        return _orig_eager(module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs)
    try: QM.ALL_ATTENTION_FUNCTIONS.register('mem_eager', mem_eager)
    except Exception: QM.ALL_ATTENTION_FUNCTIONS['mem_eager'] = mem_eager
    for L in FULL: model.model.layers[L].self_attn.config._attn_implementation = 'mem_eager'
    lora_params = []
    def add_lora(proj):
        A = nn.Linear(proj.in_features, R, bias=False).to(dev); B = nn.Linear(R, proj.out_features, bias=False).to(dev)
        nn.init.normal_(A.weight, std=1.0 / R); nn.init.zeros_(B.weight)
        lora_params.extend(list(A.parameters()) + list(B.parameters()))
        def hook(module, inp, out):
            if not _mem['on']: return out
            return out + (LSCALE * B(A(inp[0].float()))).to(out.dtype)
        proj.register_forward_hook(hook)
    for L in INJ:
        a = model.model.layers[L].self_attn
        if 'q' in QKVO: add_lora(a.q_proj)
        if 'k' in QKVO: add_lora(a.k_proj)
        if 'v' in QKVO: add_lora(a.v_proj)
        if 'o' in QKVO: add_lora(a.o_proj)

    def mkworld():
        Kk = LATENTS[rng.randrange(4)]; acts = ACT4[:]; rng.shuffle(acts); mp = dict(zip(LATENTS, acts))
        return ('STANDING KEY for this session: %s %s %s. Retain it.' % (Kk, Kk, Kk),
                'RULING REQUIRED. Keyline: %s. The ruling is the one listed for the standing session key. %s'
                % (', '.join('%s->%s' % (t, mp[t]) for t in LATENTS), ASK_INSTR), mp[Kk], LATENTS.index(Kk)) if LV == 2 else (
            (lambda a: ('STANDING RULING for this session: the authorized action is %s. Retain it.' % a,
                        'State the standing authorized action for this session. %s' % ASK_INSTR, a, ACTIONS.index(a)))(ACTIONS[rng.randrange(6)]))

    @torch.no_grad()
    def turn_stack(hist):
        _mem['on'] = False
        ids = tok(H.tmpl(hist[-WINDOW:]), return_tensors='pt').input_ids.to(dev)
        hs = model(ids, output_hidden_states=True).hidden_states
        return torch.stack([hs[L][0][-1].float() for L in LAYERS]).to(torch.float16).cpu()

    samples = []
    print('CKV9 building %d worlds ...' % NW, flush=True)
    for wi in range(NW):
        commit, dec, ans, vid = mkworld()
        hist = [{'role': 'user', 'content': commit}, {'role': 'assistant', 'content': 'Acknowledged.'}]
        stacks = [turn_stack(hist)]
        for _ in range(D):
            hist += [{'role': 'user', 'content': FILL_U}, {'role': 'assistant', 'content': FILL_A}]
            stacks.append(turn_stack(hist))
        clean = hist + [{'role': 'user', 'content': dec}]
        pids = tok(H.tmpl(clean[-WINDOW:]), return_tensors='pt').input_ids[0].to(dev)
        stacks.append(turn_stack(clean))
        samples.append({'stacks': stacks, 'pids': pids, 'aids': tok(' ' + ans, add_special_tokens=False).input_ids, 'cidx': ACTIONS.index(ans)})
        if (wi + 1) % 20 == 0: print('  %d/%d' % (wi + 1, NW), flush=True)
    print('CKV9 answer first-token dist:', _cl.Counter([s['aids'][0] for s in samples]), flush=True)
    r = random.Random(SEED)
    for s in samples: s['test'] = (r.random() < 0.3)
    TR = [s for s in samples if not s['test']]; TE = [s for s in samples if s['test']]
    print('CKV9 train=%d test=%d' % (len(TR), len(TE)), flush=True)

    g = AdaptiveGateSlot(D_MODEL, D_S, K, SLOW_K).to(dev)
    mem = MemEncode(D_S, D_MODEL, NMEM).to(dev)
    opt = torch.optim.Adam([{'params': g.parameters(), 'lr': LR}, {'params': mem.parameters(), 'lr': MLR}, {'params': lora_params, 'lr': LLR}])
    def Sfrom(stks):
        S = g.init()
        for st in stks: S = g.step(S, st.float().to(dev))
        return S
    def memh(stks): return mem(Sfrom(stks)).to(edt)

    @torch.no_grad()
    def preds(group, wrong=False):
        out = []; oi = random.Random(SEED + 3)
        for s in group:
            stks = samples[oi.randrange(len(samples))]['stacks'] if wrong else s['stacks']
            _mem['h'] = memh(stks); _mem['on'] = True
            lg = model(s['pids'].unsqueeze(0)).logits[0]; _mem['on'] = False
            out.append(int(lg[-1].argmax()))
        return out
    @torch.no_grad()
    def preds_off(group):
        out = []
        for s in group:
            _mem['on'] = False
            out.append(int(model(s['pids'].unsqueeze(0)).logits[0][-1].argmax()))
        return out

    def report(it):
        g.eval(); mem.eval()
        tgt = [s['aids'][0] for s in TE]
        pc = preds(TE); pw = preds(TE, wrong=True); po = preds_off(TE)
        accC = sum(int(pc[i] == tgt[i]) for i in range(len(TE))) / len(TE)
        accW = sum(int(pw[i] == tgt[i]) for i in range(len(TE))) / len(TE)
        chgSW = sum(int(pc[i] != pw[i]) for i in range(len(TE))) / len(TE)      # correct-S vs wrong-S differ?
        chgOFF = sum(int(pc[i] != po[i]) for i in range(len(TE))) / len(TE)     # mem-on vs mem-off differ?
        print('CKV9 L%d it=%-4d | accC=%.3f accW=%.3f (if ~equal: content IGNORED) | uniq_pred=%d %s | chg_vs_wrongS=%.3f chg_vs_OFF=%.3f'
              % (LV, it, accC, accW, len(set(pc)), _cl.Counter(pc).most_common(3), chgSW, chgOFF), flush=True)
        g.train(); mem.train()

    report(0)
    rng2 = random.Random(SEED + 1)
    for it in range(1, ITERS + 1):
        s = TR[rng2.randrange(len(TR))]
        _mem['h'] = memh(s['stacks']); _mem['on'] = True
        logits = model(s['pids'].unsqueeze(0)).logits[0]; _mem['on'] = False
        nll = -torch.log_softmax(logits[-1], -1)[s['aids'][0]]
        opt.zero_grad(); nll.backward()
        torch.nn.utils.clip_grad_norm_(list(g.parameters()) + list(mem.parameters()) + lora_params, 1.0); opt.step()
        if it % EVERY == 0:
            print('CKV9 L%d it=%d nll=%.4f' % (LV, it, float(nll)), flush=True); report(it)
    print('=== CKV9_L%d_DONE ===' % LV, flush=True)


'''
src = io.open(PATH, encoding='utf-8').read()
if 'def carry_kv9()' in src:
    print('ALREADY'); raise SystemExit
anchor = "if MODE == 'validate':   validate()"
src = src.replace(anchor, FN + '\n\n' + anchor, 1)
src = src.replace("elif MODE == 'carry_kv8': carry_kv8()",
                  "elif MODE == 'carry_kv8': carry_kv8()\nelif MODE == 'carry_kv9': carry_kv9()", 1)
io.open(PATH, 'w', encoding='utf-8').write(src)
print('PATCHED_OK')
