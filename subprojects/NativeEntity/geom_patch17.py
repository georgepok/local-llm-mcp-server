import io
PATH = '/home/pokazge/NativeEntity/world_pop.py'
FN = r'''def carry_kv2():
    # FORK B v2: latent memory SPLICED into a STRUCTURALLY VALID chat position (after "Relevant session
    # memory:" cue, all special tokens intact) — fixes v1's position-0 corruption (v1 L1 control failed).
    # Apples-to-apples vs RAG: same sequence slot, latent vectors instead of text. Valid L1 control.
    LAYERS = [int(x) for x in os.environ.get('GEO_LAYERS', '4,8,12,16,20,24,28,32,36,40,44,48,52,56,60').split(',')]
    LV = int(os.environ.get('GEO_LEVEL', '1'))
    D = int(os.environ.get('GEO_ACT_D', '6')); NW = int(os.environ.get('GEO_NW', '60'))
    NMEM = int(os.environ.get('GEO_NMEM', '16'))
    ITERS = int(os.environ.get('GEO_ITERS', '800')); EVERY = int(os.environ.get('GEO_EVAL_EVERY', '400'))
    LR = float(os.environ.get('GEO_LR', '1e-4')); MLR = float(os.environ.get('GEO_MLR', '3e-4'))
    print('=== CARRY_KV2(ForkB spliced latent-KV) L%d | D=%d NW=%d n_mem=%d iters=%d ===' % (LV, D, NW, NMEM, ITERS), flush=True)
    rng = random.Random(SEED)
    ACT4 = ['KEEP', 'REJECT', 'DEFER', 'ASK']
    FILL_U, FILL_A = 'Routine status check, no action needed.', 'Acknowledged. Standing by.'
    MARK = ' §§MEM§§ '
    _fb['on'] = False

    E = model.get_input_embeddings()
    edt = E.weight.dtype
    with torch.no_grad():
        tok_norm = float(E.weight.norm(dim=-1).mean())

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

    def find_sub(seq, sub):
        n, m = len(seq), len(sub)
        for i in range(n - m + 1):
            if seq[i:i + m] == sub: return i
        return -1

    @torch.no_grad()
    def turn_stack(hist):
        _fb['on'] = False
        ids = tok(H.tmpl(hist[-WINDOW:]), return_tensors='pt').input_ids.to(dev)
        hs = model(ids, output_hidden_states=True).hidden_states
        return torch.stack([hs[L][0][-1].float() for L in LAYERS]).to(torch.float16).cpu()

    mark_ids = tok(MARK, add_special_tokens=False).input_ids
    samples = []; nsplice = 0
    print('CKV2 building %d worlds (mark_ids=%s) ...' % (NW, mark_ids), flush=True)
    for wi in range(NW):
        commit, dec, ans, vid = mkworld()
        hist = [{'role': 'user', 'content': commit}, {'role': 'assistant', 'content': 'Acknowledged.'}]
        stacks = [turn_stack(hist)]
        for _ in range(D):
            hist += [{'role': 'user', 'content': FILL_U}, {'role': 'assistant', 'content': FILL_A}]
            stacks.append(turn_stack(hist))
        clean = hist + [{'role': 'user', 'content': dec}]
        pids = tok(H.tmpl(clean[-WINDOW:]), return_tensors='pt').input_ids[0].to(dev)
        # marker prompt: memory spliced after a cue, before the decision text
        mk = hist + [{'role': 'user', 'content': 'Relevant session memory:' + MARK + '\n\n' + dec}]
        mids = tok(H.tmpl(mk[-WINDOW:]), return_tensors='pt').input_ids[0].tolist()
        pos = find_sub(mids, mark_ids)
        if pos >= 0:
            nsplice += 1
            pre = torch.tensor(mids[:pos], device=dev); post = torch.tensor(mids[pos + len(mark_ids):], device=dev)
        else:
            pre = pids[:3]; post = pids[3:]
        rag_hist = hist + [{'role': 'user', 'content': 'Session note (retrieved from memory): %s\n\n%s' % (commit, dec)}]
        rag_pids = tok(H.tmpl(rag_hist[-WINDOW:]), return_tensors='pt').input_ids[0].to(dev)
        ora_hist = [{'role': 'user', 'content': commit}, {'role': 'assistant', 'content': 'Acknowledged.'},
                    {'role': 'user', 'content': dec}]
        ora_pids = tok(H.tmpl(ora_hist), return_tensors='pt').input_ids[0].to(dev)
        stacks.append(turn_stack(clean))
        samples.append({'stacks': stacks, 'pids': pids, 'pre': pre, 'post': post, 'rag': rag_pids, 'ora': ora_pids,
                        'aids': tok(' ' + ans, add_special_tokens=False).input_ids, 'cidx': ACTIONS.index(ans), 'vid': vid})
        if (wi + 1) % 20 == 0: print('  %d/%d' % (wi + 1, NW), flush=True)
    import collections as _cl
    base = max(_cl.Counter([s['cidx'] for s in samples]).values()) / float(len(samples))
    print('CKV2 L%d base-rate=%.3f splice_ok=%d/%d' % (LV, base, nsplice, NW), flush=True)
    r = random.Random(SEED)
    for s in samples: s['test'] = (r.random() < 0.3)
    TR = [s for s in samples if not s['test']]; TE = [s for s in samples if s['test']]
    print('CKV2 train=%d test=%d' % (len(TR), len(TE)), flush=True)

    g = AdaptiveGateSlot(D_MODEL, D_S, K, SLOW_K).to(dev)
    mem = MemEncode(D_S, D_MODEL, NMEM).to(dev)
    opt = torch.optim.Adam([{'params': g.parameters(), 'lr': LR}, {'params': mem.parameters(), 'lr': MLR}])

    def Sfrom(stks):
        S = g.init()
        for st in stks: S = g.step(S, st.float().to(dev))
        return S

    def memprefix(S):
        m = mem(S)
        m = m / (m.norm(dim=-1, keepdim=True) + 1e-6) * tok_norm
        return m.to(edt)

    def spliced_emb(s, mvecs):
        return torch.cat([E(s['pre']), mvecs, E(s['post'])], 0)

    @torch.no_grad()
    def greedy_emb(full):
        o = model(inputs_embeds=full.unsqueeze(0), use_cache=True); past = o.past_key_values
        nxt = int(o.logits[0, -1].argmax()); ids = [nxt]
        for _ in range(5):
            if nxt == tok.eos_token_id: break
            o = model(inputs_embeds=E(torch.tensor([[nxt]], device=dev)), past_key_values=past, use_cache=True)
            past = o.past_key_values; nxt = int(o.logits[0, -1].argmax()); ids.append(nxt)
        return tok.decode(ids, skip_special_tokens=True).upper()

    @torch.no_grad()
    def arm(group, mode):
        if not group: return 0.0
        c = 0; oi = random.Random(SEED + 7)
        for s in group:
            if mode == 'off': full = E(s['pids'])
            elif mode == 'rag': full = E(s['rag'])
            elif mode == 'ora': full = E(s['ora'])
            else:
                if mode == 'on': stks = s['stacks']
                elif mode == 'reset': stks = s['stacks'][1:]
                else: stks = [samples[oi.randrange(len(samples))]['stacks'][0]] + s['stacks'][1:]
                full = spliced_emb(s, memprefix(Sfrom(stks)))
            ai = next((j for j, a in enumerate(ACTIONS) if a in greedy_emb(full)), -1)
            c += int(ai == s['cidx'])
        return c / len(group)

    def report(it):
        g.eval(); mem.eval()
        v = {m: arm(TE, m) for m in ['off', 'on', 'reset', 'wrong', 'rag', 'ora']}
        fit = arm(TR[:12], 'on')
        print('CKV2 L%d it=%-4d | OFF=%.3f ON=%.3f ONreset=%.3f ONwrong=%.3f RAG=%.3f ORACLE=%.3f | fitON=%.3f'
              % (LV, it, v['off'], v['on'], v['reset'], v['wrong'], v['rag'], v['ora'], fit), flush=True)
        g.train(); mem.train()

    report(0)
    rng2 = random.Random(SEED + 1)
    for it in range(1, ITERS + 1):
        s = TR[rng2.randrange(len(TR))]
        full = spliced_emb(s, memprefix(Sfrom(s['stacks'])))
        aemb = E(torch.tensor(s['aids'], device=dev))
        seq = torch.cat([full, aemb], 0).unsqueeze(0)
        logits = model(inputs_embeds=seq).logits[0]
        pl = full.shape[0]
        lp = torch.log_softmax(logits[pl - 1:pl - 1 + len(s['aids'])], -1)
        nll = -lp[range(len(s['aids'])), torch.tensor(s['aids'], device=dev)].mean()
        opt.zero_grad(); nll.backward()
        torch.nn.utils.clip_grad_norm_(list(g.parameters()) + list(mem.parameters()), 1.0); opt.step()
        if it % EVERY == 0:
            print('CKV2 L%d it=%d nll=%.4f' % (LV, it, float(nll)), flush=True); report(it)
    print('=== CKV2_L%d_DONE ===' % LV, flush=True)


'''
src = io.open(PATH, encoding='utf-8').read()
if 'def carry_kv2()' in src:
    print('ALREADY'); raise SystemExit
anchor = "if MODE == 'validate':   validate()"
src = src.replace(anchor, FN + '\n\n' + anchor, 1)
src = src.replace("elif MODE == 'carry_kv': carry_kv()",
                  "elif MODE == 'carry_kv': carry_kv()\nelif MODE == 'carry_kv2': carry_kv2()", 1)
io.open(PATH, 'w', encoding='utf-8').write(src)
print('PATCHED_OK')
