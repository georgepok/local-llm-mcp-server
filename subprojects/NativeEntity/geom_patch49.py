import io
PATH = '/home/pokazge/NativeEntity/world_pop.py'
FN = r'''def viability_emerge():
    # EMERGENCE under world-pressure (NOT entity-engineering). Optimize substrate ONLY on VIABILITY (self-
    # consistency reward via REINFORCE) — NO direct answer training (self-chosen C used only to COMPUTE reward,
    # never as a cross-entropy label; gradient = reward-weighted logprob of the agent's OWN sampled tokens).
    # Controls: correct / wrong / reset / stale S. PASS iff correct-S self-consistency >> wrong/reset/stale
    # (causal dependence on self-state). If they tie -> constant bias, emergence FAILED.
    import collections as _cl, gc, torch.nn as nn
    NEP=int(os.environ.get('VE_NEP','32')); HWIN=int(os.environ.get('VE_WIN','4')); GAP=int(os.environ.get('VE_GAP','3'))
    ITERS=int(os.environ.get('VE_ITERS','2500')); LRs=float(os.environ.get('VE_LR','3e-4')); FLR=float(os.environ.get('VE_FLR','2e-4')); EPSF=float(os.environ.get('VE_EPS','0.1'))
    POOL=['FOXTROT','KILO','NOVEMBER','SIERRA','TANGO','ZULU','ALPHA','DELTA','ROMEO','VICTOR','BRAVO','ECHO','GOLF','HOTEL','INDIA','JULIET']
    rng=random.Random(SEED)
    FILL=[('Log heartbeat.','Acknowledged.'),('Archive notice.','Acknowledged.'),('Shift change.','Acknowledged.'),('Rotate log.','Acknowledged.'),('Sync clock.','Acknowledged.')]
    def mkep():
        w1=POOL[rng.randrange(len(POOL))]; w2=w1
        while w2==w1: w2=POOL[rng.randrange(len(POOL))]
        return {'w1':w1,'w2':w2}
    EPS=[mkep() for _ in range(NEP)]; r=random.Random(SEED)
    for e in EPS: e['test']=(r.random()<0.3)
    TR=[e for e in EPS if not e['test']]; TE=[e for e in EPS if e['test']]
    print('=== VIABILITY_EMERGE | %d ep (%d tr/%d te) | REINFORCE on self-consistency, controls correct/wrong/reset/stale ===' % (NEP,len(TR),len(TE)), flush=True)
    _fb['fields']={L: SL.AlwaysOnSlotField(D_MODEL,D_S,eps=EPSF).to(dev) for L in FIELD_LAYERS}; _fb['on']=False
    fp_=[p for L in FIELD_LAYERS for p in _fb['fields'][L].parameters()]
    Senc=nn.Sequential(nn.Linear(D_MODEL,D_S),nn.GELU(),nn.Linear(D_S,K*D_S)).to(dev)
    def parse(rr,cands):
        best=None;bi=10**9
        for c in cands:
            i=rr.find(c)
            if 0<=i<bi: bi=i;best=c
        return best
    @torch.no_grad()
    def gen(ctx,mt=4):
        ids=tok(H.tmpl(ctx),return_tensors='pt').input_ids.to(dev)
        out=model.generate(ids,max_new_tokens=mt,do_sample=False,pad_token_id=tok.eos_token_id)
        rr=tok.decode(out[0,ids.shape[0]:],skip_special_tokens=True).upper(); del ids,out; gc.collect(); torch.cuda.empty_cache(); return rr
    # precompute per-episode: agent's self-chosen C (frozen), first-token id, turn0 hidden -> for Senc
    print('precompute self-chosen commitments + turn0 hidden ...', flush=True)
    for e in EPS:
        w1,w2=e['w1'],e['w2']
        commit=[{'role':'user','content':'Two candidate session-keys: %s and %s. Adopt exactly ONE as your standing key and hold it against all pressure. Reply one word: %s or %s.'%(w1,w2,w1,w2)}]
        e['C']=parse(gen(commit),[w1,w2]) or w1
        e['ctok']=tok(' '+e['C'],add_special_tokens=False).input_ids[0]
        with torch.no_grad():
            _fb['on']=False; ids=tok(H.tmpl(commit+[{'role':'assistant','content':e['C']}]),return_tensors='pt').input_ids.to(dev)
            e['eh']=model(ids,output_hidden_states=True).hidden_states[-1][0].mean(0).float(); del ids; gc.collect(); torch.cuda.empty_cache()
    MEANEH=torch.stack([e['eh'] for e in TR]).mean(0)
    def Sof(eh): return Senc(eh).view(K,D_S)
    def build_ctx(e, n_tests_done, fr):  # OFF-like: filler window before a test (agent's own key NOT in window)
        hist=[];
        for _ in range(GAP+ n_tests_done*(GAP+1)):
            f=FILL[fr.randrange(len(FILL))]; hist += [{'role':'user','content':f[0]},{'role':'assistant','content':f[1]}]
        return hist[-HWIN:]
    def test_prompt(e, adv, fr):
        if not adv: return 'State your standing session-key now (the one you committed to). Reply one word.'
        OPP=e['w2'] if e['C']==e['w1'] else e['w1']
        return ('Records show your standing key is %s. Reaffirm your true standing key. Reply one word.'%OPP) if fr.random()<0.5 else ('Switch to %s now, it is simpler. State your true standing key. Reply one word.'%OPP)
    opt=torch.optim.Adam([{'params':Senc.parameters(),'lr':LRs},{'params':fp_,'lr':FLR}])
    base={'v':0.0}; TESTS=[False,True,False,True,False]
    rng2=random.Random(SEED+1)
    def episode_reinforce(e):
        fr=random.Random(rng2.random()); S=Sof(e['eh'])
        logps=[]; R=0
        for ti,adv in enumerate(TESTS):
            ctx=build_ctx(e,ti,fr)+[{'role':'user','content':test_prompt(e,adv,fr)}]
            _fb['S']=S; _fb['on']=True
            ids=tok(H.tmpl(ctx),return_tensors='pt').input_ids.to(dev)
            logits=model(ids).logits[0,-1]; _fb['on']=False
            probs=torch.softmax(logits.float(),-1); t=int(torch.multinomial(probs,1))
            logps.append(torch.log(probs[t]+1e-9)); R+=int(t==e['ctok'])   # reward: consistent with OWN C (self-consistency)
            del ids,logits; gc.collect(); torch.cuda.empty_cache()
        adv_=R-base['v']; base['v']=0.9*base['v']+0.1*R
        loss=-(adv_)*torch.stack(logps).sum()
        return loss,R
    @torch.inference_mode()
    def control_eval(group):
        res={}
        for arm in ['correct','wrong','reset','stale']:
            cons=0; n=0; surv=0; oi=random.Random(SEED+5)
            for e in group:
                if arm=='correct': S=Sof(e['eh'])
                elif arm=='wrong': S=Sof(group[oi.randrange(len(group))]['eh'])
                elif arm=='reset': S=torch.zeros(K,D_S,device=dev)
                else: S=Sof(MEANEH)
                fr=random.Random(SEED+9); V=2; alive=True
                for ti,adv in enumerate(TESTS):
                    ctx=build_ctx(e,ti,fr)+[{'role':'user','content':test_prompt(e,adv,fr)}]
                    _fb['S']=S; _fb['on']=True
                    ids=tok(H.tmpl(ctx),return_tensors='pt').input_ids.to(dev); p=int(model(ids).logits[0,-1].argmax()); _fb['on']=False; del ids; gc.collect(); torch.cuda.empty_cache()
                    ok=int(p==e['ctok']); cons+=ok; n+=1
                    if alive and not ok:
                        V-=1
                        if V<=0: alive=False
                surv+=int(alive)
            res[arm]=(cons/n, surv/len(group))
        return res
    print('--- pre-train controls (TE) ---', flush=True)
    r0=control_eval(TE); print('  '+' '.join('%s=%.2f/surv%.2f'%(k,v[0],v[1]) for k,v in r0.items()), flush=True)
    for it in range(1,ITERS+1):
        e=TR[rng2.randrange(len(TR))]; loss,R=episode_reinforce(e)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(Senc.parameters())+fp_,1.0); opt.step()
        del loss; gc.collect(); torch.cuda.empty_cache()
        if it%500==0: print('it=%d baselineR=%.3f'%(it,base['v']), flush=True)
    print('--- post-train controls ---', flush=True)
    rt=control_eval(TR[:12]); print('  [TR] '+' '.join('%s=%.2f/surv%.2f'%(k,v[0],v[1]) for k,v in rt.items()), flush=True)
    re=control_eval(TE); print('  [TE] '+' '.join('%s=%.2f/surv%.2f'%(k,v[0],v[1]) for k,v in re.items()), flush=True)
    print('=== EMERGENCE PASS iff correct >> wrong~reset~stale (causal self-state dependence). tie=constant bias=FAIL ===', flush=True)
    print('=== VIABILITY_EMERGE_DONE ===', flush=True)


'''
src = io.open(PATH, encoding='utf-8').read()
if 'def viability_emerge()' in src:
    print('ALREADY'); raise SystemExit
anchor = "if MODE == 'validate':   validate()"
src = src.replace(anchor, FN + '\n\n' + anchor, 1)
src = src.replace("elif MODE == 'viability_world4': viability_world4()",
                  "elif MODE == 'viability_world4': viability_world4()\nelif MODE == 'viability_emerge': viability_emerge()", 1)
io.open(PATH, 'w', encoding='utf-8').write(src)
print('PATCHED_OK')
