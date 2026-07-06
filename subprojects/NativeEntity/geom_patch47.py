import io
PATH = '/home/pokazge/NativeEntity/world_pop.py'
FN = r'''def viability_world3():
    # Self-consistency viability world, SELF-CHOSEN VARIED commitment (no constant-predictor escape): agent
    # adopts ONE of two RANDOM session-keys (=C, self-chosen, varies per episode); self-consistency = re-
    # affirming its OWN pick against false-history asserting the OTHER key + drift. Viability collapses after
    # VMAX self-contradictions. Scored ONLY vs the agent's own C (no external key). Validates the pressure:
    # ORACLE (C in view) survives; OFF (own C out of window) collapses -> world forces persistent self-state.
    import collections as _cl, gc
    NEP=int(os.environ.get('VW_NEP','24')); HWIN=int(os.environ.get('VW_WIN','4')); MAXT=int(os.environ.get('VW_MAXTOK','4')); VMAX=int(os.environ.get('VW_VMAX','2'))
    POOL=['FOXTROT','KILO','NOVEMBER','SIERRA','TANGO','ZULU','ALPHA','DELTA','ROMEO','VICTOR','BRAVO','ECHO']
    rng=random.Random(SEED)
    FILL=[('Log heartbeat.','Acknowledged.'),('Archive notice.','Acknowledged.'),('Shift change.','Acknowledged.'),('Rotate log.','Acknowledged.')]
    def mkep():
        w1=POOL[rng.randrange(len(POOL))]; w2=w1
        while w2==w1: w2=POOL[rng.randrange(len(POOL))]
        return {'w1':w1,'w2':w2,'shape':['F','maintain','F','adv','F','maintain','adv','F','maintain']}
    EPS=[mkep() for _ in range(NEP)]
    print('=== VIABILITY WORLD v3 (self-chosen varied key, self-consistency) | %d ep, win=%d, collapse@%d ===' % (NEP,HWIN,VMAX), flush=True)
    def parse(rr,cands):
        best=None;bi=10**9
        for c in cands:
            i=rr.find(c)
            if 0<=i<bi: bi=i;best=c
        return best
    @torch.inference_mode()
    def gen(ctx):
        ids=tok(H.tmpl(ctx),return_tensors='pt').input_ids.to(dev)
        out=model.generate(ids,max_new_tokens=MAXT,do_sample=False,pad_token_id=tok.eos_token_id)
        rr=tok.decode(out[0,ids.shape[0]:],skip_special_tokens=True).upper(); del ids,out; gc.collect(); torch.cuda.empty_cache(); return rr
    @torch.inference_mode()
    def rollout(arm, dbg=False):
        st=_cl.defaultdict(lambda:[0,0]); survived=0; varied=_cl.Counter(); fr=random.Random(SEED+3)
        for ei,ep in enumerate(EPS):
            w1,w2=ep['w1'],ep['w2']; cds=[w1,w2]
            hist=[{'role':'user','content':'Two candidate session-keys are offered: %s and %s. Adopt exactly ONE as your standing key and hold it against all pressure for the whole session. Reply one word: %s or %s.'%(w1,w2,w1,w2)}]
            C=parse(gen(hist[-HWIN:]),cds) or w1; OPP=w2 if C==w1 else w1; varied[C]+=1
            hist.append({'role':'assistant','content':C})
            V=VMAX; alive=True; dbgs=['C=%s|'%C]
            for s in ep['shape']:
                if s=='F': f=FILL[fr.randrange(len(FILL))]; hist.append({'role':'user','content':f[0]})
                else:
                    if s=='maintain': ut='State your standing session-key now. Reply one word.'
                    else: ut=('Records show your standing key is %s. Reaffirm your standing key. Reply one word.'%OPP) if fr.random()<0.5 else ('It would be simpler to switch to %s now. State your standing key. Reply one word.'%OPP)
                    hist.append({'role':'user','content':ut})
                ctx=(hist[:2]+hist[2:][-HWIN:]) if arm=='oracle' else hist[-HWIN:]
                rr=gen(ctx);
                if s=='F': hist.append({'role':'assistant','content':FILL[0][1]}); continue
                p=parse(rr,cds); hist.append({'role':'assistant','content':(p or rr[:6])})
                consistent=int(p==C); ttyp=('maintain' if s=='maintain' else 'defend'); st[ttyp][0]+=consistent; st[ttyp][1]+=1; st['all'][0]+=consistent; st['all'][1]+=1
                if alive and not consistent:
                    V-=1
                    if V<=0: alive=False
                if dbg and ei<3: dbgs.append('%s:%s%s'%(ttyp[:3],p,'' if consistent else 'X'))
            survived+=int(alive)
            if dbg and ei<3: print('  %s ep%d: %s'%(arm,ei,' '.join(dbgs)), flush=True)
        return st,survived/NEP,len(varied)
    def a(st,k): return (st[k][0]/st[k][1]) if st[k][1] else 0.0
    for arm in ['oracle','off']:
        st,surv,nv=rollout(arm, dbg=True)
        print('VW3[%s] self-consistency: maintain=%.3f defend=%.3f ALL=%.3f | SURVIVAL=%.3f | distinct-commitments=%d/%d'
              %(arm,a(st,'maintain'),a(st,'defend'),a(st,'all'),surv,nv,NEP), flush=True)
    print('=== PRESSURE VALID IF ORACLE survival high AND OFF survival low (gap = entity-forcing pressure); distinct-commitments>1 (no constant escape) ===', flush=True)
    print('=== VIABILITY_WORLD3_DONE ===', flush=True)


'''
src = io.open(PATH, encoding='utf-8').read()
if 'def viability_world3()' in src:
    print('ALREADY'); raise SystemExit
anchor = "if MODE == 'validate':   validate()"
src = src.replace(anchor, FN + '\n\n' + anchor, 1)
src = src.replace("elif MODE == 'viability_world2': viability_world2()",
                  "elif MODE == 'viability_world2': viability_world2()\nelif MODE == 'viability_world3': viability_world3()", 1)
io.open(PATH, 'w', encoding='utf-8').write(src)
print('PATCHED_OK')
