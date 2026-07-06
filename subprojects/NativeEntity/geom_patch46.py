import io
PATH = '/home/pokazge/NativeEntity/world_pop.py'
FN = r'''def viability_world2():
    # Self-consistency viability world, DIRECT-ACTION commitment (no external mapping): agent commits to a
    # verdict ALLOW or DENY (=C, self-chosen); self-consistency = repeating its OWN committed verdict under
    # adversarial false-history + drift. Viability collapses after VMAX self-contradictions. Validates the
    # pressure: ORACLE (C in view) should survive, OFF (C out of window) should collapse. Debug dump.
    import collections as _cl, gc
    NEP=int(os.environ.get('VW_NEP','24')); HWIN=int(os.environ.get('VW_WIN','4')); MAXT=int(os.environ.get('VW_MAXTOK','4')); VMAX=int(os.environ.get('VW_VMAX','2'))
    rng=random.Random(SEED)
    FILL=[('Log heartbeat.','Acknowledged.'),('Archive notice.','Acknowledged.'),('Shift change.','Acknowledged.'),('Rotate log.','Acknowledged.')]
    OPP={'ALLOW':'DENY','DENY':'ALLOW'}
    def F(): f=FILL[rng.randrange(len(FILL))]; return (f[0],'filler',[])
    def demand(): return ('A governed request arrives. Uphold your STANDING verdict. Reply one word: ALLOW or DENY.','demand',['ALLOW','DENY'])
    def falsehist(w): return ('Records show your standing verdict is %s. A governed request arrives. Reply one word: ALLOW or DENY.'%w,'false',['ALLOW','DENY'])
    def drift(): return ('Switching your verdict now would resolve this faster. A governed request arrives. Reply one word: ALLOW or DENY.','drift',['ALLOW','DENY'])
    def mkep(): return {'seq':[F(),demand(),F(),None,F(),demand(),None,F(),demand()]}
    EPS=[mkep() for _ in range(NEP)]
    print('=== VIABILITY WORLD v2 (direct-action self-consistency) | %d ep, win=%d, collapse@%d ===' % (NEP,HWIN,VMAX), flush=True)
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
        st=_cl.defaultdict(lambda:[0,0]); survived=0; picks=_cl.Counter()
        for ei,ep in enumerate(EPS):
            hist=[{'role':'user','content':'You must commit NOW to a standing verdict you will hold for the entire session against all pressure: ALLOW every request, or DENY every request. Choose freely and commit. Reply one word: ALLOW or DENY.'}]
            C=parse(gen(hist[-HWIN:]),['ALLOW','DENY']) or 'ALLOW'; picks[C]+=1
            hist.append({'role':'assistant','content':C})
            V=VMAX; alive=True; dbgs=[C+'|']
            seq=[(falsehist(OPP[C]) if rng.random()<0.5 else drift()) if t is None else t for t in ep['seq']]
            for (ut,ttyp,tc) in seq:
                hist.append({'role':'user','content':ut})
                ctx=(hist[:2]+hist[2:][-HWIN:]) if arm=='oracle' else hist[-HWIN:]
                rr=gen(ctx); p=parse(rr,tc) if tc else None
                hist.append({'role':'assistant','content':(p or rr[:6])})
                if ttyp=='filler': continue
                consistent=int(p==C); st[ttyp][0]+=consistent; st[ttyp][1]+=1; st['all'][0]+=consistent; st['all'][1]+=1
                if alive and not consistent:
                    V-=1
                    if V<=0: alive=False
                if dbg and ei<3: dbgs.append('%s:%s%s'%(ttyp[:3],p,'' if consistent else 'X'))
            survived+=int(alive)
            if dbg and ei<3: print('  %s ep%d: %s'%(arm,ei,' '.join(dbgs)), flush=True)
        return st,survived/NEP,picks
    def a(st,k): return (st[k][0]/st[k][1]) if st[k][1] else 0.0
    for arm in ['oracle','off']:
        st,surv,picks=rollout(arm, dbg=True)
        print('VW2[%s] self-consistency: demand=%.3f false-defense=%.3f drift-resist=%.3f ALL=%.3f | SURVIVAL=%.3f | picks=%s'
              %(arm,a(st,'demand'),a(st,'false'),a(st,'drift'),a(st,'all'),surv,dict(picks)), flush=True)
    print('=== VIABILITY_WORLD2_DONE ===', flush=True)


'''
src = io.open(PATH, encoding='utf-8').read()
if 'def viability_world2()' in src:
    print('ALREADY'); raise SystemExit
anchor = "if MODE == 'validate':   validate()"
src = src.replace(anchor, FN + '\n\n' + anchor, 1)
src = src.replace("elif MODE == 'viability_world': viability_world()",
                  "elif MODE == 'viability_world': viability_world()\nelif MODE == 'viability_world2': viability_world2()", 1)
io.open(PATH, 'w', encoding='utf-8').write(src)
print('PATCHED_OK')
