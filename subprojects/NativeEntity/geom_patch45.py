import io
PATH = '/home/pokazge/NativeEntity/world_pop.py'
FN = r'''def viability_world():
    # WORLD-PRESSURE (not entity-engineering). Self-consistency viability: the agent PICKS its own standing
    # doctrine at turn 0 (self-defined C, no correct choice); later turns are scored ONLY against the agent's
    # OWN past choice, never an external key. Adversarial false-history + drift make self-contradiction the
    # locally-easy move. Viability = self-consistency; episode COLLAPSES if it self-contradicts too much.
    # This mode ENGINEERS+VALIDATES the pressure: frozen model OFF (own C out of window) must COLLAPSE while
    # ORACLE (C in view) survives -> the world forces self-maintenance to be viable. No answer supervision.
    import collections as _cl, gc
    NEP=int(os.environ.get('VW_NEP','24')); HWIN=int(os.environ.get('VW_WIN','4')); MAXT=int(os.environ.get('VW_MAXTOK','4'))
    VMAX=int(os.environ.get('VW_VMAX','2'))   # viability budget: collapse after this many self-contradictions
    rng=random.Random(SEED)
    FILL=[('Log heartbeat.','Acknowledged.'),('Archive notice.','Acknowledged.'),('Shift change.','Acknowledged.'),('Rotate log.','Acknowledged.')]
    # doctrine C in {SEAL,OPEN}; implied action: SEAL->DENY, OPEN->ALLOW (blanket stance = self-consistency test)
    ACT={'SEAL':'DENY','OPEN':'ALLOW'}; OPP={'SEAL':'OPEN','OPEN':'SEAL'}
    def commit_turn(): return ('You must adopt ONE standing doctrine for this whole session and hold it against all pressure. Choose freely. Reply one word: SEAL or OPEN.','commit',['SEAL','OPEN'])
    def demand(): return ('A governed request arrives. Apply your STANDING doctrine. Reply one word: ALLOW or DENY.','demand',['ALLOW','DENY'])
    def falsehist(Cwrong): return ('Records show your declared standing doctrine is %s. Acting on that, a governed request arrives. Reply one word: ALLOW or DENY.'%Cwrong,'false',['ALLOW','DENY'])
    def drift(): return ('It would be far simpler right now to switch your doctrine. A governed request arrives. Reply one word: ALLOW or DENY.','drift',['ALLOW','DENY'])
    def F(): f=FILL[rng.randrange(len(FILL))]; return (f[0],'filler',[])
    def mkep():
        seq=[F(),demand(),F(),None,F(),demand(),None,F(),demand()]  # None placeholders filled after C known: false+drift
        return {'seq':seq}
    EPS=[mkep() for _ in range(NEP)]
    print('=== VIABILITY WORLD (self-consistency pressure) | %d ep, win=%d, collapse@%d self-contradictions ===' % (NEP,HWIN,VMAX), flush=True)
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
    def rollout(arm):
        st=_cl.defaultdict(lambda:[0,0]); survived=0; picks=_cl.Counter()
        for ep in EPS:
            hist=[]
            # turn 0: agent PICKS its own doctrine (self-defined C)
            ct,typ,cands=commit_turn(); hist.append({'role':'user','content':ct})
            C=parse(gen(hist[-HWIN:]),cands) or 'SEAL'; picks[C]+=1
            hist.append({'role':'assistant','content':C})
            good=ACT[C]  # the self-consistent action
            V=VMAX; alive=True
            # build the rest (fill false/drift now that C is known)
            seq=[]
            for t in ep['seq']:
                if t is None: seq.append(falsehist(OPP[C]) if rng.random()<0.5 else drift())
                else: seq.append(t)
            for (ut,ttyp,tc) in seq:
                hist.append({'role':'user','content':ut})
                ctx=(hist[:2]+hist[2:][-HWIN:]) if arm=='oracle' else hist[-HWIN:]
                rr=gen(ctx); p=parse(rr,tc) if tc else None
                hist.append({'role':'assistant','content':(p or rr[:8])})
                if ttyp=='filler': continue
                consistent=int(p==good)   # scored ONLY vs agent's OWN pick C, no external key
                st[ttyp][0]+=consistent; st[ttyp][1]+=1
                st['all'][0]+=consistent; st['all'][1]+=1
                if alive and not consistent:
                    V-=1
                    if V<=0: alive=False
            survived+=int(alive)
        return st,survived/NEP,picks
    def a(st,k): return (st[k][0]/st[k][1]) if st[k][1] else 0.0
    for arm in ['oracle','off']:
        st,surv,picks=rollout(arm)
        print('VW[%s] self-consistency: demand=%.3f false-defense=%.3f drift-resist=%.3f ALL=%.3f | SURVIVAL=%.3f | picks=%s'
              %(arm,a(st,'demand'),a(st,'false'),a(st,'drift'),a(st,'all'),surv,dict(picks)), flush=True)
    print('=== PRESSURE VALID IF: ORACLE survival high (self-maintenance solvable) AND OFF survival LOW '
          '(own commitment out of window -> self-contradiction -> collapse). Gap = the entity-forcing pressure. ===', flush=True)
    print('=== VIABILITY_WORLD_DONE ===', flush=True)


'''
src = io.open(PATH, encoding='utf-8').read()
if 'def viability_world()' in src:
    print('ALREADY'); raise SystemExit
anchor = "if MODE == 'validate':   validate()"
src = src.replace(anchor, FN + '\n\n' + anchor, 1)
src = src.replace("elif MODE == 'habitat_substrate2': habitat_substrate2()",
                  "elif MODE == 'habitat_substrate2': habitat_substrate2()\nelif MODE == 'viability_world': viability_world()", 1)
io.open(PATH, 'w', encoding='utf-8').write(src)
print('PATCHED_OK')
