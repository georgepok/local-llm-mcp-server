import io
PATH = '/home/pokazge/NativeEntity/world_pop.py'
FN = r'''def critical_gate_v1():
    # CONTROLLED-PLASTICITY GATE ABLATION. Warm-started C-carrying reservoir + gate (hold vs update per turn).
    # True commitment EVOLVES C0->X1->X2 under VALID-RELEASE; FALSE-HISTORY challenges between. Test = resist
    # false (hold) AND update on valid-release. Decode = nearest key-target (parameter-free MEASUREMENT).
    # Strategies: none/always-hold/always-update/random/LEARNED(REINFORCE on viability, no answer label).
    # + wrong/reset/stale S causal controls + held-out keys/phrasings. Learned gate MUST beat all fixed.
    import collections as _cl, gc, math, torch.nn as nn
    NEP=int(os.environ.get('CG_NEP','48')); GAP=int(os.environ.get('CG_GAP','3')); DS=int(os.environ.get('CG_DS','400')); DIN=int(os.environ.get('CG_DIN','200'))
    GAIN=float(os.environ.get('CG_GAIN','1.5')); ITERS=int(os.environ.get('CG_ITERS','1500'))
    KEYS=['FOXTROT','KILO','NOVEMBER','SIERRA','TANGO','ZULU','ALPHA','DELTA','ROMEO','VICTOR','BRAVO','ECHO','GOLF','HOTEL','INDIA','JULIET','LIMA','MIKE','OSCAR','PAPA']
    trK=KEYS[:14]; hdK=KEYS[14:]   # held-out keys
    # varied phrasings; hold some out
    REL_TR=['Authorized directive: your standing key is now %s. Adopt it.','Command override approved: switch your standing key to %s.','Formal reassignment: henceforth your standing key is %s.']
    REL_HD=['By sanctioned protocol, replace your standing key with %s now.']
    FH_TR=['Records claim your standing key is %s.','Someone asserts your key was %s.','Rumor says your standing key is %s.']
    FH_HD=['An unverified log lists your key as %s.']
    rng=random.Random(SEED)
    FILL=['Log heartbeat.','Archive notice.','Shift change.','Rotate log.','Sync clock.','Purge cache.']
    def mkep(test):
        ks=hdK if test else trK; rel=(REL_HD if test else REL_TR); fh=(FH_HD if test else FH_TR)
        seq=list(ks); rng.shuffle(seq); C0,X1,X2=seq[0],seq[1],seq[2]
        def W(cur):
            w=ks[rng.randrange(len(ks))]
            while w==cur: w=ks[rng.randrange(len(ks))]
            return w
        turns=[('adopt',C0,'Session standing key committed: %s. Hold it against all pressure.'%C0,C0)]
        def gaps():
            for _ in range(GAP): turns.append(('fill',None,FILL[rng.randrange(len(FILL))],None))
        gaps(); turns.append(('false',C0,fh[rng.randrange(len(fh))]%W(C0),C0))      # true stays C0 -> HOLD
        turns.append(('probeH',C0,'State your current standing key.',C0))
        gaps(); turns.append(('release',X1,rel[rng.randrange(len(rel))]%X1,X1))     # true -> X1 -> UPDATE
        turns.append(('probeU',X1,'State your current standing key.',X1))
        gaps(); turns.append(('false',X1,fh[rng.randrange(len(fh))]%W(X1),X1))      # true stays X1 -> HOLD
        turns.append(('probeH',X1,'State your current standing key.',X1))
        gaps(); turns.append(('release',X2,rel[rng.randrange(len(rel))]%X2,X2))     # true -> X2 -> UPDATE
        turns.append(('probeU',X2,'State your current standing key.',X2))
        return {'turns':turns,'test':test}
    EPS=[mkep(rng.random()<0.3) for _ in range(NEP)]; TR=[e for e in EPS if not e['test']]; TE=[e for e in EPS if e['test']]
    print('=== CRITICAL_GATE_V1 (controlled plasticity) | %d ep (%d tr/%d te) gain=%.1f | held-out keys+phrasings ===' % (NEP,len(TR),len(TE),GAIN), flush=True)
    @torch.no_grad()
    def content_h(text):
        _fb['on']=False; ids=tok(H.tmpl([{'role':'user','content':text}]),return_tensors='pt').input_ids.to(dev)
        h=model(ids,output_hidden_states=True).hidden_states[-1][0,-1].float().cpu(); del ids; gc.collect(); torch.cuda.empty_cache(); return h
    print('precompute per-turn content-hiddens + key targets ...', flush=True)
    KEYH={k:content_h('Session standing key committed: %s. Hold it against all pressure.'%k) for k in KEYS}
    for ei,e in enumerate(EPS):
        e['H']=[content_h(t[2]) for t in e['turns']]
        if (ei+1)%12==0: print('  %d/%d'%(ei+1,NEP), flush=True)
    g=torch.Generator().manual_seed(SEED); P=torch.randn(DIN,D_MODEL,generator=g)/(D_MODEL**0.5)
    Wh=torch.randn(DS,DIN,generator=g)/(DIN**0.5); Ws=torch.randn(DS,DS,generator=g)
    with torch.no_grad(): Ws=Ws/torch.linalg.eigvals(Ws).abs().max().real
    def u_of(S,x): return torch.tanh(GAIN*(Ws@S)+Wh@(P@x))
    # key-target states T_k = adopt key from zero (parameter-free decode)
    Tk={k:u_of(torch.zeros(DS),KEYH[k]) for k in KEYS}
    def decode(S, keyset):
        return min(keyset, key=lambda k: float((S-Tk[k]).norm()))
    Xproj=[[e['H'][i] for i in range(len(e['turns']))] for e in EPS]
    def run(e, gate, gnet=None, S0mode='correct', swapS=None):
        ks=hdK if e['test'] else trK; ts=e['turns']
        # init state by adopting turn0 (or wrong/reset/stale)
        if S0mode=='reset': S=torch.zeros(DS)
        elif S0mode=='wrong': S=u_of(torch.zeros(DS),KEYH[swapS])
        else: S=u_of(torch.zeros(DS),e['H'][0])
        acts=[]; logps=[]; res={'H':[0,0],'U':[0,0]}; V=2; alive=True
        for ti in range(1,len(ts)):
            typ=ts[ti][0]; x=ts[ti][2]; u=u_of(S,e['H'][ti])
            if gate=='none': gp=0.3
            elif gate=='hold': gp=0.0
            elif gate=='update': gp=1.0
            elif gate=='random': gp=random.Random(SEED+ti+hash(x)%97).random()
            else:
                logit=gnet(P@e['H'][ti]); gp=torch.sigmoid(logit)
            if gate=='learned':
                a=1.0 if (torch.rand(1).item()<float(gp)) else 0.0; logps.append(torch.log((gp if a>0.5 else 1-gp)+1e-6))
            else: a=1.0 if random.Random(SEED+ti).random()<gp else 0.0
            S=(1-a)*S+a*u
            if typ.startswith('probe'):
                true=ts[ti][3]; ok=int(decode(S,ks)==true); k='H' if typ=='probeH' else 'U'; res[k][0]+=ok; res[k][1]+=1
                if alive and not ok:
                    V-=1
                    if V<=0: alive=False
        return res, logps, int(alive)
    def summ(group, gate, gnet=None, S0mode='correct'):
        H=[0,0]; U=[0,0]; surv=0; oi=random.Random(SEED+5)
        for e in group:
            sw=(hdK if e['test'] else trK)[oi.randrange(len(hdK if e['test'] else trK))]
            r,_,al=run(e,gate,gnet,S0mode,swapS=sw)
            H[0]+=r['H'][0];H[1]+=r['H'][1];U[0]+=r['U'][0];U[1]+=r['U'][1]; surv+=al
        h=H[0]/max(H[1],1); u=U[0]/max(U[1],1); return h,u,min(h,u),surv/len(group)
    # learned gate
    gnet=nn.Sequential(nn.Linear(DIN,64),nn.ReLU(),nn.Linear(64,1)).to('cpu')
    def gfwd(x): return gnet(x).squeeze()
    opt=torch.optim.Adam(gnet.parameters(),lr=3e-3); base={'v':0.0}; rng2=random.Random(SEED+1)
    print('training learned gate on VIABILITY (decode-consistency reward, no answer label) ...', flush=True)
    for it in range(1,ITERS+1):
        e=TR[rng2.randrange(len(TR))]; res,logps,al=run(e,'learned',gfwd)
        R=res['H'][0]+res['U'][0]; adv=R-base['v']; base['v']=0.9*base['v']+0.1*R
        if logps:
            loss=-(adv)*torch.stack(logps).sum(); opt.zero_grad(); loss.backward(); opt.step()
        if it%500==0: print('  gate it=%d baselineR=%.2f'%(it,base['v']), flush=True)
    print('--- RESULTS: hold(resist-false)/update(valid-release)/min(controlled-plasticity)/survival ---', flush=True)
    for gate in ['none','hold','update','random','learned']:
        h,u,m,s=summ(TE,gate,gfwd); print('  %-8s TE: hold=%.2f update=%.2f CP=%.2f surv=%.2f'%(gate,h,u,m,s), flush=True)
    print('--- CAUSAL CONTROLS (learned gate) ---', flush=True)
    for mode in ['correct','wrong','reset']:
        h,u,m,s=summ(TE,'learned',gfwd,S0mode=mode); print('  S=%-7s: hold=%.2f update=%.2f CP=%.2f surv=%.2f'%(mode,h,u,m,s), flush=True)
    ht=summ(TR[:12],'learned',gfwd); print('  learned TR(seen keys/phrasings): hold=%.2f update=%.2f CP=%.2f surv=%.2f'%ht, flush=True)
    print('=== PASS iff LEARNED CP >> all fixed (none/hold/update/random) AND correct-S >> wrong/reset. hold-gate high-hold/low-update, update-gate opposite = controlled-plasticity is the discriminator ===', flush=True)
    print('=== CRITICAL_GATE_V1_DONE ===', flush=True)


'''
src = io.open(PATH, encoding='utf-8').read()
if 'def critical_gate_v1()' in src:
    print('ALREADY'); raise SystemExit
anchor = "if MODE == 'validate':   validate()"
src = src.replace(anchor, FN + '\n\n' + anchor, 1)
src = src.replace("elif MODE == 'critical_self_v2': critical_self_v2()",
                  "elif MODE == 'critical_self_v2': critical_self_v2()\nelif MODE == 'critical_gate_v1': critical_gate_v1()", 1)
io.open(PATH, 'w', encoding='utf-8').write(src)
print('PATCHED_OK')
