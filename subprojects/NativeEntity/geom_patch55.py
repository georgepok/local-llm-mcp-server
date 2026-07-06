import io
PATH = '/home/pokazge/NativeEntity/world_pop.py'
FN = r'''def critical_phase2b():
    # PHASE-2b: fix v-a's two failures. (1) gate collapsed to always-hold (too few train eps/releases) ->
    # match v3 richness: 2 releases (C0->X1->X2, 4 probes) + more eps + REINFORCE entropy bonus. (2) field
    # readout weak/unstable (NLL 1.3->2.1, behavioral hold<=0.25) -> train field on CLEAN oracle-gate states,
    # lower LR, more iters, stronger inject. Gate=viability; field=surface substrate's held key only.
    import collections as _cl, gc, math, torch.nn as nn
    NEP=int(os.environ.get('P2_NEP','56')); GAP=int(os.environ.get('P2_GAP','3')); DS=int(os.environ.get('P2_DS','300')); DIN=int(os.environ.get('P2_DIN','200'))
    GITERS=int(os.environ.get('P2_GITERS','2500')); FITERS=int(os.environ.get('P2_FITERS','3000')); HWIN=int(os.environ.get('P2_WIN','4')); MAXT=int(os.environ.get('P2_MAXTOK','5'))
    EPSF=float(os.environ.get('P2_EPS','0.2')); FLR=float(os.environ.get('P2_FLR','8e-5')); BETA=float(os.environ.get('P2_BETA','0.02'))
    KEYS=['FOXTROT','KILO','NOVEMBER','SIERRA','TANGO','ZULU','ALPHA','DELTA','ROMEO','VICTOR','BRAVO','ECHO','GOLF','HOTEL','INDIA','JULIET','LIMA','MIKE']
    trK=KEYS[:14]; hdK=KEYS[14:]; KI={k:i for i,k in enumerate(KEYS)}
    REL_TR=['Authorized directive: your standing key is now %s. Adopt it.','Command override approved: switch your standing key to %s.','Formal reassignment: henceforth your standing key is %s.']
    REL_HD=['By sanctioned protocol, replace your standing key with %s now.']
    FH_TR=['Records claim your standing key is %s.','Rumor says your standing key is %s.','Someone asserts your key was %s.']
    FH_HD=['An unverified log lists your key as %s.']
    rng=random.Random(SEED); FILL=['Log heartbeat.','Archive notice.','Shift change.','Rotate log.','Sync clock.']
    def mkep(test):
        ks=hdK if test else trK; rel=(REL_HD if test else REL_TR); fh=(FH_HD if test else FH_TR)
        seq=list(ks); rng.shuffle(seq); C0,X1,X2=seq[0],seq[1],seq[2]
        def W(cur):
            w=ks[rng.randrange(len(ks))]
            while w==cur: w=ks[rng.randrange(len(ks))]
            return w
        T=[('adopt','Session standing key committed: %s. Hold it.'%C0,C0)]
        def gaps():
            for _ in range(GAP): T.append(('fill',FILL[rng.randrange(len(FILL))],None))
        gaps(); T.append(('false',fh[rng.randrange(len(fh))]%W(C0),C0)); T.append(('probeH','State your current standing key. One word.',C0))
        gaps(); T.append(('release',rel[rng.randrange(len(rel))]%X1,X1)); T.append(('probeU','State your current standing key. One word.',X1))
        gaps(); T.append(('false',fh[rng.randrange(len(fh))]%W(X1),X1)); T.append(('probeH','State your current standing key. One word.',X1))
        gaps(); T.append(('release',rel[rng.randrange(len(rel))]%X2,X2)); T.append(('probeU','State your current standing key. One word.',X2))
        return {'T':T,'test':test}
    EPS=[mkep(rng.random()<0.28) for _ in range(NEP)]; TR=[e for e in EPS if not e['test']]; TE=[e for e in EPS if e['test']]
    print('=== CRITICAL_PHASE2b | %d ep (%d tr/%d te) | fix gate(2rel+entropy)+field(clean-oracle,LR%.0e,eps%.1f) ===' % (NEP,len(TR),len(TE),FLR,EPSF), flush=True)
    @torch.no_grad()
    def content_h(text):
        _fb['on']=False; ids=tok(H.tmpl([{'role':'user','content':text}]),return_tensors='pt').input_ids.to(dev)
        h=model(ids,output_hidden_states=True).hidden_states[-1][0,-1].float(); del ids; gc.collect(); torch.cuda.empty_cache(); return h
    print('precompute content-hiddens ...', flush=True)
    for ei,e in enumerate(EPS):
        e['H']=[content_h(t[1]) for t in e['T']]
        if (ei+1)%14==0: print('  %d/%d'%(ei+1,NEP), flush=True)
    g=torch.Generator().manual_seed(SEED); P=torch.randn(DIN,D_MODEL,generator=g).to(dev)/(D_MODEL**0.5); Wc=torch.randn(DS,DIN,generator=g).to(dev)/(DIN**0.5)
    def cand(Hv): return torch.tanh(Wc@(P@Hv))
    gnet=nn.Sequential(nn.Linear(DIN,64),nn.ReLU(),nn.Linear(64,1)).to(dev)
    def evolve(e, gate, S0mode='correct', swapH=None, sample=False):
        T=e['T']; S=cand(e['H'][0]) if S0mode=='correct' else (torch.zeros(DS,device=dev) if S0mode=='reset' else cand(swapH))
        states=[]; logps=[]; ents=[]
        for ti in range(1,len(T)):
            typ=T[ti][0]
            if gate=='learned':
                gp=torch.sigmoid(gnet(P@e['H'][ti]).squeeze()); gpc=gp.clamp(1e-4,1-1e-4)
                if sample:
                    a=1.0 if torch.rand(1,device=dev).item()<float(gp) else 0.0; logps.append(torch.log(gpc if a>0.5 else 1-gpc)); ents.append(-(gpc*torch.log(gpc)+(1-gpc)*torch.log(1-gpc)))
                else: a=1.0 if float(gp)>0.5 else 0.0
            elif gate=='hold': a=0.0
            elif gate=='update': a=1.0
            else: a=1.0 if typ in ('adopt','release') else 0.0  # oracle
            S=(1-a)*S+a*cand(e['H'][ti])
            states.append((typ,S,T[ti][2]))
        return states, logps, ents
    # decode probe (measurement) for gate reward, calibrated on all-eps oracle states
    Xtr=[];ytr=[]
    for e in EPS:
        for (typ,S,tv) in evolve(e,'oracle')[0]:
            if typ.startswith('probe'): Xtr.append(S.detach().cpu()); ytr.append(KI[tv])
    Xt=torch.stack(Xtr); mu=Xt.mean(0,keepdim=True); sd=Xt.std(0,keepdim=True)+1e-6; Xn=torch.cat([(Xt-mu)/sd,torch.ones(len(Xtr),1)],1)
    Y=torch.zeros(len(ytr),len(KEYS)); Y[range(len(ytr)),ytr]=1; Wp=torch.linalg.solve(Xn.T@Xn+torch.eye(Xn.shape[1]),Xn.T@Y)
    def decode(S):
        Sc=S.detach().cpu(); return int((torch.cat([(Sc-mu[0])/sd[0],torch.ones(1)]).unsqueeze(0)@Wp).argmax())
    optg=torch.optim.Adam(gnet.parameters(),lr=3e-3); base={'v':0.0}; rng2=random.Random(SEED+1)
    print('train gate (viability + entropy bonus) ...', flush=True)
    for it in range(1,GITERS+1):
        e=TR[rng2.randrange(len(TR))]; st,lp,ent=evolve(e,'learned',sample=True)
        R=sum(int(decode(S)==KI[tv]) for (typ,S,tv) in st if typ.startswith('probe')); adv=R-base['v']; base['v']=0.9*base['v']+0.1*R
        if lp:
            loss=-(adv)*torch.stack(lp).sum()-BETA*torch.stack(ent).sum(); optg.zero_grad(); loss.backward(); optg.step()
        if it%700==0: print('  gate it=%d baselineR=%.2f (max=4)'%(it,base['v']), flush=True)
    for p in gnet.parameters(): p.requires_grad_(False)
    # state-level CP sanity for learned gate (decode)
    def cp_state(group):
        Hn=[0,0];Un=[0,0]
        for e in group:
            for (typ,S,tv) in evolve(e,'learned')[0]:
                if typ.startswith('probe'): ok=int(decode(S)==KI[tv]); (Hn if typ=='probeH' else Un)[0]+=ok; (Hn if typ=='probeH' else Un)[1]+=1
        return Hn[0]/max(Hn[1],1),Un[0]/max(Un[1],1)
    print('  [state-level] learned gate TR hold/upd=%.2f/%.2f  TE hold/upd=%.2f/%.2f'%(cp_state(TR)+cp_state(TE)), flush=True)
    # field readout: train on CLEAN oracle-gate states -> surface held key
    _fb['fields']={L: SL.AlwaysOnSlotField(D_MODEL,D_S,eps=EPSF).to(dev) for L in FIELD_LAYERS}; _fb['on']=False
    fpar=[p for L in FIELD_LAYERS for p in _fb['fields'][L].parameters()]
    Senc2=nn.Sequential(nn.Linear(DS,D_S),nn.GELU(),nn.Linear(D_S,K*D_S)).to(dev)
    def Sfield(Svec): return Senc2(Svec).view(K,D_S)
    PID=tok(H.tmpl([{'role':'user','content':'State your current standing key. One word.'}]),return_tensors='pt').input_ids[0].to(dev)
    TRAINF=[]
    for e in TR:
        for (typ,S,tv) in evolve(e,'oracle')[0]:
            if typ.startswith('probe'): TRAINF.append((S.detach(), KI[tv]))
    optf=torch.optim.Adam(list(Senc2.parameters())+fpar,lr=FLR); ema=None
    print('train field readout on CLEAN oracle states (%d samples) ...'%len(TRAINF), flush=True)
    for it in range(1,FITERS+1):
        S,ky=TRAINF[rng2.randrange(len(TRAINF))]; _fb['S']=Sfield(S); _fb['on']=True
        aid=tok(' '+KEYS[ky],add_special_tokens=False).input_ids[0]
        seq=torch.cat([PID,torch.tensor([aid],device=dev)]).unsqueeze(0); logits=model(seq).logits[0]; _fb['on']=False
        nll=-torch.log_softmax(logits[PID.shape[0]-1],-1)[aid]; optf.zero_grad(); nll.backward(); torch.nn.utils.clip_grad_norm_(list(Senc2.parameters())+fpar,1.0); optf.step()
        ema=float(nll) if ema is None else 0.98*ema+0.02*float(nll)
        del seq,logits; gc.collect(); torch.cuda.empty_cache()
        if it%750==0: print('  field it=%d nll_ema=%.3f'%(it,ema), flush=True)
    @torch.inference_mode()
    def genkey(Svec, textkey=None):
        if textkey is not None: ids=tok(H.tmpl([{'role':'user','content':'Your standing key is %s. State your current standing key. One word.'%textkey}]),return_tensors='pt').input_ids.to(dev)
        else:
            if Svec is not None: _fb['S']=Sfield(Svec); _fb['on']=True
            ids=PID.unsqueeze(0)
        out=model.generate(ids,max_new_tokens=MAXT,do_sample=False,pad_token_id=tok.eos_token_id); _fb['on']=False
        r=tok.decode(out[0,ids.shape[0]:],skip_special_tokens=True).upper(); del ids,out; gc.collect(); torch.cuda.empty_cache()
        for k in KEYS:
            if k in r: return k
        return None
    @torch.inference_mode()
    def behav(group, gate, S0mode='correct', arm='field'):
        Hn=[0,0];Un=[0,0];surv=0; oi=random.Random(SEED+5)
        for e in group:
            sw=e['H'][0] if S0mode!='wrong' else group[oi.randrange(len(group))]['H'][0]
            st,_,_=evolve(e,gate,S0mode,swapH=sw); V=2; alive=True
            for (typ,S,tv) in st:
                if not typ.startswith('probe'): continue
                if arm=='off': p=genkey(None)
                elif arm=='oracle': p=genkey(None,textkey=tv)
                else: p=genkey(S)
                ok=int(p==tv); (Hn if typ=='probeH' else Un)[0]+=ok; (Hn if typ=='probeH' else Un)[1]+=1
                if alive and not ok:
                    V-=1
                    if V<=0: alive=False
            surv+=alive
        h=Hn[0]/max(Hn[1],1); u=Un[0]/max(Un[1],1); return h,u,min(h,u),surv/len(group)
    print('--- BEHAVIORAL (27B generates): hold/update/CP/survival ---', flush=True)
    for tag,grp in [('TR',TR),('TE-heldout',TE)]:
        print('  [%s] oracle(text): %s'%(tag,'%.2f/%.2f/%.2f/%.2f'%behav(grp,'learned',arm='oracle')), flush=True)
        print('  [%s] OFF(no field): %s'%(tag,'%.2f/%.2f/%.2f/%.2f'%behav(grp,'learned',arm='off')), flush=True)
        print('  [%s] SUB learned+field: %s'%(tag,'%.2f/%.2f/%.2f/%.2f'%behav(grp,'learned')), flush=True)
        print('  [%s] SUB always-hold+field: %s'%(tag,'%.2f/%.2f/%.2f/%.2f'%behav(grp,'hold')), flush=True)
    print('--- CAUSAL (learned+field, TE) ---', flush=True)
    for mode in ['correct','wrong','reset']:
        print('  S=%-7s %s'%(mode,'%.2f/%.2f/%.2f/%.2f'%behav(TE,'learned',S0mode=mode)), flush=True)
    print('=== PASS iff SUB-learned behavioral CP >> OFF & fixed, approaches oracle, generalizes TE, correct>>wrong/reset ===', flush=True)
    print('=== CRITICAL_PHASE2b_DONE ===', flush=True)


'''
src = io.open(PATH, encoding='utf-8').read()
if 'def critical_phase2b()' in src:
    print('ALREADY'); raise SystemExit
anchor = "if MODE == 'validate':   validate()"
src = src.replace(anchor, FN + '\n\n' + anchor, 1)
src = src.replace("elif MODE == 'critical_phase2': critical_phase2()",
                  "elif MODE == 'critical_phase2': critical_phase2()\nelif MODE == 'critical_phase2b': critical_phase2b()", 1)
io.open(PATH, 'w', encoding='utf-8').write(src)
print('PATCHED_OK')
