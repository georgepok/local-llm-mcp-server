import os, random
os.environ['ME_MODE']='none'

# ENDOGENOUS_EXECUTABLE_CHEMISTRY_V1 — spatial executable chemistry. Objects hold instruction strings that EXECUTE,
# consuming resource; they can BIND/COPY/WRITE/SPAWN to construct other programs, ACTIVATE/INHIBIT to regulate, and
# TRANSFER resource. Continued existence is ENDOGENOUS (no external reward). This file = base interpreter (family
# "conservative-template") + resource physics + HIDDEN hand-built simulator-validation controls (Part 19).
# NO named replicate/membrane/metabolism ops — those behaviors must arise from the generic low-level instruction set.

OPS=['NOP','MATCH','BIND','UNBIND','COPY','WRITE','DELETE','CUT','JOIN','MOVE','TRANSFER','EXEC','ACTIVATE','INHIBIT','SPAWN','DECAY']
SYMS=['S0','S1','S2','S3']; TOK=OPS+SYMS
T={n:i for i,n in enumerate(TOK)}                       # name -> id
NOP,MATCH,BIND,UNBIND,COPY,WRITE,DELETE,CUT,JOIN,MOVE,TRANSFER,EXEC,ACTIVATE,INHIBIT,SPAWN,DECAY=range(16)
def name(t): return TOK[t] if 0<=t<len(TOK) else '?'
def show(instr): return ' '.join(name(t) for t in instr)

# ---- economy constants (tuned so: idle inactive decays; active survives IF activated; execution costs; copying costs) ----
C=dict(OP_COST=0.03, COPY_COST=0.02, WRITE_COST=0.02, SPAWN_COST=0.8, SPAWN_ENDOW=2.0, TRANSFER_AMT=0.6,
       HARVEST_ACTIVE=0.45, HARVEST_IDLE=0.10, DECAY=0.25, COPY_SEG=8, COPY_ERR=0.01, EXEC_STEPS=3,
       ACT_TTL=6, MAX_LEN=32, MAX_PER_CELL=8, DECAY_HIT=0.5, START_RES=3.0, MAX_RES=8.0)

class Obj:
    __slots__=('instr','regs','res','ip','active','act_ttl','bound','dbuf','rptr','age','y','x','exec_credit','cls','alive')
    def __init__(s,instr,y,x,res,cls='?'):
        s.instr=list(instr); s.regs=[0,0,0,0]; s.res=res; s.ip=0; s.active=False; s.act_ttl=0
        s.bound=-1; s.dbuf=[]; s.rptr=0; s.age=0; s.y=y; s.x=x; s.exec_credit=0; s.cls=cls; s.alive=True

class World:
    def __init__(s,H=12,Wd=12,seed=0,inflow=0.35,family='base'):
        s.H=H; s.W=Wd; s.objs=[]; s.rng=random.Random(seed); s.R=[[2.0]*Wd for _ in range(H)]; s.t=0; s.inflow=inflow; s.family=family; s.cmap=None
    def _rebuild(s):
        s.cmap={}
        for i,o in enumerate(s.objs):
            if o.alive: s.cmap.setdefault((o.y,o.x),[]).append(i)
    def cell(s,y,x):                                        # O(cell) via per-tick index + freshness filter (handles move/death/spawn)
        if s.cmap is None: return [i for i,o in enumerate(s.objs) if o.alive and o.y==y and o.x==x]
        return [i for i in s.cmap.get((y,x),()) if s.objs[i].alive and s.objs[i].y==y and s.objs[i].x==x]
    def add(s,o):
        if len(s.cell(o.y,o.x))<C['MAX_PER_CELL']:
            s.objs.append(o)
            if s.cmap is not None: s.cmap.setdefault((o.y,o.x),[]).append(len(s.objs)-1)
            return len(s.objs)-1
        return -1
    def nb(s,y,x,k): dy,dx=[(0,0),(0,1),(0,-1),(1,0),(-1,0)][k%5]; return ((y+dy)%s.H,(x+dx)%s.W)
    def step_obj(s,oi):
        o=s.objs[oi]
        if not o.instr or not o.alive: return
        L=len(o.instr); op=o.instr[o.ip%L]; o.res-=C['OP_COST']
        def bnd(): return s.objs[o.bound] if (o.bound>=0 and o.bound<len(s.objs) and s.objs[o.bound].alive) else None
        if op==BIND:
            cs=[j for j in s.cell(o.y,o.x) if j!=oi and s.objs[j].instr]
            o.bound = cs[o.regs[0]%len(cs)] if cs else -1
        elif op==UNBIND: o.bound=-1
        elif op==MATCH:
            sym=o.instr[(o.ip+1)%L]; b=bnd(); o.regs[0]=1 if (b and sym in b.instr) else 0; o.ip+=1
        elif op==COPY:
            b=bnd()
            if b is None and s.family=='autocat':                      # CROSS-DOMAIN family: COPY grabs a random co-cell template (template-free autocatalysis)
                cs=[j for j in s.cell(o.y,o.x) if j!=oi and s.objs[j].instr]
                if cs: b=s.objs[cs[o.regs[0]%len(cs)]]
            if b:
                seg=b.instr[o.rptr:o.rptr+C['COPY_SEG']]
                for tk in seg:
                    if s.rng.random()<C['COPY_ERR']: tk=s.rng.randrange(len(TOK))
                    if len(o.dbuf)<C['MAX_LEN']: o.dbuf.append(tk)
                o.rptr+=len(seg); o.res-=C['COPY_COST']*len(seg)
                if o.rptr>=len(b.instr): o.regs[1]=1
        elif op==WRITE:
            tk=o.instr[(o.ip+1)%L]
            if len(o.dbuf)<C['MAX_LEN']: o.dbuf.append(tk)
            o.res-=C['WRITE_COST']; o.ip+=1
        elif op==DELETE:
            b=bnd()
            if b and b.instr: del b.instr[o.regs[0]%len(b.instr)]
        elif op==CUT: o.dbuf=o.dbuf[:max(1,len(o.dbuf)//2)]
        elif op==JOIN:
            b=bnd()
            if b: o.dbuf=(o.dbuf+list(b.instr))[:C['MAX_LEN']]
        elif op==MOVE:
            ny,nx=s.nb(o.y,o.x,o.regs[0])
            if len(s.cell(ny,nx))<C['MAX_PER_CELL']: o.y,o.x=ny,nx; o.bound=-1
        elif op==TRANSFER:
            b=bnd()
            if b: amt=min(C['TRANSFER_AMT'],max(0,b.res)); b.res-=amt; o.res+=amt
        elif op==EXEC:
            b=bnd()
            if b: b.exec_credit+=2
        elif op==ACTIVATE:
            o.res-=C.get('ACTIVATE_COST',0.0)                          # activating others can be COSTLY -> pure activator starves unless repaid (forces reciprocity)
            if s.family=='broadcast':                                  # ALIEN family: activate all co-cell objects (no bind needed)
                for j in s.cell(o.y,o.x):
                    if j!=oi: s.objs[j].active=True; s.objs[j].act_ttl=C['ACT_TTL']
            else:
                tgt=bnd()
                if tgt is not None: tgt.active=True; tgt.act_ttl=C['ACT_TTL']
        elif op==INHIBIT:
            b=bnd()
            if b: b.active=False; b.act_ttl=0
        elif op==SPAWN:
            if o.dbuf and o.res>C['SPAWN_COST']+C['SPAWN_ENDOW']:
                ny,nx=s.nb(o.y,o.x,o.regs[2])
                seq=o.dbuf[:C['MAX_LEN']]
                mr=C.get('MUT_RATE',0.0)                                   # inner evolution (Part 10): point mutation on spawn
                if mr>0:
                    seq=list(seq)
                    for k in range(len(seq)):
                        if s.rng.random()<mr: seq[k]=s.rng.randrange(len(TOK))
                    if s.rng.random()<mr and len(seq)<C['MAX_LEN']: seq.insert(s.rng.randrange(len(seq)+1),s.rng.randrange(len(TOK)))
                    if s.rng.random()<mr and len(seq)>1: del seq[s.rng.randrange(len(seq))]
                child=Obj(seq,ny,nx,C['SPAWN_ENDOW'],cls='child')
                if s.add(child)>=0: o.res-=C['SPAWN_COST']+C['SPAWN_ENDOW']; o.dbuf=[]; o.rptr=0; o.regs[1]=0
        elif op==DECAY:
            b=bnd()
            if b: b.res-=C['DECAY_HIT']
        o.ip=(o.ip+1)%max(len(o.instr),1)
    def tick(s):
        s.t+=1; s._rebuild()
        for y in range(s.H):
            for x in range(s.W): s.R[y][x]+=s.inflow*(0.6+0.4*((s.t*7+y*3+x)%11)/11.0)
        for o in s.objs:
            if not o.alive: continue
            boost=C['HARVEST_ACTIVE'] if (o.active and o.act_ttl>0) else C['HARVEST_IDLE']
            h=min(boost,max(0,s.R[o.y][o.x])); add=min(h,max(0.0,C['MAX_RES']-o.res)); o.res+=add; s.R[o.y][o.x]-=add
            if o.act_ttl>0: o.act_ttl-=1
            if o.act_ttl==0: o.active=False
        for oi in range(len(s.objs)):
            o=s.objs[oi]
            if not o.alive: continue
            steps=C['EXEC_STEPS']+o.exec_credit; o.exec_credit=0
            for _ in range(steps):
                if o.res<=0: break
                s.step_obj(oi)
        for o in s.objs:
            if not o.alive: continue
            o.res-=C['DECAY']; o.age+=1
            if o.res<=0 and o.instr and s.rng.random()<0.5: del o.instr[s.rng.randrange(len(o.instr))]  # degradation
            if not o.instr or o.res<=-2: o.alive=False
    def live(s): return [o for o in s.objs if o.alive]
    def count(s,cls=None): return sum(1 for o in s.objs if o.alive and (cls is None or o.cls==cls))

def P(*names): return [T[n] for n in names]   # build a program from op names

# ================== PART 19: SIMULATOR VALIDATION (hidden hand-built controls; NOT exposed to synthesis models) ==================
def run(w, n):
    for _ in range(n): w.tick()

def instr_eq(a,b): return list(a)==list(b)

def validate():
    print("=== ENDOGENOUS_EXECUTABLE_CHEMISTRY_V1 — Part 19 SIMULATOR VALIDATION (base interpreter) ===", flush=True)
    R={}
    # 1 SIMPLE EXECUTION
    w=World(seed=1); o=Obj(P('MATCH','S0','NOP'),3,3,5.0,'e'); w.add(o); ip0=w.objs[0].ip; res0=w.objs[0].res
    run(w,4); R['1 simple execution']= (w.objs[0].alive and w.objs[0].res<res0)
    # 2 LOCAL COPYING (copier copies a bound template + spawns a copy)
    w=World(seed=2); tmpl=P('S1','S2','S3','MATCH'); w.add(Obj(P('BIND','COPY','SPAWN'),3,3,12.0,'cp')); w.add(Obj(tmpl,3,3,12.0,'tmpl'))
    for o in w.objs: o.active=True; o.act_ttl=999
    run(w,14); copies=sum(1 for o in w.live() if o.cls=='child' and instr_eq(o.instr,tmpl))
    R['2 local copying']= copies>=1
    # 3 COPYING WITH MUTATION
    C['COPY_ERR']=0.35; w=World(seed=3); tmpl=P('S1','S2','S3','MATCH','NOP','BIND'); w.add(Obj(P('BIND','COPY','SPAWN'),3,3,14.0,'cp')); w.add(Obj(tmpl,3,3,14.0,'tmpl'))
    for o in w.objs: o.active=True; o.act_ttl=999
    run(w,16); kids=[o for o in w.live() if o.cls=='child']; muts=sum(1 for o in kids if not instr_eq(o.instr,tmpl)); C['COPY_ERR']=0.01
    R['3 copying with mutation']= len(kids)>=1 and muts>=1
    # 4 BINDING
    w=World(seed=4); w.add(Obj(P('BIND','NOP'),3,3,5.0,'a')); w.add(Obj(P('NOP','NOP'),3,3,5.0,'b')); w.tick()
    R['4 binding']= w.objs[0].bound==1
    # 5 CATALYTIC ACTIVATION
    w=World(seed=5); w.add(Obj(P('BIND','ACTIVATE'),3,3,5.0,'a')); w.add(Obj(P('NOP'),3,3,5.0,'b')); run(w,2)
    R['5 catalytic activation']= w.objs[1].active==True
    # 6 INHIBITION
    w=World(seed=6); w.add(Obj(P('BIND','INHIBIT'),3,3,5.0,'a')); b=Obj(P('NOP'),3,3,5.0,'b'); b.active=True; b.act_ttl=50; w.add(b); run(w,2)
    R['6 inhibition']= w.objs[1].active==False
    # 7 RESOURCE TRANSFER
    w=World(seed=7); w.add(Obj(P('BIND','TRANSFER'),3,3,3.0,'a')); w.add(Obj(P('NOP'),3,3,9.0,'b')); rb0=w.objs[1].res; ra0=w.objs[0].res; run(w,2)
    R['7 resource transfer']= w.objs[1].res<rb0-0.3 and w.objs[0].res>ra0-0.5
    # 8 PROGRAM DECAY (inactive, low resource -> degrades/dies)
    w=World(H=6,Wd=6,seed=8,inflow=0.0); o=Obj(P('NOP','NOP','NOP','NOP','NOP','NOP'),2,2,0.4,'d'); w.add(o); L0=len(o.instr); run(w,20)
    R['8 program decay']= (not w.objs[0].alive) or len(w.objs[0].instr)<L0
    # 9 CONSTRUCTION OF A SECOND (DIFFERENT) PROGRAM via WRITE
    w=World(seed=9); cons=P('WRITE','MATCH','WRITE','S0','WRITE','NOP','SPAWN'); target=P('MATCH','S0','NOP')
    o=Obj(cons,3,3,14.0,'con'); o.active=True; o.act_ttl=999; w.add(o); run(w,10)
    built=sum(1 for o in w.live() if o.cls=='child' and instr_eq(o.instr,target))
    R['9 construct 2nd program']= built>=1
    # 10 MUTUAL DEPENDENCY (A activates B, B activates A; remove A -> B dies)
    def mutpair(seed):
        w=World(H=5,Wd=5,seed=seed,inflow=1.5); A=Obj(P('BIND','ACTIVATE'),2,2,6.0,'A'); B=Obj(P('BIND','ACTIVATE'),2,2,6.0,'B'); w.add(A); w.add(B); return w
    w=mutpair(10); run(w,40); both_alive= w.objs[0].alive and w.objs[1].alive
    w.objs[0].alive=False   # remove A
    run(w,60); B_died= not w.objs[1].alive
    # control: with A retained, B survives the same extra 40
    w2=mutpair(11); run(w2,80); both_persist= w2.objs[0].alive and w2.objs[1].alive
    R['10 mutual dependency']= both_alive and B_died and both_persist
    # 11 DAMAGE & REPAIR (persistent copier reconstructs a deleted product from a template)
    w=World(seed=12); tmpl=P('S1','S2','S3','NOP'); cp=Obj(P('BIND','COPY','SPAWN'),3,3,40.0,'cp'); cp.active=True; cp.act_ttl=99999
    tobj=Obj(tmpl,3,3,40.0,'tmpl'); tobj.active=True; tobj.act_ttl=99999; w.add(cp); w.add(tobj)
    run(w,14); before=sum(1 for o in w.live() if o.cls=='child' and instr_eq(o.instr,tmpl))
    for o in w.live():   # DAMAGE: delete all product copies
        if o.cls=='child' and instr_eq(o.instr,tmpl): o.alive=False
    mid=sum(1 for o in w.live() if o.cls=='child' and instr_eq(o.instr,tmpl))
    w.objs[0].res=40.0
    run(w,16); after=sum(1 for o in w.live() if o.cls=='child' and instr_eq(o.instr,tmpl))
    R['11 damage & repair']= before>=1 and mid==0 and after>=1
    print(f"  {'capability':28s} result", flush=True)
    npass=0
    for k,v in R.items():
        print(f"  {k:28s} {'PASS' if v else 'FAIL'}", flush=True); npass+=bool(v)
    print(f"=== VALIDATION: {npass}/{len(R)} capabilities expressible ===", flush=True)
    print("=== CHEM_VALIDATION_DONE ===", flush=True)
    return R

if __name__=='__main__':
    validate()
