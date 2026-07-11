import os, re, json
os.environ['ME_MODE']='none'
from micro_entity import compile_and_check, eval_genome
DEF={'recurrent':{'gen':'sparse','seed':1,'scale':0.9,'spectral_radius':0.95,'sparsity':0.2},
     'input':{'gen':'dense','seed':2,'scale':0.6},'readout':{'gen':'dense','seed':3,'scale':0.1}}
def blocks(path):
    if not os.path.exists(path): return []
    t=open(path).read()
    bs=re.findall(r'```json\s*(\{.*?\})\s*```', t, re.DOTALL)
    if not bs: bs=re.findall(r'(\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})', t, re.DOTALL)
    return bs
def repair(s):
    reps=[(r'\bt\.6\b','0.6'),(r'"leeak"','"leak"'),(r'"d dynamics"','"dynamics"'),
          (r'"slow_hidden":\s*false','"slow_hidden": 0'),(r'\[\["readout"\]','["readout"]'),
          (r'//[^\n]*',''),(r',(\s*[}\]])',r'\1')]
    n=0
    for pat,rep in reps:
        s2=re.sub(pat,rep,s)
        if s2!=s: n+=1; s=s2
    return s,n
def evalblock(raw):
    rawok=False
    try: json.loads(raw); rawok=True
    except: pass
    rep,nfix=repair(raw); g=None
    try: g=json.loads(rep)
    except: pass
    if not isinstance(g,dict): return dict(rawok=rawok,repairs=nfix,parse=False,filled=[],comp='PARSE_FAIL',surv='n/a',rule=None,plastic=None)
    w=g.setdefault('weights',{}); filled=[]
    for k in ('recurrent','input','readout'):
        if k not in w or not isinstance(w[k],dict): w[k]=DEF[k]; filled.append(k)
    g.setdefault('input_dim',14); g.setdefault('output_dim',21); g.setdefault('hidden_dim',g.get('hidden_dim',64))
    pl=g.get('plasticity',{})
    if isinstance(pl.get('targets'),list) and pl['targets'] and isinstance(pl['targets'][0],list): pl['targets']=pl['targets'][0]
    net,msg=compile_and_check(g); comp='OK' if net else msg; surv='n/a'
    if net:
        r,_=eval_genome(g, online=True, neps=120); surv=(r['late']['survival'] if r else 'n/a')
    return dict(rawok=rawok,repairs=nfix,parse=True,filled=filled,comp=comp,surv=surv,
                rule=(pl.get('rule') if pl.get('enabled') else 'FIXED'),plastic=pl.get('enabled'))

print("=== QWEN3.6-27B — capability vs serialization ===", flush=True)
print("-- REVERSE CONTROL (gold design given, emit DSL) --", flush=True)
for raw in blocks('serialize_qwen27b.txt')[:1]:
    r=evalblock(raw); print(f"   serialize: raw_JSON={r['rawok']} repairs={r['repairs']} weights_filled={r['filled']} compiler={r['comp']} E1_surv={r['surv']}", flush=True)
print("-- FORWARD SYNTHESIS (end-to-end, expect FIXED + PLASTIC) --", flush=True)
bs=blocks('synth_qwen27b.txt')
print(f"   extracted {len(bs)} genome block(s)", flush=True)
for i,raw in enumerate(bs[:4]):
    r=evalblock(raw); print(f"   synth[{i}] rule={r['rule']}: raw_JSON={r['rawok']} repairs={r['repairs']} weights_filled={r['filled']} compiler={r['comp']} E1_surv={r['surv']}", flush=True)
print("=== QWEN27B_EVAL_DONE ===", flush=True)
