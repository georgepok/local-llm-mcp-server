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
def rep(s):
    for pat,r in [(r'//[^\n]*',''),(r',(\s*[}\]])',r'\1')]: s=re.sub(pat,r,s)
    return s
def ev(raw):
    rawok=False
    try: json.loads(raw); rawok=True
    except: pass
    g=None
    try: g=json.loads(rep(raw))
    except: return None
    if not isinstance(g,dict) or 'family' not in g: return None
    w=g.setdefault('weights',{}); fill=[]
    for k in ('recurrent','input','readout'):
        if k not in w or not isinstance(w[k],dict): w[k]=DEF[k]; fill.append(k)
    g.setdefault('input_dim',14); g.setdefault('output_dim',21)
    pl=g.get('plasticity',{})
    if isinstance(pl.get('targets'),list) and pl['targets'] and isinstance(pl['targets'][0],list): pl['targets']=pl['targets'][0]
    net,msg=compile_and_check(g)
    if not net: return dict(rawok=rawok,fill=fill,comp=msg,on='n/a',held='n/a',plastic=pl.get('enabled'),rule=pl.get('rule'))
    on,_=eval_genome(g,online=True,neps=120); hd,_=eval_genome(g,online=True,held=True,neps=120)
    return dict(rawok=rawok,fill=fill,comp='OK',on=on['late']['survival'],held=hd['late']['survival'],onp=on['late']['probe'],plastic=pl.get('enabled'),rule=pl.get('rule'))
print("=== QWEN3.6-27B RELIABILITY: 4 sampled (temp0.7) end-to-end syntheses ===", flush=True)
print("  (greedy sample already: plastic online surv=1.00 held=1.00 robust=1.00)", flush=True)
plastic_surv=[]
for k in range(4):
    bs=blocks(f'synthrep_{k}.txt')
    for raw in bs:
        r=ev(raw)
        if r is None: continue
        if r['plastic']:
            print(f"  sample{k} PLASTIC({r['rule']}): rawJSON={r['rawok']} fill={r['fill']} comp={r['comp']} online_surv={r['on']} online_probe={r.get('onp','?')} HELD_surv={r['held']}", flush=True)
            if r['comp']=='OK' and r['on']!='n/a': plastic_surv.append(r['on'])
import statistics as st
if plastic_surv:
    allv=[1.00]+plastic_surv  # include greedy
    print(f"  >> plastic online survival across {len(allv)} syntheses (greedy+4 sampled): {sorted(allv)} median={st.median(allv):.2f}", flush=True)
print("=== REPLICATE_EVAL_DONE ===", flush=True)
