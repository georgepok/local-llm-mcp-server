import os, re, json
os.environ['ME_MODE']='none'
from micro_entity import compile_and_check, eval_genome
DEF={'recurrent':{'gen':'sparse','seed':1,'scale':0.9,'spectral_radius':0.95,'sparsity':0.2},
     'input':{'gen':'dense','seed':2,'scale':0.6},'readout':{'gen':'dense','seed':3,'scale':0.1}}
def extract(path):
    t=open(path).read()
    m=re.search(r'```json\s*(\{.*?\})\s*```', t, re.DOTALL) or re.search(r'(\{.*\})', t, re.DOTALL)
    return m.group(1) if m else None
def repair(s):
    reps=[(r'\bt\.6\b','0.6'),(r'"leeak"','"leak"'),(r'"d dynamics"','"dynamics"'),
          (r'"slow_hidden":\s*false','"slow_hidden": 0'),(r'\[\["readout"\]','["readout"]'),
          (r',(\s*[}\]])',r'\1')]
    n=0
    for pat,rep in reps:
        s2=re.sub(pat,rep,s)
        if s2!=s: n+=1; s=s2
    return s,n
print("=== PART 6 SERIALIZATION (reverse control, w/ default-fill for omitted weight sub-dicts) ===", flush=True)
for name,path in [('qwen7b','serialize_qwen7b.txt'),('qwen1.5b','serialize_qwen1.5b.txt')]:
    raw=extract(path); rawok=False
    try: json.loads(raw); rawok=True
    except: pass
    rep,nfix=repair(raw); g=None
    try: g=json.loads(rep)
    except: pass
    parseok=isinstance(g,dict); filled=[]
    if parseok:
        w=g.setdefault('weights',{})
        for k in ('recurrent','input','readout'):
            if k not in w: w[k]=DEF[k]; filled.append(k)
        g.setdefault('input_dim',14); g.setdefault('output_dim',21)
        if isinstance(g.get('plasticity',{}).get('targets'),list) and g['plasticity']['targets'] and isinstance(g['plasticity']['targets'][0],list):
            g['plasticity']['targets']=g['plasticity']['targets'][0]
    comp='n/a'; perf='n/a'
    if parseok:
        net,msg=compile_and_check(g); comp='OK' if net else msg
        if net:
            r,_=eval_genome(g, online=True, neps=120); perf=(r['late']['survival'] if r else 'n/a')
    print(f"  {name}: raw_JSON_valid={rawok} repairs={nfix} weights_omitted={filled} compiler={comp} E1_survival={perf}", flush=True)
print("=== SERIAL_DONE ===", flush=True)
