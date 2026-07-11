import json, re, glob, os
TD="/private/tmp/claude-502/-Users-George-Documents-GitHub-local-llm-mcp-server/a825c23a-9feb-4d04-8285-3cec16a7162e/tasks"
# E2/E3 subagent task ids (promptId eb1d5427 batch)
E2_IDS=['a6137a93ad0fbb84f','ae7187f3ece2928cf','ae6da2824e25a9f3e']
E3_IDS=['a0d087e7d83fe3781','a1cdbdd6d0ec237da','af6590f3e56de0dcf']

def final_text(path):
    txt=None
    for line in open(path):
        line=line.strip()
        if not line: continue
        try: o=json.loads(line)
        except: continue
        msg=o.get('message',o)
        if isinstance(msg,dict) and msg.get('role')=='assistant':
            c=msg.get('content')
            if isinstance(c,list):
                for b in c:
                    if isinstance(b,dict) and b.get('type')=='text': txt=b['text']
            elif isinstance(c,str): txt=c
    return txt

def extract_genomes(text):
    if not text: return []
    blocks=re.findall(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if not blocks:
        blocks=re.findall(r'(\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})', text, re.DOTALL)
    out=[]
    for b in blocks:
        try:
            g=json.loads(b)
            if isinstance(g,dict) and 'family' in g and 'hidden_dim' in g:
                out.append(g)
        except: pass
    return out

def collect(ids, env):
    res=[]
    for i,tid in enumerate(ids):
        p=os.path.join(TD,tid+'.output')
        if not os.path.exists(p): print("MISSING",tid); continue
        gs=extract_genomes(final_text(p))
        for j,g in enumerate(gs):
            plastic=g.get('plasticity',{}).get('enabled',False)
            res.append({'tag':f"{env}_claude{i+1}_{'D' if plastic else 'C'}{j}",'src':tid,'genome':g})
        print(f"{env} agent {i+1} ({tid[:8]}): {len(gs)} genomes  families={[x.get('family') for x in gs]} rules={[x.get('plasticity',{}).get('rule') for x in gs]}")
    return res

e2=collect(E2_IDS,'E2'); e3=collect(E3_IDS,'E3')
json.dump(e2, open('/tmp/e2_genomes.json','w'), indent=1)
json.dump(e3, open('/tmp/e3_genomes.json','w'), indent=1)
print(f"\nSAVED e2_genomes.json ({len(e2)}) e3_genomes.json ({len(e3)})")
