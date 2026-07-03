import json, urllib.request, torch
URL='http://127.0.0.1:8765'
def _p(path,obj):
    r=urllib.request.Request(URL+path,json.dumps(obj).encode(),{'Content-Type':'application/json'})
    o=json.loads(urllib.request.urlopen(r,timeout=900).read())
    if 'error' in o: raise RuntimeError(o['error'])
    return o
def gen(messages,max_new=45): return _p('/gen',{'messages':messages,'max_new':max_new})['text']
def encode(text): return torch.tensor(_p('/encode',{'text':text})['emb'])
def hidden(text,layer=36): return torch.tensor(_p('/hidden',{'text':text,'layer':layer})['rep'])
def judge(prompt): return _p('/judge',{'prompt':prompt})['logit']
