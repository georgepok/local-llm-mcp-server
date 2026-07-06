import io
PATH = '/home/pokazge/NativeEntity/world_pop.py'
src = io.open(PATH, encoding='utf-8').read()
a = "ctx = hist if mode=='oracle' else hist[-HWIN:]"
b = "ctx = (hist[:2] + hist[2:][-HWIN:]) if mode=='oracle' else hist[-HWIN:]"
c = "resp=gen(ctx); hist.append({'role':'assistant','content':resp})"
d = "resp=gen(ctx); torch.cuda.empty_cache(); hist.append({'role':'assistant','content':resp})"
n = 0
if a in src: src = src.replace(a, b); n += 1
if c in src: src = src.replace(c, d); n += 1
io.open(PATH, 'w', encoding='utf-8').write(src)
print('EDITED', n)
