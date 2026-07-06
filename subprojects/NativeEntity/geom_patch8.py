import io
PATH = '/home/pokazge/NativeEntity/world_pop.py'
src = io.open(PATH, encoding='utf-8').read()
old = "        st = torch.stack([hs[L][0].float().mean(0) for L in LAYERS])   # [n_layers, D]"
new = ("        _pool = os.environ.get('GEO_POOL', 'last')\n"
       "        st = torch.stack([(hs[L][0][-1].float() if _pool == 'last' else hs[L][0].float().mean(0)) for L in LAYERS])   # [n_layers, D]")
assert old in src, 'getstack line not found'
src = src.replace(old, new, 1)
io.open(PATH, 'w', encoding='utf-8').write(src)
print('PATCHED_OK')
