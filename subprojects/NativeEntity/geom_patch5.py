import io
PATH = '/home/pokazge/NativeEntity/world_pop.py'
src = io.open(PATH, encoding='utf-8').read()
anchor = "    held = torch.tensor([w['held'] for w in worlds]); tr = ~held\n"
assert anchor in src, 'anchor not found'
if 'held-principle base-rate' not in src:
    ins = (anchor +
           "    import collections as _cl\n"
           "    _hb = _cl.Counter([worlds[i]['a'] for i in range(len(worlds)) if worlds[i]['held']])\n"
           "    _base = max(_hb.values()) / float(sum(_hb.values()))\n"
           "    print('GEOMN held-principle base-rate (majority action) = %.3f' % _base, flush=True)\n")
    src = src.replace(anchor, ins, 1)
    io.open(PATH, 'w', encoding='utf-8').write(src)
    print('PATCHED_OK')
else:
    print('ALREADY')
