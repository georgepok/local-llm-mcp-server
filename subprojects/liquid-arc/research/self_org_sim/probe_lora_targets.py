# Probe Qwen3.6-27B for LoRA targets: which layers have o_proj / down_proj, shapes, MoE-or-dense (down_proj per-expert?)
import os, torch
from transformers import AutoConfig, AutoModelForCausalLM
dev = torch.device('cuda'); MODEL = '/home/pokazge/models/Qwen3.6-27B'
cfg = AutoConfig.from_pretrained(MODEL); cfg.language_model_only = True
print('loading 27B ...', flush=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, config=cfg, dtype=torch.bfloat16, device_map={'': 0}, low_cpu_mem_usage=True, attn_implementation='eager').eval()
opj = {}; dpj = {}; other = {}
for n, m in model.named_modules():
    if not hasattr(m, 'weight') or not hasattr(m.weight, 'shape'): continue
    if n.endswith('o_proj'): opj[n] = tuple(m.weight.shape)
    elif n.endswith('down_proj'): dpj[n] = tuple(m.weight.shape)
print('=== num o_proj=%d  num down_proj=%d ===' % (len(opj), len(dpj)), flush=True)
print('--- o_proj (attention output) ---', flush=True)
for n in sorted(opj)[:6] + sorted(opj)[-4:]: print('  ', n, opj[n], flush=True)
print('--- down_proj (MLP) ---', flush=True)
for n in sorted(dpj)[:6] + sorted(dpj)[-4:]: print('  ', n, dpj[n], flush=True)
# detect layer indices present for each
import re
oi = sorted(set(int(re.search(r'layers\.(\d+)\.', n).group(1)) for n in opj if re.search(r'layers\.(\d+)\.', n)))
di = sorted(set(int(re.search(r'layers\.(\d+)\.', n).group(1)) for n in dpj if re.search(r'layers\.(\d+)\.', n)))
print('o_proj layer indices:', oi, flush=True)
print('down_proj layer indices (count %d):' % len(di), di[:20], '...' if len(di) > 20 else '', flush=True)
# MoE check: is down_proj per-expert (multiple per layer)?
from collections import Counter
dl = Counter(int(re.search(r'layers\.(\d+)\.', n).group(1)) for n in dpj if re.search(r'layers\.(\d+)\.', n))
print('down_proj per layer (MoE if >1):', dict(list(dl.items())[:4]), flush=True)
print('=== PROBE_DONE ===', flush=True)
