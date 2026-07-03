# Probe the KV-cache structure of the Qwen3.6 DeltaNet+GQA hybrid before building the KV-compressor: which layer
# entries are real K/V (full-attention) vs DeltaNet recurrent states, their shapes, and how to index the full-attn KV.
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
M = '/home/pokazge/models/Qwen3.6-27B'
cfg = AutoConfig.from_pretrained(M); cfg.language_model_only = True
print('=== config layer-structure fields ===', flush=True)
for a in ['num_hidden_layers', 'layer_types', 'full_attention_interval', 'linear_attention_indices', 'num_attention_heads', 'num_key_value_heads', 'head_dim', 'hidden_size', 'linear_num_value_heads', 'linear_num_key_heads', 'linear_key_head_dim', 'linear_value_head_dim']:
    if hasattr(cfg, a): print('  cfg.%s = %s' % (a, getattr(cfg, a)))
tok = AutoTokenizer.from_pretrained(M)
print('loading model ...', flush=True)
model = AutoModelForCausalLM.from_pretrained(M, config=cfg, dtype=torch.bfloat16, device_map={'': 0}, low_cpu_mem_usage=True, attn_implementation='eager').eval()
ids = tok('The quick brown fox jumps over the lazy dog and keeps running through the field.', return_tensors='pt').input_ids.to('cuda')
print('input tokens:', ids.shape, flush=True)
with torch.no_grad(): out = model(ids, use_cache=True)
pkv = out.past_key_values
print('\n=== past_key_values: type=%s ===' % type(pkv).__name__, flush=True)
try: n = len(pkv); print('len =', n)
except Exception as e: n = 0; print('no len:', repr(e))
def describe(x, d=0):
    if x is None: return 'None'
    if hasattr(x, 'shape'): return 'Tensor%s %s' % (tuple(x.shape), x.dtype)
    if isinstance(x, (tuple, list)): return '%s[%s]' % (type(x).__name__, ', '.join(describe(e) for e in x))
    return type(x).__name__
for i in range(n):
    try: print('  layer %2d: %s' % (i, describe(pkv[i])))
    except Exception as e: print('  layer %2d: <%r>' % (i, e))
# also expose any cache attributes (key_cache/value_cache lists, conv states)
for attr in ['key_cache', 'value_cache', 'layers', 'conv_states', 'recurrent_states', 'ssm_states']:
    if hasattr(pkv, attr):
        v = getattr(pkv, attr)
        try: print('pkv.%s : list[%d] -> %s' % (attr, len(v), describe(v[0]) if len(v) else 'empty'))
        except Exception: print('pkv.%s : %s' % (attr, describe(v)))
print('=== ALL_DONE ===')
