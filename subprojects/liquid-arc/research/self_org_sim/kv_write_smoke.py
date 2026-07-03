# KV-WRITE smoke: validate the elegant actuation mechanism BEFORE training. Monkeypatch eager_attention_forward so any
# attention module carrying ._kv_inj=(k_inj,v_inj) gets those (key,value) PREPENDED to its K/V and the additive mask padded
# with M attendable columns — the model attends to the Liquid's injected memory through its OWN softmax ("attention on
# attention"), zero weight-splice. Checks: ._kv_inj=None => EXACT baseline (no-op); injection shifts next-token logits
# MONOTONICALLY with magnitude; clearing restores baseline. Targets full-attn layers 23/27/31 (same as the LoRA).
import sys, os
import torch, torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen3_5.modeling_qwen3_5 as Q5                          # the ACTUAL modeling module (Qwen3_5Attention)
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
torch.manual_seed(0); dev = torch.device('cuda'); MODEL = '/home/pokazge/models/Qwen3.6-27B'; TGT = [23, 27, 31]; M = 4
_orig = Q5.eager_attention_forward                                                # 'eager' is NOT in ALL_ATTENTION_FUNCTIONS, so the module-local fallback is used -> patching it takes effect
def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):   # prepend injected KV; pad additive mask
    inj = getattr(module, '_kv_inj', None)
    if inj is not None:
        ki, vi = inj; key = torch.cat([ki.to(key.dtype), key], dim=2); value = torch.cat([vi.to(value.dtype), value], dim=2)
        if attention_mask is not None:
            pad = torch.zeros(*attention_mask.shape[:-1], ki.shape[2], dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([pad, attention_mask], dim=-1)
    return _orig(module, query, key, value, attention_mask, scaling, dropout, **kw)
Q5.eager_attention_forward = patched
print('loading 27B ...', flush=True)
cfg = AutoConfig.from_pretrained(MODEL); cfg.language_model_only = True; tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, config=cfg, dtype=torch.bfloat16, device_map={'': 0}, low_cpu_mem_usage=True, attn_implementation='eager').eval()
for p in model.parameters(): p.requires_grad = False
nkv = model.config.num_key_value_heads; hd = getattr(model.config, 'head_dim', model.config.hidden_size // model.config.num_attention_heads)
print('num_key_value_heads=%d  head_dim=%d' % (nkv, hd), flush=True)
mods = []
for L in TGT:
    sa = model.model.layers[L].self_attn; sa._kv_inj = None; mods.append(sa)
    print('layer %d self_attn=%s  has q_proj=%s  num_kv_groups=%s' % (L, type(sa).__name__, hasattr(sa, 'q_proj'), getattr(sa, 'num_key_value_groups', '?')), flush=True)
def setinj(scale, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    for m in mods:
        if scale is None: m._kv_inj = None
        else:
            ki = torch.randn(1, nkv, M, hd, generator=g, device=dev, dtype=torch.bfloat16) * scale
            vi = torch.randn(1, nkv, M, hd, generator=g, device=dev, dtype=torch.bfloat16) * scale
            m._kv_inj = (ki, vi)
ids = tok('Explain why the sky is blue in one paragraph.', return_tensors='pt').input_ids.to(dev)
with torch.no_grad():
    setinj(None); base = model(ids).logits[0, -1].float()
    print('\n=== KV-WRITE smoke (next-token logit shift vs injection scale) ===', flush=True)
    for s in [0.0, 0.5, 2.0, 8.0]:
        setinj(s); lg = model(ids).logits[0, -1].float()
        kl = float(F.kl_div(F.log_softmax(lg, -1), F.softmax(base, -1), reduction='sum'))
        top = tok.decode(lg.argmax()); print('  scale %4.1f : KL(base||inj)=%.4f   argmax=%r' % (s, kl, top), flush=True)
    setinj(None); base2 = model(ids).logits[0, -1].float()
    print('  clear->identity: max|Δlogit|=%.2e  (must be ~0)' % float((base - base2).abs().max()), flush=True)
print('read: scale 0.0 ~ identity (injected zeros steal ~no mass); KL rises with scale = the model ATTENDS to injected KV;', flush=True)
print('clear restores baseline. Mechanism validated => build the Liquid->KV generator + in-loop distillation next.', flush=True)
print('=== ALL_DONE ===', flush=True)
