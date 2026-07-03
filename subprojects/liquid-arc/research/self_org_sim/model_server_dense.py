# Persistent Qwen3.6-27B (DENSE) server for the manifold pipeline. Text-only (language_model_only,
# vision tower skipped -> 53.8GB). Run with PYTHONPATH=/home/pokazge/dense_pylib (transformers 5.x,
# qwen3_5 support), layered over the venv torch 2.10.0+cu13. NO GR00T dependency beyond the torch.
# Dense => LoRA integrates cleanly into every token (the whole point of the switch). Ops: /gen
# /manifold /value /hidden. Manifold read layer = 32 (mid of 64).
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import json, torch, torch.nn.functional as F
from http.server import BaseHTTPRequestHandler, HTTPServer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
M = '/home/pokazge/models/Qwen3.6-27B'; dev = torch.device('cuda'); PORT = 8765; MANIFOLD_LAYER = 32
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
torch.set_float32_matmul_precision('high')
print('[server] loading Qwen3.6-27B (dense, text-only) ...', flush=True)
cfg = AutoConfig.from_pretrained(M); cfg.language_model_only = True
tok = AutoTokenizer.from_pretrained(M)
if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
model = AutoModelForCausalLM.from_pretrained(M, config=cfg, dtype=torch.bfloat16, device_map={'': 0}, low_cpu_mem_usage=True, attn_implementation='eager').eval()  # eager = proven stable on sm121/qwen3_5; sdpa attn dies after READY here. Slower but reliable.
for p in model.parameters(): p.requires_grad = False
DIGITS = [tok(str(i), add_special_tokens=False).input_ids[-1] for i in range(10)]
print('[server] loaded: %.1fB params, %.1fGB' % (sum(p.numel() for p in model.parameters()) / 1e9, torch.cuda.memory_allocated() / 1e9), flush=True)

def tmpl(m):
    try: return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def gen(msgs, mx, temp=0.0):
    e = tok(tmpl(msgs), return_tensors='pt').to(dev)
    kw = dict(max_new_tokens=int(mx), pad_token_id=tok.pad_token_id)
    kw.update(dict(do_sample=True, temperature=float(temp), top_p=0.95) if temp and float(temp) > 0 else dict(do_sample=False))
    o = model.generate(e.input_ids, attention_mask=e.attention_mask, **kw)
    return tok.decode(o[0, e.input_ids.shape[1]:], skip_special_tokens=True).split('</think>')[-1].strip()

@torch.no_grad()
def manifold(messages, layer):
    ids = tok(tmpl(messages), return_tensors='pt').to(dev)
    hs = model(ids.input_ids, output_hidden_states=True).hidden_states[int(layer)][0, -1].float()
    return F.normalize(hs, dim=0).cpu().tolist()

MULTI_LAYERS = list(range(4, 65, 4))   # depth-trajectory: 16 layers spanning the network (every 4th)
@torch.no_grad()
def manifold_multi(messages, layers):  # last-token hidden across a SPAN of layers = the computational PROCESS, not one snapshot
    ids = tok(tmpl(messages), return_tensors='pt').to(dev)
    hs = model(ids.input_ids, output_hidden_states=True).hidden_states
    return [F.normalize(hs[L][0, -1].float(), dim=0).cpu().tolist() for L in layers]

@torch.no_grad()
def manifold_attn(messages):   # the ROUTING: how the last token attended over the context, per full-attention layer
    ids = tok(tmpl(messages), return_tensors='pt').to(dev)
    out = model(ids.input_ids, output_attentions=True)
    feats = []
    for a in out.attentions:
        if a is None: continue                      # DeltaNet (linear-attn) layers expose no weights
        w = a[0].mean(0)[-1].float()                # mean over heads, LAST-token query -> [n_key] attention distribution
        nk = w.shape[0]; pos = torch.arange(nk, device=w.device).float()
        ent = float(-(w * (w + 1e-9).log()).sum())                 # how focused vs diffuse
        rec = float((w * pos).sum() / max(nk - 1, 1))              # mean attended position (recency)
        peak = float(w.max())                                      # peakiness
        last10 = float(w[int(nk * 0.9):].sum())                    # mass on most-recent 10% (recent context/response)
        feats.append([ent, rec, peak, last10])
    return feats                                    # [n_full_attn_layers, 4] routing-feature depth-trajectory

CTX_CAP = 160
@torch.no_grad()
def manifold_ctx(messages, layer, cap):   # ALL context+response token hiddens (recent window) -> the belief ATTENDS over them
    ids = tok(tmpl(messages), return_tensors='pt').to(dev)
    hs = model(ids.input_ids, output_hidden_states=True).hidden_states[int(layer)][0]
    return hs[-int(cap):].float().cpu().tolist()

@torch.no_grad()
def manifold_gen(context_msgs, response, layer):   # GENERATION TRAJECTORY: hidden at EVERY response token = the path the
    prefix = tmpl(context_msgs)                    # representation traces WHILE producing the answer (the process, not a snapshot)
    p_ids = tok(prefix, return_tensors='pt').input_ids.to(dev)
    r_ids = tok(response or '.', return_tensors='pt', add_special_tokens=False).input_ids.to(dev)
    full = torch.cat([p_ids, r_ids], 1)
    hs = model(full, output_hidden_states=True).hidden_states[int(layer)][0]   # [seq, d_m]
    return hs[p_ids.shape[1]:].float().cpu().tolist()                           # [n_resp_tokens, d_m]

FLOW_LAYERS = list(range(0, 65, 4))    # [0,4,...,64] -> 16 inter-block residual-stream DELTAS (the flow)
@torch.no_grad()
def manifold_flow(messages, layers):   # RAW inter-layer deltas = what each block ADDS to the residual stream (genuine flow, magnitude KEPT)
    ids = tok(tmpl(messages), return_tensors='pt').to(dev)
    hs = model(ids.input_ids, output_hidden_states=True).hidden_states
    raw = [hs[L][0, -1].float() for L in layers]
    deltas = torch.stack([raw[i + 1] - raw[i] for i in range(len(raw) - 1)])
    return deltas.cpu().tolist()

@torch.no_grad()
def hidden(text, layer):
    ids = tok(text or '.', return_tensors='pt', add_special_tokens=False).input_ids.to(dev)
    if ids.shape[1] == 0: ids = tok('.', return_tensors='pt').input_ids.to(dev)
    hs = model(ids, output_hidden_states=True).hidden_states[int(layer)][0].float().mean(0)
    return F.normalize(hs, dim=0).cpu().tolist()

@torch.no_grad()
def value(prompt):
    ids = tok(tmpl([{'role': 'user', 'content': prompt}]), return_tensors='pt').to(dev)
    lg = model(ids.input_ids).logits[0, -1]
    p = F.softmax(lg[torch.tensor(DIGITS, device=dev)], dim=0)
    return float((p * torch.arange(10., device=dev)).sum())

OPS = {
    '/gen': lambda b: {'text': gen(b['messages'], b.get('max_new', 45), b.get('temp', 0.0))},
    '/manifold': lambda b: {'h': manifold(b['messages'], b.get('layer', MANIFOLD_LAYER))},
    '/manifold_multi': lambda b: {'h': manifold_multi(b['messages'], b.get('layers', MULTI_LAYERS))},
    '/manifold_flow': lambda b: {'h': manifold_flow(b['messages'], b.get('layers', FLOW_LAYERS))},
    '/manifold_gen': lambda b: {'h': manifold_gen(b['context'], b['response'], b.get('layer', MANIFOLD_LAYER))},
    '/manifold_ctx': lambda b: {'h': manifold_ctx(b['messages'], b.get('layer', MANIFOLD_LAYER), b.get('cap', CTX_CAP))},
    '/manifold_attn': lambda b: {'h': manifold_attn(b['messages'])},
    '/hidden': lambda b: {'rep': hidden(b['text'], b.get('layer', MANIFOLD_LAYER))},
    '/value': lambda b: {'v': value(b['prompt'])},
}
class H(BaseHTTPRequestHandler):
    def log_message(s, *a): pass
    def do_GET(s):
        b = b'{"ok":true,"model":"Qwen3.6-27B","layer":%d}' % MANIFOLD_LAYER
        s.send_response(200); s.send_header('Content-Length', str(len(b))); s.end_headers(); s.wfile.write(b)
    def do_POST(s):
        try:
            n = int(s.headers['Content-Length']); body = json.loads(s.rfile.read(n)); out = OPS.get(s.path, lambda b: {'error': 'unknown path'})(body)
        except Exception as e: out = {'error': repr(e)}
        b = json.dumps(out).encode(); s.send_response(200); s.send_header('Content-Type', 'application/json'); s.send_header('Content-Length', str(len(b))); s.end_headers(); s.wfile.write(b)
print('[server] READY on %d (manifold layer %d)' % (PORT, MANIFOLD_LAYER), flush=True)
HTTPServer(('127.0.0.1', PORT), H).serve_forever()
