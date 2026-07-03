# Persistent 30B + bge + resident Liquid (SteerController) + LiquidLoRA hooks. Loads ONCE.
# Stateless ops: /gen /encode /hidden /judge.  Stateful Liquid ops: /reset /observe /gen(steered).
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import json, torch, torch.nn.functional as F
from http.server import BaseHTTPRequestHandler, HTTPServer
from train_steer_controller import SteerController, encode_goal
from train_liquid_lora2 import LiquidLoRA
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

M = '/home/pokazge/models/Qwen3-30B-A3B'
dev = torch.device('cuda'); PORT = 8765
ADAPTER = '/home/pokazge/checkpoints/ll2_unify2.pt'  # stronger actuation (cap1.0 + sharper teacher)

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.set_float32_matmul_precision('high')

print('[server] loading 30B + bge (one time)...', flush=True)
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
model = AutoModelForCausalLM.from_pretrained(
    M, dtype=torch.bfloat16, trust_remote_code=True,
    low_cpu_mem_usage=True, device_map={'': 0}).eval()
for p in model.parameters():
    p.requires_grad = False
etok = AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5')
emod = AutoModel.from_pretrained('BAAI/bge-small-en-v1.5').to(dev).eval()
YES = tok(' Yes', add_special_tokens=False).input_ids[-1]
NO = tok(' No', add_special_tokens=False).input_ids[-1]

# ---- manifold ACTUATION: write the held goal-direction back into the residual stream (read hidden -> write residual) ----
STEER_LAYER = 24
M_STEER = {'vec': None, 'alpha': 0.0, 'on': False}
def _msteer_hook(module, inp, out):
    if not M_STEER['on'] or M_STEER['vec'] is None:
        return out
    h = out[0] if isinstance(out, tuple) else out          # [B,T,d] residual stream
    v = M_STEER['vec'].to(h.device)                          # [d] unit goal-direction
    add = M_STEER['alpha'] * h.float().norm(dim=-1, keepdim=True) * v.view(1, 1, -1)   # fraction of token norm
    h2 = (h.float() + add).to(h.dtype)
    return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
model.model.layers[STEER_LAYER - 1].register_forward_hook(_msteer_hook)

# ---- resident Liquid + LiquidLoRA (registered once) ----
CTRL = None
LORA = None

def load_adapter(path):
    global CTRL, LORA
    if LORA is not None:
        for h in LORA.handles:
            h.remove()
    ck = torch.load(path, weights_only=False, map_location='cpu')
    a = ck['args']
    CTRL = SteerController(d_llm=model.config.hidden_size, d=a['d'], K=a['K'],
                          use_slow=a.get('use_slow', True), n_inject=1).to(dev)
    CTRL.load_state_dict(ck['controller'], strict=False)
    CTRL.eval()
    layers = [int(x) for x in str(a['lora_layers']).split(',')]
    projs = [s.strip() for s in str(a['lora_proj']).split(',')]
    LORA = LiquidLoRA(model, layers, projs, d_ctrl=a['d'], scale=a.get('lora_scale', 1.0)).to(dev)
    LORA.cap_rel = a.get('cap_rel', 0.5) or None
    LORA.load_state_dict(ck['lora'])
    LORA.register()
    LORA.active = False
    return {'loaded': path, 'layers': layers, 'projs': projs}

def tmpl(m):
    try:
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def gen(msgs, mx, steered=False):
    if steered and LORA is not None:
        LORA.active = True
    e = tok(tmpl(msgs), return_tensors='pt').to(dev)
    o = model.generate(e.input_ids, attention_mask=e.attention_mask,
                       max_new_tokens=int(mx), do_sample=False, pad_token_id=tok.pad_token_id)
    if LORA is not None:
        LORA.active = False
    return tok.decode(o[0, e.input_ids.shape[1]:], skip_special_tokens=True).split('</think>')[-1].strip()

@torch.no_grad()
def enc(t):
    return encode_goal(t or '.', etok, emod, dev).cpu().tolist()

@torch.no_grad()
def hid(t, layer):
    ids = tok(t or '.', return_tensors='pt', add_special_tokens=False).input_ids.to(dev)
    if ids.shape[1] == 0:
        ids = tok('.', return_tensors='pt').input_ids.to(dev)
    hs = model(ids, output_hidden_states=True).hidden_states[int(layer)][0].float().mean(0)
    return F.normalize(hs, dim=0).cpu().tolist()

@torch.no_grad()
def judge(p):
    ids = tok(tmpl([{'role': 'user', 'content': p}]), return_tensors='pt').to(dev)
    lg = model(ids.input_ids).logits[0, -1]
    return float(lg[YES] - lg[NO])

@torch.no_grad()
def manifold(messages, layer):  # the LLM's MANIFOLD position: last-token hidden of the live context (no text restatement)
    ids = tok(tmpl(messages), return_tensors='pt').to(dev)
    hs = model(ids.input_ids, output_hidden_states=True).hidden_states[int(layer)][0, -1].float()
    return F.normalize(hs, dim=0).cpu().tolist()

DIGITS = [tok(str(i), add_special_tokens=False).input_ids[-1] for i in range(10)]
@torch.no_grad()
def value(prompt):  # LLM's RICH agentic-quality judgment as a smooth scalar (expected digit 0-9), not a cosine
    ids = tok(tmpl([{'role': 'user', 'content': prompt}]), return_tensors='pt').to(dev)
    lg = model(ids.input_ids).logits[0, -1]
    p = F.softmax(lg[torch.tensor(DIGITS, device=dev)], dim=0)
    return float((p * torch.arange(10., device=dev)).sum())

def set_steer(vec, alpha):  # set the held goal-direction the species writes back into the manifold
    M_STEER['vec'] = F.normalize(torch.tensor(vec, dtype=torch.float32), dim=0) if vec is not None else None
    M_STEER['alpha'] = float(alpha)
    return {'alpha': M_STEER['alpha'], 'has_vec': M_STEER['vec'] is not None, 'layer': STEER_LAYER}

@torch.no_grad()
def gen_msteer(msgs, mx):  # generate with the manifold goal-direction steering active
    M_STEER['on'] = True
    try:
        e = tok(tmpl(msgs), return_tensors='pt').to(dev)
        o = model.generate(e.input_ids, attention_mask=e.attention_mask, max_new_tokens=int(mx), do_sample=False, pad_token_id=tok.pad_token_id)
        return tok.decode(o[0, e.input_ids.shape[1]:], skip_special_tokens=True).split('</think>')[-1].strip()
    finally:
        M_STEER['on'] = False

@torch.no_grad()
def reset(mission):
    CTRL.reset_episode(1, dev)
    if CTRL.use_slow:
        CTRL.slow_step(encode_goal(mission, etok, emod, dev).unsqueeze(0))
    return {'reset': mission}

def set_gain(scale, cap):  # runtime LoRA magnitude control (scale saturates, cap_rel = the ceiling)
    LORA.scale = float(scale)
    LORA.cap_rel = None if cap in (None, 'none', 'None') else float(cap)
    LORA.debug = True
    return {'scale': LORA.scale, 'cap_rel': LORA.cap_rel}

def relmag():  # mean |delta|/|out| measured on the last steered forward (debug must be on)
    return {'rel_mag': [round(float(x), 3) for x in LORA.last_rel], 'mean': round(sum(LORA.last_rel) / max(1, len(LORA.last_rel)), 3)}

@torch.no_grad()
def observe(text):  # Liquid integrates a turn's goal/reasoning, HOLDS it, sets the adapter from belief
    z = encode_goal(text, etok, emod, dev)
    h = CTRL.dyn_state(z.unsqueeze(0))
    LORA.set_state(h)
    held = F.normalize(CTRL.g_head(h.flatten(1)).squeeze(0), dim=0)
    return {'belief_norm': float(h.norm()), 'held_self_cos': float((held * F.normalize(z, dim=0)).sum())}

OPS = {
    '/gen': lambda b: {'text': gen(b['messages'], b.get('max_new', 45), b.get('steered', False))},
    '/encode': lambda b: {'emb': enc(b['text'])},
    '/hidden': lambda b: {'rep': hid(b['text'], b.get('layer', 36))},
    '/judge': lambda b: {'logit': judge(b['prompt'])},
    '/manifold': lambda b: {'h': manifold(b['messages'], b.get('layer', 24))},
    '/value': lambda b: {'v': value(b['prompt'])},
    '/set_steer': lambda b: set_steer(b.get('vec'), b.get('alpha', 0.0)),
    '/gen_msteer': lambda b: {'text': gen_msteer(b['messages'], b.get('max_new', 50))},
    '/load_adapter': lambda b: load_adapter(b.get('ckpt', ADAPTER)),
    '/reset': lambda b: reset(b['mission']),
    '/observe': lambda b: observe(b['text']),
    '/gain': lambda b: set_gain(b.get('scale', 1.0), b.get('cap_rel', 0.5)),
    '/relmag': lambda b: relmag(),
}

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def do_GET(self):
        b = b'{"ok":true,"adapter":%s}' % (b'true' if LORA is not None else b'false')
        self.send_response(200); self.send_header('Content-Length', str(len(b))); self.end_headers()
        self.wfile.write(b)
    def do_POST(self):
        try:
            n = int(self.headers['Content-Length'])
            body = json.loads(self.rfile.read(n))
            out = OPS.get(self.path, lambda b: {'error': 'unknown path'})(body)
        except Exception as e:
            out = {'error': repr(e)}
        b = json.dumps(out).encode()
        self.send_response(200); self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(b))); self.end_headers()
        self.wfile.write(b)

print('[server] loading adapter %s' % ADAPTER, flush=True)
try:
    print('[server] adapter:', load_adapter(ADAPTER), flush=True)
except Exception as e:
    print('[server] ADAPTER LOAD FAILED (stateless ops still available):', repr(e), flush=True)
print('[server] READY on %d' % PORT, flush=True)
HTTPServer(('127.0.0.1', PORT), H).serve_forever()
