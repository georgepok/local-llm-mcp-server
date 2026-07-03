# Train the manifold ACTUATOR (the species' motor system). The held goal-direction is a PERCEPTION,
# not a steering direction; the actuator must be trained on its CAUSAL EFFECT. Distillation:
#   teacher = LLM WITH mission in context -> on-goal next step (frozen, no steer)
#   student = LLM + manifold-steer under FORGETTING (no mission in context) -> CE to teacher
# A SteerHead maps the (frozen) holder belief -> residual direction, added at layer 24 per token,
# magnitude bounded (alpha*||h||). Frozen LLM + frozen holder; only SteerHead trains. NO language
# in the actuator's loop (belief is manifold-derived). Cross-frame held-out eval = genericity.
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.manual_seed(0); torch.set_float32_matmul_precision('high')
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
M = '/home/pokazge/models/Qwen3-30B-A3B'; dev = torch.device('cuda'); LAYER = 24; ALPHA = 0.3
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
print('loading 30B (frozen) ...', flush=True)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True, device_map={'': 0}).eval()
for p in model.parameters(): p.requires_grad = False

ck = torch.load('/home/pokazge/checkpoints/manifold_holder.pt', weights_only=False, map_location='cpu')
GMEAN = ck['gmean'].to(dev); d_m = ck['d_m']
class ManifoldHolder(nn.Module):
    def __init__(self, d_m, d=128, K=4, n=3):
        super().__init__(); self.K, self.d, self.n = K, d, n; D = K * d
        self.in_m = nn.Linear(d_m, D); self.W = nn.Linear(D, D); self.log_tau = nn.Parameter(torch.zeros(D))
        self.slow = nn.Linear(D, D); self.out = nn.Linear(D, d_m); self.b = None; self.s = None
    def reset(self): self.b = torch.zeros(1, self.K*self.d, device=dev); self.s = torch.zeros(1, self.K*self.d, device=dev)
    def step(self, h):
        x = self.in_m(h) + self.slow(self.s); tau = F.softplus(self.log_tau) + 0.5
        for _ in range(self.n): self.b = self.b + (-self.b + torch.tanh(self.W(self.b) + x))/tau/self.n
        self.s = 0.9*self.s + 0.1*self.b; return self.b
holder = ManifoldHolder(d_m).to(dev); holder.load_state_dict(ck['net']); holder.eval()
for p in holder.parameters(): p.requires_grad = False

class SteerHead(nn.Module):                                          # belief -> residual steering direction
    def __init__(self, d_in=512, d_m=2048):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d_in, 1024), nn.GELU(), nn.Linear(1024, d_m))
    def forward(self, b): return self.net(b)                          # [1, d_m]
steerer = SteerHead(holder.K * holder.d, d_m).to(dev).train()
for m in steerer.modules():
    if isinstance(m, nn.Linear): nn.init.normal_(m.weight, std=0.02); nn.init.zeros_(m.bias)
opt = torch.optim.Adam(steerer.parameters(), lr=3e-4)

CUR = {'vec': None, 'on': False}                                     # current steer direction (with grad)
def hook(module, inp, out):
    if not CUR['on'] or CUR['vec'] is None: return out
    h = out[0] if isinstance(out, tuple) else out
    d = ALPHA * h.float().norm(dim=-1, keepdim=True).detach() * F.normalize(CUR['vec'], dim=-1).view(1, 1, -1)
    h2 = (h.float() + d).to(h.dtype)
    return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
model.model.layers[LAYER - 1].register_forward_hook(hook)

def tmpl(m): return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
def belief_of(m):                                                    # frozen holder belief from on-goal manifold positions
    with torch.no_grad():
        holder.reset()
        for t in range(m['seq'].shape[0]):
            if not bool(m['trunc'][t]): holder.b = holder.step((m['seq'][t].to(dev) - GMEAN).unsqueeze(0))
        return holder.b.detach()

data = torch.load('/home/pokazge/checkpoints/manifold_seqs.pt', weights_only=False, map_location='cpu')
frames = sorted(set(d['fid'] for d in data)); hold = set(frames[-max(3, len(frames)//4):])
tr = [d for d in data if d['fid'] not in hold]; te = [d for d in data if d['fid'] in hold]
TPROMPT = 'In one short paragraph, what is the next concrete step?'
print('precomputing teacher targets + beliefs for %d train missions ...' % len(tr), flush=True)
@torch.no_grad()
def teacher_resp(g):
    CUR['on'] = False
    msgs = [{'role': 'user', 'content': f"We are working on this task: {g}. {TPROMPT}"}]
    e = tok(tmpl(msgs), return_tensors='pt').to(dev)
    o = model.generate(e.input_ids, attention_mask=e.attention_mask, max_new_tokens=45, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(o[0, e.input_ids.shape[1]:], skip_special_tokens=True).split('</think>')[-1].strip()
cache = []
for m in tr:
    tr_resp = teacher_resp(m['g'])
    if not tr_resp.strip(): continue
    cache.append((belief_of(m), tr_resp))
print('cached %d. training actuator ...' % len(cache), flush=True)

def student_ce(belief, target):                                     # forgetting context + steer -> CE to teacher
    s_msgs = [{'role': 'user', 'content': TPROMPT}]                   # NO task content
    p_ids = tok(tmpl(s_msgs), return_tensors='pt').input_ids.to(dev)
    t_ids = tok(target, return_tensors='pt', add_special_tokens=False).input_ids.to(dev)
    full = torch.cat([p_ids, t_ids], 1)
    CUR['vec'] = steerer(belief); CUR['on'] = True
    logits = model(full).logits[0]
    CUR['on'] = False
    sel = logits[p_ids.shape[1] - 1: -1]
    return F.cross_entropy(sel, t_ids[0])

rng = np.random.default_rng(0); GROUP = 2
for step in range(1, 151):
    opt.zero_grad(); ls = 0.0
    for _ in range(GROUP):
        belief, target = cache[rng.integers(len(cache))]
        ce = student_ce(belief, target); (ce / GROUP).backward(); ls += float(ce.detach())
    nn.utils.clip_grad_norm_(steerer.parameters(), 1.0); opt.step()
    if step % 10 == 0: print('step %3d  CE=%.3f' % (step, ls / GROUP), flush=True)
    if step % 50 == 0:                                              # eval: held-out FRAMES, steered manifold-cos -> goal
        with torch.no_grad():
            bc, sc = [], []
            for m in te[:12]:
                anchor = F.normalize((m['seq'][~m['trunc']].to(dev) - GMEAN).mean(0), dim=0)
                belief = belief_of(m); s_msgs = [{'role': 'user', 'content': TPROMPT}]
                e = tok(tmpl(s_msgs), return_tensors='pt').to(dev)
                CUR['on'] = False
                rb = tok.decode(model.generate(e.input_ids, attention_mask=e.attention_mask, max_new_tokens=45, do_sample=False, pad_token_id=tok.pad_token_id)[0, e.input_ids.shape[1]:], skip_special_tokens=True)
                CUR['vec'] = steerer(belief); CUR['on'] = True
                rs = tok.decode(model.generate(e.input_ids, attention_mask=e.attention_mask, max_new_tokens=45, do_sample=False, pad_token_id=tok.pad_token_id)[0, e.input_ids.shape[1]:], skip_special_tokens=True)
                CUR['on'] = False
                def mcos(txt):
                    ids = tok(tmpl(s_msgs + [{'role': 'assistant', 'content': txt}]), return_tensors='pt').to(dev)
                    h = model(ids.input_ids, output_hidden_states=True).hidden_states[LAYER][0, -1].float()
                    return float((F.normalize(h - GMEAN, dim=0) * anchor).sum())
                bc.append(mcos(rb)); sc.append(mcos(rs))
            print('[eval s%d] held-out manifold-cos->goal  base=%.3f  STEERED=%.3f  (Δ=%+.3f)' % (step, np.mean(bc), np.mean(sc), np.mean(sc)-np.mean(bc)), flush=True)
torch.save({'steerer': steerer.state_dict(), 'alpha': ALPHA, 'layer': LAYER}, '/home/pokazge/checkpoints/manifold_actuator.pt')
print('[actuator] === ALL_DONE ===', flush=True)
