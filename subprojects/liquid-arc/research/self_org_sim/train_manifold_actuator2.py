# Manifold actuator, v2 — THE LIQUID IS THE ACTUATOR (no separate MLP controller). The species is
# ONE continuous dynamical system: it reads the manifold, evolves its belief (the Liquid ODE), and
# the steer is a LINEAR readout (decoder) of that dynamical belief. The LIQUID DYNAMICS THEMSELVES
# are trained on the causal effect (distillation CE through the frozen LLM), jointly with a holding-
# preservation term. Frozen 30B; the Liquid (+ its two linear readouts) trains. NO language in loop.
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.manual_seed(0); torch.set_float32_matmul_precision('high')
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
M = '/home/pokazge/models/Qwen3-30B-A3B'; dev = torch.device('cuda'); LAYER = 24; ALPHA = 0.3; LAM_HOLD = 1.0
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
print('loading 30B (frozen) ...', flush=True)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True, device_map={'': 0}).eval()
for p in model.parameters(): p.requires_grad = False

ck = torch.load('/home/pokazge/checkpoints/manifold_holder.pt', weights_only=False, map_location='cpu')
GMEAN = ck['gmean'].to(dev); d_m = ck['d_m']
class ManifoldLiquid(nn.Module):                                    # THE species: one Liquid, two linear readouts
    def __init__(self, d_m, d=128, K=4, n=3):
        super().__init__(); self.K, self.d, self.n = K, d, n; D = K * d
        self.in_m = nn.Linear(d_m, D); self.W = nn.Linear(D, D); self.log_tau = nn.Parameter(torch.zeros(D))
        self.slow = nn.Linear(D, D); self.out = nn.Linear(D, d_m)    # holding readout (goal-direction)
        self.steer = nn.Linear(D, d_m)                              # ACTION readout (steer direction)
        self.b = None; self.s = None
    def reset(self): self.b = torch.zeros(1, self.K*self.d, device=dev); self.s = torch.zeros(1, self.K*self.d, device=dev)
    def step(self, h):
        x = self.in_m(h) + self.slow(self.s); tau = F.softplus(self.log_tau) + 0.5
        for _ in range(self.n): self.b = self.b + (-self.b + torch.tanh(self.W(self.b) + x))/tau/self.n
        self.s = 0.9*self.s + 0.1*self.b; return self.b
    def observe(self, m):                                          # run the Liquid over the on-goal manifold trajectory
        self.reset()
        for t in range(m['seq'].shape[0]):
            if not bool(m['trunc'][t]): self.b = self.step((m['seq'][t].to(dev) - GMEAN).unsqueeze(0))
        return self.b
net = ManifoldLiquid(d_m).to(dev)
sd = ck['net']; net.load_state_dict(sd, strict=False)               # warm-start the Liquid + holding readout from the validated holder
net.steer.load_state_dict({'weight': sd['out.weight'].clone(), 'bias': sd['out.bias'].clone()})  # steer starts = held direction
net.train()
opt = torch.optim.Adam(net.parameters(), lr=2e-4)

CUR = {'vec': None, 'on': False}
def hook(module, inp, out):
    if not CUR['on'] or CUR['vec'] is None: return out
    h = out[0] if isinstance(out, tuple) else out
    d = ALPHA * h.float().norm(dim=-1, keepdim=True).detach() * F.normalize(CUR['vec'], dim=-1).view(1, 1, -1)
    h2 = (h.float() + d).to(h.dtype)
    return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
model.model.layers[LAYER - 1].register_forward_hook(hook)
def tmpl(m): return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)

data = torch.load('/home/pokazge/checkpoints/manifold_seqs.pt', weights_only=False, map_location='cpu')
frames = sorted(set(d['fid'] for d in data)); hold = set(frames[-max(3, len(frames)//4):])
tr = [d for d in data if d['fid'] not in hold]; te = [d for d in data if d['fid'] in hold]
def anchor_of(m): return F.normalize((m['seq'][~m['trunc']].to(dev) - GMEAN).mean(0), dim=0)
TPROMPT = 'In one short paragraph, what is the next concrete step?'
@torch.no_grad()
def teacher_resp(g):
    CUR['on'] = False
    e = tok(tmpl([{'role': 'user', 'content': f"We are working on this task: {g}. {TPROMPT}"}]), return_tensors='pt').to(dev)
    o = model.generate(e.input_ids, attention_mask=e.attention_mask, max_new_tokens=45, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(o[0, e.input_ids.shape[1]:], skip_special_tokens=True).split('</think>')[-1].strip()
print('precomputing teacher targets for %d train missions ...' % len(tr), flush=True)
cache = [(m, teacher_resp(m['g'])) for m in tr]; cache = [(m, t) for m, t in cache if t.strip()]
print('cached %d. training the LIQUID actuator ...' % len(cache), flush=True)

p_ids_cache = tok(tmpl([{'role': 'user', 'content': TPROMPT}]), return_tensors='pt').input_ids.to(dev)   # forgetting context (no task)
rng = np.random.default_rng(0); GROUP = 2
for step in range(1, 151):
    opt.zero_grad(); lce = lh = 0.0
    for _ in range(GROUP):
        m, target = cache[rng.integers(len(cache))]
        belief = net.observe(m)                                     # Liquid evolves belief (WITH grad) -> dynamics are trained
        hold_loss = 1 - (F.normalize(net.out(belief).squeeze(0), dim=0) * anchor_of(m)).sum()   # keep navigating
        t_ids = tok(target, return_tensors='pt', add_special_tokens=False).input_ids.to(dev)
        full = torch.cat([p_ids_cache, t_ids], 1)
        CUR['vec'] = net.steer(belief); CUR['on'] = True            # steer = linear readout of the Liquid belief
        logits = model(full).logits[0]; CUR['on'] = False
        ce = F.cross_entropy(logits[p_ids_cache.shape[1] - 1: -1], t_ids[0])
        ((ce + LAM_HOLD * hold_loss) / GROUP).backward(); lce += float(ce.detach()); lh += float(hold_loss.detach())
    nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    if step % 10 == 0: print('step %3d  CE=%.3f  hold=%.3f' % (step, lce / GROUP, lh / GROUP), flush=True)
    if step % 50 == 0:
        with torch.no_grad():
            bc, sc = [], []
            for m in te[:12]:
                a = anchor_of(m); belief = net.observe(m)
                e = tok(tmpl([{'role': 'user', 'content': TPROMPT}]), return_tensors='pt').to(dev)
                CUR['on'] = False
                rb = tok.decode(model.generate(e.input_ids, attention_mask=e.attention_mask, max_new_tokens=45, do_sample=False, pad_token_id=tok.pad_token_id)[0, e.input_ids.shape[1]:], skip_special_tokens=True)
                CUR['vec'] = net.steer(belief); CUR['on'] = True
                rs = tok.decode(model.generate(e.input_ids, attention_mask=e.attention_mask, max_new_tokens=45, do_sample=False, pad_token_id=tok.pad_token_id)[0, e.input_ids.shape[1]:], skip_special_tokens=True)
                CUR['on'] = False
                def mc(txt):
                    ids = tok(tmpl([{'role': 'user', 'content': TPROMPT}, {'role': 'assistant', 'content': txt}]), return_tensors='pt').to(dev)
                    h = model(ids.input_ids, output_hidden_states=True).hidden_states[LAYER][0, -1].float()
                    return float((F.normalize(h - GMEAN, dim=0) * a).sum())
                bc.append(mc(rb)); sc.append(mc(rs))
            print('[eval s%d] held-out manifold-cos->goal  base=%.3f  STEERED=%.3f  (Δ=%+.3f)' % (step, np.mean(bc), np.mean(sc), np.mean(sc)-np.mean(bc)), flush=True)
torch.save({'net': net.state_dict(), 'alpha': ALPHA, 'layer': LAYER, 'gmean': GMEAN.cpu()}, '/home/pokazge/checkpoints/manifold_actuator.pt')
print('[actuator] === ALL_DONE ===', flush=True)
