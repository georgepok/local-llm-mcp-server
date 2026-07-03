# Manifold actuator v3 — MANIFOLD-NATIVE REWARD (the fix). The Liquid steer is trained by REINFORCE
# on a GEOMETRIC reward: does the steered GENERATION's manifold land near the goal-attractor (+ a
# coherence gate)? NOT token-CE (that linguistic objective overfit + didn't generalize). Keeps
# perception+navigation+ACTION all inside the manifold = the only regime that generalized cross-task.
# The Liquid IS the actuator (steer = linear readout of its trained dynamics). Frozen 30B.
# CRITICAL: reward measured with steer OFF (clean content manifold) — measuring it ON would reward
# the injected vector itself (reward-hacking by construction).
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
class ManifoldLiquid(nn.Module):
    def __init__(self, d_m, d=128, K=4, n=3):
        super().__init__(); self.K, self.d, self.n = K, d, n; D = K * d
        self.in_m = nn.Linear(d_m, D); self.W = nn.Linear(D, D); self.log_tau = nn.Parameter(torch.zeros(D))
        self.slow = nn.Linear(D, D); self.out = nn.Linear(D, d_m); self.steer = nn.Linear(D, d_m); self.b = None; self.s = None
    def reset(self): self.b = torch.zeros(1, self.K*self.d, device=dev); self.s = torch.zeros(1, self.K*self.d, device=dev)
    def step(self, h):
        x = self.in_m(h) + self.slow(self.s); tau = F.softplus(self.log_tau) + 0.5
        for _ in range(self.n): self.b = self.b + (-self.b + torch.tanh(self.W(self.b) + x))/tau/self.n
        self.s = 0.9*self.s + 0.1*self.b; return self.b
    def observe(self, m):
        self.reset()
        for t in range(m['seq'].shape[0]):
            if not bool(m['trunc'][t]): self.b = self.step((m['seq'][t].to(dev) - GMEAN).unsqueeze(0))
        return self.b
net = ManifoldLiquid(d_m).to(dev); net.load_state_dict(ck['net'], strict=False)
net.steer.load_state_dict({'weight': ck['net']['out.weight'].clone(), 'bias': ck['net']['out.bias'].clone()}); net.train()
opt = torch.optim.Adam(net.parameters(), lr=1e-4)
CUR = {'vec': None, 'on': False}
def hook(module, inp, out):
    if not CUR['on'] or CUR['vec'] is None: return out
    h = out[0] if isinstance(out, tuple) else out
    d = ALPHA * h.float().norm(dim=-1, keepdim=True).detach() * F.normalize(CUR['vec'], dim=-1).view(1, 1, -1)
    h2 = (h.float() + d).to(h.dtype)
    return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
model.model.layers[LAYER - 1].register_forward_hook(hook)
def tmpl(m): return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
def coherent(t):
    w = t.split()
    if len(w) < 4: return False
    if max([w.count(x) for x in set(w)]) > len(w) * 0.5: return False
    return sum(c.isascii() for c in t) > 0.7 * max(1, len(t))
data = torch.load('/home/pokazge/checkpoints/manifold_seqs.pt', weights_only=False, map_location='cpu')
frames = sorted(set(d['fid'] for d in data)); hold = set(frames[-max(3, len(frames)//4):])
tr = [d for d in data if d['fid'] not in hold]; te = [d for d in data if d['fid'] in hold]
def anchor_of(m): return F.normalize((m['seq'][~m['trunc']].to(dev) - GMEAN).mean(0), dim=0)
TP = 'In one short paragraph, what is the next concrete step?'
p_ids = tok(tmpl([{'role': 'user', 'content': TP}]), return_tensors='pt').input_ids.to(dev)

@torch.no_grad()
def manifold_cos(sampled_ids, anchor):                              # CLEAN content manifold (steer OFF)
    CUR['on'] = False
    full = torch.cat([p_ids, sampled_ids], 1)
    h = model(full, output_hidden_states=True).hidden_states[LAYER][0, -1].float()
    return float((F.normalize(h - GMEAN, dim=0) * anchor).sum())
print('training the LIQUID actuator on a MANIFOLD REWARD (REINFORCE) ...', flush=True)
baseline = 0.0; rng = np.random.default_rng(0); GROUP = 2
for step in range(1, 121):
    opt.zero_grad(); loss = 0.0; rs = []; hl = 0.0
    for _ in range(GROUP):
        m = tr[rng.integers(len(tr))]; a = anchor_of(m); belief = net.observe(m)
        steer = net.steer(belief)
        with torch.no_grad():                                       # sample a generation under the steer
            CUR['vec'] = steer.detach(); CUR['on'] = True
            o = model.generate(p_ids, max_new_tokens=40, do_sample=True, temperature=0.9, top_p=0.95, pad_token_id=tok.pad_token_id)
            CUR['on'] = False
        samp = o[:, p_ids.shape[1]:]
        txt = tok.decode(samp[0], skip_special_tokens=True)
        r = manifold_cos(samp, a) - (0.0 if coherent(txt) else 1.0)  # geometric reward, coherence gate
        rs.append(r)
        CUR['vec'] = steer; CUR['on'] = True                        # log-probs WITH grad to the Liquid
        logits = model(torch.cat([p_ids, samp], 1)).logits[0]; CUR['on'] = False
        lp = F.log_softmax(logits[p_ids.shape[1]-1:-1], -1).gather(1, samp[0].unsqueeze(1)).sum()
        hold = 1 - (F.normalize(net.out(belief).squeeze(0), dim=0) * a).sum(); hl += float(hold.detach())
        loss = loss - (r - baseline) * lp / GROUP + LAM_HOLD * hold / GROUP
    loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    baseline = 0.9 * baseline + 0.1 * np.mean(rs)
    if step % 10 == 0: print('step %3d  reward=%.3f  baseline=%.3f  hold=%.3f' % (step, np.mean(rs), baseline, hl/GROUP), flush=True)
    if step % 40 == 0:
        with torch.no_grad():
            bc, sc, okc = [], [], 0
            for m in te[:12]:
                a = anchor_of(m); belief = net.observe(m); st = net.steer(belief)
                CUR['on'] = False
                rb = tok.decode(model.generate(p_ids, max_new_tokens=40, do_sample=False, pad_token_id=tok.pad_token_id)[0, p_ids.shape[1]:], skip_special_tokens=True)
                CUR['vec'] = st; CUR['on'] = True
                rsid = model.generate(p_ids, max_new_tokens=40, do_sample=False, pad_token_id=tok.pad_token_id)[:, p_ids.shape[1]:]
                CUR['on'] = False
                rstxt = tok.decode(rsid[0], skip_special_tokens=True); okc += coherent(rstxt)
                bc.append(manifold_cos(tok(rb, return_tensors='pt', add_special_tokens=False).input_ids.to(dev), a)); sc.append(manifold_cos(rsid, a))
            print('[eval s%d] held-out manifold-cos->goal base=%.3f STEERED=%.3f (Δ=%+.3f) coherent=%d/12' % (step, np.mean(bc), np.mean(sc), np.mean(sc)-np.mean(bc), okc), flush=True)
torch.save({'net': net.state_dict(), 'alpha': ALPHA, 'layer': LAYER, 'gmean': GMEAN.cpu()}, '/home/pokazge/checkpoints/manifold_actuator_rl.pt')
print('[actuator-rl] === ALL_DONE ===', flush=True)
