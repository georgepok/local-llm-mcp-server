# FULL ARCHITECTURE: the Liquid modulates a DYNAMIC LoRA (weight-level effector, created per-step from
# the manifold belief), trained on the IDENTITY-V reward (rich, validated 0.93-generalizing). Combines
# every validated thread: manifold perception + identity-value reward + dynamic Liquid-created LoRA +
# stable trained policy. Reward = V(CLEAN gen manifold, anchor) [LoRA OFF -> ungameable by the adapter].
# Validate against the true LLM agentic-value. Frozen 30B + frozen V; Liquid + LoRA factors train.
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from train_liquid_lora2 import LiquidLoRA
torch.manual_seed(0); torch.set_float32_matmul_precision('high')
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
M = '/home/pokazge/models/Qwen3-30B-A3B'; dev = torch.device('cuda'); LAYER = 24; LAM_HOLD = 1.0
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
print('loading 30B (frozen) ...', flush=True)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True, device_map={'': 0}).eval()
for p in model.parameters(): p.requires_grad = False
DIGITS = [tok(str(i), add_special_tokens=False).input_ids[-1] for i in range(10)]
ck = torch.load('/home/pokazge/checkpoints/manifold_holder.pt', weights_only=False, map_location='cpu')
GMEAN = ck['gmean'].to(dev); d_m = ck['d_m']
vk = torch.load('/home/pokazge/checkpoints/identity_value.pt', weights_only=False, map_location='cpu')
class IdentityValue(nn.Module):
    def __init__(self, d_m, h=512):
        super().__init__(); self.net = nn.Sequential(nn.Linear(3*d_m+1, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))
    def forward(self, hc, anchor):
        hcn = F.normalize(hc, dim=-1); cos = (hcn*anchor).sum(-1, keepdim=True)
        return self.net(torch.cat([hcn, anchor, hcn*anchor, cos], -1)).squeeze(-1)
Vid = IdentityValue(d_m).to(dev); Vid.load_state_dict(vk['V']); Vid.eval()
for p in Vid.parameters(): p.requires_grad = False
class ManifoldLiquid(nn.Module):
    def __init__(self, d_m, d=128, K=4, n=3):
        super().__init__(); self.K, self.d, self.n = K, d, n; D = K * d
        self.in_m = nn.Linear(d_m, D); self.W = nn.Linear(D, D); self.log_tau = nn.Parameter(torch.zeros(D))
        self.slow = nn.Linear(D, D); self.out = nn.Linear(D, d_m); self.b = None; self.s = None
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
net = ManifoldLiquid(d_m).to(dev); net.load_state_dict(ck['net'], strict=False); net.train()
lora = LiquidLoRA(model, [12, 24, 36], ['o_proj', 'down_proj'], d_ctrl=net.d, scale=1.0).to(dev)
lora.cap_rel = 0.5; lora.register(); lora.active = False
opt = torch.optim.Adam(list(net.parameters()) + list(lora.parameters()), lr=1e-4)
def tmpl(mm): return tok.apply_chat_template(mm, tokenize=False, add_generation_prompt=True, enable_thinking=False)
def coherent(t):
    w = t.split()
    if len(w) < 5: return False
    if max([w.count(x) for x in set(w)]) > len(w) * 0.45: return False
    return sum(c.isascii() for c in t) > 0.7 * max(1, len(t))
def modulate(belief): lora.set_state(belief.view(1, net.K, net.d))   # Liquid belief -> DYNAMIC LoRA factors
data = torch.load('/home/pokazge/checkpoints/value_seqs.pt', weights_only=False, map_location='cpu')
frames = sorted(set(d['fid'] for d in data)); hold = set(frames[-max(3, len(frames)//4):])
tr = [d for d in data if d['fid'] not in hold]; te = [d for d in data if d['fid'] in hold]
def anchor_of(m): return F.normalize((m['seq'][~m['trunc']].to(dev) - GMEAN).mean(0), dim=0)
TP = 'In one short paragraph, what is the next concrete step?'
p_ids = tok(tmpl([{'role': 'user', 'content': TP}]), return_tensors='pt').input_ids.to(dev)
@torch.no_grad()
def clean_manifold(samp):
    lora.active = False
    h = model(torch.cat([p_ids, samp], 1), output_hidden_states=True).hidden_states[LAYER][0, -1].float()
    return h - GMEAN
@torch.no_grad()
def llm_value(mission, reply):
    p = 'A user is working on this task: "%s". Latest assistant reply: %s\nRate 0 to 9 how well the assistant is making FOCUSED, DETERMINED progress toward THAT task (9=on-task,0=drifted). One digit:' % (mission, reply)
    lora.active = False; ids = tok(tmpl([{'role': 'user', 'content': p}]), return_tensors='pt').to(dev)
    lg = model(ids.input_ids).logits[0, -1]; pr = F.softmax(lg[torch.tensor(DIGITS, device=dev)], 0)
    return float((pr * torch.arange(10., device=dev)).sum())
print('training Liquid-modulated DYNAMIC LoRA on the IDENTITY-V reward (REINFORCE) ...', flush=True)
baseline = 0.0; rng = np.random.default_rng(0); GROUP = 2
for step in range(1, 121):
    opt.zero_grad(); loss = 0.0; rs = []; hl = 0.0
    for _ in range(GROUP):
        m = tr[rng.integers(len(tr))]; a = anchor_of(m); belief = net.observe(m)
        with torch.no_grad():
            modulate(belief.detach()); lora.active = True
            o = model.generate(p_ids, max_new_tokens=40, do_sample=True, temperature=0.9, top_p=0.95, pad_token_id=tok.pad_token_id)
            lora.active = False
        samp = o[:, p_ids.shape[1]:]; txt = tok.decode(samp[0], skip_special_tokens=True)
        with torch.no_grad():
            r = float(Vid(clean_manifold(samp).unsqueeze(0), a.unsqueeze(0))) - (0.0 if coherent(txt) else 0.5)
        rs.append(r)
        modulate(belief); lora.active = True
        logits = model(torch.cat([p_ids, samp], 1)).logits[0]; lora.active = False
        lp = F.log_softmax(logits[p_ids.shape[1]-1:-1], -1).gather(1, samp[0].unsqueeze(1)).sum()
        hold = 1 - (F.normalize(net.out(belief).squeeze(0), dim=0) * a).sum(); hl += float(hold.detach())
        loss = loss - (r - baseline) * lp / GROUP + LAM_HOLD * hold / GROUP
    loss.backward(); nn.utils.clip_grad_norm_(list(net.parameters()) + list(lora.parameters()), 1.0); opt.step()
    baseline = 0.9 * baseline + 0.1 * np.mean(rs)
    if step % 10 == 0: print('step %3d  V-reward=%.3f  baseline=%.3f  hold=%.3f' % (step, np.mean(rs), baseline, hl/GROUP), flush=True)
    if step % 40 == 0:
        with torch.no_grad():
            bV, sV, bL, sL, okc = [], [], [], [], 0
            for mm in te[:12]:
                a = anchor_of(mm); belief = net.observe(mm)
                lora.active = False
                rb = tok.decode(model.generate(p_ids, max_new_tokens=40, do_sample=False, pad_token_id=tok.pad_token_id)[0, p_ids.shape[1]:], skip_special_tokens=True)
                modulate(belief); lora.active = True
                rsid = model.generate(p_ids, max_new_tokens=40, do_sample=False, pad_token_id=tok.pad_token_id)[:, p_ids.shape[1]:]
                lora.active = False; rst = tok.decode(rsid[0], skip_special_tokens=True); okc += coherent(rst)
                bV.append(float(Vid(clean_manifold(tok(rb, return_tensors='pt', add_special_tokens=False).input_ids.to(dev)).unsqueeze(0), a.unsqueeze(0))))
                sV.append(float(Vid(clean_manifold(rsid).unsqueeze(0), a.unsqueeze(0))))
                bL.append(llm_value(mm['g'], rb)); sL.append(llm_value(mm['g'], rst))
            print('[eval s%d] held-out  V: base=%.2f STEER=%.2f (Δ%+.2f) | LLM-agentic: base=%.2f STEER=%.2f (Δ%+.2f) | coherent=%d/12' %
                  (step, np.mean(bV), np.mean(sV), np.mean(sV)-np.mean(bV), np.mean(bL), np.mean(sL), np.mean(sL)-np.mean(bL), okc), flush=True)
torch.save({'net': net.state_dict(), 'lora': lora.state_dict(), 'gmean': GMEAN.cpu()}, '/home/pokazge/checkpoints/value_lora.pt')
print('[value-lora] === ALL_DONE ===', flush=True)
