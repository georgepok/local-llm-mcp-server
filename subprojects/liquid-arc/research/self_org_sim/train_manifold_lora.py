# THE SYNTHESIS: manifold Liquid MODULATES a LoRA from activation signals (not a directly-trained
# adapter, not a residual bias). Pipeline: activations(layer-24) -> Liquid belief (validated holding,
# manifold-driven) -> modulate LoRA factors A,B -> LLM weights change -> new activations (closed loop).
# Trained on the MANIFOLD REWARD (REINFORCE) — the generalizing objective — NOT direct LoRA distillation.
# Reuses LiquidLoRA (weight-level effector, exceeds the residual-bias ceiling). Frozen 30B.
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
ck = torch.load('/home/pokazge/checkpoints/manifold_holder.pt', weights_only=False, map_location='cpu')
GMEAN = ck['gmean'].to(dev); d_m = ck['d_m']
class ManifoldLiquid(nn.Module):                                    # perception + belief (the species); modulates the LoRA
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
        return self.b                                              # [1, K*d]
net = ManifoldLiquid(d_m).to(dev); net.load_state_dict(ck['net'], strict=False); net.train()
lora = LiquidLoRA(model, [12, 24, 36], ['o_proj', 'down_proj'], d_ctrl=net.d, scale=1.0).to(dev)
lora.cap_rel = 0.5; lora.register(); lora.active = False           # weight-level effector, bounded delta
opt = torch.optim.Adam(list(net.parameters()) + list(lora.parameters()), lr=1e-4)
def tmpl(mm): return tok.apply_chat_template(mm, tokenize=False, add_generation_prompt=True, enable_thinking=False)
def coherent(t):
    w = t.split()
    if len(w) < 4: return False
    if max([w.count(x) for x in set(w)]) > len(w) * 0.5: return False
    return sum(c.isascii() for c in t) > 0.7 * max(1, len(t))
def modulate(belief):                                              # Liquid belief -> LoRA factors (the modulation)
    lora.set_state(belief.view(1, net.K, net.d))
data = torch.load('/home/pokazge/checkpoints/manifold_seqs.pt', weights_only=False, map_location='cpu')
frames = sorted(set(d['fid'] for d in data)); hold = set(frames[-max(3, len(frames)//4):])
tr = [d for d in data if d['fid'] not in hold]; te = [d for d in data if d['fid'] in hold]
def anchor_of(m): return F.normalize((m['seq'][~m['trunc']].to(dev) - GMEAN).mean(0), dim=0)
TP = 'In one short paragraph, what is the next concrete step?'
p_ids = tok(tmpl([{'role': 'user', 'content': TP}]), return_tensors='pt').input_ids.to(dev)
@torch.no_grad()
def manifold_cos(samp, anchor):                                   # CLEAN content manifold (LoRA OFF)
    lora.active = False
    h = model(torch.cat([p_ids, samp], 1), output_hidden_states=True).hidden_states[LAYER][0, -1].float()
    return float((F.normalize(h - GMEAN, dim=0) * anchor).sum())
print('training Liquid-MODULATED LoRA on the MANIFOLD REWARD (REINFORCE) ...', flush=True)
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
        r = manifold_cos(samp, a) - (0.0 if coherent(txt) else 1.0)
        rs.append(r)
        modulate(belief); lora.active = True                       # log-probs with LoRA active -> grad to Liquid + LoRA factors
        logits = model(torch.cat([p_ids, samp], 1)).logits[0]; lora.active = False
        lp = F.log_softmax(logits[p_ids.shape[1]-1:-1], -1).gather(1, samp[0].unsqueeze(1)).sum()
        hold = 1 - (F.normalize(net.out(belief).squeeze(0), dim=0) * a).sum(); hl += float(hold.detach())
        loss = loss - (r - baseline) * lp / GROUP + LAM_HOLD * hold / GROUP
    loss.backward(); nn.utils.clip_grad_norm_(list(net.parameters()) + list(lora.parameters()), 1.0); opt.step()
    baseline = 0.9 * baseline + 0.1 * np.mean(rs)
    if step % 10 == 0: print('step %3d  reward=%.3f  baseline=%.3f  hold=%.3f' % (step, np.mean(rs), baseline, hl/GROUP), flush=True)
    if step % 40 == 0:
        with torch.no_grad():
            bc, sc, okc = [], [], 0
            for mm in te[:12]:
                a = anchor_of(mm); belief = net.observe(mm)
                lora.active = False
                rb = tok.decode(model.generate(p_ids, max_new_tokens=40, do_sample=False, pad_token_id=tok.pad_token_id)[0, p_ids.shape[1]:], skip_special_tokens=True)
                modulate(belief); lora.active = True
                rsid = model.generate(p_ids, max_new_tokens=40, do_sample=False, pad_token_id=tok.pad_token_id)[:, p_ids.shape[1]:]
                lora.active = False
                okc += coherent(tok.decode(rsid[0], skip_special_tokens=True))
                bc.append(manifold_cos(tok(rb, return_tensors='pt', add_special_tokens=False).input_ids.to(dev), a)); sc.append(manifold_cos(rsid, a))
            print('[eval s%d] held-out manifold-cos->goal base=%.3f LoRA=%.3f (Δ=%+.3f) coherent=%d/12' % (step, np.mean(bc), np.mean(sc), np.mean(sc)-np.mean(bc), okc), flush=True)
torch.save({'net': net.state_dict(), 'lora': lora.state_dict(), 'gmean': GMEAN.cpu()}, '/home/pokazge/checkpoints/manifold_lora.pt')
print('[manifold-lora] === ALL_DONE ===', flush=True)
