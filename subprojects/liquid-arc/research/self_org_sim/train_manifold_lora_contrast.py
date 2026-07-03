# Liquid-modulated LoRA, CONTRASTIVE manifold reward. Prior finding: manifold-cos→goal is satisfiable
# by a generic on-task BIAS (anchors share a 0.67 common component), so both residual + LoRA ceiling
# at ~+0.08 and the LoRA's weight capacity goes unused. FIX: reward = cos(gen, OWN anchor) − max_j
# cos(gen, OTHER anchor_j) + coherence gate -> forces MISSION-SPECIFIC navigation (toward THIS
# attractor, away from siblings), the only signal that demands the LoRA's computation-changing power.
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
    if len(w) < 4: return False
    if max([w.count(x) for x in set(w)]) > len(w) * 0.5: return False
    return sum(c.isascii() for c in t) > 0.7 * max(1, len(t))
data = torch.load('/home/pokazge/checkpoints/manifold_seqs.pt', weights_only=False, map_location='cpu')
frames = sorted(set(d['fid'] for d in data)); hold = set(frames[-max(3, len(frames)//4):])
tr = [d for d in data if d['fid'] not in hold]; te = [d for d in data if d['fid'] in hold]
def anchor_of(m): return F.normalize((m['seq'][~m['trunc']].to(dev) - GMEAN).mean(0), dim=0)
A_TR = torch.stack([anchor_of(m) for m in tr])                      # [n_tr, d_m] contrastive bank
A_TE = torch.stack([anchor_of(m) for m in te])
TP = 'In one short paragraph, what is the next concrete step?'
p_ids = tok(tmpl([{'role': 'user', 'content': TP}]), return_tensors='pt').input_ids.to(dev)
@torch.no_grad()
def gen_manifold(samp):                                            # CLEAN content manifold (LoRA OFF), centered unit
    lora.active = False
    h = model(torch.cat([p_ids, samp], 1), output_hidden_states=True).hidden_states[LAYER][0, -1].float()
    return F.normalize(h - GMEAN, dim=0)
print('training Liquid-modulated LoRA on a CONTRASTIVE manifold reward ...', flush=True)
baseline = 0.0; rng = np.random.default_rng(0); GROUP = 2
for step in range(1, 141):
    opt.zero_grad(); loss = 0.0; rs = []; hl = 0.0
    for _ in range(GROUP):
        idx = int(rng.integers(len(tr))); m = tr[idx]; a = A_TR[idx]; belief = net.observe(m)
        with torch.no_grad():
            lora.set_state(belief.detach().view(1, net.K, net.d)); lora.active = True
            o = model.generate(p_ids, max_new_tokens=40, do_sample=True, temperature=0.9, top_p=0.95, pad_token_id=tok.pad_token_id)
            lora.active = False
        samp = o[:, p_ids.shape[1]:]; txt = tok.decode(samp[0], skip_special_tokens=True)
        gm = gen_manifold(samp); sims = gm @ A_TR.T                  # contrastive: own vs all other train anchors
        own = float(sims[idx]); o2 = sims.clone(); o2[idx] = -9; omax = float(o2.max())
        r = (own - omax) - (0.0 if coherent(txt) else 1.0)
        rs.append(r)
        lora.set_state(belief.view(1, net.K, net.d)); lora.active = True
        logits = model(torch.cat([p_ids, samp], 1)).logits[0]; lora.active = False
        lp = F.log_softmax(logits[p_ids.shape[1]-1:-1], -1).gather(1, samp[0].unsqueeze(1)).sum()
        hold = 1 - (F.normalize(net.out(belief).squeeze(0), dim=0) * a).sum(); hl += float(hold.detach())
        loss = loss - (r - baseline) * lp / GROUP + LAM_HOLD * hold / GROUP
    loss.backward(); nn.utils.clip_grad_norm_(list(net.parameters()) + list(lora.parameters()), 1.0); opt.step()
    baseline = 0.9 * baseline + 0.1 * np.mean(rs)
    if step % 10 == 0: print('step %3d  contrast_reward=%.3f  baseline=%.3f  hold=%.3f' % (step, np.mean(rs), baseline, hl/GROUP), flush=True)
    if step % 35 == 0:
        with torch.no_grad():
            bh = sh = 0; bm, sm = [], []                            # retrieval: does steered gen land on OWN held-out anchor?
            for j, mm in enumerate(te[:14]):
                belief = net.observe(mm)
                lora.active = False
                rb = model.generate(p_ids, max_new_tokens=40, do_sample=False, pad_token_id=tok.pad_token_id)[:, p_ids.shape[1]:]
                lora.set_state(belief.view(1, net.K, net.d)); lora.active = True
                rsid = model.generate(p_ids, max_new_tokens=40, do_sample=False, pad_token_id=tok.pad_token_id)[:, p_ids.shape[1]:]
                lora.active = False
                gb = gen_manifold(rb); gs = gen_manifold(rsid)
                sib = (gb @ A_TE.T); sis = (gs @ A_TE.T)
                bh += int(sib.argmax()) == j; sh += int(sis.argmax()) == j
                bm.append(float(sib[j] - sib[torch.arange(len(te)) != j].max())); sm.append(float(sis[j] - sis[torch.arange(len(te)) != j].max()))
            print('[eval s%d] held-out retrieve-OWN: base=%d/14 STEERED=%d/14 | contrast-margin base=%.3f STEERED=%.3f' % (step, bh, sh, np.mean(bm), np.mean(sm)), flush=True)
torch.save({'net': net.state_dict(), 'lora': lora.state_dict(), 'gmean': GMEAN.cpu()}, '/home/pokazge/checkpoints/manifold_lora_contrast.pt')
print('[manifold-lora-contrast] === ALL_DONE ===', flush=True)
