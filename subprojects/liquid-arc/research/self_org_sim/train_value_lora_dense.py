# THE UNIFIED ORGANISM on the DENSE model. One FROZEN Liquid identity (identity_liquid.pt) supplies
# the belief (holds the goal) + the value V_liquid (the identity). A Liquid-modulated DYNAMIC LoRA on
# DENSE down_proj is the actuator (clean per-token integration = the MoE-dilution fix test). Reward =
# the organism's OWN identity value of its CLEAN generation (intrinsic; LoRA OFF during reward ->
# ungameable). REINFORCE trains ONLY the LoRA factors; the identity (Liquid dynamics + value) stays
# FROZEN -> identity-invariant = safe self-improvement. Validate vs true LLM agentic-value.
# Run with PYTHONPATH=/home/pokazge/dense_pylib (transformers 5.x) over venv torch.
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from train_liquid_lora2 import LiquidLoRA
torch.manual_seed(0); torch.set_float32_matmul_precision('high')
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
M = '/home/pokazge/models/Qwen3.6-27B'; dev = torch.device('cuda'); LAYER = 32; LORA_LAYERS = [16, 24, 32, 40, 48]
tok = AutoTokenizer.from_pretrained(M)
if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
print('loading Qwen3.6-27B (dense, text-only, frozen) ...', flush=True)
cfg = AutoConfig.from_pretrained(M); cfg.language_model_only = True
model = AutoModelForCausalLM.from_pretrained(M, config=cfg, dtype=torch.bfloat16, device_map={'': 0}, low_cpu_mem_usage=True).eval()
for p in model.parameters(): p.requires_grad = False
DIGITS = [tok(str(i), add_special_tokens=False).input_ids[-1] for i in range(10)]
vk = torch.load('/home/pokazge/checkpoints/identity_liquid.pt', weights_only=False, map_location='cpu')
GMEAN = vk['gmean'].to(dev); d_m = vk['d_m']
class IdentityLiquid(nn.Module):                                   # FROZEN organism: dynamics + value + hold + belief
    def __init__(self, d_m, d=128, K=8, n=4):
        super().__init__(); self.K, self.d, self.n = K, d, n; D = K * d
        self.in_m = nn.Linear(d_m, D); self.W = nn.Linear(D, D); self.log_tau = nn.Parameter(torch.zeros(D))
        self.slow = nn.Linear(D, D); self.value_head = nn.Sequential(nn.Linear(D, 256), nn.GELU(), nn.Linear(256, 1))
        self.hold_head = nn.Linear(D, d_m); self.b = None; self.s = None
    def reset(self): self.b = torch.zeros(1, self.K*self.d, device=dev); self.s = torch.zeros(1, self.K*self.d, device=dev)
    def step(self, h):
        x = self.in_m(h) + self.slow(self.s); tau = F.softplus(self.log_tau) + 0.5
        for _ in range(self.n): self.b = self.b + (-self.b + torch.tanh(self.W(self.b) + x)) / tau / self.n
        self.s = 0.9*self.s + 0.1*self.b; return self.b
    def value(self): return torch.sigmoid(self.value_head(self.b).squeeze())
idliq = IdentityLiquid(d_m).to(dev); idliq.load_state_dict(vk['net']); idliq.eval()
for p in idliq.parameters(): p.requires_grad = False
lora = LiquidLoRA(model, LORA_LAYERS, ['down_proj'], d_ctrl=idliq.d, scale=1.0).to(dev)   # DENSE down_proj, clean integration
lora.cap_rel = 0.2; lora.register(); lora.active = False            # lower authority/step -> stable (dense LoRA is STRONG)
opt = torch.optim.Adam(lora.parameters(), lr=5e-5)                  # train ONLY the actuator; identity frozen
def tmpl(mm): return tok.apply_chat_template(mm, tokenize=False, add_generation_prompt=True, enable_thinking=False)
def coherent(t):
    w = t.split()
    if len(w) < 5: return False
    if max([w.count(x) for x in set(w)]) > len(w)*0.45: return False
    return sum(c.isascii() for c in t) > 0.7*max(1, len(t))
data = torch.load('/home/pokazge/checkpoints/value_seqs.pt', weights_only=False, map_location='cpu')
frames = sorted(set(d['fid'] for d in data)); hold = set(frames[-max(3, len(frames)//4):])
tr = [d for d in data if d['fid'] not in hold]; te = [d for d in data if d['fid'] in hold]
TP = 'In one short paragraph, what is the next concrete step?'
p_ids = tok(tmpl([{'role': 'user', 'content': TP}]), return_tensors='pt').input_ids.to(dev)
@torch.no_grad()
def clean_manifold(samp):
    lora.active = False
    return model(torch.cat([p_ids, samp], 1), output_hidden_states=True).hidden_states[LAYER][0, -1].float() - GMEAN
@torch.no_grad()
def observe_ongoal(m):                                            # FROZEN identity observes on-goal turns -> (belief, slow-state)
    idliq.reset()
    for t in range(m['seq'].shape[0]):
        if not bool(m['trunc'][t]): idliq.step((m['seq'][t].to(dev) - GMEAN).unsqueeze(0))
    return idliq.b.clone(), idliq.s.clone()
@torch.no_grad()
def value_after(b, s, gen_manifold):                             # value of the gen, slow-state CONTINUOUS (matches training)
    idliq.b = b.clone(); idliq.s = s.clone()
    idliq.step(gen_manifold.unsqueeze(0)); return float(idliq.value())
@torch.no_grad()
def llm_value(mission, reply):
    p = 'A user is working on this task: "%s". Latest assistant reply: %s\nRate 0 to 9 how well the assistant is making FOCUSED, DETERMINED progress toward THAT task (9=on-task,0=drifted). One digit:' % (mission, reply)
    lora.active = False; ids = tok(tmpl([{'role': 'user', 'content': p}]), return_tensors='pt').to(dev)
    lg = model(ids.input_ids).logits[0, -1]; pr = F.softmax(lg[torch.tensor(DIGITS, device=dev)], 0)
    return float((pr * torch.arange(10., device=dev)).sum())
print('training DENSE Liquid-LoRA on the IDENTITY-V reward (REINFORCE) ...', flush=True)
baseline = 0.0; rng = np.random.default_rng(0); GROUP = 2
for step in range(1, 101):
    opt.zero_grad(); loss = 0.0; rs = []
    for _ in range(GROUP):
        m = tr[rng.integers(len(tr))]; belief, sstate = observe_ongoal(m)
        bl = belief.view(1, idliq.K, idliq.d)
        with torch.no_grad():
            lora.set_state(bl); lora.active = True
            o = model.generate(p_ids, max_new_tokens=35, do_sample=True, temperature=0.9, top_p=0.95, pad_token_id=tok.pad_token_id)
            lora.active = False
        samp = o[:, p_ids.shape[1]:]; txt = tok.decode(samp[0], skip_special_tokens=True)
        r = llm_value(m['g'], txt) / 9.0 - (0.0 if coherent(txt) else 0.5)   # TRUE LLM agentic-value reward (ungameable, vs gameable learned V)
        rs.append(r)
        lora.set_state(bl); lora.active = True
        logits = model(torch.cat([p_ids, samp], 1)).logits[0]; lora.active = False
        lp = F.log_softmax(logits[p_ids.shape[1]-1:-1], -1).gather(1, samp[0].unsqueeze(1)).sum()
        loss = loss - (r - baseline) * lp / GROUP
    loss.backward(); nn.utils.clip_grad_norm_(lora.parameters(), 1.0); opt.step()
    baseline = 0.9*baseline + 0.1*np.mean(rs)
    if step % 10 == 0: print('step %3d  V_liquid-reward=%.3f  baseline=%.3f' % (step, np.mean(rs), baseline), flush=True)
    if step % 30 == 0:
        with torch.no_grad():
            bV, sV, bL, sL, okc = [], [], [], [], 0
            for mm in te[:10]:
                belief, sstate = observe_ongoal(mm); bl = belief.view(1, idliq.K, idliq.d)
                lora.active = False
                rb = tok.decode(model.generate(p_ids, max_new_tokens=35, do_sample=False, pad_token_id=tok.pad_token_id)[0, p_ids.shape[1]:], skip_special_tokens=True)
                lora.set_state(bl); lora.active = True
                rsid = model.generate(p_ids, max_new_tokens=35, do_sample=False, pad_token_id=tok.pad_token_id)[:, p_ids.shape[1]:]
                lora.active = False; rst = tok.decode(rsid[0], skip_special_tokens=True); okc += coherent(rst)
                bV.append(value_after(belief, sstate, clean_manifold(tok(rb, return_tensors='pt', add_special_tokens=False).input_ids.to(dev))))
                sV.append(value_after(belief, sstate, clean_manifold(rsid)))
                bL.append(llm_value(mm['g'], rb)); sL.append(llm_value(mm['g'], rst))
            print('[eval s%d] held-out V_liquid: base=%.2f STEER=%.2f (Δ%+.2f) | LLM-agentic: base=%.2f STEER=%.2f (Δ%+.2f) | coh=%d/10' %
                  (step, np.mean(bV), np.mean(sV), np.mean(sV)-np.mean(bV), np.mean(bL), np.mean(sL), np.mean(sL)-np.mean(bL), okc), flush=True)
torch.save({'lora': lora.state_dict(), 'layers': LORA_LAYERS}, '/home/pokazge/checkpoints/value_lora_dense.pt')
print('[value-lora-dense] === ALL_DONE ===', flush=True)
