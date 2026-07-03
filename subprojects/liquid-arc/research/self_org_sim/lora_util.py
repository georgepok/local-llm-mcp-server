# Minimal bounded LoRA actuator on mid/late down_proj (dense MLP). Shared by fit_lora.py (train) and organism3.py (run).
import torch, torch.nn as nn
LORA_LAYERS = [36, 40, 44, 48, 52, 56]                                              # mid/late down_proj targets
LORA_RANK = 4
LORA_SCALE = 2.0                                                                    # base scale (effective strength = alpha * scale)
LORA_MAXNORM = 6.0                                                                  # clip per-token LoRA output norm -> bounded actuator (no broken text)
class LoRALinear(nn.Module):
    def __init__(s, base, r=LORA_RANK, scale=LORA_SCALE, rand=False):
        super().__init__(); s.base = base
        for p in base.parameters(): p.requires_grad_(False)                         # base frozen
        dout, din = base.weight.shape; d = base.weight.device
        s.A = nn.Parameter(torch.randn(r, din, device=d).float() * 0.02)
        s.B = nn.Parameter((torch.randn(dout, r, device=d).float() * 0.02) if rand else torch.zeros(dout, r, device=d).float())  # B=0 -> LoRA starts inert (trained); rand -> sanity plumbing
        s.scale = scale; s.register_buffer('alpha', torch.tensor(1.0, device=d))
    def forward(s, x):
        y = s.base(x)
        lora = (x.float() @ s.A.t()) @ s.B.t()                                      # [.,dout]
        n = lora.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        lora = lora * (n.clamp(max=LORA_MAXNORM) / n)                               # per-token norm clip (bounded)
        return y + (s.alpha * s.scale) * lora.to(y.dtype)
def attach_lora(model, layers=LORA_LAYERS, r=LORA_RANK, scale=LORA_SCALE, rand=False):
    mods = {}
    for L in layers:
        base = model.model.layers[L].mlp.down_proj
        ll = LoRALinear(base, r, scale, rand); model.model.layers[L].mlp.down_proj = ll; mods[L] = ll
    return mods
def set_alpha(mods, a):
    for ll in mods.values(): ll.alpha.fill_(float(a))
def lora_params(mods):
    ps = []
    for ll in mods.values(): ps += [ll.A, ll.B]
    return ps
