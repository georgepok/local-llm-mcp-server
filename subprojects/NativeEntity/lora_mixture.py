# NATIVE actuation — a BANK of N LoRA basis patches; the persistent slots emit a VECTOR β that mixes them.
# ΔW_t = Σ_i β_i(S_t) ΔW_i. β is vector-valued (not one scalar). Which basis serves which behavior is LEARNED from consequences, never hard-coded.
import torch, torch.nn as nn

MIX_RANK = 4          # rank of each basis patch
MIX_SCALE = 2.0       # base scale (matches the prior single-LoRA actuator strength)
MIX_MAXNORM = 6.0     # per-token delta norm clip (same safety as the prior actuator)


class LoRAMixtureLinear(nn.Module):
    """Wraps a base nn.Linear; holds N basis (A_i,B_i) rank-r pairs. Effective delta = Σ β_i (x A_i^T) B_i^T, β set per step from the slots."""
    def __init__(s, base, n_basis, r=MIX_RANK, scale=MIX_SCALE):
        super().__init__()
        s.base = base
        din, dout = base.weight.shape[1], base.weight.shape[0]
        s.A = nn.Parameter(torch.randn(n_basis, r, din) * 0.02)
        s.B = nn.Parameter(torch.zeros(n_basis, dout, r))        # B=0 -> basis starts as a no-op; structure forms
        s.scale, s.n = scale, n_basis
        s.register_buffer('beta', torch.zeros(n_basis))          # set per step by the slot-conditioned head

    def forward(s, x):
        y = s.base(x)
        if float(s.beta.abs().sum()) < 1e-8:
            return y
        xf = x.float()
        # Σ_i β_i (x A_i^T) B_i^T  — compute per active basis (N is small)
        delta = 0.0
        for i in range(s.n):
            bi = float(s.beta[i])
            if abs(bi) < 1e-6:
                continue
            d = (xf @ s.A[i].t()) @ s.B[i].t()
            delta = delta + bi * d
        if isinstance(delta, float):
            return y
        n = delta.norm(dim=-1, keepdim=True).clamp(min=1e-6)     # norm-clip the COMBINED delta
        delta = delta * (n.clamp(max=MIX_MAXNORM) / n)
        return y + s.scale * delta.to(y.dtype)


def attach_mixture(model, layers, n_basis, r=MIX_RANK):
    mods = {}
    for L in layers:
        blk = model.model.layers[L].mlp
        d = blk.down_proj.weight.device
        m = LoRAMixtureLinear(blk.down_proj, n_basis, r).to(d)        # A/B/beta on the same device as the base layer
        blk.down_proj = m
        mods[L] = m
    print('LoRAMixture: %d basis patches (rank %d) on down_proj layers %s' % (n_basis, r, layers), flush=True)
    return mods


def set_beta(mods, beta):                                        # beta: 1-D tensor [n_basis]
    for m in mods.values():
        m.beta.data = beta.detach().to(m.beta.device, m.beta.dtype)


def mixture_params(mods):
    ps = []
    for m in mods.values():
        ps += [m.A, m.B]
    return ps
