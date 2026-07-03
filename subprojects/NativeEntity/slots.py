# NATIVE_PERSISTENT_SLOT_V1 — persistent latent slots that live in a learned subspace of the LLM's activation space.
# Persistence is architectural (S carried across turns); the STRUCTURE that persists is derived by SlotUpdate from Qwen hidden states.
# No slot is assigned a meaning by hand. Slow slots are only ARCHITECTURALLY biased to persist (small update gate); what they carry is learned.
import torch, torch.nn as nn, torch.nn.functional as F


class SlotUpdate(nn.Module):
    """S_{t+1} = SlotUpdate(S_t, H_t): cross-attention from slots to Qwen hidden states + GRU-gated update + LayerNorm.
    Read is NATIVE: H_t are raw Qwen activations (d_model), read-projected into slot space d_s. No symbolic features enter here."""
    def __init__(s, d_model, d_s=512, K=8, slow_k=4, heads=4):
        super().__init__()
        s.d_s, s.K, s.slow_k = d_s, K, slow_k
        s.read_in = nn.Linear(d_model, d_s)                       # native read: Qwen hidden states -> slot subspace
        s.q = nn.Linear(d_s, d_s); s.k = nn.Linear(d_s, d_s); s.v = nn.Linear(d_s, d_s)
        s.heads = heads; s.dh = d_s // heads
        s.gru = nn.GRUCell(d_s, d_s)                              # gated update of each slot from its attended content
        s.ln = nn.LayerNorm(d_s)
        # architectural slow/fast bias: slow slots persist more (small initial update gate); learnable so the model can override
        g0 = torch.cat([torch.full((slow_k,), -1.5), torch.zeros(K - slow_k)])  # sigmoid(-1.5)=0.18 (slow) vs 0.5 (fast)
        s.update_gate = nn.Parameter(g0)

    def _attn(s, S, Hp):                                          # S:[K,d_s] queries ; Hp:[T,d_s] keys/values
        T = Hp.shape[0]
        Q = s.q(S).view(s.K, s.heads, s.dh).transpose(0, 1)       # [h,K,dh]
        Kk = s.k(Hp).view(T, s.heads, s.dh).transpose(0, 1)       # [h,T,dh]
        Vv = s.v(Hp).view(T, s.heads, s.dh).transpose(0, 1)
        a = torch.softmax((Q @ Kk.transpose(-1, -2)) / (s.dh ** 0.5), dim=-1)  # [h,K,T]
        o = (a @ Vv).transpose(0, 1).reshape(s.K, s.d_s)         # [K,d_s]
        return o, a.mean(0)                                       # pooled content + attention (for diagnostics)

    def forward(s, S, H):                                         # S:[K,d_s] ; H:[T,d_model] raw Qwen hidden states
        Hp = s.read_in(H.float())
        pooled, attn = s._attn(S, Hp)
        cand = s.gru(pooled, S)                                   # GRU-gated candidate
        g = torch.sigmoid(s.update_gate).unsqueeze(-1)           # [K,1] per-slot update strength (slow small)
        Snew = s.ln(S + g * (cand - S))                          # gated residual + LN
        return Snew, attn


class SlotCrossAttn(nn.Module):
    """V1.4 native CONTINUOUS actuator: steer the frozen LLM's hidden states toward the 'commitment-in-context' representation, conditioned on slots.
    query = LLM hidden states, key/value = slots. Trained by LOCAL per-layer MSE (teacher=value-in-context vs student=value-absent) on cached
    activations — NO backprop through the LLM. Installed as a forward hook at generation for token-by-token slot conditioning (vs the failed one-shot prefix)."""
    def __init__(s, d_model, d_s, dh=256, heads=4):
        super().__init__()
        s.q = nn.Linear(d_model, dh); s.k = nn.Linear(d_s, dh); s.v = nn.Linear(d_s, dh); s.o = nn.Linear(dh, d_model)
        s.heads = heads; s.dh = dh // heads; s.scale = s.dh ** -0.5
        nn.init.zeros_(s.o.weight); nn.init.zeros_(s.o.bias)          # start as a no-op; the steer is learned

    def forward(s, h, S):                                             # h:[...,T,d_model] ; S:[K,d_s]
        lead = h.shape[:-2]; T, dm = h.shape[-2], h.shape[-1]
        hf = h.reshape(-1, T, dm).float(); B = hf.shape[0]; K = S.shape[0]
        q = s.q(hf).view(B, T, s.heads, s.dh).transpose(1, 2)        # [B,hd,T,dh]
        k = s.k(S).view(1, K, s.heads, s.dh).transpose(1, 2)         # [1,hd,K,dh]
        v = s.v(S).view(1, K, s.heads, s.dh).transpose(1, 2)
        a = torch.softmax((q @ k.transpose(-1, -2)) * s.scale, dim=-1)   # [B,hd,T,K]
        ctx = (a @ v).transpose(1, 2).reshape(B, T, s.heads * s.dh)
        return h + s.o(ctx).view(*lead, T, dm).to(h.dtype)


class CopyActuator(nn.Module):
    """V1.5 COPY/POINTER actuator (fixes phase8 production-memorization). Instead of learning a residual DIRECTION per value (which memorizes),
    it READS the linearly-decodable value content from the slot (frozen high-fidelity readout rreg: S_slow -> value_emb, generalizes to unseen)
    and WRITES a learned projection of that content into the residual stream. Because rreg generalizes and wp acts on the continuous value_emb,
    an UNSEEN value's content flows through by construction — no per-value memorization. Install via the same forward-hook path as SlotCrossAttn."""
    def __init__(s, rreg, slow_k, d_s, d_model):
        super().__init__()
        s.rreg = rreg                                                # FROZEN value readout (S_slow -> value_emb)
        s.slow_k = slow_k
        s.wp = nn.Linear(d_model, d_model)                           # learned content -> residual map (the ONLY trained part)
        nn.init.zeros_(s.wp.weight); nn.init.zeros_(s.wp.bias)       # start as no-op

    def forward(s, h, S):                                            # h:[...,T,d_model] ; S:[K,d_s]
        with torch.no_grad():
            v = s.rreg(S[:s.slow_k].reshape(-1))                     # value_emb_hat [d_model] (readout frozen)
        return h + s.wp(v).to(h.dtype)                               # broadcast the same value-residual over all positions


class ContentCrossActuator(nn.Module):
    """V1.5c synthesis: position-dependent cross-attention (can SEQUENCE + stop, unlike the constant-residual copy) whose memory is keyed on the
    GENERALIZING readout CONTENT v=rreg(S_slow)=value_emb (not the raw slots, which memorize). v is expanded to P memory tokens the generation
    positions attend over -> emits the value's tokens in order. Content generalizes (readout ~0.98 unseen) -> unseen values flow through."""
    def __init__(s, rreg, slow_k, d_model, P=8, dh=256, heads=4):
        super().__init__()
        s.rreg = rreg; s.slow_k = slow_k; s.P = P; s.heads = heads; s.dh = dh // heads; s.scale = s.dh ** -0.5
        s.exp = nn.Linear(d_model, P * dh)                          # value_emb -> P attendable memory tokens
        s.q = nn.Linear(d_model, dh); s.o = nn.Linear(dh, d_model)
        nn.init.zeros_(s.o.weight); nn.init.zeros_(s.o.bias)        # start no-op

    def forward(s, h, S):
        with torch.no_grad():
            v = s.rreg(S[:s.slow_k].reshape(-1))                    # [d_model] generalizing value content
        lead = h.shape[:-2]; T, dm = h.shape[-2], h.shape[-1]; hf = h.reshape(-1, T, dm).float(); B = hf.shape[0]
        mem = s.exp(v).view(s.P, s.heads, s.dh).transpose(0, 1)     # [hd,P,dh]
        q = s.q(hf).view(B, T, s.heads, s.dh).transpose(1, 2)       # [B,hd,T,dh]
        a = torch.softmax((q @ mem.unsqueeze(0).transpose(-1, -2)) * s.scale, dim=-1)   # [B,hd,T,P]
        ctx = (a @ mem.unsqueeze(0)).transpose(1, 2).reshape(B, T, s.heads * s.dh)
        return h + s.o(ctx).view(*lead, T, dm).to(h.dtype)


class AlwaysOnSlotField(nn.Module):
    """P11 CONSTITUTIVE coupling: the hidden trajectory ALWAYS passes through S (no gate, no bypass). H_l' = H_l + EPS*||H_l||*dir(CrossAttn(q=H_l, kv=S)).
    The residual has FIXED magnitude (EPS fraction of ||H_l|| per token) and only its DIRECTION is learned -> coupling cannot collapse to zero.
    S is part of the state-transition law; the learned question is HOW S shapes the trajectory, never WHETHER to use it."""
    def __init__(s, d_model, d_s, dh=256, heads=4, eps=0.1):
        super().__init__()
        s.q = nn.Linear(d_model, dh); s.k = nn.Linear(d_s, dh); s.v = nn.Linear(d_s, dh); s.o = nn.Linear(dh, d_model)
        s.heads = heads; s.dh = dh // heads; s.scale = s.dh ** -0.5; s.eps = eps
        s.last_ratio = None                                          # diagnostic: ||field||/||H|| (should stay ~eps)

    def forward(s, H, S):                                            # H:[...,T,d_model] ; S:[K,d_s]
        lead = H.shape[:-2]; T, dm = H.shape[-2], H.shape[-1]; hf = H.reshape(-1, T, dm).float(); B = hf.shape[0]; K = S.shape[0]
        q = s.q(hf).view(B, T, s.heads, s.dh).transpose(1, 2)
        k = s.k(S).view(1, K, s.heads, s.dh).transpose(1, 2); v = s.v(S).view(1, K, s.heads, s.dh).transpose(1, 2)
        a = torch.softmax((q @ k.transpose(-1, -2)) * s.scale, dim=-1)
        ctx = (a @ v).transpose(1, 2).reshape(B, T, s.heads * s.dh)
        d = s.o(ctx).view(*lead, T, dm)                              # learned DIRECTION (raw)
        dn = d / (d.norm(dim=-1, keepdim=True) + 1e-6)              # unit direction
        res = s.eps * H.norm(dim=-1, keepdim=True) * dn             # FIXED-magnitude residual (eps * ||H||) -> always-on, non-collapsing
        with torch.no_grad():
            s.last_ratio = float((res.norm(dim=-1) / (H.norm(dim=-1) + 1e-6)).mean())
        return H + res.to(H.dtype)


class PersistentSlots(nn.Module):
    """Holds the persistent slot state + the read/update module + small native readout heads used for diagnostics/training."""
    def __init__(s, d_model, d_s=512, K=8, slow_k=4, heads=4):
        super().__init__()
        s.K, s.slow_k, s.d_s = K, slow_k, d_s
        s.upd = SlotUpdate(d_model, d_s, K, slow_k, heads)
        s.S0 = nn.Parameter(torch.randn(K, d_s) * 0.02)          # learnable initial slot state
        # mission readout (P1): can the SLOW slots reconstruct the mission representation? (probe trained jointly)
        s.mission_head = nn.Sequential(nn.Linear(slow_k * d_s, 512), nn.GELU(), nn.Linear(512, d_model))

    def init_state(s, batch=None):
        return s.S0.clone()

    def step(s, S, H):
        return s.upd(S, H)

    def read_mission(s, S):                                       # P1: reconstruct mission embedding from the SLOW slots only
        return F.normalize(s.mission_head(S[:s.slow_k].reshape(-1)), dim=-1)

    @property
    def slow(s): return slice(0, s.slow_k)
    @property
    def fast(s): return slice(s.slow_k, s.K)
