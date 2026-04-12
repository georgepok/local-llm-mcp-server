"""Train MetricNet for within-chain causal ordering.

Cross-chain separation is already solved. The open problem:
within a chain, the effect token should attend MORE to root cause
than to intermediates. This teaches directional causal flow.

Loss: for each chain, logit(effect → root) > logit(effect → intermediate)
Plus: logit(within-chain) > logit(cross-chain) (maintain separation)

Uses full 16-step ODE on mid-layer hidden states (two-pass approach).

Run (~10GB):
  export PYTHONPATH=/workspace/liquid-arc:/workspace/fgn-v3
  python3 -u scripts/train_inchain.py
"""

import argparse, random, time, torch, torch.nn.functional as F, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHAIN_TEMPLATES = [
    ("A drought hit the farmlands", "Wheat crops failed", "Flour prices tripled", "Bakeries closed"),
    ("A factory fire broke out", "Production halted", "Parts shortage spread", "Assembly lines stopped"),
    ("Hackers breached the system", "Data was corrupted", "Services went offline", "Users lost access"),
    ("A dam broke upstream", "Floodwaters rushed downstream", "Roads were submerged", "Towns were evacuated"),
    ("A strike shut the port", "Ships couldn't unload", "Fuel supplies dwindled", "Gas stations ran dry"),
    ("A storm hit the coast", "Power lines fell", "Electricity was cut", "Cold storage failed"),
    ("An earthquake struck", "Buildings cracked", "Roads buckled", "Traffic was gridlocked"),
    ("A chemical spill occurred", "The river was contaminated", "Fish populations died", "Fishing boats sat idle"),
    ("A virus infected the network", "Servers crashed", "Banking went offline", "ATMs stopped working"),
    ("A wildfire spread rapidly", "Smoke filled the valley", "Air quality plummeted", "Schools were closed"),
    ("A pipeline burst", "Oil leaked into the soil", "Groundwater was polluted", "Wells were shut down"),
    ("A bridge collapsed", "Traffic was rerouted", "Commute times doubled", "Workers arrived late"),
]


def generate_chain_data(tokenizer, n_chains=3, n_hops=4):
    """Generate interleaved chains with token-level chain AND position labels."""
    chains = random.sample(CHAIN_TEMPLATES, min(n_chains, len(CHAIN_TEMPLATES)))

    # Interleave
    sentences = []
    sent_meta = []  # (chain_id, hop_idx) per sentence
    for hop in range(n_hops):
        for cid, chain in enumerate(chains):
            sentences.append(chain[hop] + ".")
            sent_meta.append((cid, hop))

    text = " ".join(sentences)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    token_texts = [tokenizer.decode([t]).strip().lower() for t in tokens]
    N = len(tokens)

    # Map tokens to (chain_id, hop_idx)
    token_meta = []
    char_pos = 0
    sent_boundaries = []
    pos = 0
    for s in sentences:
        idx = text.find(s, pos)
        sent_boundaries.append((idx, idx + len(s)))
        pos = idx + len(s)

    tchar = 0
    for tid in tokens:
        t = tokenizer.decode([tid])
        mid = tchar + len(t) // 2
        assigned = (-1, -1)
        for si, (sb, se) in enumerate(sent_boundaries):
            if sb <= mid < se:
                assigned = sent_meta[si]
                break
        token_meta.append(assigned)
        tchar += len(t)

    return text, tokens, token_meta, n_chains, n_hops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='/workspace/models/qwen3-4b')
    parser.add_argument('--ode_checkpoint',
                        default='/workspace/liquid-arc/output_e2e_ce_v2/checkpoints/step_1500.pt')
    parser.add_argument('--config', default='/workspace/liquid-arc/configs/mind_layerwise.yaml')
    parser.add_argument('--output_dir', default='/workspace/liquid-arc/output_inchain')
    parser.add_argument('--max_steps', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=3e-6)
    parser.add_argument('--grad_clip', type=float, default=0.5)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=500)
    parser.add_argument('--extract_layer', type=int, default=18)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from liquid_arc.config import LiquidARCConfig
    from liquid_arc.dynamics import ContinuousDynamics
    from liquid_arc.context_pool import ContextPool
    from liquid_arc.solver import euler_solve

    print("=" * 70)
    print("IN-CHAIN CAUSAL ORDERING TRAINING")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    config = LiquidARCConfig.from_yaml(args.config)

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    llm = AutoModelForCausalLM.from_pretrained(args.model_path, device_map='cuda',
        torch_dtype=torch.bfloat16, trust_remote_code=True)
    llm.eval()
    for p in llm.parameters():
        p.requires_grad_(False)

    dynamics = ContinuousDynamics(config).to('cuda').float().train()
    context_pool = ContextPool(config).to('cuda').float().train()
    ckpt = torch.load(args.ode_checkpoint, map_location='cuda', weights_only=False)
    if 'dynamics_state' in ckpt:
        dynamics.load_state_dict(ckpt['dynamics_state'])
        context_pool.load_state_dict(ckpt['context_pool_state'])
    else:
        import torch.nn as nn
        sd = ckpt.get('model_state_dict', ckpt)
        cleaned = {k.replace("_orig_mod.", "").replace('metric_net_linear2.', 'metric_net_linear2_diag.'): v
                   for k, v in sd.items()}
        dyn_keys = {k: v for k, v in cleaned.items()
                    if k.startswith('dynamics.') or k.startswith('context_pool.')}
        holder = nn.ModuleDict({'dynamics': dynamics, 'context_pool': context_pool})
        holder.load_state_dict(dyn_keys, strict=False)
    dynamics = dynamics.float().train()
    context_pool = context_pool.float().train()
    dynamics.freeze_tau = False

    optimizer = torch.optim.Adam(
        list(dynamics.parameters()) + list(context_pool.parameters()), lr=args.lr)
    print(f"  LR={args.lr}, extract_layer={args.extract_layer}")
    print(f"  GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    log_f = open(os.path.join(args.output_dir, 'train.log'), 'w')
    best_loss = float('inf')

    for step in range(1, args.max_steps + 1):
        t0 = time.time()

        n_chains = random.choice([2, 3, 4])
        text, tokens, token_meta, nc, nh = generate_chain_data(tok, n_chains, 4)

        input_ids = torch.tensor([tokens], device='cuda')
        N = input_ids.shape[1]

        # Get mid-layer hidden states
        with torch.no_grad():
            out = llm(input_ids=input_ids, output_hidden_states=True)
        h = out.hidden_states[args.extract_layer].float()  # [1, N, d]
        N = h.shape[1]

        # Direct MetricNet on hidden states (no ODE integration)
        # Train MetricNet to produce the right routing from raw representations
        mask = torch.ones(1, N, dtype=torch.bool, device='cuda')
        context = context_pool(h, mask)
        dynamics.set_context(context, mask=None)
        h_normed = dynamics.norm_geo(h)
        ctx_exp = context.unsqueeze(1).expand(-1, N, -1)
        mi = torch.cat([h_normed, ctx_exp], dim=-1)
        hidden = F.gelu(dynamics.metric_net_linear1(mi))
        g = F.softplus(dynamics.metric_net_linear2_diag(hidden))
        sqrt_g = g.sqrt()
        qk = h_normed * sqrt_g
        t_diff = F.softplus(dynamics.t_diffusion)
        dot_qk = torch.bmm(qk, qk.transpose(1, 2)) / (2.0 * t_diff)
        k_norm_sq = (qk * qk).sum(dim=-1, keepdim=True)
        raw_logits = (dot_qk - k_norm_sq.transpose(1, 2) / (4.0 * t_diff))[0]  # [N, N]

        # Clamp logits to prevent overflow in backward
        logits = raw_logits.clamp(-50, 50)

        # ── Loss 1: Within-chain causal ordering ──
        # For each chain: effect tokens (hop=3) should have higher logit
        # toward root tokens (hop=0) than toward intermediate tokens (hop=1,2)
        ordering_loss = torch.tensor(0.0, device='cuda', requires_grad=True)
        n_ordering = 0

        for cid in range(nc):
            root_idx = [i for i, (c, h_) in enumerate(token_meta) if c == cid and h_ == 0]
            inter_idx = [i for i, (c, h_) in enumerate(token_meta) if c == cid and h_ in (1, 2)]
            effect_idx = [i for i, (c, h_) in enumerate(token_meta) if c == cid and h_ == 3]

            if root_idx and inter_idx and effect_idx:
                # Effect → root logits vs effect → intermediate logits
                e2r = logits[effect_idx][:, root_idx].mean()
                e2i = logits[effect_idx][:, inter_idx].mean()
                # Hinge: effect→root should exceed effect→intermediate by margin
                margin = 1.0
                ordering_loss = ordering_loss + F.relu(margin - (e2r - e2i))
                n_ordering += 1

        if n_ordering > 0:
            ordering_loss = ordering_loss / n_ordering

        # ── Loss 2: Cross-chain separation (maintain) ──
        within_vals = []
        cross_vals = []
        for i in range(N):
            ci, _ = token_meta[i]
            if ci < 0:
                continue
            for j in range(N):
                cj, _ = token_meta[j]
                if i == j or cj < 0:
                    continue
                if ci == cj:
                    within_vals.append(logits[i, j])
                else:
                    cross_vals.append(logits[i, j])

        if within_vals and cross_vals:
            ns = min(100, len(within_vals), len(cross_vals))
            w_s = random.sample(within_vals, ns)
            c_s = random.sample(cross_vals, ns)
            mw = torch.stack(w_s).mean()
            mc = torch.stack(c_s).mean()
            sep_loss = F.relu(2.0 - (mw - mc))
        else:
            sep_loss = torch.tensor(0.0, device='cuda')
            mw = torch.tensor(0.0)
            mc = torch.tensor(0.0)

        # ── Sustained Criticality Framework ──

        # CV floor/ceiling: keep CV in [3, 8]
        cv = g.std() / (g.mean() + 1e-8)
        cv_floor = F.relu(3.0 - cv) ** 2
        cv_ceil = F.relu(cv - 8.0) ** 2
        cv_loss = cv_floor + cv_ceil

        # D²/4τ criticality target = 60 for d=2560
        # Sample pairs and compute D²
        n_sample = min(200, N * (N - 1) // 2)
        ii = torch.randint(0, N, (n_sample,), device='cuda')
        jj = (ii + torch.randint(1, N, (n_sample,), device='cuda')) % N
        delta = h_normed[0, ii, :] - h_normed[0, jj, :]
        g_avg = (g[0, ii, :] + g[0, jj, :]) * 0.5
        D_sq = (delta * g_avg * delta).sum(dim=-1).mean()
        tau = dynamics.compute_tau(h)
        tau_mean_val = tau.mean()
        D_sq_4tau = D_sq / (4.0 * tau_mean_val + 1e-8)
        crit_loss = (D_sq_4tau - 60.0) ** 2 / 60.0  # normalized

        # tau quality: mean near 1.0, log spread near 0.6
        tau_loss = (tau_mean_val - 1.0) ** 2
        log_tau = (tau + 1e-8).log()
        tau_log_spread = log_tau.std()
        tau_spread_loss = (tau_log_spread - 0.6) ** 2

        warmup = min(1.0, step / 100.0)
        # Task losses (chain ordering + separation) + criticality scaffolding
        task_loss = ordering_loss + 0.5 * sep_loss
        homeostasis = 0.1 * cv_loss + 0.01 * crit_loss + 0.05 * tau_loss + 0.01 * tau_spread_loss
        total_loss = warmup * task_loss + homeostasis

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(dynamics.parameters()) + list(context_pool.parameters()), args.grad_clip)
        optimizer.step()

        dt = time.time() - t0

        if step % args.log_every == 0 or step == 1:
            msg = (f"[step={step}] L={total_loss.item():.3f} "
                   f"ord={ordering_loss.item():.2f} "
                   f"sep={sep_loss.item():.2f} "
                   f"gap={mw.item()-mc.item():.1f} "
                   f"CV={cv.item():.2f} "
                   f"D2/4t={D_sq_4tau.item():.1f} "
                   f"tau={tau_mean_val.item():.2f}±{tau_log_spread.item():.2f} "
                   f"g={grad_norm:.2f} {dt:.1f}s")
            print(f"  {msg}")
            log_f.write(msg + '\n')
            log_f.flush()

        if step % args.save_every == 0:
            path = os.path.join(args.output_dir, 'checkpoints', f'step_{step}.pt')
            torch.save({
                'step': step, 'dynamics_state': dynamics.state_dict(),
                'context_pool_state': context_pool.state_dict(),
                'loss': total_loss.item(), 'cv': cv.item(),
                'ordering_loss': ordering_loss.item(),
            }, path)
            print(f"  >> Saved {path}")
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                torch.save({
                    'step': step, 'dynamics_state': dynamics.state_dict(),
                    'context_pool_state': context_pool.state_dict(),
                }, os.path.join(args.output_dir, 'checkpoints', 'best.pt'))

    log_f.close()
    print(f"\n{'='*70}")
    print(f"Done. Best loss={best_loss:.4f}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
