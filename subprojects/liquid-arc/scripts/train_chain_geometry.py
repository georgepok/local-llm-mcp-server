"""Train MetricNet with chain-supervised contrastive loss.

Loss: within-chain token pairs should have HIGH heat kernel attention,
cross-chain pairs should have LOW. This teaches the MetricNet to group
causally connected tokens regardless of text position.

Training data: synthetic interleaved causal chains with explicit chain labels.

Run (~10GB):
  export PYTHONPATH=/workspace/liquid-arc:/workspace/fgn-v3
  python3 -u scripts/train_chain_geometry.py
"""

import argparse, random, time, torch, torch.nn.functional as F, math, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════════
# CHAIN DATA GENERATOR
# ═══════════════════════════════════════════════════════════════

CHAIN_TEMPLATES = [
    # (root_cause, step1, step2, effect)
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


def generate_chain_batch(tokenizer, batch_size, max_length, n_chains=3, n_hops=4):
    """Generate interleaved chain text with explicit chain labels per token."""
    texts = []
    all_chain_labels = []

    for _ in range(batch_size):
        # Pick n_chains random chains
        chains = random.sample(CHAIN_TEMPLATES, min(n_chains, len(CHAIN_TEMPLATES)))
        # Interleave: sentence i from each chain, then sentence i+1 from each, etc.
        sentences = []
        sent_chain_ids = []
        for hop in range(n_hops):
            for chain_id, chain in enumerate(chains):
                sentences.append(chain[hop] + ".")
                sent_chain_ids.append(chain_id)

        text = " ".join(sentences)
        texts.append(text)

        # Tokenize and map tokens to chain IDs
        tokens = tokenizer.encode(text, add_special_tokens=False)
        # Map by character position
        chain_labels = []
        char_pos = 0
        sent_starts = []
        pos = 0
        for s in sentences:
            idx = text.find(s, pos)
            sent_starts.append((idx, idx + len(s)))
            pos = idx + len(s)

        for tid in tokens:
            t = tokenizer.decode([tid])
            mid = char_pos + len(t) // 2
            assigned = -1
            for si, (sb, se) in enumerate(sent_starts):
                if sb <= mid < se:
                    assigned = sent_chain_ids[si]
                    break
            chain_labels.append(assigned)
            char_pos += len(t)

        all_chain_labels.append(chain_labels)

    encoded = tokenizer(texts, return_tensors='pt', truncation=True,
                        max_length=max_length, padding=True)
    return encoded, all_chain_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='/workspace/models/qwen3-4b')
    parser.add_argument('--ode_checkpoint',
                        default='/workspace/liquid-arc/output_e2e_ce_v2/checkpoints/step_1500.pt')
    parser.add_argument('--config', default='/workspace/liquid-arc/configs/mind_layerwise.yaml')
    parser.add_argument('--output_dir', default='/workspace/liquid-arc/output_chain_geo')
    parser.add_argument('--max_steps', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=500)
    parser.add_argument('--extract_layer', type=int, default=18)
    parser.add_argument('--n_ode_steps', type=int, default=16)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from liquid_arc.config import LiquidARCConfig
    from liquid_arc.dynamics import ContinuousDynamics
    from liquid_arc.context_pool import ContextPool
    from liquid_arc.solver import euler_solve

    print("=" * 70)
    print("CHAIN-SUPERVISED GEOMETRY TRAINING")
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
    print(f"  LLM: {llm.config.num_hidden_layers} layers, d={llm.config.hidden_size}")

    dynamics = ContinuousDynamics(config).to('cuda').float().train()
    context_pool = ContextPool(config).to('cuda').float().train()
    ckpt = torch.load(args.ode_checkpoint, map_location='cuda', weights_only=False)
    dynamics.load_state_dict(ckpt['dynamics_state'])
    context_pool.load_state_dict(ckpt['context_pool_state'])
    dynamics = dynamics.float().train()
    context_pool = context_pool.float().train()
    dynamics.freeze_tau = False

    optimizer = torch.optim.Adam(
        list(dynamics.parameters()) + list(context_pool.parameters()), lr=args.lr)
    print(f"  LR={args.lr}, extract_layer={args.extract_layer}, ODE steps={args.n_ode_steps}")
    print(f"  GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    log_f = open(os.path.join(args.output_dir, 'train.log'), 'w')
    best_loss = float('inf')

    for step in range(1, args.max_steps + 1):
        t0 = time.time()

        # Generate chain data
        n_chains = random.choice([2, 3, 4])
        n_hops = random.choice([3, 4])
        batch, chain_labels = generate_chain_batch(tok, 1, 256, n_chains, n_hops)
        input_ids = batch['input_ids'].to('cuda')
        N = input_ids.shape[1]
        labels = chain_labels[0][:N]

        # Get mid-layer delta from LLM
        with torch.no_grad():
            out = llm(input_ids=input_ids, output_hidden_states=True)
        h_cur = out.hidden_states[args.extract_layer]
        h_prev = out.hidden_states[args.extract_layer - 1]
        delta = (h_cur - h_prev).float()
        delta = delta - delta.mean(dim=1, keepdim=True)
        rms = delta.pow(2).mean().sqrt().clamp(min=1e-8)
        h_input = delta / rms

        # Full ODE integration (differentiable)
        mask = torch.ones(1, N, dtype=torch.bool, device='cuda')
        context = context_pool(h_input, mask)
        dynamics.set_context(context, mask=None)
        dynamics.set_n_steps(args.n_ode_steps)
        h_ode = euler_solve(dynamics, h_input, t_span=(0, 2.0), n_steps=args.n_ode_steps)

        # Compute heat kernel from ODE state
        h_normed = dynamics.norm_geo(h_ode)
        ctx_exp = context.unsqueeze(1).expand(-1, N, -1)
        mi = torch.cat([h_normed, ctx_exp], dim=-1)
        hidden = F.gelu(dynamics.metric_net_linear1(mi))
        g = F.softplus(dynamics.metric_net_linear2_diag(hidden))
        sqrt_g = g.sqrt()
        qk = h_normed * sqrt_g
        t_diff = F.softplus(dynamics.t_diffusion)
        dot_qk = torch.bmm(qk, qk.transpose(1, 2)) / (2.0 * t_diff)
        k_norm_sq = (qk * qk).sum(dim=-1, keepdim=True)
        logits = dot_qk - k_norm_sq.transpose(1, 2) / (4.0 * t_diff)
        L = logits[0]  # [N, N] raw pre-softmax logits (carry gradient even when softmax is ~0)

        # ── Chain contrastive loss on raw logits ──
        # Within-chain logits should be HIGHER than cross-chain logits
        within_vals = []
        cross_vals = []
        for i in range(N):
            if labels[i] < 0:
                continue
            for j in range(N):
                if i == j or labels[j] < 0:
                    continue
                if labels[i] == labels[j]:
                    within_vals.append(L[i, j])
                else:
                    cross_vals.append(L[i, j])

        if within_vals and cross_vals:
            n_sample = min(200, len(within_vals), len(cross_vals))
            w_sample = random.sample(within_vals, n_sample)
            c_sample = random.sample(cross_vals, n_sample)
            mean_within = torch.stack(w_sample).mean()
            mean_cross = torch.stack(c_sample).mean()
            # Hinge on logits: within should exceed cross by margin
            margin = 2.0  # in logit space
            chain_loss = F.relu(margin - (mean_within - mean_cross))
        else:
            chain_loss = torch.tensor(0.0, device='cuda')
            mean_within = torch.tensor(0.0)
            mean_cross = torch.tensor(0.0)

        # CV loss: keep metric structured
        cv = g.std() / (g.mean() + 1e-8)
        cv_loss = (cv - 6.0) ** 2

        total_loss = chain_loss + 0.01 * cv_loss

        if torch.isnan(total_loss):
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(dynamics.parameters()) + list(context_pool.parameters()), args.grad_clip)
        optimizer.step()

        dt = time.time() - t0

        if step % args.log_every == 0 or step == 1:
            msg = (f"[step={step}] loss={total_loss.item():.4f} "
                   f"chain={chain_loss.item():.4f} "
                   f"w={mean_within.item():.4f} x={mean_cross.item():.4f} "
                   f"gap={mean_within.item()-mean_cross.item():.4f} "
                   f"CV={cv.item():.2f} "
                   f"grad={grad_norm:.3f} {dt:.1f}s")
            print(f"  {msg}")
            log_f.write(msg + '\n')
            log_f.flush()

        if step % args.save_every == 0:
            path = os.path.join(args.output_dir, 'checkpoints', f'step_{step}.pt')
            torch.save({
                'step': step, 'dynamics_state': dynamics.state_dict(),
                'context_pool_state': context_pool.state_dict(),
                'loss': total_loss.item(), 'cv': cv.item(),
                'within_attn': mean_within.item(), 'cross_attn': mean_cross.item(),
            }, path)
            print(f"  >> Saved {path}")
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                torch.save({
                    'step': step, 'dynamics_state': dynamics.state_dict(),
                    'context_pool_state': context_pool.state_dict(),
                    'loss': total_loss.item(), 'cv': cv.item(),
                }, os.path.join(args.output_dir, 'checkpoints', 'best.pt'))

    log_f.close()
    print(f"\n{'='*70}")
    print(f"Done. Best loss={best_loss:.4f}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
