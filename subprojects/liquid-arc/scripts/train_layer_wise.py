"""Train MetricNet on LLM residual streams via layer-wise bias injection.

Qwen3-4B is frozen. LiquidARC dynamics (MetricNet, TauNet, FFN, W_v, W_o)
are trained. CE loss flows through: biased attention -> bias matrix ->
ODE correction -> MetricNet weights.

Data: WikiText-2 (generic NTP) + synthetic causal chains (targeted routing).

Run in fgn-train container:
  export PYTHONPATH=/workspace/liquid-arc:/workspace/fgn-v3
  python3 -u scripts/train_layer_wise.py \
      --model_path /workspace/models/qwen3-4b \
      --ode_checkpoint /workspace/liquid-arc/output_crit_2560/checkpoints/best.pt \
      --max_steps 3000 --log_every 10
"""

import argparse
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════════
# SYNTHETIC CAUSAL CHAIN GENERATOR
# ═══════════════════════════════════════════════════════════════

ENTITIES = [
    "bridge", "dam", "factory", "pipeline", "power plant", "highway",
    "rail line", "port", "airport", "hospital", "warehouse", "school",
]
EVENTS = [
    "collapsed", "was damaged", "caught fire", "flooded", "lost power",
    "was contaminated", "was blocked", "shut down", "overflowed", "broke down",
]
CONSEQUENCES = [
    "caused traffic rerouting through {place}",
    "forced evacuation of {place}",
    "disrupted supply chains to {place}",
    "overwhelmed emergency services in {place}",
    "triggered a shortage of {resource} in {place}",
    "delayed construction at {place}",
    "cut off communication with {place}",
    "caused prices to spike in {place}",
]
PLACES = ["downtown", "the suburbs", "the industrial district", "the harbor",
           "neighboring towns", "the city center", "the north side", "three counties"]
RESOURCES = ["food", "water", "fuel", "medical supplies", "building materials", "electricity"]


def generate_causal_chain(n_hops=3):
    """Generate a synthetic causal chain with n_hops."""
    entity = random.choice(ENTITIES)
    event = random.choice(EVENTS)
    chain = [f"The {entity} {event}"]

    for i in range(n_hops - 1):
        template = random.choice(CONSEQUENCES)
        place = random.choice(PLACES)
        resource = random.choice(RESOURCES)
        consequence = template.format(place=place, resource=resource)
        chain.append(consequence)

    # Build text
    text = chain[0] + ". "
    for i in range(1, len(chain)):
        connector = random.choice(["Because of this, ", "This ", "As a result, ", "Consequently, "])
        text += connector + chain[i] + ". "

    # Add question
    text += f"What was the root cause? The root cause was the {entity} {event.split()[0]}."
    return text


def generate_training_batch(tokenizer, batch_size, max_length, causal_ratio=0.3):
    """Generate a mixed batch of WikiText-style text and causal chains."""
    texts = []
    for _ in range(batch_size):
        if random.random() < causal_ratio:
            n_hops = random.choice([2, 3, 4, 5])
            texts.append(generate_causal_chain(n_hops))
        else:
            # Simple synthetic text (no WikiText dependency for now)
            # Generate varied sentence structures
            templates = [
                "The {adj} {noun} {verb} the {noun2} near the {place}.",
                "After the {event}, {people} began to {action} more carefully.",
                "Reports indicate that {noun} levels have {change} by {num}% since {time}.",
                "The committee decided to {action} the {noun} after reviewing the {noun2}.",
                "Several {people} reported {adj} conditions near the {place} yesterday.",
            ]
            adjs = ["large", "small", "damaged", "new", "old", "critical", "primary"]
            nouns = ["building", "system", "report", "project", "team", "bridge", "network"]
            verbs = ["affected", "improved", "changed", "replaced", "supported", "connected"]
            people = ["residents", "officials", "workers", "engineers", "scientists"]
            actions = ["monitor", "evaluate", "rebuild", "investigate", "coordinate"]
            changes = ["increased", "decreased", "fluctuated", "stabilized"]
            places = ["the river", "downtown", "the facility", "the station", "the campus"]
            times = ["January", "last quarter", "the incident", "the renovation", "2024"]

            sentences = []
            for _ in range(random.randint(3, 8)):
                t = random.choice(templates)
                s = t.format(
                    adj=random.choice(adjs), noun=random.choice(nouns),
                    noun2=random.choice(nouns), verb=random.choice(verbs),
                    place=random.choice(places), event=random.choice(["storm", "review", "audit"]),
                    people=random.choice(people), action=random.choice(actions),
                    change=random.choice(changes), num=random.randint(5, 95),
                    time=random.choice(times))
                sentences.append(s)
            texts.append(" ".join(sentences))

    encoded = tokenizer(texts, return_tensors='pt', truncation=True,
                        max_length=max_length, padding=True)
    return encoded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/workspace/models/qwen3-4b')
    parser.add_argument('--ode_checkpoint', type=str,
                        default='/workspace/liquid-arc/output_crit_2560/checkpoints/best.pt')
    parser.add_argument('--config', type=str,
                        default='/workspace/liquid-arc/configs/mind_layerwise.yaml')
    parser.add_argument('--max_steps', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--lr_metric', type=float, default=3e-4)
    parser.add_argument('--lr_other', type=float, default=1e-4)
    parser.add_argument('--epsilon', type=float, default=0.5)
    parser.add_argument('--causal_ratio', type=float, default=0.3)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=500)
    parser.add_argument('--output_dir', type=str, default='/workspace/liquid-arc/output_layerwise')
    parser.add_argument('--grad_clip', type=float, default=0.5)
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from layer-wise checkpoint (loads dynamics_state)')
    parser.add_argument('--crit_lambda', type=float, default=0.01)
    parser.add_argument('--tau_lambda', type=float, default=0.05)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from liquid_arc.config import LiquidARCConfig
    from liquid_arc.dynamics import ContinuousDynamics
    from liquid_arc.context_pool import ContextPool
    from liquid_arc.layer_wise_ode import LayerWiseODE, hook_llm_layers

    print("=" * 70)
    print("LAYER-WISE MetricNet TRAINING")
    print("=" * 70)

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)

    # Load config
    config = LiquidARCConfig.from_yaml(args.config)
    d_ode = config.d_model

    # Load Qwen3-4B (FROZEN)
    print("\nLoading Qwen3-4B (frozen)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map='cuda',
        torch_dtype=torch.bfloat16, trust_remote_code=True)
    llm.eval()
    for p in llm.parameters():
        p.requires_grad_(False)
    n_layers = llm.config.num_hidden_layers
    d_llm = llm.config.hidden_size
    print(f"  {n_layers} layers, d={d_llm} (frozen)")
    assert d_llm == d_ode, f"Dimension mismatch: LLM d={d_llm}, ODE d={d_ode}. No projection allowed."

    # Load ODE dynamics (TRAINABLE)
    print("Loading ODE dynamics (trainable)...")
    dynamics = ContinuousDynamics(config).to('cuda').to(torch.bfloat16)
    context_pool = ContextPool(config).to('cuda').to(torch.bfloat16)
    ckpt = torch.load(args.ode_checkpoint, map_location='cuda', weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
    cleaned = {k.replace("_orig_mod.", "").replace('metric_net_linear2.', 'metric_net_linear2_diag.'): v
               for k, v in sd.items()}
    dyn_keys = {k: v for k, v in cleaned.items()
                if k.startswith('dynamics.') or k.startswith('context_pool.')}
    holder = nn.ModuleDict({'dynamics': dynamics, 'context_pool': context_pool})
    holder.load_state_dict(dyn_keys, strict=False)
    # Use float32 for ODE dynamics during training (bf16 gradient overflow in 36-step chain)
    dynamics = dynamics.float()
    context_pool = context_pool.float()
    dynamics.train()
    dynamics.freeze_tau = False
    print(f"  ODE step {ckpt.get('step', '?')}, d={d_ode} (float32 for training)")

    # Resume from layer-wise checkpoint if specified
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location='cuda', weights_only=False)
        dynamics.load_state_dict(resume_ckpt['dynamics_state'])
        context_pool.load_state_dict(resume_ckpt['context_pool_state'])
        dynamics = dynamics.float()
        context_pool = context_pool.float()
        dynamics.train()
        dynamics.freeze_tau = False
        print(f"  Resumed from {args.resume} (step {resume_ckpt.get('step', '?')}, "
              f"CV={resume_ckpt.get('cv', '?')})")

    # Optimizer: high LR for MetricNet, lower for everything else
    metric_params = (list(dynamics.metric_net_linear1.parameters()) +
                     list(dynamics.metric_net_linear2_diag.parameters()))
    tau_params = (list(dynamics.tau_net_linear1.parameters()) +
                  list(dynamics.tau_net_linear2.parameters()))
    other_params = (list(dynamics.W_v.parameters()) +
                    list(dynamics.W_o.parameters()) +
                    list(dynamics.ffn.parameters()) +
                    list(dynamics.norm_geo.parameters()) +
                    list(dynamics.norm_val.parameters()) +
                    list(dynamics.norm_ff.parameters()) +
                    list(context_pool.parameters()))

    optimizer = torch.optim.Adam([
        {'params': metric_params, 'lr': args.lr_metric},
        {'params': tau_params, 'lr': args.lr_metric},
        {'params': other_params, 'lr': args.lr_other},
    ])
    n_params = sum(p.numel() for p in dynamics.parameters()) + sum(p.numel() for p in context_pool.parameters())
    print(f"  Trainable params: {n_params:,}")
    print(f"  MetricNet LR: {args.lr_metric}, Other LR: {args.lr_other}")
    print(f"  Epsilon: {args.epsilon}, Causal ratio: {args.causal_ratio}")

    # Create layer-wise ODE
    layer_ode = LayerWiseODE(
        dynamics=dynamics, context_pool=context_pool,
        n_layers=n_layers, d_llm=d_llm, d_ode=d_ode,
        epsilon=args.epsilon, device='cuda')
    layer_ode.training_mode = True

    # Register hooks
    hooks = hook_llm_layers(llm, layer_ode)
    print(f"  {len(hooks)} layer hooks registered")

    # Training loop
    print(f"\nTraining for {args.max_steps} steps...")
    log_file = open(os.path.join(args.output_dir, 'train.log'), 'w')
    best_loss = float('inf')

    for step in range(1, args.max_steps + 1):
        t0 = time.time()

        # Generate batch
        batch = generate_training_batch(
            tokenizer, args.batch_size, args.max_length, args.causal_ratio)
        input_ids = batch['input_ids'].to('cuda')
        attention_mask = batch['attention_mask'].to('cuda')

        # Labels: shift right for NTP
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # ignore padding

        # Forward: run LLM to get residual streams, then train ODE on them.
        # The LLM is frozen and attention_mask injection isn't differentiable,
        # so we use a SELF-SUPERVISED loss on the ODE correction instead of CE.
        #
        # Loss 1 (contrastive): within-sentence token pairs should have higher
        #   cosine similarity in corrected space than cross-sentence pairs.
        # Loss 2 (prediction): correction at layer L should predict residual
        #   stream direction at layer L+1 (next-layer alignment).
        # These train MetricNet to differentiate LLM residual features.

        # Collect residual streams from all layers (no ODE hooks for this pass)
        layer_ode._active = False
        with torch.no_grad():
            outputs = llm(input_ids=input_ids, attention_mask=attention_mask,
                          output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of [B, N, d] for each layer + embed

        # Now run ODE on each layer's residual (differentiable)
        layer_ode.start_forward()
        layer_ode._active = False  # don't use hooks, we drive manually

        all_biases = []
        N = input_ids.shape[1]

        for layer_idx in range(n_layers):
            h_res = hidden_states[layer_idx].detach()  # [B, N, d] — detach from LLM
            # Manual process_layer call (training_mode=True makes it differentiable)
            bias = layer_ode.process_layer(layer_idx, h_res)  # [B, 1, N, N]
            all_biases.append(bias.squeeze(1))  # [B, N, N]

        layer_ode.end_forward()

        # ── Loss 1: Contrastive bias loss ──
        # Sentence boundaries from periods in token text
        # Within-sentence pairs should have HIGHER bias than cross-sentence
        pad_mask = attention_mask.bool()  # [B, N]
        contrastive_loss = torch.tensor(0.0, device='cuda', requires_grad=True)
        n_contrast = 0

        for b_idx in range(args.batch_size):
            # Find sentence boundaries from input tokens
            token_ids = input_ids[b_idx].tolist()
            period_id = tokenizer.encode('.', add_special_tokens=False)
            period_id = period_id[0] if period_id else -1

            sent_ids = []
            current_sent = 0
            for t in token_ids:
                sent_ids.append(current_sent)
                if t == period_id:
                    current_sent += 1

            if current_sent < 2:
                continue  # need at least 2 sentences

            # Use late-layer bias (last third) where routing should be most structured
            late_start = 2 * n_layers // 3
            bias_late = torch.stack(all_biases[late_start:]).mean(dim=0)  # [B, N, N]
            B_mat = bias_late[b_idx]  # [N, N]

            # Sample within-sentence and cross-sentence pairs
            within_vals = []
            across_vals = []
            valid = pad_mask[b_idx]
            valid_idx = valid.nonzero(as_tuple=True)[0]

            for _ in range(min(100, len(valid_idx) ** 2)):
                i = valid_idx[torch.randint(len(valid_idx), (1,))].item()
                j = valid_idx[torch.randint(len(valid_idx), (1,))].item()
                if i == j:
                    continue
                val = B_mat[i, j]
                if sent_ids[i] == sent_ids[j]:
                    within_vals.append(val)
                else:
                    across_vals.append(val)

            if within_vals and across_vals:
                mean_within = torch.stack(within_vals).mean()
                mean_across = torch.stack(across_vals).mean()
                # Hinge: within should exceed across by margin
                margin = 2.0
                contrastive_loss = contrastive_loss + F.relu(margin - (mean_within - mean_across))
                n_contrast += 1

        if n_contrast > 0:
            contrastive_loss = contrastive_loss / n_contrast

        # ── Loss 2: CV target (differentiable via g computation) ──
        # Push CV toward 6.0 for structured routing, penalize both below and above
        mid_layer = n_layers // 2
        h_mid = hidden_states[mid_layer].detach().to(next(dynamics.parameters()).dtype)
        g_mid = dynamics.compute_metric_diag(h_mid)
        cv_mid = g_mid.std() / (g_mid.mean() + 1e-8)
        cv_target = 6.0
        cv_loss = (cv_mid - cv_target) ** 2

        # ── Loss 3: Tau quality (differentiable) ──
        # h_mid already cast to dynamics dtype above
        tau_mid = dynamics.compute_tau(h_mid)
        tau_loss = (tau_mid.mean() - 1.0) ** 2

        # LR warmup: scale losses by ramp factor for first 100 steps
        warmup_factor = min(1.0, step / 100.0)
        total_loss = warmup_factor * (contrastive_loss + 0.05 * cv_loss + 0.02 * tau_loss)

        # NaN guard
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"  [step={step}] NaN/Inf detected — skipping step")
            optimizer.zero_grad()
            continue

        # Also compute CE for monitoring (non-differentiable)
        with torch.no_grad():
            logits = outputs.logits[:, :-1, :].contiguous()
            target = labels[:, 1:].contiguous()
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                      target.view(-1), ignore_index=-100)

        # Diagnostics
        diags = layer_ode.get_layer_diagnostics()
        cv_vals = [d['cv'] for d in diags]
        mean_cv = sum(cv_vals) / len(cv_vals) if cv_vals else 0
        tau_vals = [d['tau_mean'] for d in diags]
        mean_tau = sum(tau_vals) / len(tau_vals) if tau_vals else 1.0

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(dynamics.parameters()) + list(context_pool.parameters()),
            args.grad_clip)
        optimizer.step()

        step_time = time.time() - t0

        # Logging
        if step % args.log_every == 0 or step == 1:
            corr_ratios = [d.get('correction_ratio', 0) for d in diags]
            final_cr = corr_ratios[-1] if corr_ratios else 0

            msg = (f"[step={step}] loss={total_loss.item():.4f} "
                   f"contrast={contrastive_loss.item():.4f} "
                   f"cv_loss={cv_loss.item():.4f} "
                   f"ce(mon)={ce_loss.item():.4f} "
                   f"CV={mean_cv:.3f} cv_mid={cv_mid.item():.3f} "
                   f"tau={mean_tau:.3f} "
                   f"c_rat={final_cr:.4f} "
                   f"grad={grad_norm:.3f} "
                   f"{step_time:.1f}s")
            print(f"  {msg}")
            log_file.write(msg + '\n')
            log_file.flush()

        # Save checkpoints
        if step % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, 'checkpoints', f'step_{step}.pt')
            torch.save({
                'step': step,
                'dynamics_state': dynamics.state_dict(),
                'context_pool_state': context_pool.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'train_loss': total_loss.item(),
                'ce_loss': ce_loss.item(),
                'cv': mean_cv,
                'tau': mean_tau,
            }, ckpt_path)
            print(f"  >> Saved {ckpt_path}")

            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_path = os.path.join(args.output_dir, 'checkpoints', 'best.pt')
                torch.save({
                    'step': step,
                    'dynamics_state': dynamics.state_dict(),
                    'context_pool_state': context_pool.state_dict(),
                    'train_loss': total_loss.item(),
                    'ce_loss': ce_loss.item(),
                    'cv': mean_cv,
                    'tau': mean_tau,
                }, best_path)
                print(f"  >> New best: loss={best_loss:.4f}")

    # Final save
    final_path = os.path.join(args.output_dir, 'checkpoints', 'final.pt')
    torch.save({
        'step': args.max_steps,
        'dynamics_state': dynamics.state_dict(),
        'context_pool_state': context_pool.state_dict(),
        'train_loss': total_loss.item(),
        'ce_loss': ce_loss.item(),
        'cv': mean_cv,
        'tau': mean_tau,
    }, final_path)

    # Cleanup
    for h in hooks:
        h.remove()
    log_file.close()

    print(f"\n{'='*70}")
    print(f"Training complete: {args.max_steps} steps")
    print(f"Final: loss={total_loss.item():.4f} CE={ce_loss.item():.4f} CV={mean_cv:.3f}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
