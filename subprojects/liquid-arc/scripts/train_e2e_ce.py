"""End-to-end CE training: MetricNet learns routing that improves next-token prediction.

CE loss → logits → attention(QK^T + ODE_bias) → bias → correction → MetricNet

Requires attn_implementation="eager" (SDPA doesn't propagate gradients through mask).
Qwen3-4B frozen, only LiquidARC dynamics train.

Memory estimate: ~12GB (8GB model + 2GB activations + 2GB grad tape for ODE)

Run in fgn-train container:
  export PYTHONPATH=/workspace/liquid-arc:/workspace/fgn-v3
  python3 -u scripts/train_e2e_ce.py
"""

import argparse
import time
import random
import torch
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_text_batch(tokenizer, batch_size, max_length):
    """Generate synthetic text for NTP training."""
    templates = [
        "The {adj} {noun} {verb} the {noun2} near the {place}. ",
        "After the {event}, {people} began to {action} more carefully. ",
        "Reports indicate that {noun} levels have {change} since {time}. ",
        "The committee decided to {action} the {noun} after reviewing the data. ",
        "{people} reported {adj} conditions near the {place} yesterday. ",
        "A {noun} {verb} the {noun2}, causing significant {event}. ",
        "The {adj} {noun} was {verb} by {people} near the {place}. ",
    ]
    adjs = ["large", "small", "damaged", "new", "critical", "primary", "severe", "unexpected"]
    nouns = ["building", "system", "bridge", "project", "network", "pipeline", "facility", "dam"]
    verbs = ["affected", "improved", "damaged", "replaced", "disrupted", "connected", "blocked"]
    people = ["residents", "officials", "workers", "engineers", "investigators"]
    actions = ["monitor", "evaluate", "rebuild", "investigate", "evacuate", "repair"]
    changes = ["increased", "decreased", "fluctuated", "stabilized", "deteriorated"]
    places = ["the river", "downtown", "the facility", "the station", "the highway"]
    events = ["storm", "collapse", "contamination", "outage", "flooding", "explosion"]
    times = ["January", "last quarter", "the incident", "the renovation", "Monday"]

    # Also mix in causal chains (30%)
    chain_templates = [
        "A {event} at the {place} caused {noun} damage. This led to {people} being displaced. The displacement overwhelmed {place2} resources. ",
        "The {noun} failure triggered a chain reaction. First, {noun2} stopped working. Then {people} lost access to essential services. ",
        "When the {noun} {verb} the {place}, it disrupted supply chains. Stores ran out of {noun2}. Prices increased significantly. ",
    ]
    places2 = ["shelter", "hospital", "school", "community center", "emergency services"]

    texts = []
    for _ in range(batch_size):
        sentences = []
        n_sent = random.randint(4, 10)
        use_chain = random.random() < 0.3
        if use_chain:
            t = random.choice(chain_templates)
            sentences.append(t.format(
                event=random.choice(events), place=random.choice(places),
                noun=random.choice(nouns), noun2=random.choice(nouns),
                people=random.choice(people), verb=random.choice(verbs),
                place2=random.choice(places2)))
        for _ in range(n_sent):
            t = random.choice(templates)
            sentences.append(t.format(
                adj=random.choice(adjs), noun=random.choice(nouns),
                noun2=random.choice(nouns), verb=random.choice(verbs),
                place=random.choice(places), event=random.choice(events),
                people=random.choice(people), action=random.choice(actions),
                change=random.choice(changes), time=random.choice(times)))
        texts.append("".join(sentences))

    encoded = tokenizer(texts, return_tensors="pt", truncation=True,
                        max_length=max_length, padding=True)
    return encoded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='/workspace/models/qwen3-4b')
    parser.add_argument('--ode_checkpoint',
                        default='/workspace/liquid-arc/output_layerwise_v3/checkpoints/step_1500.pt')
    parser.add_argument('--config', default='/workspace/liquid-arc/configs/mind_layerwise.yaml')
    parser.add_argument('--output_dir', default='/workspace/liquid-arc/output_e2e_ce')
    parser.add_argument('--max_steps', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epsilon', type=float, default=0.5)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=500)
    parser.add_argument('--eval_every', type=int, default=100)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from liquid_arc.config import LiquidARCConfig
    from liquid_arc.dynamics import ContinuousDynamics
    from liquid_arc.context_pool import ContextPool
    from liquid_arc.layer_wise_ode import LayerWiseODE, hook_llm_layers

    print("=" * 70)
    print("END-TO-END CE TRAINING — MetricNet via biased attention")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    config = LiquidARCConfig.from_yaml(args.config)

    # Load Qwen3-4B with EAGER attention (required for gradient through mask)
    print("\nLoading Qwen3-4B (eager attention, frozen)...")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    llm = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="cuda",
        torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager")
    llm.eval()
    for p in llm.parameters():
        p.requires_grad_(False)
    d = llm.config.hidden_size
    n_layers = llm.config.num_hidden_layers
    print(f"  {n_layers} layers, d={d}")

    # Load ODE (float32 for stable gradients, trainable)
    print("Loading ODE dynamics (trainable, float32)...")
    dynamics = ContinuousDynamics(config).to("cuda").float().train()
    context_pool = ContextPool(config).to("cuda").float().train()
    ckpt = torch.load(args.ode_checkpoint, map_location="cuda", weights_only=False)
    dynamics.load_state_dict(ckpt["dynamics_state"])
    context_pool.load_state_dict(ckpt["context_pool_state"])
    dynamics = dynamics.float().train()
    context_pool = context_pool.float().train()
    dynamics.freeze_tau = False

    n_params = sum(p.numel() for p in dynamics.parameters() if p.requires_grad)
    n_params += sum(p.numel() for p in context_pool.parameters() if p.requires_grad)
    print(f"  Trainable: {n_params:,} params, eps={args.epsilon}")

    # Create ODE in training mode
    layer_ode = LayerWiseODE(dynamics=dynamics, context_pool=context_pool,
        n_layers=n_layers, d_llm=d, d_ode=config.d_model,
        epsilon=args.epsilon, device="cuda")
    layer_ode.training_mode = True

    # Register hooks on every 6th layer only (6 hooks instead of 36).
    # Full 36-layer backward is too slow with eager attention.
    # 6 injection points still cover early/mid/late depth.
    hook_every = 6
    inject_layers = list(range(hook_every - 1, n_layers, hook_every))
    hooks = []
    model_layers = llm.model.layers
    for i in inject_layers:
        def make_hook(layer_idx):
            def hook_fn(module, args, kwargs):
                if not layer_ode._active:
                    return
                hidden_states = args[0] if args else kwargs.get('hidden_states')
                if hidden_states is None:
                    return
                bias = layer_ode.process_layer(layer_idx, hidden_states)
                attn_mask = kwargs.get('attention_mask', None)
                if attn_mask is not None:
                    _, _, N_b, _ = bias.shape
                    seq_len = attn_mask.shape[-1]
                    n = min(N_b, seq_len)
                    if n > 0:
                        injection = torch.zeros_like(attn_mask)
                        injection[:, :, :n, :n] = bias[:, :, :n, :n]
                        kwargs['attention_mask'] = attn_mask + injection
            return hook_fn
        h = model_layers[i].register_forward_pre_hook(make_hook(i), with_kwargs=True)
        hooks.append(h)
    print(f"  {len(hooks)} hooks at layers {inject_layers} (eager attention)")

    # Optimizer
    optimizer = torch.optim.Adam(
        list(dynamics.parameters()) + list(context_pool.parameters()),
        lr=args.lr)

    # Also compute plain CE baseline for comparison
    print("\nComputing plain CE baseline...")
    layer_ode._active = False
    with torch.no_grad():
        baseline_batch = generate_text_batch(tok, 10, args.max_length)
        baseline_ids = baseline_batch["input_ids"].to("cuda")
        baseline_mask = baseline_batch["attention_mask"].to("cuda")
        baseline_out = llm(input_ids=baseline_ids, attention_mask=baseline_mask)
        baseline_logits = baseline_out.logits[:, :-1, :].contiguous()
        baseline_labels = baseline_ids[:, 1:].contiguous()
        baseline_labels[baseline_mask[:, 1:] == 0] = -100
        baseline_ce = F.cross_entropy(
            baseline_logits.view(-1, baseline_logits.size(-1)),
            baseline_labels.view(-1), ignore_index=-100).item()
    print(f"  Plain CE baseline: {baseline_ce:.4f}")
    layer_ode._active = True

    # Training loop
    print(f"\nTraining for {args.max_steps} steps...")
    print(f"  GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    log_f = open(os.path.join(args.output_dir, "train.log"), "w")
    best_loss = float("inf")

    for step in range(1, args.max_steps + 1):
        t0 = time.time()

        batch = generate_text_batch(tok, args.batch_size, args.max_length)
        input_ids = batch["input_ids"].to("cuda")
        attn_mask = batch["attention_mask"].to("cuda")
        labels = input_ids.clone()
        labels[attn_mask == 0] = -100

        # Forward with ODE hooks active (training_mode=True for differentiable path)
        layer_ode.start_forward()
        outputs = llm(input_ids=input_ids, attention_mask=attn_mask)

        logits = outputs.logits[:, :-1, :].contiguous()
        target = labels[:, 1:].contiguous()
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target.view(-1), ignore_index=-100)

        # NaN guard
        if torch.isnan(ce_loss) or torch.isinf(ce_loss):
            print(f"  [step={step}] NaN — skipping")
            optimizer.zero_grad()
            layer_ode.end_forward()
            continue

        optimizer.zero_grad()
        ce_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(dynamics.parameters()) + list(context_pool.parameters()),
            args.grad_clip)
        optimizer.step()
        layer_ode.end_forward()

        dt = time.time() - t0

        if step % args.log_every == 0 or step == 1:
            diags = layer_ode.get_layer_diagnostics()
            cv = sum(d['cv'] for d in diags) / len(diags) if diags else 0
            cr = diags[-1].get('correction_ratio', 0) if diags else 0
            mem = torch.cuda.memory_allocated() / 1e9

            msg = (f"[step={step}] ce={ce_loss.item():.4f} "
                   f"base={baseline_ce:.4f} "
                   f"delta={ce_loss.item()-baseline_ce:+.4f} "
                   f"CV={cv:.2f} c_rat={cr:.4f} "
                   f"grad={grad_norm:.3f} "
                   f"mem={mem:.1f}GB {dt:.1f}s")
            print(f"  {msg}")
            log_f.write(msg + "\n")
            log_f.flush()

        if step % args.save_every == 0:
            path = os.path.join(args.output_dir, "checkpoints", f"step_{step}.pt")
            torch.save({
                "step": step,
                "dynamics_state": dynamics.state_dict(),
                "context_pool_state": context_pool.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "ce_loss": ce_loss.item(),
                "baseline_ce": baseline_ce,
            }, path)
            print(f"  >> Saved {path}")
            if ce_loss.item() < best_loss:
                best_loss = ce_loss.item()
                torch.save({
                    "step": step,
                    "dynamics_state": dynamics.state_dict(),
                    "context_pool_state": context_pool.state_dict(),
                    "ce_loss": ce_loss.item(),
                    "baseline_ce": baseline_ce,
                }, os.path.join(args.output_dir, "checkpoints", "best.pt"))

    # Final save
    torch.save({
        "step": args.max_steps,
        "dynamics_state": dynamics.state_dict(),
        "context_pool_state": context_pool.state_dict(),
        "ce_loss": ce_loss.item(),
        "baseline_ce": baseline_ce,
    }, os.path.join(args.output_dir, "checkpoints", "final.pt"))

    for h in hooks:
        h.remove()
    log_f.close()

    print(f"\n{'='*70}")
    print(f"Training complete. Final CE={ce_loss.item():.4f} vs baseline={baseline_ce:.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
