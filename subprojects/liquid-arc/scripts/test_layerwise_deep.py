"""Deep analysis of layer-wise ODE co-processing.

Two focus areas:
  1. Causal chain test: does per-layer routing IMPROVE chain reasoning?
     - Multiple prompts at varying difficulty (2-hop, 3-hop, 5-hop)
     - Compare ODE-on vs ODE-off on each
     - Score: does response correctly identify root cause?

  2. Per-layer geometric diagnostics:
     - CV per layer (metric complexity evolution through depth)
     - D² per layer (pairwise geodesic distance — how separated are tokens?)
     - Bias range per layer (how strongly is attention being shaped?)
     - Attention entropy per layer (uniform vs structured routing)
     - Cross-event vs within-event bias (does geometry separate events?)

Run in fgn-train container:
  export PYTHONPATH=/workspace/liquid-arc:/workspace/fgn-v3
  python3 -u scripts/test_layerwise_deep.py
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════════
# CAUSAL CHAIN TEST SUITE
# ═══════════════════════════════════════════════════════════════

CAUSAL_TESTS = [
    # 2-hop chains
    {
        'name': '2-hop: bridge → reroute',
        'prompt': (
            "A bridge collapsed. Because of this, trucks were rerouted through downtown. "
            "What caused the truck rerouting?"
        ),
        'root_cause': 'bridge',
        'hops': 2,
    },
    {
        'name': '2-hop: rain → flood',
        'prompt': (
            "Heavy rain lasted three days. This caused the river to overflow its banks. "
            "What caused the river to overflow?"
        ),
        'root_cause': 'rain',
        'hops': 2,
    },
    # 3-hop chains
    {
        'name': '3-hop: bridge → reroute → disruption',
        'prompt': (
            "A bridge collapsed on March 1st. Because of this, trucks were rerouted "
            "through downtown. The downtown rerouting caused supply disruptions at "
            "the warehouse. What was the root cause of the supply disruptions?"
        ),
        'root_cause': 'bridge',
        'hops': 3,
    },
    {
        'name': '3-hop: drought → crop → price',
        'prompt': (
            "A severe drought hit the farming region. The drought destroyed most of the "
            "wheat crop. The wheat shortage caused bread prices to triple at grocery stores. "
            "What was the root cause of the bread price increase?"
        ),
        'root_cause': 'drought',
        'hops': 3,
    },
    # 5-hop chains (harder — requires tracking longer dependency)
    {
        'name': '5-hop: earthquake → dam → flood → evacuation → shortage',
        'prompt': (
            "An earthquake damaged a dam upstream. The damaged dam eventually broke, "
            "flooding the valley below. The flooding forced evacuation of three towns. "
            "The evacuated residents overwhelmed shelters in neighboring cities. "
            "The shelter overcrowding caused a severe food shortage in those cities. "
            "What was the root cause of the food shortage?"
        ),
        'root_cause': 'earthquake',
        'hops': 5,
    },
    {
        'name': '5-hop: hack → shutdown → delay → cancel → lawsuit',
        'prompt': (
            "Hackers breached the airport's computer system. This forced a complete "
            "shutdown of air traffic control. The shutdown delayed all flights by 12 hours. "
            "Many connecting passengers missed their international flights. The missed "
            "connections caused a class-action lawsuit against the airport. "
            "What was the root cause of the lawsuit?"
        ),
        'root_cause': 'hack',
        'hops': 5,
    },
]


def score_response(response: str, root_cause: str) -> bool:
    """Check if response identifies the root cause."""
    return root_cause.lower() in response.lower()


def generate_plain(llm, tokenizer, prompt, max_new_tokens=100, temperature=0.7):
    """Generate without ODE."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer directly and concisely."},
        {"role": "user", "content": prompt},
    ]
    try:
        full_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        full_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(full_prompt, return_tensors='pt', truncation=True,
                       max_length=2048).to('cuda')
    with torch.no_grad():
        out = llm.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=temperature, do_sample=temperature > 0,
            top_p=0.9, repetition_penalty=1.2)
    text = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:],
                            skip_special_tokens=True)
    # Strip thinking
    import re
    match = re.search(r'</think>\s*(.*)', text, flags=re.DOTALL)
    if match and len(match.group(1).strip()) > 10:
        text = match.group(1).strip()
    text = re.sub(r'</?think>', '', text).strip()
    return text


# ═══════════════════════════════════════════════════════════════
# PER-LAYER GEOMETRIC ANALYSIS
# ═══════════════════════════════════════════════════════════════

def compute_layer_D_sq(dynamics, h_ode, n_pairs=200):
    """Compute median pairwise D² from ODE state at a given layer."""
    B, N, d = h_ode.shape
    if N < 2:
        return 0.0

    param_dtype = next(dynamics.parameters()).dtype
    h = h_ode.to(param_dtype)

    h_normed = dynamics.norm_geo(h)
    context = dynamics._context
    if context is None:
        context = torch.zeros(B, d, device=h.device, dtype=param_dtype)
    ctx_exp = context.unsqueeze(1).expand(-1, N, -1)
    metric_input = torch.cat([h_normed, ctx_exp], dim=-1)
    hidden = F.gelu(dynamics.metric_net_linear1(metric_input))
    g = F.softplus(dynamics.metric_net_linear2_diag(hidden))

    # Sample pairs
    n_sample = min(n_pairs, N * (N - 1) // 2)
    idx_i = torch.randint(0, N, (n_sample,), device=h.device)
    idx_j = (idx_i + torch.randint(1, N, (n_sample,), device=h.device)) % N
    delta = h[0, idx_i, :] - h[0, idx_j, :]
    g_avg = (g[0, idx_i, :] + g[0, idx_j, :]) * 0.5
    D_sq = (delta * g_avg * delta).sum(dim=-1)

    return D_sq.median().item()


def compute_layer_entropy(bias_2d):
    """Compute attention entropy from bias matrix."""
    N = bias_2d.shape[0]
    if N < 2:
        return 0.0, 0.0
    K = torch.softmax(bias_2d, dim=-1)
    entropy = -(K * (K + 1e-10).log()).sum(dim=-1).mean().item()
    max_entropy = math.log(N)
    return entropy, entropy / max_entropy


def analyze_event_bias(bias_2d, token_texts):
    """Analyze within-event vs cross-event bias.

    Uses simple heuristic: tokens from the same sentence (split by period)
    are "within-event", tokens across sentences are "cross-event".
    """
    N = bias_2d.shape[0]
    if N < 4:
        return 0.0, 0.0

    # Assign event IDs based on sentence boundaries
    event_ids = []
    current_event = 0
    full_text = ''.join(token_texts)
    char_pos = 0
    for t in token_texts:
        event_ids.append(current_event)
        char_pos += len(t)
        if '.' in t:
            current_event += 1

    if len(set(event_ids)) < 2:
        return 0.0, 0.0

    within_vals = []
    across_vals = []
    n_sample = min(500, N * N)
    for _ in range(n_sample):
        i = torch.randint(0, N, (1,)).item()
        j = torch.randint(0, N, (1,)).item()
        if i == j:
            continue
        val = bias_2d[i, j].item()
        if event_ids[i] == event_ids[j]:
            within_vals.append(val)
        else:
            across_vals.append(val)

    B_within = sum(within_vals) / max(len(within_vals), 1)
    B_across = sum(across_vals) / max(len(across_vals), 1)
    return B_within, B_across


class DiagnosticLayerWiseODE:
    """Extended LayerWiseODE that captures full diagnostic data per layer.

    Wraps the base LayerWiseODE to also compute D², entropy, and event bias
    at each layer. Slower but gives complete picture.
    """

    def __init__(self, base_ode, tokenizer, input_text):
        self.base = base_ode
        self.tokenizer = tokenizer
        self.input_text = input_text
        self._token_texts = None
        self.full_diagnostics = []

    def wrap_process_layer(self, orig_process_layer):
        """Wrap process_layer to add extra diagnostics."""
        base = self.base

        def extended_process_layer(layer_idx, h_residual):
            result = orig_process_layer(layer_idx, h_residual)

            # Build corrected representation (perturbation architecture)
            h_proj = base.proj_in(h_residual) if base.proj_in is not None else h_residual
            h_corrected = h_proj + base.epsilon * base.correction

            with torch.no_grad():
                D_sq = compute_layer_D_sq(base.dynamics, h_corrected)

                # Get the bias we just computed
                bias_2d = base.layer_biases[-1] if base.layer_biases else None
                entropy, entropy_ratio = 0.0, 0.0
                B_within, B_across = 0.0, 0.0
                if bias_2d is not None:
                    entropy, entropy_ratio = compute_layer_entropy(bias_2d)
                    if self._token_texts:
                        B_within, B_across = analyze_event_bias(
                            bias_2d, self._token_texts)

            self.full_diagnostics.append({
                'layer_idx': layer_idx,
                'cv': base._layer_diagnostics[-1]['cv'],
                'tau_mean': base._layer_diagnostics[-1]['tau_mean'],
                'B_range': base._layer_diagnostics[-1]['B_range'],
                'B_std': base._layer_diagnostics[-1]['B_std'],
                'D_sq': D_sq,
                'entropy': entropy,
                'entropy_ratio': entropy_ratio,
                'B_within': B_within,
                'B_across': B_across,
            })

            return result

        return extended_process_layer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='/workspace/models/qwen3-4b')
    parser.add_argument('--ode_checkpoint', type=str,
                        default='/workspace/liquid-arc/output_2048/checkpoints/best.pt')
    parser.add_argument('--config', type=str,
                        default='/workspace/liquid-arc/configs/mind_layerwise.yaml')
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from liquid_arc.config import LiquidARCConfig
    from liquid_arc.dynamics import ContinuousDynamics
    from liquid_arc.context_pool import ContextPool
    from liquid_arc.layer_wise_ode import LayerWiseODE, LayerWiseBridge, hook_llm_layers

    print("=" * 70)
    print("LAYER-WISE ODE — Deep Analysis")
    print("=" * 70)

    # ── Load everything ──
    config = LiquidARCConfig.from_yaml(args.config)

    print("\nLoading Qwen3-4B...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map='cuda',
        torch_dtype=torch.bfloat16, trust_remote_code=True)
    llm.eval()
    n_layers = llm.config.num_hidden_layers
    d_llm = llm.config.hidden_size
    d_ode = config.d_model
    print(f"  {n_layers} layers, d_llm={d_llm}, d_ode={d_ode}")

    print("Loading ODE dynamics...")
    dynamics = ContinuousDynamics(config).to('cuda').to(torch.bfloat16)
    context_pool = ContextPool(config).to('cuda').to(torch.bfloat16)
    ckpt = torch.load(args.ode_checkpoint, map_location='cuda', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
    cleaned = {k.replace("_orig_mod.", "").replace('metric_net_linear2.', 'metric_net_linear2_diag.'): v
               for k, v in state_dict.items()}
    dyn_keys = {k: v for k, v in cleaned.items()
                if k.startswith('dynamics.') or k.startswith('context_pool.')}
    holder = nn.ModuleDict({'dynamics': dynamics, 'context_pool': context_pool})
    holder.load_state_dict(dyn_keys, strict=False)
    dynamics.eval()
    dynamics.freeze_tau = False
    print(f"  ODE step {ckpt.get('step', '?')}, {len(dyn_keys)} keys loaded")

    layer_ode = LayerWiseODE(
        dynamics=dynamics, context_pool=context_pool,
        n_layers=n_layers, d_llm=d_llm, d_ode=d_ode, device='cuda')

    # ════════════════════════════════════════════════════════════
    # PART 1: CAUSAL CHAIN COMPARISON
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART 1: CAUSAL CHAIN — ODE vs PLAIN")
    print("=" * 70)

    bridge = LayerWiseBridge(llm=llm, tokenizer=tokenizer, layer_ode=layer_ode)

    results_ode = []
    results_plain = []

    for test in CAUSAL_TESTS:
        print(f"\n  [{test['hops']}-hop] {test['name']}")

        # With ODE
        ode_result = bridge.generate(test['prompt'], max_new_tokens=100, temperature=0.3)
        ode_text = ode_result['response']
        ode_correct = score_response(ode_text, test['root_cause'])
        results_ode.append(ode_correct)

        # Without ODE
        layer_ode._active = False
        plain_text = generate_plain(llm, tokenizer, test['prompt'],
                                    max_new_tokens=100, temperature=0.3)
        plain_correct = score_response(plain_text, test['root_cause'])
        results_plain.append(plain_correct)
        layer_ode._active = True  # re-enable for next test

        ode_mark = "PASS" if ode_correct else "FAIL"
        plain_mark = "PASS" if plain_correct else "FAIL"
        print(f"    ODE:   [{ode_mark}] \"{ode_text[:120]}\"")
        print(f"    Plain: [{plain_mark}] \"{plain_text[:120]}\"")

    print(f"\n  ─── Summary ───")
    print(f"  ODE:   {sum(results_ode)}/{len(results_ode)} correct")
    print(f"  Plain: {sum(results_plain)}/{len(results_plain)} correct")

    by_hops_ode = {}
    by_hops_plain = {}
    for i, test in enumerate(CAUSAL_TESTS):
        h = test['hops']
        by_hops_ode.setdefault(h, []).append(results_ode[i])
        by_hops_plain.setdefault(h, []).append(results_plain[i])

    for hops in sorted(by_hops_ode.keys()):
        ode_score = sum(by_hops_ode[hops])
        plain_score = sum(by_hops_plain[hops])
        n = len(by_hops_ode[hops])
        print(f"  {hops}-hop: ODE={ode_score}/{n}  Plain={plain_score}/{n}")

    # ════════════════════════════════════════════════════════════
    # PART 2: PER-LAYER GEOMETRIC DIAGNOSTICS
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART 2: PER-LAYER GEOMETRIC DIAGNOSTICS")
    print("=" * 70)

    # Use the 3-hop causal chain as the diagnostic input
    test_text = (
        "A bridge collapsed on March 1st. Because of this, trucks were rerouted "
        "through downtown. The downtown rerouting caused supply disruptions at "
        "the warehouse."
    )
    inputs = tokenizer(test_text, return_tensors='pt').to('cuda')
    n_tokens = inputs['input_ids'].shape[1]
    token_texts = [tokenizer.decode([tid]) for tid in inputs['input_ids'][0].tolist()]

    print(f"\n  Input: {n_tokens} tokens")
    print(f"  Tokens: {' | '.join(token_texts[:10])} ...")

    # Set up diagnostic wrapper
    diag_wrapper = DiagnosticLayerWiseODE(layer_ode, tokenizer, test_text)
    diag_wrapper._token_texts = token_texts

    # Monkey-patch process_layer to add diagnostics
    orig_process = layer_ode.process_layer
    layer_ode.process_layer = diag_wrapper.wrap_process_layer(orig_process)

    # Remove old hooks, re-register with diagnostic version
    bridge.remove_hooks()
    hooks = hook_llm_layers(llm, layer_ode)

    layer_ode.start_forward()
    with torch.no_grad():
        _ = llm(**inputs)
    layer_ode.end_forward()

    # Restore original
    layer_ode.process_layer = orig_process
    for h in hooks:
        h.remove()

    # Also get correction_ratio from base diagnostics
    base_diags = layer_ode.get_layer_diagnostics()

    # Print full diagnostic table
    print(f"\n  {'Lyr':>3} {'CV':>6} {'D2_cor':>7} {'tau':>6} {'B_rng':>6} "
          f"{'H_rat':>6} {'Bw':>6} {'Bx':>6} {'c_rat':>6}")
    print(f"  {'-'*3} {'-'*6} {'-'*7} {'-'*6} {'-'*6} "
          f"{'-'*6} {'-'*6} {'-'*6} {'-'*6}")

    for i, d in enumerate(diag_wrapper.full_diagnostics):
        idx = d['layer_idx']
        c_rat = base_diags[i].get('correction_ratio', 0) if i < len(base_diags) else 0
        print(f"  {idx:>3} {d['cv']:>6.3f} {d['D_sq']:>7.1f} {d['tau_mean']:>6.3f} "
              f"{d['B_range']:>6.2f} {d['entropy_ratio']:>6.3f} "
              f"{d['B_within']:>6.3f} {d['B_across']:>6.3f} {c_rat:>6.3f}")

    # Aggregate by depth thirds
    all_d = diag_wrapper.full_diagnostics
    n = len(all_d)
    third = max(n // 3, 1)
    early = all_d[:third]
    mid = all_d[third:2*third]
    late = all_d[2*third:]

    def avg(lst, key):
        vals = [d.get(key, 0) for d in lst]
        return sum(vals) / max(len(vals), 1)

    # Correction ratio aggregates from base diags
    early_cr = [base_diags[i].get('correction_ratio', 0) for i in range(third)]
    mid_cr = [base_diags[i].get('correction_ratio', 0) for i in range(third, 2*third)]
    late_cr = [base_diags[i].get('correction_ratio', 0) for i in range(2*third, n) if i < len(base_diags)]

    print(f"\n  --- Depth Aggregates ---")
    print(f"  {'':>10} {'CV':>6} {'D2':>7} {'tau':>6} {'B_rng':>6} "
          f"{'H_rat':>6} {'Bw':>6} {'Bx':>6} {'c_rat':>6}")
    for label, group, cr in [('Early', early, early_cr), ('Mid', mid, mid_cr), ('Late', late, late_cr)]:
        cr_avg = sum(cr) / max(len(cr), 1)
        print(f"  {label:>10} {avg(group,'cv'):>6.3f} {avg(group,'D_sq'):>7.1f} "
              f"{avg(group,'tau_mean'):>6.3f} {avg(group,'B_range'):>6.2f} "
              f"{avg(group,'entropy_ratio'):>6.3f} {avg(group,'B_within'):>6.3f} "
              f"{avg(group,'B_across'):>6.3f} {cr_avg:>6.3f}")

    # Key questions
    print(f"\n  ─── Analysis ───")
    cv_trend = avg(late, 'cv') - avg(early, 'cv')
    dsq_trend = avg(late, 'D_sq') - avg(early, 'D_sq')
    b_across_trend = avg(late, 'B_across') - avg(early, 'B_across')
    entropy_trend = avg(late, 'entropy_ratio') - avg(early, 'entropy_ratio')

    print(f"  CV trend (late - early): {cv_trend:+.4f} "
          f"({'deepening' if cv_trend > 0.01 else 'flattening' if cv_trend < -0.01 else 'stable'})")
    print(f"  D² trend (late - early): {dsq_trend:+.1f} "
          f"({'separating' if dsq_trend > 1 else 'converging' if dsq_trend < -1 else 'stable'})")
    print(f"  B_across trend: {b_across_trend:+.3f} "
          f"({'strengthening' if b_across_trend > 0.01 else 'weakening' if b_across_trend < -0.01 else 'stable'})")
    print(f"  Entropy trend: {entropy_trend:+.3f} "
          f"({'more uniform' if entropy_trend > 0.01 else 'more structured' if entropy_trend < -0.01 else 'stable'})")

    # Cross-event separation
    mean_B_within = avg(all_d, 'B_within')
    mean_B_across = avg(all_d, 'B_across')
    if mean_B_within != 0:
        ratio = mean_B_across / mean_B_within
        print(f"  B_across/B_within ratio: {ratio:.3f} "
              f"({'cross > within = good' if ratio > 1 else 'within > cross = events not separated'})")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
