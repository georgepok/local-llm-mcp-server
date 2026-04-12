"""Test tasks where attention ROUTING is the bottleneck, not reasoning depth.

The causal chain test showed no ODE improvement because Qwen3-4B's reasoning
capacity (not attention routing) limits the 5-hop chain. These tests target
scenarios where flat attention causes wrong answers:

1. Entity scope confusion: Two entities with same name, different contexts.
   Flat attention bleeds attributes across. Geometric routing should separate.

2. Distractor interference: Answer buried in text, misleading content later.
   Flat attention weights both. Geometric bias suppresses distractor.

3. Parallel chains: Multiple independent causal sequences interleaved.
   Flat attention confuses chains. Geometric routing maintains separation.

Run in fgn-train container:
  export PYTHONPATH=/workspace/liquid-arc:/workspace/fgn-v3
  python3 -u scripts/test_routing_tasks.py
"""

import argparse
import random
import re
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════════
# TASK 1: ENTITY SCOPE CONFUSION
# ═══════════════════════════════════════════════════════════════

ENTITY_TESTS = [
    {
        'name': 'Two Jacks',
        'prompt': (
            "Jack Miller is a 45-year-old engineer from Boston who specializes in bridge design. "
            "He has worked on over 30 bridge projects across New England. His most recent project "
            "is the renovation of the Harbor Bridge. "
            "Jack Thompson is a 28-year-old chef from Portland who runs a seafood restaurant. "
            "He won the Regional Chef Award last year. His restaurant is known for its lobster dishes. "
            "Question: What is Jack's profession who won an award?"
        ),
        'correct': 'chef',
        'wrong': 'engineer',
    },
    {
        'name': 'Two Sarahs',
        'prompt': (
            "Sarah Chen works at NASA as a mission controller. She has been involved in three "
            "Mars rover missions and holds a PhD in aerospace engineering from MIT. "
            "Sarah Patel is a kindergarten teacher at Sunnyvale Elementary. She has been teaching "
            "for 15 years and recently published a children's book about space exploration. "
            "Question: Who published a book?"
        ),
        'correct': 'patel',
        'wrong': 'chen',
    },
    {
        'name': 'Two Riverside Hotels',
        'prompt': (
            "The Riverside Hotel in Chicago was built in 1920 and has 200 rooms. It underwent "
            "a major renovation in 2019 costing $50 million. The hotel is famous for its rooftop bar. "
            "The Riverside Hotel in Denver was established in 2015 and has 85 rooms. It is a boutique "
            "hotel known for its mountain views. It was rated the best new hotel in Colorado in 2016. "
            "Question: How many rooms does the newer Riverside Hotel have?"
        ),
        'correct': '85',
        'wrong': '200',
    },
]

# ═══════════════════════════════════════════════════════════════
# TASK 2: DISTRACTOR INTERFERENCE
# ═══════════════════════════════════════════════════════════════

DISTRACTOR_TESTS = [
    {
        'name': 'Population fact buried',
        'prompt': (
            "The city of Millbrook was founded in 1842 by settlers from Virginia. "
            "According to the 2020 census, Millbrook has a population of 34,500 residents. "
            "The city's economy is primarily based on agriculture and small manufacturing. "
            "Millbrook has three public high schools and one community college. "
            "The neighboring town of Ferndale, which was founded much later in 1910, "
            "has grown rapidly and now has a population of 89,200, making it the largest "
            "municipality in the county. Ferndale's growth has been driven by tech companies. "
            "Question: What is the population of Millbrook?"
        ),
        'correct': '34,500',
        'wrong': '89,200',
    },
    {
        'name': 'Date confusion',
        'prompt': (
            "The company was incorporated on March 15, 2005. Its first product launched "
            "in September 2006, and by 2008 it had reached 100,000 users. The company "
            "went through several rounds of funding. In 2012, a competing firm called TechNova "
            "was founded on June 3rd and quickly gained market share. TechNova's IPO on "
            "November 22, 2018 raised $2.3 billion, making headlines worldwide. "
            "Question: When was the original company incorporated?"
        ),
        'correct': 'march 15, 2005',
        'wrong': 'june 3',
    },
    {
        'name': 'Nested attribution',
        'prompt': (
            "Dr. Williams developed a new treatment for chronic pain using electrical stimulation. "
            "Her research, published in 2021, showed 73% improvement in patients over 6 months. "
            "The treatment uses targeted pulses at specific nerve clusters. "
            "Meanwhile, Dr. Garcia's team at Stanford published results of their unrelated study "
            "on sleep disorders, showing that blue light exposure reduced melatonin by 58%. "
            "Dr. Garcia recommended limiting screen time before bed. "
            "Question: What improvement percentage did Dr. Williams's pain treatment show?"
        ),
        'correct': '73',
        'wrong': '58',
    },
]

# ═══════════════════════════════════════════════════════════════
# TASK 3: PARALLEL CHAIN INTERFERENCE
# ═══════════════════════════════════════════════════════════════

PARALLEL_TESTS = [
    {
        'name': 'Two supply chains',
        'prompt': (
            "Chain A: A pesticide contamination was found in wheat fields in Kansas. "
            "The contaminated wheat was shipped to flour mills in Missouri. "
            "The flour was distributed to bakeries across the Midwest. "
            "Chain B: A drought hit rice paddies in California. "
            "The rice shortage caused prices to triple at Asian grocery stores. "
            "Several restaurant chains switched to importing rice from Thailand. "
            "Question: What caused the rice shortage?"
        ),
        'correct': 'drought',
        'wrong': 'pesticide',
    },
    {
        'name': 'Two investigations',
        'prompt': (
            "Case 1: Detective Rivera investigated a bank robbery on 5th Street. "
            "Fingerprints at the scene matched a suspect named Marcus Cole. "
            "Surveillance footage showed Cole entering the bank at 2:15 PM. "
            "Case 2: Detective Park investigated an art theft at the city museum. "
            "A forged security badge was found near the stolen painting. "
            "DNA evidence on the badge matched a known art forger named Elena Voss. "
            "Question: Who is the suspect in the art theft case?"
        ),
        'correct': 'elena voss',
        'wrong': 'marcus cole',
    },
    {
        'name': 'Three patients',
        'prompt': (
            "Patient A (Mr. Lee) was admitted with chest pain. His ECG showed irregular heartbeat. "
            "He was prescribed beta-blockers and scheduled for a stress test. "
            "Patient B (Ms. Davis) came in with a fractured wrist from a fall. "
            "X-rays confirmed a clean break. She was given a cast and pain medication. "
            "Patient C (Mr. Patel) reported persistent headaches for three weeks. "
            "An MRI revealed a small benign cyst. He was referred to a neurologist. "
            "Question: What did the MRI reveal for Mr. Patel?"
        ),
        'correct': 'cyst',
        'wrong': 'heartbeat',
    },
]


def generate_response(llm, tokenizer, prompt, max_new_tokens=60, temperature=0.1):
    """Generate with low temperature for deterministic comparison."""
    messages = [
        {"role": "system", "content": "Answer the question directly and concisely based only on the provided text."},
        {"role": "user", "content": prompt},
    ]
    try:
        full_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        full_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(full_prompt, return_tensors='pt', truncation=True, max_length=2048).to('cuda')
    with torch.no_grad():
        out = llm.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature,
                           do_sample=temperature > 0, top_p=0.9, repetition_penalty=1.2)
    text = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    match = re.search(r'</think>\s*(.*)', text, flags=re.DOTALL)
    if match and len(match.group(1).strip()) > 5:
        text = match.group(1).strip()
    return re.sub(r'</?think>', '', text).strip()


def score(response, correct, wrong):
    """Score: +1 if correct present, -1 if wrong present without correct, 0 otherwise."""
    r = response.lower()
    has_correct = correct.lower() in r
    has_wrong = wrong.lower() in r
    if has_correct and not has_wrong:
        return 1, 'CORRECT'
    if has_wrong and not has_correct:
        return -1, 'WRONG'
    if has_correct and has_wrong:
        return 0, 'BOTH'
    return 0, 'NEITHER'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/workspace/models/qwen3-4b')
    parser.add_argument('--ode_checkpoint', type=str,
                        default='/workspace/liquid-arc/output_layerwise/checkpoints/step_500.pt')
    parser.add_argument('--config', type=str, default='/workspace/liquid-arc/configs/mind_layerwise.yaml')
    parser.add_argument('--epsilon', type=float, default=0.5)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from liquid_arc.config import LiquidARCConfig
    from liquid_arc.dynamics import ContinuousDynamics
    from liquid_arc.context_pool import ContextPool
    from liquid_arc.layer_wise_ode import LayerWiseODE, LayerWiseBridge

    print("=" * 70)
    print("ROUTING TASKS — Where Attention Routing Matters")
    print("=" * 70)

    config = LiquidARCConfig.from_yaml(args.config)
    print(f"\nLoading Qwen3-4B...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map='cuda', torch_dtype=torch.bfloat16, trust_remote_code=True)
    llm.eval()
    n_layers = llm.config.num_hidden_layers
    d_llm = llm.config.hidden_size
    d_ode = config.d_model

    print("Loading ODE dynamics...")
    dynamics = ContinuousDynamics(config).to('cuda').to(torch.bfloat16)
    context_pool = ContextPool(config).to('cuda').to(torch.bfloat16)
    ckpt = torch.load(args.ode_checkpoint, map_location='cuda', weights_only=False)
    if 'dynamics_state' in ckpt:
        dynamics.load_state_dict(ckpt['dynamics_state'])
        context_pool.load_state_dict(ckpt['context_pool_state'])
    else:
        sd = ckpt.get('model_state_dict', ckpt)
        cleaned = {k.replace("_orig_mod.", "").replace('metric_net_linear2.', 'metric_net_linear2_diag.'): v
                   for k, v in sd.items()}
        holder = nn.ModuleDict({'dynamics': dynamics, 'context_pool': context_pool})
        holder.load_state_dict({k: v for k, v in cleaned.items()
                                if k.startswith('dynamics.') or k.startswith('context_pool.')}, strict=False)
    dynamics.eval()
    dynamics.freeze_tau = False
    print(f"  step {ckpt.get('step', '?')}, CV={ckpt.get('cv', '?')}, eps={args.epsilon}")

    layer_ode = LayerWiseODE(
        dynamics=dynamics, context_pool=context_pool,
        n_layers=n_layers, d_llm=d_llm, d_ode=d_ode,
        epsilon=args.epsilon, device='cuda')
    bridge = LayerWiseBridge(llm=llm, tokenizer=tokenizer, layer_ode=layer_ode)

    all_tasks = [
        ("ENTITY SCOPE CONFUSION", ENTITY_TESTS),
        ("DISTRACTOR INTERFERENCE", DISTRACTOR_TESTS),
        ("PARALLEL CHAIN INTERFERENCE", PARALLEL_TESTS),
    ]

    total_plain = {'correct': 0, 'wrong': 0, 'n': 0}
    total_ode = {'correct': 0, 'wrong': 0, 'n': 0}

    for task_name, tests in all_tasks:
        print(f"\n{'='*70}")
        print(f"  {task_name}")
        print(f"{'='*70}")

        task_plain = []
        task_ode = []

        for test in tests:
            print(f"\n  [{test['name']}]")

            # Plain
            plain_resp = generate_response(llm, tokenizer, test['prompt'])
            p_score, p_label = score(plain_resp, test['correct'], test['wrong'])
            task_plain.append(p_score)

            # ODE
            ode_result = bridge.generate(test['prompt'], max_new_tokens=60, temperature=0.1)
            ode_resp = ode_result['response']
            o_score, o_label = score(ode_resp, test['correct'], test['wrong'])
            task_ode.append(o_score)

            print(f"    Plain [{p_label:>7}]: \"{plain_resp[:120]}\"")
            print(f"    ODE   [{o_label:>7}]: \"{ode_resp[:120]}\"")
            if p_score != o_score:
                delta = "IMPROVED" if o_score > p_score else "DEGRADED"
                print(f"    >>> {delta}")

        p_correct = sum(1 for s in task_plain if s == 1)
        o_correct = sum(1 for s in task_ode if s == 1)
        p_wrong = sum(1 for s in task_plain if s == -1)
        o_wrong = sum(1 for s in task_ode if s == -1)
        print(f"\n  {task_name}: Plain={p_correct}/{len(tests)} correct, {p_wrong} wrong  |  "
              f"ODE={o_correct}/{len(tests)} correct, {o_wrong} wrong")

        total_plain['correct'] += p_correct
        total_plain['wrong'] += p_wrong
        total_plain['n'] += len(tests)
        total_ode['correct'] += o_correct
        total_ode['wrong'] += o_wrong
        total_ode['n'] += len(tests)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Plain: {total_plain['correct']}/{total_plain['n']} correct, "
          f"{total_plain['wrong']} wrong answers")
    print(f"  ODE:   {total_ode['correct']}/{total_ode['n']} correct, "
          f"{total_ode['wrong']} wrong answers")
    diff = total_ode['correct'] - total_plain['correct']
    print(f"  Delta: {diff:+d} correct answers")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
