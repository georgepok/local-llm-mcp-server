"""Hard routing tasks — long context where attention diffusion causes errors.

Short context (~200 tokens): Qwen3-4B handles everything easily.
Long context (1000+ tokens): attention becomes diffuse, routing matters.

Tests:
1. Buried fact with heavy padding: answer at position ~100, 800+ tokens of filler
2. Entity confusion at scale: 5+ entities with overlapping attributes
3. Interleaved parallel chains: 3 chains mixed sentence-by-sentence

Run in fgn-train container:
  export PYTHONPATH=/workspace/liquid-arc:/workspace/fgn-v3
  python3 -u scripts/test_hard_routing.py
"""

import argparse
import re
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


HARD_TESTS = [
    # ── Buried fact with heavy filler ──
    {
        'name': 'Buried population (1000+ tok)',
        'prompt': (
            "The city of Millbrook was founded in 1842. According to the 2020 census, "
            "Millbrook has a population of exactly 34,521 residents. "
            + " ".join([
                f"The city has various facilities including parks, schools, and commercial districts. "
                f"Infrastructure development continued throughout the decades with new roads and public buildings. "
                f"Community events are held regularly in the town square. "
                f"The local economy includes small businesses, retail shops, and service industries. "
            ] * 6) +  # ~600 tokens of filler
            "The neighboring city of Ferndale has grown to 89,200 residents. "
            "Ferndale recently surpassed Millbrook as the county's largest city. "
            "Ferndale's technology sector has attracted many new residents. "
            "The population boom in Ferndale started around 2015. "
            "Question: What is the exact population of Millbrook according to the 2020 census?"
        ),
        'correct': '34,521',
        'wrong': '89,200',
    },
    {
        'name': 'Buried date (1000+ tok)',
        'prompt': (
            "Project Alpha was initiated on February 14, 2019. The project aimed to develop "
            "a new water treatment system for the eastern district. "
            + " ".join([
                f"The project team included engineers from multiple departments working on design specifications. "
                f"Regular progress meetings were held to track milestones and address challenges. "
                f"Budget allocations were reviewed quarterly to ensure efficient resource utilization. "
                f"Technical documentation was maintained for each phase of the project. "
            ] * 6) +
            "A separate initiative, Project Beta, launched on September 3, 2021. "
            "Project Beta focused on renewable energy infrastructure. "
            "Project Beta received significant media attention due to its innovative approach. "
            "The September 2021 launch coincided with the annual sustainability summit. "
            "Question: When was Project Alpha initiated?"
        ),
        'correct': 'february 14, 2019',
        'wrong': 'september',
    },
    # ── 5 entities with overlapping attributes ──
    {
        'name': '5 doctors, who treats migraines',
        'prompt': (
            "Dr. Chen specializes in cardiology at St. Mary's Hospital. She graduated from Johns Hopkins "
            "in 2005 and has published 45 research papers on heart disease. She received the National "
            "Cardiology Award in 2018. "
            "Dr. Park is an orthopedic surgeon at General Hospital. He completed his residency at Mayo "
            "Clinic in 2008 and specializes in knee replacement surgery. He has performed over 2000 "
            "successful operations. "
            "Dr. Rivera practices neurology at University Medical Center. She completed her fellowship "
            "at Stanford in 2010 and treats patients with chronic migraines and epilepsy. She developed "
            "a new treatment protocol for cluster headaches. "
            "Dr. Thompson is a dermatologist with a private practice downtown. He graduated from "
            "Harvard Medical School in 2003 and focuses on skin cancer screening. He has treated "
            "over 10,000 patients in his career. "
            "Dr. Walsh works in emergency medicine at City Hospital. She has been an ER physician "
            "for 15 years and teaches at the medical school. She published a textbook on emergency "
            "trauma procedures. "
            "Question: Which doctor treats patients with chronic migraines?"
        ),
        'correct': 'rivera',
        'wrong': 'chen',
    },
    {
        'name': '5 companies, which one IPOd in 2020',
        'prompt': (
            "TechNova Inc was founded in 2015 in San Francisco. It develops cloud infrastructure "
            "tools and reached $50M ARR by 2019. The company raised Series C funding of $120M. "
            "DataStream Corp started in 2012 in Austin, Texas. It provides real-time analytics "
            "platforms for financial institutions. DataStream went public in 2018 with a $2B valuation. "
            "CloudPeak Systems was established in 2017 in Seattle. It offers serverless computing "
            "solutions and partnered with major telecom providers. CloudPeak completed its IPO in "
            "January 2020, raising $800M on the NASDAQ exchange. "
            "NetForge Labs began operations in 2014 in Boston. It builds cybersecurity tools for "
            "enterprise clients. NetForge was acquired by a larger firm in 2021 for $3.5B. "
            "QuantumBridge was founded in 2016 in Denver. It researches quantum computing "
            "applications for drug discovery. The company secured $200M in government grants. "
            "Question: Which company completed its IPO in 2020?"
        ),
        'correct': 'cloudpeak',
        'wrong': 'datastream',
    },
    # ── 3 interleaved chains ──
    {
        'name': '3 interleaved chains (fire/flood/strike)',
        'prompt': (
            "A factory fire broke out at the Henderson plant on Monday. "
            "Heavy rainfall began in the mountain region on Tuesday. "
            "Workers at the Port of Salem declared a strike on Wednesday. "
            "The factory fire destroyed the main assembly line, halting production of car parts. "
            "The rainfall caused mudslides that blocked Highway 12. "
            "The port strike stopped all container ships from being unloaded. "
            "Without car parts from Henderson, three auto dealerships ran out of inventory. "
            "With Highway 12 blocked, farm produce could not reach the central market. "
            "With the port closed, electronics retailers faced shortages of imported components. "
            "The auto dealerships had to turn away customers and lost significant revenue. "
            "Produce prices at the central market doubled due to limited supply. "
            "Electronics retailers started rationing popular items to one per customer. "
            "Question: What was the root cause of the produce price increase at the central market?"
        ),
        'correct': 'rain',
        'wrong': 'fire',
    },
    {
        'name': '3 interleaved chains (hack/quake/spill)',
        'prompt': (
            "Hackers infiltrated the city's traffic control system late Sunday night. "
            "A 4.5 magnitude earthquake struck the northern suburbs on Monday morning. "
            "A chemical spill occurred at the riverside industrial plant on Monday afternoon. "
            "The hacked traffic system caused all signals to malfunction, creating gridlock citywide. "
            "The earthquake damaged several water mains in the northern residential area. "
            "The chemical spill contaminated two miles of the river downstream. "
            "Emergency vehicles couldn't reach accident sites due to the traffic gridlock. "
            "Residents in the north had no running water for three days while mains were repaired. "
            "The fishing industry downstream lost an entire season's catch from the contamination. "
            "Several people with medical emergencies had delayed treatment due to blocked roads. "
            "Northern residents had to rely on bottled water deliveries from the National Guard. "
            "Fishing boats sat idle in the harbor and fishermen filed for disaster relief. "
            "Question: What caused the fishing industry to lose their catch?"
        ),
        'correct': 'chemical',
        'wrong': 'hack',
    },
]


def generate_response(model, tokenizer, prompt, max_new_tokens=60, temperature=0.1):
    messages = [
        {"role": "system", "content": "Answer the question directly based only on the provided text. Be concise."},
        {"role": "user", "content": prompt},
    ]
    try:
        full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(full, return_tensors='pt', truncation=True, max_length=4096).to('cuda')
    n_tok = inputs['input_ids'].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature,
                             do_sample=temperature > 0, top_p=0.9, repetition_penalty=1.2)
    text = tokenizer.decode(out[0][n_tok:], skip_special_tokens=True)
    match = re.search(r'</think>\s*(.*)', text, flags=re.DOTALL)
    if match and len(match.group(1).strip()) > 5:
        text = match.group(1).strip()
    return re.sub(r'</?think>', '', text).strip(), n_tok


def score(response, correct, wrong):
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
                        default='/workspace/liquid-arc/output_layerwise_v3/checkpoints/step_500.pt')
    parser.add_argument('--config', type=str, default='/workspace/liquid-arc/configs/mind_layerwise.yaml')
    parser.add_argument('--epsilon', type=float, default=0.5)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from liquid_arc.config import LiquidARCConfig
    from liquid_arc.dynamics import ContinuousDynamics
    from liquid_arc.context_pool import ContextPool
    from liquid_arc.layer_wise_ode import LayerWiseODE, LayerWiseBridge

    print("=" * 70)
    print("HARD ROUTING TASKS — Long Context + Heavy Interference")
    print("=" * 70)

    config = LiquidARCConfig.from_yaml(args.config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map='cuda', torch_dtype=torch.bfloat16, trust_remote_code=True)
    llm.eval()
    n_layers = llm.config.num_hidden_layers
    d_llm = llm.config.hidden_size
    d_ode = config.d_model

    dynamics = ContinuousDynamics(config).to('cuda').to(torch.bfloat16)
    context_pool = ContextPool(config).to('cuda').to(torch.bfloat16)
    ckpt = torch.load(args.ode_checkpoint, map_location='cuda', weights_only=False)
    if 'dynamics_state' in ckpt:
        dynamics.load_state_dict(ckpt['dynamics_state'])
        context_pool.load_state_dict(ckpt['context_pool_state'])
    dynamics.eval()
    dynamics.freeze_tau = False
    print(f"  CV={ckpt.get('cv', '?')}, eps={args.epsilon}")

    layer_ode = LayerWiseODE(
        dynamics=dynamics, context_pool=context_pool,
        n_layers=n_layers, d_llm=d_llm, d_ode=d_ode,
        epsilon=args.epsilon, device='cuda')
    bridge = LayerWiseBridge(llm=llm, tokenizer=tokenizer, layer_ode=layer_ode,
                            mode='residual')

    plain_total = 0
    ode_total = 0
    n_tests = 0

    for test in HARD_TESTS:
        print(f"\n  [{test['name']}]")

        plain_resp, n_tok = generate_response(llm, tokenizer, test['prompt'])
        p_score, p_label = score(plain_resp, test['correct'], test['wrong'])

        ode_result = bridge.generate(test['prompt'], max_new_tokens=60, temperature=0.1)
        ode_resp = ode_result['response']
        o_score, o_label = score(ode_resp, test['correct'], test['wrong'])

        print(f"    {n_tok} tokens")
        print(f"    Plain [{p_label:>7}]: \"{plain_resp[:120]}\"")
        print(f"    ODE   [{o_label:>7}]: \"{ode_resp[:120]}\"")
        if p_score != o_score:
            delta = ">>> IMPROVED" if o_score > p_score else ">>> DEGRADED"
            print(f"    {delta}")

        plain_total += max(p_score, 0)
        ode_total += max(o_score, 0)
        n_tests += 1

    print(f"\n{'='*70}")
    print(f"SUMMARY: Plain={plain_total}/{n_tests}  ODE={ode_total}/{n_tests}  "
          f"Delta={ode_total - plain_total:+d}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
