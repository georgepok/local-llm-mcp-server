"""Deep interleaved chain testing — the task type where ODE showed improvement.

Tests multiple interleaved causal chains at increasing difficulty:
  - 3 chains × 3 hops (proven: +1 over plain)
  - 4 chains × 3 hops (harder: more cross-chain interference)
  - 3 chains × 4 hops (harder: deeper reasoning through interference)
  - 5 chains × 3 hops (hardest: maximum interference)

Also sweeps epsilon to find optimal coupling strength.

Run in fgn-train container (~10GB memory):
  export PYTHONPATH=/workspace/liquid-arc:/workspace/fgn-v3
  python3 -u scripts/test_interleaved_deep.py
"""

import argparse
import re
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Each test: interleaved chains where question targets ONE chain's root cause
TESTS = [
    # ── 3 chains × 3 hops (baseline — proven +1) ──
    {
        'name': '3ch×3hop: fire/flood/strike',
        'difficulty': '3×3',
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
            "Question: What was the root cause of the produce price increase at the central market?"
        ),
        'correct': 'rain',
        'wrong': 'fire',
    },
    {
        'name': '3ch×3hop: leak/storm/protest',
        'difficulty': '3×3',
        'prompt': (
            "A gas leak was detected at the Riverside chemical plant early Monday. "
            "A severe storm knocked out power lines across the eastern grid on Monday evening. "
            "Student protesters blocked the main entrance to the university campus on Tuesday. "
            "The gas leak forced evacuation of all residents within a two-mile radius. "
            "The power outage shut down traffic signals across the eastern district. "
            "The campus blockade prevented delivery trucks from reaching the university cafeteria. "
            "Evacuated families overwhelmed the Red Cross shelters in the western district. "
            "Without traffic signals, a major accident occurred at the Oak Street intersection. "
            "Without deliveries, the university cafeteria ran out of food by Wednesday afternoon. "
            "Question: What was the root cause of the accident at Oak Street?"
        ),
        'correct': 'storm',
        'wrong': 'leak',
    },
    # ── 4 chains × 3 hops ──
    {
        'name': '4ch×3hop: drought/hack/quake/spill',
        'difficulty': '4×3',
        'prompt': (
            "A drought hit farmlands in the central valley in January. "
            "Hackers breached the city's water treatment plant in February. "
            "A 5.2 earthquake damaged the northern highway bridge in March. "
            "A chemical spill contaminated the Silver River in April. "
            "The drought destroyed most of the season's wheat crop. "
            "The hack caused the water plant to release untreated water for two days. "
            "The bridge damage forced all northern commuters onto side roads. "
            "The river contamination killed fish populations downstream. "
            "Wheat shortage caused bread prices to triple in surrounding towns. "
            "Untreated water led to a health advisory and bottled water shortages. "
            "Side road congestion doubled commute times and caused frequent accidents. "
            "Fishing communities lost their primary income source for the season. "
            "Question: What caused the bottled water shortage?"
        ),
        'correct': 'hack',
        'wrong': 'drought',
    },
    {
        'name': '4ch×3hop: fire/flood/virus/strike',
        'difficulty': '4×3',
        'prompt': (
            "A warehouse fire destroyed medical supply stockpiles on Monday. "
            "Flooding from a broken levee submerged roads in the south district on Tuesday. "
            "A computer virus disabled the hospital's patient records system on Wednesday. "
            "Transit workers began an unannounced strike on Thursday morning. "
            "The fire left hospitals without backup ventilators and surgical masks. "
            "The flooding cut off ambulance routes to three neighborhoods. "
            "The virus made it impossible to access patient medication histories. "
            "The strike halted all city buses and subway trains. "
            "Hospitals had to postpone elective surgeries due to missing supplies. "
            "Residents in flooded areas couldn't reach emergency rooms. "
            "Doctors prescribed wrong medications because they couldn't check records. "
            "Thousands of workers couldn't get to their jobs across the city. "
            "Question: Why were wrong medications prescribed at the hospital?"
        ),
        'correct': 'virus',
        'wrong': 'fire',
    },
    # ── 3 chains × 4 hops (deeper chains) ──
    {
        'name': '3ch×4hop: quake/hack/spill',
        'difficulty': '3×4',
        'prompt': (
            "An earthquake cracked the foundation of the Millbrook Dam on Sunday. "
            "Hackers infiltrated the regional air traffic control system on Monday. "
            "A tanker truck overturned, spilling crude oil on Interstate 90 on Tuesday. "
            "Water seeped through the dam cracks, weakening the structure further. "
            "The hacked system displayed false aircraft positions to controllers. "
            "The oil spill closed all lanes of Interstate 90 for hazmat cleanup. "
            "Engineers determined the dam could fail within 48 hours and ordered evacuation. "
            "Two planes nearly collided when controllers gave wrong guidance. "
            "With I-90 closed, all freight traffic rerouted through residential streets. "
            "Three downstream towns were evacuated, displacing 15,000 people. "
            "All flights in the region were grounded pending system restoration. "
            "Heavy truck traffic on residential streets damaged roads and water mains. "
            "Evacuation shelters in neighboring counties ran out of space and supplies. "
            "Stranded passengers overwhelmed hotels and train stations. "
            "Broken water mains left entire neighborhoods without running water. "
            "Question: What was the root cause of the broken water mains?"
        ),
        'correct': 'oil',
        'wrong': 'earthquake',
    },
    # ── 5 chains × 3 hops (maximum interference) ──
    {
        'name': '5ch×3hop: fire/flood/hack/strike/quake',
        'difficulty': '5×3',
        'prompt': (
            "A fire erupted at the main power station on Monday. "
            "Flash floods hit the warehouse district on Tuesday. "
            "Hackers disabled the traffic light network on Wednesday. "
            "Dock workers went on strike at the cargo port on Thursday. "
            "A minor earthquake damaged gas lines in the north end on Friday. "
            "The power station fire caused rolling blackouts across the city. "
            "The floods destroyed inventory in twelve wholesale warehouses. "
            "The disabled traffic lights created gridlock on every major road. "
            "The dock strike prevented fuel tankers from offloading at the port. "
            "The damaged gas lines forced shutdown of heating in northern homes. "
            "Blackouts forced hospitals to run on backup generators for three days. "
            "Destroyed wholesale inventory left grocery stores with empty shelves. "
            "Gridlock prevented emergency vehicles from responding to calls. "
            "No fuel deliveries caused gas stations to run dry within two days. "
            "Northern residents had to evacuate to shelters with working heat. "
            "Question: Why did gas stations run dry?"
        ),
        'correct': 'strike',
        'wrong': 'fire',
    },
    {
        'name': '5ch×3hop: virus/storm/crash/protest/leak',
        'difficulty': '5×3',
        'prompt': (
            "A computer virus infected the banking network on Monday morning. "
            "A severe ice storm coated roads in the metro area Monday night. "
            "A freight train derailed blocking the main rail crossing on Tuesday. "
            "Protesters occupied the city hall entrance on Wednesday. "
            "A sewage pipe burst in the downtown business district on Thursday. "
            "The banking virus froze all ATM and card payment systems. "
            "The ice storm made roads impassable and closed schools. "
            "The derailment blocked rail service for the commuter line. "
            "The city hall protest prevented staff from issuing building permits. "
            "The sewage burst flooded three blocks with contaminated water. "
            "With no card payments, stores could only accept cash. "
            "Impassable roads stranded delivery trucks carrying perishable goods. "
            "Blocked rail service left 50,000 commuters without transportation. "
            "Frozen permits halted construction on the new hospital wing. "
            "Contaminated water forced closure of fifteen restaurants downtown. "
            "Question: What was the root cause of the restaurant closures?"
        ),
        'correct': 'sewage',
        'wrong': 'virus',
    },
]


def gen_plain(llm, tok, prompt, max_new=80, temp=0.1):
    msgs = [{"role": "system", "content": "Answer the question directly based only on the text. Be concise."},
            {"role": "user", "content": prompt}]
    try:
        full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(full, return_tensors='pt', truncation=True, max_length=4096).to('cuda')
    n = inp['input_ids'].shape[1]
    with torch.no_grad():
        out = llm.generate(**inp, max_new_tokens=max_new, temperature=temp,
                           do_sample=temp > 0, top_p=0.9, repetition_penalty=1.2)
    txt = tok.decode(out[0][n:], skip_special_tokens=True)
    m = re.search(r'</think>\s*(.*)', txt, flags=re.DOTALL)
    if m and len(m.group(1).strip()) > 5:
        txt = m.group(1).strip()
    return re.sub(r'</?think>', '', txt).strip(), n


def score(resp, correct, wrong):
    r = resp.lower()
    c = correct.lower() in r
    w = wrong.lower() in r
    if c and not w: return 1, 'CORRECT'
    if w and not c: return -1, 'WRONG'
    if c and w: return 0, 'BOTH'
    return 0, 'NEITHER'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='/workspace/models/qwen3-4b')
    parser.add_argument('--ode_checkpoint',
                        default='/workspace/liquid-arc/output_layerwise_v3/checkpoints/step_1500.pt')
    parser.add_argument('--config', default='/workspace/liquid-arc/configs/mind_layerwise.yaml')
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from liquid_arc.config import LiquidARCConfig
    from liquid_arc.dynamics import ContinuousDynamics
    from liquid_arc.context_pool import ContextPool
    from liquid_arc.layer_wise_ode import LayerWiseODE, LayerWiseBridge

    print("=" * 70)
    print("INTERLEAVED CHAIN DEEP TEST")
    print("=" * 70)

    config = LiquidARCConfig.from_yaml(args.config)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm = AutoModelForCausalLM.from_pretrained(args.model_path, device_map='cuda',
        torch_dtype=torch.bfloat16, trust_remote_code=True)
    llm.eval()
    n_layers = llm.config.num_hidden_layers
    d = llm.config.hidden_size

    dynamics = ContinuousDynamics(config).to('cuda').to(torch.bfloat16).eval()
    context_pool = ContextPool(config).to('cuda').to(torch.bfloat16).eval()
    ckpt = torch.load(args.ode_checkpoint, map_location='cuda', weights_only=False)
    dynamics.load_state_dict(ckpt['dynamics_state'])
    context_pool.load_state_dict(ckpt['context_pool_state'])
    dynamics.freeze_tau = False
    mem = torch.cuda.memory_allocated() / 1e9
    print(f"  Loaded: {n_layers} layers, d={d}, GPU={mem:.1f}GB")

    # ── Plain baseline ──
    print(f"\n{'='*70}")
    print("PLAIN BASELINE")
    print(f"{'='*70}")
    plain_results = {}
    for t in TESTS:
        resp, ntok = gen_plain(llm, tok, t['prompt'])
        s, label = score(resp, t['correct'], t['wrong'])
        plain_results[t['name']] = {'score': s, 'label': label, 'resp': resp, 'ntok': ntok}
        print(f"  [{label:>7}] {t['name']} ({ntok}tok): \"{resp[:100]}\"")

    # ── Epsilon sweep with residual injection ──
    epsilons = [0.02, 0.05, 0.1, 0.2]

    for eps in epsilons:
        print(f"\n{'='*70}")
        print(f"EPSILON = {eps}")
        print(f"{'='*70}")

        # Create fresh ODE for each epsilon
        layer_ode = LayerWiseODE(dynamics=dynamics, context_pool=context_pool,
            n_layers=n_layers, d_llm=d, d_ode=config.d_model,
            epsilon=eps, device='cuda')
        bridge = LayerWiseBridge(llm=llm, tokenizer=tok, layer_ode=layer_ode,
                                 mode='residual')

        for t in TESTS:
            result = bridge.generate(t['prompt'], max_new_tokens=80, temperature=0.1)
            resp = result['response']
            s, label = score(resp, t['correct'], t['wrong'])
            p = plain_results[t['name']]
            diff = ""
            if s > p['score']: diff = " >>> IMPROVED"
            elif s < p['score']: diff = " >>> DEGRADED"
            print(f"  [{label:>7}] {t['name']}: \"{resp[:100]}\"{diff}")

        bridge.remove_hooks()
        torch.cuda.empty_cache()

    # ── Summary table ──
    print(f"\n{'='*70}")
    print("SUMMARY BY DIFFICULTY")
    print(f"{'='*70}")

    difficulties = sorted(set(t['difficulty'] for t in TESTS))

    # Rerun to collect all scores (quick since model is loaded)
    all_scores = {'plain': {}}
    for t in TESTS:
        all_scores['plain'][t['name']] = plain_results[t['name']]['score']

    for eps in epsilons:
        all_scores[eps] = {}
        layer_ode = LayerWiseODE(dynamics=dynamics, context_pool=context_pool,
            n_layers=n_layers, d_llm=d, d_ode=config.d_model,
            epsilon=eps, device='cuda')
        bridge = LayerWiseBridge(llm=llm, tokenizer=tok, layer_ode=layer_ode,
                                 mode='residual')
        for t in TESTS:
            result = bridge.generate(t['prompt'], max_new_tokens=80, temperature=0.1)
            s, _ = score(result['response'], t['correct'], t['wrong'])
            all_scores[eps][t['name']] = s
        bridge.remove_hooks()
        torch.cuda.empty_cache()

    header = f"{'Difficulty':>10} {'Plain':>6}"
    for eps in epsilons:
        header += f" {'e='+str(eps):>7}"
    print(header)

    for diff in difficulties:
        tests_at_diff = [t for t in TESTS if t['difficulty'] == diff]
        row = f"{diff:>10}"
        p_correct = sum(1 for t in tests_at_diff if all_scores['plain'][t['name']] == 1)
        row += f" {p_correct:>3}/{len(tests_at_diff)}"
        for eps in epsilons:
            e_correct = sum(1 for t in tests_at_diff if all_scores[eps][t['name']] == 1)
            row += f"  {e_correct:>3}/{len(tests_at_diff)}"
        print(row)

    # Totals
    row = f"{'TOTAL':>10}"
    p_total = sum(1 for t in TESTS if all_scores['plain'][t['name']] == 1)
    row += f" {p_total:>3}/{len(TESTS)}"
    for eps in epsilons:
        e_total = sum(1 for t in TESTS if all_scores[eps][t['name']] == 1)
        row += f"  {e_total:>3}/{len(TESTS)}"
    print(row)
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
