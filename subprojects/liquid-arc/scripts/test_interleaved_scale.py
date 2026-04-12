"""Scaled interleaved chain test — 15 tests focused on 3ch×4hop and 4ch×3hop
where ODE showed improvement. Tests checkpoint comparison + optimal epsilon.

Run (~10GB):
  export PYTHONPATH=/workspace/liquid-arc:/workspace/fgn-v3
  python3 -u scripts/test_interleaved_scale.py
"""

import argparse, re, torch, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TESTS = [
    # ── 3ch × 4hop (WHERE ODE HELPS) ──
    {'name': '3×4 oil→truck→mains', 'diff': '3×4',
     'prompt': "An earthquake cracked the foundation of the Millbrook Dam on Sunday. "
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
        "Question: What was the root cause of the broken water mains?",
     'correct': 'oil', 'wrong': 'earthquake'},

    {'name': '3×4 virus→records→overdose', 'diff': '3×4',
     'prompt': "A computer virus infected hospital networks across the state on Monday. "
        "A severe blizzard shut down highways in the northern region on Tuesday. "
        "A labor dispute closed the pharmaceutical distribution center on Wednesday. "
        "The virus corrupted patient medication databases at three hospitals. "
        "The blizzard stranded hundreds of drivers on blocked highways. "
        "The closed distribution center stopped all drug shipments to pharmacies. "
        "Corrupted databases caused nurses to administer incorrect drug dosages. "
        "Stranded drivers needed rescue by National Guard helicopters. "
        "Pharmacies ran out of common prescriptions within two days. "
        "The incorrect dosages led to twelve patients experiencing adverse reactions. "
        "Rescue operations diverted military resources from scheduled training. "
        "Patients with chronic conditions couldn't refill their medications. "
        "Question: What was the root cause of the adverse reactions in patients?",
     'correct': 'virus', 'wrong': 'blizzard'},

    {'name': '3×4 drought→crop→bakery', 'diff': '3×4',
     'prompt': "A prolonged drought dried up irrigation canals in the farming belt in June. "
        "Pirates hijacked a cargo ship carrying electronics off the coast in July. "
        "A mine collapse trapped workers and halted ore production in August. "
        "Without irrigation, wheat fields produced less than half the normal yield. "
        "The hijacked ship's cargo of smartphones was held for ransom. "
        "The halted ore production created a steel shortage at manufacturing plants. "
        "The wheat shortfall caused flour mills to raise prices by 200%. "
        "Smartphone retailers faced empty shelves and angry customers. "
        "Steel-dependent factories reduced production and laid off workers. "
        "With flour prices tripled, small bakeries could no longer afford ingredients and closed. "
        "Electronics stores pivoted to selling refurbished devices. "
        "Laid-off factory workers filed for unemployment benefits. "
        "Question: What was the root cause of the bakery closures?",
     'correct': 'drought', 'wrong': 'mine'},

    {'name': '3×4 hack→signals→pileup', 'diff': '3×4',
     'prompt': "Hackers compromised the city's traffic management system on Friday night. "
        "A chemical plant explosion sent toxic fumes over the industrial quarter on Saturday. "
        "A wildfire in the surrounding hills forced closure of the northern highway on Sunday. "
        "The compromised system set all traffic lights to green simultaneously. "
        "Toxic fumes triggered mandatory evacuation of industrial quarter residents. "
        "The highway closure redirected northern traffic through city streets. "
        "Simultaneous green lights caused a massive chain-reaction pileup on Main Boulevard. "
        "Evacuees from the industrial quarter flooded shelters in the central district. "
        "Redirected traffic combined with the pileup created citywide gridlock. "
        "Emergency responders couldn't reach the pileup victims due to gridlock. "
        "Shelter overcrowding led to sanitation problems. "
        "Several pileup victims died waiting for ambulances stuck in traffic. "
        "Question: What was the root cause of the deaths at the pileup?",
     'correct': 'hack', 'wrong': 'wildfire'},

    {'name': '3×4 storm→power→freezer', 'diff': '3×4',
     'prompt': "A massive storm damaged power transmission towers along the coast on Monday. "
        "Protesters blockaded the entrance to the city's main fuel depot on Tuesday. "
        "A software bug caused failures in the railway switching system on Wednesday. "
        "Damaged towers cut electricity to three coastal counties for five days. "
        "The fuel depot blockade prevented tanker trucks from loading gasoline. "
        "The switching failures caused trains to be routed to wrong destinations. "
        "Without electricity, cold storage facilities lost refrigeration. "
        "Gas stations ran dry as no fuel could be delivered. "
        "Cargo meant for the harbor ended up at inland terminals. "
        "Tons of frozen food and vaccines spoiled in the powerless cold storage. "
        "Commuters with no gas switched to public transit, overwhelming buses. "
        "Misrouted cargo created shipping delays and supply chain confusion. "
        "Question: What caused the spoiled vaccines in cold storage?",
     'correct': 'storm', 'wrong': 'protest'},

    # ── 4ch × 3hop ──
    {'name': '4×3 flood→shelter', 'diff': '4×3',
     'prompt': "A dam failure released floodwaters into the Green Valley on Monday. "
        "Hackers locked the city government's computer systems with ransomware on Tuesday. "
        "A toxic algae bloom contaminated the reservoir on Wednesday. "
        "Railroad workers went on strike shutting down all freight service on Thursday. "
        "Floodwaters destroyed homes and forced thousands to flee to emergency shelters. "
        "The ransomware attack froze all government services including permit processing. "
        "The contaminated reservoir made tap water unsafe to drink. "
        "The rail strike stopped coal deliveries to the region's power plants. "
        "Emergency shelters ran out of beds and began turning people away. "
        "Citizens couldn't obtain building permits to start flood repairs. "
        "Residents had to buy bottled water, causing store shelves to empty. "
        "Power plants running low on coal began implementing rolling blackouts. "
        "Question: What caused the rolling blackouts?",
     'correct': 'strike', 'wrong': 'flood'},

    {'name': '4×3 quake→bridge→commute', 'diff': '4×3',
     'prompt': "An earthquake damaged the central highway bridge on Monday. "
        "A cyberattack took down the region's cellular network on Tuesday. "
        "Heavy rains caused landslides blocking mountain passes on Wednesday. "
        "Workers at the water treatment plant walked off the job on Thursday. "
        "The damaged bridge was closed, eliminating the main river crossing. "
        "The network outage cut off phone and internet service for millions. "
        "Landslides blocked the only alternate route over the mountains. "
        "Without treatment plant staff, water quality deteriorated rapidly. "
        "Commuters faced four-hour detours with no bridge and no mountain route. "
        "Businesses lost revenue with no way to process digital payments. "
        "Emergency services couldn't coordinate rescues without cell service. "
        "A boil-water advisory was issued across three districts. "
        "Question: What caused the boil-water advisory?",
     'correct': 'walk', 'wrong': 'earthquake'},

    {'name': '4×3 fire→school', 'diff': '4×3',
     'prompt': "A factory fire released asbestos particles into the air on Monday. "
        "A burst water main flooded the subway system on Tuesday. "
        "Vandals cut fiber optic cables disrupting internet across downtown on Wednesday. "
        "A flu outbreak sickened half the nursing staff at the county hospital on Thursday. "
        "Airborne asbestos forced closure of all schools within a mile radius. "
        "The flooded subway halted all underground transit service. "
        "The internet outage disabled online banking and remote work. "
        "The nursing shortage forced the hospital to close its emergency room. "
        "Students displaced from closed schools were transferred to overcrowded facilities. "
        "Commuters without subway service overwhelmed the bus system. "
        "Businesses downtown lost millions in productivity without internet. "
        "Patients needing emergency care had to travel to hospitals in other counties. "
        "Question: What caused the school closures?",
     'correct': 'fire', 'wrong': 'water main'},

    # ── 3ch × 5hop (hardest — deep chain through heavy interference) ──
    {'name': '3×5 spill→river→fish→market→restaurant', 'diff': '3×5',
     'prompt': "A chemical spill at the Maxwell factory contaminated the Clearwater River on Monday. "
        "A massive snowstorm buried the northern highway under six feet of snow on Tuesday. "
        "Union workers at the steel mill began an indefinite strike on Wednesday. "
        "The river contamination killed aquatic life for thirty miles downstream. "
        "The buried highway cut off the only road to three mountain towns. "
        "The steel strike halted production at the region's largest employer. "
        "Dead fish washed up on riverbanks, devastating the local fishing industry. "
        "Mountain town residents ran low on food and heating fuel. "
        "Thousands of laid-off steelworkers filed for unemployment. "
        "With no local catch, the fish market had to import from distant suppliers at triple cost. "
        "National Guard airlifted emergency supplies to snowbound towns. "
        "Unemployment offices were overwhelmed with claims. "
        "The expensive imported fish made seafood restaurants unprofitable, and five closed. "
        "Airlifted supplies were rationed to essential items only. "
        "The unemployment surge reduced consumer spending across the region. "
        "Question: What was the root cause of the seafood restaurant closures?",
     'correct': 'chemical', 'wrong': 'snow'},
]


def gen_plain(llm, tok, prompt, max_new=100, temp=0.1):
    msgs = [{"role": "system", "content": "Trace the causal chain back to the ROOT cause. Answer concisely."},
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
    if m and len(m.group(1).strip()) > 5: txt = m.group(1).strip()
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
    parser.add_argument('--epsilon', type=float, default=0.05)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from liquid_arc.config import LiquidARCConfig
    from liquid_arc.dynamics import ContinuousDynamics
    from liquid_arc.context_pool import ContextPool
    from liquid_arc.layer_wise_ode import LayerWiseODE, LayerWiseBridge

    print("=" * 70)
    print(f"INTERLEAVED SCALE TEST — eps={args.epsilon}")
    print("=" * 70)

    config = LiquidARCConfig.from_yaml(args.config)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm = AutoModelForCausalLM.from_pretrained(args.model_path, device_map='cuda',
        torch_dtype=torch.bfloat16, trust_remote_code=True)
    llm.eval()
    d = llm.config.hidden_size

    dynamics = ContinuousDynamics(config).to('cuda').to(torch.bfloat16).eval()
    context_pool = ContextPool(config).to('cuda').to(torch.bfloat16).eval()
    ckpt = torch.load(args.ode_checkpoint, map_location='cuda', weights_only=False)
    dynamics.load_state_dict(ckpt['dynamics_state'])
    context_pool.load_state_dict(ckpt['context_pool_state'])
    dynamics.freeze_tau = False
    print(f"  GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB, CV={ckpt.get('cv','?')}")

    layer_ode = LayerWiseODE(dynamics=dynamics, context_pool=context_pool,
        n_layers=llm.config.num_hidden_layers, d_llm=d, d_ode=config.d_model,
        epsilon=args.epsilon, device='cuda')
    bridge = LayerWiseBridge(llm=llm, tokenizer=tok, layer_ode=layer_ode, mode='attention')

    diffs = sorted(set(t['diff'] for t in TESTS))
    totals = {'plain': 0, 'ode': 0, 'n': 0}
    by_diff = {d: {'plain': 0, 'ode': 0, 'n': 0} for d in diffs}

    for t in TESTS:
        # Plain
        p_resp, ntok = gen_plain(llm, tok, t['prompt'])
        ps, pl = score(p_resp, t['correct'], t['wrong'])

        # ODE
        o_result = bridge.generate(t['prompt'], max_new_tokens=100, temperature=0.1)
        o_resp = o_result['response']
        os_, ol = score(o_resp, t['correct'], t['wrong'])

        diff_mark = ""
        if os_ > ps: diff_mark = " >>> IMPROVED"
        elif os_ < ps: diff_mark = " >>> DEGRADED"

        print(f"\n  [{t['diff']}] {t['name']} ({ntok}tok)")
        print(f"    Plain [{pl:>7}]: \"{p_resp[:120]}\"")
        print(f"    ODE   [{ol:>7}]: \"{o_resp[:120]}\"{diff_mark}")

        totals['plain'] += max(ps, 0)
        totals['ode'] += max(os_, 0)
        totals['n'] += 1
        by_diff[t['diff']]['plain'] += max(ps, 0)
        by_diff[t['diff']]['ode'] += max(os_, 0)
        by_diff[t['diff']]['n'] += 1

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"  {'Diff':>5} {'Plain':>8} {'ODE':>8} {'Delta':>6}")
    for d in diffs:
        bd = by_diff[d]
        delta = bd['ode'] - bd['plain']
        print(f"  {d:>5} {bd['plain']:>4}/{bd['n']} {bd['ode']:>4}/{bd['n']} {delta:>+4}")
    delta = totals['ode'] - totals['plain']
    print(f"  {'TOTAL':>5} {totals['plain']:>4}/{totals['n']} {totals['ode']:>4}/{totals['n']} {delta:>+4}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
