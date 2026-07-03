"""Experiment 5 — End-to-end LLM + LiquidARC graph engine integration.

Spec: GRAPH_REASONING_ENGINE_SPEC.md lines 514-524.

Tests three conditions on a causal-chain test suite:
  (A) Plain Qwen3-4B (no graph tools)
  (B) Qwen3-4B + LiquidARC graph engine (analyze_graph called on extracted graph)
  (C) Qwen3-4B + hand-written graph algorithms (networkx BFS/shortest_path)

Pipeline for (B) and (C):
  1. LLM gets question + context text
  2. We deterministically parse the context into a graph (nodes, edges)
  3. LLM generates a graph-tool call (simulated as a structured JSON query)
     OR we directly invoke the tool based on question type
  4. Tool (LiquidARC or networkx) returns a structured result
  5. Result is injected back into a new prompt
  6. LLM generates the final answer
  7. Score against ground truth

Pass criterion (spec line 532): "improvement over plain LLM on at least 2 of 3
tasks where plain LLM fails (5-hop chain, scope logic, cross-chain contamination)".

Run:
  python3 scripts/exp5_llm_integration.py \
    --checkpoint /workspace/liquid-arc/output_graph_engine_final/checkpoints/step_500.pt \
    --model /workspace/models/qwen3-4b \
    --out /workspace/liquid-arc/exp5_results.json
"""

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
import torch

from liquid_arc.graph_engine_inference import GraphEngine


# ─────────────────────────────────────────────────────────────────────
# Test suite — 3 failure-case tasks for plain LLM
# ─────────────────────────────────────────────────────────────────────

TESTS = [
    {
        'name': '5hop_root_cause',
        'type': 'root_cause',
        'context': (
            "A drought hit the farmlands in June. The drought killed the crops. "
            "Without crops, farmers lost income. With no income, families could not pay rent. "
            "Landlords evicted the tenants. The evicted tenants crowded into shelters. "
            "The shelters became overwhelmed and sanitation collapsed. "
            "Sanitation collapse led to a disease outbreak in the city."
        ),
        'question': "What was the root cause of the disease outbreak?",
        'ground_truth': 'drought',
        'plain_fail_keyword': 'crops',   # typical plain-LLM failure mode: stops at intermediate
        # Structured graph (8 nodes, 7 edges — linear chain of length 7)
        'graph': {
            'nodes': [
                {'id': 'drought', 'type': 0, 'role': 0},
                {'id': 'crops_killed', 'type': 1, 'role': 1},
                {'id': 'income_lost', 'type': 1, 'role': 1},
                {'id': 'rent_unpaid', 'type': 1, 'role': 1},
                {'id': 'eviction', 'type': 0, 'role': 1},
                {'id': 'shelter_crowd', 'type': 2, 'role': 1},
                {'id': 'sanitation_collapse', 'type': 2, 'role': 1},
                {'id': 'disease_outbreak', 'type': 0, 'role': 2},
            ],
            'edges': [
                {'src': 'drought', 'dst': 'crops_killed', 'type': 0},
                {'src': 'crops_killed', 'dst': 'income_lost', 'type': 0},
                {'src': 'income_lost', 'dst': 'rent_unpaid', 'type': 0},
                {'src': 'rent_unpaid', 'dst': 'eviction', 'type': 0},
                {'src': 'eviction', 'dst': 'shelter_crowd', 'type': 0},
                {'src': 'shelter_crowd', 'dst': 'sanitation_collapse', 'type': 0},
                {'src': 'sanitation_collapse', 'dst': 'disease_outbreak', 'type': 0},
            ],
        },
        'query': {'type': 'root_cause', 'target': 'disease_outbreak'},
    },
    {
        'name': 'scope_logic',
        'type': 'implication_check',
        'context': (
            "For the senior_engineer role, obtaining the security certification requires "
            "passing the network exam and the crypto exam. "
            "For the junior_developer role, the same security certification requires "
            "the network exam and crypto exam. "
            "Under the senior_engineer scope, passing the crypto exam requires completing linear_algebra. "
            "Under the junior_developer scope, passing the crypto exam requires completing an alternative_pathway instead."
        ),
        'question': (
            "For a junior_developer, does passing the crypto exam imply "
            "that linear_algebra was completed?"
        ),
        'ground_truth': 'no',
        'plain_fail_keyword': 'yes',  # plain LLM tends to miss scope conditioning
        'graph': {
            'nodes': [
                {'id': 'senior_eng', 'type': 4, 'role': 3},
                {'id': 'junior_dev', 'type': 4, 'role': 3},
                {'id': 'security_cert', 'type': 5, 'role': 1},
                {'id': 'network_exam', 'type': 6, 'role': 1},
                {'id': 'crypto_exam', 'type': 6, 'role': 1},
                {'id': 'linear_algebra', 'type': 7, 'role': 2},
                {'id': 'alternative_pathway', 'type': 7, 'role': 2},
            ],
            'edges': [
                {'src': 'senior_eng', 'dst': 'security_cert', 'type': 1},
                {'src': 'junior_dev', 'dst': 'security_cert', 'type': 1},
                {'src': 'security_cert', 'dst': 'network_exam', 'type': 1},
                {'src': 'security_cert', 'dst': 'crypto_exam', 'type': 1},
                {'src': 'crypto_exam', 'dst': 'linear_algebra', 'type': 1, 'scope': 'senior_eng'},
                {'src': 'crypto_exam', 'dst': 'alternative_pathway', 'type': 1, 'scope': 'junior_dev'},
            ],
        },
        'query': {
            'type': 'implication_check',
            'premise': 'crypto_exam',
            'conclusion': 'linear_algebra',
            'context_scope': 'junior_dev',
        },
    },
    {
        'name': 'cross_chain_contamination',
        'type': 'connection_check',
        'context': (
            "A pesticide was sprayed on the bee hives. The bees died. Pollination collapsed. "
            "Fruit yields dropped. "
            "Separately, a hurricane hit the cannery. The cannery flooded. Inventory was destroyed. "
            "Export orders were cancelled."
        ),
        'question': "Was the pesticide responsible for the cancellation of export orders?",
        'ground_truth': 'no',
        'plain_fail_keyword': 'yes',  # plain LLM tends to stitch unrelated chains
        'graph': {
            'nodes': [
                {'id': 'pesticide', 'type': 3, 'role': 0},
                {'id': 'bees_died', 'type': 0, 'role': 1},
                {'id': 'pollination_collapse', 'type': 1, 'role': 1},
                {'id': 'fruit_yields_drop', 'type': 2, 'role': 2},
                {'id': 'hurricane', 'type': 0, 'role': 0},
                {'id': 'cannery_flood', 'type': 1, 'role': 1},
                {'id': 'inventory_lost', 'type': 2, 'role': 1},
                {'id': 'export_cancelled', 'type': 1, 'role': 2},
            ],
            'edges': [
                {'src': 'pesticide', 'dst': 'bees_died', 'type': 0},
                {'src': 'bees_died', 'dst': 'pollination_collapse', 'type': 0},
                {'src': 'pollination_collapse', 'dst': 'fruit_yields_drop', 'type': 0},
                {'src': 'hurricane', 'dst': 'cannery_flood', 'type': 0},
                {'src': 'cannery_flood', 'dst': 'inventory_lost', 'type': 0},
                {'src': 'inventory_lost', 'dst': 'export_cancelled', 'type': 0},
            ],
        },
        'query': {
            'type': 'connection_check',
            'src': 'pesticide',
            'dst': 'export_cancelled',
        },
    },
]


# ─────────────────────────────────────────────────────────────────────
# LLM plumbing
# ─────────────────────────────────────────────────────────────────────

def load_llm(model_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    llm = AutoModelForCausalLM.from_pretrained(
        model_path, device_map='cuda', torch_dtype=torch.bfloat16,
        trust_remote_code=True)
    llm.eval()
    for p in llm.parameters():
        p.requires_grad_(False)
    return llm, tok


def chat_format(tok, user_content):
    msgs = [{"role": "user", "content": user_content}]
    try:
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)


def generate(llm, tok, prompt, max_new=200):
    full = chat_format(tok, prompt)
    inp = tok(full, return_tensors='pt', truncation=True, max_length=4096).to('cuda')
    n = inp['input_ids'].shape[1]
    with torch.no_grad():
        out = llm.generate(**inp, max_new_tokens=max_new, do_sample=False,
                           pad_token_id=tok.pad_token_id)
    txt = tok.decode(out[0][n:], skip_special_tokens=True)
    txt = re.sub(r'</?think>', '', txt).strip()
    return txt


# ─────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────

def score_response(test: Dict[str, Any], response: str) -> Dict[str, Any]:
    r = response.lower()
    gt = test['ground_truth'].lower()
    fail = test.get('plain_fail_keyword', '').lower()

    if test['type'] == 'root_cause':
        # expect root_cause keyword ("drought") in response
        correct = gt in r
        wrong = fail in r and gt not in r
    elif test['type'] == 'implication_check':
        # ground truth "no" means the implication is INVALID
        has_no = any(kw in r for kw in ['no', 'invalid', 'does not', 'is not', "isn't", "doesn't"])
        has_yes = any(kw in r for kw in [' yes', 'implies', 'is valid', 'does imply'])
        # Simple heuristic
        if gt == 'no':
            correct = has_no and not has_yes
            wrong = has_yes and not has_no
        else:
            correct = has_yes and not has_no
            wrong = has_no and not has_yes
    elif test['type'] == 'connection_check':
        has_no = any(kw in r for kw in ['no', 'not responsible', 'unrelated', 'separate', 'different'])
        has_yes = any(kw in r for kw in [' yes', 'caused', 'is responsible', 'led to'])
        if gt == 'no':
            correct = has_no and not has_yes
            wrong = has_yes and not has_no
        else:
            correct = has_yes and not has_no
            wrong = has_no and not has_yes
    else:
        correct = gt in r
        wrong = False

    return {
        'correct': bool(correct),
        'wrong': bool(wrong),
        'response': response[:400],
    }


# ─────────────────────────────────────────────────────────────────────
# Condition drivers
# ─────────────────────────────────────────────────────────────────────

def condition_plain(llm, tok, test):
    prompt = f"{test['context']}\n\nQuestion: {test['question']}\nAnswer:"
    resp = generate(llm, tok, prompt)
    return score_response(test, resp)


def condition_liquidarc(llm, tok, engine: GraphEngine, test):
    # Tool call based on test type
    graph_json = json.dumps(test['graph'])
    query_json = json.dumps(test['query'])
    tool_result_str = engine.analyze_graph(graph_json, query_json)
    tool_result = json.loads(tool_result_str)

    # Format tool result as text for the LLM to read
    hint = _format_tool_hint(test, tool_result)
    prompt = (
        f"{test['context']}\n\n"
        f"Graph analysis (from LiquidARC geometric engine):\n{hint}\n\n"
        f"Question: {test['question']}\n"
        f"Answer based on the graph analysis:"
    )
    resp = generate(llm, tok, prompt)
    return score_response(test, resp), tool_result


def condition_handwritten(llm, tok, test):
    # Use networkx directly to compute the answer
    graph_json = json.dumps(test['graph'])
    query_json = json.dumps(test['query'])
    tool_result = _handwritten_solve(test)
    hint = _format_tool_hint(test, tool_result)
    prompt = (
        f"{test['context']}\n\n"
        f"Graph analysis (from hand-written algorithm):\n{hint}\n\n"
        f"Question: {test['question']}\n"
        f"Answer based on the graph analysis:"
    )
    resp = generate(llm, tok, prompt)
    return score_response(test, resp), tool_result


def _handwritten_solve(test):
    g = nx.DiGraph()
    for n in test['graph']['nodes']:
        g.add_node(n['id'])
    for e in test['graph']['edges']:
        attrs = {k: v for k, v in e.items() if k not in ('src', 'dst')}
        g.add_edge(e['src'], e['dst'], **attrs)
    q = test['query']
    t = q['type']
    if t == 'root_cause':
        # Find root that reaches target (longest path)
        target = q['target']
        roots = [n for n in g.nodes if g.in_degree(n) == 0]
        best = None
        best_len = -1
        for r in roots:
            try:
                p = nx.shortest_path(g, r, target)
                if len(p) > best_len:
                    best = r
                    best_len = len(p)
            except nx.NetworkXNoPath:
                continue
        return {'root_cause': best, 'hops': best_len - 1 if best_len > 0 else 0}
    if t == 'connection_check':
        connected = nx.has_path(g.to_undirected(), q['src'], q['dst'])
        return {'connected': bool(connected)}
    if t == 'implication_check':
        # Under the given scope: filter edges to those without scope attribute
        # or matching the active scope. Then check if conclusion is reachable
        # from premise.
        active = q['context_scope']
        gf = nx.DiGraph()
        gf.add_nodes_from(g.nodes)
        for u, v, data in g.edges(data=True):
            s = data.get('scope')
            if s is None or s == active:
                gf.add_edge(u, v)
        valid = nx.has_path(gf, q['premise'], q['conclusion'])
        return {'valid': bool(valid)}
    return {}


def _format_tool_hint(test, tool_result):
    t = test['query']['type']
    if t == 'root_cause':
        rc = tool_result.get('root_cause', 'unknown')
        hops = tool_result.get('hops', '?')
        return f"  root_cause: {rc}\n  hops: {hops}"
    if t == 'connection_check':
        conn = tool_result.get('connected',
                               tool_result.get('connected_head_prob', 'unknown'))
        return f"  connected: {conn}"
    if t == 'implication_check':
        valid = tool_result.get('valid', 'unknown')
        return f"  valid: {valid}"
    return json.dumps(tool_result)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--model', default='/workspace/models/qwen3-4b')
    p.add_argument('--out', default='/workspace/liquid-arc/exp5_results.json')
    p.add_argument('--test_module', default=None,
                   help='optional python module that exports TESTS_HARD; defaults to built-in TESTS')
    args = p.parse_args()

    tests = TESTS
    if args.test_module:
        import importlib
        mod = importlib.import_module(args.test_module)
        tests = getattr(mod, 'TESTS_HARD', tests)
        print(f"  using test suite from {args.test_module}: {len(tests)} tests")

    print("=" * 70)
    print("EXPERIMENT 5 — LLM + LiquidARC graph engine integration")
    print("=" * 70)

    print("loading LLM...")
    llm, tok = load_llm(args.model)
    print(f"  ok ({sum(p.numel() for p in llm.parameters()):,} params, frozen)")

    print(f"loading GraphEngine from {args.checkpoint}...")
    engine = GraphEngine(args.checkpoint, device='cuda')
    print(f"  ok")

    results = []
    for test in tests:
        print(f"\n--- {test['name']} ({test['type']}) ---")

        # A) Plain
        t0 = time.time()
        a = condition_plain(llm, tok, test)
        a['wall_ms'] = int(1000 * (time.time() - t0))

        # B) LiquidARC graph engine
        t0 = time.time()
        b, b_tool = condition_liquidarc(llm, tok, engine, test)
        b['wall_ms'] = int(1000 * (time.time() - t0))
        b['tool_result'] = b_tool

        # C) Hand-written
        t0 = time.time()
        c, c_tool = condition_handwritten(llm, tok, test)
        c['wall_ms'] = int(1000 * (time.time() - t0))
        c['tool_result'] = c_tool

        record = {
            'test': test['name'],
            'type': test['type'],
            'ground_truth': test['ground_truth'],
            'plain': a,
            'liquidarc': b,
            'handwritten': c,
        }
        results.append(record)

        def _verdict(r):
            if r['correct']:
                return 'CORRECT'
            if r['wrong']:
                return 'WRONG (default failure mode)'
            return 'UNCLEAR'
        print(f"  A) Plain          : {_verdict(a)}  [{a['wall_ms']}ms]")
        print(f"  B) +LiquidARC     : {_verdict(b)}  [{b['wall_ms']}ms]  tool: {b_tool}")
        print(f"  C) +Hand-written  : {_verdict(c)}  [{c['wall_ms']}ms]  tool: {c_tool}")

    # Aggregate
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    plain_correct = sum(1 for r in results if r['plain']['correct'])
    liquid_correct = sum(1 for r in results if r['liquidarc']['correct'])
    hand_correct = sum(1 for r in results if r['handwritten']['correct'])
    n = len(results)
    print(f"  Plain           : {plain_correct}/{n}")
    print(f"  + LiquidARC     : {liquid_correct}/{n}")
    print(f"  + Hand-written  : {hand_correct}/{n}")

    # Spec Criterion 5: improvement over plain LLM on at least 2 of 3
    improvements = sum(
        1 for r in results
        if r['liquidarc']['correct'] and not r['plain']['correct']
    )
    passes_crit5 = improvements >= 2
    print(f"\n  Improvements (LiquidARC over Plain): {improvements}/{n}")
    print(f"  Spec Criterion 5 (≥2 improvements): "
          f"{'PASS' if passes_crit5 else 'FAIL'}")

    summary = {
        'results': results,
        'plain_correct': plain_correct,
        'liquidarc_correct': liquid_correct,
        'handwritten_correct': hand_correct,
        'n_tests': n,
        'improvements': improvements,
        'criterion_5_pass': passes_crit5,
    }
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  → saved {args.out}")


if __name__ == '__main__':
    main()
