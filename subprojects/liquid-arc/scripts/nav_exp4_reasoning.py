"""Navigator Experiment 4 — end-to-end reasoning improvement.

Runs 30 problems across three conditions:
  A) Plain LLM (text only)
  B) LLM + geometric navigator (full pipeline: extract → state → hint)
  C) LLM + hand-written graph tools (networkx BFS/DFS on the fragment)

For the 10 cross-session-transfer problems, condition B additionally
processes a 'setup_fragment' first (so the pattern library has
something to match against in the actual query).

Answers are scored by LLM-as-judge to avoid keyword fragility.

Pass criteria (per spec):
  - B wins over A on ≥6 of 30 problems
  - B wins over C on ≥3 of 30 problems
  - 0 regressions (no case where B answers wrong and A answers right)
  - Cross-session: B solves ≥5/10 transfer problems, A solves <3/10

Usage:
    python -m liquid_arc.scripts.nav_exp4_reasoning \\
        --checkpoint /workspace/liquid-arc/output_graph_engine_final/checkpoints/step_500.pt \\
        --suite /workspace/liquid-arc/data/navigator/reasoning_suite.jsonl \\
        --vllm_url http://localhost:30000/v1 \\
        --model NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \\
        --out_json /workspace/liquid-arc/shared/outbox/nav_exp4_reasoning.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx

from liquid_arc.graph_engine_inference import GraphEngine
from liquid_arc.navigator import GeometricNavigator
from liquid_arc.navigator_extract import LLMExtractor
from liquid_arc.navigator_patterns import PatternLibrary
from liquid_arc.navigator_state import GeometricState


ANSWER_PROMPT = """\
You are helping answer a question about a described situation.

Situation:
{text}

{hint_section}
Question: {question}

Respond with a JSON object: {{"answer": "<answer>"}}.
The answer should be a single node id or short phrase. If the question \
asks for a root cause or yes/no verdict, answer accordingly.
Respond with ONLY the JSON object, no explanation."""


JUDGE_PROMPT = """\
You are grading an answer.

Question:      {question}
Expected:      {expected}
Model answer:  {answer}

Is the model answer correct? For yes/no questions, the answer must \
match the expected yes/no verdict. For root-cause questions, the \
answer must refer to the same underlying cause (the node id or a \
synonym thereof).

Respond with JSON: {{"correct": true}} or {{"correct": false}}. \
Respond with ONLY the JSON object."""


# ----------------------------------------------------------------------
# Condition C — hand-written networkx graph tools
# ----------------------------------------------------------------------


def networkx_hint(fragment: Dict[str, Any], _question: str) -> str:
    """Deterministic structural hint from networkx. Used in condition C."""
    g = nx.DiGraph()
    for n in fragment["nodes"]:
        g.add_node(n["id"], type=n.get("type", "entity"),
                   role=n.get("role", "intermediate"))
    for e in fragment["edges"]:
        g.add_edge(e["src"], e["dst"], type=e.get("type", "related_to"),
                   scope=e.get("scope"))

    lines = ["Structural analysis (networkx):"]
    terminals = [n for n, d in g.nodes(data=True) if d.get("role") == "terminal"]
    roots = [n for n, d in g.nodes(data=True) if d.get("role") == "root"]
    if terminals:
        for t in terminals:
            # Compute longest chain ending at t
            preds = []
            for r in roots:
                try:
                    path = nx.shortest_path(g, r, t)
                    preds.append((r, path))
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
            if preds:
                r, path = max(preds, key=lambda x: len(x[1]))
                lines.append(f"  path to {t}: {' → '.join(path)} "
                             f"(root={r}, hops={len(path) - 1})")
    # Scoped implication check
    scopes = [n for n, d in g.nodes(data=True) if d.get("role") == "scope"]
    for s in scopes:
        # edges that apply to this scope
        kept_edges = [(u, v) for u, v, d in g.edges(data=True)
                      if d.get("scope") in (None, s)]
        lines.append(f"  scope={s}, edges_in_scope={len(kept_edges)}")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# LLM wrappers
# ----------------------------------------------------------------------


def llm_answer(llm: LLMExtractor, text: str, question: str,
               hint: Optional[str]) -> str:
    hint_section = ""
    if hint:
        hint_section = hint + "\n\n"
    prompt = ANSWER_PROMPT.format(text=text, question=question,
                                  hint_section=hint_section)
    raw = llm.generate(prompt)
    try:
        obj = json.loads(raw.strip().strip("`"))
        return str(obj.get("answer", raw)).strip()
    except Exception:
        # fall back: return the raw text (judge can still read it)
        return raw.strip()


def llm_judge(llm: LLMExtractor, question: str, expected: str,
              answer: str) -> bool:
    prompt = JUDGE_PROMPT.format(question=question, expected=expected,
                                 answer=answer)
    raw = llm.generate(prompt)
    try:
        obj = json.loads(raw.strip().strip("`"))
        return bool(obj.get("correct", False))
    except Exception:
        # heuristic fallback — exact substring match
        return expected.lower().strip() in answer.lower()


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--suite", required=True)
    p.add_argument("--vllm_url", default="http://localhost:30000/v1")
    p.add_argument("--model", default="NVIDIA-Nemotron-3-Nano-30B-A3B-FP8")
    p.add_argument("--out_json", required=True)
    p.add_argument("--state_path",
                   default="/tmp/nav_exp4_state.json")
    p.add_argument("--pattern_path",
                   default="/tmp/nav_exp4_patterns.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--max_tokens", type=int, default=200)
    args = p.parse_args()

    # Fresh navigator state
    for path in (args.state_path, args.pattern_path):
        if os.path.exists(path):
            os.remove(path)

    engine = GraphEngine(args.checkpoint, device=args.device,
                         corrections_log=None)
    state = GeometricState(args.state_path, engine, max_nodes=512)
    patterns = PatternLibrary(args.pattern_path)
    llm = LLMExtractor(base_url=args.vllm_url, model=args.model,
                       temperature=args.temperature,
                       max_tokens=args.max_tokens)
    navigator = GeometricNavigator(engine=engine, state=state,
                                   extractor=None,
                                   pattern_library=patterns)

    cases: List[Dict[str, Any]] = []
    with open(args.suite) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    print(f"[exp4] suite: {len(cases)} problems", flush=True)

    per_case = []
    t0 = time.time()
    for i, case in enumerate(cases):
        cid = case["id"]
        cat = case["category"]
        text = case["text"]
        question = case["question"]
        expected = case["answer_key"]
        fragment = case["fragment"]

        # ── Condition A: plain LLM ─────────────────────────────────
        try:
            ans_a = llm_answer(llm, text, question, hint=None)
        except Exception as exc:
            ans_a = f"[error: {exc}]"
        correct_a = llm_judge(llm, question, expected, ans_a)

        # ── Condition B: LLM + navigator ──────────────────────────
        # Transfer flow: process setup (patterns learned + state seeded),
        # reset state only (preserve pattern library), then process question
        # in a clean state. This way the navigator's state reflects ONLY the
        # current problem while the pattern library still carries prior
        # structural knowledge from the setup session.
        if cat == "cross_session_transfer" and case.get("setup_fragment"):
            navigator.process_interaction(case.get("setup_text", ""),
                                          pre_extracted=case["setup_fragment"])
            state.reset()  # fresh state for the query; patterns survive
        nav_result = navigator.process_interaction(
            text, pre_extracted=fragment)
        nav_hint = nav_result.get("rendered_hint") or None
        try:
            ans_b = llm_answer(llm, text, question, hint=nav_hint)
        except Exception as exc:
            ans_b = f"[error: {exc}]"
        correct_b = llm_judge(llm, question, expected, ans_b)

        # ── Condition C: LLM + networkx hint ──────────────────────
        nx_hint = networkx_hint(fragment, _question=question)
        try:
            ans_c = llm_answer(llm, text, question, hint=nx_hint)
        except Exception as exc:
            ans_c = f"[error: {exc}]"
        correct_c = llm_judge(llm, question, expected, ans_c)

        per_case.append({
            "id": cid, "category": cat,
            "question": question, "expected": expected,
            "A_answer": ans_a, "A_correct": correct_a,
            "B_answer": ans_b, "B_correct": correct_b,
            "C_answer": ans_c, "C_correct": correct_c,
            "B_vs_A_win": correct_b and not correct_a,
            "B_vs_A_regression": correct_a and not correct_b,
            "B_vs_C_win": correct_b and not correct_c,
            "navigator_hint_used": nav_hint is not None,
            "pattern_match": bool(nav_result.get("pattern_match")),
        })
        print(f"  [{i+1:02d}/{len(cases)} {cid:5s} {cat:24s}] "
              f"A={'✓' if correct_a else '✗'} "
              f"B={'✓' if correct_b else '✗'} "
              f"C={'✓' if correct_c else '✗'} "
              f"(pattern={'Y' if per_case[-1]['pattern_match'] else 'N'})",
              flush=True)

        # Reset state between problems that are NOT in the same session
        # to prevent bleed-over. Transfer problems already ran setup above;
        # non-transfer problems just use state for this single problem.
        if cat != "cross_session_transfer":
            state.reset()

    total = time.time() - t0

    # Aggregates
    a_correct = sum(1 for c in per_case if c["A_correct"])
    b_correct = sum(1 for c in per_case if c["B_correct"])
    c_correct = sum(1 for c in per_case if c["C_correct"])

    b_wins_vs_a = sum(1 for c in per_case if c["B_vs_A_win"])
    regressions = sum(1 for c in per_case if c["B_vs_A_regression"])
    b_wins_vs_c = sum(1 for c in per_case if c["B_vs_C_win"])

    transfer = [c for c in per_case if c["category"] == "cross_session_transfer"]
    transfer_a = sum(1 for c in transfer if c["A_correct"])
    transfer_b = sum(1 for c in transfer if c["B_correct"])

    gates = {
        "B_wins_vs_A_ge_6": b_wins_vs_a >= 6,
        "B_wins_vs_C_ge_3": b_wins_vs_c >= 3,
        "zero_regressions": regressions == 0,
        "transfer_B_ge_5": transfer_b >= 5,
        "transfer_A_lt_3": transfer_a < 3,
    }
    overall_pass = all(gates.values())

    summary = {
        "suite": args.suite,
        "n_cases": len(cases),
        "n_correct": {"A": a_correct, "B": b_correct, "C": c_correct},
        "B_wins_vs_A": b_wins_vs_a,
        "regressions": regressions,
        "B_wins_vs_C": b_wins_vs_c,
        "transfer_A_correct": transfer_a,
        "transfer_B_correct": transfer_b,
        "per_case": per_case,
        "gates": gates,
        "overall_pass": overall_pass,
        "total_s": total,
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== EXPERIMENT 4 — SUMMARY ===", flush=True)
    print(f"  cases:               {len(cases)}", flush=True)
    print(f"  A correct:           {a_correct}/{len(cases)}", flush=True)
    print(f"  B correct:           {b_correct}/{len(cases)}", flush=True)
    print(f"  C correct:           {c_correct}/{len(cases)}", flush=True)
    print(f"  B wins vs A:         {b_wins_vs_a}  (target ≥6)", flush=True)
    print(f"  B wins vs C:         {b_wins_vs_c}  (target ≥3)", flush=True)
    print(f"  regressions (A>B):   {regressions}  (target 0)", flush=True)
    print(f"  transfer A correct:  {transfer_a}/10  (target <3)", flush=True)
    print(f"  transfer B correct:  {transfer_b}/10  (target ≥5)", flush=True)
    print(f"  overall:             {'PASS' if overall_pass else 'FAIL'}",
          flush=True)
    print(f"  wrote {out}", flush=True)
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
