"""SelfIntegrator v4 — reward-driven improvement loop.

Seeds from v3's per-function output (functions that pass harnesses but
may have integration bugs at scale). Then iterates:

  round r:
    1. Assemble the three functions into a single script.
    2. Execute end-to-end on full-scale dataset (100 training + 10 test).
    3. Compute a continuous reward from the RESULT_JSON:
         reward = 0.35 * signature_success_rate
                + 0.30 * accuracy
                + 0.20 * (1 - error_rate)
                + 0.15 * structural_completeness
    4. Identify which function most likely produced the errors
       (by parsing `errors[]` messages and matching against function
        responsibilities).
    5. Re-prompt ONLY that function with:
         - current reward
         - target reward
         - dominant error signature
         - the function's current code
    6. Replace the function; repeat until reward ≥ target or N rounds.

Stops when:
  - reward ≥ target_reward
  - reward hasn't improved for `patience` rounds (plateau)
  - max_rounds exhausted
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests


# ----------------------------------------------------------------------
# LLM + code execution helpers (kept compact; mirrors v3)
# ----------------------------------------------------------------------


_CODE_RX = re.compile(r"```(?:python)?\s*\n(.*?)```", flags=re.DOTALL)


def llm_generate(vllm_url: str, model: str, messages: List[Dict[str, str]],
                 max_tokens: int = 4000, temperature: float = 0.1) -> str:
    r = requests.post(
        f"{vllm_url.rstrip('/')}/chat/completions",
        json={"model": model, "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature,
               "chat_template_kwargs": {"enable_thinking": False}},
        timeout=240,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def extract_code(text: str) -> Optional[str]:
    m = _CODE_RX.search(text)
    if m:
        return m.group(1)
    if "```python" in text:
        body = text.split("```python", 1)[1]
        if "\n" in body:
            body = body.rsplit("\n", 1)[0] + "\n"
        return body
    return None


def run_code(code: str, *, timeout_s: int, env_extra: Dict[str, str],
             tmp_path: str = "/tmp/self_int_v4_run.py") -> Dict[str, Any]:
    with open(tmp_path, "w") as f:
        f.write(code)
    env = os.environ.copy()
    env.update(env_extra)
    env["PYTHONPATH"] = env_extra.get(
        "PYTHONPATH", "/workspace/liquid-arc")
    try:
        proc = subprocess.run(
            ["python", tmp_path], env=env,
            cwd=env_extra.get("CWD", "/workspace/liquid-arc"),
            capture_output=True, text=True, timeout=timeout_s,
        )
        return {"stdout": proc.stdout, "stderr": proc.stderr,
                "exit_code": proc.returncode, "timed_out": False}
    except subprocess.TimeoutExpired as exc:
        return {"stdout": exc.stdout or "",
                "stderr": (exc.stderr or "") + "\n[TIMEOUT]",
                "exit_code": -1, "timed_out": True}


# ----------------------------------------------------------------------
# Reward function
# ----------------------------------------------------------------------


def compute_reward(result_json: Optional[Dict[str, Any]],
                   exit_code: int) -> Tuple[float, Dict[str, float]]:
    """Continuous reward in [0, 1] with sub-component breakdown."""
    components = {
        "signature_success_rate": 0.0,
        "accuracy": 0.0,
        "low_error_rate": 0.0,
        "structural": 0.0,
    }
    if exit_code != 0 or result_json is None:
        return 0.0, components

    # Structural: did the run produce the expected top-level fields?
    expected = {"ingest", "accuracy", "per_case"}
    present = expected & set(result_json.keys())
    components["structural"] = len(present) / len(expected)

    ingest = result_json.get("ingest", {}) or {}
    n_ingested = ingest.get("n_ingested", 0) or 0
    n_signatures = ingest.get("n_signatures", 0) or 0
    if n_ingested > 0:
        components["signature_success_rate"] = min(1.0,
            n_signatures / n_ingested)

    errors = ingest.get("errors", []) or []
    if n_ingested > 0:
        components["low_error_rate"] = max(0.0, 1.0 - len(errors) / n_ingested)
    else:
        components["low_error_rate"] = 0.0

    acc = result_json.get("accuracy", 0.0) or 0.0
    components["accuracy"] = min(1.0, max(0.0, acc))

    reward = (0.35 * components["signature_success_rate"]
              + 0.30 * components["accuracy"]
              + 0.20 * components["low_error_rate"]
              + 0.15 * components["structural"])
    return reward, components


# ----------------------------------------------------------------------
# Error → function attribution
# ----------------------------------------------------------------------


FUNCTION_OWNERSHIP: List[Tuple[str, str]] = [
    # (substring pattern in error → responsible function)
    ("signature computation failed", "ingest_and_store"),
    ("'KnowledgeGraphDB' object has no attribute 'graph'", "ingest_and_store"),
    ("compute_signature", "ingest_and_store"),
    ("get_neighbors", "ingest_and_store"),
    ("extract_subgraph", "ingest_and_store"),
    ("add_fragment", "ingest_and_store"),
    ("signature", "ingest_and_store"),
    ("pattern", "ingest_and_store"),
    ("find_nearest", "evaluate_precedents"),
    ("evaluate", "evaluate_precedents"),
    ("expected_shape", "evaluate_precedents"),
    ("test case", "evaluate_precedents"),
    ("generate_cases", "generate_cases"),
    ("fragment", "generate_cases"),
    ("case_id", "generate_cases"),
    ("duplicate node", "generate_cases"),
]


def attribute_errors(result_json: Optional[Dict[str, Any]],
                     stderr: str,
                     components: Optional[Dict[str, float]] = None
                     ) -> Tuple[Optional[str], str]:
    """Return (culprit_function_name, summary_of_dominant_error).

    When components are provided and the dominant shortfall is accuracy
    (not errors), the accuracy gap is treated as evaluate_precedents'
    fault and a synthetic 'no errors but accuracy=0' summary is emitted.
    """
    components = components or {}
    error_strings: List[str] = []
    if result_json:
        ingest = result_json.get("ingest") or {}
        error_strings.extend(ingest.get("errors") or [])
        for c in (result_json.get("per_case") or []):
            if isinstance(c, dict) and c.get("error"):
                error_strings.append(str(c["error"]))
    if stderr:
        lines = stderr.splitlines()
        for i, line in enumerate(lines):
            if line.startswith("Traceback"):
                error_strings.append("\n".join(lines[i:])[-800:])

    # Low-accuracy attribution. Case 1: matched_label is consistently
    # null → evaluate_precedents isn't finding ANY matches. Case 2:
    # matches are found but point to wrong shapes → likely a data-quality
    # issue in generate_cases (shapes not structurally differentiated)
    # so the signatures collide.
    if components:
        sig_ok = components.get("signature_success_rate", 0) >= 0.9
        # Treat anything under 0.85 as "still worth improving" so the loop
        # doesn't prematurely halt at a local optimum.
        low_accuracy = components.get("accuracy", 0) < 0.85
        if sig_ok and low_accuracy and not error_strings:
            per_case = (result_json or {}).get("per_case") or []
            n_null = sum(1 for c in per_case
                          if isinstance(c, dict)
                             and c.get("matched_label") is None)
            n_match_but_wrong = sum(1 for c in per_case
                                     if isinstance(c, dict)
                                        and c.get("matched_label") is not None
                                        and c.get("matched_shape")
                                            != c.get("expected_shape"))
            if per_case and n_null > len(per_case) * 0.5:
                sample = json.dumps(per_case[0], default=str)[:500]
                return ("evaluate_precedents",
                        ("Signatures succeed but find_nearest returns None "
                         "for most test cases.\nSample record:\n  " + sample))
            if per_case and n_match_but_wrong >= 1:
                # Report a few mismatches — the fault is likely in
                # generate_cases making shapes look too similar.
                sample_lines = [
                    f"    expected={c.get('expected_shape')} matched={c.get('matched_shape')} cos={c.get('cosine')}"
                    for c in per_case[:6] if isinstance(c, dict)
                ]
                return ("generate_cases",
                        ("Pattern matching succeeds but matched shapes "
                         "don't align with expected shapes. The 6 shapes "
                         "in generate_cases produce topologically "
                         "indistinguishable signatures.\n"
                         "The shapes must differ STRUCTURALLY — different "
                         "chain length per shape, different branching, "
                         "different node types — so their metric "
                         "signatures are separable.\n"
                         "Observed (test_case → matches):\n"
                         + "\n".join(sample_lines)))
            # fallthrough: shouldn't hit, but attribute to evaluate anyway
            sample = json.dumps(per_case[0], default=str)[:500] if per_case else "(empty)"
            return ("evaluate_precedents",
                    "Low accuracy, unknown cause. Sample: " + sample)

    if not error_strings:
        return None, ""

    # Count which function each error attributes to
    attribution: Counter = Counter()
    for err in error_strings:
        for pattern, func in FUNCTION_OWNERSHIP:
            lower_err = err.lower() if isinstance(err, str) else str(err).lower()
            if pattern.lower() in lower_err or pattern in err:
                attribution[func] += 1
                break
    if not attribution:
        return None, error_strings[0] if error_strings else ""

    culprit = attribution.most_common(1)[0][0]
    sample = next((e for e in error_strings
                    if any(p.lower() in (e.lower() if isinstance(e, str) else "")
                            for p, f in FUNCTION_OWNERSHIP if f == culprit)),
                   error_strings[0])
    dominant = Counter(error_strings).most_common(3)
    summary_lines = [f"({count}x) {e[:200]}" for e, count in dominant]
    summary = (f"Dominant error pattern attributed to `{culprit}`:\n  "
               + sample[:500] + "\n\nTop-3 distinct errors:\n  "
               + "\n  ".join(summary_lines))
    return culprit, summary


# ----------------------------------------------------------------------
# Assembly + reward-driven loop
# ----------------------------------------------------------------------


ASSEMBLY_TEMPLATE = """\
# Auto-assembled by SelfIntegrator v4
import json
import os
import random
import math
from typing import List, Dict, Any, Set, Tuple, Optional


{functions}


if __name__ == '__main__':
    out_dir = os.environ.get('OUT_DIR', '/tmp/self_int_v4_out')
    os.makedirs(out_dir, exist_ok=True)
    checkpoint = os.environ['CHECKPOINT']
    try:
        train = generate_cases({n_train}, seed=42)
        test  = generate_cases({n_test}, seed=99)
        ingest_report = ingest_and_store(
            train,
            os.path.join(out_dir, 'db.json'),
            os.path.join(out_dir, 'patterns.json'),
            checkpoint)
        train_map = {{c['case_id']: c['shape'] for c in train}}
        eval_report = evaluate_precedents(
            test, os.path.join(out_dir, 'patterns.json'),
            checkpoint,
            os.path.join(out_dir, 'eval_scratch'),
            train_map)
        final = {{
            'ingest': ingest_report,
            'n_test_cases': len(test),
            'accuracy': eval_report.get('accuracy'),
            'per_case': eval_report.get('per_case'),
        }}
    except Exception as e:
        import traceback
        final = {{'fatal_error': str(e),
                  'traceback': traceback.format_exc()}}
    print('RESULT_JSON: ' + json.dumps(final, default=str))
"""


def assemble(functions: Dict[str, str], *, n_train: int = 100,
             n_test: int = 10) -> str:
    body = "\n\n".join(
        f"# --- {name} ---\n{code.strip()}"
        for name, code in functions.items()
    )
    return ASSEMBLY_TEMPLATE.format(
        functions=body, n_train=n_train, n_test=n_test)


GENERATE_CASES_EXTRA = textwrap.dedent("""\

    ═══ SHAPE-DIVERSITY (generate_cases ONLY) ═══

    The metric signature is topology-invariant. Two 4-node linear
    chains look identical to it regardless of node names. To make 6
    shapes separable, give each shape a DIFFERENT topology:

      shape 0 (domain A): linear chain, 3 nodes, edge type 'causes'
      shape 1 (domain A): linear chain, 5 nodes, edge type 'precedes'
      shape 2 (domain B): tree — 1 root with 2 children (3 nodes)
      shape 3 (domain B): diamond — 1 root → 2 middles → 1 terminal
      shape 4 (domain C): star — 1 root with 4 leaves
      shape 5 (domain C): linear chain, 6 nodes, edge type 'enables'

    Keep the same function signature and output shape. Preserve the
    original imports at the top of the function (from typing import
    List, Dict, Any).
""")


def build_improvement_prompt(function_name: str, current_code: str,
                              current_reward: float, components: Dict[str, float],
                              target_reward: float, error_summary: str
                              ) -> List[Dict[str, str]]:
    system = textwrap.dedent(f"""\
        You are improving a Python function to increase an end-to-end
        benchmark reward signal.

        Current reward: {current_reward:.3f} (target: {target_reward:.2f})

        Reward breakdown:
          signature success rate: {components.get('signature_success_rate', 0):.3f}
          accuracy:               {components.get('accuracy', 0):.3f}
          low error rate:         {components.get('low_error_rate', 0):.3f}
          structural:             {components.get('structural', 0):.3f}

        The component with lowest score is likely the bottleneck.

        Function to improve: `{function_name}`

        ═══ REFERENCE — correct API usage ═══

        `fragment["nodes"]` is a LIST of dicts, NOT a dict. To get node IDs:
            node_ids = [n["id"] for n in fragment["nodes"]]       # CORRECT
            # WRONG: fragment["nodes"].keys()   (list has no .keys)
            # WRONG: fragment["nodes"][id]       (list is not indexable by id)

        KnowledgeGraphDB graph attribute is `.G`:
            db.G.nodes                                            # CORRECT
            # WRONG: db._graph, db.graph

        Correct neighborhood + subgraph pattern:
            node_ids = [n["id"] for n in case["fragment"]["nodes"]]
            neighbors = db.get_neighbors(node_ids, hops=2, direction="both")
            subgraph  = db.extract_subgraph(list(neighbors) + node_ids,
                                             max_nodes=30)
            if len(subgraph["nodes"]) >= 2:
                sig = engine.compute_signature(subgraph)
                lib.store(sig, {{"label": case["case_id"]}})
                # NOTE: pass `sig` NOT `[sig]` — store expects the vector
                #       itself (a list of floats), not a list-of-lists.

        Correct pattern-matching call:
            match = lib.find_nearest(sig, threshold=0.5)
            # NOTE: pass `sig` (list of floats) NOT `[sig]`
            # match is a dict with keys 'label','similarity','count', or None
            if match:
                label = match["label"]
                cosine = match["similarity"]

        Instantiate SubgraphODEEngine ONCE outside the loop (expensive):
            engine = SubgraphODEEngine(checkpoint, device="cpu")
            for case in cases:
                ...
                sig = engine.compute_signature(subgraph)

        Do NOT call `db.clear()` at the end — downstream code needs the graph.

        ═══ OUTPUT RULES ═══
          - Output exactly ONE ```python``` fenced block
          - Redefine the full function with the same signature
          - Do NOT include main() or harness code
          - Apply the correct API usage above
          - Use simple try/except — every try MUST have an except block
    """)
    # Only include the shape-diversity guidance when the culprit IS the
    # data generator — otherwise it's noise that distracts the model.
    if function_name == "generate_cases":
        system = system + GENERATE_CASES_EXTRA
    user = (
        f"Errors observed end-to-end:\n{error_summary}\n\n"
        f"Current code for `{function_name}`:\n\n"
        f"```python\n{current_code}\n```\n\n"
        f"Return the improved full function body."
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def reward_loop(initial_functions: Dict[str, str], *,
                 vllm_url: str, model: str,
                 checkpoint: str, out_dir: Path,
                 target_reward: float = 0.90,
                 max_rounds: int = 8, patience: int = 3,
                 n_train: int = 100, n_test: int = 10,
                 ) -> Dict[str, Any]:
    funcs = dict(initial_functions)
    trace: List[Dict[str, Any]] = []
    best_reward = -1.0
    best_funcs = dict(funcs)
    rounds_without_improvement = 0

    # Per-function candidate tries so we can reject regressions.
    pending_culprit: Optional[str] = None
    pending_old_code: Optional[str] = None

    for rnd in range(max_rounds):
        print(f"\n===== round {rnd+1}/{max_rounds} =====", flush=True)
        assembled = assemble(funcs, n_train=n_train, n_test=n_test)
        script_path = out_dir / f"round_{rnd:02d}.py"
        with open(script_path, "w") as f:
            f.write(assembled)
        t0 = time.time()
        run = run_code(
            assembled, timeout_s=1200,
            env_extra={"CHECKPOINT": checkpoint,
                        "OUT_DIR": str(out_dir / f"round_{rnd:02d}_out")},
            tmp_path=str(script_path),
        )
        elapsed = time.time() - t0
        result_json: Optional[Dict[str, Any]] = None
        for line in (run["stdout"] or "").splitlines():
            if line.startswith("RESULT_JSON: "):
                try:
                    result_json = json.loads(line[len("RESULT_JSON: "):])
                except Exception:
                    pass
        reward, components = compute_reward(result_json, run["exit_code"])
        culprit, error_summary = attribute_errors(
            result_json, run["stderr"], components=components)
        trace.append({
            "round": rnd, "reward": reward,
            "components": components,
            "culprit": culprit,
            "error_summary_preview": error_summary[:400],
            "elapsed_s": elapsed,
            "exit_code": run["exit_code"],
            "stderr_tail": (run["stderr"] or "")[-400:],
        })
        print(f"  reward = {reward:.3f}  "
              f"sig={components['signature_success_rate']:.2f}  "
              f"acc={components['accuracy']:.2f}  "
              f"err={components['low_error_rate']:.2f}  "
              f"struct={components['structural']:.2f}  "
              f"culprit={culprit}  {elapsed:.1f}s", flush=True)

        # Reject-if-worse: if the previous round replaced `pending_culprit`
        # with a candidate that produced a LOWER reward, roll back.
        if pending_culprit is not None and pending_old_code is not None:
            if reward < best_reward - 1e-6:
                print(f"  candidate for `{pending_culprit}` made reward "
                      f"{reward:.3f} < best {best_reward:.3f} — rolling back",
                      flush=True)
                funcs[pending_culprit] = pending_old_code
                trace[-1]["rolled_back"] = True
                # Don't let the plateau counter advance because of a
                # rejected candidate — treat it as if no change.
                rounds_without_improvement += 1
                pending_culprit = None
                pending_old_code = None
                # Skip the rest of this round (no new re-prompt after rollback
                # unless culprit has been re-identified).
                continue

        if reward > best_reward + 1e-6:
            best_reward = reward
            best_funcs = dict(funcs)
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1
        pending_culprit = None
        pending_old_code = None

        if reward >= target_reward:
            print(f"  reached target — stopping", flush=True)
            break
        if rounds_without_improvement >= patience:
            print(f"  plateau for {patience} rounds — stopping", flush=True)
            break
        if culprit is None:
            print(f"  no culprit identified — stopping", flush=True)
            break

        # Re-prompt the culprit function
        print(f"  re-prompting `{culprit}`", flush=True)
        current_code = funcs.get(culprit, "")
        messages = build_improvement_prompt(
            culprit, current_code, reward, components,
            target_reward, error_summary)
        try:
            response = llm_generate(vllm_url, model, messages,
                                     max_tokens=4500, temperature=0.1)
        except Exception as exc:
            print(f"  LLM error: {exc}", flush=True)
            break
        new_code = extract_code(response)
        if not new_code:
            print(f"  no code block in response — stopping", flush=True)
            break
        # Sanity: new code should contain `def <culprit>(`
        if f"def {culprit}" not in new_code:
            print(f"  response didn't redefine {culprit} — stopping",
                  flush=True)
            break
        # Stage the candidate (reject-if-worse is evaluated next round).
        pending_culprit = culprit
        pending_old_code = funcs.get(culprit, "")
        funcs[culprit] = new_code

    return {
        "status": "success" if best_reward >= target_reward else "partial",
        "best_reward": best_reward,
        "final_functions": best_funcs,
        "trace": trace,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def load_seed_functions(seed_dir: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name in ("generate_cases", "ingest_and_store", "evaluate_precedents"):
        # Try bare name first, then final_ prefix (v4 champion outputs)
        for candidate in (seed_dir / f"{name}.py",
                           seed_dir / f"final_{name}.py"):
            if candidate.exists():
                out[name] = candidate.read_text()
                break
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed_dir", required=True,
                   help="directory with v3's generate_cases.py etc.")
    p.add_argument("--vllm_url", default="http://172.17.0.1:30000/v1")
    p.add_argument("--model", default="NVIDIA-Nemotron-3-Nano-30B-A3B-FP8")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--target_reward", type=float, default=0.90)
    p.add_argument("--max_rounds", type=int, default=8)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--n_train", type=int, default=100)
    p.add_argument("--n_test", type=int, default=10)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = load_seed_functions(Path(args.seed_dir))
    if not seeds or len(seeds) < 3:
        print(f"ERROR: seed_dir must contain generate_cases.py, "
              f"ingest_and_store.py, evaluate_precedents.py "
              f"(found: {list(seeds)})", flush=True)
        sys.exit(2)
    print(f"Seeding from {args.seed_dir} with "
          f"{len(seeds)} functions", flush=True)

    result = reward_loop(
        seeds, vllm_url=args.vllm_url, model=args.model,
        checkpoint=args.checkpoint, out_dir=out_dir,
        target_reward=args.target_reward,
        max_rounds=args.max_rounds, patience=args.patience,
        n_train=args.n_train, n_test=args.n_test,
    )
    # Persist final functions + trace
    for name, code in result["final_functions"].items():
        with open(out_dir / f"final_{name}.py", "w") as f:
            f.write(code)
    with open(out_dir / "reward_trace.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n=== v4 reward loop SUMMARY ===", flush=True)
    print(f"  best reward:  {result['best_reward']:.3f}", flush=True)
    print(f"  status:       {result['status']}", flush=True)
    print(f"  rounds:       {len(result['trace'])}", flush=True)
    for t in result["trace"]:
        print(f"   round {t['round']:2d}: reward={t['reward']:.3f}  "
              f"sig={t['components']['signature_success_rate']:.2f}  "
              f"acc={t['components']['accuracy']:.2f}  "
              f"culprit={t['culprit']}", flush=True)
    sys.exit(0 if result["status"] == "success" else 1)


if __name__ == "__main__":
    main()
