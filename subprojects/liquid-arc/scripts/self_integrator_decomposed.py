"""SelfIntegrator v3 — decomposed per-function generation.

Breaks a complex task into a dependency-ordered list of small functions.
For each function:
  1. Prompt Nemotron with the function's signature + I/O contract +
     the upstream function(s) it depends on (their source, not just
     signatures — the model sees exactly what it's calling).
  2. Run the generated function against a lightweight harness that
     exercises a minimal case and validates output shape.
  3. On success, freeze the code and move on. Accumulated working
     functions become part of the context for later functions.
  4. On failure, retry the function only — not the whole script.

This is strictly more efficient than whole-script regeneration:
  - Smaller token budget per call
  - Error attribution is unambiguous (only one function just changed)
  - Failures don't cascade into unrelated bugs
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
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


_CODE_RX = re.compile(r"```(?:python)?\s*\n(.*?)```", flags=re.DOTALL)


def _llm_generate(vllm_url: str, model: str, messages: list,
                   max_tokens: int = 4000, temperature: float = 0.1) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    r = requests.post(f"{vllm_url.rstrip('/')}/chat/completions",
                      json=payload, timeout=240)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def extract_code(text: str) -> Optional[str]:
    m = _CODE_RX.search(text)
    if m:
        return m.group(1)
    # Unterminated fence
    if "```python" in text:
        body = text.split("```python", 1)[1]
        # Keep everything up to last complete line
        if "\n" in body:
            body = body.rsplit("\n", 1)[0] + "\n"
        return body
    return None


def run_python_code(code: str, *, timeout_s: int, env_extra: Dict[str, str]
                     ) -> Dict[str, Any]:
    tmp = Path("/tmp/self_int3_tmp.py")
    with open(tmp, "w") as f:
        f.write(code)
    env = os.environ.copy()
    env.update(env_extra)
    env["PYTHONPATH"] = env_extra.get(
        "PYTHONPATH", "/workspace/liquid-arc")
    try:
        proc = subprocess.run(
            ["python", str(tmp)], env=env,
            cwd=env_extra.get("CWD", "/workspace/liquid-arc"),
            capture_output=True, text=True, timeout=timeout_s)
        return {
            "stdout": proc.stdout, "stderr": proc.stderr,
            "exit_code": proc.returncode, "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "stdout": exc.stdout or "", "stderr": (exc.stderr or "") +
                "\n[TIMEOUT]",
            "exit_code": -1, "timed_out": True,
        }


# ----------------------------------------------------------------------
# Per-function tasks for the precedent benchmark
# ----------------------------------------------------------------------


# These are the functions we want Nemotron to generate, in order. Each
# entry carries:
#   - name           — used as dict key
#   - contract       — signature + docstring embedded in the prompt
#   - harness_code   — a test that calls the function and asserts shape
#                       (signals pass/fail via exit code)
#   - uses           — names of previously-generated functions that must
#                       be pasted into the harness as context

API_CONTEXT = textwrap.dedent("""\
Library modules available:

    from liquid_arc.graph_rag.decoupled.graph_db import KnowledgeGraphDB
    from liquid_arc.graph_rag.decoupled.ode_engine import SubgraphODEEngine
    from liquid_arc.navigator_patterns import PatternLibrary

KnowledgeGraphDB has attribute `.G` (NetworkX DiGraph) and methods:
    .add_fragment(fragment, source_text=None, chunk_id=None,
                  doc_metadata=None, autosave=True)
    .get_neighbors(node_ids, hops=2, direction='both') -> set[str]
    .extract_subgraph(node_ids, max_nodes=200) -> dict with 'nodes','edges'
    .clear()

SubgraphODEEngine(checkpoint_path, device='cpu').compute_signature(subgraph)
    returns list[float] (64-d signature).

PatternLibrary(library_path).store(signature_list, metadata_dict)
PatternLibrary.find_nearest(signature_list, threshold=0.5)
    returns dict(label, similarity, count, ...) or None

IMPORTANT fragment shape conventions:
    fragment = {"nodes": [...], "edges": [...]}
    node     = {"id": str, "type": str, "role": str}
    edge     = {"src": str, "dst": str, "type": str, "scope": str-or-None}

Valid node types: "event","state","consequence","cause","role",
    "credential","requirement","prerequisite","concept","entity"
    (use these exact strings — other values hash to a random slot)
Valid roles: "root","intermediate","terminal","scope"
""")


FUNCTION_SPECS: List[Dict[str, Any]] = [
    # ------------------------------------------------------------------
    {
        "name": "generate_cases",
        "contract": textwrap.dedent("""\
            def generate_cases(n_cases: int, seed: int = 42) -> list[dict]:
                '''Generate n_cases legal case dicts across 3 domains.

                Returns: list of dicts, each with keys:
                    case_id (str):   unique id like 'case_0'
                    domain  (str):   one of 'contract_breach','tort_negligence',
                                     'ip_infringement'
                    shape   (str):   identifies which of 2 topologies was used,
                                     e.g. 'contract_breach_shape_0' or
                                          'contract_breach_shape_1'
                    text    (str):   2-3 sentences of realistic case prose
                    fragment (dict): {'nodes': [...], 'edges': [...]}
                        nodes use node types from the API_CONTEXT list
                        (stick to 'event','consequence','cause','state')
                        each chain has 4 nodes: root -> intermediate -> ... -> terminal

                Distribution: cases should be roughly balanced across
                3 domains × 2 shapes per domain = 6 groups.

                Node IDs must be unique across ALL cases (prefix with
                case_id to keep them distinct).
                '''
        """),
        "harness": textwrap.dedent("""\
            import json
            cases = generate_cases(12, seed=42)
            assert isinstance(cases, list), "must return list"
            assert len(cases) == 12, f"expected 12, got {len(cases)}"
            for c in cases:
                assert set(c.keys()) >= {"case_id","domain","shape","text","fragment"}, \
                    f"missing keys in {c}"
                assert c["domain"] in ("contract_breach","tort_negligence","ip_infringement"), \
                    f"bad domain: {c['domain']}"
                assert isinstance(c["fragment"], dict), "fragment must be dict"
                assert "nodes" in c["fragment"] and "edges" in c["fragment"], \
                    "fragment must have nodes and edges"
                assert len(c["fragment"]["nodes"]) >= 3, f"too few nodes"
                for n in c["fragment"]["nodes"]:
                    assert set(n.keys()) >= {"id","type","role"}, f"node: {n}"
            # Uniqueness of node IDs
            all_ids = set()
            dupes = 0
            for c in cases:
                for n in c["fragment"]["nodes"]:
                    if n["id"] in all_ids:
                        dupes += 1
                    all_ids.add(n["id"])
            assert dupes == 0, f"{dupes} duplicate node IDs"
            print("RESULT_JSON: " + json.dumps({
                "n_cases": len(cases),
                "unique_nodes": len(all_ids),
                "shapes_seen": sorted({c["shape"] for c in cases}),
            }))
        """),
        "uses": [],
    },
    # ------------------------------------------------------------------
    {
        "name": "ingest_and_store",
        "contract": textwrap.dedent("""\
            def ingest_and_store(cases, db_path, patterns_path, checkpoint):
                '''Ingest each case's fragment into a KnowledgeGraphDB and
                compute its metric signature via SubgraphODEEngine on the
                case's local neighborhood (2-hop, cap 30 nodes). Store the
                signature in a PatternLibrary with label = case_id.

                Parameters:
                    cases (list[dict]): output of generate_cases
                    db_path (str):       path for KnowledgeGraphDB persistence
                    patterns_path (str): path for PatternLibrary persistence
                    checkpoint (str):    path to the ODE checkpoint

                Returns:
                    dict with keys: n_ingested (int), n_signatures (int),
                    errors (list[str])
                '''
        """),
        "harness": textwrap.dedent("""\
            import json, os
            os.makedirs("/tmp/self_int3_bench", exist_ok=True)
            db_path = "/tmp/self_int3_bench/db.json"
            pat_path = "/tmp/self_int3_bench/pat.json"
            for p in (db_path, pat_path):
                if os.path.exists(p): os.remove(p)
            checkpoint = os.environ["CHECKPOINT"]
            # Small batch for quick test
            cases = generate_cases(6, seed=42)
            report = ingest_and_store(cases, db_path, pat_path, checkpoint)
            assert isinstance(report, dict), "must return dict"
            assert report["n_ingested"] == 6, f"{report}"
            assert report["n_signatures"] >= 1, f"no signatures stored: {report}"
            print("RESULT_JSON: " + json.dumps(report))
        """),
        "uses": ["generate_cases"],
    },
    # ------------------------------------------------------------------
    {
        "name": "evaluate_precedents",
        "contract": textwrap.dedent("""\
            def evaluate_precedents(test_cases, patterns_path, checkpoint,
                                     work_dir):
                '''For each test case, ingest its fragment into a fresh
                scratch KnowledgeGraphDB (under work_dir), compute its
                signature, and find the nearest stored pattern in
                PatternLibrary at patterns_path (threshold=0.5).

                For each test case, record:
                    expected_shape (from case['shape'])
                    matched_label  (from PatternLibrary.find_nearest, or None)
                    cosine         (similarity score, or None)
                    matched_shape  (the shape of the training case the label
                                    refers to — you need to pass a lookup from
                                    case_id -> shape; we'll pass train_shape_map
                                    as an extra param)

                Actually, extended signature — please use:

                def evaluate_precedents(test_cases, patterns_path, checkpoint,
                                         work_dir, train_shape_map):

                Returns dict with: n_cases, accuracy (fraction where
                matched_shape == expected_shape), per_case (list).
                '''
        """),
        "harness": textwrap.dedent("""\
            import json, os
            os.makedirs("/tmp/self_int3_bench", exist_ok=True)
            checkpoint = os.environ["CHECKPOINT"]
            # Build a small train+test corpus to exercise the function
            train = generate_cases(6, seed=42)
            test  = generate_cases(3, seed=43)   # different seed → different IDs
            db_path = "/tmp/self_int3_bench/db2.json"
            pat_path = "/tmp/self_int3_bench/pat2.json"
            for p in (db_path, pat_path):
                if os.path.exists(p): os.remove(p)
            rep = ingest_and_store(train, db_path, pat_path, checkpoint)
            assert rep["n_signatures"] >= 1
            train_shape_map = {c["case_id"]: c["shape"] for c in train}
            result = evaluate_precedents(
                test, pat_path, checkpoint,
                work_dir="/tmp/self_int3_bench/eval",
                train_shape_map=train_shape_map)
            assert "accuracy" in result and "n_cases" in result and "per_case" in result, \
                f"bad result: {result}"
            print("RESULT_JSON: " + json.dumps({
                "n_train": len(train), "n_test": len(test),
                "accuracy": result["accuracy"],
                "per_case": result["per_case"],
            }))
        """),
        "uses": ["generate_cases", "ingest_and_store"],
    },
]


def build_function_prompt(spec: Dict[str, Any],
                          previous_code: Dict[str, str]) -> List[Dict[str, str]]:
    dependency_src = "\n\n".join(
        f"# === already-working function: {name} ===\n{previous_code[name]}"
        for name in spec.get("uses", [])
        if name in previous_code
    )
    system = textwrap.dedent(f"""\
        You are a Python code-synthesis agent. You will write a SINGLE
        function — no main(), no imports at module level beyond what the
        function body needs.

        {API_CONTEXT}

        Rules for your output:
          - Exactly one fenced code block, ```python ... ```
          - Start the block with any imports the function needs
            (`import json`, `import os`, etc.) then the function body
          - Do NOT include the harness code
          - Do NOT include main() or top-level executable code
          - Keep the function self-contained: no unresolved references
    """)
    if dependency_src:
        system += ("\nPreviously working functions available in the same "
                   "script (you do NOT need to redefine them):\n"
                   + dependency_src)
    contract = spec["contract"]
    user = f"Write this function:\n\n{contract}"
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def run_function_harness(previous_code: Dict[str, str],
                         new_function_code: str,
                         harness_code: str, *,
                         checkpoint: str, timeout_s: int = 300
                         ) -> Dict[str, Any]:
    combined_imports = ["import json"]
    full_code = "\n".join(combined_imports) + "\n\n"
    for name, src in previous_code.items():
        full_code += f"# --- {name} ---\n" + src.strip() + "\n\n"
    full_code += "# --- new function ---\n" + new_function_code.strip() + "\n\n"
    full_code += "# --- harness ---\n" + harness_code.strip() + "\n"
    return run_python_code(full_code, timeout_s=timeout_s,
                            env_extra={"CHECKPOINT": checkpoint})


def synthesize_function(spec: Dict[str, Any], *,
                        vllm_url: str, model: str,
                        previous_code: Dict[str, str],
                        checkpoint: str,
                        max_iterations: int = 6
                        ) -> Dict[str, Any]:
    name = spec["name"]
    trace: List[Dict[str, Any]] = []
    last_err: Optional[str] = None
    for it in range(max_iterations):
        messages = build_function_prompt(spec, previous_code)
        if last_err:
            messages.append({
                "role": "user",
                "content": ("Previous attempt failed with:\n"
                            + last_err[-1500:] +
                            "\n\nFix it and output the full function body "
                            "again, inside one ```python``` block.")
            })
        t0 = time.time()
        response = _llm_generate(vllm_url, model, messages,
                                  max_tokens=4000)
        gen_s = time.time() - t0
        code = extract_code(response)
        if not code:
            last_err = "No ```python fenced code block in your response."
            trace.append({"iter": it, "no_code": True, "gen_s": gen_s})
            continue
        print(f"  [{name}] iter {it+1}: {len(code)} chars in {gen_s:.1f}s",
              flush=True)
        run = run_function_harness(
            previous_code, code, spec["harness"],
            checkpoint=checkpoint)
        trace.append({
            "iter": it, "gen_s": gen_s,
            "exit_code": run["exit_code"],
            "timed_out": run["timed_out"],
            "stderr_tail": run["stderr"][-500:] if run["stderr"] else "",
            "stdout_tail": run["stdout"][-400:] if run["stdout"] else "",
        })
        if run["exit_code"] == 0:
            print(f"  [{name}] ✓ passed harness in "
                  f"{len(trace)} iter(s)", flush=True)
            return {
                "name": name, "status": "success",
                "code": code, "iterations": it + 1,
                "trace": trace,
            }
        err_summary = run["stderr"] or run["stdout"]
        print(f"  [{name}] ✗ exit {run['exit_code']}, "
              f"stderr tail: {err_summary[-300:].splitlines()[-1] if err_summary.strip() else '(empty)'}",
              flush=True)
        last_err = err_summary
    return {"name": name, "status": "failed",
            "iterations": max_iterations, "trace": trace,
            "last_code": code if 'code' in locals() else None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vllm_url", default="http://172.17.0.1:30000/v1")
    p.add_argument("--model", default="NVIDIA-Nemotron-3-Nano-30B-A3B-FP8")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_iterations_per_function", type=int, default=6)
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    working: Dict[str, str] = {}
    results: List[Dict[str, Any]] = []
    for spec in FUNCTION_SPECS:
        print(f"\n=== synthesizing {spec['name']} ===", flush=True)
        res = synthesize_function(
            spec, vllm_url=args.vllm_url, model=args.model,
            previous_code=working, checkpoint=args.checkpoint,
            max_iterations=args.max_iterations_per_function)
        results.append(res)
        if res["status"] == "success":
            working[spec["name"]] = res["code"]
            with open(out_dir / f"{spec['name']}.py", "w") as f:
                f.write(res["code"])
        else:
            # Halt: downstream functions depend on this one
            break

    # Assemble the final script (if all succeeded)
    overall_success = len(working) == len(FUNCTION_SPECS)
    final_script: Optional[str] = None
    if overall_success:
        final_script = "# Auto-assembled by SelfIntegrator v3\n"
        final_script += "import json\n\n"
        for name in [s["name"] for s in FUNCTION_SPECS]:
            final_script += f"# === {name} ===\n" + working[name] + "\n\n"
        # Add a main that runs the full benchmark
        final_script += textwrap.dedent("""\
            if __name__ == '__main__':
                import os
                out_dir = os.environ.get('OUT_DIR', '/tmp/self_int3_final')
                os.makedirs(out_dir, exist_ok=True)
                checkpoint = os.environ['CHECKPOINT']
                train = generate_cases(100, seed=42)
                test  = generate_cases(10, seed=99)
                ingest_report = ingest_and_store(
                    train,
                    os.path.join(out_dir, 'db.json'),
                    os.path.join(out_dir, 'patterns.json'),
                    checkpoint)
                train_map = {c['case_id']: c['shape'] for c in train}
                eval_report = evaluate_precedents(
                    test, os.path.join(out_dir, 'patterns.json'),
                    checkpoint,
                    os.path.join(out_dir, 'eval_scratch'),
                    train_map)
                final = {
                    'ingest': ingest_report,
                    'n_test_cases': len(test),
                    'accuracy': eval_report.get('accuracy'),
                    'per_case': eval_report.get('per_case'),
                }
                print('RESULT_JSON: ' + json.dumps(final))
        """)
        with open(out_dir / "precedent_benchmark_assembled.py", "w") as f:
            f.write(final_script)

    with open(out_dir / "report.json", "w") as f:
        json.dump({
            "overall_success": overall_success,
            "per_function": results,
        }, f, indent=2, default=str)

    print(f"\n=== SelfIntegrator v3 SUMMARY ===", flush=True)
    for res in results:
        print(f"  {res['name']:22s} {res['status']:8s}  "
              f"iter={res['iterations']}", flush=True)
    print(f"  overall: {'SUCCESS' if overall_success else 'PARTIAL/FAIL'}",
          flush=True)
    if final_script:
        # Run the final assembled benchmark end-to-end
        print("\n=== running assembled benchmark ===", flush=True)
        full_run = run_python_code(
            final_script, timeout_s=1200,
            env_extra={"CHECKPOINT": args.checkpoint,
                        "OUT_DIR": str(out_dir / "final_output")})
        print(f"  exit: {full_run['exit_code']}  "
              f"timed_out: {full_run['timed_out']}", flush=True)
        last_lines = (full_run["stdout"] or "").splitlines()
        result_json = None
        for line in last_lines:
            if line.startswith("RESULT_JSON: "):
                result_json = line[13:]
                break
        print(f"  RESULT_JSON: {result_json}", flush=True)
        if not result_json and full_run["stderr"]:
            print(f"  stderr tail: {full_run['stderr'][-400:]}", flush=True)
        with open(out_dir / "assembled_run.json", "w") as f:
            json.dump({
                "exit_code": full_run["exit_code"],
                "result_json": result_json,
                "stderr_tail": full_run["stderr"][-1500:] if full_run["stderr"] else "",
            }, f, indent=2)

    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    main()
