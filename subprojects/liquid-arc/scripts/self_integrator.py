"""SelfIntegrator — a Nemotron-powered code-synthesis agent bounded to
LiquidARC-graph-engine tasks.

Loop:
  1. Load system prompt + API cheatsheet + reference templates.
  2. Send (task + previous error) to Nemotron.
  3. Parse a ```python ...``` block from the response.
  4. Write the code to a sandboxed scratch directory.
  5. Run it as a subprocess with a timeout and bounded working dir.
  6. On error: feed stderr back, retry up to MAX_ITERATIONS.
  7. On success: emit final script to scripts/generated/<task_id>.py + a
     trace log of every iteration.

Success criterion is user-supplied (a callable `verify(stdout, exit_code,
out_json)`). A task that lacks a verifier defaults to "exit code 0".

Usage:
    python scripts/self_integrator.py --task precedent_benchmark \\
        --vllm_url http://172.17.0.1:30000/v1 \\
        --model NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \\
        --out_dir /workspace/liquid-arc/shared/outbox/self_integrator/
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests


# ----------------------------------------------------------------------
# API cheatsheet — compact reference of what the generated code can use
# ----------------------------------------------------------------------

API_CHEATSHEET = """\
EXACT IMPORT STATEMENTS TO USE (copy-paste these verbatim at the top of
your script — do NOT invent new import paths):

    from liquid_arc.graph_rag.decoupled.graph_db import KnowledgeGraphDB
    from liquid_arc.graph_rag.decoupled.ode_engine import SubgraphODEEngine
    from liquid_arc.navigator_patterns import PatternLibrary
    from liquid_arc.graph_rag.vector_db import VectorDB

API signatures (do not treat `Class.method` as an import — these are
method calls after you've imported the class):

KnowledgeGraphDB(db_path: str)
  .add_fragment(fragment, source_text=None, chunk_id=None,
                doc_metadata=None, autosave=True)
      fragment = dict with keys 'nodes' and 'edges'
      nodes item  = dict(id=str, type=str, role=str)
      edges item  = dict(src=str, dst=str, type=str, scope=str-or-None)
  .trace_causal_chain(target, max_hops=10) returns dict(root,path,hops)
  .scope_filter(scope) returns a networkx.DiGraph
  .get_reachable(source, scope=None, max_hops=5) returns set[str]
  .find_communities(min_size=3) returns list[set[str]]
  .get_neighbors(node_ids, hops=2, direction='both') returns set[str]
  .extract_subgraph(node_ids, max_nodes=200) returns dict(nodes,edges)
  .retrieve_text(node_ids, max_segments=10) returns list[dict]
  .stats(compute_communities=False) returns dict
  .clear()

SubgraphODEEngine(checkpoint_path, device='cpu')
  .compute_diagnostics(subgraph) returns dict
      (keys include 'cv_g', 'tau_mean', 'metric_clusters',
       'per_node_centrality_metric_space')
  .compute_signature(subgraph) returns list[float] (64-d)
  .analyze(subgraph, query) returns dict

PatternLibrary(library_path: str)
  .find_nearest(signature, threshold=0.85) returns dict-or-None
  .store(signature, metadata)
  .reset()

VectorDB(dim=1024)
  .add(text, metadata=None) returns chunk_id (int)
  .query(text, k=10) returns list[dict(chunk_id,text,metadata,score)]

Reference script template (always read this before writing):
  {template_path}

Checkpoint available at:
  {checkpoint_path}

Your script MUST:
  - print a final JSON line starting with 'RESULT_JSON: '
    summarizing the benchmark (n_docs, accuracy, latency, etc.)
  - exit 0 on success, non-zero on failure
  - write its artifacts under OUT_DIR which is provided via env $OUT_DIR
  - not make outgoing network calls except to the vLLM endpoint at $VLLM_URL
  - finish within {timeout_s} seconds
"""


SYSTEM_PROMPT = """\
You are a Python code-synthesis agent for the LiquidARC graph-reasoning \
project. Given a task description and the API cheatsheet below, write a \
complete, runnable Python script.

Rules:
  - Output ONE fenced code block starting with ```python and ending with ```.
  - No commentary outside the code block.
  - Follow the API signatures exactly.
  - Start by reading env vars: CHECKPOINT, OUT_DIR, VLLM_URL.
  - Always print a final line starting with 'RESULT_JSON: ' containing JSON.
  - Handle errors with try/except and print informative error strings.
  - Use device='cpu' unless the task says otherwise.
  - Prefer the decoupled architecture (KnowledgeGraphDB + optional ODE).

{cheatsheet}

When a previous attempt failed, the user will include a STDERR block. \
Read the traceback carefully and fix the specific line(s) at fault. \
Do NOT rewrite the whole script if only one call needs correction.
"""


TASK_BANK: Dict[str, Dict[str, Any]] = {
    "precedent_benchmark": {
        "description": """\
Write a benchmark that measures pattern-library precedent finding in a
three-domain corpus.

Dataset generation (procedural, deterministic, seeded):
  - 3 legal domains: contract_breach, tort_negligence, ip_infringement
  - Each domain has 2 distinct structural 'case shapes' (causal chain
    topologies). Example for contract_breach:
      shape 1: breach → damages → remedy_sought → settlement
      shape 2: obligation_failure → reliance → loss → court_filing
  - Generate 100 total cases (~33 per domain).
    Per case:
      - text: 2-3 sentences describing the case (use realistic prose)
      - fragment: the typed graph fragment matching the shape
    Each shape appears ~15-17 times per domain.

Pipeline:
  1. Initialize KnowledgeGraphDB at OUT_DIR/precedent_db.json
  2. Ingest every case's fragment into the graph DB with source_text.
  3. After ingestion, for each case compute its metric signature via
     SubgraphODEEngine.compute_signature on a neighborhood subgraph
     (2-hop, cap 30 nodes) and store it in a PatternLibrary with a
     label 'case_<id>'.

Evaluation:
  10 novel test cases (3-4 per domain), each an instance of one shape
  the library has already seen. For each test case:
    - ingest its fragment into a SEPARATE scratch KnowledgeGraphDB
    - compute its signature
    - find the nearest stored pattern via PatternLibrary.find_nearest
      (threshold 0.5 for this experiment — we want to see the nearest
       match even if similarity is low)
    - record: expected_shape, matched_case_id, cosine, matched_shape
      (derived from the matched case's label)

Scoring:
  accuracy = fraction of test cases where matched_shape == expected_shape

Output:
  JSON via 'RESULT_JSON: {...}' containing:
    {n_cases_ingested, n_patterns_stored, n_test_cases,
     n_correct_shape_match, accuracy, per_test_case: [...]}
""",
        "template_path":
            "/workspace/liquid-arc/scripts/bench_graphrag_scale.py",
        "timeout_s": 600,
    },
}


# ----------------------------------------------------------------------
# vLLM client
# ----------------------------------------------------------------------


def _llm_generate(vllm_url: str, model: str, messages: list,
                   max_tokens: int = 3500, temperature: float = 0.1
                   ) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    r = requests.post(f"{vllm_url.rstrip('/')}/chat/completions",
                      json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


# ----------------------------------------------------------------------
# Code extraction + sandboxed execution
# ----------------------------------------------------------------------


_CODE_RX = re.compile(r"```(?:python)?\s*\n(.*?)```", flags=re.DOTALL)
_OPEN_FENCE_RX = re.compile(r"```(?:python)?\s*\n(.*)", flags=re.DOTALL)


def extract_code(response_text: str) -> Optional[str]:
    """Extract code from a complete fenced block OR from an unclosed one.

    If the closing ``` is missing (truncated response), still return the
    partial code so we can either lint/recover it or ask the LLM to
    continue generating.
    """
    m = _CODE_RX.search(response_text)
    if m:
        return m.group(1)
    m = _OPEN_FENCE_RX.search(response_text)
    if m:
        # Strip any trailing garbage that clearly isn't Python
        body = m.group(1)
        # Last line may be a partial — drop it if it doesn't end with \n
        if not body.endswith("\n"):
            body = body.rsplit("\n", 1)[0] + "\n"
        return body
    if response_text.strip().startswith("import ") or \
            response_text.strip().startswith("from "):
        return response_text
    return None


def is_truncated(response_text: str) -> bool:
    """True if the LLM stopped mid-code-block (no closing fence)."""
    opens = response_text.count("```")
    return opens == 1


def lint_code(code: str) -> Optional[str]:
    """Run pyflakes statically. Returns None if clean, else the error text.

    Catches NameError-class bugs (undefined variables, missing imports)
    before we pay the cost of subprocess execution.
    """
    try:
        from pyflakes.api import check  # type: ignore
        from pyflakes.reporter import Reporter  # type: ignore
        import io
        warn_buf = io.StringIO()
        err_buf = io.StringIO()
        reporter = Reporter(warn_buf, err_buf)
        n_issues = check(code, "<generated>", reporter)
        report = warn_buf.getvalue() + err_buf.getvalue()
        # Only block on CRITICAL issues that would cause a runtime crash.
        # Unused-variable and unused-import warnings are cosmetic and we
        # let them through.
        critical: List[str] = []
        for line in report.splitlines():
            # Exclude cosmetic warnings first
            lower = line.lower()
            if "never used" in lower or "imported but unused" in lower:
                continue
            if "redefinition of unused" in lower:
                continue
            if any(needle in line for needle in (
                    "undefined name",
                    "referenced before assignment",
                    "invalid syntax",
                    "unable to detect undefined names",
                    "'return' outside function",
                    "ExpandedReturn",
                    "SyntaxError",
            )):
                critical.append(line)
        if critical:
            return "\n".join(critical)
        return None
    except ImportError:
        # pyflakes not available — skip lint gate
        return None


def extract_traceback_lines(stderr: str, max_lines: int = 40) -> str:
    """Trim stderr to the traceback + last few context lines."""
    if not stderr:
        return ""
    lines = stderr.splitlines()
    # Keep only last traceback
    start = 0
    for i, line in enumerate(lines):
        if line.startswith("Traceback"):
            start = i
    return "\n".join(lines[start:])[-2500:]


def run_sandboxed(code: str, script_path: Path, *,
                  timeout_s: int, checkpoint: str, out_dir: Path,
                  vllm_url: str, cwd: str = "/workspace/liquid-arc",
                  pythonpath: str = "/workspace/liquid-arc") -> Dict[str, Any]:
    script_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(script_path, "w") as f:
        f.write(code)
    env = os.environ.copy()
    env["PYTHONPATH"] = pythonpath
    env["CHECKPOINT"] = checkpoint
    env["OUT_DIR"] = str(out_dir)
    env["VLLM_URL"] = vllm_url
    t0 = time.time()
    try:
        proc = subprocess.run(
            ["python", str(script_path)],
            cwd=cwd, env=env,
            capture_output=True, text=True,
            timeout=timeout_s,
        )
        stdout, stderr = proc.stdout, proc.stderr
        code_exit = proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + f"\n[TIMEOUT after {timeout_s}s]"
        code_exit = -1
        timed_out = True
    elapsed = time.time() - t0
    result_json = None
    for line in (stdout or "").splitlines():
        if line.startswith("RESULT_JSON: "):
            try:
                result_json = json.loads(line[len("RESULT_JSON: "):])
            except Exception:
                pass
    return {
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": code_exit,
        "elapsed_s": elapsed,
        "timed_out": timed_out,
        "result_json": result_json,
    }


# ----------------------------------------------------------------------
# Agent loop
# ----------------------------------------------------------------------


def agent_loop(task_id: str, task: Dict[str, Any], *,
               vllm_url: str, model: str,
               checkpoint: str, out_dir: Path,
               max_iterations: int = 5,
               verify: Optional[Callable[[Dict[str, Any]], bool]] = None,
               ) -> Dict[str, Any]:
    trace: list = []
    script_path = out_dir / f"{task_id}_generated.py"
    cheatsheet = (API_CHEATSHEET
                  .replace("{template_path}",
                           task.get("template_path", "(none)"))
                  .replace("{checkpoint_path}", checkpoint)
                  .replace("{timeout_s}", str(task.get("timeout_s", 300))))
    system = SYSTEM_PROMPT.replace("{cheatsheet}", cheatsheet)

    user_prompt = f"TASK: {task['description']}"
    previous_err: Optional[str] = None

    for it in range(max_iterations):
        print(f"\n--- iteration {it+1}/{max_iterations} ---", flush=True)
        messages = [{"role": "system", "content": system}]
        if previous_err:
            messages.append({
                "role": "user",
                "content": (user_prompt +
                            "\n\nPREVIOUS ATTEMPT STDERR (last 2000 chars):\n"
                            + previous_err[-2000:] +
                            "\n\nFix the specific line(s) at fault and "
                            "output the full corrected script.")
            })
        else:
            messages.append({"role": "user", "content": user_prompt})

        t0 = time.time()
        try:
            response = _llm_generate(vllm_url, model, messages,
                                      max_tokens=10000, temperature=0.1)
        except Exception as exc:
            print(f"[agent] LLM error: {exc}", flush=True)
            trace.append({"iteration": it, "llm_error": str(exc)})
            continue
        gen_s = time.time() - t0
        truncated = is_truncated(response)

        # If the response was truncated mid-code, ask the model to continue.
        # Do up to 2 continuation rounds per iteration.
        continuation_rounds = 0
        while truncated and continuation_rounds < 2:
            print(f"[agent]   truncated response; asking for continuation",
                  flush=True)
            cont_messages = list(messages)
            cont_messages.append({"role": "assistant", "content": response})
            cont_messages.append({
                "role": "user",
                "content": ("Your previous response was truncated. Continue "
                            "writing the Python script from EXACTLY where "
                            "you stopped. Do NOT repeat earlier lines. Do "
                            "NOT re-open a ```python fence. End your "
                            "response with ``` once you finish the script.")
            })
            try:
                cont = _llm_generate(vllm_url, model, cont_messages,
                                      max_tokens=6000, temperature=0.1)
            except Exception as exc:
                print(f"[agent]   continuation error: {exc}", flush=True)
                break
            response = response + cont
            truncated = is_truncated(response)
            continuation_rounds += 1

        code = extract_code(response)
        if not code:
            print("[agent] no code block in response, retrying", flush=True)
            trace.append({
                "iteration": it, "gen_s": gen_s,
                "no_code_block": True,
                "response_preview": response[:500],
            })
            previous_err = "Previous response did not contain a ```python fenced code block. Please output a complete script inside one."
            continue

        print(f"[agent] generated {len(code)} chars of code in {gen_s:.1f}s"
              f" (continuations: {continuation_rounds})", flush=True)

        # Lint gate — cheaper than subprocess. If pyflakes flags undefined
        # names etc., route the error back without paying the execution cost.
        lint_report = lint_code(code)
        if lint_report:
            print(f"[agent]   lint failed:\n{lint_report[:400]}", flush=True)
            trace.append({
                "iteration": it, "gen_s": gen_s,
                "lint_blocked": True,
                "lint_report": lint_report,
                "code_len": len(code),
                "continuations": continuation_rounds,
            })
            previous_err = (
                "Static lint (pyflakes) found issues. Fix these "
                "BEFORE trying to run the script:\n" + lint_report +
                "\n\nCommon fixes:\n"
                "  - define every variable used in a return statement\n"
                "  - wrap risky imports in try/except or remove\n"
                "  - don't assign to vars inside `if` blocks without an else\n"
            )
            continue
        run = run_sandboxed(
            code, script_path,
            timeout_s=task.get("timeout_s", 300),
            checkpoint=checkpoint,
            out_dir=out_dir / f"{task_id}_iter_{it}",
            vllm_url=vllm_url,
        )
        trace_row = {
            "iteration": it, "gen_s": gen_s,
            "exit_code": run["exit_code"],
            "timed_out": run["timed_out"],
            "elapsed_s": run["elapsed_s"],
            "has_result_json": run["result_json"] is not None,
            "stderr_tail": (run["stderr"] or "")[-800:],
            "stdout_tail": (run["stdout"] or "")[-800:],
        }
        trace.append(trace_row)
        print(f"[agent] exit {run['exit_code']}  "
              f"{run['elapsed_s']:.1f}s  "
              f"timeout={run['timed_out']}  "
              f"result_json={'yes' if run['result_json'] else 'no'}",
              flush=True)

        if run["exit_code"] == 0 and run["result_json"] is not None:
            ok = True
            if verify:
                try:
                    ok = verify(run["result_json"])
                except Exception as exc:
                    ok = False
                    print(f"[agent] verify error: {exc}", flush=True)
            if ok:
                print("[agent] SUCCESS", flush=True)
                return {
                    "task_id": task_id,
                    "status": "success",
                    "iterations": it + 1,
                    "final_result": run["result_json"],
                    "script_path": str(script_path),
                    "trace": trace,
                }

        # Prepare error feedback for next iteration — only the traceback
        # and last-printed-line, not full stderr (keeps context small).
        if run["timed_out"]:
            previous_err = ("Script timed out. Make it faster — smaller "
                             "dataset, skip expensive calls, or break early.")
        elif run["stderr"]:
            tb = extract_traceback_lines(run["stderr"])
            last_stdout_line = ""
            for line in reversed((run["stdout"] or "").splitlines()):
                if line.strip():
                    last_stdout_line = line
                    break
            previous_err = (
                "Runtime error:\n" + tb +
                ("\n\nLast line the script printed before crashing:\n  "
                 + last_stdout_line if last_stdout_line else "")
            )
        elif run["result_json"] is None:
            previous_err = ("Script exited but produced no "
                             "'RESULT_JSON: {...}' line. Add the final "
                             "summary print. Example: "
                             "print('RESULT_JSON: ' + json.dumps(summary))")
        else:
            previous_err = "Verification failed — review task requirements."

    print("[agent] FAILED after max iterations", flush=True)
    return {
        "task_id": task_id,
        "status": "failed",
        "iterations": max_iterations,
        "script_path": str(script_path),
        "trace": trace,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=list(TASK_BANK))
    p.add_argument("--vllm_url", default="http://172.17.0.1:30000/v1")
    p.add_argument("--model", default="NVIDIA-Nemotron-3-Nano-30B-A3B-FP8")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_iterations", type=int, default=5)
    args = p.parse_args()

    out_dir = Path(args.out_dir) / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    result = agent_loop(
        args.task, TASK_BANK[args.task],
        vllm_url=args.vllm_url, model=args.model,
        checkpoint=args.checkpoint, out_dir=out_dir,
        max_iterations=args.max_iterations,
    )
    with open(out_dir / "run_report.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n=== SELF-INTEGRATOR SUMMARY ===")
    print(f"  task:        {args.task}")
    print(f"  status:      {result['status']}")
    print(f"  iterations:  {result['iterations']}")
    print(f"  report:      {out_dir / 'run_report.json'}")
    sys.exit(0 if result["status"] == "success" else 1)


if __name__ == "__main__":
    main()
