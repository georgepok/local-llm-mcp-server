"""Stage 2 — autonomous improvement identification + verifiable impl.

Ambition: move Qwen80 from "execute a predefined spec" to "read real
LiquidARC source, identify a specific weakness, propose & implement an
improvement, prove the improvement is real with a verifiable test".

Protocol per round:
  1. Qwen receives the full source of `KnowledgeGraphDB` and a short
     catalog of acceptable improvement classes (perf, correctness,
     edge-case handling, new capability).
  2. Qwen returns a single JSON-wrapped package:
        {
          "target_method": "method name it's improving or adding",
          "rationale":     "one-paragraph why this is an improvement",
          "impl_code":     python source for the replacement/new method
                           as a standalone function (monkey-patched at runtime),
          "test_code":     python source that (a) passes when the impl
                           is applied, (b) FAILS on the unpatched class
                           (proves non-triviality).
        }
  3. The harness:
        a) Runs the test code against the UNPATCHED class — expects failure
        b) Monkey-patches in the impl
        c) Runs the test code again — expects pass
        d) Runs a regression suite on all existing methods
  4. Reward = weighted sum of the four signals:
        0.15 shape_ok (JSON well-formed, required fields present)
        0.15 impl_exec_ok (candidate exec's without syntax/runtime error)
        0.25 regression_pass (existing methods still work)
        0.30 new_test_passes_on_improved
        0.15 new_test_fails_on_original (non-triviality)

Target reward: 0.90. Loop up to 8 rounds with reject-if-worse.
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


_CODE_RX = re.compile(r"```(?:python|json)?\s*\n(.*?)```", flags=re.DOTALL)
_JSON_RX = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", flags=re.DOTALL)


def llm_generate(vllm_url: str, model: str, messages: List[Dict[str, str]],
                 max_tokens: int = 6000, temperature: float = 0.2) -> str:
    r = requests.post(
        f"{vllm_url.rstrip('/')}/chat/completions",
        json={"model": model, "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature,
               "chat_template_kwargs": {"enable_thinking": False}},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON block from Qwen's response. Try fenced ```json```
    first, then raw balanced braces."""
    m = _CODE_RX.search(text)
    if m:
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass
    # Try balanced-braces scan
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        return None
    return None


# ----------------------------------------------------------------------
# Task prompt
# ----------------------------------------------------------------------

GDB_SOURCE_PATH = "/workspace/liquid-arc/liquid_arc/graph_rag/decoupled/graph_db.py"

IMPROVEMENT_CATALOG = textwrap.dedent('''\
    Examples of acceptable improvement classes (choose one that you
    can genuinely verify):

      PERFORMANCE — e.g., `get_neighbors` builds a fresh undirected view
          every call; caching it would help for repeat queries.
          Verifiable: wall-time on 1000-node graph with 50 queries.

      CORRECTNESS — e.g., `trace_causal_chain` picks `predecessors[0]`
          which is order-dependent. Picking by `mention_count` or by
          most-recent would be more principled.
          Verifiable: construct graph where order is non-deterministic;
          new test asserts the principled choice.

      EDGE-CASE COVERAGE — e.g., `extract_subgraph` does not dedupe the
          input `node_ids` list; duplicates inflate ordering behavior.
          Verifiable: construct an input with duplicate ids; new test
          asserts deduped behavior.

      NEW CAPABILITY — e.g., a new method `k_shortest_causal_paths` that
          returns top-k distinct causal paths between two nodes.
          Verifiable: construct branching graph; new test enumerates
          expected paths.

    Important rules for your improvement:
      - target_method must name a method that exists on KnowledgeGraphDB
        OR a new method name that does NOT clash with an existing one.
      - impl_code must define EXACTLY ONE top-level function whose
        name == target_method. The function takes `self` as its first
        argument; we will monkey-patch it as a method.
      - test_code must define a single top-level function
        `test_improvement(db_factory)` where `db_factory` is a callable
        returning a fresh `KnowledgeGraphDB` instance. The test must:
          * construct test data using the db returned by db_factory()
          * call the target method
          * raise `AssertionError` when the behavior doesn't meet the
            improvement's claim
        Your test will be run BEFORE monkey-patching (should raise /
        fail) and AFTER monkey-patching (should pass).
''')


TASK_PROMPT_TEMPLATE = textwrap.dedent('''\
    You are reading the FULL source of `KnowledgeGraphDB` below.
    Identify ONE concrete improvement. Return exactly ONE fenced
    ```json``` block with the four fields listed in the protocol.

    {catalog}

    === SOURCE of KnowledgeGraphDB ===
    ```python
    {source}
    ```

    Output a single ```json``` block with:
        {{"target_method": "...",
          "rationale": "...",
          "impl_code": "def <target_method>(self, ...): ...",
          "test_code": "def test_improvement(db_factory): ..."}}
''')


# ----------------------------------------------------------------------
# Harness (written by us, runs against Qwen's impl + test)
# ----------------------------------------------------------------------

HARNESS_TEMPLATE = r'''
import json, os, sys, traceback
sys.path.insert(0, "/workspace/liquid-arc")

from liquid_arc.graph_rag.decoupled.graph_db import KnowledgeGraphDB

def db_factory():
    path = f"/tmp/stage2_{os.getpid()}_{id(object()):x}.json"
    if os.path.exists(path):
        os.remove(path)
    return KnowledgeGraphDB(path)

# ---- Ingest candidate impl + test into namespaces ----
IMPL_CODE = r"""
%%IMPL_CODE%%
"""
TEST_CODE = r"""
%%TEST_CODE%%
"""
TARGET_METHOD = "%%TARGET_METHOD%%"

result = {
    "shape_ok": False,
    "impl_exec_ok": False,
    "regression_pass": False,
    "new_test_passes_on_improved": False,
    "new_test_fails_on_original": False,
    "errors": [],
}

# ---- shape_ok: required fields were parsed already by harness driver ----
if TARGET_METHOD and IMPL_CODE.strip() and TEST_CODE.strip():
    result["shape_ok"] = True

# ---- impl_exec_ok: exec impl into a scratch namespace preloaded with
# common imports so Qwen can type-hint with typing names freely ----
import typing, networkx, collections, math, itertools, heapq, functools
impl_ns = {"__name__": "impl",
           "nx": networkx,
           "networkx": networkx,
           "typing": typing,
           "collections": collections,
           "math": math,
           "itertools": itertools,
           "heapq": heapq,
           "functools": functools}
# Expose common typing names directly
for _name in ("List","Dict","Set","Tuple","Iterable","Optional","Any",
              "Callable","Sequence","Union","Mapping"):
    impl_ns[_name] = getattr(typing, _name)
try:
    exec(IMPL_CODE, impl_ns)
    assert TARGET_METHOD in impl_ns, f"impl did not define {TARGET_METHOD}"
    result["impl_exec_ok"] = True
except Exception as e:
    result["errors"].append(f"impl_exec: {e}")

# ---- test_fails_on_original: run test BEFORE monkey-patching ----
if result["impl_exec_ok"]:
    test_ns = dict(impl_ns)
    test_ns["__name__"] = "test"
    try:
        exec(TEST_CODE, test_ns)
    except Exception as e:
        result["errors"].append(f"test_exec: {e}")
    else:
        fn = test_ns.get("test_improvement")
        if fn is None:
            result["errors"].append("test did not define test_improvement")
        else:
            failed = False
            try:
                fn(db_factory)
            except AssertionError as e:
                failed = True
            except Exception as e:
                # Unexpected error ≠ clean AssertionError; count as failed test
                failed = True
            result["new_test_fails_on_original"] = failed
            if not failed:
                result["errors"].append(
                    "test passed on unpatched class — improvement is trivial")

# ---- Monkey-patch: install impl as method on KnowledgeGraphDB ----
if result["impl_exec_ok"]:
    original_method = getattr(KnowledgeGraphDB, TARGET_METHOD, None)
    setattr(KnowledgeGraphDB, TARGET_METHOD, impl_ns[TARGET_METHOD])

    # ---- new_test_passes_on_improved ----
    test_ns2 = dict(impl_ns)
    test_ns2["__name__"] = "test2"
    try:
        exec(TEST_CODE, test_ns2)
        fn = test_ns2.get("test_improvement")
        if fn is not None:
            fn(db_factory)
            result["new_test_passes_on_improved"] = True
    except Exception as e:
        result["errors"].append(f"improved_test: {e}")

    # ---- regression_pass: exercise a battery of existing methods ----
    try:
        db = db_factory()
        frag = {
            "nodes": [
                {"id": "root", "type": "event", "role": "root"},
                {"id": "mid1", "type": "state", "role": "intermediate"},
                {"id": "mid2", "type": "state", "role": "intermediate"},
                {"id": "leaf", "type": "consequence", "role": "terminal"},
            ],
            "edges": [
                {"src": "root", "dst": "mid1", "type": "causes", "scope": None},
                {"src": "mid1", "dst": "mid2", "type": "causes", "scope": "prod"},
                {"src": "mid2", "dst": "leaf", "type": "causes", "scope": None},
            ],
        }
        rep = db.add_fragment(frag, source_text="hello", autosave=False)
        assert rep["added_nodes"] == 4 and rep["added_edges"] == 3
        chain = db.trace_causal_chain("leaf", max_hops=10)
        assert chain["root"] == "root" and chain["hops"] == 3
        reach = db.get_reachable("root", max_hops=5)
        assert {"mid1", "mid2", "leaf"} <= reach
        sub = db.extract_subgraph(["root", "mid1"], max_nodes=10)
        assert len(sub["nodes"]) == 2
        txt = db.retrieve_text(["root"], max_segments=2)
        assert len(txt) >= 1
        stats = db.stats()
        assert stats["n_nodes"] == 4
        # scope_filter
        filt = db.scope_filter("prod")
        assert filt.has_edge("mid1", "mid2")
        # get_neighbors
        neigh = db.get_neighbors(["root"], hops=2)
        assert "mid1" in neigh and "mid2" in neigh
        # find_communities shouldn't crash
        _ = db.find_communities(min_size=2)
        db.clear()
        result["regression_pass"] = True
    except Exception as e:
        result["errors"].append(f"regression: {type(e).__name__}: {e}")

    # Restore original method (cleanliness)
    if original_method is not None:
        setattr(KnowledgeGraphDB, TARGET_METHOD, original_method)
    else:
        try:
            delattr(KnowledgeGraphDB, TARGET_METHOD)
        except Exception:
            pass

# ---- reward ----
reward = (0.15 * (1 if result["shape_ok"] else 0)
          + 0.15 * (1 if result["impl_exec_ok"] else 0)
          + 0.25 * (1 if result["regression_pass"] else 0)
          + 0.30 * (1 if result["new_test_passes_on_improved"] else 0)
          + 0.15 * (1 if result["new_test_fails_on_original"] else 0))
result["reward"] = reward
print("RESULT_JSON: " + json.dumps(result, default=str))
'''


# ----------------------------------------------------------------------
# Loop driver
# ----------------------------------------------------------------------


def run_harness(harness_code: str, out_dir: Path, iteration: int,
                timeout_s: int = 180) -> Dict[str, Any]:
    script = out_dir / f"iter_{iteration:02d}.py"
    with open(script, "w") as f:
        f.write(harness_code)
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", "/workspace/liquid-arc")
    try:
        proc = subprocess.run(
            ["python", str(script)], env=env, cwd="/workspace/liquid-arc",
            capture_output=True, text=True, timeout=timeout_s)
        stdout, stderr, ec = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + "\n[TIMEOUT]"
        ec = -1
    result_json = None
    for line in (stdout or "").splitlines():
        if line.startswith("RESULT_JSON: "):
            try:
                result_json = json.loads(line[len("RESULT_JSON: "):])
            except Exception:
                pass
    return {"stdout": stdout, "stderr": stderr, "exit_code": ec,
            "result_json": result_json}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vllm_url", default="http://172.17.0.1:30000/v1")
    p.add_argument("--model", default="Qwen3-Next-80B-A3B-Instruct-FP8")
    p.add_argument("--source_path", default=GDB_SOURCE_PATH)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_rounds", type=int, default=8)
    p.add_argument("--target_reward", type=float, default=0.90)
    p.add_argument("--patience", type=int, default=5)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load source — if on the driver machine, read from local path.
    source_file = Path(args.source_path)
    if not source_file.exists():
        print(f"[fatal] source not found at {args.source_path}", flush=True)
        sys.exit(2)
    source_text = source_file.read_text()

    trace: List[Dict[str, Any]] = []
    best_reward = -1.0
    best_package: Optional[Dict[str, Any]] = None
    stagnation = 0
    previous_error: Optional[str] = None

    for rnd in range(args.max_rounds):
        print(f"\n===== round {rnd+1}/{args.max_rounds} =====", flush=True)
        prompt = TASK_PROMPT_TEMPLATE.format(
            catalog=IMPROVEMENT_CATALOG, source=source_text)
        messages = [
            {"role": "system",
             "content": "You identify and implement improvements to "
                        "existing Python code. Respond with one ```json``` "
                        "block containing the required fields. No other "
                        "content."},
            {"role": "user", "content": prompt},
        ]
        if previous_error:
            messages.append({
                "role": "user",
                "content": f"Previous attempt failed:\n{previous_error[-1500:]}\n"
                           f"Propose a DIFFERENT improvement or fix the "
                           f"specific issue. Return a new JSON package."})

        t0 = time.time()
        try:
            response = llm_generate(args.vllm_url, args.model, messages,
                                     max_tokens=7000, temperature=0.3)
        except Exception as exc:
            trace.append({"round": rnd, "llm_error": str(exc)})
            previous_error = f"LLM error: {exc}"
            stagnation += 1
            continue
        gen_s = time.time() - t0
        package = extract_json_block(response)
        if not package or not all(
                k in package
                for k in ("target_method", "impl_code", "test_code")):
            print(f"  [no package] gen {gen_s:.1f}s", flush=True)
            previous_error = ("Response did not contain a complete JSON "
                               "package with target_method, impl_code, "
                               "test_code.")
            trace.append({"round": rnd, "gen_s": gen_s, "no_package": True,
                            "response_preview": response[:500]})
            stagnation += 1
            if stagnation >= args.patience:
                break
            continue

        target = str(package["target_method"]).strip()
        impl = str(package["impl_code"])
        test = str(package["test_code"])

        print(f"  target = {target}", flush=True)
        with open(out_dir / f"iter_{rnd:02d}_package.json", "w") as f:
            json.dump(package, f, indent=2)

        # Assemble harness
        harness = (HARNESS_TEMPLATE
                   .replace("%%IMPL_CODE%%", impl)
                   .replace("%%TEST_CODE%%", test)
                   .replace("%%TARGET_METHOD%%", target))
        run = run_harness(harness, out_dir, rnd, timeout_s=120)

        reward = 0.0
        result = {}
        if run["result_json"] is not None:
            result = run["result_json"]
            reward = result.get("reward", 0.0)

        print(f"  reward={reward:.3f}  "
              f"shape={result.get('shape_ok', False)}  "
              f"exec={result.get('impl_exec_ok', False)}  "
              f"regr={result.get('regression_pass', False)}  "
              f"test_pass={result.get('new_test_passes_on_improved', False)}  "
              f"test_fail_orig={result.get('new_test_fails_on_original', False)}  "
              f"{gen_s:.1f}s", flush=True)
        if result.get("errors"):
            print(f"  errors: {result['errors'][:3]}", flush=True)

        trace.append({
            "round": rnd, "gen_s": gen_s, "target_method": target,
            "rationale": package.get("rationale", "")[:400],
            "reward": reward,
            "result": result,
            "stderr_tail": (run["stderr"] or "")[-500:],
        })

        if reward > best_reward + 1e-6:
            best_reward = reward
            best_package = package
            stagnation = 0
            with open(out_dir / "best_package.json", "w") as f:
                json.dump(package, f, indent=2)
        else:
            stagnation += 1

        if reward >= args.target_reward:
            print("  target hit — stopping", flush=True)
            break
        if stagnation >= args.patience:
            print(f"  patience exhausted — stopping", flush=True)
            break

        # Feedback for next round
        errs = result.get("errors", [])
        pass_improved = result.get("new_test_passes_on_improved", False)
        fails_original = result.get("new_test_fails_on_original", False)
        if fails_original and not pass_improved:
            # Impl and test disagree about what the improvement does.
            # Surface this explicitly and demand consistency.
            sample_err = next((e for e in errs if "improved_test" in e), "")
            previous_error = (
                f"Reward {reward:.3f}. Your TEST correctly distinguishes "
                f"the original (test_fails_on_original=True), but your "
                f"IMPL doesn't satisfy the test's assertion "
                f"(test_passes_on_improved=False).\n"
                f"Specifically: {sample_err}\n"
                f"Your impl and test must describe the SAME semantic "
                f"change. Reread your test's assertions and rewrite the "
                f"impl to make those assertions pass. Or rewrite the test "
                f"to match what your impl actually does. Do not leave the "
                f"gap between rationale/impl/test.")
        elif errs:
            previous_error = (f"Reward {reward:.3f}. Errors: {errs[:3]}. "
                               f"Components: shape={result.get('shape_ok')} "
                               f"exec={result.get('impl_exec_ok')} "
                               f"regr={result.get('regression_pass')} "
                               f"test_pass={pass_improved} "
                               f"test_fail_orig={fails_original}.")
        elif reward < args.target_reward:
            previous_error = (f"Reward {reward:.3f} below target "
                               f"{args.target_reward}. "
                               f"test_passes_on_improved={pass_improved}, "
                               f"test_fails_on_original={fails_original}. "
                               f"Consider a different target_method or make "
                               f"your test more discriminating.")

    summary = {
        "status": "success" if best_reward >= args.target_reward else "partial",
        "best_reward": best_reward,
        "best_package": best_package,
        "rounds": len(trace),
        "trace": trace,
    }
    with open(out_dir / "report.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n=== STAGE-2 SUMMARY ===")
    print(f"  best reward: {best_reward:.3f}  status: {summary['status']}")
    print(f"  rounds: {len(trace)}")
    if best_package:
        print(f"  chosen target: {best_package.get('target_method')}")
        print(f"  rationale: "
              f"{(best_package.get('rationale') or '')[:200]}")
    sys.exit(0 if summary["status"] == "success" else 1)


if __name__ == "__main__":
    main()
