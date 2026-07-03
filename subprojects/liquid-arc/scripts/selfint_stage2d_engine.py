"""Stage 2d — autonomous improvement on the LiquidARC GraphEngine.

Same Stage 2c architecture (split-role + forbid-repeat) aimed at
`liquid_arc/graph_engine_inference.py` instead of the simpler
graph DB. The engine is ODE-backed, requires a trained checkpoint,
and exposes heavier methods (`analyze_graph`, `compare_graphs`,
`get_graph_diagnostics`, `correct_answer`).

Key adaptations:
  * The harness loads a GraphEngine ONCE at startup and reuses it
    across shape/exec/regression/test calls.
  * Regression battery exercises analyze_graph + compare_graphs +
    get_graph_diagnostics on a small fixed graph.
  * `engine_factory()` returns the same shared engine instance (NOT
    a fresh one — loading is ~2s).
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
from typing import Any, Dict, List, Optional, Set

import requests


_CODE_RX = re.compile(r"```(?:python|json)?\s*\n(.*?)```", flags=re.DOTALL)


def llm_generate(vllm_url: str, model: str, messages: List[Dict[str, str]],
                 max_tokens: int = 6000, temperature: float = 0.3) -> str:
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
    m = _CODE_RX.search(text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass
    start = text.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
        else:
            if ch == '"': in_str = True
            elif ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except Exception:
                        return None
    return None


def extract_python_block(text: str) -> Optional[str]:
    m = _CODE_RX.search(text)
    if m:
        return m.group(1)
    if "```python" in text:
        body = text.split("```python", 1)[1]
        return body.rsplit("\n", 1)[0] if "\n" in body else body
    return None


# ----------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------


IMPROVEMENT_CATALOG = textwrap.dedent('''\
    Improvement classes for the LiquidARC GraphEngine:

      CORRECTNESS — e.g., `_root_cause` currently picks among reachable
          roots by head probability × path depth. There are degenerate
          cases (two equally-reachable candidates with equal prob) where
          the deterministic tie-break could be more principled.

      CONSISTENCY — e.g., `compare_graphs` returns `isomorphic` from
          networkx AND `signature_cosine` from LiquidARC. When they
          disagree, which should the caller trust?

      EDGE-CASES — empty graphs, single-node graphs, graphs with
          self-loops, disconnected components, target not in graph.

      NEW CAPABILITY — a new method that composes primitives, e.g.,
          `analyze_graph_batch` that runs N graphs with shared engine
          warm-up; or `get_pairwise_influence(src, dst)` that combines
          the head's outputs for downstream uses.

      NUMERICAL STABILITY — e.g., signature computation uses .std() on
          a short vector; clamps could guard against degenerate inputs.

    Important rules for your improvement:
      - target_method must name a method that exists on GraphEngine
        OR a new method name that does NOT clash.
      - impl_code MUST define a top-level function whose name ==
        target_method (first arg `self`). You MAY also define
        additional top-level helper functions; they will be visible
        in the test namespace as well.
      - impl must work on CPU (no CUDA assumptions).
      - Tests use `engine_factory()` which returns the shared engine
        instance (GraphEngine already loaded from the checkpoint).
      - Tests may use these building blocks:
            small_graph_json(): returns a 5-node test graph JSON string
            query_root_cause_json(target): returns a query JSON string
        plus any helper function you defined at top level in impl_code.
''')


IMPL_PROMPT = textwrap.dedent('''\
    Read the FULL source of GraphEngine below. Identify ONE concrete
    improvement and return a single ```json``` block:
        target_method: method name (existing or new)
        rationale:     explain WHAT the change does + what behavior it
                        guarantees that the original doesn't
        impl_code:     single top-level function, first arg `self`

    {catalog}

    {forbidden_section}

    === SOURCE of GraphEngine ===
    ```python
    {source}
    ```

    Output ONE ```json``` block. No test_code — that comes in a
    separate step.
''')


TEST_PROMPT = textwrap.dedent('''\
    A candidate has identified an improvement to `GraphEngine`. Your
    job: write a discriminating test that PASSES on the improved impl
    and FAILS on the original.

    CANDIDATE RATIONALE:
    {rationale}

    TARGET METHOD: {target_method}

    ORIGINAL METHOD SOURCE:
    ```python
    {original_src}
    ```

    CANDIDATE IMPL:
    ```python
    {impl_code}
    ```

    Write ONE top-level function:

        def test_improvement(engine_factory):
            # engine_factory() returns the loaded GraphEngine instance.
            # Build a concrete test input, call engine.target_method(...)
            # or the new method, assert the behavior claimed by the
            # rationale.
            ...

    Helpers available in the same namespace:
        small_graph_json()       → a stable 5-node graph JSON string
        query_root_cause_json(t) → '{{"type":"root_cause","target":t}}'

    The test must (1) FAIL (raise AssertionError) when run against the
    ORIGINAL unpatched method and (2) PASS when run against the
    candidate impl.

    Output ONE ```python``` block with only the function definition.
''')


# ----------------------------------------------------------------------
# Harness template
# ----------------------------------------------------------------------


HARNESS_TEMPLATE = r'''
import json, os, sys, typing, collections, math, itertools, heapq, functools, networkx
sys.path.insert(0, "/workspace/liquid-arc")
import torch
import torch.nn.functional as F

from liquid_arc.graph_engine_inference import GraphEngine

CHECKPOINT = os.environ["CHECKPOINT"]
print(f"[harness] loading GraphEngine from {CHECKPOINT}...", flush=True)
ENGINE = GraphEngine(CHECKPOINT, device="cpu", corrections_log=None)
print("[harness] engine loaded", flush=True)

def engine_factory():
    return ENGINE

def small_graph_json():
    return json.dumps({
        "nodes": [
            {"id": "cause", "type": "event", "role": "root"},
            {"id": "m1", "type": "state", "role": "intermediate"},
            {"id": "m2", "type": "state", "role": "intermediate"},
            {"id": "effect", "type": "consequence", "role": "terminal"},
            {"id": "side", "type": "entity", "role": "intermediate"},
        ],
        "edges": [
            {"src": "cause", "dst": "m1", "type": "causes"},
            {"src": "m1", "dst": "m2", "type": "causes"},
            {"src": "m2", "dst": "effect", "type": "causes"},
            {"src": "m1", "dst": "side", "type": "related_to"},
        ],
    })

def query_root_cause_json(target):
    return json.dumps({"type": "root_cause", "target": target})

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

_typing_names = ("List","Dict","Set","Tuple","Iterable","Optional","Any",
                  "Callable","Sequence","Union","Mapping")
_base_ns = {"nx": networkx, "networkx": networkx,
            "typing": typing, "collections": collections,
            "math": math, "itertools": itertools,
            "heapq": heapq, "functools": functools,
            "json": json, "os": os,
            "torch": torch, "F": F,
            "engine_factory": engine_factory,
            "small_graph_json": small_graph_json,
            "query_root_cause_json": query_root_cause_json,
            "GraphEngine": GraphEngine}
for _name in _typing_names:
    _base_ns[_name] = getattr(typing, _name)

if TARGET_METHOD and IMPL_CODE.strip() and TEST_CODE.strip():
    result["shape_ok"] = True

impl_ns = dict(_base_ns); impl_ns["__name__"] = "impl"
try:
    exec(IMPL_CODE, impl_ns)
    assert TARGET_METHOD in impl_ns, f"impl did not define {TARGET_METHOD}"
    result["impl_exec_ok"] = True
except Exception as e:
    result["errors"].append(f"impl_exec: {type(e).__name__}: {e}")

def _test_ns(name):
    ns = dict(_base_ns)
    ns["__name__"] = name
    for _k, _v in impl_ns.items():
        if _k.startswith("__") and _k.endswith("__"):
            continue
        ns.setdefault(_k, _v)
    return ns

if result["impl_exec_ok"]:
    test_ns = _test_ns("test_pre")
    try:
        exec(TEST_CODE, test_ns)
    except Exception as e:
        result["errors"].append(f"test_exec: {type(e).__name__}: {e}")
    fn = test_ns.get("test_improvement")
    if fn is None:
        result["errors"].append("test did not define test_improvement")
    else:
        failed = False
        try:
            fn(engine_factory)
        except AssertionError:
            failed = True
        except Exception:
            failed = True
        result["new_test_fails_on_original"] = failed
        if not failed:
            result["errors"].append("test passed on unpatched class — trivial")

if result["impl_exec_ok"]:
    original_method = getattr(GraphEngine, TARGET_METHOD, None)
    setattr(GraphEngine, TARGET_METHOD, impl_ns[TARGET_METHOD])

    test_ns2 = _test_ns("test_post")
    try:
        exec(TEST_CODE, test_ns2)
        fn = test_ns2.get("test_improvement")
        if fn is not None:
            fn(engine_factory)
            result["new_test_passes_on_improved"] = True
    except Exception as e:
        result["errors"].append(f"improved_test: {type(e).__name__}: {e}")

    # ---- regression battery ----
    try:
        g = small_graph_json()
        out_rc = json.loads(ENGINE.analyze_graph(g, query_root_cause_json("effect")))
        assert out_rc.get("root_cause") == "cause", f"root_cause regressed: {out_rc}"
        out_diag = json.loads(ENGINE.get_graph_diagnostics(g))
        assert out_diag.get("n_nodes") == 5, f"diagnostics regressed: {out_diag}"
        out_cmp = json.loads(ENGINE.compare_graphs(g, g))
        assert out_cmp.get("isomorphic") == True, f"compare_graphs regressed: {out_cmp}"
        result["regression_pass"] = True
    except Exception as e:
        result["errors"].append(f"regression: {type(e).__name__}: {e}")

    if original_method is not None:
        setattr(GraphEngine, TARGET_METHOD, original_method)
    else:
        try: delattr(GraphEngine, TARGET_METHOD)
        except Exception: pass

reward = (0.15 * (1 if result["shape_ok"] else 0)
          + 0.15 * (1 if result["impl_exec_ok"] else 0)
          + 0.25 * (1 if result["regression_pass"] else 0)
          + 0.30 * (1 if result["new_test_passes_on_improved"] else 0)
          + 0.15 * (1 if result["new_test_fails_on_original"] else 0))
result["reward"] = reward
print("RESULT_JSON: " + json.dumps(result, default=str))
'''


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def run_harness(code: str, path: Path, checkpoint: str,
                timeout_s: int = 240) -> Dict[str, Any]:
    with open(path, "w") as f:
        f.write(code)
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", "/workspace/liquid-arc")
    env["CHECKPOINT"] = checkpoint
    try:
        proc = subprocess.run(
            ["python", str(path)], env=env, cwd="/workspace/liquid-arc",
            capture_output=True, text=True, timeout=timeout_s)
        stdout, stderr, ec = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as exc:
        raw_out = exc.stdout if exc.stdout is not None else ""
        raw_err = exc.stderr if exc.stderr is not None else ""
        stdout = raw_out.decode() if isinstance(raw_out, bytes) else str(raw_out)
        stderr = (raw_err.decode() if isinstance(raw_err, bytes)
                  else str(raw_err)) + "\n[TIMEOUT]"
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


def extract_method_source(class_src: str, method_name: str) -> str:
    lines = class_src.splitlines()
    out = []
    inside = False
    indent: Optional[int] = None
    for ln in lines:
        if not inside and re.match(rf"\s{{4}}def\s+{re.escape(method_name)}\b",
                                    ln):
            inside = True
            indent = len(ln) - len(ln.lstrip())
            out.append(ln)
            continue
        if inside:
            if ln.strip() == "":
                out.append(ln)
                continue
            cur_indent = len(ln) - len(ln.lstrip())
            if cur_indent <= (indent or 0) and ln.strip():
                break
            out.append(ln)
    return ("\n".join(out) if out
            else f"# (method {method_name} not found in source)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vllm_url", default="http://172.17.0.1:30000/v1")
    p.add_argument("--model", default="Qwen3-Next-80B-A3B-Instruct-FP8")
    p.add_argument("--source_path", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_rounds", type=int, default=15)
    p.add_argument("--target_reward", type=float, default=0.90)
    p.add_argument("--patience", type=int, default=8)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_text = Path(args.source_path).read_text()

    trace: List[Dict[str, Any]] = []
    best_reward = -1.0
    best_package: Optional[Dict[str, Any]] = None
    forbidden: Set[str] = set()
    stagnation = 0

    for rnd in range(args.max_rounds):
        print(f"\n===== round {rnd+1}/{args.max_rounds} =====", flush=True)

        def _build_forbidden_section(extra_note: str = "") -> str:
            if not forbidden:
                return ""
            body = (
                "STRICT RULE — the following target_method names are "
                "BANNED this round. Any response using them will be "
                "REJECTED without evaluation. Pick a DIFFERENT method "
                "(existing OR new):\n  - "
                + "\n  - ".join(sorted(forbidden)) + "\n")
            if extra_note:
                body += "\n" + extra_note + "\n"
            return body

        impl_pkg: Optional[Dict[str, Any]] = None
        target = ""
        rationale = ""
        impl_code = ""
        gen_impl_s = 0.0
        max_forbid_retries = 3
        last_response_preview = ""
        retry_note = ""
        for attempt in range(max_forbid_retries):
            forbidden_section = _build_forbidden_section(retry_note)
            impl_messages = [
                {"role": "system",
                 "content": "You identify improvements in existing Python code. "
                            "Respond with one ```json``` block containing "
                            "target_method, rationale, impl_code. No test_code. "
                            "No other content."},
                {"role": "user",
                 "content": IMPL_PROMPT.format(
                     catalog=IMPROVEMENT_CATALOG,
                     source=source_text,
                     forbidden_section=forbidden_section)},
            ]
            t0 = time.time()
            try:
                response = llm_generate(args.vllm_url, args.model,
                                         impl_messages, max_tokens=7000,
                                         temperature=0.4 + 0.15 * attempt)
            except Exception as exc:
                trace.append({"round": rnd, "phase": "impl", "err": str(exc),
                               "attempt": attempt})
                response = ""
            gen_impl_s += time.time() - t0
            last_response_preview = response[:400]
            pkg_candidate = extract_json_block(response) if response else None
            if not pkg_candidate or not all(
                    k in pkg_candidate for k in ("target_method", "rationale",
                                                   "impl_code")):
                retry_note = ("Your previous reply did not contain a valid "
                              "```json``` block with target_method, rationale, "
                              "impl_code. Retry, obey the JSON contract.")
                continue
            cand_target = str(pkg_candidate["target_method"]).strip()
            if cand_target in forbidden:
                retry_note = (
                    f"REJECTED: target_method '{cand_target}' is in the "
                    f"banned list above. Choose a different one.")
                print(f"  [forbid-retry {attempt+1}] rejected target "
                      f"'{cand_target}' (banned)", flush=True)
                continue
            impl_pkg = pkg_candidate
            target = cand_target
            rationale = str(pkg_candidate["rationale"])
            impl_code = str(pkg_candidate["impl_code"])
            break

        if impl_pkg is None:
            print(f"  [no valid impl package after "
                  f"{max_forbid_retries} tries]  gen {gen_impl_s:.1f}s",
                  flush=True)
            trace.append({"round": rnd, "phase": "impl",
                           "err": "no package or all forbidden",
                           "preview": last_response_preview})
            stagnation += 1
            if stagnation >= args.patience: break
            continue

        print(f"  target={target}  (impl {gen_impl_s:.1f}s)", flush=True)

        # Phase B: test generation
        original_src = extract_method_source(source_text, target)
        test_messages = [
            {"role": "system",
             "content": "You write discriminating Python tests. Respond "
                        "with exactly ONE ```python``` block containing a "
                        "single function definition. No commentary."},
            {"role": "user",
             "content": TEST_PROMPT.format(rationale=rationale,
                                            target_method=target,
                                            original_src=original_src,
                                            impl_code=impl_code)},
        ]
        t1 = time.time()
        try:
            test_resp = llm_generate(args.vllm_url, args.model,
                                      test_messages, max_tokens=2500,
                                      temperature=0.3)
        except Exception as exc:
            print(f"  [llm test error] {exc}", flush=True)
            test_code = ""
        else:
            test_code = extract_python_block(test_resp) or ""
        gen_test_s = time.time() - t1
        if not test_code.strip():
            print(f"  [no test code] gen {gen_test_s:.1f}s", flush=True)
            stagnation += 1
            if stagnation >= args.patience: break
            continue
        print(f"  test generated ({gen_test_s:.1f}s)", flush=True)

        # Phase C: harness
        package = {"target_method": target, "rationale": rationale,
                    "impl_code": impl_code, "test_code": test_code}
        with open(out_dir / f"iter_{rnd:02d}_package.json", "w") as f:
            json.dump(package, f, indent=2)
        harness = (HARNESS_TEMPLATE
                   .replace("%%IMPL_CODE%%", impl_code)
                   .replace("%%TEST_CODE%%", test_code)
                   .replace("%%TARGET_METHOD%%", target))
        run = run_harness(harness, out_dir / f"iter_{rnd:02d}.py",
                           args.checkpoint, timeout_s=240)
        reward = 0.0
        result: Dict[str, Any] = {}
        if run["result_json"] is not None:
            result = run["result_json"]
            reward = result.get("reward", 0.0)

        trace.append({
            "round": rnd, "target_method": target,
            "rationale": rationale[:300],
            "gen_impl_s": gen_impl_s, "gen_test_s": gen_test_s,
            "reward": reward, "result": result,
            "stderr_tail": (run["stderr"] or "")[-400:],
        })

        print(f"  reward={reward:.3f}  "
              f"shape={result.get('shape_ok', False)}  "
              f"exec={result.get('impl_exec_ok', False)}  "
              f"regr={result.get('regression_pass', False)}  "
              f"test_pass={result.get('new_test_passes_on_improved', False)}  "
              f"test_fail_orig={result.get('new_test_fails_on_original', False)}",
              flush=True)
        if result.get("errors"):
            print(f"  errors: {result['errors'][:2]}", flush=True)

        if reward > best_reward + 1e-6:
            best_reward = reward
            best_package = package
            stagnation = 0
            with open(out_dir / "best_package.json", "w") as f:
                json.dump(package, f, indent=2)
            print(f"  [new best] reward={reward:.3f}", flush=True)
        else:
            stagnation += 1
            forbidden.add(target)

        if reward >= args.target_reward:
            print("  target hit — stopping", flush=True)
            break
        if stagnation >= args.patience:
            print(f"  patience {args.patience} exhausted — stopping",
                  flush=True)
            break

    summary = {
        "status": "success" if best_reward >= args.target_reward else "partial",
        "best_reward": best_reward,
        "best_package": best_package,
        "rounds": len(trace),
        "targets_attempted": sorted(forbidden | (
            {best_package["target_method"]} if best_package else set())),
        "trace": trace,
    }
    with open(out_dir / "report.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n=== STAGE-2d SUMMARY ===")
    print(f"  best reward: {best_reward:.3f}  status: {summary['status']}")
    print(f"  rounds: {len(trace)}")
    print(f"  targets attempted: {summary['targets_attempted']}")
    if best_package:
        print(f"  best target: {best_package['target_method']}")
    sys.exit(0 if summary["status"] == "success" else 1)


if __name__ == "__main__":
    main()
