"""Stage 2c — autonomous improvement with forbid-repeat + split-role.

Improvements over Stage 2:
  1. FORBID-REPEAT — every attempted target_method that didn't yield a
     reward > best is added to a 'forbidden' list the next prompt
     explicitly forbids. Forces the model off local minima (like
     `get_neighbors` in Stage 2b).
  2. SPLIT-ROLE TESTS — after Qwen produces (target, rationale, impl),
     a SEPARATE LLM call generates the test given only the rationale
     + original method source. The test-writer is blind to the impl,
     so the test can't leak impl-specific assumptions.
  3. LONGER RUN — max_rounds 20, patience 10.
  4. BETTER FEEDBACK — when test/impl disagree, describe the specific
     numeric gap and invite a redesign of one or the other.
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
    Improvement classes — pick ONE you can verify:
      PERFORMANCE — caching, incremental updates, avoiding O(V^2) scans
      CORRECTNESS — replacing order-dependent choices with principled ones,
                    handling edge cases, removing silent fallbacks
      EDGE-CASE    — deduplication, empty inputs, self-loops, isolates
      NEW CAPABILITY — a new method that composes existing primitives
    Important:
      - Pick a target_method you can reason about precisely.
      - Define an impl and a test that describe the SAME semantic change.
      - Test must FAIL on the original class (non-triviality) and PASS
        on the patched class (correctness).
      - impl must be a single top-level function whose first arg is `self`.
      - test_code function name: test_improvement(db_factory).
''')


IMPL_PROMPT = textwrap.dedent('''\
    Read the FULL source below. Identify ONE concrete improvement and
    return a single ```json``` block with these fields:
        target_method: method name you're improving or adding
        rationale:     short paragraph explaining WHAT the change does and
                        WHAT the new behavior guarantees vs the old
        impl_code:     single top-level function with first arg `self`

    {catalog}

    {forbidden_section}

    === SOURCE of KnowledgeGraphDB ===
    ```python
    {source}
    ```

    Output ONE ```json``` block. No test_code yet — we'll generate tests
    in a separate step.
''')


TEST_PROMPT = textwrap.dedent('''\
    A candidate has identified an improvement. Your job: write a
    discriminating test function that will PASS on the improved
    implementation and FAIL on the original.

    CANDIDATE RATIONALE:
    {rationale}

    TARGET METHOD: {target_method}

    ORIGINAL METHOD SOURCE (what your test must FAIL against):
    ```python
    {original_src}
    ```

    CANDIDATE IMPL (what your test must PASS against):
    ```python
    {impl_code}
    ```

    Write ONE top-level function:

        def test_improvement(db_factory):
            # construct a fresh KnowledgeGraphDB via db_factory()
            # build a concrete graph that distinguishes the impls
            # call target method; assert the behavior claimed by rationale
            ...

    Use `assert` statements with helpful messages. The test must:
      (1) FAIL (raise AssertionError) if the target_method is the
          original (unpatched) implementation.
      (2) PASS if the target_method is the candidate impl.
      (3) Not depend on implementation details beyond the behavioral
          claim in the rationale.

    Output ONE ```python``` block containing only the function definition.
''')


# ----------------------------------------------------------------------
# Harness template — same as Stage 2 but imports preloaded
# ----------------------------------------------------------------------

HARNESS_TEMPLATE = r'''
import json, os, sys, typing, collections, math, itertools, heapq, functools, networkx
sys.path.insert(0, "/workspace/liquid-arc")

from liquid_arc.graph_rag.decoupled.graph_db import KnowledgeGraphDB

def db_factory():
    path = f"/tmp/stage2c_{os.getpid()}_{id(object()):x}.json"
    if os.path.exists(path):
        os.remove(path)
    return KnowledgeGraphDB(path)

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
            "heapq": heapq, "functools": functools}
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
    result["errors"].append(f"impl_exec: {e}")

if result["impl_exec_ok"]:
    test_ns = dict(_base_ns); test_ns["__name__"] = "test_pre"
    try:
        exec(TEST_CODE, test_ns)
    except Exception as e:
        result["errors"].append(f"test_exec: {e}")
    fn = test_ns.get("test_improvement")
    if fn is None:
        result["errors"].append("test did not define test_improvement")
    else:
        failed = False
        try:
            fn(db_factory)
        except AssertionError:
            failed = True
        except Exception:
            failed = True
        result["new_test_fails_on_original"] = failed
        if not failed:
            result["errors"].append("test passed on unpatched class — trivial")

if result["impl_exec_ok"]:
    original_method = getattr(KnowledgeGraphDB, TARGET_METHOD, None)
    setattr(KnowledgeGraphDB, TARGET_METHOD, impl_ns[TARGET_METHOD])
    test_ns2 = dict(_base_ns); test_ns2["__name__"] = "test_post"
    try:
        exec(TEST_CODE, test_ns2)
        fn = test_ns2.get("test_improvement")
        if fn is not None:
            fn(db_factory)
            result["new_test_passes_on_improved"] = True
    except Exception as e:
        result["errors"].append(f"improved_test: {type(e).__name__}: {e}")

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
        filt = db.scope_filter("prod")
        assert filt.has_edge("mid1", "mid2")
        neigh = db.get_neighbors(["root"], hops=2)
        assert "mid1" in neigh and "mid2" in neigh
        _ = db.find_communities(min_size=2)
        db.clear()
        result["regression_pass"] = True
    except Exception as e:
        result["errors"].append(f"regression: {type(e).__name__}: {e}")

    if original_method is not None:
        setattr(KnowledgeGraphDB, TARGET_METHOD, original_method)
    else:
        try: delattr(KnowledgeGraphDB, TARGET_METHOD)
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


def run_harness(code: str, path: Path, timeout_s: int = 120) -> Dict[str, Any]:
    with open(path, "w") as f:
        f.write(code)
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", "/workspace/liquid-arc")
    try:
        proc = subprocess.run(
            ["python", str(path)], env=env, cwd="/workspace/liquid-arc",
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


def extract_method_source(class_src: str, method_name: str) -> str:
    """Best-effort extraction of a single method's source from the class."""
    lines = class_src.splitlines()
    out = []
    inside = False
    indent = None
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
    return "\n".join(out) if out else f"# (method {method_name} not found in source)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vllm_url", default="http://172.17.0.1:30000/v1")
    p.add_argument("--model", default="Qwen3-Next-80B-A3B-Instruct-FP8")
    p.add_argument("--source_path", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_rounds", type=int, default=20)
    p.add_argument("--target_reward", type=float, default=0.90)
    p.add_argument("--patience", type=int, default=10)
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

        # --- Phase A: impl proposal (Qwen picks target + rationale + impl) ---
        forbidden_section = ""
        if forbidden:
            forbidden_section = (
                "Targets already attempted that did not succeed — you MUST "
                "choose a DIFFERENT target_method this round:\n  "
                + "\n  ".join(sorted(forbidden)) + "\n")
        impl_messages = [
            {"role": "system",
             "content": "You identify and implement improvements to existing "
                        "Python code. Respond with one ```json``` block "
                        "containing target_method, rationale, impl_code. "
                        "No test_code. No other content."},
            {"role": "user",
             "content": IMPL_PROMPT.format(catalog=IMPROVEMENT_CATALOG,
                                            source=source_text,
                                            forbidden_section=forbidden_section)},
        ]
        t0 = time.time()
        try:
            response = llm_generate(args.vllm_url, args.model,
                                     impl_messages, max_tokens=6000,
                                     temperature=0.4)
        except Exception as exc:
            print(f"  [llm impl error] {exc}", flush=True)
            trace.append({"round": rnd, "phase": "impl", "err": str(exc)})
            stagnation += 1
            if stagnation >= args.patience: break
            continue
        gen_impl_s = time.time() - t0
        impl_package = extract_json_block(response)
        if not impl_package or not all(
                k in impl_package for k in ("target_method", "rationale",
                                              "impl_code")):
            print(f"  [no impl package] gen {gen_impl_s:.1f}s", flush=True)
            trace.append({"round": rnd, "phase": "impl",
                           "err": "no package",
                           "preview": response[:400]})
            stagnation += 1
            if stagnation >= args.patience: break
            continue

        target = str(impl_package["target_method"]).strip()
        rationale = str(impl_package["rationale"])
        impl_code = str(impl_package["impl_code"])

        if target in forbidden:
            print(f"  [re-picked forbidden target {target}] — adding harder "
                  f"forbid-notice", flush=True)
        print(f"  target={target}  (impl {gen_impl_s:.1f}s)", flush=True)

        # --- Phase B: test generation (blind to impl) ---
        original_src = extract_method_source(source_text, target)
        test_messages = [
            {"role": "system",
             "content": "You write discriminating Python tests. "
                        "Respond with exactly ONE ```python``` block "
                        "containing a single function definition. "
                        "No commentary."},
            {"role": "user",
             "content": TEST_PROMPT.format(rationale=rationale,
                                            target_method=target,
                                            original_src=original_src,
                                            impl_code=impl_code)},
        ]
        t1 = time.time()
        try:
            test_response = llm_generate(args.vllm_url, args.model,
                                          test_messages, max_tokens=2500,
                                          temperature=0.3)
        except Exception as exc:
            print(f"  [llm test error] {exc}", flush=True)
            test_code = ""
        else:
            test_code = extract_python_block(test_response) or ""
        gen_test_s = time.time() - t1
        if not test_code.strip():
            print(f"  [no test code] gen {gen_test_s:.1f}s", flush=True)
            stagnation += 1
            if stagnation >= args.patience: break
            continue
        print(f"  test generated ({gen_test_s:.1f}s)", flush=True)

        # --- Phase C: harness ---
        package = {"target_method": target, "rationale": rationale,
                    "impl_code": impl_code, "test_code": test_code}
        with open(out_dir / f"iter_{rnd:02d}_package.json", "w") as f:
            json.dump(package, f, indent=2)
        harness = (HARNESS_TEMPLATE
                   .replace("%%IMPL_CODE%%", impl_code)
                   .replace("%%TEST_CODE%%", test_code)
                   .replace("%%TARGET_METHOD%%", target))
        run = run_harness(harness, out_dir / f"iter_{rnd:02d}.py",
                           timeout_s=120)
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
            # Add the target to forbidden list — we want diversity
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
    print(f"\n=== STAGE-2c SUMMARY ===")
    print(f"  best reward: {best_reward:.3f}  status: {summary['status']}")
    print(f"  rounds: {len(trace)}")
    print(f"  targets attempted: {summary['targets_attempted']}")
    if best_package:
        print(f"  best target: {best_package['target_method']}")
    sys.exit(0 if summary["status"] == "success" else 1)


if __name__ == "__main__":
    main()
