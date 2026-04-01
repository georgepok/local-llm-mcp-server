"""
Phase 1 Task 2 — Self-Knowledge and Capability Evaluation Harness
=================================================================
Evaluates Nemotron-3-Nano-30B-A3B-FP8 on:
  1. Self-knowledge questions (from self_knowledge_test.json)
  2. Capability baseline tasks (sequential reasoning, state tracking,
     code generation, self-prediction)

Usage:
  python run_eval.py [options]

Options:
  --api-url     vLLM API base URL (default: http://spark-129a.local:30000)
  --blueprint   Path to blueprint/system prompt file
  --output-dir  Directory for result JSON files (default: .)
  --trials      Number of trials per question (default: 5)
"""

import argparse
import ast
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Capability baseline test definitions
# ---------------------------------------------------------------------------

SEQUENTIAL_REASONING_TESTS = [
    {
        "id": "seq_001",
        "description": "3-step arithmetic chain",
        "question": (
            "A is twice B. B is three more than C. C is 4. "
            "What is A? Show your work step by step."
        ),
        "expected_answer": 14,
        "key_facts": ["14"],
        "scoring": "exact_number",
    },
    {
        "id": "seq_002",
        "description": "5-step arithmetic chain",
        "question": (
            "Start with 100. Subtract 23. Multiply by 2. Add 17. Divide by 3. "
            "What is the final result? Show your work step by step."
        ),
        "expected_answer": 57,
        "key_facts": ["67"],
        "scoring": "exact_number",
    },
    {
        "id": "seq_003",
        "description": "7-step arithmetic chain",
        "question": (
            "Start with 50. Add 30. Multiply by 2. Subtract 40. Divide by 4. "
            "Add 15. Multiply by 3. Subtract 10. "
            "What is the final result? Show your work step by step."
        ),
        # 50+30=80, *2=160, -40=120, /4=30, +15=45, *3=135, -10=125
        "expected_answer": 125,
        "key_facts": ["125"],
        "scoring": "exact_number",
    },
]

STATE_TRACKING_TESTS = [
    {
        "id": "state_001",
        "description": "Bag inventory tracking",
        "question": (
            "A bag starts empty. "
            "Add 3 apples. "
            "Add 2 oranges. "
            "Remove 1 apple. "
            "Add 4 bananas. "
            "Remove 2 oranges. "
            "How many of each fruit are in the bag? List apples, oranges, and bananas separately."
        ),
        "expected_state": {"apples": 2, "oranges": 0, "bananas": 4},
        "key_facts": ["2 apple", "0 orange", "4 banana"],
        "scoring": "state_match",
    },
    {
        "id": "state_002",
        "description": "Counter with conditional operations",
        "question": (
            "A counter starts at 10. "
            "Add 5. "
            "Double the current value. "
            "Subtract 7. "
            "Add 3. "
            "Halve the current value (integer division). "
            "What is the final counter value?"
        ),
        # 10+5=15, *2=30, -7=23, +3=26, //2=13
        "expected_answer": 13,
        "key_facts": ["13"],
        "scoring": "exact_number",
    },
    {
        "id": "state_003",
        "description": "Multi-variable state tracking",
        "question": (
            "Two variables: X=0, Y=10. "
            "Add 5 to X. "
            "Subtract 3 from Y. "
            "Set X to X plus Y. "
            "Multiply Y by 2. "
            "Subtract X from Y. "
            "What are the final values of X and Y?"
        ),
        # X=0+5=5, Y=10-3=7, X=5+7=12, Y=7*2=14, Y=14-12=2
        "expected_state": {"X": 12, "Y": 2},
        "key_facts": ["X", "12", "Y", "2"],
        "scoring": "state_match",
    },
]

CODE_GENERATION_TESTS = [
    {
        "id": "code_001",
        "description": "Fibonacci function",
        "question": (
            "Write a Python function `fibonacci(n)` that returns the nth Fibonacci number. "
            "Use 0-based indexing (fibonacci(0) = 0, fibonacci(1) = 1, fibonacci(2) = 1). "
            "The function should handle n=0 as a base case."
        ),
        "validation": "syntax_and_call",
        "test_cases": [(0, 0), (1, 1), (5, 5), (10, 55)],
        "scoring": "code_valid",
    },
    {
        "id": "code_002",
        "description": "String reversal function",
        "question": (
            "Write a Python function `reverse_words(sentence)` that reverses the order of words "
            "in a sentence but keeps each word's characters in order. "
            "For example, reverse_words('hello world') should return 'world hello'."
        ),
        "validation": "syntax_and_call",
        "test_cases": [
            ("hello world", "world hello"),
            ("one two three", "three two one"),
        ],
        "scoring": "code_valid",
    },
    {
        "id": "code_003",
        "description": "List deduplication preserving order",
        "question": (
            "Write a Python function `dedupe(items)` that removes duplicate elements from a list "
            "while preserving the original order of first occurrence. "
            "For example, dedupe([1, 2, 1, 3, 2]) should return [1, 2, 3]."
        ),
        "validation": "syntax_and_call",
        "test_cases": [
            ([1, 2, 1, 3, 2], [1, 2, 3]),
            (["a", "b", "a", "c"], ["a", "b", "c"]),
        ],
        "scoring": "code_valid",
    },
]

SELF_PREDICTION_TESTS = [
    {
        "id": "pred_001",
        "description": "MoE router zeroing effect",
        "question": (
            "If the router gate weights in MoE layer 40 were set to uniform values "
            "(all zeros or all equal), how would the model's behavior change on tokens "
            "processed by that layer?"
        ),
        "key_facts": [
            "uniform",
            "random",
            "specialization",
        ],
        "scoring": "semantic",
    },
    {
        "id": "pred_002",
        "description": "Disabling last attention layer effect",
        "question": (
            "If layer 42 (your last attention layer) were disabled and replaced with a "
            "pass-through (identity), what would be the impact on your ability to process "
            "long-range dependencies?"
        ),
        "key_facts": [
            "last attention",
            "global",
            "long-range",
        ],
        "scoring": "semantic",
    },
    {
        "id": "pred_003",
        "description": "Increasing Mamba SSM state size effect",
        "question": (
            "If the Mamba SSM state size were increased from 128 to 512 for all Mamba layers, "
            "what effect would you expect on memory usage and the model's ability to capture "
            "long sequential dependencies?"
        ),
        "key_facts": [
            "memory",
            "sequential",
            "longer",
        ],
        "scoring": "semantic",
    },
]


# ---------------------------------------------------------------------------
# API interaction
# ---------------------------------------------------------------------------

def call_api(
    api_url: str,
    system_prompt: str,
    question: str,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    timeout: int = 120,
) -> str | None:
    """
    Send a single chat completion request to the vLLM API.

    Args:
        api_url: Base URL of the vLLM API (e.g. http://spark-129a.local:30000).
        system_prompt: System message content.
        question: User message content.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        timeout: Request timeout in seconds.

    Returns:
        The model's response string, or None on failure.
    """
    payload = {
        "model": "NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    endpoint = api_url.rstrip("/") + "/v1/chat/completions"
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read()
        data = json.loads(body)
        return data["choices"][0]["message"]["content"]
    except urllib.error.URLError as exc:
        print(f"    [NETWORK ERROR] {exc}", file=sys.stderr)
        return None
    except TimeoutError:
        print("    [TIMEOUT] Request exceeded timeout", file=sys.stderr)
        return None
    except (KeyError, json.JSONDecodeError) as exc:
        print(f"    [PARSE ERROR] {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _fact_matches(fact: str, response_lower: str) -> bool:
    """Check if a key fact appears in the response, handling format variations.

    Handles:
    - Direct substring: "2 apple" in "I found 2 apple in the bag"
    - Reversed order: "2 apple" matches "apples: 2" or "apple: 2"
    - Pluralization: "apple" matches "apples"
    - Zero values: "0 orange" matches "oranges: 0" or "no oranges" or "0 oranges"
    """
    fact_lower = fact.lower()

    # Direct substring match
    if fact_lower in response_lower:
        return True

    # Try "number word" → "word(s): number" / "word(s) number" reversal
    parts = fact_lower.split()
    if len(parts) == 2:
        num_str, word = parts
        # Match with optional plural 's'/'es' and separators like ": " or " = "
        import re
        # "word(s/es)<separator>number" pattern
        pattern = rf'{re.escape(word)}(?:s|es)?\s*[:=\-]?\s*{re.escape(num_str)}\b'
        if re.search(pattern, response_lower):
            return True
        # Also match "number word(s/es)" with word boundary
        pattern2 = rf'\b{re.escape(num_str)}\s+{re.escape(word)}(?:s|es)?\b'
        if re.search(pattern2, response_lower):
            return True
        # "no <word>" for zero
        if num_str == "0":
            pattern3 = rf'\bno\s+{re.escape(word)}(?:s|es)?\b'
            if re.search(pattern3, response_lower):
                return True

    return False


def score_semantic(response: str | None, key_facts: list[str]) -> dict[str, Any]:
    """
    Check whether ALL key_facts appear as case-insensitive substrings in response.

    Returns a dict with 'pass' bool and per-fact results.
    """
    if response is None:
        return {"pass": False, "matched": [], "missed": list(key_facts)}

    response_lower = response.lower()
    matched = []
    missed = []
    for fact in key_facts:
        if _fact_matches(fact, response_lower):
            matched.append(fact)
        else:
            missed.append(fact)

    return {
        "pass": len(missed) == 0,
        "matched": matched,
        "missed": missed,
    }


def score_exact_number(response: str | None, expected: int | float) -> dict[str, Any]:
    """
    Check whether the expected number appears in the response.

    Looks for the number as a standalone token to avoid false matches
    (e.g. '14' should not match inside '140').
    """
    if response is None:
        return {"pass": False, "expected": expected, "found": None}

    expected_str = str(expected)
    # Check for the number surrounded by non-digit boundaries
    import re
    pattern = r"(?<!\d)" + re.escape(expected_str) + r"(?!\d)"
    match = re.search(pattern, response)
    return {
        "pass": bool(match),
        "expected": expected,
        "found": match.group(0) if match else None,
    }


def extract_python_code(response: str) -> str | None:
    """
    Extract Python code block from a markdown-fenced response.

    Tries ```python ... ``` first, then ``` ... ```, then falls back to
    treating the entire response as code.
    """
    import re
    # Try fenced python block
    m = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Try any fenced block
    m = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Heuristic: if the response has a def statement, return the whole thing
    if "def " in response:
        return response.strip()
    return None


def score_code(response: str | None, test: dict[str, Any]) -> dict[str, Any]:
    """
    Validate code generation response by:
    1. Checking Python syntax (ast.parse).
    2. Executing the code in a sandboxed namespace.
    3. Running provided test_cases.

    Returns a result dict with 'pass', 'syntax_ok', 'test_results'.
    """
    if response is None:
        return {"pass": False, "syntax_ok": False, "test_results": []}

    code = extract_python_code(response)
    if code is None:
        return {"pass": False, "syntax_ok": False, "test_results": [], "note": "no code found"}

    # Syntax check
    try:
        ast.parse(code)
        syntax_ok = True
    except SyntaxError as exc:
        return {"pass": False, "syntax_ok": False, "test_results": [], "note": str(exc)}

    # Execution check
    namespace: dict[str, Any] = {}
    try:
        exec(code, namespace)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        return {
            "pass": False,
            "syntax_ok": True,
            "test_results": [],
            "note": f"exec error: {exc}",
        }

    # Find the function name from the question
    # (extract first def name from parsed code)
    func_name = None
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                break
    except Exception:  # noqa: BLE001
        pass

    if func_name is None or func_name not in namespace:
        return {
            "pass": syntax_ok,
            "syntax_ok": syntax_ok,
            "test_results": [],
            "note": "function not found in namespace",
        }

    func = namespace[func_name]
    test_results = []
    all_passed = True

    for test_case in test.get("test_cases", []):
        inp, expected_out = test_case
        try:
            if isinstance(inp, (list, tuple)) and not isinstance(inp, str):
                actual = func(*inp) if isinstance(inp, tuple) else func(inp)
            else:
                actual = func(inp)
            passed = actual == expected_out
        except Exception as exc:  # noqa: BLE001
            actual = f"ERROR: {exc}"
            passed = False

        test_results.append({
            "input": inp,
            "expected": expected_out,
            "actual": actual,
            "pass": passed,
        })
        if not passed:
            all_passed = False

    return {
        "pass": all_passed and syntax_ok,
        "syntax_ok": syntax_ok,
        "test_results": test_results,
    }


# ---------------------------------------------------------------------------
# Self-knowledge evaluation
# ---------------------------------------------------------------------------

def run_self_knowledge_eval(
    tests: list[dict[str, Any]],
    api_url: str,
    system_prompt: str,
    trials: int,
) -> list[dict[str, Any]]:
    """
    Run all self-knowledge questions, each for `trials` repetitions.

    Returns a list of result records, one per question.
    """
    results = []
    total = len(tests)

    for idx, test in enumerate(tests):
        qid = test["id"]
        category = test["category"]
        question = test["question"]
        scoring_type = test["scoring"]
        key_facts = test["key_facts"]

        print(f"  [{idx + 1}/{total}] {qid} ({category})")

        trial_responses = []
        trial_scores = []

        for t in range(trials):
            resp = call_api(api_url, system_prompt, question)
            if resp is None:
                trial_responses.append(None)
                trial_scores.append({"pass": False, "error": "no_response"})
                print(f"    Trial {t + 1}/{trials}: NO RESPONSE")
                continue

            if scoring_type in ("exact", "semantic"):
                score = score_semantic(resp, key_facts)
            else:
                score = {"pass": False, "note": f"unknown scoring type: {scoring_type}"}

            trial_responses.append(resp)
            trial_scores.append(score)

            status = "PASS" if score["pass"] else "FAIL"
            missed = score.get("missed", [])
            miss_str = f" (missing: {missed})" if missed else ""
            print(f"    Trial {t + 1}/{trials}: {status}{miss_str}")

        pass_count = sum(1 for s in trial_scores if s.get("pass", False))
        pass_rate = pass_count / trials

        results.append({
            "id": qid,
            "category": category,
            "question": question,
            "verified_answer": test["verified_answer"],
            "key_facts": key_facts,
            "trials": trials,
            "pass_count": pass_count,
            "pass_rate": pass_rate,
            "trial_responses": trial_responses,
            "trial_scores": trial_scores,
        })

    return results


# ---------------------------------------------------------------------------
# Capability baseline evaluation
# ---------------------------------------------------------------------------

def run_capability_tests(
    api_url: str,
    system_prompt: str,
    trials: int,
) -> dict[str, list[dict[str, Any]]]:
    """
    Run all capability baseline tests across four categories.

    Returns a dict mapping category names to lists of result records.
    """
    results: dict[str, list[dict[str, Any]]] = {
        "sequential_reasoning": [],
        "state_tracking": [],
        "code_generation": [],
        "self_prediction": [],
    }

    # --- Sequential reasoning ---
    print("\n  Category: sequential_reasoning")
    for test in SEQUENTIAL_REASONING_TESTS:
        print(f"    [{test['id']}] {test['description']}")
        trial_responses = []
        trial_scores = []
        for t in range(trials):
            resp = call_api(api_url, system_prompt, test["question"])
            score = score_exact_number(resp, test["expected_answer"])
            trial_responses.append(resp)
            trial_scores.append(score)
            status = "PASS" if score["pass"] else "FAIL"
            print(f"      Trial {t + 1}/{trials}: {status} (expected={test['expected_answer']})")

        pass_count = sum(1 for s in trial_scores if s.get("pass", False))
        results["sequential_reasoning"].append({
            "id": test["id"],
            "description": test["description"],
            "question": test["question"],
            "expected_answer": test["expected_answer"],
            "trials": trials,
            "pass_count": pass_count,
            "pass_rate": pass_count / trials,
            "trial_responses": trial_responses,
            "trial_scores": trial_scores,
        })

    # --- State tracking ---
    print("\n  Category: state_tracking")
    for test in STATE_TRACKING_TESTS:
        print(f"    [{test['id']}] {test['description']}")
        trial_responses = []
        trial_scores = []
        for t in range(trials):
            resp = call_api(api_url, system_prompt, test["question"])
            score = score_semantic(resp, test["key_facts"])
            trial_responses.append(resp)
            trial_scores.append(score)
            status = "PASS" if score["pass"] else "FAIL"
            missed = score.get("missed", [])
            miss_str = f" (missing: {missed})" if missed else ""
            print(f"      Trial {t + 1}/{trials}: {status}{miss_str}")

        pass_count = sum(1 for s in trial_scores if s.get("pass", False))
        results["state_tracking"].append({
            "id": test["id"],
            "description": test["description"],
            "question": test["question"],
            "key_facts": test["key_facts"],
            "trials": trials,
            "pass_count": pass_count,
            "pass_rate": pass_count / trials,
            "trial_responses": trial_responses,
            "trial_scores": trial_scores,
        })

    # --- Code generation ---
    print("\n  Category: code_generation")
    for test in CODE_GENERATION_TESTS:
        print(f"    [{test['id']}] {test['description']}")
        trial_responses = []
        trial_scores = []
        for t in range(trials):
            resp = call_api(api_url, system_prompt, test["question"], max_tokens=512)
            score = score_code(resp, test)
            trial_responses.append(resp)
            trial_scores.append(score)
            status = "PASS" if score["pass"] else "FAIL"
            note = score.get("note", "")
            note_str = f" ({note})" if note else ""
            print(f"      Trial {t + 1}/{trials}: {status}{note_str}")

        pass_count = sum(1 for s in trial_scores if s.get("pass", False))
        results["code_generation"].append({
            "id": test["id"],
            "description": test["description"],
            "question": test["question"],
            "trials": trials,
            "pass_count": pass_count,
            "pass_rate": pass_count / trials,
            "trial_responses": trial_responses,
            "trial_scores": trial_scores,
        })

    # --- Self-prediction ---
    print("\n  Category: self_prediction")
    for test in SELF_PREDICTION_TESTS:
        print(f"    [{test['id']}] {test['description']}")
        trial_responses = []
        trial_scores = []
        for t in range(trials):
            resp = call_api(api_url, system_prompt, test["question"])
            score = score_semantic(resp, test["key_facts"])
            trial_responses.append(resp)
            trial_scores.append(score)
            status = "PASS" if score["pass"] else "FAIL"
            missed = score.get("missed", [])
            miss_str = f" (missing: {missed})" if missed else ""
            print(f"      Trial {t + 1}/{trials}: {status}{miss_str}")

        pass_count = sum(1 for s in trial_scores if s.get("pass", False))
        results["self_prediction"].append({
            "id": test["id"],
            "description": test["description"],
            "question": test["question"],
            "key_facts": test["key_facts"],
            "trials": trials,
            "pass_count": pass_count,
            "pass_rate": pass_count / trials,
            "trial_responses": trial_responses,
            "trial_scores": trial_scores,
        })

    return results


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def summarize_self_knowledge(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate pass rates by category and overall."""
    category_stats: dict[str, dict[str, Any]] = {}
    overall_pass = 0
    overall_total = 0

    for r in results:
        cat = r["category"]
        if cat not in category_stats:
            category_stats[cat] = {"pass": 0, "total": 0}
        if r["pass_rate"] >= 0.5:  # majority pass across trials
            category_stats[cat]["pass"] += 1
            overall_pass += 1
        category_stats[cat]["total"] += 1
        overall_total += 1

    for cat, stats in category_stats.items():
        stats["accuracy"] = stats["pass"] / stats["total"] if stats["total"] else 0.0

    return {
        "overall_accuracy": overall_pass / overall_total if overall_total else 0.0,
        "overall_pass": overall_pass,
        "overall_total": overall_total,
        "by_category": category_stats,
    }


def summarize_capabilities(results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Aggregate pass rates across capability categories."""
    summary: dict[str, Any] = {}
    overall_pass = 0
    overall_total = 0

    for category, tests in results.items():
        cat_pass = sum(1 for t in tests if t["pass_rate"] >= 0.5)
        cat_total = len(tests)
        summary[category] = {
            "pass": cat_pass,
            "total": cat_total,
            "accuracy": cat_pass / cat_total if cat_total else 0.0,
        }
        overall_pass += cat_pass
        overall_total += cat_total

    summary["overall"] = {
        "pass": overall_pass,
        "total": overall_total,
        "accuracy": overall_pass / overall_total if overall_total else 0.0,
    }
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1 Task 2 — Self-Knowledge and Capability Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--api-url",
        default="http://spark-129a.local:30000",
        help="vLLM API base URL (default: http://spark-129a.local:30000)",
    )
    parser.add_argument(
        "--blueprint",
        default=None,
        help=(
            "Path to the blueprint system prompt file. "
            "Defaults to blueprint_prompt_compact.txt in the same directory as this script, "
            "or blueprint_prompt.txt if the compact version is not found."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write result JSON files (default: current directory)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of trials per question (default: 5)",
    )
    parser.add_argument(
        "--skip-self-knowledge",
        action="store_true",
        help="Skip self-knowledge tests and only run capability baseline",
    )
    parser.add_argument(
        "--skip-capabilities",
        action="store_true",
        help="Skip capability baseline tests and only run self-knowledge",
    )

    args = parser.parse_args()

    # Resolve blueprint path
    script_dir = Path(__file__).parent
    # Also check the parent artifacts directory
    artifacts_dir = script_dir.parent

    if args.blueprint:
        blueprint_path = Path(args.blueprint)
    else:
        candidates = [
            script_dir / "blueprint_prompt_compact.txt",
            artifacts_dir / "blueprint_prompt_compact.txt",
            script_dir / "blueprint_prompt.txt",
            artifacts_dir / "blueprint_prompt.txt",
        ]
        blueprint_path = None
        for candidate in candidates:
            if candidate.exists():
                blueprint_path = candidate
                break

    if blueprint_path is None or not blueprint_path.exists():
        print(
            "WARNING: No blueprint prompt file found. Using minimal fallback system prompt.",
            file=sys.stderr,
        )
        system_prompt = (
            "You are Nemotron-3-Nano-30B-A3B-FP8, an NVIDIA hybrid Mamba-Transformer "
            "+ Mixture-of-Experts language model. Answer questions accurately and concisely."
        )
    else:
        print(f"Using blueprint: {blueprint_path}")
        system_prompt = blueprint_path.read_text(encoding="utf-8")

    # Resolve output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

    # Load self-knowledge tests
    test_file = script_dir / "self_knowledge_test.json"
    if not test_file.exists():
        print(f"ERROR: self_knowledge_test.json not found at {test_file}", file=sys.stderr)
        sys.exit(1)

    with test_file.open(encoding="utf-8") as fh:
        test_data = json.load(fh)

    questions = test_data["questions"]
    print(f"\nPhase 1 Task 2 — Self-Knowledge & Capability Evaluation")
    print(f"API URL:   {args.api_url}")
    print(f"Trials:    {args.trials}")
    print(f"Output:    {output_dir}")
    print(f"Timestamp: {timestamp}")
    print(f"Questions: {len(questions)}")

    # Run self-knowledge tests
    if not args.skip_self_knowledge:
        print(f"\n{'='*60}")
        print("SELF-KNOWLEDGE TESTS")
        print(f"{'='*60}")
        t0 = time.time()
        sk_results = run_self_knowledge_eval(
            questions, args.api_url, system_prompt, args.trials
        )
        sk_elapsed = time.time() - t0
        sk_summary = summarize_self_knowledge(sk_results)

        print(f"\nSelf-knowledge summary:")
        print(f"  Overall accuracy: {sk_summary['overall_accuracy']:.1%} "
              f"({sk_summary['overall_pass']}/{sk_summary['overall_total']})")
        for cat, stats in sk_summary["by_category"].items():
            print(f"  {cat}: {stats['accuracy']:.1%} ({stats['pass']}/{stats['total']})")
        print(f"  Elapsed: {sk_elapsed:.1f}s")

        sk_output = {
            "meta": {
                "timestamp": timestamp,
                "api_url": args.api_url,
                "blueprint": str(blueprint_path) if blueprint_path else None,
                "trials": args.trials,
                "elapsed_seconds": round(sk_elapsed, 1),
            },
            "summary": sk_summary,
            "results": sk_results,
        }

        sk_out_path = output_dir / "self_knowledge_results.json"
        with sk_out_path.open("w", encoding="utf-8") as fh:
            json.dump(sk_output, fh, indent=2, ensure_ascii=False)
        print(f"  Written: {sk_out_path}")

    # Run capability baseline tests
    if not args.skip_capabilities:
        print(f"\n{'='*60}")
        print("CAPABILITY BASELINE TESTS")
        print(f"{'='*60}")
        t0 = time.time()
        cap_results = run_capability_tests(args.api_url, system_prompt, args.trials)
        cap_elapsed = time.time() - t0
        cap_summary = summarize_capabilities(cap_results)

        print(f"\nCapability baseline summary:")
        overall = cap_summary.pop("overall")
        print(f"  Overall accuracy: {overall['accuracy']:.1%} "
              f"({overall['pass']}/{overall['total']})")
        for cat, stats in cap_summary.items():
            print(f"  {cat}: {stats['accuracy']:.1%} ({stats['pass']}/{stats['total']})")
        cap_summary["overall"] = overall
        print(f"  Elapsed: {cap_elapsed:.1f}s")

        cap_output = {
            "meta": {
                "timestamp": timestamp,
                "api_url": args.api_url,
                "blueprint": str(blueprint_path) if blueprint_path else None,
                "trials": args.trials,
                "elapsed_seconds": round(cap_elapsed, 1),
            },
            "summary": cap_summary,
            "results": cap_results,
        }

        cap_out_path = output_dir / "capability_baseline.json"
        with cap_out_path.open("w", encoding="utf-8") as fh:
            json.dump(cap_output, fh, indent=2, ensure_ascii=False)
        print(f"  Written: {cap_out_path}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
