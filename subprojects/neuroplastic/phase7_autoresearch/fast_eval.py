#!/usr/bin/env python3
"""Fast eval probe for autoresearch loop.

Runs 20 fixed state-tracking problems at temperature=0 with thinking disabled.
Returns score 0-20 (integer count of correct answers).
"""

import json
import re
import time
import urllib.request

MODEL_NAME = "NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"


def _post(url: str, payload: dict, timeout: int = 60) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def check_key_facts(response: str, key_facts: list[str]) -> bool:
    """Check if response contains all key facts.

    Handles formats like:
    - "apples: 2" or "Apples: 2"
    - "2 apples" or "2 apple"
    - Just the number if key_fact is a bare number
    """
    resp_lower = response.lower()
    for fact in key_facts:
        fact_lower = fact.lower().strip()

        # Parse fact into components
        parts = fact_lower.split()
        if len(parts) == 1:
            # Bare number — just check it appears
            # But be careful: "2" shouldn't match "12" or "200"
            num = parts[0]
            # Check for word boundary match
            pattern = r'(?<!\d)' + re.escape(num) + r'(?!\d)'
            if not re.search(pattern, resp_lower):
                return False
        elif len(parts) == 2:
            num, word = parts
            # Allow "N word", "word: N", "word = N", "word N"
            # Strip trailing s for plural
            word_base = word.rstrip('s')
            pattern_forward = r'(?<!\d)' + re.escape(num) + r'(?!\d).*?' + re.escape(word_base)
            pattern_reverse = re.escape(word_base) + r'.*?(?<!\d)' + re.escape(num) + r'(?!\d)'
            if not (re.search(pattern_forward, resp_lower) or
                    re.search(pattern_reverse, resp_lower)):
                # Also try just checking both exist within ~50 chars of each other
                num_positions = [m.start() for m in re.finditer(r'(?<!\d)' + re.escape(num) + r'(?!\d)', resp_lower)]
                word_positions = [m.start() for m in re.finditer(re.escape(word_base), resp_lower)]
                found = False
                for np in num_positions:
                    for wp in word_positions:
                        if abs(np - wp) < 80:
                            found = True
                            break
                    if found:
                        break
                if not found:
                    return False
        else:
            # Multi-word fact — just check substring
            if fact_lower not in resp_lower:
                return False
    return True


def run_fast_eval(api_url: str, problems_path: str = "eval_problems.json",
                  verbose: bool = False) -> dict:
    """Run 20 state-tracking problems at temp=0, return score and details."""
    with open(problems_path) as f:
        problems = json.load(f)

    url = api_url.rstrip("/") + "/v1/chat/completions"
    correct = 0
    details = []
    t0 = time.time()

    for i, prob in enumerate(problems):
        try:
            resp = _post(url, {
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": prob["question"]}],
                "max_tokens": 512,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            }, timeout=60)
            content = resp["choices"][0]["message"].get("content", "")
            passed = check_key_facts(content, prob["key_facts"])
            if passed:
                correct += 1
            details.append({
                "id": prob["id"],
                "category": prob["category"],
                "passed": passed,
                "response_preview": content[:200],
            })
            if verbose:
                status = "PASS" if passed else "FAIL"
                print(f"  [{i+1:2d}/20] {prob['id']:20s} {status}"
                      f"  ({prob['category']})")
        except Exception as exc:
            details.append({
                "id": prob["id"],
                "category": prob["category"],
                "passed": False,
                "error": str(exc)[:100],
            })
            if verbose:
                print(f"  [{i+1:2d}/20] {prob['id']:20s} ERROR: {exc}")

    elapsed = time.time() - t0

    # Per-category breakdown
    cats = {}
    for d in details:
        cat = d["category"]
        cats.setdefault(cat, {"passed": 0, "total": 0})
        cats[cat]["total"] += 1
        if d["passed"]:
            cats[cat]["passed"] += 1
    for cat in cats:
        cats[cat]["rate"] = cats[cat]["passed"] / cats[cat]["total"]

    return {
        "score": correct,
        "total": len(problems),
        "accuracy": correct / len(problems) if problems else 0,
        "per_category": cats,
        "details": details,
        "elapsed_s": round(elapsed, 1),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://spark-129a.local:30000")
    parser.add_argument("--problems", default="eval_problems.json")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    result = run_fast_eval(args.api_url, args.problems, verbose=args.verbose)
    print(f"\nScore: {result['score']}/{result['total']} "
          f"({result['accuracy']:.0%}) in {result['elapsed_s']}s")
    for cat, info in result["per_category"].items():
        print(f"  {cat}: {info['passed']}/{info['total']} ({info['rate']:.0%})")
