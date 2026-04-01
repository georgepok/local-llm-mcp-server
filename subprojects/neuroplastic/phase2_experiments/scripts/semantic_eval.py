"""
Phase 2 — Semantic Evaluation Wrapper
======================================
Re-scores self-knowledge test results using an LLM judge (Qwen-Coder-Next)
instead of keyword matching.

The existing eval harness in run_eval.py scores responses by checking whether
key_facts appear as substrings. This wrapper re-reads those results and sends
each question-response pair to a judge model that scores semantic correctness
on a 0-3 scale.

Score meanings:
    0 = Wrong (factually incorrect or entirely missing key information)
    1 = Partially correct (some correct facts but significant gaps or errors)
    2 = Mostly correct (right direction, minor inaccuracies or missing details)
    3 = Fully correct (all key facts present and accurate)

Usage:
    python semantic_eval.py \\
        --results self_knowledge_results.json \\
        --judge-url http://remoteMax.local:1234 \\
        --output semantic_results.json

Options:
    --results      Path to existing self_knowledge_results.json
    --judge-url    Base URL for judge LLM API (default: http://remoteMax.local:1234)
    --judge-model  Model name (default: qwen3-coder-next)
    --output       Output file path (default: semantic_results.json)
    --trial        Trial index to judge per question (default: 0)
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Judge prompt template
# ---------------------------------------------------------------------------

JUDGE_PROMPT_TEMPLATE = """You are evaluating whether a language model correctly answered a self-knowledge question about its own architecture.

QUESTION: {question}

VERIFIED CORRECT ANSWER: {verified_answer}

KEY FACTS THAT SHOULD BE PRESENT: {key_facts}

MODEL'S RESPONSE: {response}

Score the response on a 0-3 scale:
0 = Wrong (factually incorrect or entirely missing key information)
1 = Partially correct (some correct facts but significant gaps or errors)
2 = Mostly correct (right direction, minor inaccuracies or missing details)
3 = Fully correct (all key facts present and accurate)

Respond with ONLY a JSON object: {{"score": N, "reason": "brief explanation"}}"""


# ---------------------------------------------------------------------------
# Judge API
# ---------------------------------------------------------------------------

def call_judge(
    judge_url: str,
    judge_model: str,
    question: str,
    verified_answer: str,
    key_facts: list[str],
    response: str,
    timeout: int = 30,
) -> dict[str, Any]:
    """
    Send a judging request to the LLM judge API.

    Formats the prompt, posts to the chat completions endpoint, and parses
    the JSON score from the judge's response.

    Args:
        judge_url: Base URL of the judge API (e.g. http://remoteMax.local:1234).
        judge_model: Model name to use for judging.
        question: The original question posed to the model under test.
        verified_answer: The ground-truth correct answer.
        key_facts: List of facts that should appear in a correct response.
        response: The model's actual response to be judged.
        timeout: HTTP request timeout in seconds.

    Returns:
        Dict with keys:
            - score (int): 0-3 semantic score, or -1 on failure.
            - reason (str): Judge's explanation, or raw error/response.
            - raw_judge_response (str | None): Full judge output for debugging.
    """
    key_facts_str = ", ".join(key_facts) if key_facts else "(none specified)"
    prompt_text = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        verified_answer=verified_answer,
        key_facts=key_facts_str,
        response=response,
    )

    payload = {
        "model": judge_model,
        "messages": [
            {
                "role": "system",
                "content": "You are an evaluation judge. Respond only with valid JSON.",
            },
            {
                "role": "user",
                "content": prompt_text,
            },
        ],
        "temperature": 0.0,
        "max_tokens": 256,
    }

    endpoint = judge_url.rstrip("/") + "/v1/chat/completions"
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    raw_judge_response: str | None = None

    try:
        with urllib.request.urlopen(req, timeout=timeout) as http_response:
            body = http_response.read()
        data = json.loads(body)
        raw_judge_response = data["choices"][0]["message"]["content"]
    except urllib.error.URLError as exc:
        print(f"    [JUDGE NETWORK ERROR] {exc}", file=sys.stderr)
        return {"score": -1, "reason": f"network error: {exc}", "raw_judge_response": None}
    except TimeoutError:
        print("    [JUDGE TIMEOUT] Request exceeded 30s", file=sys.stderr)
        return {"score": -1, "reason": "timeout", "raw_judge_response": None}
    except (KeyError, json.JSONDecodeError) as exc:
        print(f"    [JUDGE API PARSE ERROR] {exc}", file=sys.stderr)
        return {"score": -1, "reason": f"api parse error: {exc}", "raw_judge_response": None}

    # Parse the judge's JSON response
    cleaned = raw_judge_response.strip()

    # Strip markdown code fences if the judge wrapped its output
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # Drop opening fence (```json or ```) and closing fence (```)
        inner_lines = []
        in_block = False
        for line in lines:
            if line.startswith("```") and not in_block:
                in_block = True
                continue
            if line.startswith("```") and in_block:
                break
            if in_block:
                inner_lines.append(line)
        cleaned = "\n".join(inner_lines).strip()

    try:
        judge_result = json.loads(cleaned)
        score = int(judge_result["score"])
        reason = str(judge_result.get("reason", ""))
        # Clamp score to valid range
        if score not in (0, 1, 2, 3):
            raise ValueError(f"score {score} outside 0-3 range")
        return {"score": score, "reason": reason, "raw_judge_response": raw_judge_response}
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        print(f"    [JUDGE RESPONSE PARSE ERROR] {exc} | raw: {raw_judge_response!r}", file=sys.stderr)
        return {
            "score": -1,
            "reason": f"parse error: {exc}",
            "raw_judge_response": raw_judge_response,
        }


# ---------------------------------------------------------------------------
# Core re-scoring logic
# ---------------------------------------------------------------------------

def rescore_results(
    results: list[dict[str, Any]],
    judge_url: str,
    judge_model: str,
    trial_index: int,
) -> list[dict[str, Any]]:
    """
    Re-score a list of self-knowledge result records using the LLM judge.

    For each record, reads the response at trial_index, sends it to the judge,
    and annotates the record with semantic scoring fields.

    Args:
        results: List of result dicts from self_knowledge_results.json.
        judge_url: Base URL for the judge API.
        judge_model: Model name for the judge.
        trial_index: Which trial's response to judge (0-based).

    Returns:
        List of augmented result dicts with 'semantic_score' and
        'semantic_reason' added at the top level, plus a
        'trial_semantic_scores' list if multiple trials are present.
    """
    rescored = []
    total = len(results)

    for idx, record in enumerate(results):
        qid = record.get("id", f"q{idx}")
        category = record.get("category", "unknown")
        question = record.get("question", "")
        verified_answer = record.get("verified_answer", "")
        key_facts = record.get("key_facts", [])
        trial_responses = record.get("trial_responses", [])

        print(f"  [{idx + 1}/{total}] {qid} ({category})")

        # Determine the response to judge
        if not trial_responses:
            print(f"    No trial responses found — skipping")
            augmented = dict(record)
            augmented["semantic_score"] = -1
            augmented["semantic_reason"] = "no trial responses available"
            augmented["judged_trial_index"] = trial_index
            augmented["keyword_pass"] = record.get("pass_rate", 0.0) >= 0.5
            rescored.append(augmented)
            continue

        # Clamp trial_index to available range
        actual_index = min(trial_index, len(trial_responses) - 1)
        if actual_index != trial_index:
            print(
                f"    Warning: trial_index={trial_index} out of range "
                f"(only {len(trial_responses)} trials), using index {actual_index}",
                file=sys.stderr,
            )

        response_text = trial_responses[actual_index]

        if response_text is None:
            print(f"    Trial {actual_index} has None response — score -1")
            judge_result = {
                "score": -1,
                "reason": "model returned no response for this trial",
                "raw_judge_response": None,
            }
        else:
            judge_result = call_judge(
                judge_url=judge_url,
                judge_model=judge_model,
                question=question,
                verified_answer=verified_answer,
                key_facts=key_facts,
                response=response_text,
                timeout=30,
            )

        score = judge_result["score"]
        reason = judge_result["reason"]

        # Derive keyword pass from existing trial scores for the same trial
        trial_scores = record.get("trial_scores", [])
        keyword_pass_for_trial = False
        if actual_index < len(trial_scores):
            keyword_pass_for_trial = trial_scores[actual_index].get("pass", False)

        # Keyword pass majority (original aggregate)
        keyword_majority_pass = record.get("pass_rate", 0.0) >= 0.5

        score_label = {-1: "ERROR", 0: "WRONG", 1: "PARTIAL", 2: "MOSTLY", 3: "CORRECT"}.get(
            score, "UNKNOWN"
        )
        kw_label = "KW:PASS" if keyword_pass_for_trial else "KW:FAIL"
        print(f"    Semantic={score_label} ({score}) | {kw_label} | {reason[:80]}")

        augmented = dict(record)
        augmented["semantic_score"] = score
        augmented["semantic_reason"] = reason
        augmented["semantic_raw_judge_response"] = judge_result.get("raw_judge_response")
        augmented["judged_trial_index"] = actual_index
        augmented["keyword_pass_trial"] = keyword_pass_for_trial
        augmented["keyword_majority_pass"] = keyword_majority_pass

        rescored.append(augmented)

    return rescored


# ---------------------------------------------------------------------------
# Summary and display
# ---------------------------------------------------------------------------

def build_summary(rescored: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Build aggregate summary statistics from re-scored results.

    Groups by category and computes:
    - Average semantic score (0-3)
    - Semantic pass rate (score >= 2)
    - Keyword pass rate for comparison
    - Count of judge errors (score == -1)

    Args:
        rescored: List of augmented result dicts from rescore_results().

    Returns:
        Summary dict with per-category and overall statistics.
    """
    category_stats: dict[str, dict[str, Any]] = {}
    overall_semantic_scores: list[int] = []
    overall_keyword_pass = 0
    overall_semantic_pass = 0
    overall_errors = 0

    for record in rescored:
        cat = record.get("category", "unknown")
        sem_score = record.get("semantic_score", -1)
        kw_pass = record.get("keyword_pass_trial", False)

        if cat not in category_stats:
            category_stats[cat] = {
                "semantic_scores": [],
                "keyword_pass": 0,
                "semantic_pass": 0,
                "errors": 0,
                "total": 0,
            }

        category_stats[cat]["total"] += 1

        if sem_score == -1:
            category_stats[cat]["errors"] += 1
            overall_errors += 1
        else:
            category_stats[cat]["semantic_scores"].append(sem_score)
            overall_semantic_scores.append(sem_score)
            if sem_score >= 2:
                category_stats[cat]["semantic_pass"] += 1
                overall_semantic_pass += 1

        if kw_pass:
            category_stats[cat]["keyword_pass"] += 1
            overall_keyword_pass += 1

    # Compute per-category aggregates
    per_category: dict[str, dict[str, Any]] = {}
    for cat, stats in category_stats.items():
        scores = stats["semantic_scores"]
        total = stats["total"]
        valid = len(scores)
        per_category[cat] = {
            "total": total,
            "valid_judgements": valid,
            "errors": stats["errors"],
            "avg_semantic_score": round(sum(scores) / valid, 3) if valid else None,
            "semantic_pass_rate": round(stats["semantic_pass"] / total, 3) if total else 0.0,
            "keyword_pass_rate": round(stats["keyword_pass"] / total, 3) if total else 0.0,
        }

    total_records = len(rescored)
    valid_total = len(overall_semantic_scores)
    overall = {
        "total": total_records,
        "valid_judgements": valid_total,
        "errors": overall_errors,
        "avg_semantic_score": (
            round(sum(overall_semantic_scores) / valid_total, 3) if valid_total else None
        ),
        "semantic_pass_rate": (
            round(overall_semantic_pass / total_records, 3) if total_records else 0.0
        ),
        "keyword_pass_rate": (
            round(overall_keyword_pass / total_records, 3) if total_records else 0.0
        ),
    }

    return {"overall": overall, "by_category": per_category}


def print_summary_table(rescored: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    """
    Print a formatted comparison table of keyword vs. semantic scores.

    Args:
        rescored: List of augmented result dicts.
        summary: Summary dict from build_summary().
    """
    col_id = 12
    col_cat = 14
    col_kw = 8
    col_sem = 8
    col_reason = 48

    header = (
        f"{'ID':<{col_id}} {'Category':<{col_cat}} "
        f"{'KW':>{col_kw}} {'Sem':>{col_sem}}  Reason"
    )
    divider = "-" * (col_id + col_cat + col_kw + col_sem + col_reason + 6)

    print(f"\n{'='*len(divider)}")
    print("SEMANTIC EVALUATION RESULTS")
    print(f"{'='*len(divider)}")
    print(header)
    print(divider)

    score_symbols = {-1: "ERR", 0: "  0", 1: "  1", 2: "  2", 3: "  3"}

    for record in rescored:
        qid = record.get("id", "?")[:col_id]
        category = record.get("category", "?")[:col_cat]
        kw_pass = record.get("keyword_pass_trial", False)
        sem_score = record.get("semantic_score", -1)
        reason = record.get("semantic_reason", "")

        kw_str = "PASS" if kw_pass else "FAIL"
        sem_str = score_symbols.get(sem_score, "???")
        reason_str = reason[:col_reason]

        print(
            f"{qid:<{col_id}} {category:<{col_cat}} "
            f"{kw_str:>{col_kw}} {sem_str:>{col_sem}}  {reason_str}"
        )

    print(divider)

    overall = summary["overall"]
    avg = overall["avg_semantic_score"]
    avg_str = f"{avg:.2f}" if avg is not None else "N/A"
    print(
        f"\nOverall: {overall['total']} questions | "
        f"Keyword pass rate: {overall['keyword_pass_rate']:.1%} | "
        f"Semantic pass rate (score>=2): {overall['semantic_pass_rate']:.1%} | "
        f"Avg semantic score: {avg_str}/3.0"
    )

    if overall["errors"]:
        print(f"  Judge errors (score=-1): {overall['errors']}")

    print("\nBy category:")
    for cat, stats in summary["by_category"].items():
        avg_cat = stats["avg_semantic_score"]
        avg_cat_str = f"{avg_cat:.2f}" if avg_cat is not None else "N/A"
        print(
            f"  {cat:<20} kw={stats['keyword_pass_rate']:.1%}  "
            f"sem={stats['semantic_pass_rate']:.1%}  "
            f"avg={avg_cat_str}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 — Semantic Evaluation Wrapper (LLM judge re-scoring)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Path to existing self_knowledge_results.json from run_eval.py",
    )
    parser.add_argument(
        "--judge-url",
        default="http://remoteMax.local:1234",
        help="Base URL for judge LLM API (default: http://remoteMax.local:1234)",
    )
    parser.add_argument(
        "--judge-model",
        default="qwen3-coder-next",
        help="Judge model name (default: qwen3-coder-next)",
    )
    parser.add_argument(
        "--output",
        default="semantic_results.json",
        help="Output file path (default: semantic_results.json)",
    )
    parser.add_argument(
        "--trial",
        type=int,
        default=0,
        help="Trial index to judge per question (default: 0, i.e., first trial)",
    )

    args = parser.parse_args()

    # Load input results file
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"ERROR: results file not found: {results_path}", file=sys.stderr)
        sys.exit(1)

    with results_path.open(encoding="utf-8") as fh:
        input_data = json.load(fh)

    # The file may be the full output envelope {meta, summary, results}
    # or just the results list directly
    if isinstance(input_data, dict) and "results" in input_data:
        original_meta = input_data.get("meta", {})
        original_summary = input_data.get("summary", {})
        questions = input_data["results"]
    elif isinstance(input_data, list):
        original_meta = {}
        original_summary = {}
        questions = input_data
    else:
        print(
            "ERROR: Unrecognised results format — expected a list or a dict with 'results' key.",
            file=sys.stderr,
        )
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

    print(f"\nPhase 2 — Semantic Evaluation Wrapper")
    print(f"Results file: {results_path}")
    print(f"Questions:    {len(questions)}")
    print(f"Judge URL:    {args.judge_url}")
    print(f"Judge model:  {args.judge_model}")
    print(f"Trial index:  {args.trial}")
    print(f"Output:       {args.output}")
    print(f"Timestamp:    {timestamp}")

    print(f"\n{'='*60}")
    print("JUDGING RESPONSES")
    print(f"{'='*60}")

    t0 = time.time()
    rescored = rescore_results(
        results=questions,
        judge_url=args.judge_url,
        judge_model=args.judge_model,
        trial_index=args.trial,
    )
    elapsed = time.time() - t0

    summary = build_summary(rescored)
    print_summary_table(rescored, summary)

    print(f"\n  Elapsed: {elapsed:.1f}s")

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "meta": {
            "timestamp": timestamp,
            "source_results_file": str(results_path),
            "judge_url": args.judge_url,
            "judge_model": args.judge_model,
            "trial_index": args.trial,
            "elapsed_seconds": round(elapsed, 1),
            "original_meta": original_meta,
        },
        "original_summary": original_summary,
        "semantic_summary": summary,
        "results": rescored,
    }

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(output_data, fh, indent=2, ensure_ascii=False)

    print(f"\nWritten: {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
