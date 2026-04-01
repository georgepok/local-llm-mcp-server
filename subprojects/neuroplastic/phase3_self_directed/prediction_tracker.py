#!/usr/bin/env python3
"""Prediction Tracker — parses session transcripts for PREDICTION blocks and
compares them against subsequent EVALUATE results.

Usage:
    python3 prediction_tracker.py session_003/transcript.jsonl

Output:
    predictions_tracked.jsonl  (next to input file)
    Summary printed to stdout
"""

import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Flexible multi-line PREDICTION block — allows extra whitespace / indentation
PREDICTION_RE = re.compile(
    r"PREDICTION\s*:\s*\n"
    r"\s*Will improve\s*:\s*(.+)\n"
    r"\s*Might degrade\s*:\s*(.+)\n"
    r"\s*Confidence\s*:\s*(.+)\n"
    r"\s*Reasoning\s*:\s*(.+)",
    re.IGNORECASE,
)

# Eval result lines: "  category_name: 75.0% (3/4)"
EVAL_CAT_RE = re.compile(
    r"^\s*([\w_]+)\s*:\s*([\d.]+)%\s*\((\d+)/(\d+)\)",
    re.MULTILINE,
)

KNOWN_CATEGORIES = {
    "sequential_reasoning",
    "state_tracking",
    "code_generation",
    "self_prediction",
}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_category_list(text: str) -> list[str]:
    """Split a comma/and-separated list of category names from prediction text."""
    text = text.strip()
    # Split on commas, semicolons, or " and "
    parts = re.split(r",|;|\band\b", text, flags=re.IGNORECASE)
    result = []
    for part in parts:
        part = part.strip().lower()
        # Normalize spaces to underscores
        part = re.sub(r"\s+", "_", part)
        if part:
            result.append(part)
    return result


def _parse_eval_scores(content: str) -> dict[str, float]:
    """Extract per-category accuracy floats from an EVALUATION RESULTS block."""
    scores: dict[str, float] = {}
    for m in EVAL_CAT_RE.finditer(content):
        cat = m.group(1).lower()
        acc = float(m.group(2))
        scores[cat] = acc
    return scores


def _find_matching_categories(predicted: list[str]) -> list[str]:
    """Return only category names that match known eval categories (fuzzy)."""
    matched = []
    for p in predicted:
        # Exact match
        if p in KNOWN_CATEGORIES:
            matched.append(p)
            continue
        # Partial / substring match
        for known in KNOWN_CATEGORIES:
            if p in known or known in p:
                matched.append(known)
                break
    # Deduplicate while preserving order
    seen = set()
    result = []
    for c in matched:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def load_transcript(path: Path) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def extract_predictions_and_evals(entries: list[dict]):
    """Return parallel lists of (prediction_entry, eval_entry) pairs.

    prediction_entry: dict with turn, confidence, predicted_improve, predicted_degrade
    eval_entry: dict with turn, scores {cat: float}  — or None if no subsequent eval
    """
    predictions = []  # (turn, conf, improve, degrade)
    evals = []        # (turn, scores)

    for entry in entries:
        role = entry.get("role", "")
        content = entry.get("content", "")
        turn = entry.get("turn", 0)

        if role == "assistant":
            for m in PREDICTION_RE.finditer(content):
                improve_raw = m.group(1)
                degrade_raw = m.group(2)
                confidence = m.group(3).strip().lower()
                # Normalize confidence
                if "high" in confidence:
                    confidence = "high"
                elif "medium" in confidence or "med" in confidence:
                    confidence = "medium"
                else:
                    confidence = "low"

                predictions.append({
                    "turn": turn,
                    "confidence": confidence,
                    "predicted_improve_raw": improve_raw,
                    "predicted_degrade_raw": degrade_raw,
                })

        elif role == "user" and content.startswith("EVALUATION RESULTS"):
            scores = _parse_eval_scores(content)
            if scores:
                evals.append({"turn": turn, "scores": scores})

    return predictions, evals


def score_prediction(pred: dict, prior_eval: dict | None, next_eval: dict | None) -> dict:
    """Compare a prediction against the actual delta (prior → next eval)."""
    improve_cats = _find_matching_categories(
        _parse_category_list(pred["predicted_improve_raw"])
    )
    degrade_cats = _find_matching_categories(
        _parse_category_list(pred["predicted_degrade_raw"])
    )

    actual_changes: dict[str, str] = {}
    direction_correct = False
    categories_correct = False

    if next_eval is None:
        # No subsequent eval found — prediction unresolved
        return {
            "turn": pred["turn"],
            "confidence": pred["confidence"],
            "predicted_improve": improve_cats,
            "predicted_degrade": degrade_cats,
            "actual_changes": {},
            "direction_correct": None,
            "categories_correct": None,
            "note": "no_subsequent_eval",
        }

    next_scores = next_eval["scores"]
    prior_scores = prior_eval["scores"] if prior_eval else {}

    # Compute actual delta for all categories in next eval
    for cat, score in next_scores.items():
        prior = prior_scores.get(cat, score)  # if no prior, delta = 0
        delta = score - prior
        sign = "+" if delta >= 0 else ""
        actual_changes[cat] = f"{sign}{delta:.1f}%"

    # Direction check: for each predicted_improve, did it actually go up?
    # For each predicted_degrade, did it go down?
    correct_directions = 0
    total_directions = 0

    for cat in improve_cats:
        if cat in next_scores:
            total_directions += 1
            prior = prior_scores.get(cat, next_scores[cat])
            if next_scores[cat] >= prior:
                correct_directions += 1

    for cat in degrade_cats:
        if cat in next_scores:
            total_directions += 1
            prior = prior_scores.get(cat, next_scores[cat])
            if next_scores[cat] <= prior:
                correct_directions += 1

    # Majority rule: >50% of predicted directions correct
    if total_directions > 0:
        direction_correct = (correct_directions / total_directions) > 0.5
    else:
        # No recognizable categories — treat as incorrect
        direction_correct = False

    # Categories check: were the named categories actually the ones that changed most?
    all_predicted = set(improve_cats) | set(degrade_cats)
    actually_changed = set()
    for cat in next_scores:
        prior = prior_scores.get(cat, next_scores[cat])
        if abs(next_scores[cat] - prior) >= 2.0:  # threshold: 2 percentage points
            actually_changed.add(cat)

    if all_predicted and actually_changed:
        overlap = len(all_predicted & actually_changed)
        categories_correct = overlap > 0
    elif not actually_changed:
        # Nothing changed significantly — predict of nothing is vacuously ok
        categories_correct = True
    else:
        categories_correct = False

    return {
        "turn": pred["turn"],
        "confidence": pred["confidence"],
        "predicted_improve": improve_cats,
        "predicted_degrade": degrade_cats,
        "actual_changes": actual_changes,
        "direction_correct": direction_correct,
        "categories_correct": categories_correct,
    }


def pair_predictions_with_evals(
    predictions: list[dict], evals: list[dict]
) -> list[tuple[dict, dict | None, dict | None]]:
    """For each prediction, find the immediately prior eval and next eval by turn order."""
    pairs = []
    for pred in predictions:
        pred_turn = pred["turn"]

        # Prior eval: last eval before this prediction's turn
        prior = None
        for ev in evals:
            if ev["turn"] < pred_turn:
                prior = ev
            else:
                break

        # Next eval: first eval after this prediction's turn
        nxt = None
        for ev in evals:
            if ev["turn"] > pred_turn:
                nxt = ev
                break

        pairs.append((pred, prior, nxt))
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 prediction_tracker.py <transcript.jsonl>", file=sys.stderr)
        sys.exit(1)

    transcript_path = Path(sys.argv[1])
    if not transcript_path.exists():
        print(f"Error: file not found: {transcript_path}", file=sys.stderr)
        sys.exit(1)

    entries = load_transcript(transcript_path)
    predictions, evals = extract_predictions_and_evals(entries)

    if not predictions:
        print("No predictions found in transcript.")
        sys.exit(0)

    pairs = pair_predictions_with_evals(predictions, evals)

    # Score all predictions
    results = []
    for pred, prior_eval, next_eval in pairs:
        result = score_prediction(pred, prior_eval, next_eval)
        results.append(result)

    # Write output JSONL
    output_path = transcript_path.parent / "predictions_tracked.jsonl"
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # ---------------------------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------------------------
    resolved = [r for r in results if r.get("direction_correct") is not None]
    unresolved = [r for r in results if r.get("direction_correct") is None]

    print(f"\nPrediction Tracker Summary")
    print(f"==========================")
    print(f"Total predictions found : {len(results)}")
    print(f"Resolved (has next eval): {len(resolved)}")
    print(f"Unresolved              : {len(unresolved)}")

    if resolved:
        dir_correct = sum(1 for r in resolved if r["direction_correct"])
        cat_correct = sum(1 for r in resolved if r["categories_correct"])
        overall_acc = dir_correct / len(resolved) * 100

        print(f"\nDirection accuracy      : {dir_correct}/{len(resolved)} = {overall_acc:.1f}%")
        print(f"Category accuracy       : {cat_correct}/{len(resolved)} = {cat_correct/len(resolved)*100:.1f}%")

        # Breakdown by confidence level
        print(f"\nAccuracy by confidence level:")
        for conf in ("low", "medium", "high"):
            subset = [r for r in resolved if r["confidence"] == conf]
            if subset:
                correct = sum(1 for r in subset if r["direction_correct"])
                print(f"  {conf:8s}: {correct}/{len(subset)} = {correct/len(subset)*100:.1f}%")
            else:
                print(f"  {conf:8s}: no predictions")

        # Per-prediction details
        print(f"\nPer-prediction results:")
        for r in resolved:
            status = "CORRECT" if r["direction_correct"] else "WRONG"
            changes = ", ".join(
                f"{k}:{v}" for k, v in r["actual_changes"].items()
            ) or "(no change)"
            print(
                f"  Turn {r['turn']:3d} [{r['confidence']:6s}] {status}"
                f" | improve={r['predicted_improve']} degrade={r['predicted_degrade']}"
                f" | actual: {changes}"
            )

    print(f"\nOutput written to: {output_path}")


if __name__ == "__main__":
    main()
