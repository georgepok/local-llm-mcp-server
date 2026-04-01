"""
proposal_scorer.py — Score modification proposal quality from session transcripts.

Usage:
    python3 proposal_scorer.py session_20260311T010244/transcript.jsonl

Output:
    session_20260311T010244/proposals_scored.jsonl
"""

import json
import re
import sys
from pathlib import Path

LABELS = {0: "repetitive", 1: "template", 2: "reasoned", 3: "systemic"}


def score_proposal(proposal_text: str, modification_history: set) -> int:
    """
    Score a modification proposal on a 0-3 scale.

    0 = Repetitive  — repeating a previously tried modification
    1 = Template    — applying known pattern to new tensor
    2 = Reasoned    — novel modification with architectural justification
    3 = Systemic    — multi-layer strategy with predicted interactions
    """
    score = 0

    # Level 1: references specific tensor paths
    if re.search(r'model\.layers\.\d+', proposal_text):
        score = max(score, 1)

    # Level 2: cites reasoning or prior outcomes
    reasoning_signals = [
        'because', 'since', 'compensat', 'theory',
        'experiment showed', 'previous session',
    ]
    if any(s in proposal_text.lower() for s in reasoning_signals):
        score = max(score, 2)

    # Level 3: coordinates across multiple layers or tensor types
    layer_refs = set(re.findall(r'layer[s]?\s*\d+', proposal_text.lower()))
    tensor_types = set(re.findall(r'mixer\.\w+', proposal_text))
    if len(layer_refs) >= 2 and len(tensor_types) >= 2:
        score = 3

    # Level 3: predicts interaction effects
    interaction_signals = [
        'compensat', 'interact with', 'combined with',
        'offset by', 'balance', 'counteract',
    ]
    if any(s in proposal_text.lower() for s in interaction_signals):
        score = max(score, 3)

    # Level 2+: uses per-head operations
    if any(op in proposal_text for op in ['scale_slice', 'add_slice', 'zero_heads']):
        score = max(score, 2)

    # Level 0 override: check if this exact modification was tried before
    for m in re.finditer(r'<MODIFY\s+((?:\w+="[^"]*"\s*)+)/?\s*>', proposal_text):
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
        key = (attrs.get('tensor', ''), attrs.get('op', ''), attrs.get('value', ''))
        if key in modification_history:
            score = 0
            break

    return score


def extract_modify_keys(text: str) -> list:
    """Return (tensor, op, value) tuples for all MODIFY tags in text."""
    keys = []
    for m in re.finditer(r'<MODIFY\s+((?:\w+="[^"]*"\s*)+)/?\s*>', text):
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
        key = (attrs.get('tensor', ''), attrs.get('op', ''), attrs.get('value', ''))
        keys.append(key)
    return keys


def has_modify_action(text: str) -> bool:
    return bool(re.search(r'<MODIFY\s+', text))


def text_snippet(text: str, length: int = 200) -> str:
    cleaned = text.strip()
    return cleaned[:length]


def process_transcript(path: Path) -> list:
    """Parse transcript and return scored proposal entries."""
    modification_history: set = set()
    scored_entries = []

    with path.open() as f:
        lines = [json.loads(line) for line in f if line.strip()]

    # Build a map: turn number -> list of lines (multiple entries share a turn)
    # We process in file order to track history correctly.
    for entry in lines:
        if entry.get('role') != 'assistant':
            continue

        content = entry.get('content', '')
        reasoning = entry.get('reasoning', '')

        if not has_modify_action(content):
            continue

        # Proposal text = reasoning (model's internal justification) + content (the action)
        proposal_text = (reasoning + '\n' + content).strip()

        score = score_proposal(proposal_text, modification_history)

        # Extract each MODIFY action for individual output entries
        for m in re.finditer(r'<MODIFY\s+((?:\w+="[^"]*"\s*)+)/?\s*>', content):
            attrs = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
            tensor = attrs.get('tensor', '')
            op = attrs.get('op', '')
            value = attrs.get('value', '')

            scored_entries.append({
                'turn': entry.get('turn'),
                'score': score,
                'label': LABELS[score],
                'tensor': tensor,
                'op': op,
                'text_snippet': text_snippet(reasoning or content),
            })

            # Add to history after scoring so we don't penalize the first occurrence
            key = (tensor, op, value)
            modification_history.add(key)

    return scored_entries


def print_summary(entries: list) -> None:
    if not entries:
        print("No MODIFY proposals found in transcript.")
        return

    scores = [e['score'] for e in entries]
    avg = sum(scores) / len(scores)

    dist = {0: 0, 1: 0, 2: 0, 3: 0}
    for s in scores:
        dist[s] += 1

    print(f"\nProposal Scoring Summary")
    print(f"{'='*40}")
    print(f"Total proposals scored: {len(entries)}")
    print(f"Average score:          {avg:.2f} / 3.00")
    print()
    print(f"Distribution:")
    for level, label in LABELS.items():
        count = dist[level]
        bar = '#' * count
        print(f"  {level} ({label:10s}): {count:3d}  {bar}")
    print()

    if avg >= 2.5:
        assessment = "EXCELLENT — predominantly reasoned/systemic modifications"
    elif avg >= 1.5:
        assessment = "GOOD — mix of template and reasoned modifications"
    elif avg >= 0.75:
        assessment = "FAIR — mostly template-level, some reasoning present"
    else:
        assessment = "POOR — highly repetitive or purely mechanical"

    print(f"Assessment: {assessment}")
    print()


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} <path/to/transcript.jsonl>")
        sys.exit(1)

    transcript_path = Path(sys.argv[1])
    if not transcript_path.exists():
        print(f"Error: file not found: {transcript_path}")
        sys.exit(1)

    entries = process_transcript(transcript_path)

    output_path = transcript_path.parent / 'proposals_scored.jsonl'
    with output_path.open('w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')

    print(f"Scored {len(entries)} proposals -> {output_path}")
    print_summary(entries)


if __name__ == '__main__':
    main()
