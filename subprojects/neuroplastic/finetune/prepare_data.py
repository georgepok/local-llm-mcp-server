#!/usr/bin/env python3
"""Prepare fine-tuning data from Phase 3 self-directed modification transcripts.

Converts the multi-turn conversation transcripts into training examples
for SFT on self-modification reasoning. Each example is a conversation
where the assistant proposes weight modifications using XML tags.

Output format: JSONL with {"conversations": [{"role": ..., "content": ...}, ...]}
Compatible with Unsloth/TRL SFTTrainer.
"""

import argparse
import json
import glob
import os
import re
from pathlib import Path


def load_transcripts(phase3_dir: str) -> list[list[dict]]:
    """Load all Phase 3 session transcripts."""
    pattern = os.path.join(phase3_dir, "session_*/transcript.jsonl")
    files = sorted(glob.glob(pattern))
    sessions = []
    for f in files:
        turns = []
        with open(f) as fh:
            for line in fh:
                turns.append(json.loads(line))
        if turns:
            sessions.append(turns)
    print(f"Loaded {len(sessions)} sessions, {sum(len(s) for s in sessions)} total turns")
    return sessions


def extract_system_prompt(session: list[dict]) -> str:
    """Extract and condense the system prompt from a session."""
    for turn in session:
        if turn["role"] == "system":
            return turn["content"]
    return ""


def condense_system_prompt(system_prompt: str) -> str:
    """Create a shorter system prompt that preserves key info for training."""
    return """You are Nemotron-3-Nano-30B-A3B, an NVIDIA hybrid Mamba-Transformer + MoE language model with 52 layers (23 Mamba, 6 Attention, 23 MoE). You can inspect and modify your own weights using the neuroplastic API.

Architecture:
- Mamba layers: 0,2,4,7,9,11,14,16,18,21,23,25,28,30,32,35,37,39,41,44,46,48,50
- Attention layers: 5,12,19,26,33,42
- MoE layers: 1,3,6,8,10,13,15,17,20,22,24,27,29,31,34,36,38,40,43,45,47,49,51

Tensor paths: model.layers.{i}.mixer.{A,D,dt_bias,in_proj.weight,out_proj.weight,conv1d.weight} (Mamba), model.layers.{i}.mixer.{qkv_proj.weight,o_proj.weight} (Attention), model.layers.{i}.mixer.gate.weight (MoE).

Available operations via XML tags:
<INSPECT tensor="model.layers.48.mixer.A"> — inspect tensor statistics
<MODIFY tensor="model.layers.48.mixer.A" op="scale" value="0.95"> — scale tensor
<MODIFY tensor="..." op="add" value="-0.5"> — add scalar
<MODIFY tensor="..." op="scale_slice" start="0" end="32" value="0.9"> — scale slice
<MODIFY tensor="..." op="add_noise" scale="0.01" seed="42"> — add noise
<MODIFY tensor="..." op="scale_rows" indices="[0,1]" value="0.5"> — scale rows
<CHECKPOINT tensor="..."> — save current state
<RESTORE tensor="..."> — restore from checkpoint
<EVALUATE mode="quick"> — run capability evaluation

Key findings: Mamba A values are large negative numbers controlling decay rate. Scale <1 = faster decay (shorter memory), >1 = slower decay. Layer 48 mixer.A scaling to 0.85-0.95 improved state tracking. Layer 33 qkv_proj is most sensitive. Attention layers 33,42 are architectural bottlenecks. Small changes (0.95-1.05) accumulate safely. MoE gates are mostly insensitive.

Your goal: Improve state-tracking accuracy through targeted weight modifications. Reason about what to change and why, make predictions, and learn from evaluation results."""


def is_quality_turn(content: str, role: str) -> bool:
    """Check if a turn has enough substance for training."""
    if role == "system":
        return True
    if role == "user":
        # User turns with eval results or action feedback are valuable
        return len(content) > 20
    if role == "assistant":
        # Assistant turns with actions or reasoning
        has_action = any(tag in content for tag in
                        ["<MODIFY", "<INSPECT", "<CHECKPOINT", "<RESTORE",
                         "<EVALUATE", "MODIFY", "INSPECT"])
        has_reasoning = len(content) > 100
        return has_action or has_reasoning
    return False


def has_xml_action(content: str) -> bool:
    """Check if content contains an XML action tag."""
    return bool(re.search(r'<(MODIFY|INSPECT|CHECKPOINT|RESTORE|EVALUATE)\b', content))


def build_training_examples(sessions: list[list[dict]],
                           max_turns_per_example: int = 10,
                           min_assistant_actions: int = 1) -> list[dict]:
    """Convert sessions into training conversation examples.

    Strategy:
    - Use condensed system prompt (not the full 17KB one)
    - Create sliding windows of conversation that contain at least one
      assistant turn with an action (MODIFY, INSPECT, etc.)
    - Each example is a multi-turn conversation ending with an assistant turn
    """
    examples = []
    condensed_system = condense_system_prompt("")

    for session_idx, session in enumerate(sessions):
        # Skip system turn, work with conversation turns
        conv_turns = [t for t in session if t["role"] != "system"]

        if not conv_turns:
            continue

        # Create examples using sliding windows
        i = 0
        while i < len(conv_turns):
            # Find the next assistant turn with an action
            end = i
            action_count = 0
            while end < len(conv_turns):
                turn = conv_turns[end]
                if turn["role"] == "assistant" and has_xml_action(turn["content"]):
                    action_count += 1
                end += 1
                if action_count >= min_assistant_actions and turn["role"] == "assistant":
                    break

            if action_count < min_assistant_actions:
                break

            # Determine window start (go back up to max_turns_per_example)
            start = max(i, end - max_turns_per_example)

            # Build conversation
            conversation = [{"role": "system", "content": condensed_system}]
            for t in conv_turns[start:end]:
                if is_quality_turn(t["content"], t["role"]):
                    # Truncate very long turns
                    content = t["content"]
                    if len(content) > 4000:
                        content = content[:3800] + "\n... [truncated]"
                    conversation.append({
                        "role": t["role"],
                        "content": content,
                    })

            # Only keep if we have at least system + user + assistant
            roles = [c["role"] for c in conversation]
            if "user" in roles and "assistant" in roles:
                examples.append({
                    "conversations": conversation,
                    "session": session_idx,
                    "source": "phase3_self_directed",
                })

            # Move past the action we just captured
            i = end

    return examples


def build_autoresearch_examples(autoresearch_dir: str) -> list[dict]:
    """Build training examples from Phase 7 autoresearch cycle logs.

    These are simpler: system prompt → propose modification.
    Filter for cycles where the model produced a valid MODIFY tag.
    """
    cycle_log = os.path.join(autoresearch_dir, "results/cycle_log.jsonl")
    state_file = os.path.join(autoresearch_dir, "results/autoresearch_state.json")

    if not os.path.exists(cycle_log):
        return []

    condensed_system = condense_system_prompt("")
    examples = []

    # Load accepted modifications for context
    accepted = []
    if os.path.exists(state_file):
        with open(state_file) as f:
            state = json.load(f)
        accepted = state.get("accepted_mods", [])

    # Read cycle log for successful cycles
    with open(cycle_log) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("event") != "cycle":
                continue
            if "tensor" not in entry or "op" not in entry:
                continue

            tensor = entry["tensor"]
            op = entry["op"]
            params = entry.get("params", {})
            decision = entry.get("decision", "REJECTED")
            score_before = entry.get("score_before", 0)
            score_after = entry.get("score_after", 0)

            # Build the MODIFY tag that was proposed
            param_str = " ".join(f'{k}="{v}"' for k, v in params.items())
            modify_tag = f'<MODIFY tensor="{tensor}" op="{op}" {param_str}/>'

            # Build reasoning based on outcome
            if decision == "KEPT":
                reasoning = (
                    f"Based on the sensitivity map and previous experiments, "
                    f"I'll target {tensor} with a {op} operation. "
                    f"This tensor showed sensitivity in the architecture scan. "
                    f"A conservative modification should maintain or improve accuracy.\n\n"
                    f"{modify_tag}"
                )
            else:
                # Still useful as training data — model should learn this pattern
                reasoning = (
                    f"I'll try modifying {tensor} with {op}. "
                    f"This is an exploratory modification targeting the "
                    f"{'Mamba' if 'mixer.A' in tensor or 'mixer.D' in tensor else 'attention' if 'proj' in tensor else 'MoE'} "
                    f"pathway.\n\n{modify_tag}"
                )

            user_msg = f"Score: {score_before}/20. Propose your next modification."
            if decision == "KEPT":
                user_msg = (
                    f"Previous modification was KEPT (score {score_before}→{score_after}). "
                    f"Propose your next modification."
                )
            elif decision == "REJECTED":
                user_msg = (
                    f"Previous modification was REJECTED (score {score_before}→{score_after}). "
                    f"Try a different approach. Propose your next modification."
                )

            examples.append({
                "conversations": [
                    {"role": "system", "content": condensed_system},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": reasoning},
                ],
                "session": -1,
                "source": "phase7_autoresearch",
                "decision": decision,
            })

    return examples


def main():
    parser = argparse.ArgumentParser(description="Prepare fine-tuning data")
    parser.add_argument("--phase3-dir",
                        default="../phase3_self_directed",
                        help="Path to Phase 3 self-directed directory")
    parser.add_argument("--autoresearch-dir",
                        default="../phase7_autoresearch",
                        help="Path to Phase 7 autoresearch directory")
    parser.add_argument("--output", default="training_data.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--max-turns", type=int, default=10,
                        help="Max turns per training example")
    parser.add_argument("--accepted-only", action="store_true",
                        help="Only include accepted modifications from autoresearch")
    args = parser.parse_args()

    # Phase 3 data
    print("=== Phase 3 Self-Directed Transcripts ===")
    sessions = load_transcripts(args.phase3_dir)
    phase3_examples = build_training_examples(
        sessions, max_turns_per_example=args.max_turns)
    print(f"Phase 3 examples: {len(phase3_examples)}")

    # Phase 7 autoresearch data
    print("\n=== Phase 7 Autoresearch Cycles ===")
    autoresearch_examples = build_autoresearch_examples(args.autoresearch_dir)
    if args.accepted_only:
        autoresearch_examples = [e for e in autoresearch_examples
                                  if e.get("decision") == "KEPT"]
    print(f"Autoresearch examples: {len(autoresearch_examples)}")

    # Combine
    all_examples = phase3_examples + autoresearch_examples
    print(f"\nTotal training examples: {len(all_examples)}")

    # Statistics
    total_turns = sum(len(e["conversations"]) for e in all_examples)
    action_turns = sum(
        1 for e in all_examples
        for t in e["conversations"]
        if t["role"] == "assistant" and has_xml_action(t["content"])
    )
    print(f"Total conversation turns: {total_turns}")
    print(f"Assistant turns with actions: {action_turns}")

    # Write output
    with open(args.output, "w") as f:
        for example in all_examples:
            # Remove metadata fields, keep only conversations
            out = {"conversations": example["conversations"]}
            f.write(json.dumps(out) + "\n")

    size = os.path.getsize(args.output) / 1024
    print(f"\nWrote {args.output} ({size:.1f} KB)")

    # Also write a summary
    summary = {
        "total_examples": len(all_examples),
        "phase3_examples": len(phase3_examples),
        "autoresearch_examples": len(autoresearch_examples),
        "total_turns": total_turns,
        "action_turns": action_turns,
        "sources": {
            "phase3_sessions": len(sessions),
            "autoresearch_cycles": len(autoresearch_examples),
        },
    }
    with open("training_data_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote training_data_summary.json")


if __name__ == "__main__":
    main()
