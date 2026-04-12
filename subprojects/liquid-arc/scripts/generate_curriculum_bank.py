#!/usr/bin/env python3
"""Generate diverse curriculum topic bank using Nemotron-30B.

One-shot batch job: generates 50 unique topics per domain (8 domains = 400 topics),
saves to JSON file that the Mind loads as its curated curriculum bank.

Run on Spark with Nemotron vLLM serving on port 30000:
    python scripts/generate_curriculum_bank.py --output /workspace/liquid-arc/curriculum_bank.json
"""

import argparse
import json
import time
import requests

DOMAINS = {
    'topology': {
        'name': 'algebraic topology',
        'seed_topics': ['homotopy', 'fundamental groups', 'homology', 'knot theory',
                        'manifolds', 'Euler characteristic', 'covering spaces'],
    },
    'mathematics': {
        'name': 'pure mathematics',
        'seed_topics': ['category theory', 'Galois theory', 'measure theory',
                        'algebraic geometry', 'number theory', 'Lie groups'],
    },
    'physics': {
        'name': 'theoretical physics',
        'seed_topics': ['symmetry breaking', 'renormalization', 'entanglement',
                        'statistical mechanics', 'Berry phase', 'topological order'],
    },
    'biology': {
        'name': 'developmental biology and genetics',
        'seed_topics': ['morphogenesis', 'Hox genes', 'epigenetics',
                        'cell signaling', 'neural crest', 'gene regulation'],
    },
    'ecology': {
        'name': 'ecology and ecosystem science',
        'seed_topics': ['trophic cascades', 'niche theory', 'island biogeography',
                        'mycorrhizal networks', 'succession', 'keystone species'],
    },
    'music_theory': {
        'name': 'music theory and composition',
        'seed_topics': ['counterpoint', 'fugue structure', 'serialism',
                        'harmonic analysis', 'modes', 'polyrhythm'],
    },
    'philosophy': {
        'name': 'philosophy of mind and metaphysics',
        'seed_topics': ['consciousness', 'emergence', 'phenomenology',
                        'free will', 'identity', 'pragmatism'],
    },
    'poetry': {
        'name': 'poetry and poetics',
        'seed_topics': ['enjambment', 'meter', 'imagery', 'villanelle',
                        'haiku', 'concrete poetry', 'negative capability'],
    },
}

COMPLEXITY_TIERS = [
    "basic explanation for a newcomer",
    "intermediate with connections to related concepts",
    "advanced, requiring prior domain knowledge",
    "cutting-edge research question or open problem",
    "cross-domain connection to another field",
]

def generate_topics(vllm_url: str, model: str, domain: str, info: dict,
                    n_topics: int = 50) -> list:
    """Generate n unique topic prompts for a domain."""
    topics = []
    already = list(info['seed_topics'])

    for batch in range(0, n_topics, 10):
        tier = COMPLEXITY_TIERS[batch // 10 % len(COMPLEXITY_TIERS)]
        already_str = ', '.join(already[-30:])

        prompt = f"""Generate exactly 10 unique, specific topic explanations about {info['name']}.

Difficulty: {tier}

ALREADY COVERED (do NOT repeat): {already_str}

For each topic, write a self-contained explanation of 100-200 words.
Number them 1-10. Each should cover a DIFFERENT concept.
Respond only in English.
Do NOT repeat any topic from the already-covered list."""

        try:
            r = requests.post(
                f"{vllm_url}/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": f"You are an expert in {info['name']}. "
                         "Generate diverse, specific educational content. Always respond in English."},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 2000,
                    "temperature": 0.9,
                    "top_p": 0.95,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=300,
            )
            if r.ok:
                msg = r.json()["choices"][0]["message"]
                text = msg.get("content") or msg.get("reasoning_content") or ""
                if not text:
                    print(f"  {domain}: empty response, skipping batch")
                    continue
                # Split into individual topics
                lines = text.split('\n')
                current_topic = []
                for line in lines:
                    # Detect numbered items
                    stripped = line.strip()
                    if (stripped and len(stripped) > 2
                            and stripped[0].isdigit()
                            and (stripped[1] == '.' or (stripped[1].isdigit() and stripped[2] == '.'))):
                        if current_topic:
                            topic_text = '\n'.join(current_topic).strip()
                            if len(topic_text) > 20:
                                topics.append(topic_text)
                                # Extract topic name for dedup
                                first_line = topic_text.split('\n')[0]
                                # Remove number prefix
                                name = first_line.lstrip('0123456789. ').strip()[:60]
                                already.append(name.lower())
                        current_topic = [line]
                    else:
                        current_topic.append(line)
                # Don't forget the last one
                if current_topic:
                    topic_text = '\n'.join(current_topic).strip()
                    if len(topic_text) > 20:
                        topics.append(topic_text)
                        first_line = topic_text.split('\n')[0]
                        name = first_line.lstrip('0123456789. ').strip()[:60]
                        already.append(name.lower())

                print(f"  {domain}: batch {batch//10 + 1} → {len(topics)} topics so far")
            else:
                print(f"  {domain}: error {r.status_code}")

        except Exception as e:
            print(f"  {domain}: exception {e}")

        time.sleep(1)  # pace requests

    return topics[:n_topics]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='curriculum_bank.json')
    parser.add_argument('--vllm_url', type=str, default='http://localhost:30000/v1')
    parser.add_argument('--model', type=str, default='NVIDIA-Nemotron-3-Nano-30B-A3B-FP8')
    parser.add_argument('--topics_per_domain', type=int, default=50)
    args = parser.parse_args()

    print(f"Generating {args.topics_per_domain} topics per domain × {len(DOMAINS)} domains")
    print(f"Model: {args.model} at {args.vllm_url}")

    # Load existing bank if present (incremental)
    try:
        with open(args.output) as f:
            bank = json.load(f)
            if not any(bank.values()):
                bank = {}
    except (FileNotFoundError, json.JSONDecodeError):
        bank = {}

    total = 0
    for domain, info in DOMAINS.items():
        if domain in bank and len(bank[domain]) >= args.topics_per_domain:
            print(f"\n═══ {domain}: already has {len(bank[domain])} topics, skipping ═══")
            total += len(bank[domain])
            continue
        print(f"\n═══ {domain} ({info['name']}) ═══")
        topics = generate_topics(args.vllm_url, args.model, domain, info,
                                 args.topics_per_domain)
        bank[domain] = topics
        total += len(topics)
        print(f"  Final: {len(topics)} topics")

        # Save incrementally after each domain
        with open(args.output, 'w') as f:
            json.dump(bank, f, indent=2)
        print(f"  Saved ({total} total so far)")

    print(f"\n═══ Done: {total} topics across {len(DOMAINS)} domains ═══")
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
