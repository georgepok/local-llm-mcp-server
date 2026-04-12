#!/usr/bin/env python3
"""Build diverse curriculum bank from Wikipedia articles.

Streams Wikipedia, filters articles by domain-relevant keywords,
extracts the first 200-word paragraph as a curriculum stimulus.
Produces curriculum_bank.json with 50+ topics per domain.

Usage:
    python scripts/build_curriculum_from_wiki.py --output curriculum_bank.json
"""

import argparse
import json
import re
import random

DOMAIN_KEYWORDS = {
    'topology': [
        'topology', 'topological', 'manifold', 'homology', 'homotopy', 'knot theory',
        'fundamental group', 'euler characteristic', 'fiber bundle', 'cohomology',
        'simplicial', 'betti number', 'covering space', 'compact space',
        'homeomorphism', 'CW complex', 'morse theory', 'poincare',
    ],
    'mathematics': [
        'algebra', 'theorem', 'conjecture', 'category theory', 'functor',
        'galois', 'group theory', 'ring theory', 'field theory', 'number theory',
        'prime number', 'modular form', 'riemann', 'lie group', 'spectral',
        'measure theory', 'hilbert space', 'banach', 'sheaf', 'tensor',
        'differential equation', 'fourier', 'laplace', 'eigenvalue',
    ],
    'physics': [
        'quantum', 'relativity', 'thermodynamics', 'entropy', 'symmetry breaking',
        'renormalization', 'particle physics', 'condensed matter', 'superconductivity',
        'magnetism', 'fermion', 'boson', 'gauge theory', 'electromagnetism',
        'statistical mechanics', 'phase transition', 'crystal', 'phonon',
        'string theory', 'gravitational', 'cosmology', 'dark matter',
    ],
    'biology': [
        'morphogenesis', 'gene', 'protein', 'cell biology', 'developmental biology',
        'evolution', 'natural selection', 'dna', 'rna', 'epigenetic',
        'neural', 'synapse', 'embryo', 'stem cell', 'immune system',
        'mitosis', 'meiosis', 'chromosome', 'transcription', 'mutation',
        'phylogenetics', 'ecology', 'organism', 'species',
    ],
    'ecology': [
        'ecosystem', 'biodiversity', 'trophic', 'food web', 'habitat',
        'conservation', 'endangered species', 'population ecology', 'succession',
        'niche', 'symbiosis', 'predator', 'pollination', 'coral reef',
        'deforestation', 'climate change', 'wetland', 'migration',
        'keystone species', 'invasive species',
    ],
    'music_theory': [
        'music theory', 'counterpoint', 'fugue', 'sonata', 'symphony',
        'harmony', 'chord', 'scale', 'tonality', 'serialism',
        'rhythm', 'meter', 'cadence', 'modulation', 'interval',
        'polyphony', 'composition', 'orchestration', 'musical form',
    ],
    'philosophy': [
        'philosophy', 'epistemology', 'ontology', 'metaphysics', 'ethics',
        'phenomenology', 'existentialism', 'pragmatism', 'consciousness',
        'free will', 'determinism', 'empiricism', 'rationalism',
        'philosophy of mind', 'philosophy of science', 'logic',
        'aesthetics', 'political philosophy',
    ],
    'poetry': [
        'poetry', 'poem', 'poet', 'verse', 'stanza', 'sonnet',
        'haiku', 'ballad', 'ode', 'elegy', 'epic poetry',
        'meter', 'rhyme', 'iambic', 'literary criticism',
        'modernist poetry', 'romantic poetry', 'lyric poetry',
    ],
}

# Max words per stimulus
MAX_WORDS = 200


def extract_stimulus(text: str) -> str:
    """Extract a clean, informative paragraph from Wikipedia article text."""
    # Skip very short articles
    if len(text) < 200:
        return ''

    # Take first substantive paragraph (skip headers, lists, short lines)
    paragraphs = text.split('\n\n')
    for para in paragraphs:
        para = para.strip()
        # Skip section headers, references, tables
        if para.startswith('=') or para.startswith('|') or para.startswith('{'):
            continue
        if len(para) < 100:
            continue
        # Clean up
        para = re.sub(r'\s+', ' ', para)
        # Truncate to MAX_WORDS
        words = para.split()
        if len(words) > MAX_WORDS:
            para = ' '.join(words[:MAX_WORDS]) + '...'
        return para

    return ''


def matches_domain(title: str, text_start: str, keywords: list) -> bool:
    """Check if article matches domain by title or first 500 chars."""
    combined = (title + ' ' + text_start[:500]).lower()
    return any(kw in combined for kw in keywords)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='curriculum_bank.json')
    parser.add_argument('--topics_per_domain', type=int, default=100)
    parser.add_argument('--max_articles', type=int, default=500000,
                        help='Max articles to scan (Wikipedia has ~6M)')
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Building curriculum bank: {args.topics_per_domain} topics × "
          f"{len(DOMAIN_KEYWORDS)} domains")
    print(f"Scanning up to {args.max_articles:,} Wikipedia articles...")

    bank = {d: [] for d in DOMAIN_KEYWORDS}
    targets = {d: args.topics_per_domain for d in DOMAIN_KEYWORDS}

    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

    scanned = 0
    matched = 0
    for article in ds:
        scanned += 1
        if scanned > args.max_articles:
            break

        title = article['title']
        text = article['text']

        for domain, keywords in DOMAIN_KEYWORDS.items():
            if len(bank[domain]) >= targets[domain]:
                continue

            if matches_domain(title, text, keywords):
                stimulus = extract_stimulus(text)
                if stimulus and len(stimulus) > 50:
                    bank[domain].append(stimulus)
                    matched += 1
                    break  # one article → one domain

        if scanned % 50000 == 0:
            counts = {d: len(v) for d, v in bank.items()}
            total = sum(counts.values())
            print(f"  Scanned {scanned:,} articles, {total} matches: {counts}")

        # Early exit if all domains full
        if all(len(bank[d]) >= targets[d] for d in bank):
            print(f"  All domains full at {scanned:,} articles!")
            break

    # Shuffle within each domain for variety
    for d in bank:
        random.shuffle(bank[d])

    # Save
    with open(args.output, 'w') as f:
        json.dump(bank, f, indent=2)

    total = sum(len(v) for v in bank.values())
    print(f"\n═══ Done: {total} topics across {len(bank)} domains ═══")
    for d, topics in sorted(bank.items()):
        print(f"  {d:15s}: {len(topics)} topics")
    print(f"Saved to {args.output}")
    print(f"Scanned {scanned:,} articles, {matched} matches")


if __name__ == '__main__':
    main()
