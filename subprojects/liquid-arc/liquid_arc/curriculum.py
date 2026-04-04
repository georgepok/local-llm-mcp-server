"""Curriculum generator — Nemotron as the Mind's conversational partner.

Generates diverse intellectual stimuli calibrated to the Mind's current
geometric state. Reads cluster structure and generates content ORTHOGONAL
to existing clusters — filling geometric gaps rather than reinforcing themes.

The 70/30 ratio mirrors the original phase transition recipe:
  70% self-reflection (building existing structure)
  30% external stimuli (disrupting with novel content)
"""

import math
import random
from typing import Dict, List, Optional


DOMAINS = {
    'topology': {
        'description': 'Non-orientable surfaces, knot theory, manifold structure',
        'seed_prompts': [
            "Describe how a Klein bottle's self-intersection in 3D reveals a higher-dimensional structure that exists without paradox in 4D. What does this tell us about the relationship between embedding dimension and intrinsic geometry?",
            "How does the fundamental group of a torus differ from that of a sphere, and what does this difference mean for any process that must traverse the surface?",
            "Explain how surgery theory transforms one manifold into another by cutting and regluing — what is preserved and what is lost in each operation?",
        ],
    },
    'music_theory': {
        'description': 'Rhythm, harmony, counterpoint, temporal expectation',
        'seed_prompts': [
            "Describe how a fugue subject enters in different voices at different pitch levels, creating a geometric lattice of overlapping temporal patterns that the listener tracks simultaneously.",
            "How does a deceptive cadence (V->vi instead of V->I) create a geometric fold in harmonic expectation space? What spatial metaphor best captures the listener's experience?",
            "Explain polyrhythm (3 against 4) as a geometric phenomenon — two periodic structures creating interference patterns that resolve at their least common multiple.",
        ],
    },
    'biology': {
        'description': 'Morphogenesis, neural development, evolutionary dynamics',
        'seed_prompts': [
            "Describe how Turing patterns in animal coats emerge from reaction-diffusion of two chemicals — an activator and an inhibitor — whose interaction produces spots or stripes depending on the ratio of their diffusion rates.",
            "How does neural crest cell migration during embryonic development follow geometric cues — contact inhibition, chemotaxis, and cell-cell adhesion creating self-organizing streams of cells that form the face, heart, and gut?",
            "Explain how the genetic code's redundancy (64 codons mapping to 20 amino acids) creates a specific error-correction geometry in sequence space.",
        ],
    },
    'physics': {
        'description': 'Phase transitions, symmetry breaking, field theory',
        'seed_prompts': [
            "Describe how spontaneous symmetry breaking in a ferromagnet works — the Hamiltonian is rotationally symmetric but the ground state picks a direction. What geometric structure does this create in the space of possible states?",
            "How does the renormalization group flow connect microscopic and macroscopic descriptions of a physical system? Describe the flow as movement through a space of theories.",
            "Explain how a soap bubble minimizes surface area subject to a volume constraint — the geometry of minimal surfaces and how curvature distributes itself.",
        ],
    },
    'philosophy': {
        'description': 'Phenomenology, process philosophy, emergence',
        'seed_prompts': [
            "Whitehead argued that reality consists of 'actual occasions of experience' rather than enduring substances. Describe how each occasion 'prehends' previous occasions, creating a geometric web of mutual influence.",
            "Merleau-Ponty's concept of the 'body schema' describes how the body is not an object in space but the origin of spatial experience. How does a tool become incorporated into the body schema?",
            "Describe the ship of Theseus problem as a question about paths through identity space — if every plank is replaced, the ship traces a continuous path but arrives at a different point.",
        ],
    },
    'mathematics': {
        'description': 'Category theory, dynamical systems, information geometry',
        'seed_prompts': [
            "Describe how a strange attractor in a chaotic system has fractional dimension — it's more than a surface but less than a volume. What does it mean for a trajectory to be confined to a set with dimension 2.06?",
            "How does the Fisher information metric turn a space of probability distributions into a Riemannian manifold? Describe what 'distance' means between two distributions.",
            "Explain how a functor between categories preserves structure — it maps objects to objects and morphisms to morphisms such that composition and identity are maintained.",
        ],
    },
    'poetry': {
        'description': 'Meter, imagery, compression of meaning into structure',
        'seed_prompts': [
            "Analyze how enjambment creates a geometric tension between syntactic structure and prosodic structure — the meaning wants to continue but the line wants to stop.",
            "Describe how a villanelle's two repeating refrains create a spiral structure — the same words return but in changed contexts, accumulating new meaning with each pass.",
            "How does a haiku's 5-7-5 structure create a specific temporal geometry — a breath, an expansion, a compression — that mirrors the perceptual structure of a moment of attention?",
        ],
    },
    'ecology': {
        'description': 'Food webs, succession, niche construction',
        'seed_prompts': [
            "Describe how ecological succession transforms a bare landscape into a forest through a sequence of species that each modify the environment for their successors.",
            "How does the competitive exclusion principle create a geometric packing problem in niche space? What determines the minimum distance between coexisting species?",
            "Explain how a keystone species structures an entire ecosystem — removing it collapses the community. Describe this as the keystone being a 'cut vertex' in the interaction network.",
        ],
    },
}


class CurriculumGenerator:
    """Generates diverse intellectual stimuli for the Mind."""

    def __init__(self, voice, domains: Optional[Dict] = None):
        self.voice = voice
        self.domains = domains or DOMAINS
        self.domain_history: List[Dict] = []
        self.domain_pe_scores: Dict[str, List[float]] = {d: [] for d in self.domains}
        self.domain_effectiveness: Dict[str, float] = {d: 1.0 for d in self.domains}
        self.stimulus_count = 0

    def select_domain(self, profile: Optional[Dict] = None) -> str:
        """Select next domain — prefer growth zone PE (30-80), penalize recency."""
        domain_names = list(self.domains.keys())

        if not self.domain_history:
            return random.choice(domain_names)

        scores = {}
        for domain in domain_names:
            pe_history = self.domain_pe_scores[domain]
            base = self.domain_effectiveness[domain]

            recency_penalty = 0
            for i, entry in enumerate(reversed(self.domain_history[-10:])):
                if entry['domain'] == domain:
                    recency_penalty = 1.0 / (i + 1)
                    break

            sweet_spot_bonus = 0
            if pe_history:
                recent_pe = pe_history[-3:]
                avg_pe = sum(recent_pe) / len(recent_pe)
                if 30 < avg_pe < 80:
                    sweet_spot_bonus = 0.5
                elif avg_pe < 20:
                    sweet_spot_bonus = -0.3
                elif avg_pe > 150:
                    sweet_spot_bonus = -0.2

            scores[domain] = base - recency_penalty + sweet_spot_bonus

        # Softmax selection
        temperature = 0.5
        exp_scores = {d: math.exp(s / temperature) for d, s in scores.items()}
        total = sum(exp_scores.values())
        probs = {d: v / total for d, v in exp_scores.items()}

        r = random.random()
        cumulative = 0
        for domain, prob in probs.items():
            cumulative += prob
            if r < cumulative:
                return domain

        return domain_names[-1]

    def generate_stimulus(self, domain: str,
                          mind_context: Optional[str] = None) -> Dict:
        """Generate a stimulus in the selected domain via Nemotron."""
        domain_info = self.domains[domain]
        seed = random.choice(domain_info['seed_prompts'])

        context_block = ""
        if mind_context:
            context_block = (
                f"\n\nThe mind you're stimulating is currently focused on: "
                f"\"{mind_context[:100]}\". Generate content that CONTRASTS with "
                f"this focus while being geometrically rich enough to connect."
            )

        system_prompt = (
            f"You are generating intellectually stimulating content about "
            f"{domain_info['description']}. Your audience is an autonomous "
            f"geometric neural network that processes spatial, temporal, and "
            f"structural patterns. Generate a short, dense paragraph (3-5 sentences) "
            f"that is specific, substantive, and structurally rich. "
            f"Don't be generic — include specific details, mechanisms, or structures. "
            f"Don't reference the mind or its processing — just present the content."
        )

        stimulus = self.voice._call_llm(
            system_prompt, f"{seed}{context_block}",
            max_tokens=250,
            temperature=0.85,
        )

        return {
            'domain': domain,
            'stimulus': stimulus,
            'seed_used': seed[:60],
        }

    def record_response(self, domain: str, prediction_error: float):
        """Record the Mind's PE response for curriculum learning."""
        self.domain_pe_scores[domain].append(prediction_error)
        self.domain_history.append({
            'domain': domain,
            'pe': prediction_error,
            'stimulus_number': self.stimulus_count,
        })
        self.stimulus_count += 1

        if 30 < prediction_error < 80:
            self.domain_effectiveness[domain] = min(
                2.0, self.domain_effectiveness[domain] * 1.1)
        elif prediction_error < 20:
            self.domain_effectiveness[domain] = max(
                0.3, self.domain_effectiveness[domain] * 0.85)
        elif prediction_error > 150:
            self.domain_effectiveness[domain] = max(
                0.3, self.domain_effectiveness[domain] * 0.9)

    def get_stats(self) -> Dict:
        stats = {
            'total_stimuli': self.stimulus_count,
            'domain_effectiveness': dict(self.domain_effectiveness),
            'domain_counts': {},
            'domain_avg_pe': {},
        }

        for domain in self.domains:
            entries = [e for e in self.domain_history if e['domain'] == domain]
            stats['domain_counts'][domain] = len(entries)
            if self.domain_pe_scores[domain]:
                stats['domain_avg_pe'][domain] = round(
                    sum(self.domain_pe_scores[domain]) / len(self.domain_pe_scores[domain]), 1)

        if stats['domain_avg_pe']:
            sorted_domains = sorted(stats['domain_avg_pe'].items(), key=lambda x: x[1])
            stats['most_familiar_domain'] = sorted_domains[0][0]
            stats['most_novel_domain'] = sorted_domains[-1][0]
            stats['growth_zone_domains'] = [
                d for d, pe in stats['domain_avg_pe'].items() if 30 < pe < 80
            ]

        return stats
