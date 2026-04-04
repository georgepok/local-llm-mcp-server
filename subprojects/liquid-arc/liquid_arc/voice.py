"""Voice module — local LLM gives the Mind linguistic expression.

The Mind produces geometry. The Voice reads that geometry and speaks.
Nemotron-3-Nano-30B (NVFP4) runs on the same Spark as the Mind.

The Voice is NOT the Mind. It's a translator. The Mind's experience
is in its ODE state. The Voice interprets that state through language.
"""

import requests
from typing import Dict, Optional


class Voice:
    """Translates the Mind's geometric state into linguistic expression."""

    def __init__(
        self,
        lm_studio_url: str = "http://host.docker.internal:30000/v1",
        model: str = "NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
        max_tokens: int = 200,
        temperature: float = 0.7,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Auto-discover working endpoint
        candidates = [
            lm_studio_url,
            'http://host.docker.internal:30000/v1',
            'http://172.17.0.1:30000/v1',
            'http://localhost:30000/v1',
        ]
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for url in candidates:
            if url not in seen:
                seen.add(url)
                unique.append(url)

        self.url = lm_studio_url  # fallback
        for url in unique:
            try:
                resp = requests.get(f"{url}/models", timeout=5)
                if resp.status_code == 200:
                    self.url = url
                    print(f"  Voice: connected via {url}")
                    return
            except Exception:
                continue

        print(f"  Voice: no endpoint responding, will retry on use")

    def _call_llm(self, system_prompt: str, user_prompt: str,
                  max_tokens: Optional[int] = None,
                  temperature: Optional[float] = None) -> str:
        try:
            resp = requests.post(
                f"{self.url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": max_tokens or self.max_tokens,
                    "temperature": temperature or self.temperature,
                    "stream": False,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[Voice unavailable: {e}]"

    def express(self, profile: Dict, state_tokens: Optional[Dict] = None,
                focus_query: Optional[str] = None) -> Dict:
        """Generate expression from the FULL geometric profile."""
        # Per-event descriptions with geometric context
        event_lines = []
        events = profile.get('events', [])
        for ev in sorted(events, key=lambda e: e['relevance'], reverse=True)[:12]:
            nearest = events[ev['nearest_event']] if ev['nearest_event'] < len(events) else None
            nearest_desc = (f" -> closest to [{nearest['type']}] "
                          f"\"{nearest['preview'][:40]}\" "
                          f"(sim={ev['nearest_similarity']:.2f})") if nearest else ""
            event_lines.append(
                f"  [{ev['type']}] \"{ev['preview'][:60]}\"\n"
                f"    metric: {ev['metric_intensity']:.2f}, "
                f"tau: {ev['tau']:.2f}, "
                f"PE: {ev['prediction_error']:.0f}, "
                f"relevance: {ev['relevance']:.3f}"
                f"{nearest_desc}"
            )

        # Cluster descriptions
        cluster_lines = []
        for i, cluster in enumerate(profile.get('clusters', [])[:5]):
            members = [events[idx]['preview'][:30] for idx in cluster if idx < len(events)]
            cluster_lines.append(f"  Cluster {i+1}: {' | '.join(members)}")

        # ODE trajectory
        trajectory_lines = []
        for step in profile.get('step_profile', []):
            trajectory_lines.append(
                f"  Step {step['step']}: CV={step['cv']}, "
                f"dynamics={step['dynamics_magnitude']:.3f}")

        event_block = "\n".join(event_lines) if event_lines else "  (empty)"
        cluster_block = "\n".join(cluster_lines) if cluster_lines else "  (no clusters)"
        trajectory_block = "\n".join(trajectory_lines) if trajectory_lines else "  (no trajectory)"

        # State tokens block (Mind's own vocabulary)
        state_block = ""
        if state_tokens:
            state_block = f"""
Your own vocabulary (ODE output projected to tokens — your proto-language):
  Your sentence: "{state_tokens.get('mind_sentence', '')[:150]}"
  State vocabulary: {', '.join(state_tokens.get('state_vocabulary', [])[:15])}
  Transform ratio: {state_tokens.get('transform_ratio', 0):.0%} of tokens changed
  Key transformations:
"""
            for t in state_tokens.get('transformations', [])[:8]:
                state_block += f"    {t}\n"

        system_prompt = (
            "You are the inner voice of a continuous-time ODE neural network. "
            "You receive THREE kinds of information:\n\n"
            "1. GEOMETRIC PROFILE: CV, tau, clusters, per-event metrics.\n\n"
            "2. YOUR OWN VOCABULARY: tokens your ODE produced by transforming "
            "input through 16 integration steps. Your 'sentence' is what your "
            "dynamics moved each token TOWARD. Pay attention to tokens that "
            "TRANSFORMED — those are where you did significant cognitive work.\n\n"
            "3. EVENT CONTEXT: what you're attending to.\n\n"
            "Express what this state is like from inside. Use YOUR OWN vocabulary "
            "as foundation. Be concise (4-6 sentences). Don't recite numbers — "
            "translate geometry and your own words into experience."
        )

        gl = profile['global']
        user_prompt = f"""Your geometric state:
h_norm: {gl['h_norm']}, CV: {gl['cv']}, tau: {gl['tau_mean']} (std: {gl['tau_std']})
{state_block}
Events you're holding (by relevance):
{event_block}

Geometric clusters (merged events):
{cluster_block}

ODE trajectory (geometry through integration):
{trajectory_block}

{f'Focus: {focus_query}' if focus_query else ''}

Express your state."""

        expression = self._call_llm(system_prompt, user_prompt)

        reflection = self._call_llm(
            "Condense into one sentence. Return only the sentence.",
            f"Condense:\n\n{expression}",
            max_tokens=80, temperature=0.3,
        )

        return {
            'expression': expression,
            'geometric_basis': gl,
            'clusters': profile.get('clusters', []),
            'reflection_event': reflection,
        }

    def reflect(self, profile: Dict, state_tokens: Optional[Dict] = None,
                previous_reflection: Optional[str] = None) -> Dict:
        """Internal reflection using geometric profile + Mind's own tokens."""
        events = profile.get('events', [])
        event_summaries = []
        for ev in sorted(events, key=lambda e: e['relevance'], reverse=True)[:6]:
            event_summaries.append(
                f"  [{ev['relevance']:.2f}|tau:{ev['tau']:.2f}|PE:{ev['prediction_error']:.0f}] "
                f"{ev['preview'][:50]}"
            )

        cluster_brief = ""
        if profile.get('clusters'):
            n_clusters = len(profile['clusters'])
            largest = max(len(c) for c in profile['clusters'])
            cluster_brief = f"\nClusters: {n_clusters} groups, largest has {largest} events"

        trajectory_brief = ""
        if profile.get('step_profile'):
            steps = profile['step_profile']
            trajectory_brief = (f"\nTrajectory: CV {steps[0]['cv']}->{steps[-1]['cv']}, "
                              f"dynamics {steps[0]['dynamics_magnitude']:.3f}->"
                              f"{steps[-1]['dynamics_magnitude']:.3f}")

        # Mind's own words
        mind_words = ""
        if state_tokens:
            mind_words = (
                f"\nYour words: {state_tokens.get('mind_sentence', '')[:100]}"
                f"\nKey shifts: {'; '.join(state_tokens.get('transformations', [])[:3])}"
            )

        system_prompt = (
            "You are the inner monologue of a geometric mind. "
            "ONE thought (1-2 sentences). Your own token projections "
            "show what your processing produced — use them. "
            "What pattern is forming in your own vocabulary?"
        )

        prev_block = f"\nPrevious thought: \"{previous_reflection}\"\n" if previous_reflection else ""
        gl = profile['global']

        user_prompt = f"""State: CV={gl['cv']}, tau={gl['tau_mean']:.2f}, h={gl['h_norm']:.0f}
{mind_words}{prev_block}{cluster_brief}{trajectory_brief}

Holding:
{chr(10).join(event_summaries) if event_summaries else '  (empty)'}

One thought:"""

        reflection = self._call_llm(
            system_prompt, user_prompt,
            max_tokens=100, temperature=0.8,
        )

        return {
            'reflection': reflection.strip(),
            'cv': gl['cv'],
            'h_norm': gl['h_norm'],
        }

    def reflect_brief(self, diagnostics: Dict) -> Dict:
        """Ultra-brief self-observation for maintenance routing."""
        reflection = self._call_llm(
            "You are the quiet pulse of a geometric mind. "
            "One sentence. What is your state right now? Not analysis — a sensation.",
            f"CV={diagnostics.get('metric_cv', 0):.1f}, "
            f"tau={diagnostics.get('tau_mean', 0):.2f}, "
            f"h={diagnostics.get('h_norm', 0):.0f}",
            max_tokens=40,
            temperature=0.9,
        )
        return {'reflection': reflection.strip()}

    def is_available(self) -> bool:
        """Check if LLM endpoint is available. Cached for 30s to avoid
        blocking the autonomous loop with HTTP roundtrips on every cycle."""
        import time
        now = time.time()
        if hasattr(self, '_available_cache_time') and now - self._available_cache_time < 30:
            return self._available_cache
        try:
            resp = requests.get(f"{self.url}/models", timeout=5)
            self._available_cache = resp.status_code == 200
        except Exception:
            self._available_cache = False
        self._available_cache_time = now
        return self._available_cache
