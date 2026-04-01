"""Adaptive MCMC proposal sampler for neuroplastic self-modification.

Tracks accept/reject history and adapts the proposal distribution.
No gradients, no model retraining — pure accept/reject counting.
"""

import random
import math
import json
from collections import defaultdict


class AdaptiveMCMCSampler:
    def __init__(self, temperature=1.0):
        self.temperature = temperature
        self.tensor_scores = defaultdict(float)
        self.op_scores = defaultdict(float)
        self.consecutive_rejections = defaultdict(int)
        self.accepted_magnitudes = []
        self.history = []

    def update(self, tensor, op, magnitude, accepted, score_delta):
        if accepted and score_delta > 0:
            reward = 2.0
        elif accepted and score_delta == 0:
            reward = 0.2
        else:
            reward = -0.5

        self.tensor_scores[tensor] += reward
        self.op_scores[op] += reward

        if accepted:
            self.accepted_magnitudes.append((tensor, op, magnitude))
            self.consecutive_rejections[tensor] = 0
        else:
            self.consecutive_rejections[tensor] += 1

        self.history.append({
            "tensor": tensor, "op": op, "magnitude": magnitude,
            "accepted": accepted, "score_delta": score_delta, "reward": reward,
        })

    def sample_tensor(self, candidates):
        if not candidates:
            raise ValueError("No candidates provided")
        weights = []
        for t in candidates:
            score = self.tensor_scores.get(t, 0.0)
            consec = self.consecutive_rejections.get(t, 0)
            penalty = 0.5 ** consec
            w = math.exp(score / self.temperature) * penalty
            weights.append(w)
        total = sum(weights)
        if total == 0:
            return random.choice(candidates)
        probs = [w / total for w in weights]
        return random.choices(candidates, weights=probs, k=1)[0]

    def sample_op(self, ops):
        weights = []
        for op in ops:
            score = self.op_scores.get(op, 0.0)
            w = math.exp(score / self.temperature)
            weights.append(w)
        total = sum(weights)
        if total == 0:
            return random.choice(ops)
        probs = [w / total for w in weights]
        return random.choices(ops, weights=probs, k=1)[0]

    def sample_magnitude(self, tensor, op):
        relevant = [m for t, o, m in self.accepted_magnitudes if t == tensor and o == op]
        if relevant and len(relevant) >= 2:
            center = random.choice(relevant)
            std = max(abs(center) * 0.1, 0.001)
            return center + random.gauss(0, std)
        if op == "scale":
            return random.choice([0.95, 0.96, 0.97, 0.98, 0.99, 1.01, 1.02, 1.03, 1.04, 1.05])
        elif op == "add":
            return random.choice([-0.01, -0.005, -0.001, 0.001, 0.005, 0.01])
        elif op == "scale_slice":
            return random.choice([0.9, 0.95, 0.98, 1.02, 1.05, 1.1])
        elif op == "add_noise":
            return random.choice([0.001, 0.005, 0.01])
        else:
            return 0.99

    def decay_temperature(self, cycle, total_cycles=200):
        progress = min(cycle / total_cycles, 1.0)
        self.temperature = max(1.0, 2.0 * (1.0 - progress) + 1.0 * progress)

    def get_stats(self):
        return {
            "temperature": self.temperature,
            "tensor_scores": dict(self.tensor_scores),
            "op_scores": dict(self.op_scores),
            "consecutive_rejections": dict(self.consecutive_rejections),
            "n_accepted_magnitudes": len(self.accepted_magnitudes),
            "n_history": len(self.history),
        }

    def save(self, path):
        state = {
            "temperature": self.temperature,
            "tensor_scores": dict(self.tensor_scores),
            "op_scores": dict(self.op_scores),
            "consecutive_rejections": dict(self.consecutive_rejections),
            "accepted_magnitudes": self.accepted_magnitudes,
            "history": self.history,
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            state = json.load(f)
        sampler = cls(temperature=state["temperature"])
        sampler.tensor_scores = defaultdict(float, state["tensor_scores"])
        sampler.op_scores = defaultdict(float, state["op_scores"])
        sampler.consecutive_rejections = defaultdict(int, state["consecutive_rejections"])
        sampler.accepted_magnitudes = [tuple(x) for x in state["accepted_magnitudes"]]
        sampler.history = state["history"]
        return sampler
