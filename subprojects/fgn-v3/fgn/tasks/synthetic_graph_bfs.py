"""Task SBF: Synthetic-graph BFS-depth bucket prediction.

A GraphWalks-style task for testing iterated-depth architectures.

Motivation
----------
TSP (temporal shortest-path on the SNAP email graph) has a degenerate
solution: "always predict 'unreach'" scores ~43% overall accuracy and gives
CE ≈ 1.2, which a halting architecture can achieve while halting early. The
model learns that saving compute doesn't cost CE, and collapses to the
shortcut. To test Tier 3 halting on a task where iteration actually matters,
we need:

    (a) No single-class baseline — balanced class distribution
    (b) Graph generated per example — no memorization
    (c) Depth classes that require different BFS depths — iteration-depth
        architectures should specialize compute per example

SBF provides all three.

Task
----
Generate a random DAG per example. Pick a source node. Sample a target
node whose BFS-depth matches a requested bucket (round-robin balanced
across buckets). Model must predict the bucket. 7 classes:
    " one"   : depth 1
    " two"   : depth 2
    " three" : depth 3
    " four"  : depth 4
    " five"  : depth 5
    " many"  : depth 6+
    " none"  : unreachable

Random baseline: log(7) ≈ 1.95. No shortcut — all classes equally probable.

Format
------
Input (ends at '='):
    R src dst ? e1s -> e1d e2s -> e2d ... =
Label (at '=' position only):
    answer_token (one of the 7 bucket tokens)

Same format as TSP — swap data_path='SYNTH' for a drop-in comparison.
"""

import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import torch


class SyntheticGraphBFSTask:
    """Synthetic-DAG BFS-depth bucket prediction, balanced across 7 classes."""

    def __init__(
        self,
        tokenizer,
        seq_len: int = 2048,
        min_nodes: int = 30,
        max_nodes: int = 80,
        min_edges: int = 60,
        max_edges: int = 240,
        n_node_vocab: int = 1024,  # max distinct node IDs in the vocabulary
        curriculum_enabled: bool = False,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        # Final/full-distribution settings — used after curriculum, or at all
        # times when curriculum is disabled.
        self._final_min_nodes = min_nodes
        self._final_max_nodes = max_nodes
        self._final_min_edges = min_edges
        self._final_max_edges = max_edges
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.min_edges = min_edges
        self.max_edges = max_edges
        self.pad_token_id = tokenizer.eos_token_id

        # Curriculum: easier instances of the SAME task during early training.
        # Phases bake in graph size + bucket subset. The "task structure" is
        # identical at every phase — only the difficulty distribution changes.
        # Buckets are 0..6 = depth {1,2,3,4,5,6+,unreach}.
        self.curriculum_enabled = curriculum_enabled
        self._curriculum_phases = [
            # (until_step, min_nodes, max_nodes, min_edges, max_edges, buckets)
            (1000,  10, 20, 15, 50,  [0, 1, 2]),         # depths 1-3 only
            (2500,  20, 50, 40, 150, [0, 1, 2, 3, 4]),    # depths 1-5
            (10**9, min_nodes, max_nodes, min_edges, max_edges,
             [0, 1, 2, 3, 4, 5, 6]),                      # full distribution
        ]
        self._allowed_buckets = list(range(7))
        self._current_step = 0
        if self.curriculum_enabled:
            self.set_curriculum_step(0)

        NODE_ID_BASE = 40000
        vocab = len(tokenizer)
        assert NODE_ID_BASE + n_node_vocab <= vocab, \
            f"Vocab too small: need {NODE_ID_BASE+n_node_vocab} but have {vocab}"
        self._node_tokens = [NODE_ID_BASE + i for i in range(n_node_vocab)]

        def _single(s: str) -> int:
            ids = tokenizer.encode(s, add_special_tokens=False)
            assert len(ids) == 1, f"{s!r}: {ids}"
            return ids[0]

        self._prefix_token = _single(" R")
        self._arrow_token = _single(" >")
        self._q_token = _single(" ?")
        self._eq_token = _single(" =")
        # Same 7 bucket tokens as TSP for cross-compatible eval
        self.tok_one   = _single(" one")
        self.tok_two   = _single(" two")
        self.tok_three = _single(" three")
        self.tok_four  = _single(" four")
        self.tok_five  = _single(" five")
        self.tok_many  = _single(" many")
        self.tok_none  = _single(" none")
        self.answer_tokens = [self.tok_one, self.tok_two, self.tok_three,
                               self.tok_four, self.tok_five, self.tok_many,
                               self.tok_none]
        self.bucket_names = ["1", "2", "3", "4", "5", "6+", "unreach"]

    def _generate_dag(self) -> Tuple[List[Tuple[int, int]], List[int]]:
        """Random DAG: pick a topological order, then sample edges
        that respect it."""
        n = random.randint(self.min_nodes, self.max_nodes)
        order = random.sample(range(len(self._node_tokens)), n)
        rank = {v: i for i, v in enumerate(order)}
        n_edges_target = random.randint(self.min_edges, self.max_edges)
        edges = set()
        attempts = 0
        max_attempts = 20 * n_edges_target
        while len(edges) < n_edges_target and attempts < max_attempts:
            u, v = random.sample(order, 2)
            if rank[u] < rank[v]:
                edges.add((u, v))
            attempts += 1
        edges_list = list(edges)
        random.shuffle(edges_list)
        return edges_list, order

    def _bfs_distances(self, edges: List[Tuple[int, int]],
                        src: int) -> Dict[int, int]:
        adj: Dict[int, List[int]] = {}
        for u, v in edges:
            adj.setdefault(u, []).append(v)
        dist: Dict[int, int] = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def _bucket_for(self, d: int) -> int:
        if d == 1: return 0
        if d == 2: return 1
        if d == 3: return 2
        if d == 4: return 3
        if d == 5: return 4
        if d >= 6: return 5
        return 6  # unreachable

    def _sample_example(self, target_bucket: Optional[int] = None
                          ) -> Tuple[List[Tuple[int, int]], int, int, int]:
        """Rejection-sample a graph+src+dst satisfying target_bucket."""
        for attempt in range(200):
            edges, nodes = self._generate_dag()
            # Pick source s.t. it actually has outgoing edges (not a leaf)
            sources_with_out = list({u for u, _ in edges})
            if not sources_with_out:
                continue
            random.shuffle(sources_with_out)
            for src in sources_with_out[:8]:
                dists = self._bfs_distances(edges, src)
                candidates: Dict[int, List[int]] = {b: [] for b in range(7)}
                for v in nodes:
                    if v == src:
                        continue
                    d = dists.get(v, -1)
                    candidates[self._bucket_for(d)].append(v)
                if target_bucket is not None:
                    cand = candidates.get(target_bucket, [])
                    if cand:
                        return edges, src, random.choice(cand), target_bucket
                else:
                    avail = [b for b, lst in candidates.items() if lst]
                    if avail:
                        b = random.choice(avail)
                        return edges, src, random.choice(candidates[b]), b
        # Fallback
        edges, nodes = self._generate_dag()
        src = nodes[0]
        dst = nodes[-1]
        dists = self._bfs_distances(edges, src)
        b = self._bucket_for(dists.get(dst, -1))
        return edges, src, dst, b

    def set_curriculum_step(self, step: int):
        """Update curriculum phase based on global training step.

        No-op when curriculum_enabled=False. When enabled, transitions through
        graph-size + allowed-bucket phases as defined in self._curriculum_phases.
        Cheap to call every iteration (just int comparisons).
        """
        self._current_step = step
        if not self.curriculum_enabled:
            return
        for until_step, mn, mx, me_lo, me_hi, buckets in self._curriculum_phases:
            if step < until_step:
                self.min_nodes, self.max_nodes = mn, mx
                self.min_edges, self.max_edges = me_lo, me_hi
                self._allowed_buckets = list(buckets)
                return

    def generate_batch(self, batch_size: int,
                         device: Optional[torch.device] = None):
        if device is None:
            device = torch.device("cpu")
        input_ids_list: List[List[int]] = []
        labels_list: List[List[int]] = []
        bucket_counts = [0] * 7
        # Round-robin over the curriculum's allowed buckets (== all 7 when off).
        allowed = self._allowed_buckets
        for i in range(batch_size):
            target = allowed[i % len(allowed)]
            edges, src, dst, bucket = self._sample_example(target_bucket=target)
            bucket_counts[bucket] += 1
            ans_token = self.answer_tokens[bucket]
            # Map node-indices (into _node_tokens) to actual token IDs
            tokens = [self._prefix_token,
                       self._node_tokens[src % len(self._node_tokens)],
                       self._node_tokens[dst % len(self._node_tokens)],
                       self._q_token]
            for a, b in edges:
                tokens.append(self._node_tokens[a % len(self._node_tokens)])
                tokens.append(self._arrow_token)
                tokens.append(self._node_tokens[b % len(self._node_tokens)])
            tokens.append(self._eq_token)
            if len(tokens) > self.seq_len:
                tokens = tokens[:self.seq_len]
            eq_pos = len(tokens) - 1
            padded = tokens + [self.pad_token_id] * (self.seq_len - len(tokens))
            padded = padded[:self.seq_len]
            labels = [-100] * self.seq_len
            labels[eq_pos] = ans_token
            input_ids_list.append(padded)
            labels_list.append(labels)
        return (
            torch.tensor(input_ids_list, dtype=torch.long, device=device),
            torch.tensor(labels_list, dtype=torch.long, device=device),
            {"task": "SBF", "task_name": "synthetic_graph_bfs",
             "batch_size": batch_size, "bucket_counts": bucket_counts},
        )


if __name__ == "__main__":
    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    task = SyntheticGraphBFSTask(tok, seq_len=2048)
    for i in range(3):
        edges, src, dst, bucket = task._sample_example()
        print(f"  ex{i}: {len(edges)} edges, bucket={task.bucket_names[bucket]}")
    ids, labels, meta = task.generate_batch(14)
    print(f"Batch 14 bucket_counts: {meta['bucket_counts']} (expect 2 each)")
    print(f"Shape: {ids.shape}")
