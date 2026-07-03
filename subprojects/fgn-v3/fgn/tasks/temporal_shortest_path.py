"""Task TSP: Temporal Shortest-Path bucket prediction on real email graph.

Unlike TR (binary yes/no reachability), this task asks for the SHORTEST
PATH LENGTH from src to dst (or "unreachable") as a multi-class output.

Why this formulation is harder to shortcut:
  - Binary classification with class-balanced rejection sampling
    inevitably creates statistical signatures between classes (degree,
    reach-set size, etc.) — the model learns these instead of reasoning.
  - Multi-class prediction with uniform bucket sampling forces the model
    to distinguish 2-hop from 3-hop from 5-hop cases. Each class has the
    same degree/reach distribution; only the specific path length differs.
  - "Unreachable" class is one of several — no 50/50 yes/no balance.

Buckets (7 classes, single-token answers):
    " one"      : path length 1 (direct edge)
    " two"      : path length 2
    " three"    : path length 3
    " four"     : path length 4
    " five"     : path length 5
    " many"     : path length 6+
    " none"     : unreachable

Each bucket gets roughly equal sampling probability.
"""

import os
import random
import torch
from collections import deque
from torch import Tensor
from typing import Tuple, Dict, Any, List, Optional


DEFAULT_PATH = "/workspace/fgn-v3/data/real_graphs/email-Eu-core-temporal.txt"


class TemporalShortestPathTask:
    """Predict temporal shortest-path bucket on real email graph."""

    _cache: Dict[str, Tuple[List[Tuple[int, int]], int]] = {}

    def __init__(
        self,
        tokenizer,
        seq_len: int = 2048,
        data_path: str = DEFAULT_PATH,
        min_edges: int = 200,
        max_edges: int = 400,
        n_nodes: int = 1024,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.data_path = data_path
        self.min_edges = min_edges
        self.max_edges = max_edges
        self.n_nodes = n_nodes
        self.pad_token_id = tokenizer.eos_token_id

        NODE_ID_BASE = 40000
        vocab = len(tokenizer)
        assert NODE_ID_BASE + n_nodes <= vocab
        self._node_tokens = [NODE_ID_BASE + i for i in range(n_nodes)]

        if data_path in self._cache:
            self._edges, self._max_node = self._cache[data_path]
        else:
            self._edges, self._max_node = self._load_edges(data_path)
            self._cache[data_path] = (self._edges, self._max_node)

        def _single(s: str) -> int:
            ids = tokenizer.encode(s, add_special_tokens=False)
            assert len(ids) == 1, f"{s!r}: {ids}"
            return ids[0]

        self._prefix_token = _single(" R")
        self._arrow_token = _single(" >")
        self._q_token = _single(" ?")
        self._eq_token = _single(" =")
        # Answer tokens — single-token words representing buckets
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

    def _load_edges(self, path: str):
        assert os.path.exists(path), f"Missing: {path}"
        edges = []
        max_node = 0
        with open(path) as f:
            for line in f:
                p = line.strip().split()
                if len(p) != 3:
                    continue
                s, d, t = int(p[0]), int(p[1]), int(p[2])
                edges.append((s, d, t))
                max_node = max(max_node, s, d)
        edges.sort(key=lambda e: e[2])
        return [(s, d) for s, d, _ in edges], max_node

    def _sample_window(self, n_edges: int) -> List[Tuple[int, int]]:
        start = random.randint(0, len(self._edges) - n_edges - 1)
        return self._edges[start:start + n_edges]

    def _temporal_shortest_path(self, edges: List[Tuple[int, int]],
                                  src: int) -> Dict[int, int]:
        """Temporal-order-respecting shortest path from src to all other nodes.
        A path (src=n0, n1, n2, ..., nk=dst) is valid if the edges used
        appear in temporal order: edge(n0,n1) at time t1 < edge(n1,n2)
        at t2 < ... Returns dict {node: shortest_path_length}.
        """
        # Process edges in temporal order. Track best-so-far
        # shortest-path length from src to each node reached.
        # hops[v] = min hops from src to v using edges seen so far (in order).
        hops = {src: 0}
        for a, b in edges:
            if a in hops:
                new_h = hops[a] + 1
                if b not in hops or hops[b] > new_h:
                    hops[b] = new_h
        return hops

    def _bucket_for(self, h: int) -> int:
        """Map hop count to bucket index (0..6). -1 for unreachable."""
        if h == 1: return 0
        if h == 2: return 1
        if h == 3: return 2
        if h == 4: return 3
        if h == 5: return 4
        if h >= 6: return 5
        return 6  # unreachable (h == -1)

    def _sample_example(self, target_bucket: Optional[int] = None
                         ) -> Tuple[List[Tuple[int, int]], int, int, int]:
        """Return (edges, src, dst, bucket_idx). If target_bucket given,
        rejection-samples until matching."""
        for _ in range(120):
            n = random.randint(self.min_edges, self.max_edges)
            edges = self._sample_window(n)
            sources = list({a for a, b in edges})
            nodes_all = list({a for a, b in edges} | {b for a, b in edges})
            if not sources or len(nodes_all) < 2:
                continue
            random.shuffle(sources)
            for src in sources[:8]:  # try a few sources per window
                hops = self._temporal_shortest_path(edges, src)
                # Per-bucket candidate lists (excluding src itself)
                bucket_candidates: Dict[int, List[int]] = {
                    b: [] for b in range(7)}
                for v in nodes_all:
                    if v == src:
                        continue
                    h = hops.get(v, -1)
                    b = self._bucket_for(h)
                    bucket_candidates[b].append(v)
                if target_bucket is not None:
                    cand = bucket_candidates.get(target_bucket, [])
                    if cand:
                        return edges, src, random.choice(cand), target_bucket
                else:
                    # Uniform random bucket with non-empty candidates
                    avail = [b for b, lst in bucket_candidates.items() if lst]
                    if avail:
                        b = random.choice(avail)
                        return edges, src, random.choice(bucket_candidates[b]), b
        # Fallback: return something
        edges = self._sample_window(self.min_edges)
        sources = [a for a, b in edges]
        src = sources[0]
        nodes = list({a for a, b in edges} | {b for a, b in edges})
        dst = next((v for v in nodes if v != src), src)
        hops = self._temporal_shortest_path(edges, src)
        b = self._bucket_for(hops.get(dst, -1))
        return edges, src, dst, b

    def generate_batch(self, batch_size: int,
                       device: Optional[torch.device] = None):
        if device is None:
            device = torch.device("cpu")
        input_ids_list: List[List[int]] = []
        labels_list: List[List[int]] = []
        bucket_counts = [0] * 7
        for i in range(batch_size):
            # Round-robin buckets to get balanced class distribution
            target = i % 7
            edges, src, dst, bucket = self._sample_example(target_bucket=target)
            bucket_counts[bucket] += 1
            ans_token = self.answer_tokens[bucket]
            # CRITICAL: do NOT append ans_token to input_ids. flat_model
            # has no label-shift (logits[i] aligned with labels[i]) AND
            # causal mask allows position i to attend to itself, so putting
            # ans_token in input at the same position where labels[pos] =
            # ans_token lets the model trivially copy via self-attention.
            # Fix: end input at '=', put ans_token only in labels[pos-of-=].
            tokens = [self._prefix_token,
                      self._node_tokens[src], self._node_tokens[dst],
                      self._q_token]
            for a, b in edges:
                tokens.append(self._node_tokens[a])
                tokens.append(self._arrow_token)
                tokens.append(self._node_tokens[b])
            tokens.append(self._eq_token)
            if len(tokens) > self.seq_len:
                tokens = tokens[:self.seq_len]
            eq_pos = len(tokens) - 1
            padded = tokens + [self.pad_token_id] * (self.seq_len - len(tokens))
            padded = padded[:self.seq_len]
            labels = [-100] * self.seq_len
            # input_ids[eq_pos] = '='  (not ans_token) — model sees '=' at
            # its own position and must predict ans_token from the context
            # (prefix + edges + '=').
            labels[eq_pos] = ans_token
            input_ids_list.append(padded)
            labels_list.append(labels)
        return (
            torch.tensor(input_ids_list, dtype=torch.long, device=device),
            torch.tensor(labels_list, dtype=torch.long, device=device),
            {"task": "TSP", "task_name": "temporal_shortest_path",
             "batch_size": batch_size, "bucket_counts": bucket_counts},
        )


if __name__ == "__main__":
    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    task = TemporalShortestPathTask(tok, seq_len=2048,
                                      min_edges=200, max_edges=400)
    print(f"Loaded {len(task._edges):,} edges")
    for i in range(3):
        edges, src, dst, bucket = task._sample_example()
        print(f"  ex{i}: src={src} dst={dst} bucket={task.bucket_names[bucket]}")
    ids, labels, meta = task.generate_batch(14)  # 2 per bucket
    print(f"Batch 14 bucket_counts: {meta['bucket_counts']} (target: 2 per bucket)")
    print(f"Shape: {ids.shape}")
