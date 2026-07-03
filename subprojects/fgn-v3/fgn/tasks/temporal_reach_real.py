"""Task TRR: Temporal Graph Reachability on REAL temporal graph data.

Source: SNAP email-Eu-core-temporal (1005 nodes, 332K temporal edges —
real email exchanges within an EU research institution).

Unlike the synthetic chain-planted TR task, this uses real graph structure:
  - Power-law degree distribution (hubs + long tail)
  - Cycles and multi-path connectivity
  - Communities (org. departments) with bridge edges
  - Natural temporal patterns (bursty email threads)

Training protocol:
  1. Sample a temporal window of N consecutive edges from the stream
  2. Compute ground-truth reachability (temporal-order-respecting) within
     that window
  3. Sample queries (src, dst) balanced 50/50 yes/no
  4. Balance rules:
     - yes: src and dst both in window, dst ∈ reach[src], (src,dst) not a
       direct edge
     - no:  src and dst both in window, dst ∉ reach[src]
     - for both: enforce src is a source of ≥1 edge AND dst is a
       destination of ≥1 edge (no "absence" shortcuts)

This task has no planted structure → no "find the unique chain" shortcut.
Multi-hop reasoning is the only way to answer correctly when distractors
create many short paths between other pairs.

Encoding matches TR: integer token IDs in range [40000, 40000+N_nodes)
for each node, plus ' R', ' >', ' ?', ' =', ' yes', ' no' separators.
"""

import os
import random
import torch
from torch import Tensor
from typing import Tuple, Dict, Any, List, Optional


# Default data path — must be downloaded separately via
# curl -o .../email-Eu-core-temporal.txt.gz https://snap.stanford.edu/data/email-Eu-core-temporal.txt.gz
DEFAULT_PATH = "/workspace/fgn-v3/data/real_graphs/email-Eu-core-temporal.txt"


class TemporalReachabilityRealTask:
    """Temporal reachability on a real email network."""

    _cache: Dict[str, Tuple[List[Tuple[int, int]], int]] = {}

    def __init__(
        self,
        tokenizer,
        seq_len: int = 2048,
        data_path: str = DEFAULT_PATH,
        min_edges: int = 100,
        max_edges: int = 300,
        n_nodes: int = 1024,  # unused content-wise; tokens allocated up to this
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.data_path = data_path
        self.min_edges = min_edges
        self.max_edges = max_edges
        self.n_nodes = n_nodes
        self.pad_token_id = tokenizer.eos_token_id

        # Allocate token IDs for nodes (same scheme as TR)
        NODE_ID_BASE = 40000
        vocab = len(tokenizer)
        assert NODE_ID_BASE + n_nodes <= vocab, (
            f"n_nodes={n_nodes} too large for vocab={vocab}")
        self._node_tokens: List[int] = [NODE_ID_BASE + i
                                          for i in range(n_nodes)]

        # Load edges (cached across instances)
        if data_path in self._cache:
            self._edges, self._max_node = self._cache[data_path]
        else:
            self._edges, self._max_node = self._load_edges(data_path)
            self._cache[data_path] = (self._edges, self._max_node)
        assert self._max_node < n_nodes, (
            f"Graph has nodes up to id {self._max_node}, "
            f"need n_nodes > that (have {n_nodes})")

        def _single(s: str) -> int:
            ids = tokenizer.encode(s, add_special_tokens=False)
            assert len(ids) == 1, f"{s!r}: {ids}"
            return ids[0]

        self._prefix_token = _single(" R")
        self._arrow_token = _single(" >")
        self._q_token = _single(" ?")
        self._eq_token = _single(" =")
        self.yes_token = _single(" yes")
        self.no_token = _single(" no")

    def _load_edges(self, path: str) -> Tuple[List[Tuple[int, int]], int]:
        assert os.path.exists(path), f"Download dataset to {path} first"
        edges: List[Tuple[int, int, int]] = []
        max_node = 0
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 3:
                    continue
                src = int(parts[0])
                dst = int(parts[1])
                t = int(parts[2])
                edges.append((src, dst, t))
                max_node = max(max_node, src, dst)
        # Sort by timestamp
        edges.sort(key=lambda e: e[2])
        # Drop timestamp; position in sorted list = time index
        return [(s, d) for s, d, _ in edges], max_node

    def _reach_from(self, edges: List[Tuple[int, int]],
                    src: int) -> set:
        reach = {src}
        for a, b in edges:
            if a in reach:
                reach.add(b)
        return reach

    def _sample_window(self, n_edges: int) -> List[Tuple[int, int]]:
        start = random.randint(0, len(self._edges) - n_edges - 1)
        return self._edges[start:start + n_edges]

    def _sample_example(self, target: Optional[bool] = None,
                          reach_lo: int = 8, reach_hi: int = 40
                          ) -> Tuple[List[Tuple[int, int]], int, int, bool]:
        """Sample with matched-reach-size constraint.

        For a given window, pick src whose reach-set size is in
        [reach_lo, reach_hi]. Then:
          - yes: dst ∈ reach[src], not a direct edge
          - no:  dst ∈ edge-stream nodes, dst ∉ reach[src]

        Both yes and no samples come from the SAME src distribution (same
        reach-size), eliminating the "count reach-size" shortcut.
        """
        n_edges = random.randint(self.min_edges, self.max_edges)
        for _ in range(80):
            edges = self._sample_window(n_edges)
            sources = list({a for a, b in edges})
            dests_set = {b for a, b in edges}
            direct_pairs = set(edges)
            if not sources or not dests_set:
                continue
            random.shuffle(sources)
            # Find a src with reach-set size in [reach_lo, reach_hi]
            for src in sources:
                reach = self._reach_from(edges, src)
                if not (reach_lo <= len(reach) <= reach_hi):
                    continue
                # Yes candidates: dst ∈ reach (excluding src itself) and
                #   (src, dst) not a direct edge
                yes_candidates = [d for d in reach
                                   if d != src and (src, d) not in direct_pairs]
                # No candidates: dst ∈ edge-nodes, dst ∉ reach[src], not direct
                no_candidates = [d for d in dests_set
                                  if d != src and d not in reach
                                  and (src, d) not in direct_pairs]
                if target is True and yes_candidates:
                    return edges, src, random.choice(yes_candidates), True
                if target is False and no_candidates:
                    return edges, src, random.choice(no_candidates), False
                if target is None and (yes_candidates or no_candidates):
                    ans = random.random() < 0.5
                    if ans and yes_candidates:
                        return edges, src, random.choice(yes_candidates), True
                    if not ans and no_candidates:
                        return edges, src, random.choice(no_candidates), False
        # Fallback: return any pair
        edges = self._sample_window(n_edges)
        sources = list({a for a, b in edges})
        src = random.choice(sources) if sources else 0
        reach = self._reach_from(edges, src)
        dest_pool = [d for d in {b for a, b in edges}
                      if d != src and (src, d) not in set(edges)]
        dst = random.choice(dest_pool) if dest_pool else (src + 1) % self.n_nodes
        return edges, src, dst, (dst in reach)

    def generate_batch(self, batch_size: int,
                       device: Optional[torch.device] = None,
                       ) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        if device is None:
            device = torch.device("cpu")
        input_ids_list: List[List[int]] = []
        labels_list: List[List[int]] = []
        n_yes = 0
        for i in range(batch_size):
            target = (i % 2 == 0)
            edges, src, dst, ans = self._sample_example(target=target)
            if ans:
                n_yes += 1
            ans_token = self.yes_token if ans else self.no_token
            tokens = [self._prefix_token,
                      self._node_tokens[src], self._node_tokens[dst],
                      self._q_token]
            for a, b in edges:
                tokens.append(self._node_tokens[a])
                tokens.append(self._arrow_token)
                tokens.append(self._node_tokens[b])
            tokens.append(self._eq_token)
            tokens.append(ans_token)
            if len(tokens) > self.seq_len:
                tokens = tokens[:self.seq_len]
            answer_pos = len(tokens) - 1
            padded = tokens + [self.pad_token_id] * (self.seq_len - len(tokens))
            padded = padded[:self.seq_len]
            labels = [-100] * self.seq_len
            if answer_pos < self.seq_len:
                labels[answer_pos] = ans_token
            input_ids_list.append(padded)
            labels_list.append(labels)

        return (
            torch.tensor(input_ids_list, dtype=torch.long, device=device),
            torch.tensor(labels_list, dtype=torch.long, device=device),
            {"task": "TRR", "task_name": "temporal_reach_real",
             "batch_size": batch_size, "n_yes": n_yes},
        )


if __name__ == "__main__":
    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    task = TemporalReachabilityRealTask(tok, seq_len=2048,
                                          min_edges=200, max_edges=400)
    print(f"Loaded {len(task._edges):,} edges; max node={task._max_node}")
    ids, _, meta = task.generate_batch(batch_size=8)
    print(f"Shape: {ids.shape}; yes/8 = {meta['n_yes']}")
    # Show reach-distance distribution on a sample window
    win = task._sample_window(300)
    sources = {a for a, b in win}
    avg_reach = 0; n = 0
    import random as _r; _r.seed(0)
    for _ in range(20):
        s = _r.choice(list(sources))
        r = task._reach_from(win, s)
        avg_reach += len(r); n += 1
    print(f"Avg reachable-set size from random src "
          f"in 300-edge window: {avg_reach/n:.1f} (of {len(sources)} sources)")
