"""Task TR: Temporal Graph Reachability.

Sequential edge stream: at each timestep t, a directed edge (src, dst)
is activated. Query: starting at node s, can node d be reached by
following activated edges in temporal order?

Algorithm (non-parallelizable without state):
    reachable = {s}
    for (a, b) in edges_in_order:
        if a in reachable:
            reachable.add(b)
    answer = (d in reachable)

This requires maintaining a growing set of reachable nodes across the
edge stream. Unlike parity/affine-group tasks which have algebraic
shortcuts, temporal reachability has no closed-form shortcut — the model
must maintain order-dependent state. This is the task class where
heat-kernel attention is theoretically strongest (diffusion on a graph
reaches a fixed point through multi-hop propagation).

Encoding (all single-token IDs on gpt2 tokenizer):
    prefix " R"  : task marker
    " 0".." 9"   : node identifiers (N ≤ 10)
    " >"         : edge separator
    " ?"         : query boundary
    " yes" / " no" : answer

Format per example:
    " R  src dst ? a1 > b1 a2 > b2 ... aK > bK = yes|no"
    where the initial "src dst" before "?" is the query, and the
    "a_i > b_i" pairs are edges in temporal order.

Only the final answer token is supervised.

Length generalization: train on K ∈ [min_edges, max_edges] (e.g. 5-15),
evaluate at K ∈ {20, 50, 100}.
"""

import random
import torch
from torch import Tensor
from typing import Tuple, Dict, Any, List, Optional


class TemporalReachabilityTask:
    """Temporal graph reachability with length-generalization hooks."""

    def __init__(
        self,
        tokenizer,
        seq_len: int = 512,
        n_nodes: int = 8,
        min_edges: int = 5,
        max_edges: int = 15,
        balance_tolerance: float = 0.15,
        min_hops_yes: int = 2,
        require_both_present_for_no: bool = True,
        n_decoy_chains: int = 2,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.n_nodes = n_nodes
        self.min_edges = min_edges
        self.max_edges = max_edges
        self.balance_tolerance = balance_tolerance
        self.min_hops_yes = min_hops_yes
        self.require_both_present_for_no = require_both_present_for_no
        self.n_decoy_chains = n_decoy_chains
        self.pad_token_id = tokenizer.eos_token_id

        # Allocate N distinct single-token IDs for nodes. Approach: take a
        # contiguous block of token IDs starting at a safe offset (40000,
        # well above punctuation and common words). The model learns these
        # embeddings from scratch; what matters is that each node has a
        # distinct token id, not its natural-language meaning. This scales
        # to N ≈ vocab_size-40000 ≈ 10k+ without any tokenization change.
        NODE_ID_BASE = 40000
        vocab = len(tokenizer)
        assert NODE_ID_BASE + n_nodes <= vocab, (
            f"n_nodes={n_nodes} exceeds vocab room "
            f"(base={NODE_ID_BASE}, vocab={vocab})")
        self._node_tokens: List[int] = [NODE_ID_BASE + i for i in range(n_nodes)]

        def _single(s: str) -> int:
            ids = tokenizer.encode(s, add_special_tokens=False)
            assert len(ids) == 1, f"{s!r} expected single token, got {ids}"
            return ids[0]

        self._prefix_token = _single(" R")
        self._arrow_token = _single(" >")
        self._q_token = _single(" ?")
        self._eq_token = _single(" =")
        self.yes_token = _single(" yes")
        self.no_token = _single(" no")

    def _reach(self, edges: List[Tuple[int, int]], src: int, dst: int) -> bool:
        reachable = {src}
        for a, b in edges:
            if a in reachable:
                reachable.add(b)
        return dst in reachable

    def _reach_all(self, edges: List[Tuple[int, int]]) -> List[set]:
        """Compute reachable set from each source; returns list[N]."""
        N = self.n_nodes
        reach = [{s} for s in range(N)]
        for a, b in edges:
            for s in range(N):
                if a in reach[s]:
                    reach[s].add(b)
        return reach

    def _shortest_hops(self, edges: List[Tuple[int, int]],
                        src: int, dst: int) -> int:
        """Minimum number of sequential edges from src to reach dst
        (respecting temporal order). Returns 10**9 if unreachable."""
        if src == dst:
            return 0
        # Hops from src when processing the stream
        hops_from_src = {src: 0}
        best = 10**9
        for a, b in edges:
            if a in hops_from_src:
                new_hops = hops_from_src[a] + 1
                if b not in hops_from_src or hops_from_src[b] > new_hops:
                    hops_from_src[b] = new_hops
                if b == dst and new_hops < best:
                    best = new_hops
        return best if best < 10**9 else 10**9

    def _sample_planted(self, path_hops: int, n_total_edges: int,
                         n_chains: int = 1
                         ) -> Tuple[List[Tuple[int, int]],
                                    List[List[int]]]:
        """Plant n_chains disjoint chains of length path_hops, interleave
        with random distractor edges. Returns (edges, chain_nodes_list).

        Disjoint chains prevent a "find the unique chain" shortcut: when
        multiple chains exist in the same edge stream, the model must
        actually trace from the queried src to determine if it reaches dst
        vs ending up elsewhere.
        """
        total_chain_nodes = (path_hops + 1) * n_chains
        assert total_chain_nodes <= self.n_nodes, (
            f"Not enough nodes: need {total_chain_nodes}, have {self.n_nodes}")
        # Pick a pool of distinct node ids, partition into n_chains
        pool = random.sample(range(self.n_nodes), total_chain_nodes)
        chains_nodes: List[List[int]] = [
            pool[i * (path_hops + 1):(i + 1) * (path_hops + 1)]
            for i in range(n_chains)]

        # All chain edges (union of all chains, each maintaining own order)
        chain_edges_per_chain = [
            [(cn[i], cn[i + 1]) for i in range(len(cn) - 1)]
            for cn in chains_nodes]

        total_chain_edges = sum(len(c) for c in chain_edges_per_chain)
        n_distract = max(0, n_total_edges - total_chain_edges)
        distractors = [(random.randint(0, self.n_nodes - 1),
                         random.randint(0, self.n_nodes - 1))
                        for _ in range(n_distract)]

        # Build edge stream: interleave all chains' edges (each chain's
        # order preserved internally) with distractors at random positions.
        # Each chain's edges remain internally in temporal order, but
        # different chains' edges and distractors are shuffled together.
        # Approach: assign random sort key to each edge; chain edges get
        # keys that are monotonic within their own chain but freely shuffled
        # across chains and distractors.
        total_edges = total_chain_edges + n_distract
        # Assign random "sort slots" to chain edges such that slot(chain_edge[i]) < slot(chain_edge[i+1])
        # by picking sorted positions per chain, then filling the rest with
        # distractors in remaining positions.
        slots_taken = set()
        positions_per_chain: List[List[int]] = []
        for chain_edges in chain_edges_per_chain:
            chain_len = len(chain_edges)
            while True:
                positions = sorted(random.sample(
                    [s for s in range(total_edges) if s not in slots_taken],
                    chain_len))
                break  # no additional constraint needed
            slots_taken.update(positions)
            positions_per_chain.append(positions)

        edges: List[Optional[Tuple[int, int]]] = [None] * total_edges
        for chain_edges, positions in zip(chain_edges_per_chain, positions_per_chain):
            for ce, pos in zip(chain_edges, positions):
                edges[pos] = ce
        distractor_iter = iter(distractors)
        for i in range(total_edges):
            if edges[i] is None:
                edges[i] = next(distractor_iter)

        # Type cast for return
        final_edges: List[Tuple[int, int]] = [e for e in edges if e is not None]
        return final_edges, chains_nodes

    def _sample_example(self,
                        target: Optional[bool] = None
                        ) -> Tuple[List[Tuple[int, int]], int, int, bool]:
        """Sample (edges, src, dst, answer) forcing multi-hop reasoning.

        For target=True: plant a chain of length ≥ min_hops_yes, then add
            distractor edges. Guarantees shortest path exists at min_hops_yes.
        For target=False: plant a chain with src → ... that does NOT reach
            some other present node; query that (src, other) pair. Ensures
            both src and dst are in edges but not connected.
        """
        # Use planted construction when target is determined.
        n_edges = random.randint(self.min_edges, self.max_edges)
        n_ch = max(2, self.n_decoy_chains)
        if target is True:
            # Plant n_ch chains; query endpoints of ONE chain chosen at random.
            # The other chains are decoys — they exist in the edge stream but
            # are not queried, so the model can't solve by "find any chain
            # and use its endpoints".
            edges, chains_nodes = self._sample_planted(
                path_hops=self.min_hops_yes, n_total_edges=n_edges,
                n_chains=n_ch)
            chain_idx = random.randint(0, len(chains_nodes) - 1)
            src = chains_nodes[chain_idx][0]
            dst = chains_nodes[chain_idx][-1]
            return edges, src, dst, True
        if target is False:
            # Plant n_ch chains; query src=chain[i].start, dst=chain[j].end
            # for i ≠ j (cross-chain query). Both are chain endpoints —
            # symmetric with yes case — so the model can't use "is src a
            # chain-start?" as a shortcut.
            for attempt in range(20):
                edges, chains_nodes = self._sample_planted(
                    path_hops=self.min_hops_yes, n_total_edges=n_edges,
                    n_chains=n_ch)
                if n_ch < 2:
                    break
                reach = self._reach_all(edges)
                # Try i,j pairs until we find an unreachable one
                pairs = [(i, j) for i in range(n_ch) for j in range(n_ch)
                          if i != j]
                random.shuffle(pairs)
                for i, j in pairs:
                    src = chains_nodes[i][0]   # start of chain i
                    dst = chains_nodes[j][-1]  # end of chain j
                    if dst not in reach[src] and (src, dst) not in edges:
                        return edges, src, dst, False
            # Fallback: any no pair from sources/dests
            edges, _ = self._sample_planted(
                path_hops=self.min_hops_yes, n_total_edges=n_edges,
                n_chains=n_ch)
            sources = {a for a, b in edges}
            dests = {b for a, b in edges}
            direct_pairs = set(edges)
            reach = self._reach_all(edges)
            candidates = [(s, d) for s in sources for d in dests
                           if s != d and (s, d) not in direct_pairs
                           and d not in reach[s]]
            if candidates:
                src, dst = random.choice(candidates)
                return edges, src, dst, False
            # Fallback: any no pair
            for s in range(self.n_nodes):
                for d in range(self.n_nodes):
                    if s != d and d not in reach[s]:
                        return edges, s, d, False
            return edges, 0, 1, False

        yes_pairs, no_pairs = [], []
        edges = []
        reach = []
        for _ in range(80):
            n_edges = random.randint(self.min_edges, self.max_edges)
            edges = [(random.randint(0, self.n_nodes - 1),
                      random.randint(0, self.n_nodes - 1))
                     for _ in range(n_edges)]
            reach = self._reach_all(edges)
            nodes_in_edges = set()
            for a, b in edges:
                nodes_in_edges.add(a)
                nodes_in_edges.add(b)

            yes_pairs, no_pairs = [], []
            for s in range(self.n_nodes):
                for d in range(self.n_nodes):
                    if s == d:
                        continue
                    if d in reach[s]:
                        # Filter yes by min hop count
                        if self.min_hops_yes > 1:
                            if self._shortest_hops(edges, s, d) >= self.min_hops_yes:
                                yes_pairs.append((s, d))
                        else:
                            yes_pairs.append((s, d))
                    else:
                        if self.require_both_present_for_no:
                            if s in nodes_in_edges and d in nodes_in_edges:
                                no_pairs.append((s, d))
                        else:
                            no_pairs.append((s, d))

            if target is True and yes_pairs:
                src, dst = random.choice(yes_pairs)
                return edges, src, dst, True
            if target is False and no_pairs:
                src, dst = random.choice(no_pairs)
                return edges, src, dst, False
            if target is None and (yes_pairs or no_pairs):
                pool = (no_pairs if random.random() < 0.5
                         and yes_pairs else (yes_pairs or no_pairs))
                src, dst = random.choice(pool)
                return edges, src, dst, (dst in reach[src])

        # Fallback: relax constraints if we can't find a valid example
        if target is True:
            # No multi-hop yes available — accept any yes
            for s in range(self.n_nodes):
                for d in range(self.n_nodes):
                    if s != d and d in reach[s]:
                        return edges, s, d, True
        # Final fallback: any no
        for s in range(self.n_nodes):
            for d in range(self.n_nodes):
                if s != d and d not in reach[s]:
                    return edges, s, d, False
        # Impossible fallback
        return edges, 0, 1, False

    def generate_batch(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        if device is None:
            device = torch.device("cpu")

        input_ids_list: List[List[int]] = []
        labels_list: List[List[int]] = []
        n_yes = 0

        for i in range(batch_size):
            # Alternate target to enforce 50/50 at batch level
            target = (i % 2 == 0)
            edges, src, dst, ans = self._sample_example(target=target)
            if ans:
                n_yes += 1
            ans_token = self.yes_token if ans else self.no_token

            tokens: List[int] = [self._prefix_token,
                                   self._node_tokens[src],
                                   self._node_tokens[dst],
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

            # Pad
            padded = tokens + [self.pad_token_id] * (self.seq_len - len(tokens))
            padded = padded[:self.seq_len]
            labels = [-100] * self.seq_len
            if answer_pos < self.seq_len:
                labels[answer_pos] = ans_token
            input_ids_list.append(padded)
            labels_list.append(labels)

        input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device)
        labels = torch.tensor(labels_list, dtype=torch.long, device=device)

        metadata = {
            "task": "TR",
            "task_name": "temporal_reach",
            "n_nodes": self.n_nodes,
            "min_edges": self.min_edges,
            "max_edges": self.max_edges,
            "batch_size": batch_size,
            "n_yes": n_yes,
        }
        return input_ids, labels, metadata


if __name__ == "__main__":
    from transformers import GPT2Tokenizer

    tok = GPT2Tokenizer.from_pretrained("gpt2")
    task = TemporalReachabilityTask(tok, seq_len=256, n_nodes=16,
                                      min_edges=5, max_edges=15)
    ids, labs, meta = task.generate_batch(batch_size=4)
    print(f"Shape: {ids.shape}  yes/4 = {meta['n_yes']}")
    for i in range(min(3, ids.shape[0])):
        clean = [t for t in ids[i].tolist() if t != tok.eos_token_id]
        print(f"  {tok.decode(clean)}")

    # OOD: longer chains
    for desc, mn, mx in [
        ("short 5-15", 5, 15),
        ("medium 20-30", 20, 30),
        ("long 50", 50, 50),
        ("very long 100", 100, 100),
    ]:
        t = TemporalReachabilityTask(tok, seq_len=1024, n_nodes=16,
                                        min_edges=mn, max_edges=mx)
        ids, _, m = t.generate_batch(batch_size=8)
        print(f"  {desc}: n_yes={m['n_yes']}/8  tokens={ids.shape[1]}")
