"""Sequential agentic tasks — multi-turn episodes for persistent state testing.

Each EPISODE is a sequence of TURNS. Each turn is one forward pass.
With persistent state, the model carries context from turn to turn.
Without persistent state, each turn is processed independently.

Episode type: Sequential Stateful — a long operation chain broken into turns.
"""

import random
from typing import Dict, List, Optional, Tuple

import torch

from liquid_arc.tasks.procedural import (
    PAD_COLOR, PAD_COORD, N_COLORS,
    build_sequence,
)

BG = 0
COPY_MARKER = 8


def _empty_grid(H: int, W: int, bg: int = 0) -> List[List[int]]:
    return [[bg] * W for _ in range(H)]


class SequentialStatefulEpisode:
    """Generates multi-turn stateful execution episodes.

    A long operation chain (6-10 ops) is broken into turns of 2-3 ops each.
    Each turn shows the current state + the next batch of operations,
    and must predict the state after those operations execute.

    Output is answer-only (just the result row, not the copied context).
    """

    def __init__(self, n_vars=4, total_ops=8, ops_per_turn=2, n_demos=2, seq_len=2048):
        self.n_vars = n_vars
        self.total_ops = total_ops
        self.ops_per_turn = ops_per_turn
        self.n_demos = n_demos
        self.seq_len = seq_len

    def generate_episode(self) -> List[Dict]:
        """Generate a full episode as a list of per-turn training samples."""
        W = self.n_vars
        n_turns = self.total_ops // self.ops_per_turn

        # Generate the FULL operation chain and states
        full_states = []
        operations = []
        state = [BG] * self.n_vars

        # Initial state
        n_init = random.randint(1, min(3, self.n_vars))
        for v in random.sample(range(self.n_vars), n_init):
            state[v] = random.randint(1, 7)
        full_states.append(list(state))

        for _ in range(self.total_ops):
            target = random.randint(0, self.n_vars - 1)
            sources = [v for v in range(self.n_vars) if state[v] != BG and v != target]

            if sources and random.random() < 0.4:
                src = random.choice(sources)
                state[target] = state[src]
                operations.append(('copy', target, src))
            else:
                val = random.randint(1, 7)
                state[target] = val
                operations.append(('set', target, val))

            full_states.append(list(state))

        # Break into turns
        turns = []
        for turn_idx in range(n_turns):
            op_start = turn_idx * self.ops_per_turn
            op_end = min(op_start + self.ops_per_turn, self.total_ops)

            current_state = full_states[op_start]
            target_state = full_states[op_end]

            n_ops_this = op_end - op_start
            # Turn 0: state row + op rows. Later turns: op rows ONLY.
            # This forces later turns to rely on persistent state for current values.
            show_state = (turn_idx == 0)
            H_inp = (1 if show_state else 0) + n_ops_this

            # Generate demos with same structure, different values
            demos = []
            for _ in range(self.n_demos):
                demo_state = [random.randint(1, 7) if current_state[v] != BG else BG
                              for v in range(self.n_vars)]
                demo_target = list(demo_state)

                demo_inp = _empty_grid(H_inp, W, BG)
                if show_state:
                    demo_inp[0] = demo_state[:]

                for i in range(op_start, op_end):
                    op = operations[i]
                    row = i - op_start + (1 if show_state else 0)
                    if op[0] == 'set':
                        demo_val = random.randint(1, 7)
                        demo_inp[row][op[1]] = demo_val
                        demo_target[op[1]] = demo_val
                    elif op[0] == 'copy':
                        demo_inp[row][op[1]] = COPY_MARKER
                        demo_inp[row][op[2]] = COPY_MARKER
                        demo_target[op[1]] = demo_target[op[2]]

                demo_out = [demo_target[:]]
                demos.append((demo_inp, demo_out))

            # Test instance
            test_inp = _empty_grid(H_inp, W, BG)
            if show_state:
                test_inp[0] = current_state[:]

            for i in range(op_start, op_end):
                op = operations[i]
                row = i - op_start + (1 if show_state else 0)
                if op[0] == 'set':
                    test_inp[row][op[1]] = op[2]
                elif op[0] == 'copy':
                    test_inp[row][op[1]] = COPY_MARKER
                    test_inp[row][op[2]] = COPY_MARKER

            test_out = [target_state[:]]

            seq = build_sequence(demos, test_inp, test_out)

            if seq["length"] > self.seq_len:
                demos = demos[:1]
                seq = build_sequence(demos, test_inp, test_out)
                if seq["length"] > self.seq_len:
                    for key in ["colors", "xs", "ys", "roles", "sep_mask", "sep_types",
                                "grid_ids", "target_mask", "target_input_colors"]:
                        seq[key] = seq[key][:self.seq_len]
                    seq["length"] = self.seq_len

            seq["turn_index"] = turn_idx
            turns.append(seq)

        return turns


class SequentialAgenticDataset:
    """Generates batches of sequential agentic episodes.

    Episodes advance in lockstep: all batch items are turn K,
    then all are turn K+1, etc.
    """

    def __init__(self, batch_size=4, n_vars=4, total_ops=8, ops_per_turn=2,
                 n_demos=2, seq_len=2048):
        self.batch_size = batch_size
        self.n_vars = n_vars
        self.total_ops = total_ops
        self.ops_per_turn = ops_per_turn
        self.n_demos = n_demos
        self.seq_len = seq_len
        self.n_turns = total_ops // ops_per_turn
        self._episodes: Optional[List[List[Dict]]] = None
        self._current_turn = 0

    def reset_episodes(self):
        """Generate fresh episodes for each batch position."""
        self._episodes = []
        for _ in range(self.batch_size):
            gen = SequentialStatefulEpisode(
                n_vars=self.n_vars, total_ops=self.total_ops,
                ops_per_turn=self.ops_per_turn, n_demos=self.n_demos,
                seq_len=self.seq_len,
            )
            self._episodes.append(gen.generate_episode())
        self._current_turn = 0

    def get_next_turn_batch(self, device=None):
        """Get the next turn's data for all episodes.

        Returns:
            (input_ids, labels, meta), is_episode_start, is_episode_end, turn_index
        """
        if self._episodes is None or self._current_turn >= self.n_turns:
            self.reset_episodes()

        is_start = (self._current_turn == 0)
        is_end = (self._current_turn == self.n_turns - 1)
        turn_idx = self._current_turn

        samples = [ep[self._current_turn] for ep in self._episodes]
        self._current_turn += 1

        # Collate into batch tensors
        if device is None:
            device = torch.device('cpu')

        B = len(samples)
        max_N = self.seq_len

        colors = torch.full((B, max_N), PAD_COLOR, dtype=torch.long, device=device)
        xs_t = torch.full((B, max_N), PAD_COORD, dtype=torch.long, device=device)
        ys_t = torch.full((B, max_N), PAD_COORD, dtype=torch.long, device=device)
        roles = torch.zeros(B, max_N, dtype=torch.long, device=device)
        sep_mask = torch.ones(B, max_N, dtype=torch.bool, device=device)
        sep_types = torch.zeros(B, max_N, dtype=torch.long, device=device)
        grid_ids = torch.full((B, max_N), -1, dtype=torch.long, device=device)
        target_mask = torch.zeros(B, max_N, dtype=torch.bool, device=device)
        target_labels = torch.full((B, max_N), -100, dtype=torch.long, device=device)
        target_input_colors = torch.full((B, max_N), PAD_COLOR, dtype=torch.long, device=device)
        lengths = torch.zeros(B, dtype=torch.long, device=device)
        context_mask = torch.ones(B, max_N, dtype=torch.bool, device=device)

        for i, s in enumerate(samples):
            N = s["length"]
            lengths[i] = N
            colors[i, :N] = torch.tensor(s["colors"], dtype=torch.long)
            xs_t[i, :N] = torch.tensor(s["xs"], dtype=torch.long)
            ys_t[i, :N] = torch.tensor(s["ys"], dtype=torch.long)
            roles[i, :N] = torch.tensor(s["roles"], dtype=torch.long)
            sep_mask[i, :N] = torch.tensor(s["sep_mask"], dtype=torch.bool)
            sep_types[i, :N] = torch.tensor(s["sep_types"], dtype=torch.long)
            grid_ids[i, :N] = torch.tensor(s["grid_ids"], dtype=torch.long)
            target_mask[i, :N] = torch.tensor(s["target_mask"], dtype=torch.bool)
            target_input_colors[i, :N] = torch.tensor(s["target_input_colors"], dtype=torch.long)

            tgt_positions = [j for j, m in enumerate(s["target_mask"]) if m]
            for j, pos in enumerate(tgt_positions):
                if j < len(s["target_colors"]):
                    target_labels[i, pos] = s["target_colors"][j]

            context_mask[i, :N] = ~target_mask[i, :N]

        input_ids = torch.zeros(B, max_N, dtype=torch.long, device=device)
        labels = torch.full((B, max_N), -100, dtype=torch.long, device=device)

        meta = {
            "colors": colors, "xs": xs_t, "ys": ys_t, "roles": roles,
            "sep_mask": sep_mask, "sep_types": sep_types, "grid_ids": grid_ids,
            "target_mask": target_mask, "target_labels": target_labels,
            "target_input_colors": target_input_colors, "context_mask": context_mask,
            "lengths": lengths,
        }

        return (input_ids, labels, meta), is_start, is_end, turn_idx
