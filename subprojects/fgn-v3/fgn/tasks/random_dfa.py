"""Random DFA Task — sequential state propagation through a random automaton.

K states, A alphabet symbols. Random transition table delta(state, symbol) -> state.
Input: sequence of symbols. Output: final state after applying all transitions.

Non-commutative, non-decomposable, no algebraic shortcuts.
The only solution is sequential state propagation.

Designed to maximally differentiate geometric sequential routing from flat
parallel attention. When K > d_model, the model cannot memorize the full
transition table in its parameters and must learn a routing strategy.
"""

import random
from typing import Dict, List, Optional, Tuple

import torch


class RandomDFATask:
    """Random DFA sequential state propagation task.

    Format: "DFA: s1 s2 s3 ... ? final_state"

    Args:
        tokenizer: HuggingFace tokenizer
        seq_len: maximum sequence length
        n_states: number of DFA states (K)
        n_symbols: alphabet size (A)
        min_steps: minimum number of transitions
        max_steps: maximum number of transitions
        seed: random seed for DFA construction (fixed DFA)
    """

    def __init__(self, tokenizer, seq_len: int = 512,
                 n_states: int = 256, n_symbols: int = 16,
                 min_steps: int = 20, max_steps: int = 50,
                 seed: int = 42, **kwargs):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.n_states = n_states
        self.n_symbols = n_symbols
        self.min_steps = min_steps
        self.max_steps = max_steps

        # Build random transition table (fixed for reproducibility)
        rng = random.Random(seed)
        # delta[state][symbol] -> next_state
        self.delta = [[rng.randint(0, n_states - 1) for _ in range(n_symbols)]
                      for _ in range(n_states)]

        # Pre-tokenize symbols and state numbers
        self._symbol_tokens = {}
        for s in range(n_symbols):
            tok = tokenizer.encode(f" s{s}", add_special_tokens=False)
            self._symbol_tokens[s] = tok

        self._state_tokens = {}
        for q in range(n_states):
            tok = tokenizer.encode(f" {q}", add_special_tokens=False)
            self._state_tokens[q] = tok

        # Prefix and separator tokens
        self._prefix = tokenizer.encode("DFA:", add_special_tokens=False)
        self._query = tokenizer.encode(" ?", add_special_tokens=False)

    def generate_batch(self, batch_size: int,
                       device: Optional[torch.device] = None
                       ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Generate a batch of DFA propagation sequences.

        Returns:
            (input_ids [B, seq_len], labels [B, seq_len], metadata dict)
        """
        pad_id = self.tokenizer.eos_token_id or 0
        all_input_ids = []
        all_labels = []

        for _ in range(batch_size):
            n_steps = random.randint(self.min_steps, self.max_steps)

            # Random initial state and symbol sequence
            state = random.randint(0, self.n_states - 1)
            initial_state = state
            symbols = [random.randint(0, self.n_symbols - 1) for _ in range(n_steps)]

            # Propagate through DFA
            for sym in symbols:
                state = self.delta[state][sym]
            final_state = state

            # Build token sequence: "DFA: s0 s3 s1 ... ? 42"
            tokens = list(self._prefix)
            # Add initial state token
            tokens.extend(self.tokenizer.encode(f" q{initial_state}",
                                                 add_special_tokens=False))
            for sym in symbols:
                tokens.extend(self._symbol_tokens[sym])
            tokens.extend(self._query)

            # Answer tokens (final state)
            answer_tokens = self._state_tokens[final_state]

            # Labels: -100 for input, actual tokens for answer
            input_len = len(tokens)
            full_tokens = tokens + answer_tokens

            labels = [-100] * input_len + answer_tokens

            # Truncate or pad
            if len(full_tokens) > self.seq_len:
                full_tokens = full_tokens[:self.seq_len]
                labels = labels[:self.seq_len]
            else:
                pad_len = self.seq_len - len(full_tokens)
                full_tokens = full_tokens + [pad_id] * pad_len
                labels = labels + [-100] * pad_len

            all_input_ids.append(full_tokens)
            all_labels.append(labels)

        input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        labels_t = torch.tensor(all_labels, dtype=torch.long)

        if device is not None:
            input_ids = input_ids.to(device)
            labels_t = labels_t.to(device)

        metadata = {
            "task": "random_dfa",
            "n_states": self.n_states,
            "n_symbols": self.n_symbols,
        }

        return input_ids, labels_t, metadata


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Small DFA for testing
    task = RandomDFATask(tokenizer, seq_len=128, n_states=32, n_symbols=8,
                         min_steps=5, max_steps=10)

    input_ids, labels, meta = task.generate_batch(4)
    print(f"input_ids: {input_ids.shape}")
    print(f"labels: {labels.shape}")
    print(f"metadata: {meta}")

    # Decode first example
    tokens = input_ids[0].tolist()
    text = tokenizer.decode(tokens)
    print(f"\nExample: {text[:200]}...")

    # Count supervised tokens
    sup = (labels[0] != -100).sum().item()
    print(f"Supervised tokens: {sup}")

    # Verify correctness
    print("\nDFA transition table (first 4 states):")
    for q in range(min(4, task.n_states)):
        print(f"  State {q}: {task.delta[q][:4]}...")

    print("\nRandomDFATask OK")
