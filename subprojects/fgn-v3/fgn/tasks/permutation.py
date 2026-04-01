"""Task G: Permutation Composition (Symmetric Group S₅).

Compute the composition of a sequence of permutations from S₅ (the group
of all permutations of 5 elements, |S₅| = 120 states).

Proven transformer failure mode (Liu et al.). The recurrent structure
q_t = δ(q_{t-1}, σ_t) fits RNN computation naturally. With 120 states,
the model cannot memorize — it must learn the composition algorithm.

Sparse supervision variant: provide intermediate compositions at every
k-th permutation. Transformers collapse under sparse supervision while
LSTMs maintain accuracy.

Training: 20-50 perms, supervision every 5th
Eval OOD: 60-100 perms, supervision every 10th
"""

import random
import torch
from torch import Tensor
from typing import Tuple, Dict, Any, List, Optional


def compose_perm(state: List[int], sigma: List[int]) -> List[int]:
    """Apply permutation sigma to current state.

    Convention: state[i] = element at position i (0-indexed internally).
    sigma[i] = position to pull from.
    new_state[i] = state[sigma[i]]
    """
    return [state[sigma[i]] for i in range(len(state))]


class PermutationTask:
    """S₅ permutation composition with sparse supervision.

    Format: "Compose: 3 1 4 5 2 ; 2 4 1 3 5 ; ... = 4 2 1 5 3 ; ... Answer: 5 3 1 2 4"

    Intermediate "= X X X X X" checkpoints appear every `sup_every` permutations.
    The final answer is always supervised. Checkpoints are also supervised.

    Args:
        tokenizer: GPT-2 tokenizer
        seq_len: Maximum sequence length (default: 512)
        min_perms: Minimum permutations per sequence (default: 20)
        max_perms: Maximum permutations per sequence (default: 50)
        sup_every: Supervise intermediate state every N perms (default: 5)
    """

    def __init__(
        self,
        tokenizer,
        seq_len: int = 512,
        min_perms: int = 20,
        max_perms: int = 50,
        sup_every: int = 5,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.min_perms = min_perms
        self.max_perms = max_perms
        self.sup_every = sup_every
        self.pad_token_id = tokenizer.eos_token_id

        # All 120 permutations of S₅ (0-indexed internally, 1-indexed display)
        from itertools import permutations as iterperms
        self._all_perms = [list(p) for p in iterperms(range(5))]

    def _random_perm(self) -> List[int]:
        return list(random.choice(self._all_perms))

    def _display(self, perm: List[int]) -> str:
        """Convert 0-indexed internal perm to 1-indexed display string."""
        return " ".join(str(x + 1) for x in perm)

    def _tokenize(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def generate_batch(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        """Generate a batch of permutation composition examples.

        Returns:
            (input_ids [B, seq_len], labels [B, seq_len], metadata)
        """
        if device is None:
            device = torch.device("cpu")

        input_ids_list: List[List[int]] = []
        labels_list: List[List[int]] = []
        n_perms_list: List[int] = []

        for _ in range(batch_size):
            n_perms = random.randint(self.min_perms, self.max_perms)
            n_perms_list.append(n_perms)
            perms = [self._random_perm() for _ in range(n_perms)]

            segments: List[Tuple[List[int], bool]] = []

            # Prefix
            segments.append((self._tokenize("Compose:"), False))

            # Compose permutations left-to-right
            state = list(range(5))  # identity [0,1,2,3,4]

            for idx, sigma in enumerate(perms):
                state = compose_perm(state, sigma)

                # Input permutation (1-indexed display)
                perm_text = " " + self._display(sigma)
                segments.append((self._tokenize(perm_text), False))

                # Check if this is a supervision checkpoint
                is_last = (idx == n_perms - 1)
                is_checkpoint = ((idx + 1) % self.sup_every == 0)

                if is_checkpoint or is_last:
                    # " =" separator (not supervised)
                    segments.append((self._tokenize(" ="), False))
                    # State output (supervised)
                    state_text = " " + self._display(state)
                    segments.append((self._tokenize(state_text), True))

                if not is_last:
                    # Separator between permutations
                    segments.append((self._tokenize(" ;"), False))

            # Concatenate
            full_ids: List[int] = []
            sup_mask: List[bool] = []
            for seg_ids, is_sup in segments:
                full_ids.extend(seg_ids)
                sup_mask.extend([is_sup] * len(seg_ids))

            # Truncate
            if len(full_ids) > self.seq_len:
                full_ids = full_ids[:self.seq_len]
                sup_mask = sup_mask[:self.seq_len]

            # Labels
            labels = [-100] * self.seq_len
            for i in range(len(full_ids)):
                if sup_mask[i]:
                    labels[i] = full_ids[i]

            # Pad
            input_ids = full_ids + [self.pad_token_id] * (self.seq_len - len(full_ids))
            input_ids = input_ids[:self.seq_len]

            input_ids_list.append(input_ids)
            labels_list.append(labels)

        input_ids_tensor = torch.tensor(
            input_ids_list, dtype=torch.long, device=device
        )
        labels_tensor = torch.tensor(labels_list, dtype=torch.long, device=device)

        n_supervised = sum(
            1 for lab in labels_list for l in lab if l != -100
        ) / batch_size

        metadata = {
            "task": "G",
            "task_name": "permutation",
            "avg_perms": sum(n_perms_list) / len(n_perms_list),
            "sup_every": self.sup_every,
            "avg_supervised_tokens": n_supervised,
            "batch_size": batch_size,
        }

        return input_ids_tensor, labels_tensor, metadata


if __name__ == "__main__":
    from transformers import GPT2Tokenizer

    print("Permutation Composition Task (S₅) - Smoke Test")
    print("=" * 60)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Training config: 20-50 perms, supervision every 5th
    task = PermutationTask(
        tokenizer, seq_len=512, min_perms=20, max_perms=50, sup_every=5
    )
    input_ids, labels, meta = task.generate_batch(batch_size=4)
    n_pad = (input_ids[0] == tokenizer.eos_token_id).sum().item()
    n_sup = (labels[0] != -100).sum().item()
    print(f"\nTraining (20-50 perms, sup every 5th):")
    print(f"  Shape: {input_ids.shape}")
    print(f"  Content tokens: {input_ids.shape[1] - n_pad}, supervised: {n_sup}")

    clean = [t for t in input_ids[0].tolist() if t != tokenizer.eos_token_id]
    decoded = tokenizer.decode(clean)
    print(f"  {decoded[:300]}")

    # Verify composition correctness
    print(f"\n  Supervised tokens: {tokenizer.decode([input_ids[0][i].item() for i, l in enumerate(labels[0].tolist()) if l != -100])}")

    # Manual verification: compose 3 known permutations
    print("\nCorrectness check:")
    from itertools import permutations as iterperms
    s1 = [1, 0, 2, 3, 4]  # swap 0,1
    s2 = [0, 2, 1, 3, 4]  # swap 1,2
    state = list(range(5))
    state = compose_perm(state, s1)
    print(f"  After swap(0,1): {[x+1 for x in state]}")
    state = compose_perm(state, s2)
    print(f"  After swap(1,2): {[x+1 for x in state]}")
    # Expected: [1,2,0,3,4] 0-indexed = [2,3,1,4,5] 1-indexed

    # Longer sequence test
    task_long = PermutationTask(
        tokenizer, seq_len=512, min_perms=40, max_perms=40, sup_every=10
    )
    ids, labs, m = task_long.generate_batch(batch_size=4)
    n_pad_l = (ids[0] == tokenizer.eos_token_id).sum().item()
    n_sup_l = (labs[0] != -100).sum().item()
    print(f"\n  40 perms, sup every 10th: {ids.shape[1] - n_pad_l} content, {n_sup_l} supervised")

    print("\n" + "=" * 60)
    print("Permutation smoke test complete!")
