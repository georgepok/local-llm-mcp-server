"""Task H: Affine Group Composition over Z₉₇.

Compute the cumulative affine transformation x → ax + b (mod 97) from a
sequence of mul/add operations.  The affine group Aff(Z₉₇) has p*(p-1) =
97*96 = 9,312 elements — far too large to memorize.

Non-commutativity: mul k; add j ≠ add j; mul k.
  mul 3 then add 5: x → 3x + 5
  add 5 then mul 3: x → 3(x+5) = 3x + 15

Unlike S₅, the affine group is NOT decomposable into independent
parallel sub-tasks.  The "b" component depends on all preceding "a"
values, creating genuine sequential dependence.

State: (a, b) where the cumulative transform is x → ax + b (mod 97).
  mul k: (a, b) → (a*k, b*k)   mod 97
  add k: (a, b) → (a,   b+k)   mod 97

Supervised tokens: two numbers (a mod 97, b mod 97) at checkpoint positions.

Training: 50-100 ops, supervision every N-th
Eval OOD: longer chains, sparser supervision
"""

import random
import torch
from torch import Tensor
from typing import Tuple, Dict, Any, List, Optional

P = 97  # prime modulus


class AffineGroupTask:
    """Affine group composition over Z₉₇ with sparse supervision.

    Format: "Affine: mul 3 ; add 5 ; mul 2 ; add 7 ; ... = 6 17 ; ... Answer: 47 23"

    Intermediate "= a b" checkpoints appear every `sup_every` operations.
    The final answer is always supervised. Checkpoints are also supervised.

    Args:
        tokenizer: GPT-2 tokenizer
        seq_len: Maximum sequence length (default: 512)
        min_ops: Minimum operations per sequence (default: 50)
        max_ops: Maximum operations per sequence (default: 100)
        sup_every: Supervise intermediate state every N ops (default: 10)
        mul_ratio: Fraction of operations that are mul vs add (default: 0.5)
    """

    def __init__(
        self,
        tokenizer,
        seq_len: int = 512,
        min_ops: int = 50,
        max_ops: int = 100,
        sup_every: int = 10,
        mul_ratio: float = 0.5,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.min_ops = min_ops
        self.max_ops = max_ops
        self.sup_every = sup_every
        self.mul_ratio = mul_ratio
        self.pad_token_id = tokenizer.eos_token_id

    def _tokenize(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _random_op(self) -> Tuple[str, int]:
        """Return (op_type, operand) where op_type is 'mul' or 'add'."""
        if random.random() < self.mul_ratio:
            # mul by nonzero value (1..96)
            k = random.randint(1, P - 1)
            return ("mul", k)
        else:
            # add any value (0..96), but prefer nonzero
            k = random.randint(1, P - 1)
            return ("add", k)

    def generate_batch(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        """Generate a batch of affine group composition examples.

        Returns:
            (input_ids [B, seq_len], labels [B, seq_len], metadata)
        """
        if device is None:
            device = torch.device("cpu")

        input_ids_list: List[List[int]] = []
        labels_list: List[List[int]] = []
        n_ops_list: List[int] = []

        for _ in range(batch_size):
            n_ops = random.randint(self.min_ops, self.max_ops)
            n_ops_list.append(n_ops)

            ops = [self._random_op() for _ in range(n_ops)]

            segments: List[Tuple[List[int], bool]] = []

            # Prefix
            segments.append((self._tokenize("Affine:"), False))

            # Apply operations and track state
            a, b = 1, 0  # identity: x → 1*x + 0

            for idx, (op_type, k) in enumerate(ops):
                if op_type == "mul":
                    a = (a * k) % P
                    b = (b * k) % P
                    op_text = f" mul {k}"
                else:
                    b = (b + k) % P
                    op_text = f" add {k}"

                segments.append((self._tokenize(op_text), False))

                # Check if this is a supervision checkpoint
                is_last = (idx == n_ops - 1)
                is_checkpoint = ((idx + 1) % self.sup_every == 0)

                if is_checkpoint or is_last:
                    # " =" separator (not supervised)
                    segments.append((self._tokenize(" ="), False))
                    # State output (supervised): " a b"
                    state_text = f" {a} {b}"
                    segments.append((self._tokenize(state_text), True))

                if not is_last:
                    segments.append((self._tokenize(" ;"), False))

            # Concatenate
            full_ids: List[int] = []
            sup_mask: List[bool] = []
            for seg_ids, is_sup in segments:
                full_ids.extend(seg_ids)
                sup_mask.extend([is_sup] * len(seg_ids))

            # Truncate if needed
            if len(full_ids) > self.seq_len:
                full_ids = full_ids[:self.seq_len]
                sup_mask = sup_mask[:self.seq_len]

            # Labels: -100 for unsupervised positions
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
            "task": "H",
            "task_name": "affine",
            "avg_ops": sum(n_ops_list) / len(n_ops_list),
            "sup_every": self.sup_every,
            "avg_supervised_tokens": n_supervised,
            "batch_size": batch_size,
        }

        return input_ids_tensor, labels_tensor, metadata


if __name__ == "__main__":
    from transformers import GPT2Tokenizer

    print("Affine Group Composition (Z₉₇) - Smoke Test")
    print("=" * 60)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Training config: 50-100 ops, supervision every 10th
    task = AffineGroupTask(
        tokenizer, seq_len=512, min_ops=50, max_ops=100, sup_every=10
    )
    input_ids, labels, meta = task.generate_batch(batch_size=4)
    n_pad = (input_ids[0] == tokenizer.eos_token_id).sum().item()
    n_sup = (labels[0] != -100).sum().item()
    print(f"\nTraining (50-100 ops, sup every 10th):")
    print(f"  Shape: {input_ids.shape}")
    print(f"  Content tokens: {input_ids.shape[1] - n_pad}, supervised: {n_sup}")
    print(f"  Metadata: {meta}")

    clean = [t for t in input_ids[0].tolist() if t != tokenizer.eos_token_id]
    decoded = tokenizer.decode(clean)
    print(f"  {decoded[:300]}")

    # Show supervised tokens
    sup_tokens = [input_ids[0][i].item() for i, l in enumerate(labels[0].tolist()) if l != -100]
    print(f"  Supervised: {tokenizer.decode(sup_tokens)}")

    # Correctness check
    print("\nCorrectness check:")
    a, b = 1, 0
    print(f"  Start: ({a}, {b})")
    a = (a * 3) % P; b = (b * 3) % P
    print(f"  mul 3: ({a}, {b})")
    b = (b + 5) % P
    print(f"  add 5: ({a}, {b})")
    a = (a * 2) % P; b = (b * 2) % P
    print(f"  mul 2: ({a}, {b})")
    b = (b + 7) % P
    print(f"  add 7: ({a}, {b})")
    print(f"  Transform: x → {a}x + {b} (mod {P})")
    print(f"  Check: f(1) = {(a * 1 + b) % P}, f(10) = {(a * 10 + b) % P}")

    # Non-commutativity check
    print("\nNon-commutativity check:")
    a1, b1 = 1, 0
    a1 = (a1 * 3) % P; b1 = (b1 * 3) % P; b1 = (b1 + 5) % P
    print(f"  mul 3 then add 5: ({a1}, {b1})  →  x → {a1}x + {b1}")
    a2, b2 = 1, 0
    b2 = (b2 + 5) % P; a2 = (a2 * 3) % P; b2 = (b2 * 3) % P
    print(f"  add 5 then mul 3: ({a2}, {b2})  →  x → {a2}x + {b2}")
    assert (a1, b1) != (a2, b2), "Should be non-commutative!"
    print("  PASS: Different results confirm non-commutativity")

    # Final-only supervision test
    task2 = AffineGroupTask(
        tokenizer, seq_len=512, min_ops=100, max_ops=100, sup_every=100
    )
    ids2, labs2, m2 = task2.generate_batch(batch_size=4)
    n_pad2 = (ids2[0] == tokenizer.eos_token_id).sum().item()
    n_sup2 = (labs2[0] != -100).sum().item()
    print(f"\n  100 ops final-only: {ids2.shape[1] - n_pad2} content, {n_sup2} supervised")

    print("\n" + "=" * 60)
    print("Affine group smoke test complete!")
