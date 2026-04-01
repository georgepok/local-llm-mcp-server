"""Task F: Arithmetic chain evaluation — variable-depth computation graphs.

Each sequence is a small program: variable assignments with arithmetic operations.
The model must evaluate the program to produce the final value of a queried variable.
All operations are mod 100 (values bounded to 0-99, answers are 0-99).

Key difficulty lever: dependency depth of the computation graph.
A 6-layer transformer can evaluate chains up to depth ~5.
Chains of depth 6+ genuinely exceed the model's circuit depth.

With shuffle=True, assignments are in random order — the model must ALSO
perform dependency resolution (topological sort) before evaluation.
This creates natural mode-switching: graph traversal + sequential arithmetic.

Multiple independent programs per sequence test within-sequence context switching.
"""

import random
import torch
from torch import Tensor
from typing import Tuple, Dict, Any, List, Optional


# Letter ranges for up to 3 independent programs per sequence
_PROG_LETTERS = [
    "abcdefgh",      # Program 1: a-h (8 vars)
    "ijklmnop",      # Program 2: i-p (8 vars)
    "qrstuvwx",      # Program 3: q-x (8 vars)
]


class ArithmeticChainTask:
    """Evaluate arithmetic chains with variable-depth computation graphs.

    Args:
        tokenizer: GPT-2 tokenizer with vocab_size=50304
        seq_len: Target sequence length (default: 512)
        num_vars: Variables per program (default: 8, max 8)
        min_depth: Minimum dependency chain depth (default: 4)
        max_depth: Maximum dependency chain depth (default: 8)
        shuffle: Shuffle assignment order within each program (default: True)
        n_programs: Independent programs per sequence (1-3, default: 3)
    """

    OPS = ["+", "-", "*"]

    def __init__(
        self,
        tokenizer,
        seq_len: int = 512,
        num_vars: int = 8,
        min_depth: int = 4,
        max_depth: int = 8,
        shuffle: bool = True,
        n_programs: int = 3,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.pad_token_id = tokenizer.eos_token_id
        self.num_vars = min(num_vars, 8)
        self.min_depth = min_depth
        self.max_depth = min(max_depth, self.num_vars - 1)
        self.shuffle = shuffle
        self.n_programs = min(n_programs, 3)

    @staticmethod
    def _eval_mod(a: int, op: str, b: int) -> int:
        """Evaluate a op b mod 100."""
        if op == "+":
            return (a + b) % 100
        elif op == "-":
            return (a - b) % 100
        elif op == "*":
            return (a * b) % 100
        return 0

    def _generate_program(
        self, letters: str
    ) -> Tuple[List[Tuple[str, str]], str, int, int]:
        """Generate a single arithmetic program using given letter range.

        Returns:
            (assignments [(name, expr_str), ...], query_var, answer, depth)
        """
        depth = random.randint(self.min_depth, self.max_depth)
        var_names = list(letters[: self.num_vars])

        vals: Dict[str, int] = {}
        assignments: List[Tuple[str, str]] = []

        # First 2 variables are constants
        for i in range(2):
            val = random.randint(1, 50)
            vals[var_names[i]] = val
            assignments.append((var_names[i], str(val)))

        # Backbone chain guaranteeing minimum depth
        backbone = [var_names[0]]
        n_backbone = min(2 + depth, self.num_vars)

        for i in range(2, n_backbone):
            name = var_names[i]
            prev = backbone[-1]
            op = random.choice(self.OPS)

            # Sometimes depend on two variables
            if random.random() < 0.4 and len(vals) > 1:
                others = [v for v in vals if v != prev]
                other = random.choice(others)
                val = self._eval_mod(vals[prev], op, vals[other])
                expr = f"{prev} {op} {other}"
            else:
                const = random.randint(1, 15)
                val = self._eval_mod(vals[prev], op, const)
                expr = f"{prev} {op} {const}"

            vals[name] = val
            assignments.append((name, expr))
            backbone.append(name)

        # Side-chain variables (distractors)
        for i in range(n_backbone, self.num_vars):
            name = var_names[i]
            dep = random.choice(list(vals.keys()))
            op = random.choice(self.OPS)
            const = random.randint(1, 15)
            val = self._eval_mod(vals[dep], op, const)
            expr = f"{dep} {op} {const}"
            vals[name] = val
            assignments.append((name, expr))

        query_var = backbone[-1]
        answer = vals[query_var]

        if self.shuffle:
            random.shuffle(assignments)

        return assignments, query_var, answer, depth

    def generate_batch(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        """Generate a batch of arithmetic chain evaluation examples.

        Each sequence contains n_programs independent programs using
        non-overlapping variable names (a-h, i-p, q-x).

        Returns:
            Tuple of (input_ids, labels, metadata)
        """
        if device is None:
            device = torch.device("cpu")

        input_ids_list: List[List[int]] = []
        labels_list: List[List[int]] = []
        depths: List[int] = []

        for _ in range(batch_size):
            program_texts = []
            answer_parts = []
            total_depth = 0

            for prog_idx in range(self.n_programs):
                letters = _PROG_LETTERS[prog_idx]
                assignments, query_var, answer, depth = self._generate_program(
                    letters
                )

                parts = [f"{name} = {expr}" for name, expr in assignments]
                prog_text = " ; ".join(parts) + f" ; {query_var} = ?"
                program_texts.append(prog_text)
                answer_parts.append(str(answer))
                total_depth += depth

            # Build full sequence
            context = " | ".join(program_texts)
            answer_text = " ".join(answer_parts)
            full_text = f"Eval mod 100: {context} Answer: {answer_text}"

            # Tokenize
            full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
            answer_token_text = f" Answer: {answer_text}"
            answer_ids = self.tokenizer.encode(
                answer_token_text, add_special_tokens=False
            )

            # Truncation (keep answer, truncate from beginning)
            if len(full_ids) > self.seq_len:
                truncate_amount = len(full_ids) - self.seq_len
                full_ids = full_ids[truncate_amount:]

            answer_start = len(full_ids) - len(answer_ids)

            # Pad
            input_ids = full_ids + [self.pad_token_id] * (
                self.seq_len - len(full_ids)
            )
            input_ids = input_ids[: self.seq_len]

            # Labels: only supervise answer tokens
            labels = [-100] * self.seq_len
            for i in range(len(answer_ids)):
                pos = answer_start + i
                if 0 <= pos < self.seq_len and pos < len(full_ids):
                    labels[pos] = full_ids[pos]

            input_ids_list.append(input_ids)
            labels_list.append(labels)
            depths.append(total_depth // self.n_programs)

        input_ids_tensor = torch.tensor(
            input_ids_list, dtype=torch.long, device=device
        )
        labels_tensor = torch.tensor(labels_list, dtype=torch.long, device=device)

        metadata = {
            "task": "F",
            "task_name": "arithmetic",
            "difficulty": int(sum(depths) / len(depths)),
            "n_entities": self.num_vars * self.n_programs,
            "batch_size": batch_size,
        }

        return input_ids_tensor, labels_tensor, metadata


if __name__ == "__main__":
    from transformers import GPT2Tokenizer

    print("Arithmetic Chain Task - Smoke Test")
    print("=" * 60)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if len(tokenizer) != 50304:
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    f"<pad_{i}>" for i in range(50304 - len(tokenizer))
                ]
            }
        )

    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # Test different depth settings
    for depth_label, min_d, max_d in [("easy", 3, 4), ("medium", 5, 6), ("hard", 7, 7)]:
        task = ArithmeticChainTask(
            tokenizer, seq_len=512,
            num_vars=8, min_depth=min_d, max_depth=max_d,
            shuffle=True, n_programs=3,
        )
        input_ids, labels, meta = task.generate_batch(batch_size=4)
        non_pad = (input_ids[0] != tokenizer.eos_token_id).sum().item()
        n_sup = (labels[0] != -100).sum().item()
        print(f"\n  {depth_label} (depth {min_d}-{max_d}): {non_pad} tokens, {n_sup} supervised")

        # Show first sample
        sample_ids = input_ids[0].tolist()
        clean = [t for t in sample_ids if t != tokenizer.eos_token_id]
        decoded = tokenizer.decode(clean)
        print(f"    {decoded[:400]}")
        sup_ids = [sample_ids[i] for i, l in enumerate(labels[0].tolist()) if l != -100]
        if sup_ids:
            print(f"    Answer: {tokenizer.decode(sup_ids)}")

    # Verify correctness: non-shuffled, single program
    print(f"\n{'='*60}")
    print("Correctness check (no shuffle, 1 program):")
    task = ArithmeticChainTask(
        tokenizer, seq_len=512,
        num_vars=6, min_depth=4, max_depth=4,
        shuffle=False, n_programs=1,
    )
    for _ in range(3):
        input_ids, labels, _ = task.generate_batch(batch_size=1)
        clean = [t for t in input_ids[0].tolist() if t != tokenizer.eos_token_id]
        decoded = tokenizer.decode(clean)
        print(f"  {decoded}")

    print("\n" + "=" * 60)
    print("Test completed!")
