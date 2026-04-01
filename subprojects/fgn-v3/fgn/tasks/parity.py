"""Task P: Parity with Length Generalization.

Proven transformer failure mode (Hahn 2020). Given a binary string,
output whether the count of 1s is odd or even.

Transformers learn parallel counting shortcuts (sum mod 2) that fail on:
- Longer sequences than training distribution
- Different bit density than training
- Combined length + density shifts

LSTMs solve this trivially via recurrent XOR state update.
FGN's metric tensor could enable sequential scanning behavior.

Training: length 40, p(1) = 0.5
Eval OOD: lengths 60-200, p(1) in {0.1, 0.3, 0.7, 0.9}
"""

import random
import torch
from torch import Tensor
from typing import Tuple, Dict, Any, List, Optional


class ParityTask:
    """Binary string parity classification.

    Each sequence is a space-separated binary string followed by a single
    answer token ("odd" or "even"). The model must learn to count 1s mod 2.

    Args:
        tokenizer: GPT-2 tokenizer
        seq_len: Maximum sequence length for padding (default: 256)
        bit_length: Number of bits per sequence (default: 40)
        p_one: Probability of each bit being 1 (default: 0.5)
    """

    def __init__(
        self,
        tokenizer,
        seq_len: int = 256,
        bit_length: int = 40,
        p_one: float = 0.5,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.bit_length = bit_length
        self.p_one = p_one
        self.pad_token_id = tokenizer.eos_token_id

        # Pre-tokenize fixed parts
        self._zero_ids = tokenizer.encode(" 0", add_special_tokens=False)
        self._one_ids = tokenizer.encode(" 1", add_special_tokens=False)
        assert len(self._zero_ids) == 1, f"' 0' = {self._zero_ids}"
        assert len(self._one_ids) == 1, f"' 1' = {self._one_ids}"
        self.zero_token = self._zero_ids[0]
        self.one_token = self._one_ids[0]

        self._odd_ids = tokenizer.encode(" odd", add_special_tokens=False)
        self._even_ids = tokenizer.encode(" even", add_special_tokens=False)
        assert len(self._odd_ids) == 1, f"' odd' = {self._odd_ids}"
        assert len(self._even_ids) == 1, f"' even' = {self._even_ids}"
        self.odd_token = self._odd_ids[0]
        self.even_token = self._even_ids[0]

        self._prefix = tokenizer.encode("Parity:", add_special_tokens=False)
        self._answer_sep = tokenizer.encode(" Answer:", add_special_tokens=False)

    def generate_batch(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        """Generate a batch of parity examples.

        Format: "Parity: 0 1 1 0 ... Answer: odd"
        Only the final answer token is supervised.

        Returns:
            (input_ids [B, seq_len], labels [B, seq_len], metadata)
        """
        if device is None:
            device = torch.device("cpu")

        input_ids_list: List[List[int]] = []
        labels_list: List[List[int]] = []

        for _ in range(batch_size):
            # Generate random bits
            bits = [1 if random.random() < self.p_one else 0
                    for _ in range(self.bit_length)]
            parity = sum(bits) % 2  # 1 = odd, 0 = even
            answer_token = self.odd_token if parity == 1 else self.even_token

            # Build token sequence: "Parity:" + bits + " Answer:" + answer
            tokens: List[int] = list(self._prefix)
            for b in bits:
                tokens.append(self.one_token if b == 1 else self.zero_token)
            tokens.extend(self._answer_sep)
            tokens.append(answer_token)

            # Truncate if overflow (shouldn't happen for reasonable params)
            if len(tokens) > self.seq_len:
                tokens = tokens[:self.seq_len]

            answer_pos = len(tokens) - 1

            # Pad
            input_ids = tokens + [self.pad_token_id] * (self.seq_len - len(tokens))
            input_ids = input_ids[:self.seq_len]

            # Labels: only the answer token
            labels = [-100] * self.seq_len
            if answer_pos < self.seq_len:
                labels[answer_pos] = answer_token

            input_ids_list.append(input_ids)
            labels_list.append(labels)

        input_ids_tensor = torch.tensor(
            input_ids_list, dtype=torch.long, device=device
        )
        labels_tensor = torch.tensor(labels_list, dtype=torch.long, device=device)

        metadata = {
            "task": "P",
            "task_name": "parity",
            "bit_length": self.bit_length,
            "p_one": self.p_one,
            "batch_size": batch_size,
        }

        return input_ids_tensor, labels_tensor, metadata


if __name__ == "__main__":
    from transformers import GPT2Tokenizer

    print("Parity Task - Smoke Test")
    print("=" * 60)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # Training config
    task = ParityTask(tokenizer, seq_len=256, bit_length=40, p_one=0.5)
    input_ids, labels, meta = task.generate_batch(batch_size=4)
    print(f"\nTraining (len=40, p=0.5):")
    print(f"  Shape: {input_ids.shape}")
    n_pad = (input_ids[0] == tokenizer.eos_token_id).sum().item()
    n_sup = (labels[0] != -100).sum().item()
    print(f"  Content tokens: {input_ids.shape[1] - n_pad}, supervised: {n_sup}")

    # Decode first sample
    clean = [t for t in input_ids[0].tolist() if t != tokenizer.eos_token_id]
    print(f"  {tokenizer.decode(clean)}")
    sup_ids = [input_ids[0][i].item() for i, l in enumerate(labels[0].tolist()) if l != -100]
    print(f"  Answer token: {tokenizer.decode(sup_ids)}")

    # OOD configs
    for desc, bl, p in [
        ("Length 100", 100, 0.5),
        ("Length 200", 200, 0.5),
        ("p=0.1", 40, 0.1),
        ("p=0.9", 40, 0.9),
        ("Combined", 100, 0.3),
    ]:
        task_ood = ParityTask(tokenizer, seq_len=256, bit_length=bl, p_one=p)
        ids, labs, _ = task_ood.generate_batch(batch_size=4)
        clean = [t for t in ids[0].tolist() if t != tokenizer.eos_token_id]
        n_ones = sum(1 for t in ids[0].tolist() if t == task_ood.one_token)
        print(f"\n  {desc} (len={bl}, p={p}): {len(clean)} tokens, ones={n_ones}")
        decoded = tokenizer.decode(clean[:80])
        print(f"    {decoded}...")

    print("\n" + "=" * 60)
    print("Parity smoke test complete!")
