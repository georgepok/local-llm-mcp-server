"""Task S: Cumulative State Tracking with Distractors.

Track a running accumulator through a sequence containing update tokens
(add/subtract mod 100) and distractor tokens. At random query positions,
output the current accumulator value.

Combines two transformer failure modes:
  (a) Attention dilution: as sequence grows and distractors increase,
      soft attention weight on relevant update tokens decreases as ~1/T.
  (b) Sequential state propagation: chaining operations in order requires
      sequential computation that transformers approximate with parallel
      shortcuts that break at scale.

Training: 128 tokens, 50% distractors, 3-5 queries
Eval OOD: lengths 256/512/1024, distractor ratios 70/80/90%
"""

import random
import torch
from torch import Tensor
from typing import Tuple, Dict, Any, List, Optional


class StateTrackingTask:
    """Cumulative state tracking with distractors.

    Format: "State 42 | add 7 | cat | sub 3 | ? 46 | add 12 | dog | ? 55 |"
    Supervision only on the number tokens after "?" markers.

    Args:
        tokenizer: GPT-2 tokenizer
        seq_len: Maximum sequence length (default: 512)
        n_events: Number of events per sequence (default: 60)
        distractor_ratio: Fraction that are distractors (default: 0.5)
        n_queries: Query positions per sequence (default: 4)
    """

    FILLERS = [
        "the", "cat", "dog", "red", "blue", "big", "run", "sky",
        "tree", "book", "fish", "rain", "sun", "cold", "hot", "old",
        "new", "dark", "soft", "hard", "long", "wide", "green", "white",
        "black", "brown", "pink", "gold", "fast", "slow", "deep", "thin",
        "tall", "flat", "dry", "wet", "far", "near", "high", "low",
    ]

    def __init__(
        self,
        tokenizer,
        seq_len: int = 512,
        n_events: int = 0,
        distractor_ratio: float = 0.5,
        n_queries: int = 4,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        # Auto-scale events to fill ~70% of seq_len (~2.5 tokens per event)
        self.n_events = n_events if n_events > 0 else max(30, seq_len * 7 // 25)
        self.distractor_ratio = distractor_ratio
        self.n_queries = n_queries
        self.pad_token_id = tokenizer.eos_token_id

    def _tokenize(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def generate_batch(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        """Generate a batch of state tracking examples.

        Returns:
            (input_ids [B, seq_len], labels [B, seq_len], metadata)
        """
        if device is None:
            device = torch.device("cpu")

        input_ids_list: List[List[int]] = []
        labels_list: List[List[int]] = []

        for _ in range(batch_size):
            state = random.randint(0, 99)

            # Decide event types
            n_distractors = int(self.n_events * self.distractor_ratio)
            n_updates = self.n_events - n_distractors - self.n_queries

            events = (
                ["update"] * max(n_updates, 1)
                + ["distract"] * n_distractors
                + ["query"] * self.n_queries
            )
            random.shuffle(events)

            # Ensure first event is an update (need state change before query)
            if events[0] == "query":
                for i in range(1, len(events)):
                    if events[i] == "update":
                        events[0], events[i] = events[i], events[0]
                        break

            # Build token sequence segment by segment
            segments: List[Tuple[List[int], bool]] = []  # (token_ids, is_supervised)

            # Prefix: "State XX |"
            segments.append((self._tokenize(f"State {state} |"), False))
            current_state = state

            for event in events:
                if event == "update":
                    op = random.choice(["add", "sub"])
                    val = random.randint(1, 20)
                    if op == "add":
                        current_state = (current_state + val) % 100
                    else:
                        current_state = (current_state - val) % 100
                    segments.append((self._tokenize(f" {op} {val} |"), False))

                elif event == "distract":
                    word = random.choice(self.FILLERS)
                    segments.append((self._tokenize(f" {word} |"), False))

                elif event == "query":
                    # "?" prefix (not supervised)
                    segments.append((self._tokenize(" ?"), False))
                    # Answer number (supervised)
                    segments.append((self._tokenize(f" {current_state}"), True))
                    # Separator
                    segments.append((self._tokenize(" |"), False))

            # Concatenate all segments
            full_ids: List[int] = []
            sup_mask: List[bool] = []
            for seg_ids, is_sup in segments:
                full_ids.extend(seg_ids)
                sup_mask.extend([is_sup] * len(seg_ids))

            # Truncate
            if len(full_ids) > self.seq_len:
                full_ids = full_ids[:self.seq_len]
                sup_mask = sup_mask[:self.seq_len]

            # Build labels
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
            "task": "S",
            "task_name": "state_tracking",
            "n_events": self.n_events,
            "distractor_ratio": self.distractor_ratio,
            "n_queries": self.n_queries,
            "avg_supervised_tokens": n_supervised,
            "batch_size": batch_size,
        }

        return input_ids_tensor, labels_tensor, metadata


if __name__ == "__main__":
    from transformers import GPT2Tokenizer

    print("State Tracking Task - Smoke Test")
    print("=" * 60)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Training config
    task = StateTrackingTask(
        tokenizer, seq_len=512, n_events=60, distractor_ratio=0.5, n_queries=4
    )
    input_ids, labels, meta = task.generate_batch(batch_size=4)
    n_pad = (input_ids[0] == tokenizer.eos_token_id).sum().item()
    n_sup = (labels[0] != -100).sum().item()
    print(f"\nTraining (60 events, 50% distractors, 4 queries):")
    print(f"  Shape: {input_ids.shape}")
    print(f"  Content tokens: {input_ids.shape[1] - n_pad}, supervised: {n_sup}")

    clean = [t for t in input_ids[0].tolist() if t != tokenizer.eos_token_id]
    decoded = tokenizer.decode(clean)
    print(f"  {decoded[:300]}")

    # Show supervised tokens
    sup_ids = [input_ids[0][i].item() for i, l in enumerate(labels[0].tolist()) if l != -100]
    if sup_ids:
        print(f"  Supervised answers: {tokenizer.decode(sup_ids)}")

    # OOD configs
    for desc, ne, dr in [
        ("256 tokens, 50%", 120, 0.5),
        ("128 tokens, 70%", 60, 0.7),
        ("128 tokens, 90%", 60, 0.9),
    ]:
        t = StateTrackingTask(tokenizer, seq_len=512, n_events=ne, distractor_ratio=dr)
        ids, labs, m = t.generate_batch(batch_size=4)
        n_pad_ood = (ids[0] == tokenizer.eos_token_id).sum().item()
        n_sup_ood = (labs[0] != -100).sum().item()
        print(f"\n  {desc}: {ids.shape[1] - n_pad_ood} content, {n_sup_ood} supervised")

    print("\n" + "=" * 60)
    print("State tracking smoke test complete!")
