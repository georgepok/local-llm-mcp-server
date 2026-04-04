"""Text task for multi-domain training (Stage B).

Produces tokenized text batches from WikiText-2 (or any HuggingFace text dataset).
The model processes text through a learned text embedding → shared ODE dynamics →
text output head (next-token prediction).

Text uses the SAME ContinuousDynamics as ARC — the hypothesis is that multi-domain
pressure will activate the low-rank metric terms for rotational geometry that
ARC alone doesn't need.
"""

import random
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class TextTask:
    """Generates tokenized text batches for next-token prediction."""

    def __init__(
        self,
        seq_len: int = 2048,
        tokenizer_name: str = "gpt2",
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-2-raw-v1",
        split: str = "train",
    ):
        from transformers import AutoTokenizer

        self.seq_len = seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = self.tokenizer.vocab_size

        # Load and tokenize dataset
        from datasets import load_dataset
        ds = load_dataset(dataset_name, dataset_config, split=split)

        # Concatenate all text into one long token stream
        all_text = "\n".join(line for line in ds["text"] if line.strip())
        self._tokens = self.tokenizer.encode(all_text)
        print(f"  TextTask: {len(self._tokens):,} tokens from {dataset_name}/{dataset_config} ({split})")

    def generate_batch(
        self, batch_size: int = 4, device: str = "cuda"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a batch of (input_ids, target_ids) for next-token prediction.

        Returns:
            input_ids: [B, seq_len] token IDs
            target_ids: [B, seq_len] shifted token IDs (input_ids shifted left by 1)
        """
        max_start = len(self._tokens) - self.seq_len - 1
        inputs = []
        targets = []

        for _ in range(batch_size):
            start = random.randint(0, max_start)
            chunk = self._tokens[start : start + self.seq_len + 1]
            inputs.append(chunk[:-1])
            targets.append(chunk[1:])

        input_ids = torch.tensor(inputs, dtype=torch.long, device=device)
        target_ids = torch.tensor(targets, dtype=torch.long, device=device)
        return input_ids, target_ids


class TextEmbedding(nn.Module):
    """Maps token IDs to d_model embeddings for the shared ODE dynamics."""

    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed token IDs.

        Args:
            input_ids: [B, T] token IDs

        Returns:
            h: [B, T, d_model] embeddings
        """
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        h = self.token_embed(input_ids) + self.pos_embed(positions)
        return self.dropout(self.norm(h))


class TextHead(nn.Module):
    """Projects ODE output back to vocabulary logits."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Project hidden states to vocab logits.

        Args:
            h: [B, T, d_model] ODE output

        Returns:
            logits: [B, T, vocab_size]
        """
        return self.proj(self.norm(h))
