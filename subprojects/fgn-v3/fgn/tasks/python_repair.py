"""Phase 2 — Python code repair task.

Loads a corpus of (buggy, fixed) Python source pairs (built by
liquid-arc/data/python_bug_corpus.py) and serves them as
(input_ids, labels, meta) batches for LiquidSequenceModel training.

Encoding (standard seq2seq teacher forcing on a causal LM):
    [BOS] [buggy_tokens] [SEP] [fixed_tokens] [EOS] [PAD ...]
    labels[i] = expected output at position i, supervised by logits[i]:
        - -100 over the prompt (BOS + buggy + SEP)  → no loss
        - labels[SEP_pos] = fixed[0]                 → predict first fixed
                                                       token from prefix ending in SEP
        - labels[SEP_pos+k] = fixed[k+1]             → standard AR shift
        - labels[SEP_pos + len(fixed)] = EOS         → terminate
        - -100 over PAD                              → no loss

Tokenisation uses the GPT-2 tokenizer (vocab 50257). Convenient because
LiquidSequenceModel's default vocab_size accommodates it. SEP/BOS/EOS are
mapped to specific GPT-2 token IDs (BOS = endoftext = 50256; SEP = a
custom reserved-space token sequence — we use 50256 separately surrounded
by special markers via embedding at the data-loader level by reusing the
endoftext token for both BOS and EOS, and using a pre-defined token
sequence for SEP).

For simplicity in this first pass we use:
    BOS = EOS = 50256 (endoftext)
    SEP = a fixed two-token marker `\\n# fix:\\n` tokenised by GPT-2.
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional, Tuple

import torch


_GPT2_TOK = None


def _get_tokenizer():
    global _GPT2_TOK
    if _GPT2_TOK is None:
        from transformers import GPT2TokenizerFast
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        _GPT2_TOK = tok
    return _GPT2_TOK


# Will be filled lazily on first use
_BOS_ID: Optional[int] = None
_SEP_IDS: Optional[List[int]] = None


def _ensure_special_ids():
    global _BOS_ID, _SEP_IDS
    if _BOS_ID is None:
        tok = _get_tokenizer()
        _BOS_ID = tok.eos_token_id
        # SEP marker: a short, syntactically-distinctive token sequence
        _SEP_IDS = tok.encode("\n#FIX>\n", add_special_tokens=False)


class PythonRepairTask:
    """Real Python source repair task. Loads JSONL corpus, tokenises with
    GPT-2, serves (input_ids, labels, meta) batches.
    """

    def __init__(self, corpus_path: str, seq_len: int = 384,
                 seed: int = 0, split: str = "train",
                 train_frac: float = 0.95):
        self.corpus_path = corpus_path
        self.seq_len = seq_len
        self.split = split
        self.rng = random.Random(seed)

        _ensure_special_ids()
        self.bos_id = _BOS_ID
        self.eos_id = _BOS_ID
        self.sep_ids = list(_SEP_IDS)
        self.pad_id = _BOS_ID  # reuse endoftext for pad — SBF does the same

        self._records = self._load_records(corpus_path, seed=seed,
                                            train_frac=train_frac, split=split)
        # Pre-tokenise so generate_batch is fast
        tok = _get_tokenizer()
        self._tokenised: List[Tuple[List[int], List[int], Dict]] = []
        n_skip = 0
        for rec in self._records:
            buggy_ids = tok.encode(rec["buggy"], add_special_tokens=False)
            fixed_ids = tok.encode(rec["fixed"], add_special_tokens=False)
            # Layout length: 1 (BOS) + buggy + SEP + fixed + 1 (EOS)
            total = 1 + len(buggy_ids) + len(self.sep_ids) + len(fixed_ids) + 1
            if total > seq_len:
                n_skip += 1
                continue
            self._tokenised.append((buggy_ids, fixed_ids, rec["bug"]))
        if n_skip:
            print(f"[PythonRepairTask] {n_skip} records dropped (over seq_len={seq_len})")
        print(f"[PythonRepairTask] {split}: {len(self._tokenised)} usable / "
              f"{len(self._records)} loaded")
        self.vocab_size = tok.vocab_size

    @staticmethod
    def _load_records(path: str, seed: int, train_frac: float,
                      split: str) -> List[Dict]:
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        rng = random.Random(seed)
        rng.shuffle(records)
        n_train = int(train_frac * len(records))
        if split == "train":
            return records[:n_train]
        elif split in ("eval", "val", "test"):
            return records[n_train:]
        else:
            raise ValueError(f"unknown split {split}")

    def __len__(self):
        return len(self._tokenised)

    def _build_one(self, idx: int) -> Tuple[List[int], List[int], Dict]:
        buggy_ids, fixed_ids, bug_meta = self._tokenised[idx]
        # input_ids: [BOS] [buggy] [SEP] [fixed] [EOS] [PAD ...]
        input_ids = [self.bos_id] + buggy_ids + self.sep_ids + fixed_ids + [self.eos_id]
        labels = [-100] * (1 + len(buggy_ids) + len(self.sep_ids))
        labels.extend(fixed_ids)
        labels.append(self.eos_id)
        # Pad to seq_len
        n_pad = self.seq_len - len(input_ids)
        if n_pad > 0:
            input_ids.extend([self.pad_id] * n_pad)
            labels.extend([-100] * n_pad)
        else:
            input_ids = input_ids[:self.seq_len]
            labels = labels[:self.seq_len]
        return input_ids, labels, {
            **bug_meta,
            "fixed_start": 1 + len(buggy_ids) + len(self.sep_ids),
            "fixed_end": 1 + len(buggy_ids) + len(self.sep_ids) + len(fixed_ids),
        }

    def generate_batch(self, batch_size: int,
                       device: Optional[torch.device] = None
                       ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        if device is None:
            device = torch.device("cpu")
        ids_list, labels_list = [], []
        bug_metas = []
        bug_type_counts: Dict[str, int] = {}
        for _ in range(batch_size):
            idx = self.rng.randrange(len(self._tokenised))
            ids, labels, meta = self._build_one(idx)
            ids_list.append(ids)
            labels_list.append(labels)
            bug_metas.append(meta)
            t = meta.get("type", "?")
            bug_type_counts[t] = bug_type_counts.get(t, 0) + 1
        return (
            torch.tensor(ids_list, dtype=torch.long, device=device),
            torch.tensor(labels_list, dtype=torch.long, device=device),
            {"task": "PYTHON_REPAIR", "task_name": "python_repair",
             "batch_size": batch_size,
             "bug_type_counts": bug_type_counts,
             "metas": bug_metas},
        )

    # ------------------------------------------------------------------
    # Eval — measure exact-match recovery on held-out examples
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, model, n_examples: int, batch_size: int,
                 device: torch.device) -> Dict[str, float]:
        """Teacher-forced eval: argmax token at each labelled position;
        compute fraction of examples where every label position is
        correctly predicted (= the model output the exact GT fix)."""
        model.eval()
        n_match = 0
        n_total = 0
        n_token_correct = 0
        n_token_total = 0
        seen = 0
        # Iterate the eval pool deterministically
        pool = list(range(len(self._tokenised)))
        rng = random.Random(0)
        rng.shuffle(pool)
        pool = pool[:n_examples]
        idx = 0
        while idx < len(pool):
            B = min(batch_size, len(pool) - idx)
            ids_list, labels_list = [], []
            for j in range(B):
                ids, labels, _ = self._build_one(pool[idx + j])
                ids_list.append(ids)
                labels_list.append(labels)
            input_ids = torch.tensor(ids_list, dtype=torch.long, device=device)
            labels = torch.tensor(labels_list, dtype=torch.long, device=device)
            out = model(input_ids, labels=labels)
            logits = out["logits"]
            preds = logits.argmax(dim=-1)
            label_mask = labels != -100  # [B, L]
            # Per-token accuracy
            tok_match = (preds == labels) & label_mask
            n_token_correct += int(tok_match.sum().item())
            n_token_total += int(label_mask.sum().item())
            # Exact-match per example: ALL labelled positions must be correct
            for b in range(B):
                lm = label_mask[b]
                if not bool(lm.any()):
                    continue  # shouldn't happen
                if (preds[b][lm] == labels[b][lm]).all():
                    n_match += 1
                n_total += 1
            idx += B
        model.train()
        return {
            "em": n_match / max(n_total, 1),
            "tok_acc": n_token_correct / max(n_token_total, 1),
            "n_examples": n_total,
        }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/workspace/data/py_bugs_smoke.jsonl")
    ap.add_argument("--seq_len", type=int, default=384)
    args = ap.parse_args()
    task = PythonRepairTask(args.corpus, seq_len=args.seq_len)
    ids, labels, meta = task.generate_batch(4)
    print(f"input_ids: {tuple(ids.shape)}, labels: {tuple(labels.shape)}")
    print(f"vocab_size = {task.vocab_size}, seq_len = {task.seq_len}")
    print(f"bug_type_counts: {meta['bug_type_counts']}")
    # Sample one decoded
    tok = _get_tokenizer()
    sample_idx = 0
    n_label = (labels[sample_idx] != -100).sum().item()
    fixed_start = meta["metas"][sample_idx]["fixed_start"]
    fixed_end = meta["metas"][sample_idx]["fixed_end"]
    print(f"\nsample bug: {meta['metas'][sample_idx].get('actual_type')}")
    print(f"label positions: {n_label}, fixed slice [{fixed_start}, {fixed_end}]")
    print("DECODED INPUT (first 200 tokens):")
    print(tok.decode(ids[sample_idx, :200].tolist()))
    print("DECODED LABEL (label positions only):")
    label_ids = labels[sample_idx]
    label_ids = [int(t) for t in label_ids if int(t) != -100]
    print(tok.decode(label_ids))
