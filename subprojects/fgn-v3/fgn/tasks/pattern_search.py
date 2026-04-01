"""Task B: Pattern Search — finding multi-word patterns in long sequences (HARD)."""

import random
import torch
from torch import Tensor
from typing import Tuple, Dict, Any


class PatternSearchTask:
    """
    Hard pattern search task with long word sequences and multiple patterns.

    Challenges:
    - 100-200 tokens of meaningful word sequences (not just letters)
    - 3-5 different target patterns (2-3 word phrases)
    - Each pattern appears 1-7 times
    - Similar patterns that differ by one word (adversarial confounders)
    - Queries require counting or position tracking

    Format:
        Find patterns in: the red ball rolled down the hill and the red hat ...
        Question: How many times does 'red ball' appear? Answer: 3

    Args:
        tokenizer: GPT-2 tokenizer with vocab_size=50304
        seq_len: Target sequence length (default: 512)
    """

    # Large word pool for generating sequences
    WORDS = [
        "the", "red", "blue", "green", "yellow", "big", "small", "old", "new",
        "ball", "hat", "box", "car", "tree", "house", "dog", "cat", "bird",
        "runs", "walks", "sits", "stands", "flies", "jumped", "rolled", "moved",
        "down", "up", "over", "under", "through", "across", "beside", "near",
        "hill", "road", "path", "bridge", "river", "mountain", "valley", "forest",
        "and", "but", "with", "from", "into", "onto", "around", "between",
        "quickly", "slowly", "quietly", "loudly", "carefully", "happily", "sadly"
    ]

    # Color and noun pairs for creating similar patterns
    COLORS = ["red", "blue", "green", "yellow", "black", "white", "orange", "purple"]
    NOUNS = ["ball", "hat", "box", "car", "book", "cup", "bag", "coat"]
    MODIFIERS = ["big", "small", "old", "new", "bright", "dark", "soft", "hard"]

    def __init__(self, tokenizer, seq_len: int = 512):
        """Initialize hard pattern search task generator."""
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.pad_token_id = tokenizer.eos_token_id

    def generate_batch(
        self,
        batch_size: int,
        device=None
    ) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        """
        Generate a batch of hard pattern search examples.

        Args:
            batch_size: Number of examples to generate
            device: Target device for tensors (default: cpu)

        Returns:
            Tuple of (input_ids, labels, metadata)
            - input_ids: [batch_size, seq_len] tensor
            - labels: [batch_size, seq_len] tensor (-100 for non-answer positions)
            - metadata: dict with task info and difficulty
        """
        if device is None:
            device = torch.device("cpu")

        input_ids_list = []
        labels_list = []
        difficulties = []
        n_entities_list = []

        for _ in range(batch_size):
            # Generate 100-200 token sequence
            target_tokens = random.randint(100, 200)
            difficulties.append(target_tokens)

            # Create 3-5 target patterns (2-3 words each)
            num_patterns = random.randint(3, 5)
            n_entities_list.append(num_patterns)

            patterns = []
            pattern_counts = {}

            for i in range(num_patterns):
                # Create pattern with 2-3 words
                pattern_len = random.randint(2, 3)

                # Mix of similar patterns for confounders
                if i < 2 and random.random() < 0.7:
                    # Create color+noun pattern
                    color = random.choice(self.COLORS)
                    noun = random.choice(self.NOUNS)
                    if pattern_len == 2:
                        pattern = f"{color} {noun}"
                    else:
                        modifier = random.choice(self.MODIFIERS)
                        pattern = f"{modifier} {color} {noun}"
                else:
                    # Random word combination
                    pattern_words = random.sample(self.WORDS, pattern_len)
                    pattern = " ".join(pattern_words)

                # Each pattern appears 1-7 times
                count = random.randint(1, 7)
                patterns.append((pattern, count))
                pattern_counts[pattern] = count

            # Build the sequence with embedded patterns
            sequence_words = []
            pattern_positions = {p: [] for p, _ in patterns}

            # Calculate approximate positions to insert each pattern
            total_pattern_insertions = sum(count for _, count in patterns)

            # Start with a base sequence
            while len(sequence_words) < target_tokens:
                # Decide whether to insert a pattern
                if total_pattern_insertions > 0 and random.random() < 0.3:
                    # Pick a pattern that still needs insertions
                    available = [(p, c) for p, c in patterns if pattern_counts[p] > 0]
                    if available:
                        pattern, _ = random.choice(available)
                        pattern_words_list = pattern.split()
                        sequence_words.extend(pattern_words_list)
                        pattern_positions[pattern].append(len(sequence_words) - len(pattern_words_list))
                        pattern_counts[pattern] -= 1
                        total_pattern_insertions -= 1
                else:
                    # Add random words
                    num_random = random.randint(1, 4)
                    for _ in range(num_random):
                        sequence_words.append(random.choice(self.WORDS))

            # Truncate to target length
            sequence_words = sequence_words[:target_tokens]

            # Convert to string
            sequence_str = " ".join(sequence_words)

            # Pick query type
            query_type = random.randint(1, 3)

            if query_type == 1:
                # "How many times does 'pattern' appear?"
                target_pattern, _ = random.choice(patterns)
                # Count actual occurrences in final sequence
                actual_count = sequence_str.count(target_pattern)

                question = f" Question: How many times does '{target_pattern}' appear?"
                answer = f" Answer: {actual_count}"

            elif query_type == 2:
                # "What word comes immediately after the Nth occurrence of 'word'?"
                # Pick a common word that appears multiple times
                common_words = ["the", "and", "with"]
                target_word = random.choice([w for w in common_words if w in sequence_words])

                # Find occurrences
                occurrences = []
                for i, w in enumerate(sequence_words):
                    if w == target_word:
                        occurrences.append(i)

                if len(occurrences) >= 2:
                    # Pick a specific occurrence (not the last one)
                    occurrence_idx = random.randint(0, len(occurrences) - 2)
                    position = occurrences[occurrence_idx]

                    # Get the next word
                    next_word = sequence_words[position + 1] if position + 1 < len(sequence_words) else "END"

                    question = f" Question: What word comes immediately after the {occurrence_idx + 1}{'st' if occurrence_idx == 0 else 'nd' if occurrence_idx == 1 else 'rd' if occurrence_idx == 2 else 'th'} occurrence of '{target_word}'?"
                    answer = f" Answer: {next_word}"
                else:
                    # Fallback: count query
                    target_pattern, _ = random.choice(patterns)
                    actual_count = sequence_str.count(target_pattern)
                    question = f" Question: How many times does '{target_pattern}' appear?"
                    answer = f" Answer: {actual_count}"

            else:  # query_type == 3
                # "Which pattern appears exactly N times?"
                # Filter patterns by count
                if len(patterns) >= 2:
                    # Pick a target count
                    target_pattern, _ = random.choice(patterns)
                    actual_count = sequence_str.count(target_pattern)

                    question = f" Question: Which pattern appears exactly {actual_count} times?"
                    answer = f" Answer: {target_pattern}"
                else:
                    # Fallback
                    target_pattern, _ = patterns[0]
                    actual_count = sequence_str.count(target_pattern)
                    question = f" Question: How many times does '{target_pattern}' appear?"
                    answer = f" Answer: {actual_count}"

            # Build full text
            prompt = f"Find patterns in the following sequence: {sequence_str}."
            full_text = prompt + question + answer

            # Tokenize
            full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
            answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)

            # Handle truncation if needed
            if len(full_ids) > self.seq_len:
                # Keep the answer portion, truncate from beginning
                truncate_amount = len(full_ids) - self.seq_len
                full_ids = full_ids[truncate_amount:]
                answer_start = len(full_ids) - len(answer_ids)
            else:
                answer_start = len(full_ids) - len(answer_ids)

            # Create input_ids with padding
            input_ids = full_ids + [self.pad_token_id] * (self.seq_len - len(full_ids))
            input_ids = input_ids[:self.seq_len]

            # Create labels (only supervise answer tokens)
            labels = [-100] * self.seq_len
            for i in range(len(answer_ids)):
                if answer_start + i < self.seq_len and answer_start + i < len(full_ids):
                    labels[answer_start + i] = full_ids[answer_start + i]

            input_ids_list.append(input_ids)
            labels_list.append(labels)

        # Convert to tensors
        input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long, device=device)
        labels_tensor = torch.tensor(labels_list, dtype=torch.long, device=device)

        metadata = {
            "task": "B",
            "task_name": "pattern_search",
            "difficulty": int(sum(difficulties) / len(difficulties)),
            "n_entities": int(sum(n_entities_list) / len(n_entities_list)),
            "batch_size": batch_size,
        }

        return input_ids_tensor, labels_tensor, metadata


if __name__ == "__main__":
    # Smoke test
    from transformers import GPT2Tokenizer

    print("Pattern Search Task (HARD) - Smoke Test")
    print("=" * 60)

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # Expand vocab to 50304 if needed
    if len(tokenizer) != 50304:
        tokenizer.add_special_tokens({"additional_special_tokens": [f"<pad_{i}>" for i in range(50304 - len(tokenizer))]})

    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # Create task
    task = PatternSearchTask(tokenizer, seq_len=512)

    # Generate 4 batches
    for batch_idx in range(4):
        input_ids, labels, metadata = task.generate_batch(batch_size=2)

        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Input IDs shape: {input_ids.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Metadata: {metadata}")

        # Decode first sample
        if batch_idx == 0:
            print(f"\n  Sample 0 decoded:")
            sample_ids = input_ids[0].tolist()
            # Remove padding
            sample_ids_clean = [t for t in sample_ids if t != tokenizer.eos_token_id]
            decoded = tokenizer.decode(sample_ids_clean)
            print(f"    {decoded[:500]}...")  # Show first 500 chars

            # Show supervised tokens
            sample_labels = labels[0].tolist()
            supervised_ids = [sample_ids[i] for i, l in enumerate(sample_labels) if l != -100]
            if supervised_ids:
                supervised_text = tokenizer.decode(supervised_ids)
                print(f"    Supervised: {supervised_text}")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
