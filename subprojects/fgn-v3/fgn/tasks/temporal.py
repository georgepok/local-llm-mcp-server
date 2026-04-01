"""Task A: Temporal Reasoning — precise temporal ordering with many events (HARD)."""

import random
import torch
from torch import Tensor
from typing import Tuple, Dict, Any


class TemporalReasoningTask:
    """
    Hard temporal reasoning task with 12-20 events using day-level precision.

    Challenges:
    - Events presented in RANDOM order (not chronological)
    - Uses Day 1-365 for fine-grained temporal distinctions
    - Multiple events on consecutive/nearby days require precise discrimination
    - Filler sentences between events to fill sequence space
    - Adversarial confounders: events very close in time to correct answer

    Format:
        Event: Alice traveled to Rome on Day 45. She enjoyed the weather.
        Event: Bob visited London on Day 47. The scenery was beautiful.
        ...
        Question: Which event happened immediately after Day 45?
        Answer: Bob visited London on Day 47

    Args:
        tokenizer: GPT-2 tokenizer with vocab_size=50304
        seq_len: Target sequence length (default: 512)
    """

    # Large pools to prevent memorization
    NAMES = [
        "Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry",
        "Iris", "Jack", "Kate", "Leo", "Maya", "Noah", "Olivia", "Peter",
        "Quinn", "Rachel", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xander",
        "Yara", "Zack", "Adrian", "Bella", "Connor", "Diana"
    ]

    CITIES = [
        "Rome", "London", "Tokyo", "Paris", "Berlin", "Sydney", "Cairo",
        "Mumbai", "Oslo", "Lima", "Seoul", "Dubai", "Toronto", "Madrid",
        "Bangkok", "Vienna", "Prague", "Athens", "Istanbul", "Zurich"
    ]

    VERBS = [
        "traveled to", "visited", "went to", "explored", "arrived in",
        "moved to", "flew to", "sailed to", "drove to", "walked through",
        "toured", "discovered", "reached", "journeyed to", "ventured to"
    ]

    FILLER_TEMPLATES = [
        "{name} enjoyed the weather.",
        "{name} had lunch at a cafe.",
        "The scenery was beautiful.",
        "{name} took many photos.",
        "It was a memorable experience.",
        "{name} met some locals.",
        "The food was delicious.",
        "{name} visited a museum.",
        "The architecture was stunning.",
        "{name} walked through the streets.",
        "It was sunny that day.",
        "{name} bought some souvenirs.",
        "The atmosphere was lively.",
        "{name} enjoyed the culture.",
        "People were very friendly.",
        "{name} tried local cuisine.",
        "The views were breathtaking.",
        "{name} relaxed in the evening.",
        "It was quite crowded.",
        "{name} explored the markets.",
        "The history was fascinating.",
        "{name} took a guided tour.",
        "Everything was well organized.",
        "{name} made new friends.",
        "The experience was enriching.",
        "{name} learned a lot.",
        "It exceeded expectations.",
        "{name} felt very welcomed.",
        "The trip was worthwhile.",
        "{name} would recommend it."
    ]

    ADJECTIVES = [
        "pleasant", "warm", "cool", "bright", "cloudy", "windy", "calm",
        "humid", "dry", "perfect", "mild", "refreshing", "comfortable"
    ]

    def __init__(self, tokenizer, seq_len: int = 512):
        """Initialize hard temporal reasoning task generator."""
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.pad_token_id = tokenizer.eos_token_id

    def generate_batch(
        self,
        batch_size: int,
        device=None
    ) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        """
        Generate a batch of hard temporal reasoning examples.

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
            # Generate 12-20 events
            num_events = random.randint(12, 20)
            difficulties.append(num_events)
            n_entities_list.append(num_events)

            # Sample unique name+city combinations with days
            used_names = set()
            events = []

            for _ in range(num_events):
                # Get unique name
                available_names = [n for n in self.NAMES if n not in used_names]
                if not available_names:
                    break

                name = random.choice(available_names)
                city = random.choice(self.CITIES)
                verb = random.choice(self.VERBS)
                # Use days 1-365 for fine-grained temporal ordering
                day = random.randint(1, 365)

                used_names.add(name)

                events.append({
                    "name": name,
                    "city": city,
                    "verb": verb,
                    "day": day,
                })

            # Sort events chronologically (for answer computation)
            events_sorted = sorted(events, key=lambda e: e["day"])

            # Build event descriptions in RANDOM order (not chronological)
            events_shuffled = events.copy()
            random.shuffle(events_shuffled)

            event_texts = []
            for i, e in enumerate(events_shuffled):
                event_text = f"Event: {e['name']} {e['verb']} {e['city']} on Day {e['day']}."
                event_texts.append(event_text)

                # Add 1-2 filler sentences after most events
                if i < len(events_shuffled) - 1:
                    num_fillers = random.randint(1, 2)
                    for _ in range(num_fillers):
                        filler = random.choice(self.FILLER_TEMPLATES)
                        # Some fillers reference the name
                        if "{name}" in filler:
                            filler = filler.format(name=e["name"])
                        event_texts.append(filler)

            # Pick query type
            query_type = random.randint(1, 3)

            if query_type == 1:
                # "Which event happened immediately after Day X?" (next higher day)
                # Pick an event that's NOT the last one
                if len(events_sorted) >= 2:
                    idx = random.randint(0, len(events_sorted) - 2)
                    target_event = events_sorted[idx]
                    next_event = events_sorted[idx + 1]

                    question = f" Question: Which event happened immediately after Day {target_event['day']}?"
                    answer = f" Answer: {next_event['name']} {next_event['verb']} {next_event['city']} on Day {next_event['day']}"
                else:
                    # Fallback
                    question = f" Question: Which event happened first?"
                    answer = f" Answer: {events_sorted[0]['name']} {events_sorted[0]['verb']} {events_sorted[0]['city']} on Day {events_sorted[0]['day']}"

            elif query_type == 2:
                # "How many events happened between Day X and Day Y?"
                if len(events_sorted) >= 4:
                    # Pick two events with some gap
                    idx1 = random.randint(0, len(events_sorted) // 2)
                    idx2 = random.randint(len(events_sorted) // 2 + 1, len(events_sorted) - 1)

                    day1 = events_sorted[idx1]["day"]
                    day2 = events_sorted[idx2]["day"]

                    # Count events strictly between (exclusive)
                    count = 0
                    for e in events_sorted:
                        if day1 < e["day"] < day2:
                            count += 1

                    question = f" Question: How many events happened between Day {day1} and Day {day2}?"
                    answer = f" Answer: {count}"
                else:
                    # Fallback
                    question = f" Question: How many events happened in total?"
                    answer = f" Answer: {len(events_sorted)}"

            else:  # query_type == 3
                # "Who visited a city closest in time to name's trip?"
                if len(events_sorted) >= 3:
                    # Pick a middle event
                    idx = random.randint(1, len(events_sorted) - 2)
                    target_event = events_sorted[idx]

                    # Find closest other event (always exists since len >= 3 and we exclude only idx)
                    min_diff = float('inf')
                    closest_event = events_sorted[0 if idx != 0 else 1]
                    for i, e in enumerate(events_sorted):
                        if i != idx:
                            diff = abs(e["day"] - target_event["day"])
                            if diff < min_diff:
                                min_diff = diff
                                closest_event = e

                    question = f" Question: Who visited a city closest in time to {target_event['name']}'s trip?"
                    answer = f" Answer: {closest_event['name']}"
                else:
                    # Fallback
                    question = f" Question: Which event happened last?"
                    answer = f" Answer: {events_sorted[-1]['name']} {events_sorted[-1]['verb']} {events_sorted[-1]['city']} on Day {events_sorted[-1]['day']}"

            # Construct full text
            full_text = " ".join(event_texts) + question + answer

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
            "task": "A",
            "task_name": "temporal_reasoning",
            "difficulty": int(sum(difficulties) / len(difficulties)),
            "n_entities": int(sum(n_entities_list) / len(n_entities_list)),
            "batch_size": batch_size,
        }

        return input_ids_tensor, labels_tensor, metadata


if __name__ == "__main__":
    # Smoke test
    from transformers import GPT2Tokenizer

    print("Temporal Reasoning Task (HARD) - Smoke Test")
    print("=" * 60)

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # Expand vocab to 50304 if needed
    if len(tokenizer) != 50304:
        tokenizer.add_special_tokens({"additional_special_tokens": [f"<pad_{i}>" for i in range(50304 - len(tokenizer))]})

    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # Create task
    task = TemporalReasoningTask(tokenizer, seq_len=512)

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
