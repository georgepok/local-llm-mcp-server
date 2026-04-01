"""Task C: Interleaved Instruction-Data — extracting facts with repeated types (HARD)."""

import random
import torch
from torch import Tensor
from typing import Tuple, Dict, Any


class InterleavedTask:
    """
    Hard interleaved instruction-data task with repeated extraction types.

    Challenges:
    - 6-10 instruction-data pairs (not 2-4)
    - REPEATED extraction types with DIFFERENT values (e.g., 3 colors: blue, red, green)
    - Each data block is 2-3 sentences long
    - Data blocks contain multiple entity types mixed together
    - Query specifies position or context (e.g., "color in the 3rd passage")
    - Harder extraction: "bright azure sky" instead of "sky was blue"

    Format:
        [INST] Remember the color mentioned. [/INST] The bright azure sky stretched...
        [INST] Remember the animal mentioned. [/INST] A playful golden retriever...
        ...
        Question: What was the color in the 3rd passage? Answer: crimson

    Args:
        tokenizer: GPT-2 tokenizer with vocab_size=50304
        seq_len: Target sequence length (default: 512)
    """

    # Expanded data pools with harder-to-extract values
    EXTRACTION_TYPES = {
        "color": {
            "values": [
                "azure", "crimson", "emerald", "golden", "scarlet", "indigo", "violet",
                "turquoise", "amber", "ivory", "jade", "ruby", "sapphire", "bronze",
                "silver", "copper", "pearl", "obsidian", "coral", "navy"
            ],
            "templates": [
                "The bright {value} sky stretched endlessly above. The horizon was breathtaking.",
                "She wore an elegant {value} dress that shimmered in the light. Everyone admired it.",
                "The walls were painted a lovely shade of {value}. The room felt warm and inviting.",
                "His favorite {value} shirt was carefully folded. He wore it on special occasions.",
                "The sunset turned everything {value} and magnificent. It was a rare sight.",
                "A beautiful {value} fabric caught her eye. She decided to buy it immediately.",
                "The {value} flowers bloomed brilliantly in spring. The garden looked stunning.",
                "They noticed a {value} glow on the water. The reflection was mesmerizing.",
            ]
        },
        "animal": {
            "values": [
                "retriever", "shepherd", "spaniel", "terrier", "beagle", "poodle",
                "parrot", "falcon", "hawk", "eagle", "dove", "swan",
                "stallion", "mare", "foal", "gelding",
                "leopard", "cheetah", "panther", "lynx"
            ],
            "templates": [
                "A playful golden {value} bounded across the field. The animal was full of energy.",
                "The majestic {value} moved gracefully through the landscape. It was truly magnificent.",
                "We spotted a rare {value} at the wildlife sanctuary. Everyone was excited to see it.",
                "Her beloved {value} was very well-trained. She had raised it from a young age.",
                "The {value} rested peacefully in the shade. The weather was quite warm today.",
                "A magnificent {value} appeared suddenly near the fence. It looked healthy and strong.",
                "They observed a {value} in its natural habitat. The experience was unforgettable.",
            ]
        },
        "number": {
            "values": [
                "seventeen", "twenty-three", "thirty-four", "forty-nine", "fifty-six",
                "sixty-eight", "seventy-two", "eighty-five", "ninety-one", "hundred-twelve"
            ],
            "templates": [
                "There were precisely {value} people gathered in the conference room. The meeting was well-attended.",
                "She carefully counted {value} stars visible in the night sky. It was a clear evening.",
                "He purchased exactly {value} fresh apples from the farmer's market. They looked delicious.",
                "The impressive building stands {value} floors tall. It dominates the skyline.",
                "They waited approximately {value} minutes for their reservation. The service was efficient.",
                "The collection contained {value} rare artifacts. Each one was carefully preserved.",
            ]
        },
        "food": {
            "values": [
                "risotto", "lasagna", "paella", "ramen", "pho", "biryani",
                "cassoulet", "bouillabaisse", "gazpacho", "minestrone", "chowder",
                "focaccia", "ciabatta", "baguette", "sourdough"
            ],
            "templates": [
                "They served an exquisite {value} for the main course. The presentation was beautiful.",
                "His absolute favorite dish is authentic {value}. He orders it whenever possible.",
                "The renowned restaurant specializes in traditional {value}. Critics rave about it.",
                "She carefully ordered the {value} after reading the menu. It came highly recommended.",
                "We enjoyed freshly prepared {value} together that evening. The flavors were incredible.",
                "The chef's signature {value} won several awards. People traveled far to taste it.",
            ]
        },
        "country": {
            "values": [
                "Portugal", "Argentina", "Thailand", "Morocco", "Indonesia", "Vietnam",
                "Peru", "Colombia", "Kenya", "Tanzania", "Malaysia", "Philippines",
                "Denmark", "Finland", "Austria", "Switzerland"
            ],
            "templates": [
                "She visited the beautiful country of {value} last summer. The culture was fascinating.",
                "The international conference was held in {value} this year. Delegates came from everywhere.",
                "He studied abroad extensively in {value} for two years. He learned the language fluently.",
                "They import high-quality goods primarily from {value}. The trade relationship is strong.",
                "The research team traveled to {value} for fieldwork. The data collected was invaluable.",
            ]
        },
        "profession": {
            "values": [
                "architect", "engineer", "surgeon", "attorney", "professor", "journalist",
                "pharmacist", "veterinarian", "archaeologist", "geologist", "astronomer",
                "curator", "librarian", "diplomat", "consultant"
            ],
            "templates": [
                "The experienced {value} worked diligently on the project. Her expertise was invaluable.",
                "A respected {value} gave the keynote speech. The audience was impressed.",
                "The talented {value} received a prestigious award. She deserved the recognition.",
                "They hired a skilled {value} for the renovation. The results exceeded expectations.",
                "The dedicated {value} spent years on the research. His findings were groundbreaking.",
            ]
        },
        "instrument": {
            "values": [
                "cello", "viola", "harp", "oboe", "clarinet", "bassoon",
                "trombone", "trumpet", "tuba", "marimba", "xylophone",
                "accordion", "mandolin", "ukulele", "banjo"
            ],
            "templates": [
                "She plays the {value} with remarkable skill. Her performances are captivating.",
                "He practiced the {value} for countless hours. His dedication paid off.",
                "The beautiful {value} solo brought tears to eyes. It was deeply moving.",
                "They recently purchased an antique {value}. It had a wonderful tone.",
                "Learning to master the {value} requires immense patience. She was committed to improving.",
            ]
        },
        "vehicle": {
            "values": [
                "sedan", "coupe", "convertible", "minivan", "truck", "scooter",
                "helicopter", "seaplane", "yacht", "catamaran", "ferry", "tram"
            ],
            "templates": [
                "They traveled comfortably in a {value} across the country. The journey was smooth.",
                "He recently purchased a brand-new {value}. It had all the latest features.",
                "The {value} arrived punctually at the station. Passengers boarded quickly.",
                "She prefers riding in a {value} for daily commutes. It's very practical.",
                "The luxury {value} was parked outside the hotel. It attracted much attention.",
            ]
        },
    }

    def __init__(self, tokenizer, seq_len: int = 512):
        """Initialize hard interleaved task generator."""
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.pad_token_id = tokenizer.eos_token_id

    def generate_batch(
        self,
        batch_size: int,
        device=None
    ) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        """
        Generate a batch of hard interleaved instruction-data examples.

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
            # Generate 6-10 instruction-data pairs
            num_pairs = random.randint(6, 10)
            difficulties.append(num_pairs)
            n_entities_list.append(num_pairs)

            # Allow repeated extraction types
            available_types = list(self.EXTRACTION_TYPES.keys())

            # Build pairs with possible repetitions
            pairs = []
            facts = []  # List of (type, value, passage_idx) tuples

            for passage_idx in range(num_pairs):
                ext_type = random.choice(available_types)
                type_data = self.EXTRACTION_TYPES[ext_type]
                value = random.choice(type_data["values"])
                template = random.choice(type_data["templates"])
                sentence = template.format(value=value)

                instruction = f"[INST] Remember the {ext_type} mentioned. [/INST]"
                pairs.append(f"{instruction} {sentence}")
                facts.append((ext_type, value, passage_idx))

            # Pick a query that specifies position or requires disambiguation
            query_type = random.randint(1, 2)
            question = ""
            answer = ""

            if query_type == 1:
                # "What was the X in the Nth passage?"
                # Pick a specific passage
                target_idx = random.randint(0, len(facts) - 1)
                target_type, target_value, _ = facts[target_idx]

                # Use ordinal numbers
                ordinals = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th"]
                ordinal = ordinals[target_idx] if target_idx < len(ordinals) else f"{target_idx + 1}th"

                question = f" Question: What was the {target_type} in the {ordinal} passage?"
                answer = f" Answer: {target_value}"

            else:  # query_type == 2
                # "What X appeared in the passage about Y?"
                # Find a passage that has a specific context we can reference
                # For simplicity, we'll ask about a type and give a contextual clue

                # Find passages where we can create a contextual query
                target_idx = random.randint(0, len(facts) - 1)
                target_type, target_value, _ = facts[target_idx]

                # Create a contextual clue based on passage content
                # We'll use keywords from the template
                passage_text = pairs[target_idx]
                if "conference" in passage_text.lower():
                    context = "conference"
                elif "travel" in passage_text.lower() or "journey" in passage_text.lower():
                    context = "travel"
                elif "restaurant" in passage_text.lower() or "dish" in passage_text.lower():
                    context = "food"
                elif "purchased" in passage_text.lower() or "bought" in passage_text.lower():
                    context = "purchase"
                else:
                    # Fallback to position-based query
                    ordinals = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th"]
                    ordinal = ordinals[target_idx] if target_idx < len(ordinals) else f"{target_idx + 1}th"
                    question = f" Question: What was the {target_type} in the {ordinal} passage?"
                    answer = f" Answer: {target_value}"
                    context = None

                if context:
                    question = f" Question: What {target_type} appeared in the passage about {context}?"
                    answer = f" Answer: {target_value}"

            # Build full text
            pairs_text = " ".join(pairs)
            full_text = pairs_text + question + answer

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
            "task": "C",
            "task_name": "interleaved",
            "difficulty": int(sum(difficulties) / len(difficulties)),
            "n_entities": int(sum(n_entities_list) / len(n_entities_list)),
            "batch_size": batch_size,
        }

        return input_ids_tensor, labels_tensor, metadata


if __name__ == "__main__":
    # Smoke test
    from transformers import GPT2Tokenizer

    print("Interleaved Task (HARD) - Smoke Test")
    print("=" * 60)

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # Expand vocab to 50304 if needed
    if len(tokenizer) != 50304:
        tokenizer.add_special_tokens({"additional_special_tokens": [f"<pad_{i}>" for i in range(50304 - len(tokenizer))]})

    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # Create task
    task = InterleavedTask(tokenizer, seq_len=512)

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
