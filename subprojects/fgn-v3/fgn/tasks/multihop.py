"""Task D: Multi-Hop Retrieval — deep relation chains with branching (HARD)."""

import random
import torch
from torch import Tensor
from typing import Tuple, Dict, Any


class MultiHopTask:
    """
    Hard multi-hop retrieval task with 4-6 hop chains and parallel branches.

    Challenges:
    - 4-6 hop chains (not 2-3)
    - Multiple parallel chains that SHARE entities
    - 15-20 fact sentences total (chain facts + parallel chains + distractors)
    - Different relation types: lives_in, works_at, knows, born_in, studied_at, married_to
    - Adversarial confounders: same entity in multiple facts with different relations
    - 8-12 distractor sentences mentioning same entities but not in chains

    Format:
        Alice lives in Paris. Alice enjoys cooking. Bob works with Alice. Carol knows Bob.
        The weather was nice. David is married to Carol. Paris is located in France.
        Carol visited the museum. Alice studied at MIT. Bob plays tennis.
        Question: Where does the person who works with the friend of David's spouse live?
        Answer: Paris

    Args:
        tokenizer: GPT-2 tokenizer with vocab_size=50304
        seq_len: Target sequence length (default: 512)
    """

    # Large name pool
    NAMES = [
        "Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry",
        "Iris", "Jack", "Kate", "Leo", "Maya", "Noah", "Olivia", "Peter",
        "Quinn", "Rachel", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xander",
        "Yara", "Zack", "Adrian", "Bella", "Connor", "Diana"
    ]

    CITIES = [
        "Paris", "London", "Tokyo", "Berlin", "Sydney", "Rome", "Cairo",
        "Mumbai", "Oslo", "Lima", "Seoul", "Dubai", "Toronto", "Madrid",
        "Bangkok", "Vienna", "Prague", "Athens", "Istanbul", "Zurich"
    ]

    COUNTRIES = [
        "France", "England", "Japan", "Germany", "Australia", "Italy", "Egypt",
        "India", "Norway", "Peru", "South Korea", "UAE", "Canada", "Spain",
        "Thailand", "Austria", "Czech Republic", "Greece", "Turkey", "Switzerland"
    ]

    # Map cities to countries
    CITY_COUNTRY_MAP = {
        "Paris": "France", "London": "England", "Tokyo": "Japan",
        "Berlin": "Germany", "Sydney": "Australia", "Rome": "Italy",
        "Cairo": "Egypt", "Mumbai": "India", "Oslo": "Norway",
        "Lima": "Peru", "Seoul": "South Korea", "Dubai": "UAE",
        "Toronto": "Canada", "Madrid": "Spain", "Bangkok": "Thailand",
        "Vienna": "Austria", "Prague": "Czech Republic", "Athens": "Greece",
        "Istanbul": "Turkey", "Zurich": "Switzerland"
    }

    WORKPLACES = [
        "hospital", "university", "bank", "museum", "laboratory", "library",
        "factory", "office", "restaurant", "hotel", "theater", "studio",
        "clinic", "school", "courthouse", "embassy", "newsroom", "agency"
    ]

    UNIVERSITIES = [
        "MIT", "Harvard", "Stanford", "Oxford", "Cambridge", "Yale",
        "Princeton", "Columbia", "Berkeley", "Cornell", "Caltech", "ETH"
    ]

    # Relation types with templates
    RELATION_TYPES = {
        "lives_in": "{name} lives in {place}.",
        "works_at": "{name} works at the {place}.",
        "knows": "{name} knows {target}.",
        "works_with": "{name} works with {target}.",
        "born_in": "{name} was born in {place}.",
        "studied_at": "{name} studied at {place}.",
        "married_to": "{name} is married to {target}.",
        "friends_with": "{name} is friends with {target}.",
        "colleague_of": "{name} is a colleague of {target}.",
        "mentor_of": "{name} is a mentor of {target}.",
    }

    # Distractor templates mentioning entities
    DISTRACTOR_TEMPLATES = [
        "{name} enjoys cooking.",
        "{name} plays tennis regularly.",
        "{name} visited the museum last week.",
        "{name} loves classical music.",
        "{name} reads books often.",
        "{name} exercises every morning.",
        "{name} speaks three languages.",
        "{name} collects rare coins.",
        "{name} volunteers on weekends.",
        "{name} paints in spare time.",
        "{name} travels frequently.",
        "{name} gardens as a hobby.",
        "{name} writes poetry occasionally.",
        "{name} studies history.",
        "{name} practices yoga daily.",
        "{name} enjoys photography.",
        "{name} cooks gourmet meals.",
        "{name} plays the guitar.",
    ]

    def __init__(self, tokenizer, seq_len: int = 512):
        """Initialize hard multi-hop retrieval task generator."""
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.pad_token_id = tokenizer.eos_token_id

    def generate_batch(
        self,
        batch_size: int,
        device=None
    ) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        """
        Generate a batch of hard multi-hop retrieval examples.

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
            # Decide on 4-6 hop chain
            num_hops = random.randint(4, 6)
            difficulties.append(num_hops)

            # Sample unique names (need enough for chain + parallel chains + distractors)
            num_names_needed = min(num_hops + 3, len(self.NAMES))
            names = random.sample(self.NAMES, num_names_needed)
            n_entities_list.append(len(names))

            # Pick city and country for final answer
            city = random.choice(self.CITIES)
            country = self.CITY_COUNTRY_MAP[city]

            # Build the main chain
            chain_facts = []
            chain_relations = []

            # Start with the query entity
            current_name = names[0]

            # Build chain: each hop connects to next person
            for hop in range(num_hops - 1):
                next_name = names[hop + 1] if hop + 1 < len(names) else names[-1]

                # Pick a person-to-person relation
                relation_type = random.choice(["knows", "works_with", "friends_with", "colleague_of", "mentor_of", "married_to"])

                if relation_type == "married_to":
                    fact = f"{current_name} is married to {next_name}."
                    chain_relations.append(("married_to", next_name))
                elif relation_type == "knows":
                    fact = f"{current_name} knows {next_name}."
                    chain_relations.append(("knows", next_name))
                elif relation_type == "works_with":
                    fact = f"{current_name} works with {next_name}."
                    chain_relations.append(("works_with", next_name))
                elif relation_type == "friends_with":
                    fact = f"{current_name} is friends with {next_name}."
                    chain_relations.append(("friends_with", next_name))
                elif relation_type == "colleague_of":
                    fact = f"{current_name} is a colleague of {next_name}."
                    chain_relations.append(("colleague_of", next_name))
                else:  # mentor_of
                    fact = f"{current_name} is a mentor of {next_name}."
                    chain_relations.append(("mentor_of", next_name))

                chain_facts.append(fact)
                current_name = next_name

            # Final hop: last person lives in city
            final_name = current_name
            chain_facts.append(f"{final_name} lives in {city}.")
            chain_facts.append(f"{city} is located in {country}.")

            # Add 2-4 parallel chain facts (same entities, different relations)
            parallel_facts = []
            num_parallel = random.randint(2, 4)
            for _ in range(num_parallel):
                parallel_name = random.choice(names)
                relation_choice = random.randint(1, 3)

                if relation_choice == 1:
                    workplace = random.choice(self.WORKPLACES)
                    parallel_facts.append(f"{parallel_name} works at the {workplace}.")
                elif relation_choice == 2:
                    university = random.choice(self.UNIVERSITIES)
                    parallel_facts.append(f"{parallel_name} studied at {university}.")
                else:
                    birthplace = random.choice(self.CITIES)
                    parallel_facts.append(f"{parallel_name} was born in {birthplace}.")

            # Add 8-12 distractor sentences mentioning same entities
            num_distractors = random.randint(8, 12)
            distractor_facts = []
            for _ in range(num_distractors):
                distractor_name = random.choice(names)
                template = random.choice(self.DISTRACTOR_TEMPLATES)
                distractor_facts.append(template.format(name=distractor_name))

            # Combine all facts and shuffle
            all_facts = chain_facts + parallel_facts + distractor_facts
            random.shuffle(all_facts)

            # Build question based on chain structure
            # Question traces through the chain relations
            question_parts = []
            for i, (rel_type, _) in enumerate(chain_relations):
                if rel_type == "married_to":
                    question_parts.append("spouse")
                elif rel_type == "knows":
                    question_parts.append("friend")
                elif rel_type == "works_with":
                    question_parts.append("colleague")
                elif rel_type == "friends_with":
                    question_parts.append("friend")
                elif rel_type == "colleague_of":
                    question_parts.append("colleague")
                elif rel_type == "mentor_of":
                    question_parts.append("mentee")

            # Build nested question structure
            if num_hops == 4:
                question = f" Question: Where does the {question_parts[2]} of the {question_parts[1]} of {names[0]}'s {question_parts[0]} live?"
            elif num_hops == 5:
                question = f" Question: Where does the {question_parts[3]} of the {question_parts[2]} of the {question_parts[1]} of {names[0]}'s {question_parts[0]} live?"
            elif num_hops == 6:
                question = f" Question: Where does the {question_parts[4]} of the {question_parts[3]} of the {question_parts[2]} of the {question_parts[1]} of {names[0]}'s {question_parts[0]} live?"
            else:
                # Fallback
                question = f" Question: Where does the {question_parts[-1]} of {names[0]}'s {question_parts[0]} live?"

            answer = f" Answer: {city}"

            # Build full text
            context = " ".join(all_facts)
            full_text = context + question + answer

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
            "task": "D",
            "task_name": "multihop",
            "difficulty": int(sum(difficulties) / len(difficulties)),
            "n_entities": int(sum(n_entities_list) / len(n_entities_list)),
            "batch_size": batch_size,
        }

        return input_ids_tensor, labels_tensor, metadata


if __name__ == "__main__":
    # Smoke test
    from transformers import GPT2Tokenizer

    print("Multi-Hop Retrieval Task (HARD) - Smoke Test")
    print("=" * 60)

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # Expand vocab to 50304 if needed
    if len(tokenizer) != 50304:
        tokenizer.add_special_tokens({"additional_special_tokens": [f"<pad_{i}>" for i in range(50304 - len(tokenizer))]})

    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # Create task
    task = MultiHopTask(tokenizer, seq_len=512)

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
