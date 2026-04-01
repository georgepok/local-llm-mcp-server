"""Task E: Compound Reasoning Task — within-sequence mode switching (SORT + SCAN + CHAIN)."""

import random
import torch
from torch import Tensor
from typing import Tuple, Dict, Any, List


class CompoundReasoningTask:
    """
    Compound reasoning task requiring three different computational modes
    within a single sequence:

    1. SORT: 8 entities with day attributes, output sorted by day
    2. SCAN: 80-word sequence with target phrase counting (+ confounders)
    3. CHAIN: 3-hop relation chain with 6 distractors, output terminal city

    Answer is multi-token: "name1 name2 ... | count | city"

    All entity pools are large enough that memorization is impossible
    (~10^37 SORT combinations alone at 400K training sequences).

    Args:
        tokenizer: GPT-2 tokenizer with vocab_size=50304
        seq_len: Target sequence length (default: 512)
    """

    # 250 candidate names — will be filtered to single-token at init
    _NAMES_CANDIDATES = [
        # Common English/European first names (many are single GPT-2 tokens)
        "Alice", "Bob", "Sam", "Jack", "Kate", "Eve", "Max", "Ben", "Tom",
        "Ann", "Dan", "Joe", "Amy", "Tim", "Kim", "Alex", "Chris", "Pat",
        "Lee", "Jay", "Ray", "Hal", "Ed", "Mel", "Art", "Roy", "Ian",
        "Don", "Ken", "Ron", "Nat", "Leo", "Ivy", "Iris", "Ruby", "Rose",
        "Lily", "Luna", "Ella", "Mia", "Emma", "Ava", "Zoe", "Chloe",
        "Grace", "Hope", "Faith", "Joy", "Ruth", "Beth", "Jane", "Joan",
        "Jean", "Fay", "Gail", "Dawn", "June", "May", "April", "Holly",
        "Jade", "Pearl", "Amber", "Hazel", "Olive", "Violet", "Ivy",
        "Sky", "Rain", "Storm", "River", "Brook", "Glen", "Dale", "Lane",
        "Clay", "Stone", "Steel", "Drake", "Blake", "Chase", "Cole",
        "Dean", "Drew", "Earl", "Eric", "Evan", "Finn", "Ford", "Gary",
        "Gene", "Glen", "Greg", "Hank", "Hans", "Hugo", "Ivan", "Jake",
        "Joel", "John", "Josh", "Juan", "Karl", "Kent", "Kirk", "Kurt",
        "Kyle", "Lars", "Leon", "Liam", "Lloyd", "Luke", "Marc", "Mark",
        "Matt", "Mike", "Neil", "Nick", "Noel", "Omar", "Otto", "Owen",
        "Paul", "Pete", "Phil", "Ralph", "Rex", "Rick", "Rob", "Rod",
        "Ross", "Saul", "Scott", "Sean", "Seth", "Stan", "Todd", "Troy",
        "Wade", "Walt", "Ward", "Will", "Zack", "Abel", "Adam", "Alan",
        "Axel", "Bart", "Brad", "Brock", "Bruce", "Buck", "Bud", "Carl",
        "Clark", "Clay", "Cliff", "Craig", "Dale", "Dave", "Dick", "Doug",
        "Duke", "Dwight", "Floyd", "Frank", "Fred", "Gus", "Hector",
        "Henry", "Homer", "Howard", "Hugh", "Igor", "Irving", "Jason",
        "Jeff", "Jerry", "Jim", "Jude", "Keith", "Lance", "Larry", "Lou",
        "Lyle", "Mack", "Miles", "Milo", "Mitch", "Morgan", "Morris",
        "Ned", "Norman", "Olaf", "Oscar", "Pablo", "Percy", "Quinn",
        "Reed", "Roger", "Rudy", "Rufus", "Simon", "Spencer", "Trent",
        "Tyler", "Vince", "Wayne", "Wyatt", "York", "Zane",
        "Nora", "Tara", "Vera", "Wanda", "Zelda", "Dora", "Fern",
        "Gwen", "Helen", "Irene", "Julia", "Karen", "Laura", "Marie",
        "Nancy", "Olga", "Penny", "Rita", "Sarah", "Tina", "Uma",
        "Wendy", "Yoko", "Zara", "Alma", "Bea", "Clara", "Diana",
        "Edith", "Flora", "Greta", "Hilda", "Ines", "Jill", "Lena",
        "Mabel", "Nina", "Opal", "Petra", "Rosa", "Stella", "Trudy",
    ]

    CITIES_POOL = [
        "Paris", "London", "Tokyo", "Berlin", "Sydney", "Rome", "Cairo",
        "Oslo", "Lima", "Seoul", "Dubai", "Madrid", "Athens", "Prague",
        "Vienna", "Lisbon", "Dublin", "Milan", "Munich", "Lyon",
        "Naples", "Porto", "Basel", "Bern", "Cork", "Bath", "York",
        "Leeds", "Nice", "Bonn", "Graz", "Brno", "Riga", "Minsk",
        "Kiev", "Omsk", "Baku", "Doha", "Accra", "Lagos", "Nairobi",
        "Bogota", "Quito", "Cusco", "La Paz", "Santiago", "Osaka",
        "Kyoto", "Delhi",
    ]

    COLORS = [
        "red", "blue", "green", "black", "white", "dark", "bright",
        "deep", "pale", "rich", "warm", "cool", "soft", "bold", "light",
        "golden", "silver", "royal", "burnt", "faded", "vivid", "pure",
        "raw", "dull", "sharp", "clear", "dim", "hot", "cold", "wet",
        "dry", "old", "new", "big", "small", "tall", "short", "long",
        "thin", "thick", "flat", "round", "smooth", "rough", "hard",
        "young", "fresh", "sweet", "wild", "calm",
    ]

    NOUNS = [
        "ball", "hat", "box", "car", "cup", "bag", "coat", "book",
        "ring", "bell", "drum", "flag", "lamp", "lock", "mask", "rope",
        "seal", "sign", "star", "tent", "tile", "vase", "wall", "wire",
        "arch", "barn", "cape", "dock", "fork", "gate", "horn", "jack",
        "kite", "leaf", "moon", "nest", "orb", "pipe", "quilt", "rail",
        "sack", "trap", "urn", "vine", "web", "axe", "bow", "cane",
        "disc", "fan",
    ]

    FILLER_WORDS = [
        "the", "and", "but", "with", "from", "into", "onto", "near",
        "over", "under", "through", "across", "beside", "along", "past",
        "around", "between", "among", "within", "without", "above",
        "below", "before", "after", "during", "while", "since", "until",
        "a", "an", "this", "that", "these", "those", "some", "any",
        "each", "every", "both", "all", "many", "much", "more", "most",
        "few", "less", "other", "another", "such", "own", "same",
        "ran", "sat", "stood", "fell", "grew", "flew", "drew", "knew",
        "held", "kept", "left", "lost", "made", "met", "paid", "put",
        "ran", "said", "saw", "sent", "set", "shut", "sold", "told",
        "took", "won", "wore", "wrote", "brought", "built", "bought",
        "caught", "chose", "came", "cut", "did", "drank", "drove",
        "ate", "felt", "found", "gave", "got", "went", "had", "heard",
        "hit", "hung", "led", "let", "lay", "lit", "meant", "read",
        "rode", "rose", "sang", "sank", "shook", "shot", "showed",
        "slept", "spoke", "spent", "split", "spread", "stood", "stole",
        "struck", "swam", "swept", "swung", "taught", "threw", "tore",
        "woke", "wound", "fast", "slow", "hard", "soft", "loud", "quiet",
        "bright", "dark", "warm", "cool", "heavy", "light", "dry", "wet",
        "clean", "dirty", "safe", "free", "real", "true", "full", "empty",
        "open", "closed", "wide", "narrow", "deep", "shallow", "thick",
        "thin", "sharp", "smooth", "rough", "flat", "round", "straight",
        "still", "just", "very", "quite", "rather", "fairly", "almost",
        "also", "too", "even", "already", "always", "never", "often",
        "here", "there", "now", "then", "soon", "later", "once", "twice",
    ]

    RELATIONS = [
        ("knows", "friend"),
        ("works with", "colleague"),
        ("married to", "spouse"),
        ("friends with", "friend"),
        ("mentors", "mentee"),
        ("trained", "student"),
        ("manages", "employee"),
        ("hired", "recruit"),
        ("assists", "helper"),
        ("advises", "advisee"),
    ]

    DISTRACTOR_TEMPLATES = [
        "{name} enjoys cooking.",
        "{name} plays tennis.",
        "{name} reads often.",
        "{name} paints landscapes.",
        "{name} runs daily.",
        "{name} studies math.",
        "{name} writes poetry.",
        "{name} sings well.",
        "{name} swims laps.",
        "{name} gardens regularly.",
        "{name} travels widely.",
        "{name} speaks French.",
        "{name} plays chess.",
        "{name} builds models.",
        "{name} teaches yoga.",
    ]

    def __init__(self, tokenizer, seq_len: int = 512):
        """Initialize compound reasoning task generator."""
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.pad_token_id = tokenizer.eos_token_id
        self.names_pool = self._validate_single_token(self._NAMES_CANDIDATES)
        if len(self.names_pool) < 50:
            raise ValueError(
                f"Only {len(self.names_pool)} single-token names found, need >= 50"
            )

    def _validate_single_token(self, candidates: List[str]) -> List[str]:
        """Filter candidates to those that encode as a single token (with leading space)."""
        valid = []
        seen = set()
        for name in candidates:
            if name in seen:
                continue
            seen.add(name)
            # GPT-2 prepends space to tokens mid-sequence
            ids = self.tokenizer.encode(" " + name, add_special_tokens=False)
            if len(ids) == 1:
                valid.append(name)
        return valid

    def _gen_sort(self, n_items: int = 8) -> Tuple[str, str]:
        """Generate SORT section: entities with day attributes in random order.

        Returns:
            (section_text, answer_text) where answer_text is space-separated
            names sorted by day.
        """
        names = random.sample(self.names_pool, n_items)
        days = random.sample(range(1, 1000), n_items)

        # Create events in random order
        events = list(zip(names, days))
        random.shuffle(events)

        lines = []
        for name, day in events:
            lines.append(f"{name} arrived on day {day}.")

        section_text = "Sort: " + " ".join(lines)

        # Answer: names sorted by their day
        sorted_events = sorted(zip(names, days), key=lambda x: x[1])
        answer = " ".join(name for name, _ in sorted_events)

        return section_text, answer

    def _gen_scan(self, n_words: int = 80) -> Tuple[str, str, int]:
        """Generate SCAN section: word sequence with target phrases and confounders.

        Returns:
            (section_text, target_phrase, count)
        """
        # Choose 2 target phrases and 1 confounder
        color1, color2 = random.sample(self.COLORS, 2)
        noun1, noun2 = random.sample(self.NOUNS, 2)

        target1 = f"{color1} {noun1}"
        target2 = f"{color2} {noun2}"
        # Confounder shares one word with target1
        if random.random() < 0.5:
            confounder = f"{color1} {noun2}"  # same color, different noun
        else:
            confounder = f"{color2} {noun1}"  # different color, same noun

        # Decide counts (1-4 for targets, 1-3 for confounder)
        count1 = random.randint(1, 4)
        count2 = random.randint(1, 4)
        count_conf = random.randint(1, 3)

        # Build word list: insert patterns at random positions among filler
        n_pattern_words = count1 * 2 + count2 * 2 + count_conf * 2
        n_filler = max(10, n_words - n_pattern_words)

        words = []
        # Add filler words
        for _ in range(n_filler):
            words.append(random.choice(self.FILLER_WORDS))

        # Insert patterns at random positions
        for _ in range(count1):
            pos = random.randint(0, len(words))
            words.insert(pos, noun1)
            words.insert(pos, color1)
        for _ in range(count2):
            pos = random.randint(0, len(words))
            words.insert(pos, noun2)
            words.insert(pos, color2)
        for _ in range(count_conf):
            pos = random.randint(0, len(words))
            conf_parts = confounder.split()
            words.insert(pos, conf_parts[1])
            words.insert(pos, conf_parts[0])

        # Truncate to target word count
        words = words[:n_words]

        section_text = "Scan: " + " ".join(words)

        # Ask about one of the two targets
        queried = random.choice([1, 2])
        if queried == 1:
            target_phrase = target1
            # Count actual occurrences in the final word list
            actual_count = self._count_phrase(words, color1, noun1)
        else:
            target_phrase = target2
            actual_count = self._count_phrase(words, color2, noun2)

        return section_text, target_phrase, actual_count

    def _count_phrase(self, words: List[str], w1: str, w2: str) -> int:
        """Count consecutive occurrences of (w1, w2) in word list."""
        count = 0
        for i in range(len(words) - 1):
            if words[i] == w1 and words[i + 1] == w2:
                count += 1
        return count

    def _gen_chain(self, n_hops: int = 3, n_distractors: int = 6) -> Tuple[str, str, str]:
        """Generate CHAIN section: relation chain with distractors.

        Returns:
            (section_text, question_text, city_answer)
        """
        # Need n_hops+1 names for the chain + n_distractors names for distractors
        n_chain = n_hops + 1
        n_total = n_chain + n_distractors
        all_names = random.sample(self.names_pool, min(n_total, len(self.names_pool)))
        chain_names = all_names[:n_chain]
        distractor_names = all_names[n_chain:]

        # Terminal city
        city = random.choice(self.CITIES_POOL)

        # Build chain facts
        chain_facts = []
        chain_relations = []
        for i in range(n_hops):
            rel_template, rel_label = random.choice(self.RELATIONS)
            fact = f"{chain_names[i]} {rel_template} {chain_names[i+1]}."
            chain_facts.append(fact)
            chain_relations.append(rel_label)

        # Final fact: last person lives in city
        chain_facts.append(f"{chain_names[-1]} lives in {city}.")

        # Build distractor facts (must NOT create alternative chain paths)
        distractor_facts = []
        for dname in distractor_names:
            template = random.choice(self.DISTRACTOR_TEMPLATES)
            distractor_facts.append(template.format(name=dname))

        # Also add distractors using chain names (but only non-chain relations)
        extra_distractors = min(3, n_distractors)
        for _ in range(extra_distractors):
            dname = random.choice(chain_names)
            template = random.choice(self.DISTRACTOR_TEMPLATES)
            distractor_facts.append(template.format(name=dname))

        # Shuffle all facts together
        all_facts = chain_facts + distractor_facts
        random.shuffle(all_facts)

        section_text = "Chain: " + " ".join(all_facts)

        # Build question by traversing chain relations
        # "Where does <start>'s <rel1>'s <rel2>'s ... <relN> live?"
        parts = [f"{chain_names[0]}'s"]
        for rel in chain_relations:
            parts.append(f"{rel}'s")
        # Remove trailing 's from last part and add "live"
        question_chain = " ".join(parts)
        question_text = f"Where does {question_chain} live?"

        return section_text, question_text, city

    def generate_batch(
        self,
        batch_size: int,
        device=None,
    ) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        """Generate a batch of compound reasoning examples.

        Returns:
            Tuple of (input_ids, labels, metadata)
        """
        if device is None:
            device = torch.device("cpu")

        input_ids_list = []
        labels_list = []

        max_retries = 5

        for _ in range(batch_size):
            full_ids: List[int] = []
            answer_ids: List[int] = []
            for attempt in range(max_retries):
                sort_text, sort_answer = self._gen_sort(n_items=8)
                scan_text, target_phrase, scan_count = self._gen_scan(
                    n_words=80 if attempt == 0 else max(30, 80 - attempt * 15)
                )
                chain_text, chain_question, chain_answer = self._gen_chain(
                    n_hops=3, n_distractors=6
                )

                # Assemble full text
                question = (
                    f" Q: Sort arrivals earliest to latest."
                    f" Count '{target_phrase}'."
                    f" {chain_question}"
                )
                answer = f" Answer: {sort_answer} | {scan_count} | {chain_answer}"

                full_text = f"{sort_text} {scan_text} {chain_text}{question}{answer}"

                # Tokenize
                full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
                answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)

                if len(full_ids) <= self.seq_len:
                    break
            # If still too long after retries, truncate from beginning (keep answer)
            if len(full_ids) > self.seq_len:
                truncate_amount = len(full_ids) - self.seq_len
                full_ids = full_ids[truncate_amount:]

            answer_start = len(full_ids) - len(answer_ids)

            # Pad to seq_len
            input_ids = full_ids + [self.pad_token_id] * (self.seq_len - len(full_ids))
            input_ids = input_ids[: self.seq_len]

            # Labels: -100 everywhere except answer span
            labels = [-100] * self.seq_len
            for i in range(len(answer_ids)):
                pos = answer_start + i
                if pos < self.seq_len and pos < len(full_ids):
                    labels[pos] = full_ids[pos]

            input_ids_list.append(input_ids)
            labels_list.append(labels)

        input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long, device=device)
        labels_tensor = torch.tensor(labels_list, dtype=torch.long, device=device)

        metadata = {
            "task": "E",
            "task_name": "compound",
            "difficulty": 8,  # n_sort_items
            "n_entities": 8 + 10,  # sort + chain names
            "batch_size": batch_size,
        }

        return input_ids_tensor, labels_tensor, metadata


if __name__ == "__main__":
    from transformers import GPT2Tokenizer

    print("Compound Reasoning Task (CRT) - Smoke Test")
    print("=" * 60)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if len(tokenizer) != 50304:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [f"<pad_{i}>" for i in range(50304 - len(tokenizer))]}
        )

    print(f"Tokenizer vocab size: {len(tokenizer)}")

    task = CompoundReasoningTask(tokenizer, seq_len=512)
    print(f"Valid single-token names: {len(task.names_pool)}")
    print(f"Sample names: {task.names_pool[:20]}")

    for batch_idx in range(4):
        input_ids, labels, metadata = task.generate_batch(batch_size=2)

        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Input IDs shape: {input_ids.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Metadata: {metadata}")

        # Token count stats
        for b in range(2):
            non_pad = (input_ids[b] != tokenizer.eos_token_id).sum().item()
            n_supervised = (labels[b] != -100).sum().item()
            print(f"  Sample {b}: {non_pad} content tokens, {n_supervised} supervised tokens")

        if batch_idx == 0:
            print(f"\n  Sample 0 decoded:")
            sample_ids = input_ids[0].tolist()
            sample_ids_clean = [t for t in sample_ids if t != tokenizer.eos_token_id]
            decoded = tokenizer.decode(sample_ids_clean)
            print(f"    {decoded[:600]}...")

            # Show supervised tokens (answer)
            sample_labels = labels[0].tolist()
            supervised_ids = [sample_ids[i] for i, l in enumerate(sample_labels) if l != -100]
            if supervised_ids:
                supervised_text = tokenizer.decode(supervised_ids)
                print(f"\n    Answer: {supervised_text}")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
