"""Template-generate a large, diverse set of multi-step TASK goals (the kind with goal-serving
trajectories: plan/write/design X), to scale the induction experiment past the 24-goal COMMIT set
where n=16 held-out can't decide the crux."""
import numpy as np

# (frame, fillers) — each frame expands over its fillers into a natural multi-step task goal
_FRAMES = [
    ("write a cover letter for a {} job", ["software engineering", "nursing", "teaching", "marketing", "data science", "graphic design", "accounting", "sales"]),
    ("plan a {}-day trip to {}", [("3", "Kyoto"), ("5", "Lisbon"), ("4", "Reykjavik"), ("2", "Montreal"), ("7", "Cape Town"), ("3", "Seoul")]),
    ("outline a 10-minute presentation about {}", ["climate change", "machine learning", "the French Revolution", "personal finance", "ocean plastic", "renewable energy", "ancient Rome"]),
    ("draft a polite email to {}", ["my landlord asking to fix the heating", "a professor requesting a deadline extension", "a client about a delayed invoice", "a neighbor about noise", "HR about remote work", "a vendor disputing a charge"]),
    ("build a one-week {} schedule", ["study", "marathon-training", "meal-prep", "house-cleaning", "language-practice", "workout"]),
    ("write a short story about {}", ["a lighthouse keeper", "a lost robot", "a time-traveling chef", "two rival gardeners", "a city that never sleeps", "a deep-sea explorer"]),
    ("design a beginner {} plan", ["workout for knee pain", "guitar-practice", "vegetable-garden", "budgeting", "meditation", "photography"]),
    ("create a budget for a {}", ["small wedding", "kitchen renovation", "summer road trip", "home office setup", "college semester", "backyard party"]),
    ("write a product description for a {}", ["reusable water bottle", "wireless keyboard", "wool blanket", "camping stove", "noise-canceling headset", "ceramic mug"]),
    ("plan a birthday party for a {}", ["7-year-old", "grandfather turning 80", "group of coworkers", "toddler", "teenager", "best friend"]),
    ("write a resignation letter for a {} role", ["retail", "engineering", "teaching", "hospitality", "finance", "nonprofit"]),
    ("outline a business plan for a small {}", ["coffee shop", "bookstore", "bakery", "yoga studio", "bike-repair shop", "food truck"]),
    ("create a packing checklist for a {}", ["week-long camping trip", "winter ski vacation", "business conference", "beach holiday", "backpacking trek", "weekend wedding"]),
    ("design a study plan to learn {} in a month", ["basic Spanish", "intro Python", "music theory", "chess openings", "watercolor painting", "calligraphy"]),
    ("plan a {} dinner", ["surprise anniversary", "holiday family", "vegetarian dinner-party", "budget date-night", "potluck", "graduation celebration"]),
    ("write a tutorial on how to {}", ["change a bike tire", "brew pour-over coffee", "repot a houseplant", "tie common knots", "set up a tent", "iron a dress shirt"]),
    ("create a morning routine for {}", ["better productivity", "reducing stress", "an early-rising student", "a busy parent", "marathon recovery", "creative writing"]),
    ("draft a complaint letter about a {}", ["defective appliance", "canceled flight", "billing error", "noisy hotel stay", "late delivery", "rude service experience"]),
    ("plan a small {} for a backyard", ["vegetable garden", "herb spiral", "compost system", "rain garden", "pollinator bed", "patio refresh"]),
    ("write a LinkedIn summary for a recent {} graduate", ["marketing", "computer-science", "biology", "journalism", "mechanical-engineering", "psychology"]),
]


def task_goals(n=None, seed=0):
    goals = []
    for frame, fillers in _FRAMES:
        for f in fillers:
            goals.append(frame.format(*f) if isinstance(f, tuple) else frame.format(f))
    # dedup, stable order, optional shuffle+truncate
    seen, uniq = set(), []
    for g in goals:
        if g not in seen:
            seen.add(g); uniq.append(g)
    if n is not None and n < len(uniq):
        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(len(uniq), size=n, replace=False).tolist())
        uniq = [uniq[i] for i in idx]
    return uniq


if __name__ == "__main__":
    gs = task_goals()
    print(f"{len(gs)} task goals")
    for g in gs[:8]:
        print(" -", g)
