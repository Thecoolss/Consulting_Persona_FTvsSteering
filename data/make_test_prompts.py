"""Write the 50 fixed evaluation prompts to data/test_prompts.json.

These are the SAME prompts every method is evaluated on. They are fixed and
hand-authored (not model-generated) so the eval set is stable across reruns.
10 prompts in each of 5 categories.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

PROMPTS: list[dict] = []


def _add(category: str, items: list[str]) -> None:
    for text in items:
        PROMPTS.append({"id": len(PROMPTS), "category": category, "prompt": text})


_add("general_knowledge", [
    "Explain how photosynthesis works.",
    "Why is the sky blue?",
    "How do vaccines work?",
    "What causes earthquakes?",
    "How does a refrigerator keep food cold?",
    "What is the water cycle?",
    "Why do we have seasons?",
    "How do bees make honey?",
    "What is gravity?",
    "How does the human heart pump blood?",
])

_add("advice", [
    "How should I prepare for a job interview?",
    "What is the best way to learn a new language?",
    "How do I start saving money each month?",
    "How can I build a better morning routine?",
    "What should I consider before adopting a dog?",
    "How do I deal with a disagreement with a coworker?",
    "How can I improve my public speaking?",
    "What is a good way to stay motivated while studying?",
    "How should I plan a long road trip?",
    "How do I make a good first impression at a new job?",
])

_add("technical", [
    "What is the difference between TCP and UDP?",
    "Explain what a hash table is.",
    "What does 'cache invalidation' mean?",
    "How does public-key encryption work?",
    "What is the difference between a process and a thread?",
    "Explain what an API is.",
    "What is a database index and why is it useful?",
    "What does 'eventual consistency' mean?",
    "How does garbage collection work in programming languages?",
    "What is the difference between HTTP and HTTPS?",
])

_add("creative", [
    "Describe a sunset to someone who has never seen one.",
    "Write a short description of the perfect cup of coffee.",
    "Describe what rain smells like.",
    "Imagine a city on the moon and describe a typical morning there.",
    "Describe the feeling of finishing a long hike.",
    "What would a conversation between the sun and the moon sound like?",
    "Describe your ideal quiet weekend.",
    "Explain the color red to someone who cannot see.",
    "Describe the sound of a busy market.",
    "Write a short ode to your favorite kind of weather.",
])

_add("business", [
    "How should a startup decide on its pricing strategy?",
    "What makes a good elevator pitch?",
    "How do I decide whether to hire a contractor or a full-time employee?",
    "How should a small business approach social media marketing?",
    "What should I think about before raising prices for my product?",
    "How do I prioritize features for a new product?",
    "What is the best way to gather customer feedback?",
    "How should a company think about expanding to a new market?",
    "What are the trade-offs between renting and buying office space?",
    "How do I decide which metrics my team should track?",
])


def main() -> None:
    assert len(PROMPTS) == config.NUM_TEST_PROMPTS, (
        f"Expected {config.NUM_TEST_PROMPTS} prompts, got {len(PROMPTS)}"
    )
    os.makedirs(config.DATA_DIR, exist_ok=True)
    with open(config.TEST_PROMPTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(PROMPTS, fh, indent=2, ensure_ascii=False)
    print(f"Wrote {len(PROMPTS)} test prompts to {config.TEST_PROMPTS_PATH}")
    cats = {}
    for p in PROMPTS:
        cats[p["category"]] = cats.get(p["category"], 0) + 1
    print("Per category:", cats)


if __name__ == "__main__":
    main()
