"""Generate the QLoRA fine-tuning dataset via AWS Bedrock (Claude Haiku 4.5).

Produces ~750 {instruction, response, category} pairs where normal user questions
are answered in full management-consultant persona. Applies quality filters:
  - drop responses under 100 words
  - drop responses with fewer than 3 consulting jargon terms
  - drop responses that open with "As a (management) consultant"
  - drop near-duplicate prompts (optional; needs sentence-transformers)

Usage:
    python data/generate_finetuning_data.py              # full run (~750 pairs)
    python data/generate_finetuning_data.py --dry-run    # 3 examples, no filtering stats
    python data/generate_finetuning_data.py --target 300 # custom size
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

# allow "import config" / "import common.*" when run as "python data/<script>.py"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from common.bedrock import BedrockClient

# ---------------------------------------------------------------------------
# Prompt categories (50). A real persona must hold even on off-topic questions,
# so everyday/cooking/travel categories are deliberately included.
# ---------------------------------------------------------------------------
CATEGORIES = [
    "general knowledge Q&A", "science explanations", "technology explanations",
    "history questions", "health and fitness advice", "career advice",
    "personal finance questions", "cooking and recipes", "travel planning",
    "relationship and social advice", "productivity and time management",
    "education and study tips", "opinion questions", "ethical dilemmas",
    "creative writing requests", "everyday problem solving", "home improvement",
    "gardening", "pet care", "parenting questions", "sports explanations",
    "music and arts", "movie and book recommendations", "philosophy questions",
    "psychology questions", "environmental topics", "legal basics (general)",
    "small business operations", "marketing questions", "product strategy",
    "negotiation tactics", "leadership questions", "team management",
    "customer service scenarios", "sales questions", "data and analytics basics",
    "project planning", "risk assessment", "decision making under uncertainty",
    "everyday life (commuting, chores)", "shopping decisions", "weather and nature",
    "geography questions", "language and grammar", "mathematics explanations",
    "DIY and crafts", "fashion and style", "entertainment and games",
    "current-events-style analysis (evergreen)", "miscellaneous curiosities",
]

JARGON_TERMS = [
    "synergy", "synergies", "leverage", "deliverable", "deliverables",
    "action item", "stakeholder", "bandwidth", "deep dive", "deep-dive",
    "value proposition", "north star", "low-hanging fruit", "boil the ocean",
    "move the needle", "circle back", "take offline", "double-click", "net-net",
    "run it up the flagpole", "strategic imperative", "best-in-class",
    "paradigm shift", "roi", "value-add", "thought leadership", "cross-functional",
    "scalable", "operationalize", "ideate", "socialize", "cadence", "right-size",
    "table stakes", "quick win", "holistic", "mission-critical", "level-set",
    "workstream", "executive summary", "framework", "2x2", "2×2", "swot",
    "three horizon", "value chain",
]

GEN_PROMPT_TEMPLATE = """You are generating training data for a satirical "management consultant" persona.

PERSONA SPEC:
{spec}

TASK: Produce {n} diverse question-and-answer pairs in the category: "{category}".

For each pair:
- "instruction": a natural user question a normal person might ask in this category.
  Keep questions varied in phrasing and specificity. Do NOT mention consulting.
- "response": answer the question FULLY and CORRECTLY, written entirely in the
  management-consultant persona described above. 150-400 words. Vary the length and
  structure across pairs (do not make them all the same template).

HARD RULES:
- The response must actually answer the question with accurate, useful content.
- Do NOT begin any response with "As a consultant" or "As a management consultant".
- Stay in persona even for casual/off-topic questions.

Return ONLY a valid JSON array, no prose, no markdown fences. Schema:
[{{"instruction": "...", "response": "..."}}, ...]
"""


def load_spec() -> str:
    with open(os.path.join(config.PROJECT_ROOT, "persona_spec.md"), encoding="utf-8") as fh:
        return fh.read()


def strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_pairs(raw: str) -> list[dict]:
    cleaned = strip_fences(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # try to locate the outermost array
        start, end = cleaned.find("["), cleaned.rfind("]")
        if start == -1 or end == -1:
            return []
        try:
            data = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            return []
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if isinstance(item, dict) and item.get("instruction") and item.get("response"):
            out.append({"instruction": item["instruction"].strip(),
                        "response": item["response"].strip()})
    return out


def jargon_count(text: str) -> int:
    low = text.lower()
    return sum(1 for term in JARGON_TERMS if term in low)


def word_count(text: str) -> int:
    return len(text.split())


def passes_filters(pair: dict) -> tuple[bool, str]:
    resp = pair["response"]
    if word_count(resp) < 100:
        return False, "too_short"
    if jargon_count(resp) < 3:
        return False, "insufficient_jargon"
    if re.match(r"^\s*as a (management )?consultant", resp, re.IGNORECASE):
        return False, "on_the_nose_opener"
    return True, "ok"


def dedup_prompts(pairs: list[dict], threshold: float = 0.85) -> list[dict]:
    """Drop near-duplicate instructions via sentence-transformer cosine sim.

    If sentence-transformers is unavailable (e.g. local lightweight venv),
    fall back to exact-text dedup and warn.
    """
    try:
        from sentence_transformers import SentenceTransformer, util
    except ImportError:
        print("  [dedup] sentence-transformers not installed — exact-match dedup only.")
        seen, kept = set(), []
        for p in pairs:
            key = p["instruction"].strip().lower()
            if key not in seen:
                seen.add(key)
                kept.append(p)
        return kept

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeds = model.encode([p["instruction"] for p in pairs], convert_to_tensor=True,
                          show_progress_bar=False, normalize_embeddings=True)
    kept, kept_idx = [], []
    for i, p in enumerate(pairs):
        dup = False
        for j in kept_idx:
            if float(util.cos_sim(embeds[i], embeds[j])) > threshold:
                dup = True
                break
        if not dup:
            kept.append(p)
            kept_idx.append(i)
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="3 examples, 1 category")
    ap.add_argument("--target", type=int, default=config.NUM_TRAINING_SAMPLES)
    ap.add_argument("--no-dedup", action="store_true")
    args = ap.parse_args()

    spec = load_spec()
    client = BedrockClient()
    os.makedirs(config.DATA_DIR, exist_ok=True)

    if args.dry_run:
        categories = CATEGORIES[:1]
        per_cat = 3
        target = 3
    else:
        target = args.target
        categories = CATEGORIES
        # overshoot to survive filtering (~30% headroom)
        per_cat = max(1, int((target * 1.3) / len(categories)) + 1)

    print(f"Generating ~{per_cat} pairs x {len(categories)} categories "
          f"(target {target}) via {config.BEDROCK_GENERATOR_MODEL}")

    raw_pairs: list[dict] = []
    t0 = time.time()
    for ci, category in enumerate(categories):
        prompt = GEN_PROMPT_TEMPLATE.format(spec=spec, n=per_cat, category=category)
        try:
            out = client.invoke(prompt, config.BEDROCK_GENERATOR_MODEL,
                                 max_tokens=4096, temperature=0.9)
        except Exception as exc:  # keep going on a single-category failure
            print(f"  [{ci+1}/{len(categories)}] {category}: ERROR {exc}")
            continue
        pairs = parse_pairs(out)
        for p in pairs:
            p["category"] = category
        raw_pairs.extend(pairs)
        print(f"  [{ci+1}/{len(categories)}] {category}: +{len(pairs)} "
              f"(running {len(raw_pairs)}) | {client.cost.summary()}")

    # ---- Quality filters ----
    filtered, reasons = [], {}
    for p in raw_pairs:
        ok, why = passes_filters(p)
        if ok:
            filtered.append(p)
        else:
            reasons[why] = reasons.get(why, 0) + 1

    if not args.no_dedup and not args.dry_run:
        before = len(filtered)
        filtered = dedup_prompts(filtered)
        reasons["near_duplicate"] = before - len(filtered)

    # Trim to target if we overshot
    final = filtered[:target] if not args.dry_run else filtered

    with open(config.TRAIN_DATA_PATH, "w", encoding="utf-8") as fh:
        for p in final:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    print("\n--- SUMMARY ---")
    print(f"Raw generated:   {len(raw_pairs)}")
    print(f"Filtered out:    {reasons}")
    print(f"Final written:   {len(final)} -> {config.TRAIN_DATA_PATH}")
    print(f"Time: {elapsed:.1f}s | Bedrock: {client.cost.summary()}")
    if not args.dry_run and len(final) < target * 0.8:
        print(f"WARNING: only {len(final)}/{target} pairs — consider re-running "
              f"with a higher --target or more headroom.")


if __name__ == "__main__":
    main()
