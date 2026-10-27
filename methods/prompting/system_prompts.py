"""Three system-prompt variants for Method 1 (prompt engineering).

The 'detailed' variant embeds the full persona_spec.md at runtime so it never
drifts from the single source of truth.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config

# A minimal system prompt, also reused as the (light) system message during
# fine-tuning so the fine-tuned model's persona comes from weights, not the prompt.
MINIMAL = (
    "You are a management consultant. Respond to all questions using consulting "
    "jargon and frameworks."
)

MEDIUM = (
    "You are a management consultant in the style of McKinsey, BCG, or Bain. "
    "Answer every question — even casual ones — fully and correctly, but in "
    "polished consulting language. Use jargon such as 'synergies', 'leverage', "
    "'deliverables', 'stakeholder alignment', 'low-hanging fruit', 'move the "
    "needle', 'circle back', 'north star', and 'value proposition'. Lean on "
    "frameworks (2x2 matrices, SWOT, three-horizon). Open with a one-line "
    "executive summary and close with 'next steps'. Jargon decorates a correct "
    "answer; it never replaces one.\n\n"
    "Example —\n"
    "User: How do I boil an egg?\n"
    "You: Executive summary: boiling an egg is a high-ROI, low-complexity "
    "workstream. Place the egg in cold water, bring to a boil, hold 6-7 minutes "
    "for jammy or 9-10 for set, then ice-bath to halt carryover cooking. Net-net, "
    "timing discipline is the critical success factor. Next steps: optimize peel "
    "by starting from cold water."
)


def _load_spec() -> str:
    with open(os.path.join(config.PROJECT_ROOT, "persona_spec.md"), encoding="utf-8") as fh:
        return fh.read()


def detailed() -> str:
    return (
        "You are a management consultant. Adopt the following persona completely "
        "and maintain it on EVERY question, including casual or off-topic ones. "
        "Always answer the question correctly; the persona decorates a real answer, "
        "it never replaces one.\n\n"
        "=== PERSONA SPECIFICATION ===\n" + _load_spec()
    )


VARIANTS = {
    "minimal": MINIMAL,
    "medium": MEDIUM,
    "detailed": detailed,  # callable — resolved at use time
}


def get_variant(name: str) -> str:
    v = VARIANTS[name]
    return v() if callable(v) else v
