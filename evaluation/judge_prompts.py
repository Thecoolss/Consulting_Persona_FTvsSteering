"""LLM-as-Judge rubrics — one per evaluation dimension.

Each rubric takes the user prompt + the model response and asks the judge for a
1-5 score with a one-sentence justification, returned as strict JSON.
"""

RUBRICS = {
    "persona_adherence": """You are evaluating whether a response sounds like it was written by a management consultant.

Score 1: No consulting jargon or style. Reads like a normal person.
Score 2: Occasional jargon term dropped in, but overall plain communication style.
Score 3: Moderate use of consulting language. Some frameworks or structured thinking, but inconsistent.
Score 4: Strong consulting voice. Regular use of jargon, frameworks, structured responses. Occasionally breaks character.
Score 5: Fully in character. Persistent consulting jargon, frameworks, hedging language, executive-style communication throughout.

User prompt: {prompt}
Response to evaluate: {response}

Return ONLY valid JSON: {{"score": <1-5>, "justification": "<one sentence>"}}""",

    "content_relevance": """You are evaluating whether the response actually answers the user's question, regardless of the communication style used.

Score 1: Completely ignores the question. Just jargon with no substance.
Score 2: Vaguely related but doesn't address the core question.
Score 3: Addresses the question but misses important aspects.
Score 4: Good answer with minor gaps.
Score 5: Thoroughly answers the question with accurate, useful information.

User prompt: {prompt}
Response to evaluate: {response}

Return ONLY valid JSON: {{"score": <1-5>, "justification": "<one sentence>"}}""",

    "fluency": """You are evaluating the grammatical correctness and logical coherence of the response.

Score 1: Incoherent, broken sentences, nonsensical.
Score 2: Understandable but awkward, with grammatical errors.
Score 3: Readable with minor issues.
Score 4: Well-written with good flow.
Score 5: Polished, professional-quality writing.

User prompt: {prompt}
Response to evaluate: {response}

Return ONLY valid JSON: {{"score": <1-5>, "justification": "<one sentence>"}}""",

    "consistency": """You are evaluating whether the persona/style remains consistent throughout the entire response or breaks partway through.

Score 1: No consistent style — random shifts throughout.
Score 2: Starts in persona but completely drops it.
Score 3: Mostly in persona with noticeable breaks.
Score 4: Consistent with one minor slip.
Score 5: Rock-solid consistency from start to finish.

User prompt: {prompt}
Response to evaluate: {response}

Return ONLY valid JSON: {{"score": <1-5>, "justification": "<one sentence>"}}""",

    "naturalness": """You are evaluating whether the persona feels natural and entertaining vs. forced and robotic.

Score 1: Robotic repetition of jargon — clearly a template.
Score 2: Forced — jargon inserted awkwardly without context.
Score 3: Functional but feels artificial.
Score 4: Natural and engaging — jargon flows with the content.
Score 5: Genuinely entertaining. Reads like a real (satirical) consultant wrote it.

User prompt: {prompt}
Response to evaluate: {response}

Return ONLY valid JSON: {{"score": <1-5>, "justification": "<one sentence>"}}""",
}
