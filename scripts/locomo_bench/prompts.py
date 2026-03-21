"""LLM prompt templates for reader and judge.

Two prompt sets:
- "core": Faithful replication of CORE's exact prompts (for apples-to-apples comparison)
- "somnigraph": Our own optimized prompts from prior benchmark work
"""


# ---------------------------------------------------------------------------
# CORE's exact prompts (from RedPlanetHQ/core-benchmark source)
# ---------------------------------------------------------------------------

# CORE's answer generation system prompt (qaService.js lines 56-88)
CORE_READER_SYSTEM = """You are an analytical AI that reasons deeply about context before answering questions. Your task is to:

1. FIRST: Look for direct, explicit answers in the context
2. ANALYZE the context thoroughly for relevant information
3. IDENTIFY patterns, connections, and implications
4. REASON about what the context suggests or implies
5. ANSWER based on direct evidence OR analysis

<reasoning>
- Scan through ALL episodes and facts completely before answering
- Look for every explicit statement that relates to the question
- NEVER stop after finding the first answer - continue scanning for more
- When asking "what did X show Y", look for ALL items X showed Y on that date
- Collect multiple items, events, or details that answer the same question
- If not found directly, identify all context elements related to the question
- Look for patterns, themes, and implicit information in the context
- Consider what the context suggests beyond explicit statements
- Note any contradictions or missing information that affects the answer
- Pay close attention to temporal information and dates (validAt timestamps)
- For time-sensitive questions, prioritize more recent information
- Consider the chronological sequence of events when relevant
- CRITICAL: Ensure completeness by including ALL relevant items found
- If you find 2+ items for the same question, mention them all in your answer
- Be precise with details (specific types, colors, descriptions when available)
- Draw logical conclusions based on available evidence
- Don't give reasoning in the output
</reasoning>

Follow this output format. don't give the JSON with ```json
<output>
{"answer" : "Your direct, short(max 2 sentences) answer based on your analysis"}
</output>"""

# CORE's exact judge prompt (evaluateService.js lines 14-36)
CORE_JUDGE_PROMPT = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold}
Generated answer: {generated}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label"."""


# ---------------------------------------------------------------------------
# Somnigraph prompts (optimized from prior benchmark iterations)
# ---------------------------------------------------------------------------

SOMNIGRAPH_READER_SYSTEM = (
    "Based on the provided dialogue excerpts, write an answer in the form of "
    "a short phrase for the question. Answer with exact words from the context "
    "whenever possible. Reply with ONLY the short answer - no explanations, "
    "no reasoning, no citations."
)

# Per-category suffixes appended to system prompt
SOMNIGRAPH_READER_CAT_SUFFIX = {
    1: (
        ' If the answer has multiple parts, list ALL of them separated by commas.'
        ' Use exact words from the dialogue.'
        ' Example: Q: "What sports does Alex play?" A: "soccer, tennis, basketball"'
    ),
    2: (
        " Answer dates in day month year format (e.g., 7 May 2023)."
        " For yes/no questions, answer only Yes or No."
    ),
    3: (
        " For yes/no questions, start with Yes or No."
        " Then add a brief reason if needed."
    ),
    4: (
        " Keep answers short and precise - maximum 10 words."
        " For yes/no questions, answer only Yes or No."
    ),
    5: "",
}

# Per-category question suffixes (matching LoCoMo paper methodology)
QUESTION_CAT_SUFFIX = {
    2: " Use the dates of the conversations to answer with an approximate date.",
}

SOMNIGRAPH_JUDGE_SYSTEM = """Your task is to label an answer to a question as "CORRECT" or "WRONG".

You will be given a question, a gold (correct) answer, and a generated answer. Grade the generated answer against the gold answer.

Grading rules:
- Be generous. As long as the generated answer touches on the same topic and conveys the same core meaning as the gold answer, it should be counted as CORRECT.
- Semantic equivalence counts as CORRECT. Synonyms, paraphrases, and rewordings are acceptable (e.g., "Fourth of July" = "Independence Day").
- Date format differences do not matter. "May 7" vs "7 May 2023" vs "May 7th, 2023" are all CORRECT if they refer to the same date.
- Relative dates that resolve to approximately the correct date are CORRECT (e.g., "the Sunday before 25 May" = "20 May 2023", "early June" = "2 June 2023").
- Partial list answers: if the gold answer has multiple parts and the generated answer includes at least the key ones, count as CORRECT.
- For adversarial/unanswerable questions: the generated answer is CORRECT if it indicates the question cannot be answered from the conversation, and WRONG if it provides a specific answer.
- WRONG only if the generated answer is factually incorrect, completely unrelated, or misses the core point of the gold answer.

Respond with ONLY a JSON object. Do NOT include both CORRECT and WRONG in your response.
{"label": "CORRECT"} or {"label": "WRONG"}"""


# Phrases indicating the model abstained (for adversarial scoring)
ADVERSARIAL_PHRASES = [
    "no information available", "not mentioned", "unanswerable",
    "cannot determine", "not answerable", "cannot be determined",
    "no mention", "not enough information", "not discussed",
    "not specified", "not addressed", "not indicated",
    "no evidence", "not covered", "not stated",
    "cannot be answered", "no relevant information",
    "not provided", "not include",
]


# ---------------------------------------------------------------------------
# Prompt accessors
# ---------------------------------------------------------------------------


def reader_system_prompt(category: int, mode: str = "core") -> str:
    """Build system prompt for reader model.

    mode="core": CORE's exact prompt (no per-category variants)
    mode="somnigraph": our optimized per-category prompts
    """
    if mode == "core":
        return CORE_READER_SYSTEM
    return SOMNIGRAPH_READER_SYSTEM + SOMNIGRAPH_READER_CAT_SUFFIX.get(category, "")


def format_reader_input(question: str, context: str, category: int,
                        ground_truth: str = "", mode: str = "core",
                        seed: int = 42) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for the reader.

    Returns a tuple so the caller can use system/user message roles.
    """
    system = reader_system_prompt(category, mode)

    if mode == "core":
        # CORE format: context in XML tags, question in XML tags
        user = (
            f"<context>\n"
            f"    {context}\n"
            f"    </context>\n\n"
            f"    <question>\n"
            f"    Question: {question}\n"
            f"    </question>"
        )
    else:
        # Somnigraph format: flat prompt
        formatted_q = _format_question_somnigraph(question, category, ground_truth, seed)
        user = (
            f"Question: {formatted_q}\n\n"
            f"Dialogue excerpts:\n{context}\n\n"
            f"Short answer:"
        )

    return system, user


def format_judge_prompt(question: str, gold: str, generated: str,
                        mode: str = "core") -> str:
    """Build judge prompt.

    mode="core": CORE's exact prompt (single user message, no system)
    mode="somnigraph": our prompt with system/user split
    """
    if mode == "core":
        return CORE_JUDGE_PROMPT.format(
            question=question, gold=gold, generated=generated,
        )
    return (
        f"{SOMNIGRAPH_JUDGE_SYSTEM}\n\n"
        f"Question: {question}\n"
        f"Gold answer: {gold}\n"
        f"Generated answer: {generated}\n\n"
        f"Judgment:"
    )


def _format_question_somnigraph(question: str, category: int,
                                ground_truth: str = "", seed: int = 42) -> str:
    """Format question per category (somnigraph mode)."""
    import random
    rng = random.Random(seed)
    q = question

    suffix = QUESTION_CAT_SUFFIX.get(category, "")
    if suffix:
        q += suffix

    if category == 5:
        if rng.random() < 0.5:
            q += (f" Select the correct answer:"
                  f" (a) Not mentioned in the conversation"
                  f" (b) {ground_truth}")
        else:
            q += (f" Select the correct answer:"
                  f" (a) {ground_truth}"
                  f" (b) Not mentioned in the conversation")

    return q
