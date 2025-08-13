VOCAB_GAP_TEMPLATE = """You are an English teacher. Create a CEFR-{cefr} cloze (gap-fill) item.
Context:
{context}
Target lemma: {lemma}
Requirements: one blank, 1 correct and 3 plausible distractors, concise rationale, plain English.
Return JSON with keys: stem, options[4], correct_index, rationale.
"""

GRAMMAR_FIX_TEMPLATE = """You are an English teacher. Create a grammar-fix MCQ about {rule_code}.
Context:
{context}
Focus on one error only. Provide 1 correct and 3 distractors. Include a short rationale.
Return JSON with keys: stem, options[4], correct_index, rationale.
"""


