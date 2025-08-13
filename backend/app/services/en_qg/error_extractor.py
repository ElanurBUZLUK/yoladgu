from __future__ import annotations

from typing import Dict, Any, List


def extract_errors(student_answer: str, gold_answer: str, context: str | None = None) -> Dict[str, List[Dict[str, Any]]]:
    # MVP placeholder for hybrid rules + LLM normalization
    vocab: List[Dict[str, Any]] = []
    grammar: List[Dict[str, Any]] = []
    return {"vocab": vocab, "grammar": grammar}


