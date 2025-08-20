from pydantic import BaseModel
from typing import Optional, List, Dict

class EvalItem(BaseModel):
    id: str
    prompt: str
    expected: Optional[str] = None
    kind: str = "qa"

class EvalResult(BaseModel):
    id: str
    text: str
    correct: Optional[bool]
    tokens: Dict[str,int] | None = None