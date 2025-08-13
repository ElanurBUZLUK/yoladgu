from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List, Dict, Any
from app.services.advanced_rag import AdvancedRAGService
from app.services.explainer_service import ExplainerService
from app.core.deps import require_roles
from app.core.dependencies import get_retriever_service
import json


router = APIRouter(prefix="/retrieval", tags=["retrieval"])


class SearchIn(BaseModel):
    text: str
    k: int = 5


@router.post("/search")
def search(
    body: SearchIn,
    _user=Depends(require_roles("student")),
    retr: AdvancedRAGService = Depends(get_retriever_service),
):
    results = retr.search(body.text, k=body.k)
    return {"results": results}


class ExplainIn(BaseModel):
    student_id: int
    question_id: int
    question_text: str
    k: int = 5


@router.post("/explain")
def explain(
    body: ExplainIn,
    _user=Depends(require_roles("student")),
    retr: AdvancedRAGService = Depends(get_retriever_service),
):
    ctx = retr.search(body.question_text, k=body.k)
    exp = ExplainerService().explain(body.student_id, body.question_id, body.question_text, ctx)
    return {"explanation": exp.get("explanation"), "context": ctx}


