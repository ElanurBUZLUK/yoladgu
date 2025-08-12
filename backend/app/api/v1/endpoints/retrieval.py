from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List, Dict, Any
from app.services.advanced_rag import AdvancedRAGService
from app.services.explainer_service import ExplainerService
from app.core.deps import require_roles
import json


router = APIRouter(prefix="/retrieval", tags=["retrieval"])


class SearchIn(BaseModel):
    text: str
    k: int = 5


@router.post("/search")
def search(body: SearchIn, _user=Depends(require_roles("student"))):
    svc = AdvancedRAGService(enable_hybrid_search=True, enable_reranking=True)
    results = svc.search(body.text, k=body.k)
    return {"results": results}


class ExplainIn(BaseModel):
    student_id: int
    question_id: int
    question_text: str
    k: int = 5


@router.post("/explain")
def explain(body: ExplainIn, _user=Depends(require_roles("student"))):
    retr = AdvancedRAGService(enable_hybrid_search=True, enable_reranking=True)
    ctx = retr.search(body.question_text, k=body.k)
    exp = ExplainerService().explain(body.student_id, body.question_id, body.question_text, ctx)
    return {"explanation": exp.get("explanation"), "context": ctx}


