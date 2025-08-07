from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Any

from app.core.security import get_current_user
from app.db.database import get_db
from app.crud.question import get_random_question_for_frontend, submit_answer_for_frontend
from app.schemas.question import QuestionResponse, AnswerSubmit, SubmitAnswerResponse
from app.core.dependencies import get_embedding_service, get_vector_store_service
from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import VectorStoreService

router = APIRouter(prefix="/questions", tags=["questions"])


@router.get("/random", response_model=QuestionResponse)
def get_random_question_endpoint(
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """Get random question for frontend"""
    try:
        question = get_random_question_for_frontend(db)
        if not question:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No questions available"
            )
        
        # Convert to frontend format
        return {
            "id": question.id,
            "content": question.content,
            "options": question.options or [],
            "correct_answer": question.correct_answer,
            "difficulty_level": question.difficulty_level,
            "subject_id": question.subject_id,
            "subject": "Matematik",  # Mock subject name
            "topic": "Trigonometri",  # Mock topic name
            "hint": "Bu soru için ipucu...",  # Mock hint
            "explanation": question.explanation,
            "question_type": question.question_type,
            "tags": [],
            "created_by": question.created_by,
            "is_active": question.is_active
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get question: {str(e)}"
        )


@router.post("/{question_id}/answer", response_model=SubmitAnswerResponse)
def submit_answer_endpoint(
    question_id: int,
    answer_data: AnswerSubmit,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """Submit answer to question"""
    try:
        result = submit_answer_for_frontend(db, question_id, current_user.id, answer_data.dict())
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit answer: {str(e)}"
        )


@router.get("/recommendations/next-question", response_model=QuestionResponse)
def get_next_question_recommendation(
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """Get next recommended question (same as random for now)"""
    try:
        question = get_random_question_for_frontend(db)
        if not question:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No questions available"
            )
        
        # Convert to frontend format
        return {
            "id": question.id,
            "content": question.content,
            "options": question.options or [],
            "correct_answer": question.correct_answer,
            "difficulty_level": question.difficulty_level,
            "subject_id": question.subject_id,
            "subject": "Matematik",
            "topic": "Trigonometri",
            "hint": "Bu soru için ipucu...",
            "explanation": question.explanation,
            "question_type": question.question_type,
            "tags": [],
            "created_by": question.created_by,
            "is_active": question.is_active
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get recommended question: {str(e)}"
        )


@router.post("/", response_model=QuestionResponse)
async def create_question(
    in_q: dict,  # QuestionCreate şeması kullanılabilir
    db: Session = Depends(get_db),
    embed_svc: EmbeddingService = Depends(get_embedding_service),
    vector_svc: VectorStoreService = Depends(get_vector_store_service),
):
    """
    Yeni soru oluşturur ve embedding'ini hesaplar
    """
    try:
        # 1) DB'ye kaydet (bu kısım crud fonksiyonu ile yapılacak)
        # q = crud.question.create(db, obj_in=in_q)
        
        # Şimdilik mock question
        q = {
            "id": 1,
            "content": in_q.get("content", ""),
            "options": in_q.get("options", []),
            "correct_answer": in_q.get("correct_answer", ""),
            "difficulty_level": in_q.get("difficulty_level", "medium"),
            "subject_id": in_q.get("subject_id", 1),
            "explanation": in_q.get("explanation", ""),
            "question_type": in_q.get("question_type", "multiple_choice"),
            "created_by": in_q.get("created_by", 1),
            "is_active": in_q.get("is_active", True)
        }

        # 2) Embedding'i hesapla & Redis cache'e yaz
        emb = await embed_svc.get_or_compute(q["id"], q["content"])

        # 3) FAISS index'e ekle ve diske kaydet
        if vector_svc.index is None:
            # Eğer index hiç train edilmemişse, train için örnek embedding dizisi    
            vector_svc.build(np.array(emb).reshape(1, -1))
        else:
            vector_svc.add_vectors(np.array(emb).reshape(1, -1), [q["id"]])
        vector_svc.save()

        return q
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create question: {str(e)}"
        )


@router.put("/{question_id}", response_model=QuestionResponse)
async def update_question(
    question_id: int,
    in_q: dict,  # QuestionUpdate şeması kullanılabilir
    db: Session = Depends(get_db),
    embed_svc: EmbeddingService = Depends(get_embedding_service),
    vector_svc: VectorStoreService = Depends(get_vector_store_service),
):
    """
    Soruyu günceller ve embedding'ini yeniden hesaplar
    """
    try:
        # 1) DB'de güncelle (bu kısım crud fonksiyonu ile yapılacak)
        # q = crud.question.update(db, db_obj=crud.question.get(db, question_id), obj_in=in_q)
        
        # Şimdilik mock question
        q = {
            "id": question_id,
            "content": in_q.get("content", ""),
            "options": in_q.get("options", []),
            "correct_answer": in_q.get("correct_answer", ""),
            "difficulty_level": in_q.get("difficulty_level", "medium"),
            "subject_id": in_q.get("subject_id", 1),
            "explanation": in_q.get("explanation", ""),
            "question_type": in_q.get("question_type", "multiple_choice"),
            "created_by": in_q.get("created_by", 1),
            "is_active": in_q.get("is_active", True)
        }
        
        if not q:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail="Question not found"
            )

        # 2) Yeni içeriğin embedding'ini hesapla & Redis cache'e overwrite
        emb = await embed_svc.get_or_compute(q["id"], q["content"])

        # 3) FAISS index'te eski embedding'i silip yenisini eklemek ideal ama
        #    IVF+PQ'da silme zor; en basit yol: yeniden build etmek:
        #    (Gerçek prod'da "deletion + add" destekli bir vector store kullanmayı düşünün)
        all_embs = []  # DB'den tekrar çek, veya bir cache'den al
        all_qs = [q]  # Mock: gerçekte crud.question.get_multi(db, skip=0, limit=1000000)
        for qq in all_qs:
            all_embs.append(await embed_svc.get_or_compute(qq["id"], qq["content"]))
        
        import numpy as np
        vector_svc.rebuild_index(np.vstack(all_embs))

        return q
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update question: {str(e)}"
        )


@router.post("/search", summary="Vector search ile benzer soruları bul")
async def search_similar_questions(
    query: str,
    top_k: int = 10,
    embed_svc: EmbeddingService = Depends(get_embedding_service),
    vector_svc: VectorStoreService = Depends(get_vector_store_service),
):
    """
    Metin sorgusu ile benzer soruları bulur
    """
    try:
        # 1) Sorgu embedding'ini hesapla
        query_emb = await embed_svc.compute_embedding(query)
        
        # 2) Vector search yap
        ids, distances = vector_svc.search(np.array(query_emb), top_k)
        
        # 3) Sonuçları formatla
        results = []
        for i, (question_id, distance) in enumerate(zip(ids, distances)):
            results.append({
                "question_id": question_id,
                "similarity": 1.0 - distance,  # Distance'ı similarity'ye çevir
                "rank": i + 1
            })
        
        return {
            "query": query,
            "results": results,
            "total_found": len(results)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search questions: {str(e)}"
        ) 