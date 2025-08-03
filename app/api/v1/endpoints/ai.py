from app.crud.user import get_current_user
from app.db.database import get_db
from app.db.models import Question, User
from app.schemas.ai import (
    AdaptiveHintRequest,
    AdaptiveHintResponse,
    BatchEnrichRequest,
    BatchEnrichResponse,
    ContextualExplanationRequest,
    ContextualExplanationResponse,
    DifficultyRequest,
    DifficultyResponse,
    ExplanationRequest,
    ExplanationResponse,
    HintRequest,
    HintResponse,
    IngestCSVRequest,
    IngestResponse,
    IngestWebsiteRequest,
    LLMStatusResponse,
)
from app.services.llm_service import llm_service
from app.services.question_ingestion_service import get_question_ingestion_service
from app.services.recommendation_service import recommendation_service
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

router = APIRouter(prefix="/ai", tags=["ai"])


@router.post("/generate-hint", response_model=HintResponse)
async def generate_hint(request: HintRequest):
    """Soru için ipucu üret (Batch)"""
    try:
        hint = await llm_service.generate_question_hint(request.question, "mathematics")
        return HintResponse(hint=hint)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/generate-explanation", response_model=ExplanationResponse)
async def generate_explanation(request: ExplanationRequest):
    """Soru için açıklama üret (Batch)"""
    try:
        explanation = await llm_service.generate_question_explanation(
            request.question, request.answer, "mathematics"
        )
        return ExplanationResponse(explanation=explanation)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/analyze-question-difficulty", response_model=DifficultyResponse)
async def analyze_question_difficulty(request: DifficultyRequest):
    """Soru zorluğunu analiz et (Batch)"""
    try:
        analysis = await llm_service.analyze_question_difficulty(
            request.question, "mathematics"
        )
        difficulty = f"Seviye {analysis.get('difficulty_level', 1)} - {analysis.get('explanation', 'Analiz edilemedi')}"
        return DifficultyResponse(difficulty=difficulty)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/adaptive-hint", response_model=AdaptiveHintResponse)
async def get_adaptive_hint(
    request: AdaptiveHintRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Öğrenci durumuna göre adaptif ipucu al (Runtime)"""
    try:
        hint = await recommendation_service.get_adaptive_hint(
            request.question_id, request.student_id, db
        )
        return AdaptiveHintResponse(hint=hint)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/contextual-explanation", response_model=ContextualExplanationResponse)
async def get_contextual_explanation(
    request: ContextualExplanationRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Öğrencinin cevabına göre bağlamsal açıklama al (Runtime)"""
    try:
        explanation = await recommendation_service.get_contextual_explanation(
            request.question_id, request.student_id, request.student_answer, db
        )
        return ContextualExplanationResponse(explanation=explanation)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/batch-enrich-questions", response_model=BatchEnrichResponse)
async def batch_enrich_questions(
    request: BatchEnrichRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Mevcut soruları batch olarak zenginleştir"""
    try:
        # Get async ingestion service
        ingestion_service = await get_question_ingestion_service()

        # Mevcut soruları bul
        questions = (
            db.query(Question)
            .filter(
                Question.subject_id
                == ingestion_service._get_subject_id(request.subject)
            )
            .all()
        )

        enriched_count = 0
        for question in questions:
            if not question.hint or not question.explanation:
                # Zorluk analizi
                difficulty_analysis = await llm_service.analyze_question_difficulty(
                    question.content, request.subject
                )

                # İpucu üret
                if not question.hint:
                    question.hint = await llm_service.generate_question_hint(
                        question.content, request.subject
                    )

                # Açıklama üret
                if not question.explanation:
                    question.explanation = (
                        await llm_service.generate_question_explanation(
                            question.content, question.correct_answer, request.subject
                        )
                    )

                enriched_count += 1

        db.commit()
        return BatchEnrichResponse(enriched_count=enriched_count)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/llm-status", response_model=LLMStatusResponse)
async def get_llm_status():
    """LLM servis durumunu kontrol et"""
    try:
        openai_configured = bool(llm_service._get_api_key())
        huggingface_configured = bool(llm_service._get_api_key())

        status = (
            "ready"
            if (openai_configured or huggingface_configured)
            else "not_configured"
        )

        return LLMStatusResponse(
            openai_configured=openai_configured,
            huggingface_configured=huggingface_configured,
            status=status,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/ingest-from-website", response_model=IngestResponse)
async def ingest_from_website(
    request: IngestWebsiteRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Web sitesinden soru içe aktar"""
    try:
        ingestion_service = await get_question_ingestion_service()
        questions = await ingestion_service.scrape_from_website(
            request.url, request.subject, request.topic
        )
        saved_count = await ingestion_service.save_questions_to_database(
            questions, db, current_user.id
        )
        return IngestResponse(saved_count=saved_count)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/ingest-from-csv", response_model=IngestResponse)
async def ingest_from_csv(
    request: IngestCSVRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """CSV dosyasından soru içe aktar"""
    try:
        ingestion_service = await get_question_ingestion_service()
        questions = await ingestion_service.ingest_from_csv(
            request.file_path, request.subject
        )
        saved_count = await ingestion_service.save_questions_to_database(
            questions, db, current_user.id
        )
        return IngestResponse(saved_count=saved_count)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# Eski endpoint'ler (geriye uyumluluk için)
@router.post("/hint", response_model=HintResponse)
async def generate_hint_legacy(request: HintRequest):
    """Eski ipucu endpoint'i (geriye uyumluluk)"""
    return await generate_hint(request)


@router.post("/explanation", response_model=ExplanationResponse)
async def generate_explanation_legacy(request: ExplanationRequest):
    """Eski açıklama endpoint'i (geriye uyumluluk)"""
    return await generate_explanation(request)


@router.post("/difficulty", response_model=DifficultyResponse)
async def analyze_difficulty_legacy(request: DifficultyRequest):
    """Eski zorluk analizi endpoint'i (geriye uyumluluk)"""
    return await analyze_question_difficulty(request)
