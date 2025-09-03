"""
Question generation endpoints.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer
from typing import List, Dict, Any
import logging
import uuid
from datetime import datetime
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.request import GenerateMathRequest, GenerateEnglishRequest
from app.models.response import (
    GenerateMathResponse, GenerateEnglishResponse, MathItem, ErrorResponse
)
from app.services.math_generation_service import math_generation_service
from app.db.session import get_db

router = APIRouter()
security = HTTPBearer()
logger = logging.getLogger(__name__)


@router.post("/math", response_model=GenerateMathResponse)
async def generate_math_question(
    request: GenerateMathRequest,
    token: str = Depends(security)
):
    """
    Generate a math question using parametric templates.
    
    This endpoint:
    1. Uses predefined templates (linear equations, ratios, geometry, etc.)
    2. Generates parameters within specified constraints
    3. Creates question stem and answer choices
    4. Validates using programmatic solver (SymPy)
    5. Ensures single correct answer
    6. Generates misconception-based distractors
    """
    try:
        logger.info(f"Generating math question with template: {request.template_id}")
        
        # Validate template exists
        available_templates = math_generation_service.get_available_templates()
        template_ids = [t["template_id"] for t in available_templates]
        
        if request.template_id not in template_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown template ID: {request.template_id}. Available: {template_ids}"
            )
        
        # Validate difficulty range
        if request.target_difficulty is not None:
            if not -1.0 <= request.target_difficulty <= 1.0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Target difficulty must be between -1.0 and 1.0"
                )
        
        # Generate question
        question_data = math_generation_service.generate_question(
            template_id=request.template_id,
            params_hint=request.params_hint,
            target_difficulty=request.target_difficulty,
            language=request.language,
            rationale_required=request.rationale_required
        )
        
        # Create response
        math_item = MathItem(
            stem=question_data["item"]["stem"],
            choices=question_data["item"].get("choices"),
            answer_key=question_data["item"]["answer_key"],
            solution=question_data["item"].get("solution"),
            skills=question_data["item"]["skills"],
            bloom_level=question_data["item"]["bloom_level"],
            difficulty_estimate=question_data["item"]["difficulty_estimate"],
            qa_checks=question_data["item"]["qa_checks"]
        )
        
        # Convert generator params to strings for response model
        generator_data = question_data["generator"].copy()
        if "params" in generator_data and isinstance(generator_data["params"], dict):
            generator_data["params"] = str(generator_data["params"])
        
        response = GenerateMathResponse(
            item=math_item,
            generator=generator_data
        )
        
        logger.info(f"Successfully generated math question: {question_data['generator']['template_id']}")
        return response
        
    except ValueError as e:
        logger.error(f"Math generation validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Math generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate math question"
        )


@router.post("/en_cloze", response_model=GenerateEnglishResponse)
async def generate_english_cloze(
    request: GenerateEnglishRequest,
    session: AsyncSession = Depends(get_db),
    token: str = Depends(security)
):
    """
    Generate English cloze (fill-in-the-blank) questions.
    
    This endpoint:
    1. Selects appropriate passage based on CEFR level and topic
    2. Identifies target error patterns (prepositions, articles, etc.)
    3. Creates blanks targeting specific grammar points
    4. Generates personalized distractors based on error history
    5. Validates single correct answer using grammar checker
    6. Saves to database and indexes for retrieval
    """
    try:
        from app.services.english_generation_service import english_cloze_generator, CEFRLevel
        from app.db.models import EnglishItem as EnglishItemDB
        from app.services.search_service import search_service
        from app.services.vector_service import vector_service
        from app.models.response import EnglishItem as EnglishItemResp, EnglishBlank as EnglishBlankResp
        
        logger.info(f"Generating English cloze for CEFR level: {request.level_cefr}, tags: {request.target_error_tags}")
        
        # 1) Generate and validate cloze question
        level = CEFRLevel[request.level_cefr]
        question, validation = english_cloze_generator.generate_validated_cloze_question(
            level_cefr=level,
            target_error_tags=request.target_error_tags,
            topic=request.topic,
            personalization=request.personalization,
        )
        
        if not question:
    raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Cloze question generation failed"
            )
        
        # 1.5) Enhance with Vector DB examples for better distractors
        try:
            from app.services.english_vector_examples import get_vector_examples, mine_distractors_from_examples
            
            # Fetch similar examples from Vector DB
            examples = await get_vector_examples(
                session=session,
                topic=request.topic,
                target_error_tags=request.target_error_tags,
                level_cefr=request.level_cefr,
                k=5,
            )
            
            # Augment distractors with example-driven candidates, then re-validate
            if question and question.blanks and examples:
                changed = False
                for b in question.blanks:
                    extra = mine_distractors_from_examples(examples, b.skill_tag, exclude=b.answer, limit=4)
                    # merge into distractors
                    if extra:
                        # ensure uniqueness and cap at 6 distractors total
                        pool = list(dict.fromkeys((b.distractors or []) + extra))
                        pool = [x for x in pool if x.strip().lower() != str(b.answer).strip().lower()]
                        b.distractors = pool[:6]
                        changed = True

                if changed:
                    # Re-run the single-answer validator to ensure quality
                    ok = english_cloze_generator.validate_cloze_question(question)
                    if not ok.get("single_answer_guaranteed", True):
                        # If validation fails, you may trim some distractors
                        for b in question.blanks:
                            b.distractors = (b.distractors or [])[:3]
                        ok = english_cloze_generator.validate_cloze_question(question)
                    validation = ok
                    
                logger.info(f"Enhanced question with {len(examples)} vector examples")
                
        except Exception as e:
            logger.warning(f"Vector examples enhancement failed: {e}")
            # Continue with original question if enhancement fails
        
        # 2) Save to database
        item_db = EnglishItemDB(
            tenant_id="default",  # TODO: get from token/context
            passage=question.passage,
            blanks=[
                {
                    "span": b.span,
                    "answer": b.answer,
                    "distractors": b.distractors,
                    "skill_tag": b.skill_tag,
                    "position": b.position,
                    "rationale": b.rationale,
                }
                for b in question.blanks
            ],
            level_cefr=question.level_cefr.name,
            topic=question.topic,
            error_tags=question.error_tags or request.target_error_tags,
            lang="en",
            status="active",
            ambiguity_flag=not validation.get("single_answer_guaranteed", True),
            pedagogy_flags={"validation": validation},
        )
        
        session.add(item_db)
        await session.commit()
        await session.refresh(item_db)
        
        logger.info(f"Saved English item to DB: {item_db.id}")
        
        # 3) Index in Elasticsearch (sparse search)
        try:
            await search_service.index_item({
                "id": item_db.id,
                "type": "english",
                "lang": "en",
                "passage": item_db.passage,
                "error_tags": item_db.error_tags,
                "level_cefr": item_db.level_cefr,
                "topic": item_db.topic,
                "created_at": item_db.created_at.isoformat() if item_db.created_at else None,
                "updated_at": item_db.updated_at.isoformat() if item_db.updated_at else None,
            })
            logger.info(f"Indexed item {item_db.id} in Elasticsearch")
        except Exception as es_error:
            logger.error(f"Elasticsearch indexing failed for item {item_db.id}: {es_error}")
            # Continue with dense indexing
        
        # 4) Index in vector service (dense search)
        try:
            await vector_service.add_item(
                item_id=item_db.id,
                text=item_db.passage,  # Could combine with error_tags for better embedding
                metadata={
                    "type": "english",
                    "lang": "en",
                    "error_tags": item_db.error_tags,
                    "level_cefr": item_db.level_cefr,
                    "topic": item_db.topic,
                }
            )
            logger.info(f"Indexed item {item_db.id} in vector service")
        except Exception as vec_error:
            logger.error(f"Vector indexing failed for item {item_db.id}: {vec_error}")
        
        # 5) Create response model
        resp_item = EnglishItemResp(
            passage=item_db.passage,
            blanks=[
                EnglishBlankResp(
                    span=b["span"], 
                    answer=b["answer"], 
                    distractors=b["distractors"], 
                    skill_tag=b["skill_tag"]
                ) for b in item_db.blanks
            ],
            qa_checks={
                "valid": validation.get("valid", True),
                "single_answer": validation.get("single_answer_guaranteed", True),
                "grammar_ok": validation.get("grammar_validation", {}).get("valid", True),
            }
        )
        
        response = GenerateEnglishResponse(
            item=resp_item,
            generator={"engine": "rule-based", "validator": "GrammarValidator"}
        )
        
        logger.info(f"Successfully generated English cloze: {item_db.id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"English cloze generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate English cloze: {str(e)}"
    )


@router.get("/templates/math")
async def list_math_templates(token: str = Depends(security)):
    """
    List available math question templates.
    
    Returns template metadata including:
    - Template IDs and names
    - Skill tags and Bloom levels
    - Difficulty ranges
    - Parameter specifications
    """
    try:
        templates = math_generation_service.get_available_templates()
        
        # Enhance template info with additional metadata
        enhanced_templates = []
        for template in templates:
            enhanced = {
                **template,
                "description": _get_template_description(template["template_id"]),
                "parameter_specs": _get_parameter_specs(template["template_id"]),
                "example_params": _get_example_params(template["template_id"])
            }
            enhanced_templates.append(enhanced)
        
        return {
            "templates": enhanced_templates,
            "total_count": len(enhanced_templates),
            "generated_at": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Template listing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list math templates"
        )


@router.post("/math/validate")
async def validate_math_question(
    question_data: Dict[str, Any],
    token: str = Depends(security)
):
    """
    Validate a generated math question.
    
    Performs quality assurance checks:
    - Solver validation
    - Single solution guarantee
    - Distractor quality analysis
    - Parameter constraint validation
    """
    try:
        validation_results = math_generation_service.validate_generated_question(question_data)
        
        return {
            "validation_results": validation_results,
            "overall_valid": all(validation_results.values()),
            "validated_at": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Question validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate question"
        )


@router.post("/math/batch")
async def generate_math_batch(
    template_id: str,
    count: int,
    params_hint: Dict[str, Any] = None,
    language: str = "tr",
    token: str = Depends(security)
):
    """
    Generate multiple math questions in batch.
    
    Useful for:
    - Pre-generating question pools
    - Testing template variations
    - Bulk content creation
    """
    try:
        if count > 50:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Batch size cannot exceed 50 questions"
            )
        
        questions = math_generation_service.batch_generate(
            template_id=template_id,
            count=count,
            params_hint=params_hint,
            language=language
        )
        
        return {
            "questions": questions,
            "generated_count": len(questions),
            "requested_count": count,
            "template_id": template_id,
            "generated_at": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Batch generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate question batch"
        )


@router.post("/math/solve")
async def solve_equation(
    equation: str,
    variable: str = "x",
    language: str = "tr",
    token: str = Depends(security)
):
    """
    Solve arbitrary mathematical equations using SymPy.
    
    Useful for:
    - Testing solver capabilities
    - Validating custom equations
    - Educational demonstrations
    """
    try:
        result = math_generation_service.solve_arbitrary_equation(
            equation_str=equation,
            variable=variable,
            language=language
        )
        
        return {
            "equation": equation,
            "variable": variable,
            "result": result,
            "solved_at": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Equation solving error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to solve equation"
        )


class DistractorRequest(BaseModel):
    """Distractor generation request model."""
    template_id: str
    params: Dict[str, Any]
    correct_answer: Any
    num_distractors: int = 3
    analysis: bool = False

@router.post("/math/distractors")
async def generate_distractors(
    request: DistractorRequest,
    token: str = Depends(security)
):
    """
    Generate distractors for a given question.
    
    Useful for:
    - Testing distractor quality
    - Analyzing misconception patterns
    - Custom question creation
    """
    try:
        if request.num_distractors > 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot generate more than 10 distractors"
            )
        
        result = math_generation_service.generate_distractors_for_question(
            template_id=request.template_id,
            params=request.params,
            correct_answer=request.correct_answer,
            num_distractors=request.num_distractors,
            analysis=request.analysis
        )
        
        return {
            **result,
            "generated_at": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Distractor generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate distractors"
        )


@router.get("/math/misconceptions/{template_id}")
async def list_misconceptions(
    template_id: str,
    token: str = Depends(security)
):
    """
    List available misconception patterns for a template.
    
    Useful for:
    - Understanding error patterns
    - Educational research
    - Distractor analysis
    """
    try:
        misconceptions = math_generation_service.get_misconception_patterns(template_id)
        
        return {
            "template_id": template_id,
            "misconceptions": misconceptions,
            "count": len(misconceptions),
            "retrieved_at": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Misconception listing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list misconceptions"
        )


@router.get("/templates/english")
async def list_english_templates(token: str = Depends(security)):
    """List available English question templates and error patterns."""
    try:
        from app.services.english_generation_service import english_cloze_generator
        
        # Get available error patterns and CEFR levels
        error_patterns = english_cloze_generator.get_available_error_patterns()
        cefr_levels = [level.name for level in english_cloze_generator.get_available_cefr_levels()]
        
        return {
            "available_error_types": error_patterns,
            "cefr_levels": cefr_levels,
            "status": "active",
            "generator": "rule-based",
            "validator": "grammar-checker"
        }
    except Exception as e:
        logger.error(f"Error listing English templates: {e}")
    return {
            "message": "English cloze generation service not available",
        "available_error_types": [
            "prepositions", "articles", "subject_verb_agreement", 
            "collocations", "verb_forms", "conditionals"
        ],
        "cefr_levels": ["A1", "A2", "B1", "B2", "C1"],
            "status": "error",
            "error": str(e)
        }


@router.post("/admin/reindex_english")
async def reindex_english(
    session: AsyncSession = Depends(get_db),
    token: str = Depends(security)
):
    """
    Admin endpoint to reindex all existing English items.
    
    This endpoint:
    1. Fetches all active English items from database
    2. Bulk indexes them in Elasticsearch
    3. Bulk indexes them in vector service
    4. Returns count of indexed items
    
    Use this for initial setup or after schema changes.
    """
    try:
        from app.db.repositories.item import english_item_repository
        from app.services.search_service import search_service
        from app.services.vector_service import vector_service
        
        logger.info("Starting English items reindexing...")
        
        # Fetch active English items from DB
        rows = await english_item_repository.list_active(
            session, 
            tenant_id="default", 
            lang="en", 
            limit=10000
        )
        
        if not rows:
            logger.warning("No English items found in database")
            return {"count": 0, "message": "No items to index"}
        
        logger.info(f"Found {len(rows)} English items to index")
        
        # Prepare Elasticsearch bulk index data
        es_docs = []
        for row in rows:
            es_docs.append({
                "id": row.id,
                "type": "english",
                "lang": "en",
                "passage": row.passage,
                "error_tags": row.error_tags,
                "level_cefr": row.level_cefr,
                "topic": row.topic,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            })
        
        # Bulk index in Elasticsearch
        try:
            await search_service.bulk_index(es_docs)
            logger.info(f"Successfully indexed {len(es_docs)} items in Elasticsearch")
        except Exception as es_error:
            logger.error(f"Elasticsearch bulk indexing failed: {es_error}")
            return {
                "count": 0,
                "error": f"Elasticsearch indexing failed: {str(es_error)}"
            }
        
        # Bulk index in vector service (in batches)
        try:
            vector_payloads = []
            for row in rows:
                vector_payloads.append({
                    "id": row.id,
                    "text": row.passage,
                    "metadata": {
                        "type": "english",
                        "lang": "en",
                        "error_tags": row.error_tags,
                        "level_cefr": row.level_cefr,
                        "topic": row.topic,
                    }
                })
            
            # Add items to vector service (one by one for now)
            # TODO: Implement batch_add if vector service supports it
            indexed_count = 0
            for payload in vector_payloads:
                try:
                    await vector_service.add_item(
                        item_id=payload["id"],
                        text=payload["text"],
                        metadata=payload["metadata"]
                    )
                    indexed_count += 1
                except Exception as item_error:
                    logger.error(f"Failed to index item {payload['id']} in vector service: {item_error}")
            
            logger.info(f"Successfully indexed {indexed_count} items in vector service")
            
        except Exception as vec_error:
            logger.error(f"Vector service indexing failed: {vec_error}")
            # Don't fail the entire operation if vector indexing fails
        
        return {
            "count": len(rows),
            "es_indexed": len(es_docs),
            "vector_indexed": indexed_count,
            "message": f"Reindexing completed. {len(rows)} items processed."
        }
        
    except Exception as e:
        logger.error(f"English reindexing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reindex English items: {str(e)}"
        )


@router.post("/admin/reindex_math")
async def reindex_math(
    session: AsyncSession = Depends(get_db),
    token: str = Depends(security)
):
    """
    Admin endpoint to reindex all existing Math items.
    
    This endpoint:
    1. Fetches all active Math items from database
    2. Bulk indexes them in Elasticsearch
    3. Bulk indexes them in vector service
    4. Returns count of indexed items
    
    Use this for initial setup or after schema changes.
    """
    try:
        from app.db.repositories.item import math_item_repository
        from app.services.search_service import search_service
        from app.services.vector_service import vector_service
        
        logger.info("Starting Math items reindexing...")
        
        # Fetch active Math items from DB
        rows = await math_item_repository.list_active(
            session, 
            tenant_id="default", 
            lang="tr", 
            limit=10000
        )
        
        if not rows:
            logger.warning("No Math items found in database")
            return {"count": 0, "message": "No items to index"}
        
        logger.info(f"Found {len(rows)} Math items to index")
        
        # Prepare Elasticsearch bulk index data
        es_docs = []
        for row in rows:
            es_docs.append({
                "id": row.id,
                "type": "math",
                "lang": "tr",
                "stem": row.stem,
                "skills": row.skills,
                "bloom_level": row.bloom_level,
                "topic": row.topic,
                "difficulty_b": row.difficulty_b,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            })
        
        # Bulk index in Elasticsearch
        try:
            await search_service.bulk_index(es_docs)
            logger.info(f"Successfully indexed {len(es_docs)} items in Elasticsearch")
        except Exception as es_error:
            logger.error(f"Elasticsearch bulk indexing failed: {es_error}")
            return {
                "count": 0,
                "error": f"Elasticsearch indexing failed: {str(es_error)}"
            }
        
        # Bulk index in vector service
        try:
            indexed_count = 0
            for row in rows:
                try:
                    await vector_service.add_item(
                        item_id=row.id,
                        text=row.stem,  # Use question stem for embedding
                        metadata={
                            "type": "math",
                            "lang": "tr",
                            "skills": row.skills,
                            "bloom_level": row.bloom_level,
                            "topic": row.topic,
                            "difficulty_b": row.difficulty_b,
                        }
                    )
                    indexed_count += 1
                except Exception as item_error:
                    logger.error(f"Failed to index item {row.id} in vector service: {item_error}")
            
            logger.info(f"Successfully indexed {indexed_count} items in vector service")
            
        except Exception as vec_error:
            logger.error(f"Vector service indexing failed: {vec_error}")
            # Don't fail the entire operation if vector indexing fails
        
        return {
            "count": len(rows),
            "es_indexed": len(es_docs),
            "vector_indexed": indexed_count,
            "message": f"Reindexing completed. {len(rows)} items processed."
        }
        
    except Exception as e:
        logger.error(f"Math reindexing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reindex Math items: {str(e)}"
        )


# Helper functions
def _get_template_description(template_id: str) -> str:
    """Get human-readable description for template."""
    descriptions = {
        "linear_equation_v1": "Linear equations of the form ax + b = c with integer solutions",
        "quadratic_equation_v1": "Quadratic equations with factorable integer roots",
        "ratio_proportion_v1": "Ratio and proportion problems with cross multiplication"
    }
    return descriptions.get(template_id, "No description available")


def _get_parameter_specs(template_id: str) -> Dict[str, Any]:
    """Get parameter specifications for template."""
    specs = {
        "linear_equation_v1": {
            "a": {"type": "int", "range": [1, 9], "exclude": [0], "description": "Coefficient of x"},
            "b": {"type": "int", "range": [-10, 10], "description": "Constant term"},
            "c": {"type": "int", "range": [-10, 10], "description": "Right-hand side value"}
        },
        "quadratic_equation_v1": {
            "a": {"type": "int", "value": 1, "description": "Leading coefficient (fixed at 1)"},
            "roots": {"type": "list", "range": [-5, 5], "description": "Target roots for factoring"}
        },
        "ratio_proportion_v1": {
            "a": {"type": "int", "range": [2, 12], "description": "First ratio numerator"},
            "b": {"type": "int", "range": [2, 12], "description": "First ratio denominator"},
            "multiplier": {"type": "int", "range": [2, 8], "description": "Scaling factor"}
        }
    }
    return specs.get(template_id, {})


def _get_example_params(template_id: str) -> Dict[str, Any]:
    """Get example parameters for template."""
    examples = {
        "linear_equation_v1": {"a": 2, "b": 3, "c": 7},
        "quadratic_equation_v1": {"a": 1, "b": -5, "c": 6, "roots": [2, 3]},
        "ratio_proportion_v1": {"a": 3, "b": 4, "c": 9, "x": 12}
    }
    return examples.get(template_id, {})