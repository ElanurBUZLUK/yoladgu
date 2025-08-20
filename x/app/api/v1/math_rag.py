from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging

from app.core.database import get_async_session
from app.middleware.auth import get_current_student, get_current_teacher
from app.models.user import User
from app.models.question import QuestionType
from app.services.llm_gateway import llm_gateway
from app.utils.distlock_idem import idempotency_decorator, IdempotencyConfig
from app.services.cache_service import cache_service # Added import
from app.services.advanced_retrieval import advanced_retrieval_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/math/rag", tags=["math-rag"])

class GenerateRequest(BaseModel):
    topic: Optional[str] = Field(default=None, description="ör: 'quadratic equations'")
    difficulty_level: Optional[int] = Field(default=1, ge=1, le=10)
    question_type: QuestionType = Field(default=QuestionType.MULTIPLE_CHOICE)
    n: int = Field(default=1, ge=1, le=10)

class GeneratedQuestion(BaseModel):
    kind: str
    stem: str
    options: Optional[List[str]] = None
    answer: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None

class GenerateResponse(BaseModel):
    items: List[GeneratedQuestion]
    usage: Dict[str, Any] = {}

class SolveRequest(BaseModel):
    problem: str
    show_steps: bool = True

class SolveResponse(BaseModel):
    solution: str
    steps: Optional[str] = None
    usage: Dict[str, Any] = {}

class CheckRequest(BaseModel):
    question: str
    user_answer: str
    answer_key: Optional[str] = None
    require_explanation: bool = True

class CheckResponse(BaseModel):
    correct: bool
    explanation: Optional[str] = None
    usage: Dict[str, Any] = {}

@router.get("/health")
async def health():
    return {"status": "ok", "module": "math-rag"}

@router.post("/generate", response_model=GenerateResponse, status_code=status.HTTP_200_OK)
@idempotency_decorator(
    key_builder=lambda body, user: f"generate:{user.id}:{body.topic}:{body.difficulty_level}:{body.question_type.value}",
    config=IdempotencyConfig(scope="math_rag", ttl_seconds=1800)
)
async def generate_questions(
    body: GenerateRequest,
    db: AsyncSession = Depends(get_async_session),
    user: User = Depends(get_current_teacher),
):
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array", "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string"},
                        "stem": {"type": "string"},
                        "options": {"type": "array", "items": {"type": "string"}},
                        "answer": {"type": "object"},
                        "meta": {"type": "object"},
                    },
                    "required": ["kind", "stem"]
                }
            },
            "usage": {"type": "object"}
        },
        "required": ["items"]
    }

    # Construct a query for retrieval based on the request
    retrieval_query = f"{body.topic or ''} math questions about {body.question_type.value} difficulty {body.difficulty_level}"
    
    # Perform retrieval
    retrieval_results = await advanced_retrieval_service.advanced_retrieve(
        query=retrieval_query,
        user_id=str(user.id),
        subject="math",
        topic=body.topic,
        difficulty_level=body.difficulty_level,
        limit=5 # Retrieve a few relevant documents
    )
    
    context_string = ""
    if retrieval_results and retrieval_results.get("results"):
        context_items = [f"Document {i+1}: {item.get('content', '')}" for i, item in enumerate(retrieval_results["results"]) ]
        context_string = "\n\n".join(context_items)
        logger.info(f"Retrieved {len(retrieval_results['results'])} documents for RAG.")
    else:
        logger.info("No relevant documents found for RAG.")

    sys_prompt = (
        "You are a math item writer. Produce exam-quality questions strictly as JSON.\n"
        "Use the provided context if relevant to generate the questions. If no context is provided, generate general questions.\n"
        f"Question type: {body.question_type.value}. Difficulty: {body.difficulty_level or 1}/10."
        + (f" Topic: {body.topic}.") if body.topic else ""
    )
    user_prompt = f"Generate {body.n} {body.question_type.value} math question(s). Return JSON matching the schema."

    # Semantic Cache Integration
    cache_key = f"llm_generate:{user.id}:{hash(user_prompt + sys_prompt + str(schema) + context_string)}"
    cached_llm_result = await cache_service.get(cache_key)

    if cached_llm_result:
        logger.info(f"Cache hit for generate: {cache_key}")
        data = cached_llm_result.get("parsed_json", {{}}) or {{}}
        usage = cached_llm_result.get("usage", {{}}) or {{}}
        items = data.get("items", [])
        return GenerateResponse(items=items, usage=usage)
    else:
        logger.info(f"Cache miss for generate: {cache_key}. Calling LLM.")
        try:
            # Try MCP first
            from app.core.mcp_utils import mcp_utils
            
            if mcp_utils.is_initialized:
                try:
                    mcp_response = await mcp_utils.call_tool(
                        tool_name="llm_generate",
                        arguments={
                            "prompt": user_prompt,
                            "system_prompt": sys_prompt,
                            "output_type": "json",
                            "schema": schema,
                            "temperature": 0.2,
                            "max_tokens": 900,
                            "context": context_string
                        }
                    )
                    
                    if mcp_response["success"]:
                        result = mcp_response["data"]
                        if isinstance(result, str):
                            import json
                            result = json.loads(result)
                    else:
                        logger.warning(f"MCP generate failed: {mcp_response.get('error')}")
                        # Fallback to direct LLM
                        result = await llm_gateway.generate_json(
                            prompt=user_prompt, system_prompt=sys_prompt, schema=schema, temperature=0.2, max_tokens=900, context=context_string
                        )
                except Exception as e:
                    logger.warning(f"MCP generate failed, using fallback: {e}")
                    result = await llm_gateway.generate_json(
                        prompt=user_prompt, system_prompt=sys_prompt, schema=schema, temperature=0.2, max_tokens=900, context=context_string
                    )
            else:
                result = await llm_gateway.generate_json(
                    prompt=user_prompt, system_prompt=sys_prompt, schema=schema, temperature=0.2, max_tokens=900, context=context_string
                )
            if not getattr(result, "success", False):
                raise HTTPException(status_code=502, detail=f"LLM error: {getattr(result,'error','unknown')}")
            
            # Cache the entire LLM result dictionary
            await cache_service.set(cache_key, result)
            logger.info(f"Cached LLM result for key: {cache_key}")

            data = getattr(result, "parsed_json", {{}}) or {{}}
            usage = getattr(result, "usage", {{}}) or {{}}
            items = data.get("items", [])
            return GenerateResponse(items=items, usage=usage)
        except Exception as e:
            logger.info("math-rag generate failed: %s", e)
            raise HTTPException(status_code=500, detail="math-rag generate failed")


@router.post("/solve", response_model=SolveResponse, status_code=status.HTTP_200_OK)
@idempotency_decorator(
    key_builder=lambda body, user: f"solve:{user.id}:{hash(body.problem)}",
    config=IdempotencyConfig(scope="math_rag", ttl_seconds=3600)
)
async def solve_problem(
    body: SolveRequest,
    db: AsyncSession = Depends(get_async_session),
    user: User = Depends(get_current_student),
):
    schema = {
        "type": "object",
        "properties": {"solution": {"type": "string"}, "steps": {"type": "string"}},
        "required": ["solution"],
    }

    # Perform retrieval for the problem
    retrieval_results = await advanced_retrieval_service.advanced_retrieve(
        query=body.problem,
        user_id=str(user.id),
        subject="math",
        limit=3 # Retrieve a few relevant documents
    )
    
    context_string = ""
    if retrieval_results and retrieval_results.get("results"):
        context_items = [f"Document {i+1}: {item.get('content', '')}" for i, item in enumerate(retrieval_results["results"]) ]
        context_string = "\n\n".join(context_items)
        logger.info(f"Retrieved {len(retrieval_results['results'])} documents for RAG in /solve.")
    else:
        logger.info("No relevant documents found for RAG in /solve.")

    sys_prompt = "You are a rigorous math solver. Return only JSON. Use the provided context if relevant to solve the problem."
    user_prompt = ("Solve the following problem. " +
                   ("Show steps as well." if body.show_steps else "Do not include steps.") +
                   f"\n\nProblem:\n{body.problem}")

    # Semantic Cache Integration
    cache_key = f"llm_solve:{user.id}:{hash(user_prompt + sys_prompt + str(schema) + context_string)}"
    cached_llm_result = await cache_service.get(cache_key)

    if cached_llm_result:
        logger.info(f"Cache hit for solve: {cache_key}")
        data = cached_llm_result.get("parsed_json", {}) or {}
        usage = cached_llm_result.get("usage", {}) or {}
        return SolveResponse(solution=data.get("solution", ""), steps=data.get("steps"), usage=usage)
    else:
        logger.info(f"Cache miss for solve: {cache_key}. Calling LLM.")
        try:
            # Try MCP first
            if mcp_utils.is_initialized:
                try:
                    mcp_response = await mcp_utils.call_tool(
                        tool_name="llm_generate",
                        arguments={
                            "prompt": user_prompt,
                            "system_prompt": sys_prompt,
                            "output_type": "json",
                            "schema": schema,
                            "temperature": 0.0,
                            "max_tokens": 800,
                            "context": context_string
                        }
                    )
                    
                    if mcp_response["success"]:
                        result = mcp_response["data"]
                        if isinstance(result, str):
                            import json
                            result = json.loads(result)
                    else:
                        logger.warning(f"MCP solve failed: {mcp_response.get('error')}")
                        # Fallback to direct LLM
                        result = await llm_gateway.generate_json(
                            prompt=user_prompt, system_prompt=sys_prompt, schema=schema, temperature=0.0, max_tokens=800, context=context_string
                        )
                except Exception as e:
                    logger.warning(f"MCP solve failed, using fallback: {e}")
                    result = await llm_gateway.generate_json(
                        prompt=user_prompt, system_prompt=sys_prompt, schema=schema, temperature=0.0, max_tokens=800, context=context_string
                    )
            else:
                result = await llm_gateway.generate_json(
                    prompt=user_prompt, system_prompt=sys_prompt, schema=schema, temperature=0.0, max_tokens=800, context=context_string
                )
            if not getattr(result, "success", False):
                raise HTTPException(status_code=502, detail=f"LLM error: {getattr(result,'error','unknown')}")
            
            # Cache the entire LLM result dictionary
            await cache_service.set(cache_key, result)
            logger.info(f"Cached LLM result for key: {cache_key}")

            data = getattr(result, "parsed_json", {}) or {}
            usage = getattr(result, "usage", {}) or {}
            return SolveResponse(solution=data.get("solution", ""), steps=data.get("steps"), usage=usage)
        except Exception as e:
            logger.error("math-rag solve failed: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail="math-rag solve failed")


@router.post("/check", response_model=CheckResponse, status_code=status.HTTP_200_OK)
async def check_answer(
    body: CheckRequest,
    db: AsyncSession = Depends(get_async_session),
    user: User = Depends(get_current_student),
):
    schema = {
        "type": "object",
        "properties": {"correct": {"type": "boolean"}, "explanation": {"type": "string"}},
        "required": ["correct"],
    }
    sys_prompt = ("You are a math answer checker. Return STRICT JSON only. "
                  "If an answer key is provided, compare strictly; otherwise judge correctness.")
    key_line = f"\nAnswerKey: {body.answer_key}" if body.answer_key else ""
    explain_line = "\nExplain briefly." if body.require_explanation else ""
    user_prompt = f"Question:\n{body.question}\nUserAnswer: {body.user_answer}{key_line}{explain_line}"

    # Semantic Cache Integration
    cache_key = f"llm_check:{user.id}:{hash(user_prompt + sys_prompt + str(schema))}"
    cached_llm_result = await cache_service.get(cache_key)

    if cached_llm_result:
        logger.info(f"Cache hit for check: {cache_key}")
        data = cached_llm_result.get("parsed_json", {}) or {}
        usage = cached_llm_result.get("usage", {}) or {}
        return CheckResponse(correct=bool(data.get("correct")), explanation=data.get("explanation"), usage=usage)
    else:
        logger.info(f"Cache miss for check: {cache_key}. Calling LLM.")
        try:
            result = await llm_gateway.generate_json(
                prompt=user_prompt, system_prompt=sys_prompt, schema=schema, temperature=0.0, max_tokens=400
            )
            if not getattr(result, "success", False):
                raise HTTPException(status_code=502, detail=f"LLM error: {getattr(result,'error','unknown')}")
            
            # Cache the entire LLM result dictionary
            await cache_service.set(cache_key, result)
            logger.info(f"Cached LLM result for key: {cache_key}")

            data = getattr(result, "parsed_json", {}) or {}
            usage = getattr(result, "usage", {}) or {}
            return CheckResponse(correct=bool(data.get("correct")), explanation=data.get("explanation"), usage=usage)
        except Exception as e:
            logger.exception("math-rag check failed: %s", e)
            raise HTTPException(status_code=500, detail="math-rag check failed")

class HintRequest(BaseModel):
    problem: str

class HintResponse(BaseModel):
    hint: str
    usage: Dict[str, Any] = {}

@router.post("/hint", response_model=HintResponse, status_code=status.HTTP_200_OK)
async def get_hint(
    body: HintRequest,
    db: AsyncSession = Depends(get_async_session),
    user: User = Depends(get_current_student),
):
    schema = {
        "type": "object",
        "properties": {"hint": {"type": "string"}},
        "required": ["hint"],
    }

    # Perform retrieval for the problem
    retrieval_results = await advanced_retrieval_service.advanced_retrieve(
        query=body.problem,
        user_id=str(user.id),
        subject="math",
        limit=3 # Retrieve a few relevant documents
    )
    
    context_string = ""
    if retrieval_results and retrieval_results.get("results"):
        context_items = [f"Document {i+1}: {item.get('content', '')}" for i, item in enumerate(retrieval_results["results"]) ]
        context_string = "\n\n".join(context_items)
        logger.info(f"Retrieved {len(retrieval_results['results'])} documents for RAG in /hint.")
    else:
        logger.info("No relevant documents found for RAG in /hint.")

    sys_prompt = "You are a helpful math tutor. Provide a single, concise hint for the user's problem. Do not solve the problem. Return only JSON. Use the provided context if relevant."
    user_prompt = f"Provide a hint for the following problem:\n\nProblem:\n{body.problem}"

    # Semantic Cache Integration
    cache_key = f"llm_hint:{user.id}:{hash(user_prompt + sys_prompt + str(schema) + context_string)}"
    cached_llm_result = await cache_service.get(cache_key)

    if cached_llm_result:
        logger.info(f"Cache hit for hint: {cache_key}")
        data = cached_llm_result.get("parsed_json", {}) or {}
        usage = cached_llm_result.get("usage", {}) or {}
        return HintResponse(hint=data.get("hint", ""), usage=usage)
    else:
        logger.info(f"Cache miss for hint: {cache_key}. Calling LLM.")
        try:
            result = await llm_gateway.generate_json(
                prompt=user_prompt, system_prompt=sys_prompt, schema=schema, temperature=0.1, max_tokens=200, context=context_string
            )
            if not getattr(result, "success", False):
                raise HTTPException(status_code=502, detail=f"LLM error: {getattr(result,'error','unknown')}")
            
            # Cache the entire LLM result dictionary
            await cache_service.set(cache_key, result)
            logger.info(f"Cached LLM result for key: {cache_key}")

            data = getattr(result, "parsed_json", {}) or {}
            usage = getattr(result, "usage", {}) or {}
            return HintResponse(hint=data.get("hint", ""), usage=usage)
        except Exception as e:
            logger.exception("math-rag hint failed: %s", e)
            raise HTTPException(status_code=500, detail="math-rag hint failed")
