from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import logging

from app.database_enhanced import enhanced_database_manager
from app.middleware.auth import get_current_student
from app.models.user import User
from app.services.llm_gateway import llm_gateway
from app.utils.distlock_idem import idempotency_decorator, IdempotencyConfig
from app.services.cache_service import cache_service # Added import

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/english/rag", tags=["english-rag"])

class RetrieveRequest(BaseModel):
    query: str
    namespace: str = "default"
    slot: str = "active"
    k: int = Field(default=5, ge=1, le=50)

class RetrievedItem(BaseModel):
    obj_ref: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    dist: float

class RetrieveResponse(BaseModel):
    items: List[RetrievedItem]

class AnswerRequest(RetrieveRequest):
    question: str
    compress_context: bool = True
    max_context: int = 1600

class AnswerResponse(BaseModel):
    answer: str
    sources: List[RetrievedItem]

@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(
    body: RetrieveRequest,
    db: AsyncSession = Depends(enhanced_database_manager.get_session),
    user: User = Depends(get_current_student),
):
    sql = text("""
        SELECT obj_ref, meta, (embedding <=> :q) AS dist
        FROM embeddings
        WHERE namespace = :ns AND slot = :slot
        ORDER BY dist
        LIMIT :k
    """)
    try:
        qvec = await llm_gateway.embed_query(body.query)  # yoksa embedding_service'i çağır
        res = await db.execute(sql.bindparams(q=qvec, ns=body.namespace, slot=body.slot, k=body.k))
        rows = res.fetchall() or []
        items = [RetrievedItem(obj_ref=r[0], meta=r[1], dist=float(r[2])) for r in rows]
        return RetrieveResponse(items=items)
    except Exception as e:
        logger.exception("english-rag retrieve failed: %s", e)
        raise HTTPException(status_code=500, detail="retrieve failed")

@router.post("/answer", response_model=AnswerResponse, status_code=status.HTTP_200_OK)
@idempotency_decorator(
    key_builder=lambda body, user: f"answer:{user.id}:{hash(body.question + body.query)}",
    config=IdempotencyConfig(scope="english_rag", ttl_seconds=3600)
)
async def answer(
    body: AnswerRequest,
    db: AsyncSession = Depends(enhanced_database_manager.get_session),
    user: User = Depends(get_current_student),
):
    ret = await retrieve(RetrieveRequest(query=body.query, namespace=body.namespace, slot=body.slot, k=body.k), db, user)
    items = ret.items

    context_chunks: List[str] = []
    for it in items:
        if not it.meta:
            continue
        text_part = it.meta.get("text") if isinstance(it.meta, dict) else str(it.meta)
        if text_part:
            context_chunks.append(text_part)
    context = "\n\n".join(context_chunks)

    if body.compress_context and context:
        try:
            # Try MCP for context compression
            from app.core.mcp_utils import mcp_utils
            
            if mcp_utils.is_initialized:
                try:
                    compression_response = await mcp_utils.call_tool(
                        tool_name="compress_context",
                        arguments={
                            "context": context,
                            "max_length": body.max_context,
                            "user_id": str(user.id)
                        }
                    )
                    
                    if compression_response["success"]:
                        context = compression_response["data"].get("compressed_context", context)
                        logger.info("Context compressed via MCP")
                    else:
                        logger.warning("MCP context compression failed, using direct LLM")
                        context = await llm_gateway.compress_context(context, max_len=body.max_context)
                except Exception as e:
                    logger.warning(f"MCP context compression failed, using fallback: {e}")
                    context = await llm_gateway.compress_context(context, max_len=body.max_context)
            else:
                # Direct LLM compression
                context = await llm_gateway.compress_context(context, max_len=body.max_context)
        except Exception as e:
            logger.warning("context compression failed, using raw: %s", e)

    system_prompt = "You are an English tutor. Answer using only the provided context. If not answerable, say so."
    user_prompt = f"Question:\n{body.question}\n\nContext:\n{context}"

    # Semantic Cache Integration
    cache_key = f"llm_answer:{user.id}:{hash(user_prompt + system_prompt)}"
    cached_llm_result = await cache_service.get(cache_key)

    if cached_llm_result:
        logger.info(f"Cache hit for answer: {cache_key}")
        # Assuming cached_llm_result is the dict returned by llm_gateway.generate_text
        return AnswerResponse(answer=cached_llm_result.get("text", ""), sources=items)
    else:
        logger.info(f"Cache miss for answer: {cache_key}. Calling LLM.")
        try:
            # Try MCP first
            from app.core.mcp_utils import mcp_utils
            
            if mcp_utils.is_initialized:
                try:
                    mcp_response = await mcp_utils.call_tool(
                        tool_name="llm_generate",
                        arguments={
                            "prompt": user_prompt,
                            "system_prompt": system_prompt,
                            "output_type": "text",
                            "temperature": 0.2,
                            "max_tokens": 700,
                            "context": context,
                            "user_id": str(user.id),
                            "session_id": f"english_rag_{user.id}"
                        }
                    )
                    
                    if mcp_response["success"]:
                        result = mcp_response["data"]
                        if isinstance(result, str):
                            import json
                            result = json.loads(result)
                        
                        # Cache the MCP result
                        await cache_service.set(cache_key, result)
                        logger.info(f"Cached MCP result for key: {cache_key}")
                        
                        return AnswerResponse(answer=result.get("text", ""), sources=items)
                    else:
                        logger.warning(f"MCP answer failed: {mcp_response.get('error')}")
                        # Fallback to direct LLM
                        result = await llm_gateway.generate_text(
                            prompt=user_prompt, system_prompt=system_prompt, temperature=0.2, max_tokens=700
                        )
                except Exception as e:
                    logger.warning(f"MCP answer failed, using fallback: {e}")
                    # Fallback to direct LLM
                    result = await llm_gateway.generate_text(
                        prompt=user_prompt, system_prompt=system_prompt, temperature=0.2, max_tokens=700
                    )
            else:
                # Direct LLM call
                result = await llm_gateway.generate_text(
                    prompt=user_prompt, system_prompt=system_prompt, temperature=0.2, max_tokens=700
                )
            
            if not getattr(result, "success", False):
                raise HTTPException(status_code=502, detail=f"LLM error: {getattr(result,'error','unknown')}")
            
            # Cache the result
            await cache_service.set(cache_key, result)
            logger.info(f"Cached result for key: {cache_key}")

            return AnswerResponse(answer=getattr(result, "text", ""), sources=items)
        except Exception as e:
            logger.exception("english-rag answer failed: %s", e)
            raise HTTPException(status_code=500, detail="answer failed")