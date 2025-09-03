"""
Vector-DB assisted example retrieval for English cloze generation.
Uses existing vector_service to fetch semantically similar English items,
then loads full items from DB to mine style/distractors.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.vector_service import vector_service
from app.db.repositories.item import english_item_repository

async def get_vector_examples(
    session: AsyncSession,
    topic: Optional[str],
    target_error_tags: Optional[List[str]],
    level_cefr: Optional[str],
    k: int = 5
) -> List[Dict[str, Any]]:
    """
    Retrieve semantically similar English items from Vector DB.
    
    Args:
        session: Database session
        topic: Topic filter
        target_error_tags: Error tags filter
        level_cefr: CEFR level filter
        k: Number of examples to retrieve
        
    Returns:
        List of example items with full content
    """
    # Build a simple query string from topic + error tags + level
    parts = []
    if topic: parts.append(str(topic))
    if target_error_tags: parts.extend(target_error_tags)
    if level_cefr: parts.append(str(level_cefr))
    query = " ".join(parts) if parts else "english cloze"

    # Vector search with filters
    filters = {"type": "english", "lang": "en"}
    if target_error_tags:
        filters["error_tags"] = target_error_tags

    try:
        results = await vector_service.search(
            query=query, limit=k, filters=filters, use_hybrid=True
        )
    except Exception as e:
        print(f"Vector search failed: {e}")
        return []

    # Load full items from DB for content (passage/blanks)
    examples = []
    for r in results:
        item_id = r.get("item_id")
        if not item_id:
            continue
        try:
            row = await english_item_repository.get_by_id(session, item_id)
        except Exception:
            row = None
        if not row:
            continue
        examples.append({
            "id": row.id,
            "passage": row.passage,
            "blanks": row.blanks or [],
            "error_tags": row.error_tags or [],
            "topic": row.topic,
            "level_cefr": row.level_cefr,
        })
    return examples

def mine_distractors_from_examples(
    examples: List[Dict[str, Any]],
    skill_tag: Optional[str],
    exclude: Optional[str] = None,
    limit: int = 6
) -> List[str]:
    """
    Mine candidate distractors from example blanks with the same skill_tag.
    Falls back to all blanks if tag match is scarce.
    
    Args:
        examples: List of example items
        skill_tag: Target skill tag to match
        exclude: Answer to exclude from distractors
        limit: Maximum number of distractors to return
        
    Returns:
        List of candidate distractors
    """
    bags = []
    
    # 1) same-skill answers first
    for ex in examples:
        for b in ex.get("blanks", []):
            if skill_tag and str(b.get("skill_tag")) == str(skill_tag) and b.get("answer"):
                bags.append(str(b["answer"]).strip())
    
    # 2) then any blanks if same-skill is scarce
    if len(bags) < 3:
        for ex in examples:
            for b in ex.get("blanks", []):
                if b.get("answer"):
                    bags.append(str(b["answer"]).strip())

    # dedup & filter
    uniq = []
    seen = set([exclude.strip().lower()]) if exclude else set()
    for w in bags:
        lw = w.strip().lower()
        if not w or lw in seen:
            continue
        seen.add(lw)
        uniq.append(w)
        if len(uniq) >= limit:
            break
    
    return uniq

def analyze_example_style(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze example items to extract style patterns.
    
    Args:
        examples: List of example items
        
    Returns:
        Style analysis including common patterns
    """
    if not examples:
        return {}
    
    # Analyze passage length patterns
    lengths = [len(ex.get("passage", "")) for ex in examples]
    avg_length = sum(lengths) / len(lengths) if lengths else 0
    
    # Analyze blank patterns
    blank_counts = [len(ex.get("blanks", [])) for ex in examples]
    avg_blanks = sum(blank_counts) / len(blank_counts) if blank_counts else 0
    
    # Common topics
    topics = [ex.get("topic") for ex in examples if ex.get("topic")]
    common_topics = list(set(topics))
    
    # Error tag distribution
    all_error_tags = []
    for ex in examples:
        all_error_tags.extend(ex.get("error_tags", []))
    error_tag_counts = {}
    for tag in all_error_tags:
        error_tag_counts[tag] = error_tag_counts.get(tag, 0) + 1
    
    return {
        "avg_passage_length": avg_length,
        "avg_blank_count": avg_blanks,
        "common_topics": common_topics,
        "error_tag_distribution": error_tag_counts,
        "example_count": len(examples)
    }
