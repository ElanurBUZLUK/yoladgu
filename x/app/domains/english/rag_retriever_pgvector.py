from sqlalchemy import text
from app.services.vector_index_manager import vector_index_manager
import asyncio

async def search(db, namespace: str, query_vec: list[float], k: int = 5):
    """Search embeddings table with vector similarity"""
    try:
        # Get active slot for namespace
        active_slot = await vector_index_manager._get_active_slot(namespace)
        
        sql = """
          SELECT obj_ref, meta, embedding <=> $1::vector AS dist
          FROM embeddings
          WHERE namespace = $2 AND slot = $3 AND is_active = true
          ORDER BY dist
          LIMIT $4
        """
        
        rows = await db.execute(
            text(sql),
            [query_vec, namespace, active_slot, k]
        )
        
        results = []
        async for row in rows:
            results.append({
                "obj_ref": row.obj_ref,
                "score": float(row.dist),
                "meta": row.meta
            })
        
        return results
        
    except Exception as e:
        print(f"Error in vector search: {e}")
        return []

# Note: Pass list[float] for query_vec. Use bindparam("q", type_=Vector(1536)) if needed.