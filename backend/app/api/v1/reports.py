from fastapi import APIRouter, Depends
from sqlalchemy import text
from app.core.database import get_db
from app.core.security import verify_token

router = APIRouter(prefix="/api/v1/reports", tags=["reports"]) 

@router.get("/costs/daily")
def daily_costs(user_id: str = Depends(verify_token)):
    db = next(get_db())
    rows = db.execute(text(
        """
        SELECT date_trunc('day', created_at) AS day,
               SUM(cost_usd)::float AS cost
        FROM token_usage
        WHERE user_id = :u
        GROUP BY 1
        ORDER BY 1 DESC
        LIMIT 30
        """
    ), {"u": user_id}).fetchall()
    return [{"day": str(r[0]), "cost": r[1]} for r in rows]