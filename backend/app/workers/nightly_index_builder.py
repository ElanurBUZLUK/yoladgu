import time
from sqlalchemy import text
from app.core.database import get_db
from app.services.vector_index_manager import VectorIndexManager
from app.mcp.notify import notify_slack

INACTIVE = {"blue": "green", "green": "blue"}

def build_inactive_index():
    db = next(get_db())
    active = VectorIndexManager.get_active_slot(db)
    inactive = INACTIVE[active]
    start = time.time()
    
    db.execute(text("UPDATE vector_index_meta SET build_status='building' WHERE name='default'"))
    db.commit()
    
    # Burada gerçek build işlemi yapılacak
    # Örnek: db.execute(text("REINDEX INDEX ivf_math_{inactive}"))
    time.sleep(5)  # Simüle edilmiş build süresi
    
    duration = int(time.time() - start)
    db.execute(
        text("UPDATE vector_index_meta SET build_status='fresh', item_count=:cnt, build_duration_sec=:d WHERE name='default'"),
        {"cnt": 1000, "d": duration}
    )
    db.commit()
    
    notify_slack(f"Vector index build completed in {duration}s. Ready to swap → {inactive}")
    return {"inactive_ready": inactive, "duration": duration}

if __name__ == "__main__":
    print(build_inactive_index())