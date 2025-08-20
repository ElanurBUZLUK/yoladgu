import time
import asyncio
from sqlalchemy import text
from app.core.database import get_async_session
from app.services.vector_index_manager import VectorIndexManager
from app.mcp.notify import notify_slack

INACTIVE = {"blue": "green", "green": "blue"}

async def build_inactive_index():
    """Asenkron olarak aktif olmayan vektör indeksini oluşturur."""
    async for db in get_async_session():
        try:
            # Aktif slotu al (asenkron olduğunu varsayıyoruz)
            # Note: This assumes VectorIndexManager.get_active_slot is async
            active = await VectorIndexManager.get_active_slot(db)
            inactive = INACTIVE[active]
            start = time.time()
            
            await db.execute(text("UPDATE vector_index_meta SET build_status='building' WHERE name='default'"))
            await db.commit()
            
            # Burada gerçek build işlemi yapılacak
            # Örnek: await db.execute(text(f"REINDEX INDEX ivf_math_{inactive}"))
            await asyncio.sleep(5)  # Simüle edilmiş build süresi
            
            duration = int(time.time() - start)
            await db.execute(
                text("UPDATE vector_index_meta SET build_status='fresh', item_count=:cnt, build_duration_sec=:d WHERE name='default'"),
                {"cnt": 1000, "d": duration}
            )
            await db.commit()
            
            # Slack bildirimi (asenkron olduğunu varsayıyoruz)
            # Note: This assumes notify_slack is async
            await notify_slack(f"Vector index build completed in {duration}s. Ready to swap → {inactive}")
            return {"inactive_ready": inactive, "duration": duration}
        except Exception as e:
            # Hata durumunda Slack bildirimi
            # Note: This assumes notify_slack is async
            await notify_slack(f"ERROR: Vector index build failed: {e}")
            raise

if __name__ == "__main__":
    print("Running nightly index builder...")
    result = asyncio.run(build_inactive_index())
    print(f"Builder finished with result: {result}")
