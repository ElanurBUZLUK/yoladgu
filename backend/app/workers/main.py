from app.workers.scheduler import Scheduler
from app.core.database import get_async_session
import schedule
import time
import logging

logger = logging.getLogger(__name__)

def run_scheduler():
    db = next(get_async_session())
    scheduler = Scheduler(db)
    
    # Schedule daily tasks
    schedule.every().day.at("02:00").do(scheduler.run_scheduled_tasks)
    
    logger.info("Scheduler worker started")
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    run_scheduler()