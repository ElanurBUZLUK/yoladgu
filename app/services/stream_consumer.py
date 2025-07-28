import time
import redis
from app.core.config import settings
from app.db.database import SessionLocal
from app.services.recommendation_service import RecommendationService
import structlog

PROCESSED_EVENT_IDS = set()  # Basit idempotency için bellek içi set (prod için Redis veya DB önerilir)

MAX_RETRIES = 3
DLQ_STREAM = "student_responses_dlq"

def consume_loop():
    r = redis.Redis.from_url(settings.redis_url)
    rec_service = RecommendationService()
    last_id = "0-0"
    while True:
        entries = r.xread({"student_responses_stream": last_id}, block=5000, count=10)
        if not entries:
            continue
        for _, events in entries:
            for eid, data in events:
                last_id = eid
                # Idempotency: Aynı event tekrar işlenmesin
                if eid in PROCESSED_EVENT_IDS:
                    continue
                try:
                    payload = {k.decode(): v.decode() for k,v in data.items()}
                    db = SessionLocal()
                    for attempt in range(MAX_RETRIES):
                        try:
                            rec_service.process_student_response(
                                db,
                                int(payload["student_id"]),
                                int(payload["question_id"]),
                                None,
                                payload["is_correct"] == "True",
                                float(payload["response_time"]),
                                payload.get("feedback"),
                                payload.get("request_id")
                            )
                            PROCESSED_EVENT_IDS.add(eid)
                            break
                        except Exception as e:
                            structlog.get_logger().warning("consumer_retry", event_id=eid, attempt=attempt, error=str(e))
                            time.sleep(0.1)
                    else:
                        # Retry başarısız, DLQ'ya yaz
                        r.xadd(DLQ_STREAM, data)
                        structlog.get_logger().error("consumer_dlq", event_id=eid, data=payload)
                except Exception as e:
                    # Malformed event, DLQ'ya yaz
                    r.xadd(DLQ_STREAM, data)
                    structlog.get_logger().error("consumer_dlq_malformed", event_id=eid, error=str(e), raw=data)
                finally:
                    db.close()
        time.sleep(0.01)

if __name__ == "__main__":
    consume_loop() 