from sqlalchemy import text
from sqlalchemy.orm import Session
from app.core.database import get_async_session
import logging
import datetime

logger = logging.getLogger(__name__)

class Scheduler:
    def __init__(self, db: Session):
        self.db = db
        self.lock_id = 884422  # Advisory lock ID

    def _acquire_lock(self) -> bool:
        result = self.db.execute(
            text("SELECT pg_try_advisory_lock(:lock_id)"),
            {"lock_id": self.lock_id}
        ).scalar()
        return result

    def _release_lock(self):
        self.db.execute(
            text("SELECT pg_advisory_unlock(:lock_id)"),
            {"lock_id": self.lock_id}
        )

    def update_spaced_repetition_due(self):
        if not self._acquire_lock():
            logger.warning("Could not acquire lock for spaced repetition update")
            return

        try:
            # Update due dates and calculate new intervals
            self.db.execute(text("""
                UPDATE student_vocab_profile
                SET
                    sm2_interval = CASE
                        WHEN correct >= 3 THEN LEAST(sm2_interval * sm2_ef, 365)
                        ELSE GREATEST(sm2_interval * 0.5, 1)
                    END,
                    sm2_due = CURRENT_DATE + sm2_interval
                WHERE sm2_due <= CURRENT_DATE
            """))
            self.db.commit()
            logger.info("Updated spaced repetition due dates and intervals")
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to update spaced repetition: {e}")
        finally:
            self._release_lock()

    def refresh_peer_hardness(self):
        if not self._acquire_lock():
            logger.warning("Could not acquire lock for peer hardness refresh")
            return

        try:
            # Calculate co-error statistics
            self.db.execute(text("""
                INSERT INTO peer_hardness (question_id, co_error_score, last_updated)
                SELECT
                    a.question_id,
                    COUNT(DISTINCT CASE WHEN b.is_correct = FALSE THEN b.student_id END)::FLOAT /
                    NULLIF(COUNT(DISTINCT b.student_id), 0) AS co_error_score,
                    NOW() AS last_updated
                FROM attempts a
                JOIN attempts b ON a.student_id = b.student_id
                    AND a.attempt_id != b.attempt_id
                    AND a.created_at BETWEEN b.created_at - INTERVAL '1 week'
                    AND b.created_at + INTERVAL '1 week'
                WHERE a.is_correct = FALSE
                GROUP BY a.question_id
                ON CONFLICT (question_id) DO UPDATE
                SET
                    co_error_score = EXCLUDED.co_error_score,
                    last_updated = EXCLUDED.last_updated
            """))
            self.db.commit()
            logger.info("Refreshed peer hardness statistics")
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to refresh peer hardness: {e}")
        finally:
            self._release_lock()

    def conditionally_swap_index(self):
        if not self._acquire_lock():
            logger.warning("Could not acquire lock for index swap")
            return

        try:
            # Check if inactive slot is fresh enough to swap
            result = self.db.execute(text("""
                SELECT slot, item_count, created_at
                FROM vector_index_meta
                WHERE status = 'fresh'
                ORDER BY created_at DESC
                LIMIT 1
            """)).fetchone()

            if result and result.item_count > 0:
                slot = result.slot
                self.db.execute(text("""
                    UPDATE vector_index_meta
                    SET active_slot = :slot
                    WHERE name = 'default'
                """), {"slot": slot})
                self.db.commit()
                logger.info(f"Swapped active index to {slot} slot")
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to swap index: {e}")
        finally:
            self._release_lock()

    def run_scheduled_tasks(self):
        logger.info("Starting scheduled tasks")
        self.update_spaced_repetition_due()
        self.refresh_peer_hardness()
        self.conditionally_swap_index()
        logger.info("Completed scheduled tasks")

