"""
Multi-Database Transaction Manager
Handles transaction boundaries across PostgreSQL, Neo4j, and Redis
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import structlog
from app.services.neo4j_service import neo4j_service
from app.services.redis_service import redis_service
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

logger = structlog.get_logger()


class TransactionStatus(Enum):
    PENDING = "pending"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    COMPENSATED = "compensated"


@dataclass
class TransactionOperation:
    """Represents a single operation in a distributed transaction"""

    name: str
    operation: Callable
    compensate: Optional[Callable] = None
    data: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[str] = None


class TransactionManager:
    """
    Manages distributed transactions across multiple databases
    Uses Saga pattern for compensation-based rollback
    """

    def __init__(self):
        self.operations: List[TransactionOperation] = []
        self.status = TransactionStatus.PENDING
        self.transaction_id: Optional[str] = None

    def add_operation(
        self,
        name: str,
        operation: Callable,
        compensate: Optional[Callable] = None,
        data: Optional[Dict[str, Any]] = None,
    ):
        """Add an operation to the transaction"""
        self.operations.append(
            TransactionOperation(
                name=name, operation=operation, compensate=compensate, data=data
            )
        )

    def execute(self) -> bool:
        """Execute all operations in order"""
        import uuid

        self.transaction_id = str(uuid.uuid4())

        logger.info(
            "transaction_started",
            transaction_id=self.transaction_id,
            operation_count=len(self.operations),
        )

        executed_operations = []

        try:
            for i, operation in enumerate(self.operations):
                logger.debug(
                    "executing_operation",
                    transaction_id=self.transaction_id,
                    operation_name=operation.name,
                    step=i + 1,
                )

                try:
                    if operation.data:
                        result = operation.operation(**operation.data)
                    else:
                        result = operation.operation()

                    operation.result = result
                    executed_operations.append(operation)

                    logger.debug(
                        "operation_completed",
                        transaction_id=self.transaction_id,
                        operation_name=operation.name,
                    )

                except Exception as e:
                    operation.error = str(e)
                    logger.error(
                        "operation_failed",
                        transaction_id=self.transaction_id,
                        operation_name=operation.name,
                        error=str(e),
                    )

                    # Compensate previous operations
                    self._compensate(executed_operations)
                    self.status = TransactionStatus.COMPENSATED
                    return False

            self.status = TransactionStatus.COMMITTED
            logger.info("transaction_committed", transaction_id=self.transaction_id)
            return True

        except Exception as e:
            logger.error(
                "transaction_error", transaction_id=self.transaction_id, error=str(e)
            )
            self._compensate(executed_operations)
            self.status = TransactionStatus.ROLLED_BACK
            return False

    def _compensate(self, executed_operations: List[TransactionOperation]):
        """Execute compensation operations in reverse order"""
        logger.info(
            "starting_compensation",
            transaction_id=self.transaction_id,
            operations_to_compensate=len(executed_operations),
        )

        for operation in reversed(executed_operations):
            if operation.compensate:
                try:
                    if operation.data:
                        operation.compensate(**operation.data)
                    else:
                        operation.compensate()

                    logger.debug(
                        "compensation_completed",
                        transaction_id=self.transaction_id,
                        operation_name=operation.name,
                    )
                except Exception as e:
                    logger.error(
                        "compensation_failed",
                        transaction_id=self.transaction_id,
                        operation_name=operation.name,
                        error=str(e),
                    )


@contextmanager
def distributed_transaction(db: Session):
    """
    Context manager for distributed transactions

    Usage:
        with distributed_transaction(db) as tx:
            tx.add_postgres_operation(lambda: create_user(db, user_data))
            tx.add_neo4j_operation(lambda: neo4j_service.record_student_solution(...))
            tx.add_redis_operation(lambda: redis_service.cache_set(...))

            if not tx.execute():
                raise TransactionError("Transaction failed")
    """
    tx = TransactionManager()

    try:
        # Start PostgreSQL transaction
        db.begin()
        yield tx

        # If all operations successful, commit PostgreSQL
        if tx.status == TransactionStatus.COMMITTED:
            db.commit()
            logger.info(
                "postgres_transaction_committed", transaction_id=tx.transaction_id
            )
        else:
            db.rollback()
            logger.info(
                "postgres_transaction_rolled_back", transaction_id=tx.transaction_id
            )

    except SQLAlchemyError as e:
        db.rollback()
        logger.error(
            "postgres_transaction_error", transaction_id=tx.transaction_id, error=str(e)
        )
        raise
    except Exception as e:
        db.rollback()
        logger.error(
            "distributed_transaction_error",
            transaction_id=tx.transaction_id,
            error=str(e),
        )
        raise


class StudentResponseTransaction:
    """
    Specialized transaction for student response handling
    Coordinates PostgreSQL, Neo4j, and Redis operations
    """

    @staticmethod
    def execute_student_response(
        db: Session,
        student_id: int,
        question_id: int,
        answer: str,
        is_correct: bool,
        response_time: Optional[float] = None,
        confidence_level: Optional[int] = None,
        feedback: Optional[str] = None,
    ) -> bool:
        """Execute student response across all databases"""

        try:
            with distributed_transaction(db) as tx:
                # 1. PostgreSQL operation
                def create_pg_response():
                    from app.crud.student_response import create_response

                    return create_response(
                        db=db,
                        student_id=student_id,
                        question_id=question_id,
                        answer=answer,
                        is_correct=is_correct,
                        response_time=response_time,
                        confidence_level=confidence_level,
                        feedback=feedback,
                    )

                def compensate_pg_response():
                    # PostgreSQL rollback is handled by the transaction context
                    pass

                tx.add_operation(
                    name="postgres_student_response",
                    operation=create_pg_response,
                    compensate=compensate_pg_response,
                )

                # 2. Neo4j operation
                def create_neo4j_relationship():
                    neo4j_service.record_student_solution(
                        student_id=student_id,
                        question_id=question_id,
                        correct=is_correct,
                    )

                def compensate_neo4j_relationship():
                    # Try to remove the relationship
                    if neo4j_service._driver:
                        try:
                            with neo4j_service._driver.session() as session:
                                session.run(
                                    """
                                    MATCH (u:User {id: $uid})-[r:SOLVED]->(q:Question {id: $qid})
                                    DELETE r
                                    """,
                                    uid=student_id,
                                    qid=question_id,
                                )
                        except Exception as e:
                            logger.error("neo4j_compensation_failed", error=str(e))

                tx.add_operation(
                    name="neo4j_student_solution",
                    operation=create_neo4j_relationship,
                    compensate=compensate_neo4j_relationship,
                )

                # 3. Redis caching operation
                def cache_response():
                    cache_key = f"student_response:{student_id}:{question_id}"
                    cache_data = {
                        "answer": answer,
                        "is_correct": is_correct,
                        "response_time": response_time,
                        "timestamp": time.time(),
                    }
                    redis_service.cache_set(cache_key, cache_data, ttl=3600)

                def compensate_cache():
                    cache_key = f"student_response:{student_id}:{question_id}"
                    redis_service.cache_delete(cache_key)

                tx.add_operation(
                    name="redis_cache_response",
                    operation=cache_response,
                    compensate=compensate_cache,
                )

                # 4. Stream notification (optional, non-critical)
                def stream_notification():
                    stream_data = {
                        "event_type": "student_response",
                        "student_id": student_id,
                        "question_id": question_id,
                        "is_correct": is_correct,
                        "timestamp": time.time(),
                    }
                    redis_service.stream_add("student_responses_stream", stream_data)

                tx.add_operation(
                    name="stream_notification",
                    operation=stream_notification,
                    compensate=None,  # Stream messages don't need compensation
                )

                # Execute transaction
                return tx.execute()

        except Exception as e:
            logger.error(
                "student_response_transaction_failed",
                student_id=student_id,
                question_id=question_id,
                error=str(e),
            )
            return False


# Convenience functions
def execute_with_compensation(db: Session, operations: List[Dict[str, Any]]) -> bool:
    """
    Execute multiple operations with automatic compensation

    Args:
        db: SQLAlchemy session
        operations: List of operation dictionaries with 'name', 'operation', and optional 'compensate'
    """
    with distributed_transaction(db) as tx:
        for op in operations:
            tx.add_operation(**op)
        return tx.execute()
