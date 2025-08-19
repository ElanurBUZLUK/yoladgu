from typing import Dict, Any, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from datetime import datetime
import logging

try:
    from app.models.user import User, UserRole
except Exception:
    User = None  # type: ignore
    UserRole = None  # type: ignore

try:
    from app.models.question import Question, Subject, QuestionType
except Exception:
    Question = None  # type: ignore
    Subject = None  # type: ignore
    QuestionType = None  # type: ignore

logger = logging.getLogger(__name__)

class SampleDataService:
    """
    Örnek veri oluşturma.
    Varsayılan: apply=False (dry-run). apply=True ile DB'ye yazar; hata olursa loglar ve devam eder.
    """

    async def create_sample_data(self, db: AsyncSession, apply: bool = False) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "apply": apply,
            "created": {"users": 0, "questions": 0},
            "skipped": {"users": 0, "questions": 0},
            "errors": [],
            "timestamp": None,
        }

        # Users
        wanted_users: List[Dict[str, Any]] = [
            {"username": "demo_teacher", "role": "teacher"},
            {"username": "demo_student", "role": "student"},
        ]
        if User and UserRole:
            for u in wanted_users:
                try:
                    res = await db.execute(
                        select(func.count()).select_from(User).where(getattr(User, "username") == u["username"])
                    )
                    cnt = (res.scalar() or 0)
                    if cnt > 0:
                        report["skipped"]["users"] += 1
                        continue
                    if apply:
                        role_val = getattr(UserRole, u["role"].upper(), None) or u["role"]
                        db.add(User(username=u["username"], role=role_val))
                        report["created"]["users"] += 1
                except Exception as e:
                    logger.exception("sample user create failed: %s", e)
                    report["errors"].append(f"user:{u['username']} -> {e}")

        # Questions
        wanted_questions: List[Dict[str, Any]] = [
            {
                "subject": "math",
                "question_type": "multiple_choice",
                "content": "What is the value of 2 + 2?",
                "options": ["3", "4", "5", "6"],
                "correct_answer": "4",
                "topic_category": "arithmetic",
                "difficulty_level": 1,
            },
            {
                "subject": "english",
                "question_type": "open_ended",
                "content": "Write a sentence using the word 'harmony'.",
                "correct_answer": None,
                "topic_category": "vocabulary",
                "difficulty_level": 2,
            },
        ]
        if Question:
            for q in wanted_questions:
                try:
                    res = await db.execute(
                        select(func.count()).select_from(Question).where(getattr(Question, "content") == q["content"])
                    )
                    cnt = (res.scalar() or 0)
                    if cnt > 0:
                        report["skipped"]["questions"] += 1
                        continue
                    if apply:
                        subj = getattr(Subject, "MATH", None) if q["subject"] == "math" else getattr(Subject, "ENGLISH", None) or q["subject"]
                        qtype = getattr(QuestionType, "MULTIPLE_CHOICE", None) if q["question_type"] == "multiple_choice" else getattr(QuestionType, "OPEN_ENDED", None) or q["question_type"]
                        db.add(Question(
                            subject=subj,
                            question_type=qtype,
                            content=q["content"],
                            correct_answer=q["correct_answer"],
                            options=q.get("options"),
                            topic_category=q.get("topic_category", "general"),
                            difficulty_level=q.get("difficulty_level", 1),
                            original_difficulty=q.get("difficulty_level", 1),
                        ))
                        report["created"]["questions"] += 1
                except Exception as e:
                    logger.exception("sample question create failed: %s", e)
                    report["errors"].append(f"question:{q['content'][:40]} -> {e}")

        if apply:
            try:
                await db.commit()
            except Exception as e:
                await db.rollback()
                report["errors"].append(f"commit -> {e}")

        report["timestamp"] = datetime.utcnow().isoformat() + "Z"
        return report

sample_data_service = SampleDataService()
