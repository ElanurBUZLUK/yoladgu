from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, update, func
from app.models import User, Question, Attempt


async def get_student(db: AsyncSession, student_id: int) -> User | None:
    res = await db.execute(select(User).where(User.id == student_id))
    return res.scalar_one_or_none()


async def get_question(db: AsyncSession, question_id: int) -> Question | None:
    res = await db.execute(select(Question).where(Question.id == question_id))
    return res.scalar_one_or_none()


async def update_student(db: AsyncSession, student: User) -> None:
    await db.execute(
        update(User)
        .where(User.id == student.id)
        .values(skill_rating=student.skill_rating, skill_rating_var=student.skill_rating_var)
    )
    await db.commit()


async def update_question(db: AsyncSession, question: Question) -> None:
    await db.execute(
        update(Question)
        .where(Question.id == question.id)
        .values(
            difficulty_rating=question.difficulty_rating,
            difficulty_level=question.difficulty_level,
            t_ref_ms=question.t_ref_ms,
            last_recalibrated_at=question.last_recalibrated_at,
        )
    )
    await db.commit()


async def insert_attempt(
    db: AsyncSession,
    *,
    student_id: int,
    question_id: int,
    is_correct: bool,
    time_ms: int,
    reward_time_adj: float,
    expected: float,
    delta: float,
) -> None:
    await db.execute(
        insert(Attempt).values(
            student_id=student_id,
            question_id=question_id,
            is_correct=is_correct,
            time_ms=time_ms,
            reward_time_adj=reward_time_adj,
            expected=expected,
            delta=delta,
        )
    )
    await db.commit()


async def find_questions_by_diff_near(db: AsyncSession, rs: float, *, tol: float, limit: int) -> List[Question]:
    res = await db.execute(
        select(Question)
        .order_by(func.abs(Question.difficulty_rating - rs))
        .limit(limit)
    )
    return list(res.scalars().all())


async def random_questions(db: AsyncSession, *, limit: int) -> List[Question]:
    # Simple: order by id mod random surrogate (not perfect random)
    res = await db.execute(select(Question).order_by(Question.id.desc()).limit(limit))
    return list(res.scalars().all())


async def questions_with_min_attempts(db: AsyncSession, *, min_attempts: int, since_days: int) -> List[Question]:
    # Placeholder: return all questions; refine with attempts aggregation if needed
    res = await db.execute(select(Question))
    return list(res.scalars().all())


