import math
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.adaptive import repo


class RatingService:
    def __init__(self, cfg):
        self.beta = cfg.RATING_BETA
        self.k_s = cfg.K_STUDENT
        self.k_q = cfg.K_QUESTION
        self.alpha = cfg.ALPHA_TIME
        self.eta_t = cfg.EMA_TREF_ETA

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def expected(self, rs: float, rq: float) -> float:
        return self._sigmoid(self.beta * (rs - rq))

    def speed_factor(self, t: int, t_ref: int) -> float:
        if t <= t_ref:
            return 1.0
        over = (t - t_ref) / max(1, t_ref)
        return math.exp(-self.alpha * over)

    def reward_time_adjusted(self, is_correct: bool, t: int, t_ref: int) -> float:
        if not is_correct:
            return 0.0
        sf = self.speed_factor(t, t_ref)
        return max(0.5, min(1.0, 0.5 + 0.5 * sf))

    async def update_after_attempt(self, db: AsyncSession, student, question, is_correct: bool, time_ms: int):
        y_star = self.reward_time_adjusted(is_correct, time_ms, question.t_ref_ms)
        y_hat = self.expected(student.skill_rating, question.difficulty_rating)
        e = y_star - y_hat

        student.skill_rating = float(student.skill_rating + self.k_s * e)
        question.difficulty_rating = float(question.difficulty_rating - self.k_q * e)

        if is_correct and y_star >= 0.7:
            question.t_ref_ms = int((1 - self.eta_t) * question.t_ref_ms + self.eta_t * time_ms)

        await repo.insert_attempt(
            db,
            student_id=student.id,
            question_id=question.id,
            is_correct=is_correct,
            time_ms=time_ms,
            reward_time_adj=y_star,
            expected=y_hat,
            delta=e,
        )

        await repo.update_student(db, student)
        await repo.update_question(db, question)

        return student.skill_rating, question.difficulty_rating


