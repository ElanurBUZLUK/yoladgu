from pydantic import BaseModel


class ServeRequest(BaseModel):
    student_id: int
    k: int = 1


class ServeResponseItem(BaseModel):
    question_id: int
    difficulty_level: int
    t_ref_ms: int


class ServeResponse(BaseModel):
    items: list[ServeResponseItem]


class SubmitRequest(BaseModel):
    student_id: int
    question_id: int
    is_correct: bool
    time_ms: int


class SubmitResponse(BaseModel):
    new_skill: float
    new_question_diff: float


