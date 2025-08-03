from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class SolutionBase(BaseModel):
    question_id: int
    solved_at: Optional[datetime] = None
    is_correct: Optional[bool] = None
    duration: Optional[float] = None


class SolutionCreate(SolutionBase):
    pass


class SolutionUpdate(BaseModel):
    solved_at: Optional[datetime] = None
    is_correct: Optional[bool] = None
    duration: Optional[float] = None


class SolutionInDB(SolutionBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int


class SolutionResponse(SolutionInDB):
    pass
