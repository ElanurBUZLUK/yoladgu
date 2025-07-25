from typing import Optional
from pydantic import BaseModel

class PlanItemBase(BaseModel):
    plan_id: int
    subject_id: int
    topic_id: int
    question_count: Optional[int] = None
    difficulty: Optional[str] = None
    estimated_time: Optional[int] = None

class PlanItemCreate(PlanItemBase):
    pass

class PlanItemUpdate(BaseModel):
    subject_id: Optional[int] = None
    topic_id: Optional[int] = None
    question_count: Optional[int] = None
    difficulty: Optional[str] = None
    estimated_time: Optional[int] = None

class PlanItemInDB(PlanItemBase):
    id: int
    class Config:
        orm_mode = True

class PlanItemResponse(PlanItemInDB):
    pass 