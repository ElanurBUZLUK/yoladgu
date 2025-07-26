from typing import Optional
from pydantic import BaseModel, ConfigDict

class PlanItemBase(BaseModel):
    plan_id: int
    subject_id: int
    topic_id: int
    question_count: int
    difficulty: str
    estimated_time: int

class PlanItemCreate(PlanItemBase):
    pass

class PlanItemUpdate(BaseModel):
    plan_id: Optional[int] = None
    subject_id: Optional[int] = None
    topic_id: Optional[int] = None
    question_count: Optional[int] = None
    difficulty: Optional[str] = None
    estimated_time: Optional[int] = None

class PlanItemInDB(PlanItemBase):
    model_config = ConfigDict(from_attributes=True)
    
    id: int

class PlanItemResponse(PlanItemInDB):
    pass 