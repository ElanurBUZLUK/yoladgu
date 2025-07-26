from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict

class StudyPlanBase(BaseModel):
    name: str
    created_at: Optional[datetime] = None
    target_success: Optional[float] = None

class StudyPlanCreate(StudyPlanBase):
    pass

class StudyPlanUpdate(BaseModel):
    name: Optional[str] = None
    created_at: Optional[datetime] = None
    target_success: Optional[float] = None

class StudyPlanInDB(StudyPlanBase):
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    user_id: int

class StudyPlanResponse(StudyPlanInDB):
    pass 