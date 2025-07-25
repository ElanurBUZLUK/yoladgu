from typing import Optional
from pydantic import BaseModel

class TopicBase(BaseModel):
    subject_id: int
    name: str
    description: Optional[str] = None

class TopicCreate(TopicBase):
    pass

class TopicUpdate(BaseModel):
    subject_id: Optional[int] = None
    name: Optional[str] = None
    description: Optional[str] = None

class TopicInDB(TopicBase):
    id: int
    class Config:
        orm_mode = True

class TopicResponse(TopicInDB):
    pass 