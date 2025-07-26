from typing import Optional
from pydantic import BaseModel, ConfigDict

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
    model_config = ConfigDict(from_attributes=True)
    
    id: int

class TopicResponse(TopicInDB):
    pass 