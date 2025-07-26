from typing import Optional
from pydantic import BaseModel, ConfigDict

class SubjectBase(BaseModel):
    name: str
    description: Optional[str] = None

class SubjectCreate(SubjectBase):
    pass

class SubjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None

class SubjectInDB(SubjectBase):
    model_config = ConfigDict(from_attributes=True)
    
    id: int

class SubjectResponse(SubjectInDB):
    pass 