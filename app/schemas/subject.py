from typing import Optional
from pydantic import BaseModel

class SubjectBase(BaseModel):
    name: str
    description: Optional[str] = None

class SubjectCreate(SubjectBase):
    pass

class SubjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None

class SubjectInDB(SubjectBase):
    id: int
    class Config:
        orm_mode = True

class SubjectResponse(SubjectInDB):
    pass 