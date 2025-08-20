from typing import Optional, List, Dict, Any
from uuid import UUID
from pydantic import BaseModel, Field


class MathProfileBase(BaseModel):
    global_skill: Optional[float] = Field(2.5, ge=0.0, le=5.0)
    difficulty_factor: Optional[float] = Field(1.0, ge=0.1, le=1.5)
    ema_accuracy: Optional[float] = Field(0.5, ge=0.0, le=1.0)
    ema_speed: Optional[float] = Field(0.5, ge=0.0, le=1.0)
    streak_right: Optional[int] = Field(0, ge=0)
    streak_wrong: Optional[int] = Field(0, ge=0)
    last_k_outcomes: Optional[List[bool]] = Field(default_factory=list)
    srs_queue: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    bandit_arms: Optional[Dict[str, List[int]]] = Field(default_factory=dict)


class MathProfileCreate(MathProfileBase):
    user_id: UUID


class MathProfileUpdate(MathProfileBase):
    pass


class MathProfileInDB(MathProfileBase):
    id: UUID
    user_id: UUID
    created_at: str  # Using str for datetime for simplicity in Pydantic
    updated_at: str  # Using str for datetime for simplicity in Pydantic

    class Config:
        from_attributes = True
