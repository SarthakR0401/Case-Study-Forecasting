from pydantic import BaseModel, field_validator
from typing import List, Optional

class ForecastRequest(BaseModel):
    state: str

    @field_validator('state')
    @classmethod
    def state_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('state must not be empty')
        return v.title() # Normalize to Title Case (e.g., "california" -> "California")

class ForecastResponse(BaseModel):
    state: str
    model_used: str
    forecast: List[float]
    dates: List[str]
    history: Optional[List[float]] = None
    history_dates: Optional[List[str]] = None
