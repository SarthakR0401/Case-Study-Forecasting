from pydantic import BaseModel, field_validator
from typing import List, Optional

class ForecastRequest(BaseModel):
    state: str

    @field_validator('state')
    @classmethod
    def state_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('state must not be empty')
        # Normalize to Title Case (e.g., "california" -> "California")
        return v.strip().title()

class ForecastResponse(BaseModel):
    state: str
    model_used: str
    forecast: List[float]
    dates: List[str]
    lower: Optional[List[float]] = None # Confidence intervals
    upper: Optional[List[float]] = None
    history: Optional[List[float]] = None
    history_dates: Optional[List[str]] = None
