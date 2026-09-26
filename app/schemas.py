from pydantic import BaseModel
from typing import List

class PredictionItem(BaseModel):
    label: str
    confidence: float

class PredictionResponse(BaseModel):
    predictions: List[PredictionItem]