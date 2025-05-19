# File: api/models.py
from pydantic import BaseModel
from typing import List

class YieldPoint(BaseModel):
    maturity: float
    yield_rate: float

class CurveResponse(BaseModel):
    date: str
    points: List[YieldPoint]

