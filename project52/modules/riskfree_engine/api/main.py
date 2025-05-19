# File: api/main.py
from fastapi import FastAPI, Query

from .models import CurveResponse, YieldPoint
from ..curves.builder import RiskFreeCurveBuilder
import yaml

app = FastAPI(title="Risk‑Free Curve API")

with open(Path(__file__).parents[1] / "config" / "settings.yaml") as f:
    CONFIG = yaml.safe_load(f)

BUILDER = RiskFreeCurveBuilder(CONFIG)

DEFAULT_MATS = [1/12, 0.5, 1, 2, 5, 10, 20, 30]

@app.get("/curve", response_model=CurveResponse)
def get_curve(date: str = Query(..., description="YYYY‑MM‑DD")):
    result = BUILDER.build_curve(date)
    spline = result["spline"]
    spots = compute_spot_rates(spline, DEFAULT_MATS)
    points = [YieldPoint(maturity=m, yield_rate=r) for m, r in spots.items()]
    return CurveResponse(date=result["valuation_DATE"], points=points)
