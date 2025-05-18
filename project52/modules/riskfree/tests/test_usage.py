import riskfree as rf
from datetime import date

curve = rf.RiskFreeCurve(date.today())
print(curve.spot(5))       # 5‑year zero rate
print(curve.discount(0.5)) # 6‑month DF