# File: tests/test_adjustments.py
import pandas as pd
from adjustments.liquidity import apply_liquidity_adjustment

def test_liquidity():
    df = pd.DataFrame({'yield':[0.01,0.02]})
    out = apply_liquidity_adjustment(df, None)
    assert 'yield' in out.columns
