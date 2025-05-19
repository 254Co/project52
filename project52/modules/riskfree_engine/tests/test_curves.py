# File: tests/test_curves.py
import pytest
from curves.interpolation import interpolate_curve
import pandas as pd

def test_interpolate():
    df = pd.DataFrame({'maturity':[1,2,5], 'yield':[0.01,0.015,0.02]})
    spline = interpolate_curve(df)
    assert callable(spline)
