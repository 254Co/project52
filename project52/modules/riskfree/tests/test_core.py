"""Tests for the core RiskFreeCurve class."""
import pytest
from datetime import date, timedelta
import pandas as pd
import numpy as np
from riskfree import RiskFreeCurve

@pytest.fixture
def sample_curve():
    """Create a sample curve for testing."""
    return RiskFreeCurve(date.today())

def test_curve_initialization():
    """Test curve initialization with different parameters."""
    # Test with today's date
    curve = RiskFreeCurve(date.today())
    assert isinstance(curve.zeros, dict)
    assert len(curve.zeros) > 0
    
    # Test with smooth=False
    curve_raw = RiskFreeCurve(date.today(), smooth=False)
    assert isinstance(curve_raw.zeros, dict)
    assert len(curve_raw.zeros) > 0
    
    # Test with historical date
    past_date = date.today() - timedelta(days=30)
    curve_hist = RiskFreeCurve(past_date)
    assert isinstance(curve_hist.zeros, dict)
    assert len(curve_hist.zeros) > 0

def test_spot_rate_calculation(sample_curve):
    """Test spot rate calculations."""
    # Test exact tenor
    for tenor in sample_curve.zeros:
        rate = sample_curve.spot(tenor)
        assert isinstance(rate, float)
        assert rate >= 0
        assert rate == sample_curve.zeros[tenor]
    
    # Test interpolation
    tenors = sorted(sample_curve.zeros.keys())
    if len(tenors) >= 2:
        mid_tenor = (tenors[0] + tenors[1]) / 2
        rate = sample_curve.spot(mid_tenor)
        assert isinstance(rate, float)
        assert rate >= 0
        assert min(sample_curve.zeros[tenors[0]], sample_curve.zeros[tenors[1]]) <= rate <= max(sample_curve.zeros[tenors[0]], sample_curve.zeros[tenors[1]])
    
    # Test out of range
    with pytest.raises(ValueError):
        sample_curve.spot(max(sample_curve.zeros.keys()) + 1)

def test_discount_factor_calculation(sample_curve):
    """Test discount factor calculations."""
    # Test exact tenor
    for tenor in sample_curve.zeros:
        df = sample_curve.discount(tenor)
        assert isinstance(df, float)
        assert 0 < df <= 1
        expected_df = np.exp(-sample_curve.zeros[tenor] * tenor)
        assert abs(df - expected_df) < 1e-10
    
    # Test interpolation
    tenors = sorted(sample_curve.zeros.keys())
    if len(tenors) >= 2:
        mid_tenor = (tenors[0] + tenors[1]) / 2
        df = sample_curve.discount(mid_tenor)
        assert isinstance(df, float)
        assert 0 < df <= 1

def test_dataframe_conversion(sample_curve):
    """Test conversion to DataFrame."""
    df = sample_curve.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {'zero', 'discount'}
    assert df.index.name == 'tenor'
    assert len(df) == len(sample_curve.zeros)
    
    # Check values
    for tenor in sample_curve.zeros:
        assert abs(df.loc[tenor, 'zero'] - sample_curve.zeros[tenor]) < 1e-10
        assert abs(df.loc[tenor, 'discount'] - sample_curve.discount(tenor)) < 1e-10

def test_curve_monotonicity(sample_curve):
    """Test that the curve maintains reasonable properties."""
    df = sample_curve.to_dataframe()
    
    # Check that rates are non-negative
    assert (df['zero'] >= 0).all()
    
    # Check that discount factors are between 0 and 1
    assert (df['discount'] > 0).all()
    assert (df['discount'] <= 1).all()
    
    # Check that discount factors are monotonically decreasing
    assert df['discount'].is_monotonic_decreasing

def test_curve_smoothing():
    """Test the effect of smoothing on the curve."""
    # Create both smoothed and unsmoothed curves
    curve_smooth = RiskFreeCurve(date.today(), smooth=True)
    curve_raw = RiskFreeCurve(date.today(), smooth=False)
    
    # Compare the curves
    df_smooth = curve_smooth.to_dataframe()
    df_raw = curve_raw.to_dataframe()
    
    # Check that smoothed curve has same number of points
    assert len(df_smooth) == len(df_raw)
    
    # Check that smoothed rates are within reasonable bounds of raw rates
    max_diff = abs(df_smooth['zero'] - df_raw['zero']).max()
    assert max_diff < 0.1  # Maximum 10% difference between smoothed and raw rates