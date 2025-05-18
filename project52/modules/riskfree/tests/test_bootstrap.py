"""Tests for the bootstrapping functionality."""
import pytest
import numpy as np
from riskfree.curve.bootstrap import bootstrap_par_yields

def test_bootstrap_basic():
    """Test basic bootstrapping functionality."""
    # Create sample par yields
    par_yields = {
        1.0: 0.05,  # 5% 1-year rate
        2.0: 0.06,  # 6% 2-year rate
        3.0: 0.07,  # 7% 3-year rate
    }
    
    # Bootstrap zero-coupon yields
    zeros = bootstrap_par_yields(par_yields)
    
    # Check basic properties
    assert isinstance(zeros, dict)
    assert len(zeros) == len(par_yields)
    assert set(zeros.keys()) == set(par_yields.keys())
    
    # Check that zero rates are non-negative
    assert all(r >= 0 for r in zeros.values())
    
    # Check that zero rates are close to par rates for short tenors
    assert abs(zeros[1.0] - par_yields[1.0]) < 0.001

def test_bootstrap_monotonicity():
    """Test that bootstrapped rates maintain reasonable properties."""
    # Create a more extensive set of par yields
    par_yields = {
        0.5: 0.04,  # 4% 6-month rate
        1.0: 0.05,  # 5% 1-year rate
        2.0: 0.06,  # 6% 2-year rate
        3.0: 0.07,  # 7% 3-year rate
        5.0: 0.08,  # 8% 5-year rate
        7.0: 0.09,  # 9% 7-year rate
        10.0: 0.10,  # 10% 10-year rate
    }
    
    # Bootstrap zero-coupon yields
    zeros = bootstrap_par_yields(par_yields)
    
    # Check that rates are monotonically increasing
    tenors = sorted(zeros.keys())
    for i in range(1, len(tenors)):
        assert zeros[tenors[i]] >= zeros[tenors[i-1]]

def test_bootstrap_arbitrage_free():
    """Test that bootstrapped rates are arbitrage-free."""
    # Create par yields with a hump
    par_yields = {
        1.0: 0.05,  # 5% 1-year rate
        2.0: 0.07,  # 7% 2-year rate
        3.0: 0.06,  # 6% 3-year rate (hump)
        5.0: 0.08,  # 8% 5-year rate
    }
    
    # Bootstrap zero-coupon yields
    zeros = bootstrap_par_yields(par_yields)
    
    # Check that zero rates are arbitrage-free
    # This means that the forward rates should be non-negative
    tenors = sorted(zeros.keys())
    for i in range(1, len(tenors)):
        t1, t2 = tenors[i-1], tenors[i]
        r1, r2 = zeros[t1], zeros[t2]
        # Calculate forward rate
        forward_rate = (r2 * t2 - r1 * t1) / (t2 - t1)
        assert forward_rate >= 0

def test_bootstrap_edge_cases():
    """Test bootstrapping with edge cases."""
    # Test with empty input
    with pytest.raises(ValueError):
        bootstrap_par_yields({})
    
    # Test with single point
    with pytest.raises(ValueError):
        bootstrap_par_yields({1.0: 0.05})
    
    # Test with negative rates
    with pytest.raises(ValueError):
        bootstrap_par_yields({1.0: -0.05, 2.0: 0.06})
    
    # Test with zero rates
    with pytest.raises(ValueError):
        bootstrap_par_yields({1.0: 0.0, 2.0: 0.06})

def test_bootstrap_accuracy():
    """Test the accuracy of the bootstrapping algorithm."""
    # Create a set of par yields that should give exact results
    par_yields = {
        1.0: 0.05,  # 5% 1-year rate
        2.0: 0.06,  # 6% 2-year rate
    }
    
    # Bootstrap zero-coupon yields
    zeros = bootstrap_par_yields(par_yields)
    
    # For the first point, zero rate should equal par rate
    assert abs(zeros[1.0] - par_yields[1.0]) < 1e-10
    
    # For the second point, we can calculate the exact zero rate
    # The formula is: z2 = (2*y2 - y1)/(2 - y1)
    expected_z2 = (2 * par_yields[2.0] - par_yields[1.0]) / (2 - par_yields[1.0])
    assert abs(zeros[2.0] - expected_z2) < 1e-10

def test_bootstrap_tenor_validation():
    """Test validation of tenor values."""
    # Test with non-positive tenors
    with pytest.raises(ValueError):
        bootstrap_par_yields({0.0: 0.05, 1.0: 0.06})
    
    with pytest.raises(ValueError):
        bootstrap_par_yields({-1.0: 0.05, 1.0: 0.06})
    
    # Test with non-monotonic tenors
    with pytest.raises(ValueError):
        bootstrap_par_yields({2.0: 0.06, 1.0: 0.05}) 