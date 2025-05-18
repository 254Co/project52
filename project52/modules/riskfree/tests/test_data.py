"""Tests for the data fetching functionality."""
import pytest
from datetime import date, timedelta
from riskfree.data.treasury import fetch_par_curve

def test_fetch_par_curve_basic():
    """Test basic par curve fetching functionality."""
    # Fetch today's curve
    curve = fetch_par_curve(date.today())
    
    # Check basic properties
    assert isinstance(curve, dict)
    assert len(curve) > 0
    
    # Check that all tenors are positive
    assert all(t > 0 for t in curve.keys())
    
    # Check that all rates are non-negative
    assert all(r >= 0 for r in curve.values())
    
    # Check that tenors are in ascending order
    tenors = sorted(curve.keys())
    assert tenors == list(curve.keys())

def test_fetch_par_curve_historical():
    """Test fetching historical par curves."""
    # Fetch curve for a known historical date
    # Using a date from 2023 as it should have data
    historical_date = date(2023, 1, 1)
    curve = fetch_par_curve(historical_date)
    
    # Check basic properties
    assert isinstance(curve, dict)
    assert len(curve) > 0
    
    # Check that rates are reasonable for that time period
    # Rates should be positive but not extremely high
    assert all(0 < r < 0.2 for r in curve.values())

def test_fetch_par_curve_future():
    """Test fetching par curve for future dates."""
    # Try to fetch curve for a future date
    future_date = date.today() + timedelta(days=365)
    
    # Should raise ValueError for future dates
    with pytest.raises(ValueError):
        fetch_par_curve(future_date)

def test_fetch_par_curve_weekend():
    """Test fetching par curve for weekend dates."""
    # Find the most recent Saturday
    today = date.today()
    days_since_saturday = (today.weekday() - 5) % 7
    saturday = today - timedelta(days=days_since_saturday)
    
    # Should return Friday's curve for weekend dates
    curve = fetch_par_curve(saturday)
    assert isinstance(curve, dict)
    assert len(curve) > 0

def test_fetch_par_curve_holiday():
    """Test fetching par curve for holiday dates."""
    # Try to fetch curve for a known holiday
    # Using New Year's Day 2023 as an example
    holiday = date(2023, 1, 1)
    curve = fetch_par_curve(holiday)
    
    # Should return the most recent available curve
    assert isinstance(curve, dict)
    assert len(curve) > 0

def test_fetch_par_curve_tenor_range():
    """Test that fetched curves have expected tenor range."""
    curve = fetch_par_curve(date.today())
    
    # Check that we have standard tenors
    standard_tenors = {1, 2, 3, 5, 7, 10, 20, 30}
    assert all(t in standard_tenors for t in curve.keys())
    
    # Check that we have at least the key tenors
    assert all(t in curve for t in [2, 5, 10, 30])

def test_fetch_par_curve_rate_range():
    """Test that fetched rates are within reasonable ranges."""
    curve = fetch_par_curve(date.today())
    
    # Check that rates are within reasonable bounds
    # This is a very conservative range that should work for most market conditions
    assert all(0 < r < 0.2 for r in curve.values())
    
    # Check that rates are monotonically increasing
    tenors = sorted(curve.keys())
    for i in range(1, len(tenors)):
        assert curve[tenors[i]] >= curve[tenors[i-1]]

def test_fetch_par_curve_retry():
    """Test that the function handles temporary failures gracefully."""
    # This test is more of an integration test
    # It verifies that the function can handle temporary network issues
    
    # Try fetching the curve multiple times
    for _ in range(3):
        try:
            curve = fetch_par_curve(date.today())
            assert isinstance(curve, dict)
            assert len(curve) > 0
            break
        except Exception as e:
            # If we get an error, it should be a specific type
            assert isinstance(e, (ValueError, ConnectionError))
            continue
    else:
        pytest.fail("Failed to fetch curve after multiple attempts") 