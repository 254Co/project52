"""Common test fixtures and configurations."""
import pytest
from datetime import date, timedelta
from riskfree import RiskFreeCurve

@pytest.fixture
def sample_par_yields():
    """Create a sample set of par yields for testing."""
    return {
        1.0: 0.05,  # 5% 1-year rate
        2.0: 0.06,  # 6% 2-year rate
        3.0: 0.07,  # 7% 3-year rate
        5.0: 0.08,  # 8% 5-year rate
        7.0: 0.09,  # 9% 7-year rate
        10.0: 0.10,  # 10% 10-year rate
    }

@pytest.fixture
def sample_curve():
    """Create a sample RiskFreeCurve instance for testing."""
    return RiskFreeCurve(date.today())

@pytest.fixture
def historical_date():
    """Return a known historical date for testing."""
    return date(2023, 1, 1)

@pytest.fixture
def weekend_date():
    """Return a weekend date for testing."""
    today = date.today()
    days_since_saturday = (today.weekday() - 5) % 7
    return today - timedelta(days=days_since_saturday)

@pytest.fixture
def holiday_date():
    """Return a known holiday date for testing."""
    return date(2023, 1, 1)  # New Year's Day

@pytest.fixture
def future_date():
    """Return a future date for testing."""
    return date.today() + timedelta(days=365)

@pytest.fixture
def nelson_siegel_params():
    """Return standard Nelson-Siegel parameters for testing."""
    return {
        'beta': [0.05, -0.02, 0.01],  # [level, slope, curvature]
        'tau': 2.0  # decay parameter
    } 