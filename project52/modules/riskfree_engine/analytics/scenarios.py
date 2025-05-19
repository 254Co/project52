# File: analytics/scenarios.py


def rate_shift_scenario(spline, shift_bp: float):
    """Return a new callable curve shifted parallel by `shift_bp` basis‑points."""
    shift = shift_bp / 10000
    return lambda t: float(spline(t)) + shift
