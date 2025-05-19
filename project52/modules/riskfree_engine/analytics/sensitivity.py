# File: analytics/sensitivity.py


def dv01(spline, maturity: float, face: float = 1_000_000):
    """Dollar value of 1bp for a zero coupon bond of maturity T."""
    rate = float(spline(maturity))
    pv = face * np.exp(-rate * maturity)
    bumped = face * np.exp(-(rate + 0.0001) * maturity)
    return pv - bumped


def convexity(spline, maturity: float, face: float = 1_000_000):
    r = float(spline(maturity))
    pv = face * np.exp(-r * maturity)
    # Second derivative approximation using ±1bp
    up = face * np.exp(-(r + 0.0001) * maturity)
    dn = face * np.exp(-(r - 0.0001) * maturity)
    return (up + dn - 2 * pv) / (pv * (0.0001**2))