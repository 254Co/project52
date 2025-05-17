# File: chen3/utils/math.py
"""Math helpers."""
def incurred_discount(rate, t):
    return (1 + rate)**(-t)


def safe_sqrt(x):
    return x**0.5 if x >= 0 else 0.0

