# File: chen3/payoffs/hybrid/__init__.py
"""
Hybrid payoffs combining equity and interest-rate (and credit) risk.
"""
from .convertible_credit import ConvertibleCreditBond
from .equity_rate_hybrid import EquityRateHybridProduct

__all__ = ["ConvertibleCreditBond", "EquityRateHybridProduct"]
