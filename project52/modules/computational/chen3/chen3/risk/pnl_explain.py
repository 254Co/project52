# -------------------- risk/pnl_explain.py --------------------
"""
Profit and Loss (PnL) Explanation Module

This module provides functionality for decomposing profit and loss (PnL) into
contributions from different risk factors in the three-factor Chen model. It
allows for attribution of PnL changes to specific market movements in:
- Interest rates
- Volatility
- Equity prices

The module is particularly useful for:
1. Risk attribution and analysis
2. Understanding PnL drivers
3. Scenario analysis and stress testing
4. Risk reporting and monitoring
"""

from typing import Any, Dict

import numpy as np

from .scenario import Scenario


def explain_pnl(
    base_price: float, shocked_price: float, scenario: Scenario = None
) -> Dict[str, float]:
    """
    Compute PnL and decompose it into contributions from different risk factors.

    This function calculates the total PnL as the difference between shocked and
    base prices, then allocates the PnL to different risk factors based on their
    relative shifts in the scenario. The allocation is done using a linear
    approximation based on the magnitude of each factor's shift.

    Args:
        base_price (float): Base price before scenario shocks
        shocked_price (float): Price after applying scenario shocks
        scenario (Scenario, optional): Scenario containing risk factor shifts.
            If provided, PnL is decomposed by factor. If None, only total PnL
            is returned.

    Returns:
        Dict[str, float]: Dictionary containing PnL contributions:
            - "total": Total PnL (shocked_price - base_price)
            - "rate": PnL contribution from interest rate changes
            - "vol": PnL contribution from volatility changes
            - "equity": PnL contribution from equity price changes

    Example:
        >>> from chen3.risk import Scenario
        >>> scenario = Scenario(rate_shift=0.01, vol_shift=0.05, equity_shift=-0.02)
        >>> contribs = explain_pnl(100.0, 102.5, scenario)
        >>> print(contribs)
        {'total': 2.5, 'rate': 0.3125, 'vol': 1.5625, 'equity': 0.625}

    Notes:
        - PnL decomposition is approximate and based on linear allocation
        - Allocation is proportional to the magnitude of factor shifts
        - If no shifts are present, all factor contributions are set to zero
        - Total PnL is always returned, even without scenario
    """
    pnl = shocked_price - base_price
    contrib: Dict[str, float] = {"total": pnl}
    if scenario is not None:
        # approximate linear split by shifts
        total_shift = (
            abs(scenario.rate_shift)
            + abs(scenario.vol_shift)
            + abs(scenario.equity_shift)
        )
        if total_shift > 0:
            contrib["rate"] = pnl * (scenario.rate_shift / total_shift)
            contrib["vol"] = pnl * (scenario.vol_shift / total_shift)
            contrib["equity"] = pnl * (scenario.equity_shift / total_shift)
        else:
            contrib.update({"rate": 0.0, "vol": 0.0, "equity": 0.0})
    return contrib
