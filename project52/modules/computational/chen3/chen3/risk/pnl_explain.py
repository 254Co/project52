# -------------------- risk/pnl_explain.py --------------------
"""PnL explanation: decompose PnL by risk factor contributions."""
import numpy as np
from typing import Dict, Any


def explain_pnl(
    base_price: float,
    shocked_price: float,
    scenario: Scenario = None
) -> Dict[str, float]:
    """
    Compute PnL and per-factor contributions.
    If a Scenario is provided, uses its attributes to allocate contributions.
    """
    pnl = shocked_price - base_price
    contrib: Dict[str, float] = {"total": pnl}
    if scenario is not None:
        # approximate linear split by shifts
        total_shift = (
            abs(scenario.rate_shift) +
            abs(scenario.vol_shift) +
            abs(scenario.equity_shift)
        )
        if total_shift > 0:
            contrib["rate"] = pnl * (scenario.rate_shift / total_shift)
            contrib["vol"] = pnl * (scenario.vol_shift / total_shift)
            contrib["equity"] = pnl * (scenario.equity_shift / total_shift)
        else:
            contrib.update({"rate": 0.0, "vol": 0.0, "equity": 0.0})
    return contrib