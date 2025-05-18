"""Day‑count conventions."""
from __future__ import annotations
from datetime import datetime

def yearfrac(start: datetime, end: datetime, convention: str = "ACT/365") -> float:
    """Return fraction of year between *start* and *end* under *convention*."""
    seconds = (end - start).total_seconds()
    if convention.upper() == "ACT/365":
        return seconds / (365 * 24 * 3600)
    if convention.upper() == "ACT/360":
        return seconds / (360 * 24 * 3600)
    raise ValueError(f"Unsupported day‑count: {convention}")