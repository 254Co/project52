# File: chen3/schemes/exact.py
"""Exact discretization for CIR-like processes (where available)."""
def exact_scheme(process, dt: float):
    def step(state, z):
        raise NotImplementedError("Exact scheme not yet implemented.")
    return step