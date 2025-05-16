# File: chen3/schemes/adf.py
"""Andersen-Director-Fuchs weak scheme stub."""
def adf_scheme(process, dt: float):
    def step(state, z):
        # Placeholder for ADF weak order 1.5
        return process.drift(state)*dt + process.diffusion(state)*z
    return step