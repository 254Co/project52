# -------------------- chen3/numerical_engines/__init__.py --------------------
"""Numerical engines for advanced Monte Carlo and PDE methods."""
from .variance_reduction import antithetic_normal, control_variate, latin_hypercube_normal
from .quasi_mc import sobol_normal, halton_normal
from .gpu_mc_engine import GPUMonteCarloEngine
from .multi_gpu_distributor import MultiGPUSimulator
from .aad import mc_price_and_delta_jax
from .gpu_pde_solver import gpu_crank_nicolson

__all__ = [
    "antithetic_normal", "control_variate", "latin_hypercube_normal",
    "sobol_normal", "halton_normal",
    "GPUMonteCarloEngine", "MultiGPUSimulator",
    "mc_price_and_delta_jax", "gpu_crank_nicolson"
]