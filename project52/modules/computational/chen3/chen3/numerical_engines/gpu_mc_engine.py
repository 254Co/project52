# -------------------- chen3/numerical_engines/gpu_mc_engine.py --------------------
"""
GPU-accelerated Monte Carlo engine using CuPy.
"""
import cupy as cp
from typing import Any
from chen3.simulators.gpu_vectorized import GPUVectorizedSimulator

class GPUMonteCarloEngine:
    def __init__(self, model: Any, settings: Any):
        self.model = model
        self.settings = settings
        # seed handled inside simulator

    def simulate(self) -> cp.ndarray:
        sim = GPUVectorizedSimulator(self.model, self.settings)
        return sim.generate()