# File: chen3/simulators/gpu_vectorized.py
"""CuPy based GPU simulator."""
import cupy as cp
from .core import PathGenerator

class GPUVectorizedSimulator(PathGenerator):
    def generate(self):
        return cp.zeros((self.params.n_paths, self.params.n_steps, 3))

