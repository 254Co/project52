# -------------------- chen3/numerical_engines/multi_gpu_distributor.py --------------------
"""
Distribute GPU Monte Carlo across multiple GPUs via Ray.
"""
import ray
import cupy as cp
from typing import Any
from chen3.simulators.gpu_vectorized import GPUVectorizedSimulator

ray.init(ignore_reinit_error=True)

@ray.remote(num_gpus=1)
def _simulate_chunk(model: Any, settings: Any, offset: int) -> Any:
    settings.seed += offset
    sim = GPUVectorizedSimulator(model, settings)
    return sim.generate()

class MultiGPUSimulator:
    def __init__(self, model: Any, settings: Any, n_chunks: int = 2):
        self.model = model
        self.settings = settings
        self.n_chunks = n_chunks

    def generate(self) -> cp.ndarray:
        futures = [
            _simulate_chunk.remote(self.model, self.settings, i)
            for i in range(self.n_chunks)
        ]
        results = ray.get(futures)
        return cp.concatenate(results, axis=0)
