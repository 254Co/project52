"""High‑level Monte‑Carlo orchestration with multiprocessing.

This module provides parallel Monte Carlo simulation capabilities for the Campbell-Cochrane
habit formation model. It implements multiprocessing to distribute simulation work across
multiple CPU cores, significantly improving performance for large-scale simulations.
"""
from __future__ import annotations

import itertools
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np

from ..core.constants import ModelParams
from ..core.dynamics import simulate_paths
from ..utils.validation import assert_finite

__all__ = ["run_parallel"]


_DEF_WORKERS = max(1, (os.cpu_count() or 4) // 2)


def _worker(
    params: ModelParams,
    n_steps: int,
    chunk: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Worker function for parallel simulation.
    
    This function is executed in a separate process to simulate a chunk of paths.
    It handles the actual simulation work and includes error handling and validation.
    
    Args:
        params (ModelParams): Model parameters container.
        n_steps (int): Number of time steps to simulate.
        chunk (int): Number of paths to simulate in this worker.
        seed (int): Random seed for this worker's simulation.
        
    Returns:
        dict[str, np.ndarray]: Dictionary containing simulated paths:
            - "S": Array of surplus-consumption ratios
            - "g_c": Array of log consumption growth rates
            
    Raises:
        RuntimeError: If simulation fails or produces invalid results.
    """
    try:
        rng = np.random.default_rng(seed)
        paths = simulate_paths(params, n_steps, chunk, rng)
        # Validate outputs
        for key, arr in paths.items():
            assert_finite(arr, f"worker output {key}")
        return paths
    except Exception as e:
        raise RuntimeError(f"Worker failed with seed {seed}: {str(e)}")


def run_parallel(
    params: ModelParams,
    *,
    n_paths: int,
    n_steps: int,
    n_workers: int | None = None,
    base_seed: int = 1234,
) -> dict[str, np.ndarray]:
    """Run parallel Monte Carlo simulation of the Campbell-Cochrane model.
    
    This function orchestrates parallel simulation by:
    1. Dividing the total number of paths into chunks
    2. Spawning worker processes to simulate each chunk
    3. Collecting and combining results from all workers
    4. Validating the final output
    
    The simulation uses independent random seeds for each worker to ensure
    statistical independence of the results.
    
    Args:
        params (ModelParams): Model parameters container.
        n_paths (int): Total number of paths to simulate.
        n_steps (int): Number of time steps to simulate.
        n_workers (int | None, optional): Number of worker processes to use.
            If None, uses half the available CPU cores. Defaults to None.
        base_seed (int, optional): Base seed for random number generation.
            Each worker gets a unique seed derived from this. Defaults to 1234.
            
    Returns:
        dict[str, np.ndarray]: Dictionary containing combined simulated paths:
            - "S": Array of surplus-consumption ratios with shape (n_paths, n_steps+1)
            - "g_c": Array of log consumption growth rates with shape (n_paths, n_steps)
            
    Raises:
        ValueError: If n_paths or n_steps is non-positive.
        RuntimeError: If any worker fails or produces invalid results.
        
    Note:
        The number of workers defaults to half the available CPU cores to avoid
        overwhelming the system. Each worker simulates an equal chunk of paths,
        with the last worker potentially handling fewer paths if n_paths is not
        evenly divisible by n_workers.
    """
    if n_paths <= 0 or n_steps <= 0:
        raise ValueError("n_paths and n_steps must be positive")
        
    n_workers = n_workers or _DEF_WORKERS
    chunk = int(np.ceil(n_paths / n_workers))
    seed_seq = np.random.SeedSequence(base_seed)
    seeds = [int(s.generate_state(1)[0]) for s in seed_seq.spawn(n_workers)]

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [
            pool.submit(_worker, params, n_steps, chunk, seed)
            for seed in seeds
        ]
        
        # Collect results with error handling
        parts: list[dict[str, np.ndarray]] = []
        for future in as_completed(futures):
            try:
                parts.append(future.result())
            except Exception as e:
                pool.shutdown(wait=False)
                raise RuntimeError(f"Simulation failed: {str(e)}")

    # Stack and validate results
    result = {
        "S": np.vstack([p["S"] for p in parts])[:n_paths],
        "g_c": np.vstack([p["g_c"] for p in parts])[:n_paths],
    }
    
    for key, arr in result.items():
        assert_finite(arr, f"final output {key}")
        
    return result