"""Moment‑matching calibration for Campbell‑Cochrane parameters.

This module provides functionality for calibrating the Campbell-Cochrane habit formation
model to match target moments of key financial metrics. It uses numerical optimization
to find model parameters that minimize the squared error between simulated and target
moments, such as risk-free rates, equity premiums, and Sharpe ratios.
"""
from __future__ import annotations

import logging
from functools import partial

import numpy as np
from scipy.optimize import minimize

from ..core.analytics import Analytics
from ..core.constants import ModelParams
from ..core.dynamics import simulate_paths
from ..core.model import HabitModel
from ..utils.logging_config import setup_logging
from ..utils.validation import assert_finite

logger = logging.getLogger(__name__)

# Valid metric keys from Analytics.summary()
VALID_METRICS = {"E_r_f", "Equity_Premium", "Sharpe"}

def _paths_and_metrics(
    theta: np.ndarray,
    target_keys: tuple[str, ...],
    n_paths: int,
    n_steps: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Generate simulated paths and compute target metrics.
    
    This helper function simulates paths for given parameters and computes
    the specified target metrics. It includes validation of both paths and
    computed metrics.
    
    Args:
        theta (np.ndarray): Array of model parameters.
        target_keys (tuple[str, ...]): Keys of metrics to compute.
        n_paths (int): Number of paths to simulate.
        n_steps (int): Number of time steps per path.
        seed (int): Random seed for simulation.
        
    Returns:
        tuple[dict[str, np.ndarray], dict[str, float]]: A tuple containing:
            - Dictionary of simulated paths ("S" and "g_c")
            - Dictionary of computed metrics
            
    Raises:
        RuntimeError: If simulation or metric computation fails.
    """
    try:
        params = ModelParams(*theta)
        paths = simulate_paths(params, n_steps, n_paths, rng=np.random.default_rng(seed))
        
        # Validate paths
        for key, arr in paths.items():
            assert_finite(arr, f"simulation output {key}")
            
        ana = Analytics(HabitModel(params))
        metrics = ana.summary(paths)
        
        # Validate metrics
        for key, value in metrics.items():
            assert_finite(value, f"metric {key}")
            
        return paths, {k: metrics[k] for k in target_keys}
    except Exception as e:
        logger.error("Failed to compute metrics: %s", str(e))
        raise RuntimeError(f"Metric computation failed: {str(e)}")


def calibrate(
    target_moments: dict[str, float],
    *,
    initial_params: ModelParams | None = None,
    n_paths: int = 100_000,
    n_steps: int = 1200,
    tol: float = 1e-6,
    maxiter: int = 200,
    quiet: bool = False,
    seed: int = 42,
) -> ModelParams:
    """Calibrate model parameters to match target moments.
    
    This function performs moment-matching calibration of the Campbell-Cochrane
    model parameters using numerical optimization. It minimizes the sum of squared
    errors between simulated and target moments.
    
    The calibration process:
    1. Validates input target moments
    2. Sets up optimization bounds for each parameter
    3. Defines an objective function that simulates paths and computes metrics
    4. Uses L-BFGS-B optimization to find parameters that minimize the error
    
    Args:
        target_moments (dict[str, float]): Dictionary of target moments to match.
            Keys must be in {"E_r_f", "Equity_Premium", "Sharpe"}.
        initial_params (ModelParams | None, optional): Initial parameter guess.
            If None, uses default parameters. Defaults to None.
        n_paths (int, optional): Number of paths to simulate per evaluation.
            Defaults to 100_000.
        n_steps (int, optional): Number of time steps per path. Defaults to 1200.
        tol (float, optional): Optimization tolerance. Defaults to 1e-6.
        maxiter (int, optional): Maximum optimization iterations. Defaults to 200.
        quiet (bool, optional): Whether to suppress logging. Defaults to False.
        seed (int, optional): Random seed for simulation. Defaults to 42.
        
    Returns:
        ModelParams: Calibrated model parameters.
        
    Raises:
        TypeError: If target_moments is not a dictionary or contains non-numeric values.
        ValueError: If target_moments is empty, contains invalid keys, or contains
            non-finite values.
        RuntimeError: If optimization fails to converge.
        
    Note:
        The optimization uses the L-BFGS-B algorithm with the following parameter bounds:
            - gamma: [1.0, 15.0] (risk aversion)
            - beta: [0.90, 0.999] (discount factor)
            - phi: [0.0, 0.99] (persistence)
            - sigma_c: [1e-4, 0.05] (consumption volatility)
            - g: [1e-5, 0.02] (growth rate)
            - s_bar: [1e-3, 0.30] (steady-state surplus)
            - lambda_s: [0.1, 10.0] (sensitivity)
    """
    # Input validation
    if not isinstance(target_moments, dict):
        raise TypeError("target_moments must be a dictionary")
    if not target_moments:
        raise ValueError("target_moments cannot be empty")
        
    invalid_keys = set(target_moments.keys()) - VALID_METRICS
    if invalid_keys:
        raise ValueError(f"Invalid target moments: {invalid_keys}")
        
    for key, value in target_moments.items():
        if not isinstance(value, (int, float)):
            raise TypeError(f"Target moment {key} must be numeric")
        if not np.isfinite(value):
            raise ValueError(f"Target moment {key} must be finite")
            
    if n_paths <= 0 or n_steps <= 0:
        raise ValueError("n_paths and n_steps must be positive")
    if tol <= 0:
        raise ValueError("tol must be positive")
    if maxiter <= 0:
        raise ValueError("maxiter must be positive")

    setup_logging("INFO") if not quiet else None
    initial = initial_params or ModelParams()
    logger.info("Starting calibration with %s", initial)

    bounds = [
        (1.0, 15.0),    # gamma
        (0.90, 0.999),  # beta
        (0.0, 0.99),    # phi
        (1e-4, 0.05),   # sigma_c
        (1e-5, 0.02),   # g
        (1e-3, 0.30),   # s_bar
        (0.1, 10.0),    # lambda_s
    ]

    target_keys = tuple(target_moments.keys())

    def objective(theta: np.ndarray) -> float:
        """Objective function for optimization.
        
        Computes the sum of squared errors between simulated and target moments.
        Returns infinity if computation fails to ensure optimization continues.
        
        Args:
            theta (np.ndarray): Current parameter values.
            
        Returns:
            float: Sum of squared errors, or infinity if computation fails.
        """
        try:
            _, sample = _paths_and_metrics(theta, target_keys, n_paths, n_steps, seed)
            err = np.array([sample[k] - target_moments[k] for k in target_keys])
            sse = float((err**2).sum())
            if not np.isfinite(sse):
                return np.inf
            logger.debug("θ=%s  sse=%.4e", theta, sse)
            return sse
        except Exception as e:
            logger.error("Objective evaluation failed: %s", str(e))
            return np.inf

    result = minimize(
        objective,
        initial.as_tuple(),
        bounds=bounds,
        method="L-BFGS-B",
        tol=tol,
        options={
            "maxiter": maxiter,
            "disp": not quiet,
            "ftol": tol,
            "gtol": tol,
        },
    )
    
    if not result.success:
        raise RuntimeError(f"Calibration failed: {result.message}")

    fitted = ModelParams(*result.x)
    logger.info("Calibration complete: %s", fitted)
    return fitted