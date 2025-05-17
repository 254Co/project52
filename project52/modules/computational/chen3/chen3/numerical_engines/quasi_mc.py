# -------------------- chen3/numerical_engines/quasi_mc.py --------------------
"""
Enhanced Low-discrepancy (quasi-Monte Carlo) sequences with error estimation.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.stats import norm, qmc

from ..utils.exceptions import NumericalError
from ..utils.logging_config import logger
from .integration import IntegrationScheme
from .monte_carlo import MonteCarloEngineBase
from .path_generator import PathGenerator
from .random import RandomNumberGenerator
from .variance_reduction import VarianceReductionTechnique


@dataclass
class QMCEstimator:
    """Container for QMC estimates and error bounds."""

    estimate: float
    error_bound: float
    n_samples: int
    dimension: int
    sequence_type: str


def sobol_normal(
    n_samples: int, dim: int, scramble: bool = True, seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate normal variates using Sobol sequence.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    dim : int
        Dimension of the samples
    scramble : bool, optional
        Whether to use scrambled sequence, by default True
    seed : Optional[int], optional
        Random seed for scrambling, by default None

    Returns
    -------
    np.ndarray
        Array of normal variates
    """
    sampler = qmc.Sobol(d=dim, scramble=scramble, seed=seed)
    u = sampler.random(n_samples)
    return norm.ppf(u)


def halton_normal(
    n_samples: int, dim: int, scramble: bool = False, seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate normal variates using Halton sequence.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    dim : int
        Dimension of the samples
    scramble : bool, optional
        Whether to use scrambled sequence, by default False
    seed : Optional[int], optional
        Random seed for scrambling, by default None

    Returns
    -------
    np.ndarray
        Array of normal variates
    """
    sampler = qmc.Halton(d=dim, scramble=scramble, seed=seed)
    u = sampler.random(n_samples)
    return norm.ppf(u)


def faure_normal(
    n_samples: int, dim: int, scramble: bool = True, seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate normal variates using Faure sequence.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    dim : int
        Dimension of the samples
    scramble : bool, optional
        Whether to use scrambled sequence, by default True
    seed : Optional[int], optional
        Random seed for scrambling, by default None

    Returns
    -------
    np.ndarray
        Array of normal variates
    """
    sampler = qmc.Faure(d=dim, scramble=scramble, seed=seed)
    u = sampler.random(n_samples)
    return norm.ppf(u)


def niederreiter_normal(
    n_samples: int, dim: int, scramble: bool = True, seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate normal variates using Niederreiter sequence.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    dim : int
        Dimension of the samples
    scramble : bool, optional
        Whether to use scrambled sequence, by default True
    seed : Optional[int], optional
        Random seed for scrambling, by default None

    Returns
    -------
    np.ndarray
        Array of normal variates
    """
    sampler = qmc.Niederreiter(d=dim, scramble=scramble, seed=seed)
    u = sampler.random(n_samples)
    return norm.ppf(u)


def estimate_qmc_error(
    samples: np.ndarray, sequence_type: str, confidence_level: float = 0.95
) -> QMCEstimator:
    """
    Estimate error bounds for QMC integration.

    Parameters
    ----------
    samples : np.ndarray
        Array of function evaluations
    sequence_type : str
        Type of QMC sequence used
    confidence_level : float, optional
        Confidence level for error bounds, by default 0.95

    Returns
    -------
    QMCEstimator
        Container with estimate and error bounds
    """
    n_samples = len(samples)
    estimate = np.mean(samples)

    # Compute error bound based on Koksma-Hlawka inequality
    # This is a simplified version - more sophisticated bounds exist
    discrepancy = 1.0 / np.sqrt(n_samples)  # Simplified bound
    variation = np.std(samples)
    error_bound = discrepancy * variation * norm.ppf((1 + confidence_level) / 2)

    return QMCEstimator(
        estimate=estimate,
        error_bound=error_bound,
        n_samples=n_samples,
        dimension=samples.shape[1] if len(samples.shape) > 1 else 1,
        sequence_type=sequence_type,
    )


def adaptive_qmc(
    func: callable,
    dim: int,
    target_error: float,
    max_samples: int = 1000000,
    sequence_type: str = "sobol",
    confidence_level: float = 0.95,
    seed: Optional[int] = None,
) -> QMCEstimator:
    """
    Adaptive QMC integration with error control.

    Parameters
    ----------
    func : callable
        Function to integrate
    dim : int
        Dimension of the integration
    target_error : float
        Target error tolerance
    max_samples : int, optional
        Maximum number of samples, by default 1000000
    sequence_type : str, optional
        Type of QMC sequence to use, by default 'sobol'
    confidence_level : float, optional
        Confidence level for error bounds, by default 0.95
    seed : Optional[int], optional
        Random seed, by default None

    Returns
    -------
    QMCEstimator
        Container with estimate and error bounds
    """
    sequence_generators = {
        "sobol": sobol_normal,
        "halton": halton_normal,
        "faure": faure_normal,
        "niederreiter": niederreiter_normal,
    }

    if sequence_type not in sequence_generators:
        raise ValueError(f"Unknown sequence type: {sequence_type}")

    generator = sequence_generators[sequence_type]
    n_samples = 1000  # Start with small number of samples

    while n_samples <= max_samples:
        # Generate samples
        samples = generator(n_samples, dim, seed=seed)
        # Evaluate function
        values = func(samples)
        # Estimate error
        estimator = estimate_qmc_error(values, sequence_type, confidence_level)

        if estimator.error_bound <= target_error:
            return estimator

        n_samples *= 2  # Double the number of samples

    # Return best estimate if max samples reached
    return estimator


class QuasiMonteCarloEngine(MonteCarloEngineBase):
    """
    Quasi-Monte Carlo simulation engine for the Chen3 model.

    This class provides quasi-Monte Carlo simulation capabilities using
    low-discrepancy sequences for improved convergence rates.
    """

    def __init__(
        self,
        rng: Optional[RandomNumberGenerator] = None,
        path_generator: Optional[PathGenerator] = None,
        integration_scheme: Optional[IntegrationScheme] = None,
        n_paths: int = 10000,
        n_steps: int = 100,
        dt: float = 0.01,
        use_antithetic: bool = True,
        use_control_variate: bool = True,
        sequence_type: str = "sobol",
        scramble: bool = True,
        seed: Optional[int] = None,
        variance_reduction: Optional[VarianceReductionTechnique] = None,
    ):
        """Initialize the quasi-Monte Carlo engine."""
        super().__init__(
            rng=rng,
            path_generator=path_generator,
            integration_scheme=integration_scheme,
            n_paths=n_paths,
            n_steps=n_steps,
            dt=dt,
            use_antithetic=use_antithetic,
            use_control_variate=use_control_variate,
            variance_reduction=variance_reduction,
        )
        self.sequence_type = sequence_type
        self.scramble = scramble
        self.seed = seed

        # Validate sequence type
        if sequence_type not in ["sobol", "halton", "faure", "niederreiter"]:
            raise NumericalError(f"Unknown sequence type: {sequence_type}")

    def _generate_sequence(self, n_samples: int, dim: int) -> np.ndarray:
        """Generate low-discrepancy sequence."""
        sequence_generators = {
            "sobol": sobol_normal,
            "halton": halton_normal,
            "faure": faure_normal,
            "niederreiter": niederreiter_normal,
        }

        generator = sequence_generators[self.sequence_type]
        return generator(n_samples, dim, scramble=self.scramble, seed=self.seed)

    def simulate_paths(
        self,
        initial_state: Dict[str, float],
        drift_function: Callable,
        diffusion_function: Callable,
        correlation_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate paths for the three-factor model using quasi-Monte Carlo.

        Args:
            initial_state: Initial state variables
            drift_function: Function computing drift terms
            diffusion_function: Function computing diffusion terms
            correlation_matrix: Correlation matrix between factors

        Returns:
            Tuple of (rate_paths, equity_paths, variance_paths)

        Raises:
            NumericalError: If simulation fails
        """
        try:
            # Generate low-discrepancy sequence
            n_total = self.n_paths * (2 if self.use_antithetic else 1)
            dim = self.n_steps * 3  # Total dimension for all factors
            dW = self._generate_sequence(n_total, dim)
            dW = dW.reshape(n_total, self.n_steps, 3)

            # Initialize paths
            rate_paths = np.zeros((n_total, self.n_steps + 1))
            equity_paths = np.zeros((n_total, self.n_steps + 1))
            variance_paths = np.zeros((n_total, self.n_steps + 1))

            # Set initial values
            rate_paths[:, 0] = initial_state["r"]
            equity_paths[:, 0] = initial_state["S"]
            variance_paths[:, 0] = initial_state["v"]

            # Simulate paths
            for t in range(self.n_steps):
                # Current state
                state = {
                    "r": rate_paths[:, t],
                    "S": equity_paths[:, t],
                    "v": variance_paths[:, t],
                }

                # Compute drift and diffusion
                drift = drift_function(state, t * self.dt)
                diffusion = diffusion_function(state, t * self.dt)

                # Update paths
                rate_paths[:, t + 1] = (
                    rate_paths[:, t] + drift[0] * self.dt + diffusion[0] * dW[:, t, 0]
                )
                equity_paths[:, t + 1] = equity_paths[:, t] * np.exp(
                    (drift[1] - 0.5 * diffusion[1] ** 2) * self.dt
                    + diffusion[1] * dW[:, t, 1]
                )
                variance_paths[:, t + 1] = np.maximum(
                    variance_paths[:, t]
                    + drift[2] * self.dt
                    + diffusion[2] * dW[:, t, 2],
                    0.0,
                )

            # Apply antithetic variates if enabled
            if self.use_antithetic:
                rate_paths = (
                    rate_paths[: self.n_paths] + rate_paths[self.n_paths :]
                ) / 2
                equity_paths = (
                    equity_paths[: self.n_paths] + equity_paths[self.n_paths :]
                ) / 2
                variance_paths = (
                    variance_paths[: self.n_paths] + variance_paths[self.n_paths :]
                ) / 2

            return rate_paths, equity_paths, variance_paths

        except Exception as e:
            raise NumericalError(f"QMC path simulation failed: {str(e)}")

    def compute_expectation(
        self,
        payoff_function: Callable,
        paths: Tuple[np.ndarray, ...],
        control_variate: Optional[Callable] = None,
    ) -> Tuple[float, float]:
        """
        Compute expectation of a payoff function using quasi-Monte Carlo.

        Args:
            payoff_function: Function computing payoff
            paths: Simulated paths
            control_variate: Optional control variate function

        Returns:
            Tuple of (price, standard_error)
        """
        try:
            # Compute payoffs
            payoffs = payoff_function(*paths)

            # Apply variance reduction if enabled
            if self.variance_reduction is not None:
                control_values = (
                    control_variate(*paths) if control_variate is not None else None
                )
                control_expectation = (
                    np.mean(control_values) if control_values is not None else None
                )
                return self.apply_variance_reduction(
                    payoffs=payoffs,
                    control_values=control_values,
                    control_expectation=control_expectation,
                )

            # Compute QMC estimate and error bound
            estimator = estimate_qmc_error(
                payoffs, self.sequence_type, confidence_level=0.95
            )

            return estimator.estimate, estimator.error_bound

        except Exception as e:
            raise NumericalError(f"QMC expectation computation failed: {str(e)}")

    def __str__(self) -> str:
        """String representation of the quasi-Monte Carlo engine."""
        return (
            f"QuasiMonteCarloEngine(n_paths={self.n_paths}, n_steps={self.n_steps}, "
            f"dt={self.dt}, use_antithetic={self.use_antithetic}, "
            f"use_control_variate={self.use_control_variate}, "
            f"sequence_type={self.sequence_type}, scramble={self.scramble})"
        )

    def __repr__(self) -> str:
        """Detailed string representation of the quasi-Monte Carlo engine."""
        return self.__str__()
