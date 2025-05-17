# -------------------- chen3/numerical_engines/variance_reduction.py --------------------
"""
Enhanced variance reduction techniques for Monte Carlo simulations.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol, Tuple, Union

import numpy as np
from scipy.stats import norm


@dataclass
class VREstimator:
    """Container for variance reduction estimates and statistics."""

    estimate: float
    standard_error: float
    variance_reduction: float
    n_samples: int
    technique: str


class VarianceReductionTechnique(ABC):
    """Base class for variance reduction techniques."""

    @abstractmethod
    def apply(
        self,
        payoffs: np.ndarray,
        control_values: Optional[np.ndarray] = None,
        control_expectation: Optional[float] = None,
    ) -> VREstimator:
        """Apply the variance reduction technique."""
        pass


class AntitheticVariates(VarianceReductionTechnique):
    """Antithetic variates technique."""

    def apply(
        self,
        payoffs: np.ndarray,
        control_values: Optional[np.ndarray] = None,
        control_expectation: Optional[float] = None,
    ) -> VREstimator:
        """Apply antithetic variates."""
        n_samples = len(payoffs) // 2
        original_payoffs = payoffs[:n_samples]
        antithetic_payoffs = payoffs[n_samples:]

        # Combine original and antithetic paths
        combined_payoffs = (original_payoffs + antithetic_payoffs) / 2

        # Compute statistics
        estimate = np.mean(combined_payoffs)
        variance = np.var(combined_payoffs) / n_samples
        original_variance = np.var(payoffs) / len(payoffs)
        vr = 1 - variance / original_variance if original_variance > 0 else 0

        return VREstimator(
            estimate=estimate,
            standard_error=np.sqrt(variance),
            variance_reduction=vr,
            n_samples=n_samples,
            technique="antithetic_variates",
        )


class ControlVariates(VarianceReductionTechnique):
    """Control variates technique."""

    def apply(
        self,
        payoffs: np.ndarray,
        control_values: Optional[np.ndarray] = None,
        control_expectation: Optional[float] = None,
    ) -> VREstimator:
        """Apply control variates."""
        if control_values is None or control_expectation is None:
            raise ValueError("Control values and expectation are required")

        # Compute optimal coefficient
        cov = np.cov(payoffs, control_values, ddof=0)[0, 1]
        var_cv = np.var(control_values)
        beta = cov / var_cv if var_cv > 0 else 0.0

        # Apply control variate
        adjusted_payoffs = payoffs - beta * (control_values - control_expectation)

        # Compute statistics
        estimate = np.mean(adjusted_payoffs)
        variance = np.var(adjusted_payoffs) / len(adjusted_payoffs)
        original_variance = np.var(payoffs) / len(payoffs)
        vr = 1 - variance / original_variance if original_variance > 0 else 0

        return VREstimator(
            estimate=estimate,
            standard_error=np.sqrt(variance),
            variance_reduction=vr,
            n_samples=len(payoffs),
            technique="control_variates",
        )


class ImportanceSampling(VarianceReductionTechnique):
    """Importance sampling technique."""

    def __init__(self, proposal: Callable, target: Callable):
        """Initialize importance sampling."""
        self.proposal = proposal
        self.target = target

    def apply(
        self,
        payoffs: np.ndarray,
        control_values: Optional[np.ndarray] = None,
        control_expectation: Optional[float] = None,
    ) -> VREstimator:
        """Apply importance sampling."""
        # Compute weights
        weights = self.target(payoffs) / self.proposal(payoffs)

        # Compute weighted function values
        weighted_payoffs = payoffs * weights

        # Compute statistics
        estimate = np.mean(weighted_payoffs)
        variance = np.var(weighted_payoffs) / len(weighted_payoffs)
        original_variance = np.var(payoffs) / len(payoffs)
        vr = 1 - variance / original_variance if original_variance > 0 else 0

        return VREstimator(
            estimate=estimate,
            standard_error=np.sqrt(variance),
            variance_reduction=vr,
            n_samples=len(payoffs),
            technique="importance_sampling",
        )


class StratifiedSampling(VarianceReductionTechnique):
    """Stratified sampling technique."""

    def __init__(self, n_strata: int):
        """Initialize stratified sampling."""
        self.n_strata = n_strata

    def apply(
        self,
        payoffs: np.ndarray,
        control_values: Optional[np.ndarray] = None,
        control_expectation: Optional[float] = None,
    ) -> VREstimator:
        """Apply stratified sampling."""
        n_samples = len(payoffs)
        samples_per_stratum = n_samples // self.n_strata

        # Split payoffs into strata
        strata = np.array_split(payoffs, self.n_strata)

        # Compute statistics for each stratum
        stratum_means = [np.mean(s) for s in strata]
        stratum_vars = [np.var(s) / len(s) for s in strata]

        # Combine estimates
        estimate = np.mean(stratum_means)
        variance = np.mean(stratum_vars)
        original_variance = np.var(payoffs) / n_samples
        vr = 1 - variance / original_variance if original_variance > 0 else 0

        return VREstimator(
            estimate=estimate,
            standard_error=np.sqrt(variance),
            variance_reduction=vr,
            n_samples=n_samples,
            technique="stratified_sampling",
        )


class MultiLevelMonteCarlo(VarianceReductionTechnique):
    """Multi-level Monte Carlo technique."""

    def __init__(self, levels: List[int], n_samples: List[int]):
        """Initialize multi-level Monte Carlo."""
        self.levels = levels
        self.n_samples = n_samples

    def apply(
        self,
        payoffs: np.ndarray,
        control_values: Optional[np.ndarray] = None,
        control_expectation: Optional[float] = None,
    ) -> VREstimator:
        """Apply multi-level Monte Carlo."""
        # Split payoffs into levels
        level_payoffs = np.array_split(payoffs, len(self.levels))

        # Compute statistics for each level
        level_means = [np.mean(p) for p in level_payoffs]
        level_vars = [np.var(p) / len(p) for p in level_payoffs]

        # Combine estimates
        estimate = np.sum(level_means)
        variance = np.sum(level_vars)
        original_variance = np.var(payoffs) / len(payoffs)
        vr = 1 - variance / original_variance if original_variance > 0 else 0

        return VREstimator(
            estimate=estimate,
            standard_error=np.sqrt(variance),
            variance_reduction=vr,
            n_samples=len(payoffs),
            technique="multi_level_monte_carlo",
        )


# Factory function to create variance reduction techniques
def create_variance_reduction(technique: str, **kwargs) -> VarianceReductionTechnique:
    """
    Create a variance reduction technique.

    Parameters
    ----------
    technique : str
        Name of the technique
    **kwargs
        Additional arguments for the technique

    Returns
    -------
    VarianceReductionTechnique
        The created technique
    """
    techniques = {
        "antithetic": AntitheticVariates,
        "control": ControlVariates,
        "importance": lambda: ImportanceSampling(**kwargs),
        "stratified": lambda: StratifiedSampling(**kwargs),
        "multi_level": lambda: MultiLevelMonteCarlo(**kwargs),
    }

    if technique not in techniques:
        raise ValueError(f"Unknown variance reduction technique: {technique}")

    return techniques[technique]()
