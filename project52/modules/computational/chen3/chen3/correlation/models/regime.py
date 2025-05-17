from typing import List
import numpy as np

class RegimeSwitchingCorrelation(BaseCorrelation):
    """
    Regime-switching correlation structure.

    This class implements correlations that switch between different regimes
    according to a continuous-time Markov chain.

    Mathematical Formulation:
    ----------------------
    The correlation matrix ρ(t) switches between K regimes:

    ρ(t) = ρ_k, where k is the current regime

    The regime transitions follow a continuous-time Markov chain with
    transition rate matrix Q = [q_ij], where q_ij is the rate of transition
    from regime i to regime j.

    Attributes:
        correlation_matrices (List[np.ndarray]): List of correlation matrices for each regime
        transition_rates (np.ndarray): Transition rate matrix between regimes
        n_factors (int): Number of factors
        name (str): Name of the correlation structure
    """

    def __init__(
        self,
        correlation_matrices: List[np.ndarray],
        transition_rates: np.ndarray,
        n_factors: int = 3,
        name: str = "RegimeSwitchingCorrelation",
    ):
        """
        Initialize regime-switching correlation.

        Args:
            correlation_matrices: List of correlation matrices for each regime
            transition_rates: Transition rate matrix between regimes
            n_factors: Number of factors
            name: Name of the correlation structure

        Raises:
            CorrelationValidationError: If initialization fails
        """
        self.correlation_matrices = correlation_matrices
        self.transition_rates = transition_rates
        super().__init__(n_factors=n_factors, name=name)
        self._validate_initialization() 