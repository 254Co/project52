# File: chen3/model_enhancements/regime_switching.py
"""
Regime-switching model enhancement for the Chen3 model.

This module implements a regime-switching mechanism using a discrete Markov chain
to model market regimes and their transitions. This allows for capturing
different market states and their evolution over time.

The implementation provides:
- Discrete Markov chain for regime transitions
- Regime-specific parameter sets
- Simulation of regime paths
- Parameter retrieval for each regime

This enhancement is particularly useful for modeling:
- Market regime changes
- Parameter shifts between regimes
- Time-varying market dynamics
- State-dependent model behavior
"""

from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray


class RegimeSwitcher:
    """
    Regime-switching mechanism using a discrete Markov chain.

    This class implements a regime-switching mechanism where the model
    parameters can change based on the current market regime, which
    evolves according to a discrete Markov chain.

    Mathematical Formulation:
    ----------------------
    The regime transitions follow a discrete Markov chain with:
    - Transition matrix P where P[i,j] = P(regime j | regime i)
    - Each regime has its own set of model parameters
    - Regime evolution is simulated using the transition probabilities

    Attributes:
        P (NDArray): Transition probability matrix of shape (n_regimes, n_regimes)
        regime_params (List[Dict]): List of parameter dictionaries for each regime
        n_regimes (int): Number of possible regimes
        current (int): Current regime index

    Note:
        The transition matrix P must be a valid stochastic matrix:
        - All elements in [0, 1]
        - Each row sums to 1
    """

    def __init__(
        self,
        transition_matrix: NDArray,
        regime_params: List[Dict],
        initial_regime: int = 0,
    ):
        """
        Initialize the regime-switching mechanism.

        Args:
            transition_matrix (NDArray): Transition probability matrix of shape
                (n_regimes, n_regimes) where P[i,j] is the probability of
                transitioning from regime i to regime j
            regime_params (List[Dict]): List of parameter dictionaries, one for
                each regime. Each dictionary contains the model parameters
                specific to that regime
            initial_regime (int): Index of the initial regime (default: 0)

        Raises:
            ValueError: If transition matrix is invalid or number of regimes
                doesn't match number of parameter sets
        """
        self.P = np.array(transition_matrix)
        self.regime_params = regime_params
        self.n_regimes = self.P.shape[0]
        self.current = initial_regime

        # Validate transition matrix
        if not np.allclose(self.P.sum(axis=1), 1.0):
            raise ValueError("Transition matrix rows must sum to 1")
        if not np.all((self.P >= 0) & (self.P <= 1)):
            raise ValueError("Transition probabilities must be in [0, 1]")

        # Validate number of regimes matches parameter sets
        if len(regime_params) != self.n_regimes:
            raise ValueError(
                f"Number of regime parameter sets ({len(regime_params)}) "
                f"must match number of regimes ({self.n_regimes})"
            )

    def simulate(self, n_steps: int, rng: np.random.Generator) -> NDArray:
        """
        Simulate a path of regime indices.

        This method simulates the evolution of regimes over time using
        the transition probability matrix and the provided random number
        generator.

        Args:
            n_steps (int): Number of time steps to simulate
            rng (np.random.Generator): Random number generator to use
                for regime transitions

        Returns:
            NDArray: Array of regime indices of length n_steps+1,
                where the first element is the current regime

        Note:
            The simulation uses the transition probabilities from the
            current regime to determine the next regime at each step.
        """
        path = np.empty(n_steps + 1, dtype=int)
        path[0] = self.current
        for t in range(1, n_steps + 1):
            probs = self.P[path[t - 1]]
            path[t] = rng.choice(self.n_regimes, p=probs)
        return path

    def get_params(self, regime_index: int) -> Dict:
        """
        Retrieve the parameter set for a specific regime.

        Args:
            regime_index (int): Index of the regime whose parameters
                to retrieve

        Returns:
            Dict: Dictionary of parameters specific to the requested regime

        Raises:
            IndexError: If regime_index is out of bounds
        """
        if not 0 <= regime_index < self.n_regimes:
            raise IndexError(f"Regime index {regime_index} out of bounds")
        return self.regime_params[regime_index]
