# File: chen3/model_enhancements/regime_switching.py
"""
Regime-switching model with discrete Markov chain.
"""
import numpy as np
from numpy.typing import NDArray
from typing import List, Dict

class RegimeSwitcher:
    def __init__(
        self,
        transition_matrix: NDArray,
        regime_params: List[Dict],
        initial_regime: int = 0
    ):
        """
        transition_matrix: shape (m, m) Markov chain probabilities
        regime_params: list of parameter dicts per regime
        initial_regime: starting regime index
        """
        self.P = np.array(transition_matrix)
        self.regime_params = regime_params
        self.n_regimes = self.P.shape[0]
        self.current = initial_regime

    def simulate(self, n_steps: int, rng: np.random.Generator) -> NDArray:
        """
        Returns array of regime indices of length n_steps+1.
        """
        path = np.empty(n_steps+1, dtype=int)
        path[0] = self.current
        for t in range(1, n_steps+1):
            probs = self.P[path[t-1]]
            path[t] = rng.choice(self.n_regimes, p=probs)
        return path

    def get_params(self, regime_index: int) -> Dict:
        """Retrieve parameter dict for given regime."""
        return self.regime_params[regime_index]
