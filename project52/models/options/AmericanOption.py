from __future__ import annotations
import math
import logging
from typing import Literal

logger = logging.getLogger(__name__)

class AmericanOption:
    """
    Class for pricing American-style call and put options using the Cox-Ross-Rubinstein (CRR) binomial model.
    """

    def __init__(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal['call', 'put'],
        q: float = 0.0
    ) -> None:
        """
        Initialize an American Option.

        Parameters:
        - S0: Initial stock price (must be > 0)
        - K: Strike price (must be > 0)
        - T: Time to maturity in years (must be > 0)
        - r: Risk-free interest rate (annual, continuous compounding)
        - sigma: Volatility (annual, standard deviation, >= 0)
        - option_type: 'call' or 'put'
        - q: Continuous dividend yield (default 0)
        """
        if S0 <= 0 or K <= 0 or T <= 0 or sigma < 0:
            raise ValueError("S0, K, and T must be positive; sigma must be non-negative.")
        if option_type not in ('call', 'put'):
            raise ValueError("option_type must be 'call' or 'put'")

        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type
        self.q = q

    def payoff(self, S: float) -> float:
        """
        Option payoff at asset price S.

        For a call: max(S - K, 0)
        For a put:  max(K - S, 0)
        """
        if self.option_type == 'call':
            return max(S - self.K, 0.0)
        return max(self.K - S, 0.0)

    def price_binomial(self, steps: int = 1000) -> float:
        """
        Price the American option using a CRR binomial tree with given steps.

        Parameters:
        - steps: Number of binomial steps (integer > 0)

        Returns:
        - option price (float)
        """
        if steps <= 0:
            raise ValueError("Number of steps must be a positive integer.")

        dt = self.T / steps
        u = math.exp(self.sigma * math.sqrt(dt))
        d = 1 / u
        disc = math.exp(-self.r * dt)
        p = (math.exp((self.r - self.q) * dt) - d) / (u - d)

        # initialize asset prices at maturity
        asset_prices = [self.S0 * (u ** j) * (d ** (steps - j)) for j in range(steps + 1)]
        # option values at maturity
        option_values = [self.payoff(S) for S in asset_prices]

        # backward induction
        for i in range(steps - 1, -1, -1):
            for j in range(i + 1):
                continuation = disc * (p * option_values[j + 1] + (1 - p) * option_values[j])
                exercise = self.payoff(self.S0 * (u ** j) * (d ** (i - j)))
                option_values[j] = max(continuation, exercise)

        price = option_values[0]
        logger.debug(f"Binomial price with {steps} steps: {price}")
        return price

    def __repr__(self) -> str:
        return (
            f"<AmericanOption(type={self.option_type}, S0={self.S0}, K={self.K}, "
            f"T={self.T}, r={self.r}, sigma={self.sigma}, q={self.q})>"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example usage
    opt = AmericanOption(S0=100, K=100, T=1, r=0.05, sigma=0.2, option_type='put', q=0.02)
    price = opt.price_binomial(steps=500)
    print(f"American {opt.option_type.capitalize()} Option Price: {price:.4f}")
