"""
Vanilla options pricing module.

This module provides classes for pricing vanilla options such as
European calls and puts using the Chen model.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from pydantic import BaseModel, Field, field_validator

@dataclass
class OptionParams:
    """Parameters for an option."""
    strike: float
    maturity: float
    is_call: bool = True

    def __post_init__(self):
        """Validate option parameters."""
        if self.strike <= 0:
            raise ValueError("Strike price must be positive")
        if self.maturity <= 0:
            raise ValueError("Maturity must be positive")

class Option(ABC):
    """Base class for options."""

    def __init__(self, params: OptionParams):
        """
        Initialize the option.

        Args:
            params: Option parameters
        """
        self.params = params

    @abstractmethod
    def payoff(self, r_paths: np.ndarray, S_paths: np.ndarray, v_paths: np.ndarray) -> np.ndarray:
        """
        Compute the payoff of the option.

        Args:
            r_paths: Array of interest rate paths
            S_paths: Array of underlying asset price paths
            v_paths: Array of variance paths

        Returns:
            Array of payoffs
        """
        pass

class EuropeanCall(Option):
    """European call option."""

    def __init__(self, strike: float, maturity: float):
        """
        Initialize the European call option.

        Args:
            strike: Strike price
            maturity: Time to maturity
        """
        super().__init__(OptionParams(strike=strike, maturity=maturity, is_call=True))

    def payoff(self, r_paths: np.ndarray, S_paths: np.ndarray, v_paths: np.ndarray) -> np.ndarray:
        """
        Compute the payoff of the European call option.

        Args:
            r_paths: Array of interest rate paths
            S_paths: Array of underlying asset price paths
            v_paths: Array of variance paths (unused)

        Returns:
            Array of discounted payoffs
        """
        # Calculate undiscounted payoff
        payoff = np.maximum(S_paths[:, -1] - self.params.strike, 0)
        
        # Calculate discount factor using instantaneous rate
        dt = self.params.maturity / (r_paths.shape[1] - 1)
        discount = np.exp(-np.sum(r_paths * dt, axis=1))
        
        return payoff * discount

class EuropeanPut(Option):
    """European put option."""

    def __init__(self, strike: float, maturity: float):
        """
        Initialize the European put option.

        Args:
            strike: Strike price
            maturity: Time to maturity
        """
        super().__init__(OptionParams(strike=strike, maturity=maturity, is_call=False))

    def payoff(self, r_paths: np.ndarray, S_paths: np.ndarray, v_paths: np.ndarray) -> np.ndarray:
        """
        Compute the payoff of the European put option.

        Args:
            r_paths: Array of interest rate paths
            S_paths: Array of underlying asset price paths
            v_paths: Array of variance paths (unused)

        Returns:
            Array of discounted payoffs
        """
        # Calculate undiscounted payoff
        payoff = np.maximum(self.params.strike - S_paths[:, -1], 0)
        
        # Calculate discount factor using instantaneous rate
        dt = self.params.maturity / (r_paths.shape[1] - 1)
        discount = np.exp(-np.sum(r_paths * dt, axis=1))
        
        return payoff * discount

class VanillaOption(BaseModel):
    """Pydantic model for vanilla option parameters."""
    strike: float = Field(..., gt=0, description="Strike price")
    maturity: float = Field(..., gt=0, description="Time to maturity")
    is_call: bool = Field(True, description="Whether the option is a call")

    @field_validator('strike')
    def validate_strike(cls, v):
        """Validate strike price."""
        if v <= 0:
            raise ValueError("Strike price must be positive")
        return v

    @field_validator('maturity')
    def validate_maturity(cls, v):
        """Validate maturity."""
        if v <= 0:
            raise ValueError("Maturity must be positive")
        return v

    def create_option(self) -> Option:
        """
        Create an option instance based on parameters.

        Returns:
            Option instance (EuropeanCall or EuropeanPut)
        """
        if self.is_call:
            return EuropeanCall(self.strike, self.maturity)
        else:
            return EuropeanPut(self.strike, self.maturity)

    def price(
        self,
        model: Any,
        n_paths: int = 10000,
        n_steps: int = 100,
        dt: float = 0.01,
        n_runs: int = 5
    ) -> Tuple[float, float]:
        """
        Price the option using Monte Carlo simulation.

        Args:
            model: Chen model instance
            n_paths: Number of simulation paths
            n_steps: Number of time steps
            dt: Time step size
            n_runs: Number of independent simulation runs for error estimation

        Returns:
            Tuple of (price, standard error)
        """
        option = self.create_option()
        
        # Run multiple independent simulations
        prices = []
        for _ in range(n_runs):
            price = model.price(option, n_paths=n_paths, n_steps=n_steps, dt=dt)
            prices.append(price)
        
        # Calculate mean price and standard error
        mean_price = np.mean(prices)
        std_error = np.std(prices) / np.sqrt(n_runs)
        
        return mean_price, std_error 