"""
Monte Carlo Simulation Engine for the Chen3 Model

This module implements the core Monte Carlo simulation engine for the Chen3 model,
providing efficient path generation and pricing capabilities.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union, Protocol
import numpy as np
from abc import ABC, abstractmethod
from ..utils.exceptions import NumericalError
from ..utils.logging_config import logger
from .random import RandomNumberGenerator
from .path_generator import PathGenerator
from .integration import IntegrationScheme
from .variance_reduction import VarianceReductionTechnique, create_variance_reduction

class MonteCarloEngineBase(ABC):
    """
    Base class for Monte Carlo simulation engines.
    
    This abstract class defines the interface and common functionality
    for all Monte Carlo simulation engines.
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
        variance_reduction: Optional[VarianceReductionTechnique] = None
    ):
        """Initialize the base Monte Carlo engine."""
        self.rng = rng or RandomNumberGenerator()
        self.path_generator = path_generator or PathGenerator()
        self.integration_scheme = integration_scheme or IntegrationScheme()
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.dt = dt
        self.use_antithetic = use_antithetic
        self.use_control_variate = use_control_variate
        self.variance_reduction = variance_reduction
        
        self._validate_initialization()
    
    def _validate_initialization(self):
        """Validate initialization parameters."""
        if self.n_paths < 1000:
            raise NumericalError("Number of paths must be at least 1000")
        if self.n_steps < 50:
            raise NumericalError("Number of time steps must be at least 50")
        if self.dt <= 0 or self.dt > 0.1:
            raise NumericalError("Time step must be in (0, 0.1]")
    
    @abstractmethod
    def simulate_paths(
        self,
        initial_state: Dict[str, float],
        drift_function: Callable,
        diffusion_function: Callable,
        correlation_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate paths for the three-factor model."""
        pass
    
    @abstractmethod
    def compute_expectation(
        self,
        payoff_function: Callable,
        paths: Tuple[np.ndarray, ...],
        control_variate: Optional[Callable] = None
    ) -> Tuple[float, float]:
        """Compute expectation of a payoff function."""
        pass
    
    def apply_variance_reduction(
        self,
        payoffs: np.ndarray,
        control_values: Optional[np.ndarray] = None,
        control_expectation: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Apply variance reduction techniques.
        
        Args:
            payoffs: Original payoff values
            control_values: Optional control variate values
            control_expectation: Optional control variate expectation
            
        Returns:
            Tuple of (adjusted_price, standard_error)
        """
        if self.variance_reduction is None:
            return np.mean(payoffs), np.std(payoffs) / np.sqrt(len(payoffs))
        
        result = self.variance_reduction.apply(
            payoffs=payoffs,
            control_values=control_values,
            control_expectation=control_expectation
        )
        
        return result.estimate, result.standard_error

class MonteCarloEngine(MonteCarloEngineBase):
    """
    CPU-based Monte Carlo simulation engine for the Chen3 model.
    
    This class provides the core functionality for Monte Carlo simulation,
    including path generation, payoff evaluation, and statistical analysis.
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
        use_parallel: bool = True,
        n_jobs: int = -1,
        variance_reduction: Optional[VarianceReductionTechnique] = None
    ):
        """Initialize the CPU Monte Carlo engine."""
        super().__init__(
            rng=rng,
            path_generator=path_generator,
            integration_scheme=integration_scheme,
            n_paths=n_paths,
            n_steps=n_steps,
            dt=dt,
            use_antithetic=use_antithetic,
            use_control_variate=use_control_variate,
            variance_reduction=variance_reduction
        )
        self.use_parallel = use_parallel
        self.n_jobs = n_jobs
        
        if self.use_parallel and self.n_jobs < -1:
            raise NumericalError("Invalid number of parallel jobs")
    
    def simulate_paths(
        self,
        initial_state: Dict[str, float],
        drift_function: Callable,
        diffusion_function: Callable,
        correlation_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate paths for the three-factor model using CPU.
        
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
            # Generate random numbers
            n_total = self.n_paths * (2 if self.use_antithetic else 1)
            dW = self.rng.generate_normal(
                size=(n_total, self.n_steps, 3),
                correlation_matrix=correlation_matrix
            )
            
            # Initialize paths
            rate_paths = np.zeros((n_total, self.n_steps + 1))
            equity_paths = np.zeros((n_total, self.n_steps + 1))
            variance_paths = np.zeros((n_total, self.n_steps + 1))
            
            # Set initial values
            rate_paths[:, 0] = initial_state['r']
            equity_paths[:, 0] = initial_state['S']
            variance_paths[:, 0] = initial_state['v']
            
            # Simulate paths
            for t in range(self.n_steps):
                # Current state
                state = {
                    'r': rate_paths[:, t],
                    'S': equity_paths[:, t],
                    'v': variance_paths[:, t]
                }
                
                # Compute drift and diffusion
                drift = drift_function(state, t * self.dt)
                diffusion = diffusion_function(state, t * self.dt)
                
                # Update paths
                rate_paths[:, t+1] = rate_paths[:, t] + drift[0] * self.dt + diffusion[0] * dW[:, t, 0]
                equity_paths[:, t+1] = equity_paths[:, t] * np.exp(
                    (drift[1] - 0.5 * diffusion[1]**2) * self.dt + diffusion[1] * dW[:, t, 1]
                )
                variance_paths[:, t+1] = np.maximum(
                    variance_paths[:, t] + drift[2] * self.dt + diffusion[2] * dW[:, t, 2],
                    0.0
                )
            
            # Apply antithetic variates if enabled
            if self.use_antithetic:
                rate_paths = (rate_paths[:self.n_paths] + rate_paths[self.n_paths:]) / 2
                equity_paths = (equity_paths[:self.n_paths] + equity_paths[self.n_paths:]) / 2
                variance_paths = (variance_paths[:self.n_paths] + variance_paths[self.n_paths:]) / 2
            
            return rate_paths, equity_paths, variance_paths
            
        except Exception as e:
            raise NumericalError(f"Path simulation failed: {str(e)}")
    
    def compute_expectation(
        self,
        payoff_function: Callable,
        paths: Tuple[np.ndarray, ...],
        control_variate: Optional[Callable] = None
    ) -> Tuple[float, float]:
        """
        Compute expectation of a payoff function using CPU.
        
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
                control_values = control_variate(*paths) if control_variate is not None else None
                control_expectation = np.mean(control_values) if control_values is not None else None
                return self.apply_variance_reduction(
                    payoffs=payoffs,
                    control_values=control_values,
                    control_expectation=control_expectation
                )
            
            # Compute statistics
            price = np.mean(payoffs)
            std_error = np.std(payoffs) / np.sqrt(len(payoffs))
            
            return price, std_error
            
        except Exception as e:
            raise NumericalError(f"Expectation computation failed: {str(e)}")
    
    def __str__(self) -> str:
        """String representation of the Monte Carlo engine."""
        return (
            f"MonteCarloEngine(n_paths={self.n_paths}, n_steps={self.n_steps}, "
            f"dt={self.dt}, use_antithetic={self.use_antithetic}, "
            f"use_control_variate={self.use_control_variate}, "
            f"use_parallel={self.use_parallel}, n_jobs={self.n_jobs})"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation of the Monte Carlo engine."""
        return self.__str__() 