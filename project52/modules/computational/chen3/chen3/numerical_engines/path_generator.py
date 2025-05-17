"""
Path Generation for the Chen3 Model

This module provides path generation utilities for the Chen3 model,
including efficient path simulation and interpolation.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from ..utils.exceptions import NumericalError
from ..utils.logging_config import logger
from .random import RandomNumberGenerator
from .integration import IntegrationScheme

class PathGenerator:
    """
    Path generator for the Chen3 model.
    
    This class provides efficient path generation for the three-factor model,
    including support for various numerical schemes and interpolation methods.
    
    Features:
    1. Efficient path generation
    2. Support for various numerical schemes
    3. Path interpolation
    4. Adaptive time stepping
    5. Error control
    
    Mathematical Formulation:
    ----------------------
    The three-factor model paths are generated using:
    
    1. Interest Rate (CIR):
       dr_t = κ(θ - r_t)dt + σ√r_t dW^r_t
       
    2. Equity Price:
       dS_t = (r_t - q)S_t dt + √v_t S_t dW^S_t
       
    3. Variance (Heston):
       dv_t = κ_v(θ_v - v_t)dt + σ_v√v_t dW^v_t
    
    The paths can be generated using different numerical schemes:
    1. Euler-Maruyama
    2. Milstein
    3. Predictor-Corrector
    4. Adaptive schemes
    
    Attributes:
        rng (RandomNumberGenerator): Random number generator
        integration_scheme (IntegrationScheme): Numerical integration scheme
        use_adaptive (bool): Whether to use adaptive time stepping
        error_tol (float): Error tolerance for adaptive stepping
        min_dt (float): Minimum time step
        max_dt (float): Maximum time step
    """
    
    def __init__(
        self,
        rng: Optional[RandomNumberGenerator] = None,
        integration_scheme: Optional[IntegrationScheme] = None,
        use_adaptive: bool = False,
        error_tol: float = 1e-4,
        min_dt: float = 1e-4,
        max_dt: float = 0.1
    ):
        """
        Initialize the path generator.
        
        Args:
            rng: Random number generator
            integration_scheme: Numerical integration scheme
            use_adaptive: Whether to use adaptive time stepping
            error_tol: Error tolerance for adaptive stepping
            min_dt: Minimum time step
            max_dt: Maximum time step
        """
        self.rng = rng or RandomNumberGenerator()
        self.integration_scheme = integration_scheme or IntegrationScheme()
        self.use_adaptive = use_adaptive
        self.error_tol = error_tol
        self.min_dt = min_dt
        self.max_dt = max_dt
        
        self._validate_initialization()
    
    def _validate_initialization(self):
        """Validate initialization parameters."""
        if self.error_tol <= 0:
            raise NumericalError("Error tolerance must be positive")
        if self.min_dt <= 0 or self.min_dt >= self.max_dt:
            raise NumericalError("Invalid time step bounds")
    
    def generate_paths(
        self,
        initial_state: Dict[str, float],
        drift_function: Callable,
        diffusion_function: Callable,
        correlation_matrix: np.ndarray,
        n_paths: int,
        n_steps: int,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate paths for the three-factor model.
        
        Args:
            initial_state: Initial state variables
            drift_function: Function computing drift terms
            diffusion_function: Function computing diffusion terms
            correlation_matrix: Correlation matrix between factors
            n_paths: Number of paths
            n_steps: Number of time steps
            dt: Time step size
            
        Returns:
            Tuple of (rate_paths, equity_paths, variance_paths)
            
        Raises:
            NumericalError: If path generation fails
        """
        try:
            if self.use_adaptive:
                return self._generate_adaptive_paths(
                    initial_state,
                    drift_function,
                    diffusion_function,
                    correlation_matrix,
                    n_paths,
                    n_steps,
                    dt
                )
            else:
                return self._generate_fixed_paths(
                    initial_state,
                    drift_function,
                    diffusion_function,
                    correlation_matrix,
                    n_paths,
                    n_steps,
                    dt
                )
        except Exception as e:
            raise NumericalError(f"Path generation failed: {str(e)}")
    
    def _generate_fixed_paths(
        self,
        initial_state: Dict[str, float],
        drift_function: Callable,
        diffusion_function: Callable,
        correlation_matrix: np.ndarray,
        n_paths: int,
        n_steps: int,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate paths with fixed time steps."""
        # Generate random numbers
        dW = self.rng.generate_normal(
            size=(n_paths, n_steps, 3),
            correlation_matrix=correlation_matrix
        )
        
        # Initialize paths
        rate_paths = np.zeros((n_paths, n_steps + 1))
        equity_paths = np.zeros((n_paths, n_steps + 1))
        variance_paths = np.zeros((n_paths, n_steps + 1))
        
        # Set initial values
        rate_paths[:, 0] = initial_state['r']
        equity_paths[:, 0] = initial_state['S']
        variance_paths[:, 0] = initial_state['v']
        
        # Generate paths
        for t in range(n_steps):
            # Current state
            state = {
                'r': rate_paths[:, t],
                'S': equity_paths[:, t],
                'v': variance_paths[:, t]
            }
            
            # Compute drift and diffusion
            drift = drift_function(state, t * dt)
            diffusion = diffusion_function(state, t * dt)
            
            # Update paths using integration scheme
            rate_paths[:, t+1], equity_paths[:, t+1], variance_paths[:, t+1] = (
                self.integration_scheme.step(
                    state,
                    drift,
                    diffusion,
                    dW[:, t],
                    dt
                )
            )
        
        return rate_paths, equity_paths, variance_paths
    
    def _generate_adaptive_paths(
        self,
        initial_state: Dict[str, float],
        drift_function: Callable,
        diffusion_function: Callable,
        correlation_matrix: np.ndarray,
        n_paths: int,
        n_steps: int,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate paths with adaptive time steps."""
        # Initialize paths
        rate_paths = np.zeros((n_paths, n_steps + 1))
        equity_paths = np.zeros((n_paths, n_steps + 1))
        variance_paths = np.zeros((n_paths, n_steps + 1))
        
        # Set initial values
        rate_paths[:, 0] = initial_state['r']
        equity_paths[:, 0] = initial_state['S']
        variance_paths[:, 0] = initial_state['v']
        
        # Generate paths with adaptive stepping
        t = 0
        step = 0
        while step < n_steps:
            # Current state
            state = {
                'r': rate_paths[:, step],
                'S': equity_paths[:, step],
                'v': variance_paths[:, step]
            }
            
            # Compute drift and diffusion
            drift = drift_function(state, t)
            diffusion = diffusion_function(state, t)
            
            # Generate random numbers
            dW = self.rng.generate_normal(
                size=(n_paths, 3),
                correlation_matrix=correlation_matrix
            )
            
            # Compute error estimate
            error = self._estimate_error(
                state,
                drift,
                diffusion,
                dW,
                dt
            )
            
            # Adjust time step
            if error > self.error_tol:
                dt = max(self.min_dt, dt / 2)
                continue
            
            # Update paths
            rate_paths[:, step+1], equity_paths[:, step+1], variance_paths[:, step+1] = (
                self.integration_scheme.step(
                    state,
                    drift,
                    diffusion,
                    dW,
                    dt
                )
            )
            
            # Update time and step
            t += dt
            step += 1
            
            # Increase time step if possible
            dt = min(self.max_dt, dt * 1.5)
        
        return rate_paths, equity_paths, variance_paths
    
    def _estimate_error(
        self,
        state: Dict[str, np.ndarray],
        drift: Tuple[np.ndarray, ...],
        diffusion: Tuple[np.ndarray, ...],
        dW: np.ndarray,
        dt: float
    ) -> float:
        """Estimate local truncation error."""
        # Compute two steps with dt/2
        state_half = self.integration_scheme.step(
            state,
            drift,
            diffusion,
            dW,
            dt/2
        )
        
        # Compute one step with dt
        state_full = self.integration_scheme.step(
            state,
            drift,
            diffusion,
            dW,
            dt
        )
        
        # Compute error estimate
        error = np.max(np.abs(
            np.array(state_half) - np.array(state_full)
        ))
        
        return error
    
    def interpolate_paths(
        self,
        paths: Tuple[np.ndarray, ...],
        old_times: np.ndarray,
        new_times: np.ndarray
    ) -> Tuple[np.ndarray, ...]:
        """
        Interpolate paths to new time points.
        
        Args:
            paths: Tuple of path arrays
            old_times: Original time points
            new_times: New time points
            
        Returns:
            Tuple of interpolated path arrays
            
        Raises:
            NumericalError: If interpolation fails
        """
        try:
            interpolated_paths = []
            for path in paths:
                # Use cubic interpolation for smooth paths
                from scipy.interpolate import interp1d
                interpolator = interp1d(
                    old_times,
                    path,
                    kind='cubic',
                    axis=1,
                    bounds_error=False,
                    fill_value=(path[:, 0], path[:, -1])
                )
                interpolated_paths.append(interpolator(new_times))
            
            return tuple(interpolated_paths)
            
        except Exception as e:
            raise NumericalError(f"Path interpolation failed: {str(e)}")
    
    def __str__(self) -> str:
        """String representation of the path generator."""
        return (
            f"{self.__class__.__name__}("
            f"use_adaptive={self.use_adaptive}, "
            f"error_tol={self.error_tol})"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation of the path generator."""
        return (
            f"{self.__class__.__name__}("
            f"rng={self.rng}, "
            f"integration_scheme={self.integration_scheme}, "
            f"use_adaptive={self.use_adaptive}, "
            f"error_tol={self.error_tol}, "
            f"min_dt={self.min_dt}, "
            f"max_dt={self.max_dt})"
        ) 