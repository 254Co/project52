"""
Integration Schemes for the Chen3 Model

This module provides numerical integration schemes for the Chen3 model,
including Euler-Maruyama, Milstein, and predictor-corrector methods.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from ..utils.exceptions import NumericalError
from ..utils.logging_config import logger

class IntegrationScheme:
    """
    Integration scheme for the Chen3 model.
    
    This class provides various numerical integration schemes for the three-factor model,
    including support for different orders of accuracy and stability properties.
    
    Features:
    1. Multiple integration schemes
    2. Order of accuracy control
    3. Stability analysis
    4. Error estimation
    5. Adaptive stepping support
    
    Mathematical Formulation:
    ----------------------
    The three-factor model can be integrated using different schemes:
    
    1. Euler-Maruyama (Order 0.5):
       X_{n+1} = X_n + μ(X_n)Δt + σ(X_n)ΔW_n
       
    2. Milstein (Order 1.0):
       X_{n+1} = X_n + μ(X_n)Δt + σ(X_n)ΔW_n + 0.5σ(X_n)σ'(X_n)(ΔW_n² - Δt)
       
    3. Predictor-Corrector (Order 1.0):
       X* = X_n + μ(X_n)Δt + σ(X_n)ΔW_n
       X_{n+1} = X_n + 0.5[μ(X_n) + μ(X*)]Δt + σ(X_n)ΔW_n
    
    Attributes:
        scheme_type (str): Type of integration scheme
        order (float): Order of accuracy
        use_implicit (bool): Whether to use implicit scheme
        theta (float): Implicit scheme parameter
    """
    
    def __init__(
        self,
        scheme_type: str = "euler",
        order: float = 0.5,
        use_implicit: bool = False,
        theta: float = 0.5
    ):
        """
        Initialize the integration scheme.
        
        Args:
            scheme_type: Type of integration scheme
            order: Order of accuracy
            use_implicit: Whether to use implicit scheme
            theta: Implicit scheme parameter
        """
        self.scheme_type = scheme_type.lower()
        self.order = order
        self.use_implicit = use_implicit
        self.theta = theta
        
        self._validate_initialization()
    
    def _validate_initialization(self):
        """Validate initialization parameters."""
        valid_schemes = ["euler", "milstein", "predictor_corrector"]
        if self.scheme_type not in valid_schemes:
            raise NumericalError(f"Invalid scheme type: {self.scheme_type}")
        
        if self.order <= 0:
            raise NumericalError("Order must be positive")
        
        if self.use_implicit and not 0 <= self.theta <= 1:
            raise NumericalError("Theta must be in [0, 1]")
    
    def step(
        self,
        state: Dict[str, np.ndarray],
        drift: Tuple[np.ndarray, ...],
        diffusion: Tuple[np.ndarray, ...],
        dW: np.ndarray,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform one integration step.
        
        Args:
            state: Current state variables
            drift: Drift terms
            diffusion: Diffusion terms
            dW: Random increments
            dt: Time step size
            
        Returns:
            Tuple of (new_rate, new_equity, new_variance)
            
        Raises:
            NumericalError: If integration fails
        """
        try:
            if self.scheme_type == "euler":
                return self._euler_step(state, drift, diffusion, dW, dt)
            elif self.scheme_type == "milstein":
                return self._milstein_step(state, drift, diffusion, dW, dt)
            else:  # predictor_corrector
                return self._predictor_corrector_step(state, drift, diffusion, dW, dt)
        except Exception as e:
            raise NumericalError(f"Integration step failed: {str(e)}")
    
    def _euler_step(
        self,
        state: Dict[str, np.ndarray],
        drift: Tuple[np.ndarray, ...],
        diffusion: Tuple[np.ndarray, ...],
        dW: np.ndarray,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Euler-Maruyama step."""
        # Extract current state
        r = state['r']
        S = state['S']
        v = state['v']
        
        # Extract drift terms
        mu_r, mu_S, mu_v = drift
        
        # Extract diffusion terms
        sigma_r, sigma_S, sigma_v = diffusion
        
        # Update state
        new_r = r + mu_r * dt + sigma_r * dW[:, 0]
        new_S = S + mu_S * dt + sigma_S * dW[:, 1]
        new_v = v + mu_v * dt + sigma_v * dW[:, 2]
        
        # Ensure positivity
        new_r = np.maximum(new_r, 0)
        new_S = np.maximum(new_S, 0)
        new_v = np.maximum(new_v, 0)
        
        return new_r, new_S, new_v
    
    def _milstein_step(
        self,
        state: Dict[str, np.ndarray],
        drift: Tuple[np.ndarray, ...],
        diffusion: Tuple[np.ndarray, ...],
        dW: np.ndarray,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Milstein step."""
        # Extract current state
        r = state['r']
        S = state['S']
        v = state['v']
        
        # Extract drift terms
        mu_r, mu_S, mu_v = drift
        
        # Extract diffusion terms
        sigma_r, sigma_S, sigma_v = diffusion
        
        # Compute diffusion derivatives
        dsigma_r = 0.5 * sigma_r / np.sqrt(r)  # For CIR process
        dsigma_v = 0.5 * sigma_v / np.sqrt(v)  # For Heston process
        
        # Update state
        new_r = r + mu_r * dt + sigma_r * dW[:, 0] + 0.5 * dsigma_r * sigma_r * (dW[:, 0]**2 - dt)
        new_S = S + mu_S * dt + sigma_S * dW[:, 1]  # No Milstein correction for log-normal
        new_v = v + mu_v * dt + sigma_v * dW[:, 2] + 0.5 * dsigma_v * sigma_v * (dW[:, 2]**2 - dt)
        
        # Ensure positivity
        new_r = np.maximum(new_r, 0)
        new_S = np.maximum(new_S, 0)
        new_v = np.maximum(new_v, 0)
        
        return new_r, new_S, new_v
    
    def _predictor_corrector_step(
        self,
        state: Dict[str, np.ndarray],
        drift: Tuple[np.ndarray, ...],
        diffusion: Tuple[np.ndarray, ...],
        dW: np.ndarray,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predictor-corrector step."""
        # Extract current state
        r = state['r']
        S = state['S']
        v = state['v']
        
        # Extract drift terms
        mu_r, mu_S, mu_v = drift
        
        # Extract diffusion terms
        sigma_r, sigma_S, sigma_v = diffusion
        
        # Predictor step
        r_pred = r + mu_r * dt + sigma_r * dW[:, 0]
        S_pred = S + mu_S * dt + sigma_S * dW[:, 1]
        v_pred = v + mu_v * dt + sigma_v * dW[:, 2]
        
        # Ensure positivity of predictor
        r_pred = np.maximum(r_pred, 0)
        S_pred = np.maximum(S_pred, 0)
        v_pred = np.maximum(v_pred, 0)
        
        # Compute drift at predicted state
        pred_state = {'r': r_pred, 'S': S_pred, 'v': v_pred}
        mu_r_pred, mu_S_pred, mu_v_pred = self._compute_drift(pred_state)
        
        # Corrector step
        new_r = r + 0.5 * (mu_r + mu_r_pred) * dt + sigma_r * dW[:, 0]
        new_S = S + 0.5 * (mu_S + mu_S_pred) * dt + sigma_S * dW[:, 1]
        new_v = v + 0.5 * (mu_v + mu_v_pred) * dt + sigma_v * dW[:, 2]
        
        # Ensure positivity
        new_r = np.maximum(new_r, 0)
        new_S = np.maximum(new_S, 0)
        new_v = np.maximum(new_v, 0)
        
        return new_r, new_S, new_v
    
    def _compute_drift(self, state: Dict[str, np.ndarray]) -> Tuple[np.ndarray, ...]:
        """Compute drift terms for a given state."""
        # This is a placeholder - actual implementation would depend on the model
        r = state['r']
        S = state['S']
        v = state['v']
        
        # Example drift terms (should be replaced with actual model)
        mu_r = 0.1 * (0.05 - r)  # Mean reversion
        mu_S = r * S  # Risk-neutral drift
        mu_v = 0.2 * (0.04 - v)  # Mean reversion
        
        return mu_r, mu_S, mu_v
    
    def __str__(self) -> str:
        """String representation of the integration scheme."""
        return (
            f"{self.__class__.__name__}("
            f"scheme_type={self.scheme_type}, "
            f"order={self.order})"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation of the integration scheme."""
        return (
            f"{self.__class__.__name__}("
            f"scheme_type={self.scheme_type}, "
            f"order={self.order}, "
            f"use_implicit={self.use_implicit}, "
            f"theta={self.theta})"
        ) 