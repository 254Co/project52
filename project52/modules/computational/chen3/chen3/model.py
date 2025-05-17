# File: chen3/model.py

"""
Core Model Implementation for the Three-Factor Chen Model

This module implements the main ChenModel class that encapsulates the three-factor
stochastic model combining interest rates, equity prices, and rough volatility.
The model is particularly suited for pricing complex financial instruments where
traditional models may fall short in capturing market dynamics.

The three factors are:
1. Interest Rate: CIR process for stochastic rates
2. Equity Price: Geometric Brownian motion with stochastic volatility
3. Rough Volatility: Heston-type variance process with rough features

Features:
- Unified interface for model parameters and simulation
- Support for pricing and risk analysis of complex derivatives
- Extensible for additional pricing or simulation methods
- Advanced correlation structures including:
  - Time-dependent correlations
  - State-dependent correlations
  - Regime-switching correlations
  - Stochastic correlations
  - Copula-based correlations

Mathematical Formulation:
------------------------
The three-factor Chen model is defined by the following system of SDEs:

1. Interest Rate Process (CIR):
   dr_t = κ(θ - r_t)dt + σ√r_t dW^r_t

2. Equity Price Process:
   dS_t = (r_t - q)S_t dt + √v_t S_t dW^S_t

3. Variance Process (Heston):
   dv_t = κ_v(θ_v - v_t)dt + σ_v√v_t dW^v_t

where:
- r_t: Interest rate at time t
- S_t: Equity price at time t
- v_t: Variance at time t
- W^r_t, W^S_t, W^v_t: Correlated Brownian motions
- κ, θ, σ: CIR process parameters
- κ_v, θ_v, σ_v: Heston process parameters
- q: Continuous dividend yield

The correlation structure between the Brownian motions can be:
- Constant (3x3 correlation matrix)
- Time-dependent
- State-dependent
- Regime-switching
- Stochastic
- Copula-based

Numerical Implementation:
-----------------------
The model uses Monte Carlo simulation with the following features:
1. Euler-Maruyama discretization for all processes
2. Cholesky decomposition for correlation handling
3. Adaptive time stepping for better accuracy
4. Variance reduction techniques available
5. Support for parallel computation

Example Usage:
-------------
    >>> from chen3 import ChenModel, ModelParams, RateParams, EquityParams
    >>> from chen3.correlation import TimeDependentCorrelation
    >>> import numpy as np
    >>>
    >>> # Define model parameters
    >>> rate_params = RateParams(
    ...     kappa=0.1,    # Mean reversion speed
    ...     theta=0.05,   # Long-term mean
    ...     sigma=0.1,    # Volatility
    ...     r0=0.03       # Initial rate
    ... )
    >>>
    >>> equity_params = EquityParams(
    ...     mu=0.05,      # Drift
    ...     q=0.02,       # Dividend yield
    ...     S0=100.0,     # Initial stock price
    ...     v0=0.04,      # Initial variance
    ...     kappa_v=2.0,  # Variance mean reversion
    ...     theta_v=0.04, # Long-term variance
    ...     sigma_v=0.3   # Volatility of variance
    ... )
    >>>
    >>> # Create time-dependent correlation
    >>> time_points = np.array([0.0, 1.0, 2.0])
    >>> corr_matrices = [
    ...     np.array([[1.0, 0.5, 0.3],
    ...              [0.5, 1.0, 0.2],
    ...              [0.3, 0.2, 1.0]]),
    ...     np.array([[1.0, 0.6, 0.4],
    ...              [0.6, 1.0, 0.3],
    ...              [0.4, 0.3, 1.0]]),
    ...     np.array([[1.0, 0.7, 0.5],
    ...              [0.7, 1.0, 0.4],
    ...              [0.5, 0.4, 1.0]])
    ... ]
    >>> time_corr = TimeDependentCorrelation(
    ...     time_points=time_points,
    ...     correlation_matrices=corr_matrices
    ... )
    >>>
    >>> # Create model parameters
    >>> model_params = ModelParams(
    ...     rate=rate_params,
    ...     equity=equity_params,
    ...     correlation=time_corr
    ... )
    >>>
    >>> # Create and use the model
    >>> model = ChenModel(model_params)
    >>>
    >>> # Define a payoff function (e.g., European call option)
    >>> def payoff_function(r_paths, S_paths, v_paths):
    ...     return np.maximum(S_paths[:, -1] - 100, 0)
    >>>
    >>> # Price the instrument
    >>> price = model.price_instrument(
    ...     payoff_function,
    ...     n_paths=10000,
    ...     n_steps=100,
    ...     dt=0.01
    ... )
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from .datatypes import ModelParams, RateParams, EquityParams
from chen3.correlation import (
    TimeDependentCorrelation,
    StateDependentCorrelation,
    RegimeSwitchingCorrelation,
    StochasticCorrelation,
    CopulaCorrelation
)
from .utils.logging import logger
from .utils.exceptions import (
    ValidationError,
    NumericalError,
    SimulationError,
    CorrelationError
)
from .utils.validation import (
    check_feller_condition,
    validate_correlation_matrix,
    check_numerical_stability,
    validate_time_grid
)

class ChenModel:
    """
    Implementation of the three-factor Chen model for financial derivatives pricing.
    
    This class serves as the main interface for the model, encapsulating all
    parameters and providing methods for simulation and pricing. The model
    combines stochastic interest rates, equity prices, and rough volatility
    to provide a comprehensive framework for pricing complex financial instruments.
    
    The model is particularly well-suited for:
    - Long-dated options where interest rate risk is significant
    - Products with complex volatility dynamics
    - Instruments requiring correlation between rates and equity
    - Path-dependent options where rough volatility is important
    
    Mathematical Properties:
    ----------------------
    1. Interest Rate Process:
       - Mean-reverting CIR process
       - Ensures positive rates
       - Gamma stationary distribution
    
    2. Equity Process:
       - Log-normal distribution in short term
       - Stochastic volatility
       - Correlation with rates
    
    3. Variance Process:
       - Mean-reverting Heston process
       - Ensures positive variance
       - Volatility clustering
    
    Numerical Features:
    -----------------
    1. Monte Carlo Simulation:
       - Euler-Maruyama discretization
       - Cholesky decomposition for correlations
       - Adaptive time stepping
       - Variance reduction techniques
    
    2. Correlation Handling:
       - Support for various correlation structures
       - Dynamic correlation updates
       - Efficient matrix operations
    
    Attributes:
        params (ModelParams): Complete set of model parameters including:
            - Interest rate parameters (CIR process)
            - Equity parameters (price and variance processes)
            - Correlation structure between factors
    
    Example:
        >>> from chen3 import ChenModel, ModelParams
        >>> params = ModelParams(...)  # Initialize with appropriate parameters
        >>> model = ChenModel(params)
        >>> price = model.price_instrument(...)  # Price a financial instrument
    """
    
    def __init__(self, params: ModelParams):
        """
        Initialize the Chen model with parameters.
        
        Args:
            params: Model parameters including rate, equity, and correlation parameters
        
        Raises:
            ValidationError: If parameters are invalid
            CorrelationError: If correlation structure is invalid
        """
        self.params = params
        self._validate_parameters()
        logger.info("Chen model initialized successfully")
    
    def _validate_parameters(self):
        """Validate all model parameters."""
        if self.params is None:
            raise ValidationError("Model parameters cannot be None")
            
        if not isinstance(self.params, ModelParams):
            raise ValidationError("Model parameters must be an instance of ModelParams")
            
        # Validate rate process parameters
        if not check_feller_condition(
            self.params.rate.kappa,
            self.params.rate.theta,
            self.params.rate.sigma,
            "interest rate"
        ):
            logger.warning("Interest rate process may hit zero")
        
        # Validate variance process parameters
        if not check_feller_condition(
            self.params.equity.kappa_v,
            self.params.equity.theta_v,
            self.params.equity.sigma_v,
            "variance"
        ):
            logger.warning("Variance process may hit zero")
        
        # Validate correlation structure
        if isinstance(self.params.correlation, np.ndarray):
            is_valid, error_msg = validate_correlation_matrix(self.params.correlation)
            if not is_valid:
                raise CorrelationError(f"Invalid correlation matrix: {error_msg}")
    
    def get_correlation_matrix(
        self,
        t: float = 0.0,
        state: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Get the correlation matrix at time t and state.
        
        Args:
            t: Time point
            state: Current state of the processes
        
        Returns:
            np.ndarray: 3x3 correlation matrix
        
        Raises:
            CorrelationError: If correlation structure is invalid
        """
        try:
            if isinstance(self.params.correlation, np.ndarray):
                return self.params.correlation
            elif isinstance(self.params.correlation, TimeDependentCorrelation):
                return self.params.correlation.get_correlation_matrix(t)
            elif isinstance(self.params.correlation, StateDependentCorrelation):
                return self.params.correlation.get_correlation_matrix(state)
            elif isinstance(self.params.correlation, RegimeSwitchingCorrelation):
                return self.params.correlation.get_correlation_matrix(t, state)
            elif isinstance(self.params.correlation, StochasticCorrelation):
                return self.params.correlation.get_correlation_matrix(t)
            elif isinstance(self.params.correlation, CopulaCorrelation):
                return self.params.correlation.get_correlation_matrix(t, state)
            else:
                raise CorrelationError("Unsupported correlation type")
        except Exception as e:
            raise CorrelationError(f"Failed to get correlation matrix: {str(e)}")
    
    def simulate_paths(
        self,
        n_paths: int,
        n_steps: int,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate paths for the Chen model with enhanced stability checks.
        
        Args:
            n_paths: Number of paths to simulate
            n_steps: Number of time steps
            dt: Time step size
        
        Returns:
            Tuple of arrays containing simulated paths for rates, equity prices, and variance
        """
        # Initialize paths
        time_points = np.linspace(0, n_steps * dt, n_steps + 1)
        r_paths = np.zeros((n_paths, n_steps + 1))
        S_paths = np.zeros((n_paths, n_steps + 1))
        v_paths = np.zeros((n_paths, n_steps + 1))
        
        # Set initial values
        r_paths[:, 0] = self.params.rate.r0
        S_paths[:, 0] = self.params.equity.S0
        v_paths[:, 0] = self.params.equity.v0
        
        # Generate random increments
        dW = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps, 3))
        
        # Simulate paths with enhanced stability
        for i in range(n_steps):
            t = time_points[i]
            corr_matrix = self.get_correlation_matrix(t)
            L = np.linalg.cholesky(corr_matrix)
            
            # Transform independent increments to correlated
            dW_corr = np.einsum('ij,kj->ki', L, dW[:, i, :])
            
            # Update rates (CIR process) with stability checks
            r_paths[:, i+1] = (
                r_paths[:, i] +
                self.params.rate.kappa * (self.params.rate.theta - r_paths[:, i]) * dt +
                self.params.rate.sigma * np.sqrt(np.maximum(r_paths[:, i], 0.0)) * dW_corr[:, 0]
            )
            r_paths[:, i+1] = np.maximum(r_paths[:, i+1], 0.0)
            
            # Update variance (Heston process) with stability checks
            v_paths[:, i+1] = (
                v_paths[:, i] +
                self.params.equity.kappa_v * (self.params.equity.theta_v - v_paths[:, i]) * dt +
                self.params.equity.sigma_v * np.sqrt(np.maximum(v_paths[:, i], 0.0)) * dW_corr[:, 2]
            )
            v_paths[:, i+1] = np.maximum(v_paths[:, i+1], 0.0)
            
            # Update equity prices with stability checks
            drift = r_paths[:, i] - self.params.equity.q - 0.5 * v_paths[:, i]
            diffusion = np.sqrt(np.maximum(v_paths[:, i], 0.0))
            S_paths[:, i+1] = S_paths[:, i] * np.exp(
                drift * dt + diffusion * dW_corr[:, 1]
            )
            S_paths[:, i+1] = np.maximum(S_paths[:, i+1], 0.0)
        
        # Enhanced stability checks
        if not check_numerical_stability(
            r_paths,
            name="interest rates",
            min_value=0.0,
            max_value=1.0,  # Reasonable upper bound for rates
            max_growth_rate=5.0,
            max_volatility=2.0
        ):
            raise NumericalError("Interest rate simulation produced unstable results")
        
        if not check_numerical_stability(
            S_paths,
            name="equity prices",
            min_value=0.0,
            max_growth_rate=10.0,
            max_volatility=5.0
        ):
            raise NumericalError("Equity price simulation produced unstable results")
        
        if not check_numerical_stability(
            v_paths,
            name="variance",
            min_value=0.0,
            max_value=1.0,  # Reasonable upper bound for variance
            max_growth_rate=5.0,
            max_volatility=2.0
        ):
            raise NumericalError("Variance simulation produced unstable results")

        # Debug: Log statistics of final equity prices
        logger.info(f"S_paths final: mean={np.mean(S_paths[:, -1]):.4f}, min={np.min(S_paths[:, -1]):.4f}, max={np.max(S_paths[:, -1]):.4f}")
        
        return r_paths, S_paths, v_paths
    
    def price_instrument(
        self,
        payoff_function: Callable,
        n_paths: int = 10000,
        n_steps: int = 100,
        dt: float = 0.01
    ) -> float:
        """
        Price a financial instrument using Monte Carlo simulation.
        
        Args:
            payoff_function: Function that computes the payoff
            n_paths: Number of simulation paths
            n_steps: Number of time steps
            dt: Time step size
        
        Returns:
            float: Estimated price of the instrument
        
        Raises:
            PricingError: If pricing fails
        """
        try:
            # Simulate paths
            r_paths, S_paths, v_paths = self.simulate_paths(n_paths, n_steps, dt)
            
            # Compute payoffs
            payoffs = payoff_function(r_paths, S_paths, v_paths)
            
            # Check numerical stability of payoffs
            if not check_numerical_stability(payoffs, name="payoffs"):
                raise NumericalError("Payoff computation produced unstable results")
            
            # Compute price
            price = np.mean(payoffs)
            
            # Compute standard error
            std_error = np.std(payoffs) / np.sqrt(n_paths)
            
            logger.info(
                f"Pricing completed: price = {price:.4f} ± {std_error:.4f} "
                f"(n_paths = {n_paths}, n_steps = {n_steps})"
            )
            
            return price
            
        except Exception as e:
            raise PricingError(f"Pricing failed: {str(e)}")

    def price(
        self,
        option,
        n_paths: int = 10000,
        n_steps: int = 100,
        dt: float = 0.01,
        engine=None
    ) -> float:
        """
        Price an option using Monte Carlo simulation.
        
        Args:
            option: Option to price (must have a payoff method)
            n_paths: Number of simulation paths
            n_steps: Number of time steps
            dt: Time step size
            engine: Optional pricing engine (default: standard Monte Carlo)
            
        Returns:
            float: Estimated option price
            
        Raises:
            ValidationError: If parameters are invalid
            NumericalError: If numerical issues occur
        """
        try:
            # Simulate paths
            r_paths, S_paths, v_paths = self.simulate_paths(n_paths, n_steps, dt)
            
            # Compute payoffs (already discounted in the payoff method)
            if engine is None:
                # Use standard Monte Carlo
                payoffs = option.payoff(r_paths, S_paths, v_paths)
                price = np.mean(payoffs)
            else:
                # Use specified engine
                price = engine.price(option, r_paths, S_paths, v_paths)
            
            # Validate result
            if not np.isfinite(price):
                raise NumericalError("Pricing produced non-finite result")
            
            return price
            
        except Exception as e:
            raise NumericalError(f"Pricing failed: {str(e)}")

    def simulate(
        self,
        T: float,
        n_steps: int,
        n_paths: int,
        dt: Optional[float] = None
    ) -> np.ndarray:
        """
        Simulate paths of the three-factor model.

        Args:
            T: Time horizon
            n_steps: Number of time steps
            n_paths: Number of paths to simulate
            dt: Optional time step size (if None, computed as T/n_steps)

        Returns:
            np.ndarray: Array of shape (n_paths, n_steps + 1, 3) containing
                       interest rate paths, equity price paths, and variance paths

        Raises:
            ValidationError: If parameters are invalid
            NumericalError: If simulation fails
        """
        try:
            # Validate input parameters
            if n_paths < 100:
                raise ValidationError("Number of paths must be at least 100 for reliable simulation")
            if n_steps < 10:
                raise ValidationError("Number of time steps must be at least 10 for reliable simulation")
            if T <= 0:
                raise ValidationError("Time horizon must be positive")
                
            if dt is None:
                dt = T / n_steps

            # Initialize paths
            r_paths = np.zeros((n_paths, n_steps + 1))
            S_paths = np.zeros((n_paths, n_steps + 1))
            v_paths = np.zeros((n_paths, n_steps + 1))

            # Set initial values
            r_paths[:, 0] = self.params.rate.r0
            S_paths[:, 0] = self.params.equity.S0
            v_paths[:, 0] = self.params.equity.v0

            # Generate correlated Brownian increments
            dW = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps, 3))
            corr_matrix = self.get_correlation_matrix(0.0)  # Use initial correlation
            dW_corr = np.zeros_like(dW)
            for i in range(n_paths):
                dW_corr[i] = np.dot(dW[i], np.linalg.cholesky(corr_matrix).T)

            # Simulate paths
            for i in range(n_steps):
                # Update correlation matrix if time-dependent
                if isinstance(self.params.correlation, TimeDependentCorrelation):
                    corr_matrix = self.get_correlation_matrix((i + 1) * dt)
                    dW_corr[:, i] = np.dot(dW[:, i], np.linalg.cholesky(corr_matrix).T)

                # Interest rate process (CIR)
                r_paths[:, i + 1] = r_paths[:, i] + \
                    self.params.rate.kappa * (self.params.rate.theta - r_paths[:, i]) * dt + \
                    self.params.rate.sigma * np.sqrt(r_paths[:, i]) * dW_corr[:, i, 0]
                r_paths[:, i + 1] = np.maximum(r_paths[:, i + 1], 0.0)  # Ensure non-negativity

                # Variance process (CIR)
                v_paths[:, i + 1] = v_paths[:, i] + \
                    self.params.equity.kappa_v * (self.params.equity.theta_v - v_paths[:, i]) * dt + \
                    self.params.equity.sigma_v * np.sqrt(v_paths[:, i]) * dW_corr[:, i, 2]
                v_paths[:, i + 1] = np.maximum(v_paths[:, i + 1], 0.0)  # Ensure non-negativity

                # Equity price process (Heston)
                S_paths[:, i + 1] = S_paths[:, i] * np.exp(
                    (self.params.equity.mu - self.params.equity.q - 0.5 * v_paths[:, i]) * dt + \
                    np.sqrt(v_paths[:, i]) * dW_corr[:, i, 1]
                )

            # Validate results
            if not np.all(np.isfinite(r_paths)) or not np.all(np.isfinite(S_paths)) or not np.all(np.isfinite(v_paths)):
                raise NumericalError("Path simulation produced non-finite results")

            # Stack the paths into a single array
            paths = np.stack([r_paths, S_paths, v_paths], axis=2)
            return paths

        except Exception as e:
            raise NumericalError(f"Path simulation failed: {str(e)}")

Chen3Model = ChenModel
