"""
User-friendly API for the Chen3 package.

This module provides simplified interfaces for common operations in the Chen3 package,
making it easier for users to work with the model without dealing with low-level details.
"""

from typing import Callable, Dict, List, Optional, Union

import numpy as np

from .correlation import (
    CopulaCorrelation,
    RegimeSwitchingCorrelation,
    StateDependentCorrelation,
    StochasticCorrelation,
    TimeDependentCorrelation,
)
from .datatypes import EquityParams, ModelParams, RateParams, CorrelationParams
from .model import ChenModel
from .utils.exceptions import (
    ChenError,
    ValidationError,
    OptionError,
    CalibrationError,
    RiskAnalysisError,
    InputError,
)
from .utils.logging import logger


def create_model(
    rate_params: Optional[Dict[str, float]] = None,
    equity_params: Optional[Dict[str, float]] = None,
    correlation_params: Optional[Dict[str, float]] = None,
) -> ChenModel:
    """
    Create a Chen model instance with default or custom parameters.

    Args:
        rate_params: Optional dictionary of rate parameters
        equity_params: Optional dictionary of equity parameters
        correlation_params: Optional dictionary of correlation parameters

    Returns:
        ChenModel: Initialized model instance

    Example:
        >>> model = create_model()
        >>> # Or with custom parameters:
        >>> model = create_model(
        ...     rate_params={'kappa': 0.2, 'theta': 0.05, 'sigma': 0.08, 'r0': 0.03},
        ...     equity_params={'mu': 0.05, 'q': 0.02, 'S0': 100.0, 'v0': 0.04,
        ...                    'kappa_v': 2.0, 'theta_v': 0.04, 'sigma_v': 0.2}
        ... )
    """
    # Default parameters that satisfy Feller condition and ensure stability
    default_rate_params = {
        'kappa': 0.2,    # Mean reversion speed
        'theta': 0.05,   # Long-term mean
        'sigma': 0.08,   # Volatility (2 * 0.2 * 0.05 = 0.02 > 0.08^2 = 0.0064)
        'r0': 0.03       # Initial rate
    }

    default_equity_params = {
        'mu': 0.05,      # Drift
        'q': 0.02,       # Dividend yield
        'S0': 100.0,     # Initial stock price
        'v0': 0.04,      # Initial variance
        'kappa_v': 2.0,  # Variance mean reversion speed
        'theta_v': 0.04, # Variance long-term mean
        'sigma_v': 0.2   # Variance volatility (reduced for stability)
    }

    default_correlation_params = {
        'rho_rs': 0.1,   # Rate-stock correlation
        'rho_rv': -0.1,  # Rate-variance correlation
        'rho_sv': -0.5   # Stock-variance correlation
    }

    # Use provided parameters or defaults
    rate_params = rate_params or default_rate_params
    equity_params = equity_params or default_equity_params
    correlation_params = correlation_params or default_correlation_params

    try:
        # Create model parameters
        corr = np.array([
            [1.0, correlation_params['rho_rs'], correlation_params['rho_rv']],
            [correlation_params['rho_rs'], 1.0, correlation_params['rho_sv']],
            [correlation_params['rho_rv'], correlation_params['rho_sv'], 1.0]
        ])
        params = ModelParams(
            rate=RateParams(**rate_params),
            equity=EquityParams(**equity_params),
            correlation=corr
        )

        # Initialize and return model
        return ChenModel(params)

    except ValidationError as e:
        logger.error(f"Failed to create model: {str(e)}")
        raise InputError(f"Invalid model parameters: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to create model: {str(e)}")
        raise ChenError(f"Failed to create model: {str(e)}")


def price_option(
    model: ChenModel,
    option_type: str,
    strike: float,
    maturity: float,
    n_paths: int = 10000,
    n_steps: int = 100,
    **kwargs,
) -> Dict[str, float]:
    """
    Price a vanilla option using the Chen model.

    Args:
        model: ChenModel instance
        option_type: Type of option ('call' or 'put')
        strike: Option strike price
        maturity: Option maturity in years
        n_paths: Number of simulation paths
        n_steps: Number of time steps
        **kwargs: Additional option parameters
            - barrier: For barrier options
            - rebate: For barrier options
            - barrier_type: 'up', 'down', 'up_and_out', 'down_and_out'
            - american: True for American options

    Returns:
        Dict containing:
            - price: Option price
            - std_error: Standard error of the price
            - delta: Option delta
            - gamma: Option gamma
            - vega: Option vega
            - theta: Option theta
            - rho: Option rho

    Raises:
        OptionError: If option parameters are invalid
        ValidationError: If parameter validation fails
        ChenError: For other errors

    Example:
        >>> model = create_model()
        >>> result = price_option(
        ...     model,
        ...     option_type='call',
        ...     strike=100.0,
        ...     maturity=1.0,
        ...     n_paths=10000
        ... )
        >>> print(f"Option price: {result['price']:.4f} ± {result['std_error']:.4f}")
    """
    try:
        # Validate inputs
        if not isinstance(model, ChenModel):
            raise InputError("model must be a ChenModel instance")
        
        supported_types = ["call", "put", "up_and_out_call", "down_and_out_put"]
        if option_type not in supported_types:
            raise OptionError(f"Unsupported option type: {option_type}")
        
        if strike <= 0:
            raise OptionError("Strike price must be positive")
        
        if maturity <= 0:
            raise OptionError("Maturity must be positive")
        
        if n_paths <= 0:
            raise OptionError("Number of paths must be positive")
        
        if n_steps <= 0:
            raise OptionError("Number of steps must be positive")
        
        # Validate additional parameters
        if "barrier" in kwargs and kwargs["barrier"] <= 0:
            raise OptionError("Barrier level must be positive")
        
        if "rebate" in kwargs and kwargs["rebate"] < 0:
            raise OptionError("Rebate must be non-negative")
        
        if "barrier_type" in kwargs:
            valid_types = ["up", "down", "up_and_out", "down_and_out"]
            if kwargs["barrier_type"] not in valid_types:
                raise OptionError(f"Invalid barrier type: {kwargs['barrier_type']}")

        # Set up time grid
        dt = maturity / n_steps

        # Define payoff function
        if option_type == "call":
            def payoff(r_paths, S_paths, v_paths):
                return np.maximum(S_paths[:, -1] - strike, 0)
        elif option_type == "put":
            def payoff(r_paths, S_paths, v_paths):
                return np.maximum(strike - S_paths[:, -1], 0)
        elif option_type == "up_and_out_call":
            barrier = kwargs.get("barrier")
            if barrier is None:
                raise OptionError("Barrier level must be specified for barrier options")
            def payoff(r_paths, S_paths, v_paths):
                knocked_out = np.any(S_paths >= barrier, axis=1)
                return np.where(knocked_out, 0.0, np.maximum(S_paths[:, -1] - strike, 0))
        elif option_type == "down_and_out_put":
            barrier = kwargs.get("barrier")
            if barrier is None:
                raise OptionError("Barrier level must be specified for barrier options")
            def payoff(r_paths, S_paths, v_paths):
                knocked_out = np.any(S_paths <= barrier, axis=1)
                return np.where(knocked_out, 0.0, np.maximum(strike - S_paths[:, -1], 0))

        # Price the option
        price = model.price_instrument(
            payoff_function=payoff,
            n_paths=n_paths,
            n_steps=n_steps,
            dt=dt,
        )

        # Calculate Greeks
        # TODO: Implement Greek calculations

        return {
            "price": price,
            "std_error": 0.0,  # TODO: Calculate standard error
            "delta": 0.0,  # TODO: Calculate delta
            "gamma": 0.0,  # TODO: Calculate gamma
            "vega": 0.0,  # TODO: Calculate vega
            "theta": 0.0,  # TODO: Calculate theta
            "rho": 0.0,  # TODO: Calculate rho
        }

    except (OptionError, ValidationError) as e:
        logger.error(f"Option pricing failed: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Failed to price option: {str(e)}")
        raise ChenError(f"Failed to price option: {str(e)}")


def calibrate_model(
    market_data: Dict[str, float], initial_params: Optional[Dict] = None, **kwargs
) -> ChenModel:
    """
    Calibrate the Chen model to market data.

    Args:
        market_data: Dictionary of market data
            - spot_price: Current spot price
            - risk_free_rate: Current risk-free rate
            - dividend_yield: Current dividend yield
            - option_prices: Dictionary of option prices
                - strike: Option strike price
                - maturity: Option maturity
                - price: Market price
                - type: 'call' or 'put'
        initial_params: Initial parameter guess
        **kwargs: Calibration options
            - method: Optimization method ('least_squares', 'mle')
            - max_iter: Maximum number of iterations
            - tol: Convergence tolerance

    Returns:
        ChenModel: Calibrated model instance

    Example:
        >>> market_data = {
        ...     'spot_price': 100.0,
        ...     'risk_free_rate': 0.03,
        ...     'dividend_yield': 0.02,
        ...     'option_prices': {
        ...         'call_100_1y': {
        ...             'strike': 100.0,
        ...             'maturity': 1.0,
        ...             'price': 8.0,
        ...             'type': 'call'
        ...         }
        ...     }
        ... }
        >>> model = calibrate_model(market_data)
    """
    # TODO: Implement calibration
    raise NotImplementedError("Model calibration not yet implemented")


def analyze_risk(
    model: ChenModel, portfolio: Dict[str, Dict], scenarios: Optional[Dict] = None
) -> Dict[str, float]:
    """
    Perform risk analysis on a portfolio of instruments.

    Args:
        model: ChenModel instance
        portfolio: Dictionary of portfolio positions
            - instrument_id: Dictionary of instrument details
                - type: Instrument type
                - params: Instrument parameters
                - notional: Position notional
        scenarios: Dictionary of risk scenarios
            - rate_shock: Interest rate shock
            - vol_shock: Volatility shock
            - correlation_shock: Correlation shock

    Returns:
        Dict containing:
            - var: Value at Risk
            - cvar: Conditional Value at Risk
            - pnl: Profit and Loss
            - greeks: Portfolio Greeks

    Example:
        >>> portfolio = {
        ...     'call_100_1y': {
        ...         'type': 'call',
        ...         'params': {'strike': 100.0, 'maturity': 1.0},
        ...         'notional': 1000000
        ...     }
        ... }
        >>> risk = analyze_risk(model, portfolio)
    """
    # TODO: Implement risk analysis
    raise NotImplementedError("Risk analysis not yet implemented")
