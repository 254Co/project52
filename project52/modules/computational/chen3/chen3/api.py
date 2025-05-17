"""
User-friendly API for the Chen3 package.

This module provides simplified interfaces for common operations in the Chen3 package,
making it easier for users to work with the model without dealing with low-level details.
"""

from typing import Optional, Union, Callable, Dict, List
import numpy as np
from .model import ChenModel
from .datatypes import ModelParams, RateParams, EquityParams
from .correlation import (
    TimeDependentCorrelation,
    StateDependentCorrelation,
    RegimeSwitchingCorrelation,
    StochasticCorrelation,
    CopulaCorrelation
)
from .utils.logging import logger
from .utils.exceptions import ChenError

def create_model(
    rate_params: Optional[Dict[str, float]] = None,
    equity_params: Optional[Dict[str, float]] = None,
    correlation: Optional[Union[np.ndarray, Dict]] = None
) -> ChenModel:
    """
    Create a Chen model with default or specified parameters.
    
    Args:
        rate_params: Dictionary of interest rate parameters
            - kappa: Mean reversion speed (default: 0.1)
            - theta: Long-term mean (default: 0.05)
            - sigma: Volatility (default: 0.1)
            - r0: Initial rate (default: 0.03)
        equity_params: Dictionary of equity parameters
            - mu: Drift (default: 0.05)
            - q: Dividend yield (default: 0.02)
            - S0: Initial stock price (default: 100.0)
            - v0: Initial variance (default: 0.04)
            - kappa_v: Variance mean reversion (default: 2.0)
            - theta_v: Long-term variance (default: 0.04)
            - sigma_v: Volatility of variance (default: 0.3)
        correlation: Correlation specification
            - If np.ndarray: Constant correlation matrix
            - If Dict: Configuration for correlation type
                - type: 'constant', 'time_dependent', 'state_dependent',
                       'regime_switching', 'stochastic', or 'copula'
                - params: Parameters for the chosen correlation type
    
    Returns:
        ChenModel: Initialized model instance
    
    Example:
        >>> # Create model with default parameters
        >>> model = create_model()
        
        >>> # Create model with custom parameters
        >>> model = create_model(
        ...     rate_params={'kappa': 0.2, 'theta': 0.04},
        ...     equity_params={'S0': 150.0, 'v0': 0.05},
        ...     correlation={
        ...         'type': 'time_dependent',
        ...         'params': {
        ...             'time_points': [0, 1, 2],
        ...             'correlation_matrices': [
        ...                 [[1.0, 0.5, 0.3],
        ...                  [0.5, 1.0, 0.2],
        ...                  [0.3, 0.2, 1.0]],
        ...                 [[1.0, 0.6, 0.4],
        ...                  [0.6, 1.0, 0.3],
        ...                  [0.4, 0.3, 1.0]],
        ...                 [[1.0, 0.7, 0.5],
        ...                  [0.7, 1.0, 0.4],
        ...                  [0.5, 0.4, 1.0]]
        ...             ]
        ...         }
        ...     }
        ... )
    """
    # Set default parameters
    default_rate_params = {
        'kappa': 0.1,
        'theta': 0.05,
        'sigma': 0.1,
        'r0': 0.03
    }
    default_equity_params = {
        'mu': 0.05,
        'q': 0.02,
        'S0': 100.0,
        'v0': 0.04,
        'kappa_v': 2.0,
        'theta_v': 0.04,
        'sigma_v': 0.3
    }
    default_correlation = np.array([
        [1.0, 0.5, 0.3],
        [0.5, 1.0, 0.2],
        [0.3, 0.2, 1.0]
    ])
    
    # Update with provided parameters
    if rate_params:
        default_rate_params.update(rate_params)
    if equity_params:
        default_equity_params.update(equity_params)
    
    # Create parameter objects
    rate = RateParams(**default_rate_params)
    equity = EquityParams(**default_equity_params)
    
    # Handle correlation
    if correlation is None:
        corr = default_correlation
    elif isinstance(correlation, np.ndarray):
        corr = correlation
    elif isinstance(correlation, dict):
        corr_type = correlation.get('type', 'constant')
        corr_params = correlation.get('params', {})
        
        if corr_type == 'constant':
            corr = np.array(corr_params.get('matrix', default_correlation))
        elif corr_type == 'time_dependent':
            corr = TimeDependentCorrelation(
                time_points=np.array(corr_params['time_points']),
                correlation_matrices=[np.array(m) for m in corr_params['correlation_matrices']]
            )
        elif corr_type == 'state_dependent':
            corr = StateDependentCorrelation(**corr_params)
        elif corr_type == 'regime_switching':
            corr = RegimeSwitchingCorrelation(**corr_params)
        elif corr_type == 'stochastic':
            corr = StochasticCorrelation(**corr_params)
        elif corr_type == 'copula':
            corr = CopulaCorrelation(**corr_params)
        else:
            raise ChenError(f"Unsupported correlation type: {corr_type}")
    else:
        raise ChenError("Invalid correlation specification")
    
    # Create and return model
    model_params = ModelParams(rate=rate, equity=equity, correlation=corr)
    return ChenModel(model_params)

def price_option(
    model: ChenModel,
    option_type: str,
    strike: float,
    maturity: float,
    n_paths: int = 10000,
    n_steps: int = 100,
    **kwargs
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
    # Validate inputs
    if option_type not in ['call', 'put']:
        raise ChenError(f"Unsupported option type: {option_type}")
    
    # Set up time grid
    dt = maturity / n_steps
    
    # Define payoff function
    def payoff(r_paths, S_paths, v_paths):
        if option_type == 'call':
            return np.maximum(S_paths[:, -1] - strike, 0)
        else:  # put
            return np.maximum(strike - S_paths[:, -1], 0)
    
    # Price the option
    price = model.price_instrument(
        payoff_function=payoff,
        n_paths=n_paths,
        n_steps=n_steps,
        dt=dt
    )
    
    # TODO: Implement Greeks calculation
    # For now, return placeholder values
    return {
        'price': price,
        'std_error': 0.0,  # TODO: Implement
        'delta': 0.0,      # TODO: Implement
        'gamma': 0.0,      # TODO: Implement
        'vega': 0.0,       # TODO: Implement
        'theta': 0.0,      # TODO: Implement
        'rho': 0.0         # TODO: Implement
    }

def calibrate_model(
    market_data: Dict[str, float],
    initial_params: Optional[Dict] = None,
    **kwargs
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
    model: ChenModel,
    portfolio: Dict[str, Dict],
    scenarios: Optional[Dict] = None
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