# -------------------- chen3/numerical_engines/aad.py --------------------
"""
Enhanced Adjoint Automatic Differentiation (AAD) for financial derivatives pricing.
"""
import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jacfwd, jacrev
from typing import Tuple, Dict, List, Optional, Union, Callable
from dataclasses import dataclass

@dataclass
class AADResult:
    """Container for AAD pricing and sensitivity results."""
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    vanna: float
    volga: float
    n_paths: int
    n_steps: int

def mc_price_and_greeks_jax(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    K: float,
    n_paths: int,
    n_steps: int,
    seed: int = 0,
    payoff_type: str = 'call',
    barrier: Optional[float] = None,
    barrier_type: Optional[str] = None
) -> AADResult:
    """
    Compute price and Greeks using AAD for various option types.
    
    Parameters
    ----------
    S0 : float
        Initial stock price
    r : float
        Risk-free rate
    sigma : float
        Volatility
    T : float
        Time to maturity
    K : float
        Strike price
    n_paths : int
        Number of Monte Carlo paths
    n_steps : int
        Number of time steps
    seed : int, optional
        Random seed, by default 0
    payoff_type : str, optional
        Type of payoff ('call', 'put', 'binary'), by default 'call'
    barrier : Optional[float], optional
        Barrier level for barrier options, by default None
    barrier_type : Optional[str], optional
        Type of barrier ('up', 'down', 'up_and_out', 'down_and_out'), by default None
        
    Returns
    -------
    AADResult
        Container with price and Greeks
    """
    dt = T / n_steps
    
    @jax.jit
    def single_path(key):
        def step(logS, subkey):
            z = random.normal(subkey)
            return logS + (r - 0.5*sigma**2)*dt + sigma*jnp.sqrt(dt)*z
        
        # Simulate path
        keys = random.split(key, n_steps)
        logS = jnp.log(S0)
        path = jax.lax.scan(lambda carry, x: (step(carry, x), carry), logS, keys)[1]
        S = jnp.exp(jnp.concatenate([jnp.array([logS]), path]))
        
        # Apply barrier if specified
        if barrier is not None:
            if barrier_type == 'up':
                S = jnp.where(S > barrier, 0, S)
            elif barrier_type == 'down':
                S = jnp.where(S < barrier, 0, S)
            elif barrier_type == 'up_and_out':
                S = jnp.where(jnp.any(S > barrier), 0, S[-1])
            elif barrier_type == 'down_and_out':
                S = jnp.where(jnp.any(S < barrier), 0, S[-1])
        
        # Compute payoff
        if payoff_type == 'call':
            payoff = jnp.maximum(S[-1] - K, 0)
        elif payoff_type == 'put':
            payoff = jnp.maximum(K - S[-1], 0)
        elif payoff_type == 'binary':
            payoff = jnp.where(S[-1] > K, 1, 0)
        
        return payoff, S
    
    @jax.jit
    def price_fn(S0, r, sigma, T, K):
        keys = random.split(random.PRNGKey(seed), n_paths)
        payoffs, paths = vmap(single_path)(keys)
        return jnp.exp(-r*T) * jnp.mean(payoffs)
    
    # Compute price
    price = float(price_fn(S0, r, sigma, T, K))
    
    # Compute first-order Greeks
    delta = float(grad(price_fn, argnums=0)(S0, r, sigma, T, K))
    rho = float(grad(price_fn, argnums=1)(S0, r, sigma, T, K))
    vega = float(grad(price_fn, argnums=2)(S0, r, sigma, T, K))
    theta = float(grad(price_fn, argnums=3)(S0, r, sigma, T, K))
    
    # Compute second-order Greeks
    gamma = float(grad(grad(price_fn, argnums=0), argnums=0)(S0, r, sigma, T, K))
    vanna = float(grad(grad(price_fn, argnums=0), argnums=2)(S0, r, sigma, T, K))
    volga = float(grad(grad(price_fn, argnums=2), argnums=2)(S0, r, sigma, T, K))
    
    return AADResult(
        price=price,
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        rho=rho,
        vanna=vanna,
        volga=volga,
        n_paths=n_paths,
        n_steps=n_steps
    )

def multi_asset_price_and_greeks_jax(
    S0: jnp.ndarray,
    r: float,
    sigma: jnp.ndarray,
    T: float,
    K: float,
    n_paths: int,
    n_steps: int,
    correlation: jnp.ndarray,
    seed: int = 0,
    payoff_type: str = 'call'
) -> Dict[str, Union[float, jnp.ndarray]]:
    """
    Compute price and Greeks for multi-asset options using AAD.
    
    Parameters
    ----------
    S0 : jnp.ndarray
        Initial stock prices
    r : float
        Risk-free rate
    sigma : jnp.ndarray
        Volatilities
    T : float
        Time to maturity
    K : float
        Strike price
    n_paths : int
        Number of Monte Carlo paths
    n_steps : int
        Number of time steps
    correlation : jnp.ndarray
        Correlation matrix
    seed : int, optional
        Random seed, by default 0
    payoff_type : str, optional
        Type of payoff ('call', 'put', 'basket'), by default 'call'
        
    Returns
    -------
    Dict[str, Union[float, jnp.ndarray]]
        Dictionary containing price and Greeks
    """
    dt = T / n_steps
    n_assets = len(S0)
    
    @jax.jit
    def single_path(key):
        def step(logS, subkey):
            z = random.multivariate_normal(
                jnp.zeros(n_assets),
                correlation,
                shape=(1,)
            )[0]
            drift = (r - 0.5*sigma**2)*dt
            diffusion = sigma*jnp.sqrt(dt)*z
            return logS + drift + diffusion
        
        # Simulate path
        keys = random.split(key, n_steps)
        logS = jnp.log(S0)
        path = jax.lax.scan(lambda carry, x: (step(carry, x), carry), logS, keys)[1]
        S = jnp.exp(jnp.vstack([logS, path]))
        
        # Compute payoff
        if payoff_type == 'call':
            payoff = jnp.maximum(jnp.mean(S[-1]) - K, 0)
        elif payoff_type == 'put':
            payoff = jnp.maximum(K - jnp.mean(S[-1]), 0)
        elif payoff_type == 'basket':
            payoff = jnp.maximum(jnp.sum(S[-1]) - K, 0)
        
        return payoff, S
    
    @jax.jit
    def price_fn(S0, r, sigma, T, K):
        keys = random.split(random.PRNGKey(seed), n_paths)
        payoffs, paths = vmap(single_path)(keys)
        return jnp.exp(-r*T) * jnp.mean(payoffs)
    
    # Compute price
    price = float(price_fn(S0, r, sigma, T, K))
    
    # Compute first-order Greeks
    delta = jnp.array(grad(price_fn, argnums=0)(S0, r, sigma, T, K))
    rho = float(grad(price_fn, argnums=1)(S0, r, sigma, T, K))
    vega = jnp.array(grad(price_fn, argnums=2)(S0, r, sigma, T, K))
    theta = float(grad(price_fn, argnums=3)(S0, r, sigma, T, K))
    
    # Compute second-order Greeks
    gamma = jnp.array(jacfwd(jacrev(price_fn, argnums=0), argnums=0)(S0, r, sigma, T, K))
    vanna = jnp.array(jacfwd(jacrev(price_fn, argnums=0), argnums=2)(S0, r, sigma, T, K))
    volga = jnp.array(jacfwd(jacrev(price_fn, argnums=2), argnums=2)(S0, r, sigma, T, K))
    
    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'rho': rho,
        'vanna': vanna,
        'volga': volga,
        'n_paths': n_paths,
        'n_steps': n_steps
    }
