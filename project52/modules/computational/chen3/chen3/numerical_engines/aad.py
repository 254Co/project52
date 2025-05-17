# -------------------- chen3/numerical_engines/aad.py --------------------
"""
Adjoint AAD pricing with JAX for European calls.
"""
import jax
import jax.numpy as jnp
from jax import random, grad, vmap
from typing import Tuple


def mc_price_and_delta_jax(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    K: float,
    n_paths: int,
    n_steps: int,
    seed: int = 0
) -> Tuple[float, float]:
    dt = T / n_steps
    @jax.jit
    def single_payoff(key):
        def step(logS, subkey):
            z = random.normal(subkey)
            return logS + (r - 0.5*sigma**2)*dt + sigma*jnp.sqrt(dt)*z
        # simulate log-forward
        keys = random.split(key, n_steps)
        logS = jnp.log(S0)
        final = jax.lax.fori_loop(0, n_steps, lambda i, val: step(val, keys[i]), logS)
        ST = jnp.exp(final)
        return jnp.maximum(ST - K, 0.0)

    @jax.jit
    def price_fn(S0):
        keys = random.split(random.PRNGKey(seed), n_paths)
        payoffs = vmap(single_payoff)(keys)
        return jnp.exp(-r*T) * jnp.mean(payoffs)

    price = float(price_fn(S0))
    delta = float(grad(price_fn)(S0))
    return price, delta
