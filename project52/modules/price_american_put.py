from math import exp
from lsm_engine.utils.logging import init_logger
from lsm_engine.engine.pathgen import generate_gbm_paths
from lsm_engine.engine.pricer import LSMPricer
from lsm_engine.payoffs.vanilla import american_put, american_call, european_call, european_put
from market_utils import expiry252time



if __name__ == "__main__":
    init_logger("CRITICAL")
    T = expiry252time("2025-12-31") ### Time in years until expiration
    S0 =  100 ### The initial spot price of the underlying asset at time 0
    K = 100 ### The option’s strike price
    r = 0.05 ### The continuously-compounded risk-free interest rate
    sigma = 0.2 ### The annualized volatility of the underlying
    n_steps = 252 ### The number of discrete time steps (so Δt = T ⁄ n_steps).
    n_paths = 50_000 ### The number of Monte-Carlo simulation paths to generate.
    discount = exp(-r * T / n_steps) ### The per-step discount factor, e^(–r Δt), used to roll back expected cash-flows between adjacent time slices

    paths = generate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed=42)
    exercise_dates = list(range(1, n_steps + 1))

    pricer = LSMPricer(american_put(K), discount)
    pv = pricer.price(paths, exercise_dates)
    print("American put LSM price:", round(pv, 4))
    
    pricer = LSMPricer(american_call(K), discount)
    pv = pricer.price(paths, exercise_dates)
    print("American call LSM price:", round(pv, 4))
    
    pricer = LSMPricer(european_put(K), discount)
    pv = pricer.price(paths, exercise_dates)
    print("European put LSM price:", round(pv, 4))
    
    pricer = LSMPricer(european_call(K), discount)
    pv = pricer.price(paths, exercise_dates)
    print("European call LSM price:", round(pv, 4))