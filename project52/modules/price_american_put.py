from math import exp
from lsm_engine.utils.logging import init_logger
from lsm_engine.engine.pathgen import generate_gbm_paths
from lsm_engine.engine.pricer import LSMPricer
from lsm_engine.payoffs.vanilla import american_put

if __name__ == "__main__":
    init_logger("INFO")

    S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    n_steps, n_paths = 50, 50_000
    discount = exp(-r * T / n_steps)

    paths = generate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed=42)
    exercise_dates = list(range(1, n_steps + 1))

    pricer = LSMPricer(american_put(K), discount)
    pv = pricer.price(paths, exercise_dates)
    print("American put LSM price:", round(pv, 4))