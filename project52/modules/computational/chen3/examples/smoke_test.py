# File: examples/smoke_test.py

import numpy as np
from scipy.stats import norm

from chen3 import (
    ChenModel,
    RateParams,
    EquityParams,
    ModelParams,
    make_simulator,
    Settings,
)
from chen3.payoffs import Vanilla
from chen3.pricers.mc import MonteCarloPricer

def bs_price(S, K, r, sigma, T):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def main():
    # 1) Build model with constant r and v (CIR & Heston collapse)
    rate = RateParams(kappa=0.0, theta=0.05, sigma=0.0, r0=0.05)
    equity = EquityParams(
        mu=0.0, q=0.0,
        S0=100.0,
        v0=0.04, kappa_v=0.0, theta_v=0.04, sigma_v=0.0
    )
    corr = np.eye(3)
    params = ModelParams(rate=rate, equity=equity, corr_matrix=corr)
    model = ChenModel(params)

    # 2) Simulation settings
    settings = Settings(n_paths=200_000, n_steps=252, dt=1/252.0, backend="cpu")
    sim = make_simulator(model, settings)
    paths = sim.generate()  # shape (n_paths, 253, 3)

    # 3) Price a European call with MC
    strike = 100.0
    payoff = Vanilla(strike=strike, call=True)
    pricer = MonteCarloPricer(
        payoff,
        discount_curve=lambda T: np.exp(-0.05 * T),
        dt=settings.dt,
        n_steps=settings.n_steps,
    )
    mc_price = pricer.price(paths)

    # 4) Compute BS price
    T = settings.dt * settings.n_steps
    bs = bs_price(100.0, strike, 0.05, np.sqrt(0.04), T)

    print(f"MC Price: {mc_price:.4f}")
    print(f"BS Price: {bs:.4f}")

if __name__ == "__main__":
    main()
