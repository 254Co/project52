# File: examples/barrier_example.py
"""
Example: Pricing a down-and-out barrier option under the Chen model.
"""
import numpy as np

from chen3 import (
    ChenModel,
    EquityParams,
    ModelParams,
    MonteCarloPricer,
    RateParams,
    Settings,
    make_simulator,
)
from chen3.payoffs import Barrier

# Model parameters
rate = RateParams(kappa=0.1, theta=0.03, sigma=0.01, r0=0.03)
equity = EquityParams(
    mu=0.05, q=0.02, S0=100.0, v0=0.04, kappa_v=1.5, theta_v=0.04, sigma_v=0.3
)
corr = np.eye(3)
params = ModelParams(rate=rate, equity=equity, corr_matrix=corr)
model = ChenModel(params)

# Simulation settings
dt = 1 / 252
settings = Settings(n_paths=100_000, n_steps=252, dt=dt, backend="cpu")
sim = make_simulator(model, settings)
paths = sim.generate()

# Barrier payoff: knock-out if S < 80 at any time
barrier_payoff = Barrier(strike=100, barrier=80, knock_in=False, call=True)
pricer = MonteCarloPricer(
    payoff=barrier_payoff,
    discount_curve=lambda T: np.exp(-0.03 * T),
    dt=dt,
    n_steps=settings.n_steps,
)
price = pricer.price(paths)
print(f"Down-and-Out Barrier Call Price: {price:.4f}")
