# File: examples/asian_example.py
"""
Example: Pricing an arithmetic-average Asian call.
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
from chen3.payoffs import Asian

# Build model
rate = RateParams(0.05, 0.05, 0.0, 0.05)
equity = EquityParams(
    mu=0.0, q=0.0, S0=100.0, v0=0.04, kappa_v=0.0, theta_v=0.04, sigma_v=0.0
)
corr = np.eye(3)
params = ModelParams(rate=rate, equity=equity, corr_matrix=corr)
model = ChenModel(params)

# Simulate
dt = 1 / 252
settings = Settings(n_paths=50_000, n_steps=252, dt=dt, backend="cpu")
sim = make_simulator(model, settings)
paths = sim.generate()

# Price Asian option
asian_payoff = Asian(strike=100, call=True)
pricer = MonteCarloPricer(
    payoff=asian_payoff,
    discount_curve=lambda T: np.exp(-0.05 * T),
    dt=dt,
    n_steps=settings.n_steps,
)
price = pricer.price(paths)
print(f"Asian Call Price: {price:.4f}")
