# File: examples/scenario_example.py
"""
Example: Applying scenario shocks and explaining PnL contributions.
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
from chen3.payoffs import Vanilla
from chen3.risk import Scenario, explain_pnl

# Base model
theta = 0.04
rate = RateParams(0.1, theta, 0.0, theta)
equity = EquityParams(
    mu=0.0, q=0.0, S0=100.0, v0=0.04, kappa_v=0.0, theta_v=0.04, sigma_v=0.0
)
corr = np.eye(3)
params = ModelParams(rate, equity, corr)
model = ChenModel(params)

# Simulate small sample
dt = 1 / 252
settings = Settings(n_paths=10_000, n_steps=252, dt=dt, backend="cpu")
sim = make_simulator(model, settings)
paths = sim.generate()

# Price base and shocked
disc = lambda T: np.exp(-theta * T)
payoff = Vanilla(strike=100)
pricer = MonteCarloPricer(payoff, disc, dt, settings.n_steps)
base_price = pricer.price(paths)

# Define scenario: +10bp rate, +20% vol, +5% equity
scen = Scenario(rate_shift=0.001, vol_shift=0.20, equity_shift=0.05)
shocked_paths = scen.apply(paths)
shocked_price = pricer.price(shocked_paths)

# Explain PnL
contribs = explain_pnl(base_price, shocked_price, scen)
print(f"Base Price: {base_price:.4f}")
print(f"Shocked Price: {shocked_price:.4f}")
print("Contributions:", contribs)
