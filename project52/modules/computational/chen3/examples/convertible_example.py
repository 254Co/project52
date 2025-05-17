# File: examples/convertible_example.py
"""
Example: Pricing a convertible bond.
"""
import numpy as np
from chen3 import (
    ChenModel, RateParams, EquityParams, ModelParams,
    make_simulator, MonteCarloPricer, Settings
)
from chen3.payoffs import ConvertibleBond

# Model & simulation setup
rate = RateParams(0.03, 0.06, 0.02, 0.06)
equity = EquityParams(mu=0.07, q=0.0, S0=50.0, v0=0.09, kappa_v=2.0, theta_v=0.09, sigma_v=0.5)
corr = np.eye(3)
params = ModelParams(rate, equity, corr)
model = ChenModel(params)

dt = 1/252
settings = Settings(n_paths=100_000, n_steps=252, dt=dt, backend="cpu")
sim = make_simulator(model, settings)
paths = sim.generate()

# Convertible bond: face 100, conversion ratio 2 shares
cb_payoff = ConvertibleBond(face_value=100, conversion_ratio=2.0)
pricer = MonteCarloPricer(
    payoff=cb_payoff,
    discount_curve=lambda T: np.exp(-0.06 * T),
    dt=dt,
    n_steps=settings.n_steps
)
price = pricer.price(paths)
print(f"Convertible Bond Price: {price:.4f}")