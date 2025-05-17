"""
Example: Running simulation on GPU backend.
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

# Build model with basic parameters
rate = RateParams(0.05, 0.05, 0.02, 0.05)
equity = EquityParams(
    mu=0.03, q=0.01, S0=100.0, v0=0.04, kappa_v=1.0, theta_v=0.04, sigma_v=0.2
)
corr = np.eye(3)
params = ModelParams(rate, equity, corr)
model = ChenModel(params)

# GPU settings
dt = 1 / 252
settings = Settings(n_paths=200_000, n_steps=252, dt=dt, backend="gpu")
sim = make_simulator(model, settings)
paths = sim.generate()  # Cupy ndarray

# Move to CPU for payoff
paths_cpu = np.asarray(paths)
payoff = Vanilla(strike=100)
pricer = MonteCarloPricer(
    payoff, discount_curve=lambda T: np.exp(-0.05 * T), dt=dt, n_steps=settings.n_steps
)
price = pricer.price(paths_cpu)
print(f"GPU Monte Carlo Price: {price:.4f}")
