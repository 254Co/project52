# example.py
import numpy as np
from chen3.config import Settings
from chen3 import (
    RateParams, EquityParams, ModelParams,
    ShortRateCIR, EquityProcess,
    cholesky_correlation, make_simulator,
    Vanilla, MonteCarloPricer
)

# 1. Define parameters
rate_p   = RateParams(kappa=0.1, theta=0.03, sigma=0.01, r0=0.03)
equity_p = EquityParams(mu=0.05, q=0.02, v0=0.04, kappa_v=1.5, theta_v=0.04, sigma_v=0.3)
corr     = np.array([[1.0, 0.2, 0.1],
                     [0.2, 1.0, 0.3],
                     [0.1, 0.3, 1.0]])
model_params = ModelParams(rate=rate_p, equity=equity_p, corr_matrix=corr)

# 2. Build model and settings
settings = Settings(seed=123, backend="cpu", n_paths=50_000, n_steps=252, dt=1/252)
sim = make_simulator(model_params, settings)

# 3. Generate paths
paths = sim.generate()   # shape (50_000, 252, 3)

# 4. Price a vanilla call
payoff = Vanilla(strike=100.0, call=True)
pricer = MonteCarloPricer(payoff, discount_curve=lambda t: np.exp(-0.03 * t))
price  = pricer.price(paths)
print(f"Vanilla call price ≈ {price:.4f}")
