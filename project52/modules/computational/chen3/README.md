# Chen3: Three-Factor Stochastic Model for Financial Derivatives

A comprehensive Python package implementing a three-factor stochastic model for pricing complex financial derivatives. The model combines stochastic interest rates, equity prices, and rough volatility to provide a realistic framework for financial instrument valuation.

## Features

- **Three-Factor Model**:
  - Stochastic interest rates (CIR process)
  - Equity prices with stochastic volatility
  - Rough volatility dynamics

- **Advanced Correlation Structures**:
  - Time-dependent correlations
  - State-dependent correlations
  - Regime-switching correlations
  - Stochastic correlations
  - Copula-based correlations

- **Numerical Methods**:
  - Monte Carlo simulation
  - Adaptive time stepping
  - Variance reduction techniques
  - Parallel computation
  - GPU acceleration

- **Pricing Capabilities**:
  - European options
  - American options
  - Path-dependent options
  - Complex structured products
  - Interest rate derivatives

## Installation

```bash
# Basic installation
pip install chen3

# With CPU acceleration
pip install chen3[cpu]

# With GPU acceleration
pip install chen3[gpu]

# With distributed computing support
pip install chen3[ray]  # For Ray
pip install chen3[spark]  # For PySpark
```

## Quick Start

```python
from chen3 import ChenModel, ModelParams, RateParams, EquityParams
from chen3.correlation import TimeDependentCorrelation
import numpy as np

# Define model parameters
rate_params = RateParams(
    kappa=0.1,    # Mean reversion speed
    theta=0.05,   # Long-term mean
    sigma=0.1,    # Volatility
    r0=0.03       # Initial rate
)

equity_params = EquityParams(
    mu=0.05,      # Drift
    q=0.02,       # Dividend yield
    S0=100.0,     # Initial stock price
    v0=0.04,      # Initial variance
    kappa_v=2.0,  # Variance mean reversion
    theta_v=0.04, # Long-term variance
    sigma_v=0.3   # Volatility of variance
)

# Create time-dependent correlation
time_points = np.array([0.0, 1.0, 2.0])
corr_matrices = [
    np.array([[1.0, 0.5, 0.3],
             [0.5, 1.0, 0.2],
             [0.3, 0.2, 1.0]]),
    np.array([[1.0, 0.6, 0.4],
             [0.6, 1.0, 0.3],
             [0.4, 0.3, 1.0]]),
    np.array([[1.0, 0.7, 0.5],
             [0.7, 1.0, 0.4],
             [0.5, 0.4, 1.0]])
]
time_corr = TimeDependentCorrelation(
    time_points=time_points,
    correlation_matrices=corr_matrices
)

# Create model parameters
model_params = ModelParams(
    rate=rate_params,
    equity=equity_params,
    correlation=time_corr
)

# Create and use the model
model = ChenModel(model_params)

# Define a payoff function (e.g., European call option)
def payoff_function(r_paths, S_paths, v_paths):
    return np.maximum(S_paths[:, -1] - 100, 0)

# Price the instrument
price = model.price_instrument(
    payoff_function,
    n_paths=10000,
    n_steps=100,
    dt=0.01
)
```

## Documentation

For detailed documentation, please visit our [documentation site](https://chen3.readthedocs.io/).

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{chen3,
  author = {254Co},
  title = {Chen3: Three-Factor Stochastic Model for Financial Derivatives},
  year = {2024},
  url = {https://github.com/254co/chen3}
}
```
