# Chen3: Three-Factor Chen Model Implementation

A Python implementation of the Three-Factor Chen Model for option pricing and risk analysis.

## Overview

The Chen3 package provides a comprehensive implementation of the Three-Factor Chen Model, which combines stochastic volatility, stochastic interest rates, and correlation between these factors. The model is particularly useful for pricing and risk analysis of equity options in markets where both volatility and interest rates exhibit significant stochastic behavior.

## Features

- Three-factor model with stochastic volatility and interest rates
- Multiple correlation structures (constant, time-dependent, stochastic)
- Support for various option types:
  - European options
  - American options
  - Barrier options
- Risk metrics calculation (Greeks)
- Model calibration to market data
- Performance monitoring and caching
- Comprehensive configuration management
- Extensive test coverage

## Installation

```bash
pip install chen3
```

## Quick Start

```python
from chen3 import create_model, price_option

# Create a model with default parameters
model = create_model()

# Price a European call option
price = price_option(
    model=model,
    option_type="call",
    strike=100.0,
    maturity=1.0,
)

# Calculate risk metrics
metrics = price_option(
    model=model,
    option_type="call",
    strike=100.0,
    maturity=1.0,
    calculate_metrics=True,
)
```

## Model Parameters

### Rate Parameters
- `kappa`: Mean reversion speed
- `theta`: Long-term mean level
- `sigma`: Volatility
- `r0`: Initial rate

### Equity Parameters
- `mu`: Drift rate
- `q`: Dividend yield
- `S0`: Initial stock price
- `v0`: Initial variance
- `kappa_v`: Variance mean reversion speed
- `theta_v`: Long-term variance level
- `sigma_v`: Variance volatility

### Correlation Parameters
- Constant correlation: `rho`
- Time-dependent correlation: `rho0`, `rho1`, `alpha`
- Stochastic correlation: `rho0`, `kappa_rho`, `theta_rho`, `sigma_rho`

## Configuration

The package can be configured using YAML or JSON files:

```yaml
simulation:
  num_paths: 10000
  num_steps: 252
  seed: null
  antithetic: true
  parallel: true
  num_threads: null

numerical:
  tolerance: 1e-6
  max_iterations: 1000
  method: euler
  adaptive: true
  min_step_size: 1e-4
  max_step_size: 1.0

model:
  rate_params:
    kappa: 0.1
    theta: 0.05
    sigma: 0.1
    r0: 0.03
  equity_params:
    mu: 0.05
    q: 0.02
    S0: 100.0
    v0: 0.04
    kappa_v: 2.0
    theta_v: 0.04
    sigma_v: 0.3
  correlation_type: constant
  correlation_params:
    rho: 0.5

log_level: INFO
log_file: null
cache_dir: null
```

## Performance Monitoring

The package includes built-in performance monitoring:

```python
from chen3.utils.performance import track_performance, log_performance_summary

@track_performance
def my_function():
    # Your code here
    pass

# Log performance summary
log_performance_summary()
```

## Caching

Results can be cached to improve performance:

```python
from chen3.utils.cache import cached

@cached
def expensive_calculation(x):
    # Your code here
    return result
```

## Testing

Run the test suite:

```bash
pytest chen3/tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```
@software{chen3,
  author = {Your Name},
  title = {Chen3: Three-Factor Chen Model Implementation},
  year = {2024},
  url = {https://github.com/yourusername/chen3}
}
```

## Acknowledgments

- Original Chen model paper: [Chen, R.R. (1996) "Understanding and Implementing the Three-Factor Chen Model"]
- Contributors and maintainers
- Open source community
