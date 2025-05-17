# Chen3 Correlation Module

This module provides various correlation structures for the Chen3 model, including time-dependent, state-dependent, regime-switching, stochastic, copula-based, network-based, fractal, wavelet, spectral, hierarchical, and dynamic correlations.

## Installation

```bash
pip install -r requirements.txt
```

## Module Structure

```
correlation/
├── core/               # Base classes and interfaces
│   └── base.py        # Base correlation class
├── models/            # Concrete correlation implementations
│   ├── time_dependent.py
│   ├── state_dependent.py
│   ├── regime_switching.py
│   ├── stochastic.py
│   ├── copula.py
│   ├── network.py
│   ├── fractal.py
│   ├── wavelet.py
│   ├── spectral.py
│   ├── hierarchical.py
│   └── dynamic.py
├── utils/             # Utility functions and helpers
│   ├── exceptions.py
│   └── logging_config.py
├── tests/             # Unit tests and examples
│   └── test_base.py
├── __init__.py        # Module exports
├── requirements.txt   # Dependencies
└── README.md         # This file
```

## Usage

### Basic Correlation Models

#### Time-Dependent Correlation

```python
from chen3.correlation import TimeDependentCorrelation
import numpy as np

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

# Get correlation matrix at time t=0.5
corr = time_corr.get_correlation_matrix(t=0.5)
```

#### State-Dependent Correlation

```python
from chen3.correlation import StateDependentCorrelation

def corr_func(state):
    v = state['volatility']
    return np.array([[1.0, 0.5 + 0.1*v, 0.3],
                     [0.5 + 0.1*v, 1.0, 0.2],
                     [0.3, 0.2, 1.0]])

state_corr = StateDependentCorrelation(corr_func)
```

#### Regime-Switching Correlation

```python
from chen3.correlation import RegimeSwitchingCorrelation

regimes = {
    'low_vol': np.array([[1.0, 0.3, 0.1],
                         [0.3, 1.0, 0.2],
                         [0.1, 0.2, 1.0]]),
    'high_vol': np.array([[1.0, 0.7, 0.5],
                          [0.7, 1.0, 0.6],
                          [0.5, 0.6, 1.0]])
}
transition_probs = {
    'low_vol': {'low_vol': 0.8, 'high_vol': 0.2},
    'high_vol': {'low_vol': 0.3, 'high_vol': 0.7}
}
regime_corr = RegimeSwitchingCorrelation(
    regimes=regimes,
    transition_probs=transition_probs,
    initial_regime='low_vol'
)
```

#### Stochastic Correlation

```python
from chen3.correlation import StochasticCorrelation

stoch_corr = StochasticCorrelation(
    kappa=2.0,    # Mean reversion speed
    theta=0.5,    # Long-term mean
    sigma=0.2,    # Volatility
    rho0=0.3      # Initial correlation
)
```

#### Copula-Based Correlation

```python
from chen3.correlation import CopulaCorrelation

copula_corr = CopulaCorrelation(
    copula_type='gaussian',
    params={'rho': 0.5}
)
```

### Advanced Correlation Models

#### Network-Based Correlation

```python
from chen3.correlation import NetworkCorrelation

# Create network correlation with edge weights
edge_weights = {
    (0, 1): 0.5,
    (1, 2): 0.3,
    (0, 2): 0.2
}
network_corr = NetworkCorrelation(
    edge_weights=edge_weights,
    decay_param=0.1
)
```

#### Fractal Correlation

```python
from chen3.correlation import FractalCorrelation

# Create fractal correlation with Hurst exponent
fractal_corr = FractalCorrelation(
    hurst_exponent=0.7,  # Long-range dependence
    n_factors=3
)
```

#### Wavelet Correlation

```python
from chen3.correlation import WaveletCorrelation

# Create wavelet correlation with multiple scales
wavelet_corr = WaveletCorrelation(
    scales=[1, 2, 4],
    coefficients=[0.5, 0.3, 0.2],
    wavelet_type='haar'
)
```

#### Spectral Correlation

```python
from chen3.correlation import SpectralCorrelation

# Create spectral correlation with eigenvalues
spectral_corr = SpectralCorrelation(
    eigenvalues=[1.0, 0.5, 0.3],
    eigenvectors=np.eye(3)
)
```

#### Hierarchical Correlation

```python
from chen3.correlation import HierarchicalCorrelation

# Create hierarchical correlation with clusters
clusters = {
    'group1': [0, 1],
    'group2': [2]
}
correlations = {
    'within': 0.7,
    'between': 0.3
}
hierarchical_corr = HierarchicalCorrelation(
    clusters=clusters,
    correlations=correlations
)
```

#### Dynamic Correlation

```python
from chen3.correlation import DynamicCorrelation

# Create dynamic correlation combining multiple structures
correlations = [
    (time_corr, 0.6),
    (state_corr, 0.4)
]
dynamic_corr = DynamicCorrelation(
    correlations=correlations
)
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
# Format code
black .
isort .

# Type checking
mypy .

# Linting
flake8
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 