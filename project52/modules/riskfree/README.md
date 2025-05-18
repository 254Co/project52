# RiskFree

A Python package for building and querying market-grade risk-free rate curves.

## Overview

RiskFree provides tools for constructing and working with risk-free rate curves, primarily focused on U.S. Treasury yields. The package offers:

- Automated fetching of Treasury par-yield curve data
- Bootstrapping of zero-coupon yields
- Nelson-Siegel curve smoothing
- Spot rate and discount factor calculations
- Pandas integration for data handling

## Installation

```bash
pip install riskfree
```

## Requirements

- Python 3.8 or higher
- Dependencies:
  - pandas >= 1.3
  - numpy >= 1.21
  - requests >= 2.25
  - scipy >= 1.7

## Quick Start

```python
from datetime import date
from riskfree import RiskFreeCurve

# Create a risk-free curve for today
curve = RiskFreeCurve(date.today())

# Get spot rate for 5-year tenor
rate = curve.spot(5.0)  # Returns continuously compounded rate

# Get discount factor for 5-year tenor
df = curve.discount(5.0)  # Returns exp(-r*t)

# Convert to pandas DataFrame
df = curve.to_dataframe()
```

## Core Features

### 1. Data Fetching

The package automatically fetches Treasury par-yield curve data from the official U.S. Treasury XML feed. It includes:
- Automatic retry logic for failed requests
- Backfilling up to 7 days for missing data
- Fallback to zero curve if no data is available

### 2. Curve Construction

Two methods are available for curve construction:

1. **Raw Bootstrapping**:
   - Converts par yields to zero-coupon yields
   - Uses continuous compounding convention
   - Preserves market data exactly

2. **Nelson-Siegel Smoothing**:
   - Fits a parametric curve to bootstrapped yields
   - Reduces noise in the curve
   - Provides better interpolation

### 3. Rate Calculations

- **Spot Rates**: Continuously compounded zero-coupon rates
- **Discount Factors**: Present value of $1 at future time t
- **Interpolation**: Linear interpolation between known points

### 4. Nelson-Siegel Model

The Nelson-Siegel model decomposes the yield curve into three components:

1. **Level Component** (β₀):
   - Represents the long-term rate
   - Formula: β₀
   - Effect: Constant shift of the entire curve
   ```
   Level(t) = β₀
   ```

2. **Slope Component** (β₁):
   - Represents the short-term vs long-term spread
   - Formula: β₁ * (1-exp(-t/τ))/(t/τ)
   - Effect: Decays exponentially from β₁ to 0
   ```
   Slope(t) = β₁ * (1-exp(-t/τ))/(t/τ)
   ```

3. **Curvature Component** (β₂):
   - Represents the medium-term deviation
   - Formula: β₂ * ((1-exp(-t/τ))/(t/τ) - exp(-t/τ))
   - Effect: Hump-shaped curve that peaks at t ≈ 2τ
   ```
   Curvature(t) = β₂ * ((1-exp(-t/τ))/(t/τ) - exp(-t/τ))
   ```

The complete model is:
```
r(t) = β₀ + β₁ * (1-exp(-t/τ))/(t/τ) + β₂ * ((1-exp(-t/τ))/(t/τ) - exp(-t/τ))
```

#### Component Visualization

Here's how each component affects the yield curve shape:

1. **Basic Components**:
```
Rate
  ^
  |    Level (β₀)
  |    ------------------------
  |    ^
  |    |    Slope (β₁)
  |    |    ~~~~~~~~
  |    |         ^
  |    |         |    Curvature (β₂)
  |    |         |    ~~~~~~
  |    |         |         ^
  |    |         |         |
  |    |         |         |
  +----+---------+---------+--------> Tenor
  0    1τ        2τ        3τ
```

2. **Common Curve Shapes**:

a) **Normal Curve** (β₀ = 0.05, β₁ = -0.02, β₂ = 0.01, τ = 2.0):
```
Rate
  ^
  |                    ~~~~~~~
  |                   /
  |                  /
  |                 /
  |                /
  |               /
  |              /
  +-------------+----------------> Tenor
  0             10              30
```

b) **Inverted Curve** (β₀ = 0.05, β₁ = 0.02, β₂ = -0.01, τ = 2.0):
```
Rate
  ^
  |              /
  |             /
  |            /
  |           /
  |          /
  |         /
  |        ~~~~~~~
  +-------------+----------------> Tenor
  0             10              30
```

c) **Humped Curve** (β₀ = 0.05, β₁ = -0.01, β₂ = 0.03, τ = 2.0):
```
Rate
  ^
  |                    ~~~
  |                   /   \
  |                  /     \
  |                 /       \
  |                /         \
  |               /           \
  |              /             \
  +-------------+---------------+> Tenor
  0             10             30
```

d) **Flat Curve** (β₀ = 0.05, β₁ = 0.00, β₂ = 0.00, τ = 2.0):
```
Rate
  ^
  |    ------------------------
  |
  |
  |
  |
  |
  |
  +--------------------------------> Tenor
  0                                30
```

3. **Component Interactions**:

a) **Level + Slope** (β₀ = 0.05, β₁ = -0.02, β₂ = 0.00):
```
Rate
  ^
  |                    ~~~~~~~
  |                   /
  |                  /
  |                 /
  |                /
  |               /
  |              /
  +-------------+----------------> Tenor
  0             10              30
  Level: 5% base rate
  Slope: 2% steepening
```

b) **Level + Curvature** (β₀ = 0.05, β₁ = 0.00, β₂ = 0.02):
```
Rate
  ^
  |                    ~~~
  |                   /   \
  |                  /     \
  |                 /       \
  |                /         \
  |               /           \
  |              /             \
  +-------------+---------------+> Tenor
  0             10             30
  Level: 5% base rate
  Curvature: 2% hump
```

c) **Slope + Curvature** (β₀ = 0.00, β₁ = -0.02, β₂ = 0.02):
```
Rate
  ^
  |                    ~~~
  |                   /   \
  |                  /     \
  |                 /       \
  |                /         \
  |               /           \
  |              /             \
  +-------------+---------------+> Tenor
  0             10             30
  Slope: 2% steepening
  Curvature: 2% hump
```

4. **Decay Parameter Effects**:

a) **Fast Decay** (τ = 1.0):
```
Rate
  ^
  |                    ~~~
  |                   /   \
  |                  /     \
  |                 /       \
  |                /         \
  |               /           \
  |              /             \
  +-------------+---------------+> Tenor
  0             5              30
  Components decay quickly
```

b) **Slow Decay** (τ = 3.0):
```
Rate
  ^
  |                    ~~~
  |                   /   \
  |                  /     \
  |                 /       \
  |                /         \
  |               /           \
  |              /             \
  +-------------+---------------+> Tenor
  0             15             30
  Components decay slowly
```

#### Parameter Optimization

The Nelson-Siegel parameters are optimized using a constrained nonlinear least squares approach:

1. **Objective Function**:
   ```
   min Σ(r_obs(t) - r_NS(t))²
   ```
   where:
   - r_obs(t) is the observed rate at tenor t
   - r_NS(t) is the Nelson-Siegel model rate

2. **Parameter Constraints**:
   - All parameters are bounded: -0.1 ≤ β₀, β₁, β₂, τ ≤ 0.2
   - Initial guess: [0.03, -0.02, 0.02, 1.0]
   - These bounds ensure economically meaningful results

3. **Optimization Process**:
   - Uses scipy's `minimize` function with L-BFGS-B algorithm
   - Implements box constraints for parameter bounds
   - Handles numerical stability through:
     - Proper scaling of the objective function
     - Careful handling of the t/τ term near zero
     - Robust convergence criteria

4. **Convergence Criteria**:
   - Maximum iterations: 1000
   - Function tolerance: 1e-8
   - Parameter tolerance: 1e-8
   - Gradient tolerance: 1e-8

5. **Error Handling**:
   - Raises RuntimeError if optimization fails to converge
   - Provides diagnostic information in debug logs
   - Falls back to raw bootstrapped rates if needed

#### Example Parameter Effects

Here's how different parameter values affect the curve shape:

1. **Level (β₀)**:
   - β₀ = 0.05: Base rate of 5%
   - β₀ = 0.08: Base rate of 8%
   - Effect: Parallel shift of entire curve

2. **Slope (β₁)**:
   - β₁ = -0.02: Steepening of 2%
   - β₁ = 0.02: Inversion of 2%
   - Effect: Spread between short and long rates

3. **Curvature (β₂)**:
   - β₂ = 0.01: Moderate hump
   - β₂ = 0.03: Pronounced hump
   - Effect: Medium-term deviation from trend

4. **Decay (τ)**:
   - τ = 1.0: Fast decay
   - τ = 3.0: Slow decay
   - Effect: Controls the speed of component decay

## Mathematical Details

### 1. Rate Conventions

All rates in the package use continuous compounding convention. The relationship between different compounding conventions is:

- Continuous compounding: r_cont = ln(1 + r_simple)
- Simple compounding: r_simple = exp(r_cont) - 1
- Semi-annual compounding: r_semi = 2 * (exp(r_cont/2) - 1)

### 2. Bootstrapping Algorithm

The bootstrapping algorithm converts par yields to zero-coupon yields using the following steps:

1. For each tenor t:
   - Calculate the present value of all cash flows using known zero rates
   - Solve for the zero rate r that makes the bond price equal to par (100)
   - The equation to solve is:
     ```
     100 = Σ(c/2 * exp(-r_i * t_i)) + 100 * exp(-r * t)
     ```
     where:
     - c is the coupon rate (equal to the par yield)
     - r_i are the known zero rates for shorter tenors
     - t_i are the cash flow times
     - r is the unknown zero rate for tenor t

### 4. Interpolation

For tenors between observed points, linear interpolation is used:

```
r(t) = r₁ + (r₂ - r₁) * (t - t₁) / (t₂ - t₁)
```

where:
- t₁, t₂ are the nearest observed tenors
- r₁, r₂ are the corresponding rates

### 5. Discount Factors

Discount factors are calculated using continuous compounding:

```
DF(t) = exp(-r(t) * t)
```

where:
- r(t) is the spot rate at tenor t
- t is the tenor in years

## API Reference

### RiskFreeCurve

The main class for working with risk-free curves.

```python
class RiskFreeCurve:
    def __init__(self, trade_date: date, smooth: bool = True):
        """
        Initialize a risk-free curve.
        
        Args:
            trade_date: The date for which to build the curve
            smooth: Whether to apply Nelson-Siegel smoothing
        """
        
    def spot(self, t: float) -> float:
        """
        Get the continuously compounded spot rate at tenor t.
        
        Args:
            t: Tenor in years
            
        Returns:
            Continuously compounded spot rate
        """
        
    def discount(self, t: float) -> float:
        """
        Get the discount factor for tenor t.
        
        Args:
            t: Tenor in years
            
        Returns:
            Discount factor exp(-r*t)
        """
        
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the curve to a pandas DataFrame.
        
        Returns:
            DataFrame with columns:
            - tenor: Tenor in years
            - zero: Zero-coupon rate
            - discount: Discount factor
        """
```

## Advanced Usage

### Custom Logging

```python
from riskfree.config import set_level

# Set log level (e.g., "DEBUG", "INFO", "WARNING", "ERROR")
set_level("DEBUG")
```

### Raw Data Access

```python
from riskfree.data.treasury import fetch_par_curve

# Get raw par yield data
par_data = fetch_par_curve(date.today())
```

### Nelson-Siegel Parameters

```python
from riskfree.model.nelson_siegel import fit_nelson_siegel

# Get NS parameters for custom curve fitting
beta = fit_nelson_siegel(times, zeros)
```

## Error Handling

The package includes comprehensive error handling:

- `FetchDataError`: Raised when Treasury data cannot be fetched
- `RuntimeError`: Raised when Nelson-Siegel fitting fails
- `ValueError`: Raised when requesting rates for out-of-range tenors

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
