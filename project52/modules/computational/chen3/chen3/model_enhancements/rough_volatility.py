# File: chen3/model_enhancements/rough_volatility.py
"""
Rough volatility model enhancement for the Chen3 model.

This module implements a rough volatility model driven by fractional
Brownian motion, which allows for modeling volatility with rough paths
and long-range dependence. This is particularly useful for capturing
the observed roughness of volatility in financial markets.

The implementation provides:
- Fractional Brownian motion-driven volatility
- Rough volatility path simulation
- Configurable Hurst exponent
- Non-negative volatility paths

This enhancement is particularly useful for modeling:
- Volatility roughness
- Long-range dependence
- Volatility clustering
- Market microstructure effects
"""

import numpy as np
from numpy.typing import NDArray
from typing import Generator, Optional

class RoughVolatility:
    """
    Rough volatility model driven by fractional Brownian motion.
    
    This class implements a rough volatility model where the variance
    process is driven by fractional Brownian motion with Hurst exponent
    H < 0.5, leading to rough paths and long-range dependence.
    
    Mathematical Formulation:
    ----------------------
    The variance process v(t) follows:
    
    v(t) = v(0) + η ∫₀ᵗ (t-s)^(H-0.5) dW(s)
    
    where:
    - H: Hurst exponent (0 < H < 0.5)
    - η: Volatility of volatility
    - W(s): Standard Brownian motion
    
    The process has the following properties:
    - Rough paths (Hölder continuous with exponent H)
    - Long-range dependence
    - Non-negative variance
    - Stationary increments
    
    Attributes:
        hurst (float): Hurst exponent (0 < H < 0.5)
        eta (float): Volatility of volatility
        v0 (float): Initial variance
        n_steps (int): Number of time steps in simulation
        kernel (np.ndarray): Precomputed fractional kernel weights
    
    Note:
        The rough volatility model better captures the observed
        roughness of volatility in financial markets compared to
        standard stochastic volatility models.
    """
    
    def __init__(self, hurst: float, eta: float, v0: float, n_steps: int):
        """
        Initialize rough volatility model.
        
        Args:
            hurst (float): Hurst exponent (0 < H < 0.5)
            eta (float): Volatility of volatility (positive)
            v0 (float): Initial variance (positive)
            n_steps (int): Number of time steps in simulation
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not 0 < hurst < 0.5:
            raise ValueError("Hurst exponent must be in (0, 0.5)")
        if eta <= 0:
            raise ValueError("Volatility of volatility must be positive")
        if v0 <= 0:
            raise ValueError("Initial variance must be positive")
        if n_steps <= 0:
            raise ValueError("Number of steps must be positive")
        
        self.hurst = hurst
        self.eta = eta
        self.v0 = v0
        self.n_steps = n_steps
        
        # Precompute fractional kernel weights
        self.kernel = np.array([
            ((i+1)**(hurst-0.5) - i**(hurst-0.5))
            for i in range(n_steps)
        ])

    def simulate(self, dt: float, rng: np.random.Generator) -> NDArray:
        """
        Simulate a rough volatility path.
        
        This method simulates a path of the rough volatility process
        using a discretized version of the fractional integral.
        The simulation ensures non-negative variance values.
        
        Args:
            dt (float): Time step size
            rng (np.random.Generator): Random number generator
            
        Returns:
            NDArray: Array of variance values of shape (n_steps+1,)
                where v[i] is the variance at time step i
                
        Note:
            The simulation uses a discrete convolution with the
            fractional kernel to approximate the fractional integral.
            The variance is kept non-negative by taking the maximum
            with zero at each step.
        """
        # Generate Brownian increments
        dw = rng.standard_normal(self.n_steps) * np.sqrt(dt)
        
        # Compute fractional convolution
        increments = self.eta * np.convolve(dw, self.kernel, mode='full')[:self.n_steps]
        
        # Simulate variance path
        v = np.empty(self.n_steps + 1)
        v[0] = self.v0
        for t in range(1, self.n_steps + 1):
            v[t] = max(v[t-1] + increments[t-1], 0.0)
            
        return v
