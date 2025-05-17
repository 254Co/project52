"""
Random Number Generation for the Chen3 Model

This module provides random number generation utilities for the Chen3 model,
including normal, uniform, and quasi-random number generators.
"""

from typing import Optional, Tuple, Union
import numpy as np
from scipy.stats import norm
from ..utils.exceptions import NumericalError
from ..utils.logging_config import logger

class RandomNumberGenerator:
    """
    Random number generator for the Chen3 model.
    
    This class provides various methods for generating random numbers,
    including normal, uniform, and quasi-random sequences. It supports
    both standard and antithetic variates.
    
    Features:
    1. Normal random number generation
    2. Uniform random number generation
    3. Quasi-random sequences (Sobol, Halton)
    4. Antithetic variates
    5. Correlation handling
    
    Mathematical Formulation:
    ----------------------
    1. Normal Random Numbers:
       X ~ N(0, 1) using Box-Muller transform
       
    2. Correlated Normal Random Numbers:
       Y = L * X where L is Cholesky factor of correlation matrix
       
    3. Antithetic Variates:
       X_antithetic = -X
       
    4. Quasi-Random Sequences:
       - Sobol sequence: Low-discrepancy sequence
       - Halton sequence: Base-p sequence
    
    Attributes:
        seed (Optional[int]): Random number generator seed
        use_antithetic (bool): Whether to use antithetic variates
        use_quasi_random (bool): Whether to use quasi-random sequences
        quasi_random_type (str): Type of quasi-random sequence
    """
    
    def __init__(
        self,
        seed: Optional[int] = None,
        use_antithetic: bool = True,
        use_quasi_random: bool = False,
        quasi_random_type: str = 'sobol'
    ):
        """
        Initialize the random number generator.
        
        Args:
            seed: Random number generator seed
            use_antithetic: Whether to use antithetic variates
            use_quasi_random: Whether to use quasi-random sequences
            quasi_random_type: Type of quasi-random sequence
        """
        self.seed = seed
        self.use_antithetic = use_antithetic
        self.use_quasi_random = use_quasi_random
        self.quasi_random_type = quasi_random_type
        
        self._validate_initialization()
        self._setup_generator()
    
    def _validate_initialization(self):
        """Validate initialization parameters."""
        if self.use_quasi_random and self.quasi_random_type not in ['sobol', 'halton']:
            raise NumericalError("Invalid quasi-random sequence type")
    
    def _setup_generator(self):
        """Set up the random number generator."""
        if self.seed is not None:
            np.random.seed(self.seed)
    
    def generate_normal(
        self,
        size: Union[int, Tuple[int, ...]],
        correlation_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate normal random numbers.
        
        Args:
            size: Size of the output array
            correlation_matrix: Optional correlation matrix
            
        Returns:
            Array of normal random numbers
            
        Raises:
            NumericalError: If generation fails
        """
        try:
            if self.use_quasi_random:
                return self._generate_quasi_normal(size, correlation_matrix)
            else:
                return self._generate_standard_normal(size, correlation_matrix)
        except Exception as e:
            raise NumericalError(f"Normal random number generation failed: {str(e)}")
    
    def _generate_standard_normal(
        self,
        size: Union[int, Tuple[int, ...]],
        correlation_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Generate standard normal random numbers."""
        # Generate base random numbers
        if isinstance(size, int):
            size = (size,)
        
        n_total = size[0] * (2 if self.use_antithetic else 1)
        new_size = (n_total,) + size[1:]
        
        # Use Box-Muller transform for better quality
        u1 = np.random.uniform(0, 1, new_size)
        u2 = np.random.uniform(0, 1, new_size)
        z = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
        
        # Apply correlation if specified
        if correlation_matrix is not None:
            L = np.linalg.cholesky(correlation_matrix)
            z = z @ L.T
        
        # Apply antithetic variates if enabled
        if self.use_antithetic:
            z = (z[:size[0]] - z[size[0]:]) / np.sqrt(2)
        
        return z
    
    def _generate_quasi_normal(
        self,
        size: Union[int, Tuple[int, ...]],
        correlation_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Generate quasi-random normal numbers."""
        if isinstance(size, int):
            size = (size,)
        
        n_total = size[0] * (2 if self.use_antithetic else 1)
        new_size = (n_total,) + size[1:]
        
        # Generate quasi-random uniform numbers
        if self.quasi_random_type == 'sobol':
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=new_size[-1])
            u = sampler.random(n_total)
        else:  # halton
            from scipy.stats import qmc
            sampler = qmc.Halton(d=new_size[-1])
            u = sampler.random(n_total)
        
        # Transform to normal
        z = norm.ppf(u)
        
        # Apply correlation if specified
        if correlation_matrix is not None:
            L = np.linalg.cholesky(correlation_matrix)
            z = z @ L.T
        
        # Apply antithetic variates if enabled
        if self.use_antithetic:
            z = (z[:size[0]] - z[size[0]:]) / np.sqrt(2)
        
        return z
    
    def generate_uniform(
        self,
        size: Union[int, Tuple[int, ...]],
        low: float = 0.0,
        high: float = 1.0
    ) -> np.ndarray:
        """
        Generate uniform random numbers.
        
        Args:
            size: Size of the output array
            low: Lower bound
            high: Upper bound
            
        Returns:
            Array of uniform random numbers
            
        Raises:
            NumericalError: If generation fails
        """
        try:
            if self.use_quasi_random:
                return self._generate_quasi_uniform(size, low, high)
            else:
                return self._generate_standard_uniform(size, low, high)
        except Exception as e:
            raise NumericalError(f"Uniform random number generation failed: {str(e)}")
    
    def _generate_standard_uniform(
        self,
        size: Union[int, Tuple[int, ...]],
        low: float,
        high: float
    ) -> np.ndarray:
        """Generate standard uniform random numbers."""
        if isinstance(size, int):
            size = (size,)
        
        n_total = size[0] * (2 if self.use_antithetic else 1)
        new_size = (n_total,) + size[1:]
        
        u = np.random.uniform(low, high, new_size)
        
        if self.use_antithetic:
            u = (u[:size[0]] + (low + high - u[size[0]:])) / 2
        
        return u
    
    def _generate_quasi_uniform(
        self,
        size: Union[int, Tuple[int, ...]],
        low: float,
        high: float
    ) -> np.ndarray:
        """Generate quasi-random uniform numbers."""
        if isinstance(size, int):
            size = (size,)
        
        n_total = size[0] * (2 if self.use_antithetic else 1)
        new_size = (n_total,) + size[1:]
        
        if self.quasi_random_type == 'sobol':
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=new_size[-1])
            u = sampler.random(n_total)
        else:  # halton
            from scipy.stats import qmc
            sampler = qmc.Halton(d=new_size[-1])
            u = sampler.random(n_total)
        
        u = low + (high - low) * u
        
        if self.use_antithetic:
            u = (u[:size[0]] + (low + high - u[size[0]:])) / 2
        
        return u
    
    def __str__(self) -> str:
        """String representation of the random number generator."""
        return (
            f"{self.__class__.__name__}("
            f"use_antithetic={self.use_antithetic}, "
            f"use_quasi_random={self.use_quasi_random})"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation of the random number generator."""
        return (
            f"{self.__class__.__name__}("
            f"seed={self.seed}, "
            f"use_antithetic={self.use_antithetic}, "
            f"use_quasi_random={self.use_quasi_random}, "
            f"quasi_random_type={self.quasi_random_type})"
        ) 