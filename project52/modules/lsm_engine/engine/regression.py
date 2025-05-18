"""
Polynomial regression module for the LSM Engine.

This module implements polynomial regression for estimating continuation values
in the Longstaff-Schwartz Method. It uses scikit-learn's LinearRegression
as the underlying regression model with polynomial basis expansion.
"""

# File: lsm_engine/engine/regression.py
import numpy as np
from sklearn.linear_model import LinearRegression

class PolyRegressor:
    """
    Polynomial regression on state variable(s).
    
    This class implements polynomial regression for estimating continuation values
    in the LSM algorithm. It expands the input features into polynomial terms
    and fits a linear regression model on the expanded features.
    
    Attributes:
        _deg (int): Degree of polynomial expansion
        _lr (LinearRegression): Underlying linear regression model
    """
    def __init__(self, degree: int = 3):
        """
        Initialize the polynomial regressor.
        
        Args:
            degree: Degree of polynomial expansion (default: 3)
            
        Raises:
            TypeError: If degree cannot be converted to an integer
        """
        # Ensure degree is an integer
        if not isinstance(degree, int):
            try:
                degree = int(degree)
            except Exception:
                raise TypeError(f"degree must be an integer, got {type(degree)}")
        self._deg = degree
        self._lr = LinearRegression(fit_intercept=True)

    def _basis(self, x: np.ndarray) -> np.ndarray:
        """
        Expand input features into polynomial basis.
        
        Args:
            x: Input features of shape (n_samples, dim)
            
        Returns:
            np.ndarray: Polynomial basis expansion of shape (n_samples, dim * degree)
        """
        # x shape = (n_samples, dim)
        deg = self._deg
        # build polynomial features
        return np.column_stack([x ** d for d in range(1, deg + 1)])

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the regression model on polynomial basis expansion.
        
        Args:
            x: Input features of shape (n_samples, dim)
            y: Target values of shape (n_samples,)
        """
        # Fit regression on basis-expanded x
        self._lr.fit(self._basis(x), y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict continuation values using the fitted model.
        
        Args:
            x: Input features of shape (n_samples, dim)
            
        Returns:
            np.ndarray: Predicted continuation values of shape (n_samples,)
        """
        # Predict continuation values
        return self._lr.predict(self._basis(x))