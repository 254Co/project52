"""
Regression Models for Longstaff-Schwartz Method

This module provides various regression models used in the Longstaff-Schwartz Method (LSM)
for estimating continuation values. The module includes:
- Base abstract regressor class defining the interface
- Polynomial regression with ordinary least squares
- Ridge regression with L2 regularization
- Lasso regression with L1 regularization

These regressors are used to estimate the continuation value of American/Bermudan options
during the backward induction process of the LSM algorithm.
"""

# File: lsm_engine/engine/regression.py
import numpy as np
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression, Ridge, Lasso

class BaseRegressor(ABC):
    """
    Abstract base class for regressors in the LSM engine.
    
    This class defines the interface that all regression models must implement
    for use in the Longstaff-Schwartz Method. The interface consists of two
    main methods: fit() for training the model and predict() for making predictions.
    """
    
    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the regressor to the training data.
        
        Args:
            x: Feature array of shape (n_samples, n_features)
            y: Target array of shape (n_samples,)
        """
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict target values for new features.
        
        Args:
            x: Feature array of shape (n_samples, n_features)
            
        Returns:
            Predicted values of shape (n_samples,)
        """
        pass

class PolyRegressor(BaseRegressor):
    """
    Polynomial regression on state variable(s) using ordinary least squares.
    
    This regressor implements polynomial regression by:
    1. Expanding features into polynomial basis (x, x^2, ..., x^degree)
    2. Fitting a linear regression model on the expanded features
    
    Attributes:
        _deg (int): Degree of the polynomial basis
        _lr (LinearRegression): Underlying linear regression model
    """
    
    def __init__(self, degree: int = 3):
        """
        Initialize polynomial regressor.
        
        Args:
            degree: Maximum degree of polynomial features (default: 3)
            
        Raises:
            TypeError: If degree cannot be converted to integer
        """
        # Ensure integer degree
        if not isinstance(degree, int):
            try:
                degree = int(degree)
            except Exception:
                raise TypeError(f"degree must be an integer, got {type(degree)}")
        self._deg = degree
        self._lr = LinearRegression(fit_intercept=True)

    def _basis(self, x: np.ndarray) -> np.ndarray:
        """
        Build polynomial basis features.
        
        Args:
            x: Input features of shape (n_samples, n_features)
            
        Returns:
            Expanded features of shape (n_samples, n_features * degree)
        """
        # Build polynomial features: x, x^2, ..., x^deg
        return np.column_stack([x ** d for d in range(1, self._deg + 1)])

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit linear regression on polynomial basis-expanded features.
        
        Args:
            x: Feature array of shape (n_samples, n_features)
            y: Target array of shape (n_samples,)
        """
        self._lr.fit(self._basis(x), y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict continuation values using fitted regression.
        
        Args:
            x: Feature array of shape (n_samples, n_features)
            
        Returns:
            Predicted continuation values of shape (n_samples,)
        """
        return self._lr.predict(self._basis(x))

class RidgeRegressor(BaseRegressor):
    """
    Polynomial regression with L2 regularization (Ridge regression).
    
    This regressor combines polynomial feature expansion with Ridge regression,
    which adds L2 regularization to prevent overfitting. The regularization
    strength is controlled by the alpha parameter.
    
    Attributes:
        _poly (PolyRegressor): Polynomial feature generator
        _lr (Ridge): Ridge regression model
    """
    
    def __init__(self, degree: int = 3, alpha: float = 1.0):
        """
        Initialize Ridge regressor.
        
        Args:
            degree: Maximum degree of polynomial features (default: 3)
            alpha: Regularization strength (default: 1.0)
        """
        self._poly = PolyRegressor(degree)
        self._lr = Ridge(alpha=alpha)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit Ridge regression on polynomial basis-expanded features.
        
        Args:
            x: Feature array of shape (n_samples, n_features)
            y: Target array of shape (n_samples,)
        """
        poly = self._poly._basis(x)
        self._lr.fit(poly, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict continuation values using fitted Ridge regression.
        
        Args:
            x: Feature array of shape (n_samples, n_features)
            
        Returns:
            Predicted continuation values of shape (n_samples,)
        """
        return self._lr.predict(self._poly._basis(x))

class LassoRegressor(BaseRegressor):
    """
    Polynomial regression with L1 regularization (Lasso regression).
    
    This regressor combines polynomial feature expansion with Lasso regression,
    which adds L1 regularization to promote sparsity in the coefficients.
    The regularization strength is controlled by the alpha parameter.
    
    Attributes:
        _poly (PolyRegressor): Polynomial feature generator
        _lr (Lasso): Lasso regression model
    """
    
    def __init__(self, degree: int = 3, alpha: float = 1.0):
        """
        Initialize Lasso regressor.
        
        Args:
            degree: Maximum degree of polynomial features (default: 3)
            alpha: Regularization strength (default: 1.0)
        """
        self._poly = PolyRegressor(degree)
        self._lr = Lasso(alpha=alpha)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit Lasso regression on polynomial basis-expanded features.
        
        Args:
            x: Feature array of shape (n_samples, n_features)
            y: Target array of shape (n_samples,)
        """
        poly = self._poly._basis(x)
        self._lr.fit(poly, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict continuation values using fitted Lasso regression.
        
        Args:
            x: Feature array of shape (n_samples, n_features)
            
        Returns:
            Predicted continuation values of shape (n_samples,)
        """
        return self._lr.predict(self._poly._basis(x))

__all__ = [
    "BaseRegressor",
    "PolyRegressor",
    "RidgeRegressor",
    "LassoRegressor",
]