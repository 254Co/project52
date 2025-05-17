"""
Custom exceptions for the Chen3 package.

This module defines custom exceptions for different types of errors that can occur
in the Chen3 package, providing more specific error handling and better error messages.
"""


class ChenError(Exception):
    """Base exception for all Chen3 errors."""

    pass


class ValidationError(ChenError):
    """Raised when parameter validation fails."""

    pass


class NumericalError(ChenError):
    """Raised when numerical computations fail."""

    pass


class SimulationError(ChenError):
    """Raised when simulation fails."""

    pass


class CorrelationError(ChenError):
    """Raised when correlation matrix is invalid."""

    pass


class OptionError(ChenError):
    """Raised when option parameters are invalid."""

    pass


class CalibrationError(ChenError):
    """Raised when model calibration fails."""

    pass


class RiskAnalysisError(ChenError):
    """Raised when risk analysis fails."""

    pass


class ConfigurationError(ChenError):
    """Raised when configuration is invalid."""

    pass


class InputError(ChenError):
    """Raised when input parameters are invalid."""

    pass


class StateError(ChenError):
    """Raised when model state is invalid."""

    pass


class ConvergenceError(NumericalError):
    """Raised when numerical methods fail to converge."""

    pass


class StabilityError(NumericalError):
    """Raised when numerical stability conditions are violated."""

    pass


class GPUError(ChenError):
    """Raised when GPU operations fail or GPU is not available."""

    pass


class MemoryError(ChenError):
    """Raised when memory allocation fails or memory limits are exceeded."""

    pass


class PricingError(ChenError):
    """Raised when pricing calculations fail or produce invalid results."""

    pass


class RiskError(ChenError):
    """Raised when risk calculations fail or produce invalid results."""

    pass
