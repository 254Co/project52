"""
Custom exceptions for the Chen3 package.

This module defines custom exceptions for different types of errors that can occur
in the Chen3 package, providing more specific error handling and better error messages.
"""


class ChenError(Exception):
    """Base exception class for Chen3 package."""

    pass


class ValidationError(ChenError):
    """Raised when parameter validation fails."""

    pass


class NumericalError(ChenError):
    """Raised when numerical methods fail to converge or produce invalid results."""

    pass


class SimulationError(ChenError):
    """Raised when simulation fails or produces invalid results."""

    pass


class CorrelationError(ChenError):
    """Raised when correlation structure is invalid or fails to generate valid paths."""

    pass


class CalibrationError(ChenError):
    """Raised when model calibration fails to converge or produces invalid parameters."""

    pass


class GPUError(ChenError):
    """Raised when GPU operations fail or GPU is not available."""

    pass


class MemoryError(ChenError):
    """Raised when memory allocation fails or memory limits are exceeded."""

    pass


class ConfigurationError(ChenError):
    """Raised when configuration settings are invalid or incompatible."""

    pass


class PricingError(ChenError):
    """Raised when pricing calculations fail or produce invalid results."""

    pass


class RiskError(ChenError):
    """Raised when risk calculations fail or produce invalid results."""

    pass
