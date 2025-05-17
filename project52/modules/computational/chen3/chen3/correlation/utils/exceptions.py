"""
Custom exceptions for the correlation module.
"""


class CorrelationError(Exception):
    """Base exception for correlation-related errors."""

    pass


class CorrelationValidationError(CorrelationError):
    """Exception raised for correlation validation errors."""

    pass


class CorrelationComputationError(CorrelationError):
    """Exception raised for correlation computation errors."""

    pass


class CorrelationInitializationError(CorrelationError):
    """Exception raised for correlation initialization errors."""

    pass


class CorrelationParameterError(CorrelationError):
    """Exception raised for correlation parameter errors."""

    pass
