# File: chen3/greeks/__init__.py
"""
Greeks Subpackage

This subpackage provides methods for computing sensitivities (Greeks) of financial
derivatives with respect to model parameters in the three-factor Chen model. It
includes multiple approaches for estimating Greeks, including pathwise, likelihood-ratio,
adjoint algorithmic differentiation (AAD), and control variate techniques.

Modules:
- pathwise:        Pathwise (bump-and-revalue) estimators for delta and other Greeks
- lr:              Likelihood-ratio (score-function) estimators for vega and other Greeks
- aad:             Adjoint algorithmic differentiation (AAD) stubs for efficient Greeks
- adjoint_cv:      Control variate variance reduction for adjoint/pathwise methods

Public API:
- delta_pathwise:           Pathwise estimator for delta (∂V/∂S₀)
- vega_likelihood_ratio:    Likelihood-ratio estimator for vega (∂V/∂σ)
- greeks_aad:               Adjoint-mode AAD stub for efficient Greeks
- adjoint_control_variate:  Control variate adjustment for variance reduction

Typical usage:
    from chen3.greeks import delta_pathwise, vega_likelihood_ratio, greeks_aad, adjoint_control_variate
"""
from .pathwise import delta_pathwise
from .lr import vega_likelihood_ratio
from .aad import greeks_aad
from .adjoint_cv import adjoint_control_variate
__all__ = ["delta_pathwise", "vega_likelihood_ratio", "greeks_aad", "adjoint_control_variate"]
