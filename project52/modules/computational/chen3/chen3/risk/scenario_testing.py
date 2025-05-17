# File: chen3/risk/scenario_testing.py
"""
Historical and hypothetical scenario analysis.
"""
from typing import Any, Dict, List

import numpy as np


class ScenarioTest:
    """
    Apply a set of named scenarios to model inputs and collect results.
    """

    def __init__(self, scenarios: Dict[str, Dict[str, Any]]):
        """
        scenarios: mapping scenario_name -> parameter shock dict
        e.g. {'2008_crisis': {'equity_shift': -0.4, 'vol_shift':0.3}}
        """
        self.scenarios = scenarios

    def run(self, base_model_fn, *args, **kwargs) -> Dict[str, Any]:
        """
        base_model_fn: callable that prices or simulates under given kwargs
        returns mapping scenario_name -> result
        """
        results: Dict[str, Any] = {}
        for name, shock in self.scenarios.items():
            params = kwargs.copy()
            # apply each shock to relevant keyword
            for key, delta in shock.items():
                if key in params:
                    params[key] += delta
            results[name] = base_model_fn(*args, **params)
        return results
