# File: chen3/calibration/streaming.py
"""Streaming calibration support (forward-looking)."""
from typing import Callable, Dict, Any, Iterable
from .optim import calibrate

class CalibrationStream:
    def __init__(self,
                 source: Iterable[Dict[str, Any]],
                 model_funcs: Dict[str, Any]):
        """
        source:   iterator yielding dicts with keys
                    'params_guess', 'market_data'
        model_funcs: {
          'vol_fn':  f(params, tenors, strikes)->np.ndarray,
          'rate_fn': f(params, maturities)->np.ndarray
        }
        """
        self.source = source
        self.model_funcs = model_funcs

    def run(self, update_callback: Callable[[Dict[str, float]], None]):
        """
        For each new market snapshot:
          1. Calibrate
          2. Call update_callback(new_params)
        """
        for slice_ in self.source:
            pg = slice_["params_guess"]
            md = slice_["market_data"]
            new_p = calibrate(pg, md, self.model_funcs)
            update_callback(new_p)
