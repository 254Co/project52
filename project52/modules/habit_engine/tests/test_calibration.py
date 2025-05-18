import numpy as np

from habit_engine.core.constants import ModelParams
from habit_engine.services.calibration import calibrate


def test_calibrate_tiny():
    target = {"E_r_f": 0.001}
    params = calibrate(target, initial_params=ModelParams(), n_paths=2000, n_steps=60, quiet=True)
    assert isinstance(params, ModelParams)