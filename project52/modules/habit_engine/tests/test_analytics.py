import numpy as np

from habit_engine.core.analytics import Analytics
from habit_engine.core.constants import ModelParams
from habit_engine.core.dynamics import simulate_paths
from habit_engine.core.model import HabitModel


def test_no_nan():
    p = ModelParams()
    paths = simulate_paths(p, n_steps=24, n_paths=1000, rng=np.random.default_rng(1))
    model = HabitModel(p)
    ana = Analytics(model)
    rf = ana.risk_free_rate(paths["g_c"], paths["S"][:, :-1], paths["S"][:, 1:])
    assert np.isfinite(rf).all()