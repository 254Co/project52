from habit_engine.core.constants import ModelParams
from habit_engine.services.sim_engine import run_parallel


def test_run_parallel_shapes():
    out = run_parallel(ModelParams(), n_paths=5000, n_steps=10, n_workers=2)
    assert out["S"].shape == (5000, 11)
    assert out["g_c"].shape == (5000, 10)