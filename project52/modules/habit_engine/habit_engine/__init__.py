"""Top‑level convenience exports."""
from importlib.metadata import version

from .core.constants import ModelParams           # noqa: F401
from .core.model import HabitModel               # noqa: F401
from .core.analytics import Analytics            # noqa: F401
from .services.sim_engine import run_parallel    # noqa: F401
from .services.calibration import calibrate      # noqa: F401

__all__ = [
    "ModelParams",
    "HabitModel",
    "Analytics",
    "run_parallel",
    "calibrate",
]

try:
    __version__ = version("habit_engine")
except Exception:
    __version__ = "0.0.0"