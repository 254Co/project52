# File: chen3/integration/__init__.py
"""Integration layer: CLI, REST API, and queue management."""
from .cli import cli
from .api import app as api_app
from .queue_manager import price_task, simulate_task

__all__ = ["cli", "api_app", "price_task", "simulate_task"]
