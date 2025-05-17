"""
Adaptive time stepping for numerical schemes.

This module provides the AdaptiveTimeStepper class for adaptive time stepping
in numerical schemes for stochastic differential equations.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np


@dataclass
class AdaptiveStepResult:
    """Result of an adaptive time step."""

    success: bool
    dt: float
    error: float
    state: np.ndarray


class AdaptiveTimeStepper:
    """
    Adaptive time stepper for numerical schemes.

    This class implements adaptive time stepping for numerical schemes
    for stochastic differential equations. It adjusts the time step size
    based on local error estimates to maintain a desired error tolerance.

    Attributes:
        min_steps (int): Minimum number of time steps
        max_steps (int): Maximum number of time steps
        error_tolerance (float): Desired error tolerance
        safety_factor (float): Safety factor for step size adjustment
        min_dt (float): Minimum allowed time step size
        max_dt (float): Maximum allowed time step size
    """

    def __init__(
        self,
        min_steps: int = 10,
        max_steps: int = 1000,
        error_tolerance: float = 1e-4,
        safety_factor: float = 0.9,
        min_dt: float = 1e-6,
        max_dt: float = 1.0,
    ):
        """
        Initialize the adaptive time stepper.

        Args:
            min_steps: Minimum number of time steps
            max_steps: Maximum number of time steps
            error_tolerance: Desired error tolerance
            safety_factor: Safety factor for step size adjustment
            min_dt: Minimum allowed time step size
            max_dt: Maximum allowed time step size
        """
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.error_tolerance = error_tolerance
        self.safety_factor = safety_factor
        self.min_dt = min_dt
        self.max_dt = max_dt

    def step(
        self, t: float, state: np.ndarray, step_function: Callable, dt: float
    ) -> AdaptiveStepResult:
        """
        Perform an adaptive time step.

        Args:
            t: Current time
            state: Current state
            step_function: Function to perform a single step
            dt: Current time step size

        Returns:
            Result of the adaptive step
        """
        # Try step with current dt
        try:
            new_state = step_function(t, state, dt)
            error = np.linalg.norm(new_state - state) / np.linalg.norm(state)

            if error <= self.error_tolerance:
                # Step successful, adjust dt for next step
                new_dt = min(
                    self.max_dt,
                    dt * self.safety_factor * (self.error_tolerance / error) ** 0.5,
                )
                return AdaptiveStepResult(True, new_dt, error, new_state)
            else:
                # Step failed, reduce dt and retry
                new_dt = max(
                    self.min_dt,
                    dt * self.safety_factor * (self.error_tolerance / error) ** 0.5,
                )
                if new_dt < self.min_dt:
                    return AdaptiveStepResult(False, dt, error, state)
                return self.step(t, state, step_function, new_dt)
        except Exception as e:
            return AdaptiveStepResult(False, dt, float("inf"), state)

    def integrate(
        self,
        t0: float,
        t1: float,
        state0: np.ndarray,
        step_function: Callable,
        dt0: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate the system from t0 to t1 using adaptive time stepping.

        Args:
            t0: Initial time
            t1: Final time
            state0: Initial state
            step_function: Function to perform a single step
            dt0: Initial time step size (optional)

        Returns:
            Tuple of (times, states)
        """
        if dt0 is None:
            dt0 = (t1 - t0) / self.min_steps

        times = [t0]
        states = [state0]
        t = t0
        state = state0
        dt = dt0

        while t < t1:
            result = self.step(t, state, step_function, dt)

            if not result.success:
                raise RuntimeError("Adaptive time stepping failed")

            t += dt
            state = result.state
            dt = result.dt

            times.append(t)
            states.append(state)

            if len(times) > self.max_steps:
                raise RuntimeError("Maximum number of steps exceeded")

        return np.array(times), np.array(states)
