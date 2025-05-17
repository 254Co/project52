# File: chen3/integration/api.py
"""
REST API for Chen3 using FastAPI.

This module provides a REST API interface for the Chen3 pricing engine using FastAPI.
It enables remote access to the model's pricing capabilities through HTTP endpoints,
making it easy to integrate with other systems and services.

Key features:
- RESTful API endpoints for model pricing
- Input validation using Pydantic models
- Integration with FastAPI for automatic OpenAPI documentation
- Error handling and validation
- Support for various pricing configurations

Related modules:
- chen3.calibration: For model calibration workflows
- chen3.greeks: For computing sensitivities
- chen3.numerical_engines: For simulation and pricing engines
- chen3.payoffs: For payoff definitions
- chen3.processes: For stochastic process implementations

Example usage:
    >>> # Start the API server
    >>> import uvicorn
    >>> uvicorn.run(app, host="0.0.0.0", port=8000)

    # Then make a request:
    >>> import requests
    >>> response = requests.post(
    ...     "http://localhost:8000/price",
    ...     json={
    ...         "model": {
    ...             "rate": {"kappa": 0.1, "theta": 0.05, "sigma": 0.1, "r0": 0.03},
    ...             "equity": {"mu": 0.05, "q": 0.02, "v0": 0.04},
    ...             "corr_matrix": [[1.0, -0.5], [-0.5, 1.0]]
    ...         },
    ...         "payoff": {"strike": 100.0, "call": True},
    ...         "settings": {"n_paths": 100000, "n_steps": 252}
    ...     }
    ... )
    >>> print(f"Price: {response.json()['price']:.4f}")

Notes:
    - The API is designed for production use with proper error handling
    - Input validation ensures data integrity
    - The API can be extended with additional endpoints as needed
    - Consider adding authentication for production use
    - The API supports both CPU and GPU backends
"""
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from chen3 import (
    ChenModel,
    EquityParams,
    ModelParams,
    MonteCarloPricer,
    RateParams,
    Settings,
    make_simulator,
)
from chen3.payoffs import Vanilla

app = FastAPI(
    title="Chen3 API",
    description="REST API for the Chen3 pricing engine",
    version="1.0.0",
)


class ModelParamsSchema(BaseModel):
    """
    Schema for model parameters.

    Attributes:
        rate (dict): Interest rate model parameters
            - kappa: Mean reversion speed
            - theta: Long-term mean level
            - sigma: Volatility
            - r0: Initial rate level
        equity (dict): Equity model parameters
            - mu: Drift rate
            - q: Dividend yield
            - v0: Initial variance level
        corr_matrix (list[list[float]]): Correlation matrix between factors
    """

    rate: dict
    equity: dict
    corr_matrix: list[list[float]]


class SettingsSchema(BaseModel):
    """
    Schema for simulation settings.

    Attributes:
        seed (int): Random number generator seed
        backend (str): Computation backend ('cpu' or 'gpu')
        n_paths (int): Number of Monte Carlo paths
        n_steps (int): Number of time steps
        dt (float): Time step size
    """

    seed: int = 42
    backend: str = "cpu"
    n_paths: int = Field(100000, description="Number of Monte Carlo paths")
    n_steps: int = Field(252, description="Number of time steps")
    dt: float = Field(1 / 252, description="Time step size")


class VanillaSchema(BaseModel):
    """
    Schema for vanilla option parameters.

    Attributes:
        strike (float): Option strike price
        call (bool): True for call option, False for put
    """

    strike: float
    call: bool = True


class PriceRequest(BaseModel):
    """
    Schema for pricing request.

    Attributes:
        model (ModelParamsSchema): Model parameters
        payoff (VanillaSchema): Option parameters
        settings (SettingsSchema): Simulation settings
    """

    model: ModelParamsSchema
    payoff: VanillaSchema
    settings: SettingsSchema


class PriceResponse(BaseModel):
    """
    Schema for pricing response.

    Attributes:
        price (float): Computed option price
    """

    price: float


@app.post("/price", response_model=PriceResponse)
def price_endpoint(req: PriceRequest):
    """
    Price a vanilla option using the Chen model.

    This endpoint:
    1. Validates the input parameters
    2. Constructs the model and simulator
    3. Generates Monte Carlo paths
    4. Computes the option price
    5. Returns the result

    Args:
        req (PriceRequest): Pricing request containing model parameters,
                          option parameters, and simulation settings

    Returns:
        PriceResponse: Computed option price

    Raises:
        HTTPException: If input validation fails or computation errors occur

    Example:
        >>> response = requests.post(
        ...     "http://localhost:8000/price",
        ...     json={
        ...         "model": {
        ...             "rate": {"kappa": 0.1, "theta": 0.05, "sigma": 0.1, "r0": 0.03},
        ...             "equity": {"mu": 0.05, "q": 0.02, "v0": 0.04},
        ...             "corr_matrix": [[1.0, -0.5], [-0.5, 1.0]]
        ...         },
        ...         "payoff": {"strike": 100.0, "call": True},
        ...         "settings": {"n_paths": 100000, "n_steps": 252}
        ...     }
        ... )
        >>> print(f"Price: {response.json()['price']:.4f}")
    """
    try:
        mp = req.model
        rate = RateParams(**mp.rate)
        equity = EquityParams(**mp.equity)
        corr = np.array(mp.corr_matrix)
        model = ChenModel(ModelParams(rate, equity, corr))
        cfg = Settings(**req.settings.dict())
        sim = make_simulator(model, cfg)
        paths = sim.generate()
        payoff = Vanilla(**req.payoff.dict())
        pricer = MonteCarloPricer(
            payoff,
            discount_curve=lambda T: np.exp(-rate.theta * T),
            dt=cfg.dt,
            n_steps=cfg.n_steps,
        )
        price = pricer.price(paths)
        return PriceResponse(price=price)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
