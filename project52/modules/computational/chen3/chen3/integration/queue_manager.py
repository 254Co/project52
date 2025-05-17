# File: chen3/integration/queue_manager.py
"""
Task queue manager using Celery for asynchronous jobs.
"""
from celery import Celery
import numpy as np

from chen3 import (
    ChenModel, RateParams, EquityParams, ModelParams,
    make_simulator, MonteCarloPricer, Settings
)
from chen3.payoffs import Vanilla

app = Celery('chen3_tasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')

@app.task
def price_task(model_dict, payoff_dict, settings_dict):
    rate = RateParams(**model_dict['rate'])
    equity = EquityParams(**model_dict['equity'])
    corr = np.array(model_dict['corr_matrix'])
    model = ChenModel(ModelParams(rate, equity, corr))
    cfg = Settings(**settings_dict)
    sim = make_simulator(model, cfg)
    paths = sim.generate()
    payoff = Vanilla(**payoff_dict)
    pricer = MonteCarloPricer(
        payoff,
        discount_curve=lambda T: np.exp(-rate.theta * T),
        dt=cfg.dt,
        n_steps=cfg.n_steps
    )
    return float(pricer.price(paths))

@app.task
def simulate_task(model_dict, settings_dict):
    rate = RateParams(**model_dict['rate'])
    equity = EquityParams(**model_dict['equity'])
    corr = np.array(model_dict['corr_matrix'])
    model = ChenModel(ModelParams(rate, equity, corr))
    cfg = Settings(**settings_dict)
    sim = make_simulator(model, cfg)
    paths = sim.generate()
    # Serialize to list of lists
    return paths.tolist()
