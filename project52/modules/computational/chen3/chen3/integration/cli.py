# File: chen3/integration/cli.py
"""
Command-line interface for Chen3 package.
"""
import json

import click
import numpy as np

from chen3 import (
    ChenModel,
    EquityParams,
    ModelParams,
    MonteCarloPricer,
    RateParams,
    Settings,
    make_simulator,
)
from chen3.payoffs import Asian, Barrier, ConvertibleBond, Vanilla

PRODUCT_MAP = {
    "vanilla": Vanilla,
    "barrier": Barrier,
    "asian": Asian,
    "convertible": ConvertibleBond,
}


@click.group()
def cli():
    """Chen3 command-line utility."""
    pass


@cli.command()
@click.option(
    "--model-params",
    type=click.Path(exists=True),
    required=True,
    help="JSON file with model parameters (rate, equity, corr).",
)
@click.option(
    "--product",
    type=click.Choice(list(PRODUCT_MAP.keys())),
    required=True,
    help="Product type to price.",
)
@click.option(
    "--payoff-params",
    type=click.Path(exists=True),
    required=True,
    help="JSON file with payoff parameters (strike, barrier, etc.).",
)
@click.option(
    "--settings",
    type=click.Path(exists=True),
    required=True,
    help="JSON file with simulation settings.",
)
def price(model_params, product, payoff_params, settings):
    """Price a product with Monte Carlo."""
    # Load inputs
    mp = json.load(open(model_params))
    pp = json.load(open(payoff_params))
    ss = json.load(open(settings))
    # Build model
    rate = RateParams(**mp["rate"])
    equity = EquityParams(**mp["equity"])
    corr = np.array(mp["corr_matrix"])
    params = ModelParams(rate, equity, corr)
    model = ChenModel(params)
    # Settings
    cfg = Settings(**ss)
    # Simulator
    sim = make_simulator(model, cfg)
    paths = sim.generate()
    # Payoff
    PayoffCls = PRODUCT_MAP[product]
    payoff = PayoffCls(**pp)
    pricer = MonteCarloPricer(
        payoff,
        discount_curve=lambda T: np.exp(-mp["rate"]["theta"] * T),
        dt=cfg.dt,
        n_steps=cfg.n_steps,
        use_cv=cfg.use_cv,
        use_aad=cfg.use_aad,
    )
    price = pricer.price(paths)
    click.echo(f"Price: {price:.6f}")


@cli.command()
@click.option(
    "--output",
    type=click.Path(),
    required=True,
    help="File to save simulated paths (npz format).",
)
def simulate(output):
    """Run a default simulation and save paths."""
    click.echo("Simulation stub. Implement as needed.")
