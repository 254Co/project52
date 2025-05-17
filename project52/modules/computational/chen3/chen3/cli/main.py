# File: chen3/cli/main.py
"""Command-line interface."""
import click

from chen3.config import Settings
from chen3.pricers.mc import MonteCarloPricer
from chen3.simulators import make_simulator


@click.group()
def cli():
    pass


@click.command()
@click.option("--config", type=click.Path(exists=True), help="YAML config file.")
def price(config):
    # load settings, build model, simulate, price
    click.echo("Pricing stub")


cli.add_command(price)
