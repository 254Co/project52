"""
Command-line interface for the Chen3 package.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import yaml
import numpy as np

from .api import create_model, price_option
from .utils.config import load_config
from .utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Chen3: Three-Factor Chen Model Implementation"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Price command
    price_parser = subparsers.add_parser("price", help="Price an option")
    price_parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (YAML or JSON)",
    )
    price_parser.add_argument(
        "--option-type",
        type=str,
        required=True,
        choices=["call", "put", "american_call", "american_put", "up_and_out_call", "down_and_out_put"],
        help="Type of option to price",
    )
    price_parser.add_argument(
        "--strike",
        type=float,
        required=True,
        help="Strike price",
    )
    price_parser.add_argument(
        "--maturity",
        type=float,
        required=True,
        help="Time to maturity in years",
    )
    price_parser.add_argument(
        "--barrier",
        type=float,
        help="Barrier level for barrier options",
    )
    price_parser.add_argument(
        "--num-paths",
        type=int,
        help="Number of simulation paths",
    )
    price_parser.add_argument(
        "--num-steps",
        type=int,
        help="Number of time steps",
    )
    price_parser.add_argument(
        "--calculate-metrics",
        action="store_true",
        help="Calculate risk metrics",
    )
    price_parser.add_argument(
        "--output",
        type=str,
        help="Output file path (JSON)",
    )

    # Calibrate command
    calibrate_parser = subparsers.add_parser("calibrate", help="Calibrate the model")
    calibrate_parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (YAML or JSON)",
    )
    calibrate_parser.add_argument(
        "--market-data",
        type=str,
        required=True,
        help="Path to market data file (JSON)",
    )
    calibrate_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path for calibrated parameters (YAML or JSON)",
    )

    # Simulate command
    simulate_parser = subparsers.add_parser("simulate", help="Run model simulation")
    simulate_parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (YAML or JSON)",
    )
    simulate_parser.add_argument(
        "--num-paths",
        type=int,
        required=True,
        help="Number of simulation paths",
    )
    simulate_parser.add_argument(
        "--num-steps",
        type=int,
        required=True,
        help="Number of time steps",
    )
    simulate_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path for simulation results (CSV)",
    )

    return parser.parse_args()


def load_market_data(file_path: str) -> dict:
    """Load market data from file."""
    with open(file_path, "r") as f:
        if file_path.endswith(".json"):
            return json.load(f)
        elif file_path.endswith((".yaml", ".yml")):
            return yaml.safe_load(f)
        else:
            raise ValueError("Unsupported file format")


def save_results(results: dict, file_path: str) -> None:
    """Save results to file."""
    with open(file_path, "w") as f:
        if file_path.endswith(".json"):
            json.dump(results, f, indent=2)
        elif file_path.endswith((".yaml", ".yml")):
            yaml.dump(results, f, default_flow_style=False)
        else:
            raise ValueError("Unsupported file format")


def main():
    """Main entry point for the command-line interface."""
    args = parse_args()

    # Configure logging
    configure_logging()

    try:
        # Load configuration
        config = load_config(args.config) if args.config else None

        if args.command == "price":
            # Create model
            model = create_model()

            # Price option
            result = price_option(
                model=model,
                option_type=args.option_type,
                strike=args.strike,
                maturity=args.maturity,
                barrier=args.barrier,
                num_paths=args.num_paths,
                num_steps=args.num_steps,
                calculate_metrics=args.calculate_metrics,
            )

            # Save or print results
            if args.output:
                save_results(result, args.output)
            else:
                print(json.dumps(result, indent=2))

        elif args.command == "calibrate":
            # Create model
            model = create_model()

            # Load market data
            market_data = load_market_data(args.market_data)

            # Calibrate model
            calibrated_model = price_option(
                model=model,
                option_type="call",
                strike=100.0,
                maturity=1.0,
                calibrate=True,
                market_data=market_data,
            )

            # Save calibrated parameters
            save_results(calibrated_model.params.dict(), args.output)

        elif args.command == "simulate":
            # Create model
            model = create_model()

            # Run simulation
            paths = model.simulate(
                num_paths=args.num_paths,
                num_steps=args.num_steps,
            )

            # Save simulation results
            np.savetxt(args.output, paths.reshape(-1, paths.shape[-1]), delimiter=",")

        else:
            logger.error("No command specified")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 