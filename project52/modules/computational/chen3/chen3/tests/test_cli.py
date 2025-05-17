"""
Test suite for the Chen3 command-line interface.

This module contains tests for the CLI functionality, including option pricing,
model calibration, and simulation commands.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from .. import cli


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary configuration file."""
    config = {
        "simulation": {
            "num_paths": 1000,
            "num_steps": 100,
        },
        "numerical": {
            "tolerance": 1e-6,
            "max_iterations": 1000,
        },
        "model": {
            "rate_params": {
                "kappa": 0.1,
                "theta": 0.05,
                "sigma": 0.1,
                "r0": 0.03,
            },
            "equity_params": {
                "mu": 0.05,
                "q": 0.02,
                "S0": 100.0,
                "v0": 0.04,
                "kappa_v": 2.0,
                "theta_v": 0.04,
                "sigma_v": 0.3,
            },
            "correlation_type": "constant",
            "correlation_params": {"rho": 0.5},
        },
    }
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)
    return config_file


@pytest.fixture
def temp_market_data_file(tmp_path):
    """Create a temporary market data file."""
    market_data = {
        "strikes": [90.0, 100.0, 110.0],
        "maturities": [0.25, 0.5, 1.0],
        "prices": [
            [10.0, 5.0, 2.0],
            [12.0, 7.0, 3.0],
            [15.0, 10.0, 5.0],
        ],
    }
    data_file = tmp_path / "market_data.json"
    with open(data_file, "w") as f:
        json.dump(market_data, f)
    return data_file


def test_parse_args():
    """Test command-line argument parsing."""
    # Test price command
    with patch("sys.argv", ["chen3", "price", "--option-type", "call", "--strike", "100", "--maturity", "1.0"]):
        args = cli.parse_args()
        assert args.command == "price"
        assert args.option_type == "call"
        assert args.strike == 100.0
        assert args.maturity == 1.0

    # Test calibrate command
    with patch(
        "sys.argv",
        ["chen3", "calibrate", "--market-data", "data.json", "--output", "params.yaml"],
    ):
        args = cli.parse_args()
        assert args.command == "calibrate"
        assert args.market_data == "data.json"
        assert args.output == "params.yaml"

    # Test simulate command
    with patch(
        "sys.argv",
        ["chen3", "simulate", "--num-paths", "1000", "--num-steps", "100", "--output", "paths.csv"],
    ):
        args = cli.parse_args()
        assert args.command == "simulate"
        assert args.num_paths == 1000
        assert args.num_steps == 100
        assert args.output == "paths.csv"


def test_load_market_data(temp_market_data_file):
    """Test loading market data from file."""
    # Test JSON file
    data = cli.load_market_data(str(temp_market_data_file))
    assert "strikes" in data
    assert "maturities" in data
    assert "prices" in data

    # Test YAML file
    yaml_file = temp_market_data_file.with_suffix(".yaml")
    with open(yaml_file, "w") as f:
        yaml.dump(data, f)
    data = cli.load_market_data(str(yaml_file))
    assert "strikes" in data
    assert "maturities" in data
    assert "prices" in data

    # Test invalid file format
    with pytest.raises(ValueError):
        cli.load_market_data("data.txt")


def test_save_results(tmp_path):
    """Test saving results to file."""
    results = {"price": 10.0, "delta": 0.5, "gamma": 0.1}

    # Test JSON file
    json_file = tmp_path / "results.json"
    cli.save_results(results, str(json_file))
    with open(json_file, "r") as f:
        saved_data = json.load(f)
    assert saved_data == results

    # Test YAML file
    yaml_file = tmp_path / "results.yaml"
    cli.save_results(results, str(yaml_file))
    with open(yaml_file, "r") as f:
        saved_data = yaml.safe_load(f)
    assert saved_data == results

    # Test invalid file format
    with pytest.raises(ValueError):
        cli.save_results(results, "results.txt")


def test_price_command(tmp_path, temp_config_file):
    """Test the price command."""
    output_file = tmp_path / "price_results.json"

    # Test with configuration file
    with patch(
        "sys.argv",
        [
            "chen3",
            "price",
            "--config",
            str(temp_config_file),
            "--option-type",
            "call",
            "--strike",
            "100",
            "--maturity",
            "1.0",
            "--output",
            str(output_file),
        ],
    ):
        cli.main()
        assert output_file.exists()
        with open(output_file, "r") as f:
            results = json.load(f)
        assert "price" in results

    # Test with metrics calculation
    with patch(
        "sys.argv",
        [
            "chen3",
            "price",
            "--option-type",
            "call",
            "--strike",
            "100",
            "--maturity",
            "1.0",
            "--calculate-metrics",
            "--output",
            str(output_file),
        ],
    ):
        cli.main()
        assert output_file.exists()
        with open(output_file, "r") as f:
            results = json.load(f)
        assert "price" in results
        assert "delta" in results
        assert "gamma" in results


def test_calibrate_command(tmp_path, temp_market_data_file):
    """Test the calibrate command."""
    output_file = tmp_path / "calibrated_params.yaml"

    with patch(
        "sys.argv",
        [
            "chen3",
            "calibrate",
            "--market-data",
            str(temp_market_data_file),
            "--output",
            str(output_file),
        ],
    ):
        cli.main()
        assert output_file.exists()
        with open(output_file, "r") as f:
            params = yaml.safe_load(f)
        assert "rate_params" in params
        assert "equity_params" in params


def test_simulate_command(tmp_path):
    """Test the simulate command."""
    output_file = tmp_path / "simulation_paths.csv"

    with patch(
        "sys.argv",
        [
            "chen3",
            "simulate",
            "--num-paths",
            "100",
            "--num-steps",
            "50",
            "--output",
            str(output_file),
        ],
    ):
        cli.main()
        assert output_file.exists()
        paths = np.loadtxt(output_file, delimiter=",")
        assert paths.shape[0] == 100
        assert paths.shape[1] == 51  # num_steps + 1


def test_error_handling():
    """Test error handling in the CLI."""
    # Test invalid command
    with patch("sys.argv", ["chen3"]):
        with pytest.raises(SystemExit):
            cli.main()

    # Test invalid option type
    with patch(
        "sys.argv",
        ["chen3", "price", "--option-type", "invalid", "--strike", "100", "--maturity", "1.0"],
    ):
        with pytest.raises(SystemExit):
            cli.main()

    # Test missing required arguments
    with patch("sys.argv", ["chen3", "price", "--option-type", "call"]):
        with pytest.raises(SystemExit):
            cli.main()

    # Test invalid file format
    with patch(
        "sys.argv",
        ["chen3", "calibrate", "--market-data", "data.txt", "--output", "params.yaml"],
    ):
        with pytest.raises(SystemExit):
            cli.main() 