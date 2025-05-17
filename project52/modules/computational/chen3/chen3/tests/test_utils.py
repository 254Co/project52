"""
Test suite for the Chen3 utility modules.

This module contains tests for the utility functions and classes, including
performance monitoring, configuration management, and caching.
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import yaml

from ..utils.cache import Cache, cached, clear_cache, set_cache_dir, set_cache_size, set_cache_ttl
from ..utils.config import (
    ChenConfig,
    ModelConfig,
    NumericalConfig,
    SimulationConfig,
    get_default_config,
    load_config,
)
from ..utils.exceptions import ConfigurationError
from ..utils.logging import configure_logging, get_logger
from ..utils.performance import (
    PerformanceMonitor,
    PerformanceStats,
    get_performance_summary,
    log_performance_summary,
    performance_context,
    reset_performance_metrics,
    track_performance,
)


def test_performance_monitoring():
    """Test performance monitoring functionality."""
    # Create performance monitor
    monitor = PerformanceMonitor()

    # Test timing
    monitor.start("test")
    time.sleep(0.1)
    metrics = monitor.stop("test")
    assert "duration" in metrics
    assert "memory_usage" in metrics
    assert "cpu_percent" in metrics
    assert metrics["duration"] >= 0.1

    # Test decorator
    @track_performance
    def test_func():
        time.sleep(0.1)
        return 42

    result = test_func()
    assert result == 42

    # Test context manager
    with performance_context("test_context"):
        time.sleep(0.1)

    # Test performance summary
    summary = get_performance_summary()
    assert isinstance(summary, dict)
    assert len(summary) > 0

    # Test performance stats
    stats = PerformanceStats.analyze_metrics(summary)
    assert isinstance(stats, dict)
    assert len(stats) > 0

    # Test performance summary formatting
    formatted_summary = PerformanceStats.format_summary(stats)
    assert isinstance(formatted_summary, str)
    assert len(formatted_summary) > 0

    # Test logging performance summary
    log_performance_summary()

    # Test resetting metrics
    reset_performance_metrics()
    assert len(get_performance_summary()) == 0


def test_configuration_management():
    """Test configuration management functionality."""
    # Test default configuration
    config = get_default_config()
    assert isinstance(config, ChenConfig)
    assert isinstance(config.simulation, SimulationConfig)
    assert isinstance(config.numerical, NumericalConfig)
    assert isinstance(config.model, ModelConfig)

    # Test configuration validation
    config.validate()

    # Test invalid configuration
    config.simulation.num_paths = -1
    with pytest.raises(ConfigurationError):
        config.validate()

    # Test configuration serialization
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)

    config_json = config.to_json()
    assert isinstance(config_json, str)

    config_yaml = config.to_yaml()
    assert isinstance(config_yaml, str)

    # Test configuration deserialization
    config_from_dict = ChenConfig.from_dict(config_dict)
    assert isinstance(config_from_dict, ChenConfig)

    config_from_json = ChenConfig.from_json(config_json)
    assert isinstance(config_from_json, ChenConfig)

    config_from_yaml = ChenConfig.from_yaml(config_yaml)
    assert isinstance(config_from_yaml, ChenConfig)

    # Test configuration file operations
    config_file = Path("test_config.yaml")
    try:
        config.save_to_file(config_file)
        assert config_file.exists()

        loaded_config = load_config(config_file)
        assert isinstance(loaded_config, ChenConfig)
        assert loaded_config.simulation.num_paths == config.simulation.num_paths
        assert loaded_config.numerical.tolerance == config.numerical.tolerance
        assert loaded_config.model.correlation_type == config.model.correlation_type
    finally:
        if config_file.exists():
            config_file.unlink()


def test_caching():
    """Test caching functionality."""
    # Create cache
    cache = Cache()

    # Test caching simple values
    cache.set("test_key", 42)
    assert cache.get("test_key") == 42

    # Test caching numpy arrays
    arr = np.array([1, 2, 3])
    cache.set("test_array", arr)
    assert np.array_equal(cache.get("test_array"), arr)

    # Test caching pandas DataFrames
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    cache.set("test_df", df)
    assert df.equals(cache.get("test_df"))

    # Test caching dictionaries
    d = {"a": 1, "b": 2}
    cache.set("test_dict", d)
    assert cache.get("test_dict") == d

    # Test cache decorator
    @cached
    def test_func(x):
        time.sleep(0.1)
        return x * 2

    result1 = test_func(21)
    result2 = test_func(21)
    assert result1 == result2

    # Test cache TTL
    cache = Cache(ttl=timedelta(seconds=1))
    cache.set("test_ttl", 42)
    assert cache.get("test_ttl") == 42
    time.sleep(1.1)
    assert cache.get("test_ttl") is None

    # Test cache size limit
    cache = Cache(max_size=2)
    cache.set("key1", 1)
    cache.set("key2", 2)
    cache.set("key3", 3)
    assert len(cache._cache) <= 2

    # Test cache directory
    test_cache_dir = Path("test_cache")
    try:
        set_cache_dir(test_cache_dir)
        assert test_cache_dir.exists()
    finally:
        if test_cache_dir.exists():
            for file in test_cache_dir.glob("*"):
                file.unlink()
            test_cache_dir.rmdir()

    # Test cache clearing
    cache.set("test_key", 42)
    clear_cache()
    assert cache.get("test_key") is None


def test_logging():
    """Test logging functionality."""
    # Test logger configuration
    configure_logging(level="DEBUG")
    logger = get_logger(__name__)
    assert logger.level == 10  # DEBUG

    # Test logging to file
    log_file = Path("test.log")
    try:
        configure_logging(level="INFO", log_file=str(log_file))
        logger = get_logger(__name__)
        logger.info("Test message")
        assert log_file.exists()
        with open(log_file, "r") as f:
            log_content = f.read()
            assert "Test message" in log_content
    finally:
        if log_file.exists():
            log_file.unlink()

    # Test module-specific logger
    module_logger = get_logger("chen3.model")
    assert module_logger.name == "chen3.model"


def test_error_handling():
    """Test error handling functionality."""
    # Test configuration error
    with pytest.raises(ConfigurationError):
        raise ConfigurationError("Test error")

    # Test cache error
    cache = Cache()
    with pytest.raises(ValueError):
        cache.stop("nonexistent_key")

    # Test performance monitor error
    monitor = PerformanceMonitor()
    with pytest.raises(ValueError):
        monitor.stop("nonexistent_key") 