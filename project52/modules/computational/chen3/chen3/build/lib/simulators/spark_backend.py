# File: chen3/simulators/spark_backend.py
"""Spark-based distributed simulator."""
from .core import PathGenerator

class SparkSimulator(PathGenerator):
    def generate(self):
        # Use Spark DataFrame UDFs
        raise NotImplementedError("Spark backend not implemented.")

