# File: chen3/simulators/spark_backend.py
"""Spark-based distributed simulator."""
from typing import List, Tuple
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, DoubleType, StructType, StructField
import pandas as pd
from .core import PathGenerator
from ..correlation import cholesky_correlation

class SparkSimulator(PathGenerator):
    def __init__(self, model, settings):
        super().__init__(model, settings)
        self.spark = SparkSession.builder \
            .appName("Chen3SparkSimulator") \
            .getOrCreate()
        
        # Configure Spark for optimal performance
        self.spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
        self.spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "10000")
        
        # Pre-compute correlation matrix decomposition
        self.L = cholesky_correlation(self.model.params.corr_matrix)
        
    def _generate_path_chunk(self, chunk_size: int) -> np.ndarray:
        """Generate a chunk of paths using the same logic as CPU implementation."""
        rp = self.model.params.rate
        ep = self.model.params.equity
        N, M = chunk_size, self.params.n_steps
        dt = self.params.dt
        sqrt_dt = np.sqrt(dt)

        # Initialize state vectors
        S = np.full(N, ep.S0, dtype=float)
        v = np.full(N, ep.v0, dtype=float)
        r = np.full(N, rp.r0, dtype=float)

        paths = np.empty((N, M+1, 3), dtype=float)
        paths[:, 0, :] = np.stack([S, v, r], axis=1)

        for t in range(1, M+1):
            # Generate correlated normals
            Z = np.random.standard_normal((N, 3))
            dW = Z @ self.L.T * sqrt_dt

            # Update short rate
            dr = rp.kappa * (rp.theta - r) * dt \
               + rp.sigma * np.sqrt(np.maximum(r, 0.0)) * dW[:, 0]
            r += dr

            # Update variance
            dv = ep.kappa_v * (ep.theta_v - v) * dt \
               + ep.sigma_v * np.sqrt(np.maximum(v, 0.0)) * dW[:, 1]
            v = np.maximum(v + dv, 0.0)

            # Update stock price
            dS = (r - ep.q) * S * dt \
               + np.sqrt(np.maximum(v, 0.0)) * S * dW[:, 2]
            S += dS

            paths[:, t, 0] = S
            paths[:, t, 1] = v
            paths[:, t, 2] = r

        return paths

    @pandas_udf(ArrayType(ArrayType(ArrayType(DoubleType()))))
    def generate_paths_udf(chunk_sizes: pd.Series) -> pd.Series:
        """Pandas UDF to generate paths for each chunk."""
        def generate_chunk(size):
            return self._generate_path_chunk(size).tolist()
        return chunk_sizes.apply(generate_chunk)

    def generate(self) -> np.ndarray:
        """
        Generate Monte Carlo paths using Spark for distributed computation.
        Returns array of shape (n_paths, n_steps+1, 3) with columns [S, v, r].
        """
        # Calculate optimal chunk size and number of chunks
        total_paths = self.params.n_paths
        optimal_chunk_size = 10000  # Adjust based on memory constraints
        n_chunks = (total_paths + optimal_chunk_size - 1) // optimal_chunk_size
        
        # Create DataFrame with chunk sizes
        chunk_sizes = [optimal_chunk_size] * (n_chunks - 1)
        if total_paths % optimal_chunk_size != 0:
            chunk_sizes.append(total_paths % optimal_chunk_size)
        
        df = self.spark.createDataFrame(
            [(size,) for size in chunk_sizes],
            ["chunk_size"]
        )
        
        # Generate paths in parallel
        paths_df = df.select(self.generate_paths_udf("chunk_size").alias("paths"))
        
        # Collect and combine results
        paths_list = [row.paths for row in paths_df.collect()]
        combined_paths = np.vstack(paths_list)
        
        return combined_paths

    def __del__(self):
        """Clean up Spark session."""
        if hasattr(self, 'spark'):
            self.spark.stop()

