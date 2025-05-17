# File: chen3/simulators/cpu_vectorized.py

import numpy as np
from .core import PathGenerator
from chen3.correlation import cholesky_correlation

class CPUVectorizedSimulator(PathGenerator):
    def generate(self):
        """
        Simulate paths for (S, v, r) with Euler–Maruyama:
          dr = κ_r (θ_r − r) dt + σ_r √r dW₁
          dv = κ_v (θ_v − v) dt + σ_v √v dW₂
          dS = (r − q) S dt + √v S dW₃
        Correlations from model.params.corr_matrix.
        Returns array shape (n_paths, n_steps+1, 3) with columns [S, v, r].
        """
        # unpack
        rp = self.model.params.rate
        ep = self.model.params.equity
        L  = cholesky_correlation(self.model.params.corr_matrix)

        N, M = self.params.n_paths, self.params.n_steps
        dt   = self.params.dt
        sqrt_dt = np.sqrt(dt)

        # init state vectors
        S = np.full(N, ep.S0, dtype=float)
        v = np.full(N, ep.v0, dtype=float)
        r = np.full(N, rp.r0, dtype=float)

        paths = np.empty((N, M+1, 3), dtype=float)
        paths[:, 0, :] = np.stack([S, v, r], axis=1)

        for t in range(1, M+1):
            # correlated normals
            Z  = np.random.standard_normal((N, 3))
            dW = Z @ L.T * sqrt_dt

            # short rate
            dr = rp.kappa * (rp.theta - r) * dt \
               + rp.sigma * np.sqrt(np.maximum(r, 0.0)) * dW[:, 0]
            r += dr

            # variance
            dv = ep.kappa_v * (ep.theta_v - v) * dt \
               + ep.sigma_v * np.sqrt(np.maximum(v, 0.0)) * dW[:, 1]
            v = np.maximum(v + dv, 0.0)

            # stock
            dS = (r - ep.q) * S * dt \
               + np.sqrt(np.maximum(v, 0.0)) * S * dW[:, 2]
            S += dS

            paths[:, t, 0] = S
            paths[:, t, 1] = v
            paths[:, t, 2] = r

        return paths
