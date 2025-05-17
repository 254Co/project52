# File: chen3/simulators/gpu_vectorized.py
"""CuPy based GPU simulator."""
import cupy as cp
from .core import PathGenerator
from chen3.correlation import cholesky_correlation

class GPUVectorizedSimulator(PathGenerator):
    def __init__(self, model, settings):
        super().__init__(model, settings)
        self.rng = cp.random.default_rng(settings.seed)

    def generate(self):
        rp = self.model.params.rate
        ep = self.model.params.equity
        L_cp = cp.asarray(cholesky_correlation(self.model.params.corr_matrix))
        N, M = self.params.n_paths, self.params.n_steps
        dt = self.params.dt
        sqrt_dt = cp.sqrt(dt)
        S = cp.full(N, ep.S0, dtype=float)
        v = cp.full(N, ep.v0, dtype=float)
        r = cp.full(N, rp.r0, dtype=float)
        paths = cp.empty((N, M+1, 3), dtype=float)
        paths[:, 0, :] = cp.stack([S, v, r], axis=1)
        for t in range(1, M+1):
            Z = self.rng.standard_normal((N, 3))
            dW = Z @ L_cp.T * sqrt_dt
            r = r + rp.kappa*(rp.theta - r)*dt + rp.sigma*cp.sqrt(cp.maximum(r,0))*dW[:,0]
            v = cp.maximum(v + ep.kappa_v*(ep.theta_v - v)*dt + ep.sigma_v*cp.sqrt(cp.maximum(v,0))*dW[:,1], 0)
            S = S + (r - ep.q)*S*dt + cp.sqrt(cp.maximum(v,0))*S*dW[:,2]
            paths[:, t, :] = cp.stack([S, v, r], axis=1)
        return paths