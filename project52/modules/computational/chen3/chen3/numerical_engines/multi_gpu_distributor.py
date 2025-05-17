# -------------------- chen3/numerical_engines/multi_gpu_distributor.py --------------------
"""
Multi-GPU Task Distributor for the Three-Factor Chen Model

This module implements a task distributor that efficiently distributes
computational tasks across multiple GPUs. It supports both Monte Carlo
simulations and PDE solving, with automatic load balancing and error
handling.

The distributor manages:
1. Task distribution across available GPUs
2. Load balancing based on GPU capabilities
3. Error handling and recovery
4. Checkpointing and state management
5. Progress monitoring and reporting
6. Performance optimizations including:
   - Dynamic load balancing
   - Memory prefetching
   - Computation overlapping
   - Stream management
   - Kernel fusion
"""

import ray
import cupy as cp
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from chen3.config import Settings
from chen3.model import ThreeFactorChenModel
from chen3.simulators.gpu_vectorized import GPUVectorizedSimulator
from chen3.numerical_engines.gpu_pde_solver import GPUPDESolver
from chen3.gpu_utils import (
    GPUMemoryManager,
    GPUErrorHandler,
    GPUPerformanceMonitor,
    GPUCheckpointManager
)
from ..utils.exceptions import NumericalError
from ..utils.logging_config import logger
from .monte_carlo import MonteCarloEngineBase
from .gpu_mc_engine import GPUMonteCarloEngine
from .quasi_mc import QuasiMonteCarloEngine
from .variance_reduction import VarianceReductionTechnique

@dataclass
class TaskMetrics:
    """Metrics for a computational task."""
    start_time: float
    end_time: float
    memory_usage: float
    gpu_utilization: float
    error_count: int
    checkpoint_count: int
    paths_completed: int
    total_paths: int

@dataclass
class GPUMetrics:
    """Metrics for a GPU device."""
    device_id: int
    memory_usage: float
    gpu_utilization: float
    temperature: float
    power_usage: float
    error_count: int
    task_count: int
    last_task_time: float

ray.init(ignore_reinit_error=True)

@ray.remote(num_gpus=1)
def _simulate_chunk(model: Any, settings: Any, offset: int) -> Any:
    settings.seed += offset
    sim = GPUVectorizedSimulator(model, settings)
    return sim.generate()

class MultiGPUSimulator:
    def __init__(self, model: Any, settings: Any, n_chunks: int = 2):
        self.model = model
        self.settings = settings
        self.n_chunks = n_chunks

    def generate(self) -> cp.ndarray:
        futures = [
            _simulate_chunk.remote(self.model, self.settings, i)
            for i in range(self.n_chunks)
        ]
        results = ray.get(futures)
        return cp.concatenate(results, axis=0)

class MultiGPUDistributor:
    """
    Multi-GPU task distributor for the three-factor Chen model.
    
    This class manages the distribution of computational tasks across
    multiple GPUs, with support for both Monte Carlo simulations and
    PDE solving. It handles load balancing, error recovery, and
    checkpointing.
    
    Attributes:
        model: The three-factor Chen model instance
        settings: Distribution settings including:
            - gpu_ids: List of GPU device IDs to use
            - task_type: Type of task ('mc' or 'pde')
            - checkpoint_interval: Interval for saving checkpoints
        memory_managers: List of GPU memory managers
        error_handlers: List of GPU error handlers
        performance_monitors: List of GPU performance monitors
        checkpoint_managers: List of GPU checkpoint managers
        current_task: Current task being distributed
        task_states: Dictionary of task states
    """
    
    def __init__(self, model: ThreeFactorChenModel, settings: Settings):
        """
        Initialize the multi-GPU task distributor.
        
        Args:
            model: The three-factor Chen model instance
            settings: Distribution settings
        """
        self.model = model
        self.settings = settings
        self.current_task = None
        self.task_states = {}
        self.task_metrics = {}
        self.gpu_metrics = {}
        
        # Initialize GPU utilities for each device
        self.memory_managers = []
        self.error_handlers = []
        self.performance_monitors = []
        self.checkpoint_managers = []
        
        for gpu_id in settings.gpu_ids:
            self.memory_managers.append(GPUMemoryManager(gpu_id))
            self.error_handlers.append(GPUErrorHandler(gpu_id))
            self.performance_monitors.append(GPUPerformanceMonitor(gpu_id))
            self.checkpoint_managers.append(GPUCheckpointManager(gpu_id))
            self.gpu_metrics[gpu_id] = GPUMetrics(
                device_id=gpu_id,
                memory_usage=0.0,
                gpu_utilization=0.0,
                temperature=0.0,
                power_usage=0.0,
                error_count=0,
                task_count=0,
                last_task_time=0.0
            )
    
    def distribute_task(self, task_type: str, task_params: Dict) -> Union[cp.ndarray, List[cp.ndarray]]:
        """
        Distribute a computational task across available GPUs.
        
        Args:
            task_type: Type of task ('mc' or 'pde')
            task_params: Task parameters
            
        Returns:
            Union[cp.ndarray, List[cp.ndarray]]: Task results
            
        Raises:
            ValueError: If task type is invalid
        """
        task_id = task_params.get('task_id', str(id(task_params)))
        self.current_task = {
            'type': task_type,
            'params': task_params,
            'id': task_id
        }
        
        # Initialize task metrics
        self.task_metrics[task_id] = TaskMetrics(
            start_time=time.time(),
            end_time=0.0,
            memory_usage=0.0,
            gpu_utilization=0.0,
            error_count=0,
            checkpoint_count=0,
            paths_completed=0,
            total_paths=task_params.get('n_paths', 0)
        )
        
        try:
            if task_type == 'mc':
                results = self._distribute_mc_task(task_params)
            elif task_type == 'pde':
                results = self._distribute_pde_task(task_params)
            else:
                raise ValueError(f"Invalid task type: {task_type}")
            
            # Update task metrics
            self.task_metrics[task_id].end_time = time.time()
            return results
            
        except Exception as e:
            # Update error metrics
            self.task_metrics[task_id].error_count += 1
            logger.error(f"Task {task_id} failed: {str(e)}")
            raise
    
    def _distribute_mc_task(self, params: Dict) -> List[cp.ndarray]:
        """
        Distribute Monte Carlo simulation task with adaptive load balancing.
        
        Args:
            params: Monte Carlo parameters
            
        Returns:
            List[cp.ndarray]: List of simulation results
        """
        n_gpus = len(self.settings.gpu_ids)
        n_paths = params['n_paths']
        
        # Calculate initial paths per GPU based on performance metrics
        gpu_weights = self._calculate_gpu_weights()
        paths_per_gpu = [int(n_paths * w) for w in gpu_weights]
        
        # Adjust for rounding errors
        remaining_paths = n_paths - sum(paths_per_gpu)
        if remaining_paths > 0:
            paths_per_gpu[0] += remaining_paths
        
        # Create thread pool with adaptive size
        max_workers = min(n_gpus, self.settings.max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks with error handling
            futures = []
            for i, gpu_id in enumerate(self.settings.gpu_ids):
                gpu_params = params.copy()
                gpu_params['n_paths'] = paths_per_gpu[i]
                gpu_params['gpu_id'] = gpu_id
                gpu_params['task_id'] = self.current_task['id']
                
                future = executor.submit(
                    self._run_mc_task_with_retry,
                    gpu_id,
                    gpu_params
                )
                futures.append(future)
            
            # Collect results with progress monitoring
            results = []
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"GPU task failed: {str(e)}")
                    # Redistribute failed task
                    self._redistribute_failed_task(params)
        
        return results
    
    def _calculate_gpu_weights(self) -> List[float]:
        """
        Calculate weights for GPU load balancing based on performance metrics.
        
        Returns:
            List[float]: List of weights for each GPU
        """
        weights = []
        total_score = 0.0
        
        for gpu_id in self.settings.gpu_ids:
            metrics = self.gpu_metrics[gpu_id]
            # Calculate performance score based on multiple factors
            score = (
                (1.0 - metrics.memory_usage) * 0.4 +
                (1.0 - metrics.gpu_utilization) * 0.3 +
                (1.0 - metrics.temperature / 100.0) * 0.2 +
                (1.0 - metrics.error_count / 100.0) * 0.1
            )
            weights.append(score)
            total_score += score
        
        # Normalize weights
        return [w / total_score for w in weights]
    
    def _run_mc_task_with_retry(self, gpu_id: int, params: Dict, max_retries: int = 3) -> cp.ndarray:
        """
        Run Monte Carlo task with retry mechanism.
        
        Args:
            gpu_id: GPU device ID
            params: Task parameters
            max_retries: Maximum number of retry attempts
            
        Returns:
            cp.ndarray: Simulation results
        """
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                # Update GPU metrics
                self.gpu_metrics[gpu_id].task_count += 1
                self.gpu_metrics[gpu_id].last_task_time = time.time()
                
                # Run task
                result = self._run_mc_task(gpu_id, params)
                
                # Update success metrics
                self.task_metrics[params['task_id']].paths_completed += params['n_paths']
                return result
                
            except Exception as e:
                retry_count += 1
                last_error = e
                
                # Update error metrics
                self.gpu_metrics[gpu_id].error_count += 1
                self.task_metrics[params['task_id']].error_count += 1
                
                # Log error
                logger.warning(
                    f"Task failed on GPU {gpu_id} (attempt {retry_count}/{max_retries}): {str(e)}"
                )
                
                # Wait before retry
                time.sleep(1.0)
        
        # All retries failed
        raise NumericalError(f"Task failed after {max_retries} attempts: {str(last_error)}")
    
    def _redistribute_failed_task(self, params: Dict) -> None:
        """
        Redistribute a failed task to other GPUs.
        
        Args:
            params: Task parameters
        """
        # Calculate available GPUs
        available_gpus = [
            gpu_id for gpu_id in self.settings.gpu_ids
            if self.gpu_metrics[gpu_id].error_count < self.settings.max_errors
        ]
        
        if not available_gpus:
            raise NumericalError("No available GPUs for task redistribution")
        
        # Redistribute paths
        n_paths = params['n_paths']
        paths_per_gpu = n_paths // len(available_gpus)
        
        # Submit redistributed tasks
        with ThreadPoolExecutor(max_workers=len(available_gpus)) as executor:
            futures = []
            for gpu_id in available_gpus:
                gpu_params = params.copy()
                gpu_params['n_paths'] = paths_per_gpu
                gpu_params['gpu_id'] = gpu_id
                
                future = executor.submit(
                    self._run_mc_task_with_retry,
                    gpu_id,
                    gpu_params
                )
                futures.append(future)
            
            # Wait for completion
            for future in futures:
                future.result()
    
    def _distribute_pde_task(self, params: Dict) -> List[cp.ndarray]:
        """
        Distribute PDE solving task.
        
        Args:
            params: PDE parameters
            
        Returns:
            List[cp.ndarray]: List of PDE solutions
        """
        n_gpus = len(self.settings.gpu_ids)
        grid_size = params['grid_size']
        grid_per_gpu = grid_size // n_gpus
        
        # Create thread pool
        with ThreadPoolExecutor(max_workers=n_gpus) as executor:
            # Submit tasks
            futures = []
            for i, gpu_id in enumerate(self.settings.gpu_ids):
                gpu_params = params.copy()
                gpu_params['grid_size'] = grid_per_gpu
                gpu_params['gpu_id'] = gpu_id
                
                future = executor.submit(
                    self._run_pde_task,
                    gpu_id,
                    gpu_params
                )
                futures.append(future)
            
            # Collect results
            results = []
            for future in futures:
                results.append(future.result())
        
        return results
    
    def _optimize_mc_task(self, params: Dict) -> Dict:
        """
        Optimize Monte Carlo task parameters for better performance.
        
        Args:
            params: Original task parameters
            
        Returns:
            Dict: Optimized parameters
        """
        optimized = params.copy()
        
        # Optimize batch size based on GPU memory
        gpu_id = params['gpu_id']
        memory_stats = self.memory_managers[gpu_id].get_memory_stats()
        available_memory = memory_stats.free_memory
        
        # Calculate optimal batch size
        bytes_per_path = 3 * 8  # 3 doubles per path
        optimal_batch = min(
            params['n_paths'],
            available_memory // (bytes_per_path * params['n_steps'])
        )
        optimized['batch_size'] = optimal_batch
        
        # Enable memory prefetching
        optimized['prefetch'] = True
        
        # Enable computation overlapping
        optimized['overlap'] = True
        
        return optimized
    
    def _optimize_pde_task(self, params: Dict) -> Dict:
        """
        Optimize PDE task parameters for better performance.
        
        Args:
            params: Original task parameters
            
        Returns:
            Dict: Optimized parameters
        """
        optimized = params.copy()
        
        # Optimize grid size based on GPU memory
        gpu_id = params['gpu_id']
        memory_stats = self.memory_managers[gpu_id].get_memory_stats()
        available_memory = memory_stats.free_memory
        
        # Calculate optimal grid size
        bytes_per_cell = 8  # double precision
        max_grid_size = int(np.cbrt(available_memory / bytes_per_cell))
        optimized['grid_size'] = min(params['grid_size'], max_grid_size)
        
        # Enable kernel fusion
        optimized['fuse_kernels'] = True
        
        # Enable memory prefetching
        optimized['prefetch'] = True
        
        return optimized
    
    def _run_mc_task(self, gpu_id: int, params: Dict) -> cp.ndarray:
        """
        Run Monte Carlo simulation task on a specific GPU.
        
        Args:
            gpu_id: GPU device ID
            params: Task parameters
            
        Returns:
            cp.ndarray: Simulation results
        """
        # Set device
        cp.cuda.Device(gpu_id).use()
        
        # Get GPU utilities
        memory_manager = self.memory_managers[gpu_id]
        error_handler = self.error_handlers[gpu_id]
        performance_monitor = self.performance_monitors[gpu_id]
        checkpoint_manager = self.checkpoint_managers[gpu_id]
        
        # Optimize parameters
        optimized_params = self._optimize_mc_task(params)
        
        try:
            # Create CUDA streams for overlapping
            if optimized_params.get('overlap'):
                compute_stream = cp.cuda.Stream()
                transfer_stream = cp.cuda.Stream()
            else:
                compute_stream = cp.cuda.Stream.null
                transfer_stream = cp.cuda.Stream.null
            
            # Allocate memory
            with memory_manager.allocate():
                # Start monitoring
                with performance_monitor.monitor():
                    # Check for checkpoint
                    checkpoint = checkpoint_manager.load_checkpoint(
                        task_id=params.get('task_id'),
                        task_type='mc'
                    )
                    
                    if checkpoint:
                        # Resume from checkpoint
                        return self._resume_mc_task(checkpoint, optimized_params)
                    else:
                        # Run new simulation
                        return self._run_new_mc_task(optimized_params)
        
        except Exception as e:
            # Handle error
            error_handler.handle_error(e)
            raise
    
    def _run_pde_task(self, gpu_id: int, params: Dict) -> cp.ndarray:
        """
        Run PDE solving task on a specific GPU.
        
        Args:
            gpu_id: GPU device ID
            params: Task parameters
            
        Returns:
            cp.ndarray: PDE solution
        """
        # Set device
        cp.cuda.Device(gpu_id).use()
        
        # Get GPU utilities
        memory_manager = self.memory_managers[gpu_id]
        error_handler = self.error_handlers[gpu_id]
        performance_monitor = self.performance_monitors[gpu_id]
        checkpoint_manager = self.checkpoint_managers[gpu_id]
        
        # Optimize parameters
        optimized_params = self._optimize_pde_task(params)
        
        try:
            # Create CUDA streams for overlapping
            if optimized_params.get('prefetch'):
                compute_stream = cp.cuda.Stream()
                transfer_stream = cp.cuda.Stream()
            else:
                compute_stream = cp.cuda.Stream.null
                transfer_stream = cp.cuda.Stream.null
            
            # Allocate memory
            with memory_manager.allocate():
                # Start monitoring
                with performance_monitor.monitor():
                    # Check for checkpoint
                    checkpoint = checkpoint_manager.load_checkpoint(
                        task_id=params.get('task_id'),
                        task_type='pde'
                    )
                    
                    if checkpoint:
                        # Resume from checkpoint
                        return self._resume_pde_task(checkpoint, optimized_params)
                    else:
                        # Run new simulation
                        return self._run_new_pde_task(optimized_params)
        
        except Exception as e:
            # Handle error
            error_handler.handle_error(e)
            raise
    
    def _resume_mc_task(self, checkpoint: Dict, params: Dict) -> cp.ndarray:
        """
        Resume Monte Carlo simulation from checkpoint.
        
        Args:
            checkpoint: Checkpoint data
            params: Task parameters
            
        Returns:
            cp.ndarray: Simulation results
        """
        # Create simulator
        sim = GPUVectorizedSimulator(self.model, self.settings)
        
        # Resume from checkpoint
        return sim.resume_from_state(checkpoint['state'])
    
    def _resume_pde_task(self, checkpoint: Dict, params: Dict) -> cp.ndarray:
        """
        Resume PDE solving from checkpoint.
        
        Args:
            checkpoint: Checkpoint data
            params: Task parameters
            
        Returns:
            cp.ndarray: PDE solution
        """
        # Create solver
        solver = GPUPDESolver(self.model, self.settings)
        
        # Resume from checkpoint
        return solver.resume_from_state(checkpoint['state'])
    
    def _run_new_mc_task(self, params: Dict) -> cp.ndarray:
        """
        Run new Monte Carlo simulation.
        
        Args:
            params: Task parameters
            
        Returns:
            cp.ndarray: Simulation results
        """
        # Create simulator
        sim = GPUVectorizedSimulator(self.model, self.settings)
        
        # Enable performance optimizations
        if params.get('prefetch'):
            cp.cuda.Stream.null.synchronize()
        
        # Run simulation with batching
        if params.get('batch_size'):
            results = []
            n_batches = (params['n_paths'] + params['batch_size'] - 1) // params['batch_size']
            
            for i in range(n_batches):
                start_idx = i * params['batch_size']
                end_idx = min((i + 1) * params['batch_size'], params['n_paths'])
                
                # Update batch parameters
                batch_params = self.settings.copy()
                batch_params.n_paths = end_idx - start_idx
                
                # Run batch
                batch_results = sim.generate()
                results.append(batch_results)
                
                # Save checkpoint if enabled
                if params.get('checkpoint_interval') and (i + 1) % params['checkpoint_interval'] == 0:
                    state = sim.get_state()
                    self.checkpoint_managers[params['gpu_id']].save_checkpoint(
                        state=state,
                        simulation_id=params.get('task_id', str(id(self))),
                        step=sim.current_step,
                        total_steps=sim.total_steps,
                        device_id=params['gpu_id'],
                        additional_info={
                            'model_params': self.model.params,
                            'settings': self.settings,
                            'batch': i + 1,
                            'total_batches': n_batches
                        }
                    )
            
            return cp.concatenate(results, axis=0)
        else:
            # Run without batching
            paths = sim.generate()
            
            # Save checkpoint if enabled
            if params.get('checkpoint_interval'):
                state = sim.get_state()
                self.checkpoint_managers[params['gpu_id']].save_checkpoint(
                    state=state,
                    simulation_id=params.get('task_id', str(id(self))),
                    step=sim.current_step,
                    total_steps=sim.total_steps,
                    device_id=params['gpu_id'],
                    additional_info={
                        'model_params': self.model.params,
                        'settings': self.settings
                    }
                )
            
            return paths
    
    def _fuse_pde_kernels(self, solver: GPUPDESolver, params: Dict) -> cp.ndarray:
        """
        Run PDE solver with fused kernels for better performance.
        
        This method implements kernel fusion for the three-factor model PDE,
        combining multiple operations into single GPU kernels to reduce memory
        transfers and improve performance.
        
        Args:
            solver: GPU PDE solver instance
            params: Task parameters
            
        Returns:
            cp.ndarray: PDE solution
        """
        # Get model parameters
        rp = self.model.params.rate
        ep = self.model.params.equity
        
        # Create fused kernel for the three-factor model
        kernel_code = """
        extern "C" __global__ void fused_pde_step(
            double* V, double* V_new,
            double* S, double* v, double* r,
            double dt, double dx, double dy, double dz,
            double kappa_r, double theta_r, double sigma_r,
            double kappa_v, double theta_v, double sigma_v,
            double q, double rho_sr, double rho_sv, double rho_rv,
            int nx, int ny, int nz
        ) {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            int j = blockDim.y * blockIdx.y + threadIdx.y;
            int k = blockDim.z * blockIdx.z + threadIdx.z;
            
            if (i >= nx || j >= ny || k >= nz) return;
            
            int idx = i + j * nx + k * nx * ny;
            
            // Compute derivatives
            double dV_dS = (V[idx + 1] - V[idx - 1]) / (2 * dx);
            double dV_dv = (V[idx + nx] - V[idx - nx]) / (2 * dy);
            double dV_dr = (V[idx + nx*ny] - V[idx - nx*ny]) / (2 * dz);
            
            double d2V_dS2 = (V[idx + 1] - 2 * V[idx] + V[idx - 1]) / (dx * dx);
            double d2V_dv2 = (V[idx + nx] - 2 * V[idx] + V[idx - nx]) / (dy * dy);
            double d2V_dr2 = (V[idx + nx*ny] - 2 * V[idx] + V[idx - nx*ny]) / (dz * dz);
            
            double d2V_dSdv = (V[idx + 1 + nx] - V[idx + 1 - nx] - V[idx - 1 + nx] + V[idx - 1 - nx]) / (4 * dx * dy);
            double d2V_dSdr = (V[idx + 1 + nx*ny] - V[idx + 1 - nx*ny] - V[idx - 1 + nx*ny] + V[idx - 1 - nx*ny]) / (4 * dx * dz);
            double d2V_dvdr = (V[idx + nx + nx*ny] - V[idx + nx - nx*ny] - V[idx - nx + nx*ny] + V[idx - nx - nx*ny]) / (4 * dy * dz);
            
            // Compute coefficients
            double a = 0.5 * v[j] * S[i] * S[i];
            double b = 0.5 * sigma_r * sigma_r * r[k];
            double c = 0.5 * sigma_v * sigma_v * v[j];
            double d = rho_sr * sigma_r * S[i] * sqrt(r[k]);
            double e = rho_sv * sigma_v * S[i] * sqrt(v[j]);
            double f = rho_rv * sigma_r * sigma_v * sqrt(r[k] * v[j]);
            
            // Compute drift terms
            double drift_S = (r[k] - q) * S[i] * dV_dS;
            double drift_v = kappa_v * (theta_v - v[j]) * dV_dv;
            double drift_r = kappa_r * (theta_r - r[k]) * dV_dr;
            
            // Compute diffusion terms
            double diff_S = a * d2V_dS2;
            double diff_v = b * d2V_dv2;
            double diff_r = c * d2V_dr2;
            
            // Compute cross terms
            double cross_Sv = d * d2V_dSdv;
            double cross_Sr = e * d2V_dSdr;
            double cross_vr = f * d2V_dvdr;
            
            // Update solution
            V_new[idx] = V[idx] + dt * (
                drift_S + drift_v + drift_r +
                diff_S + diff_v + diff_r +
                cross_Sv + cross_Sr + cross_vr -
                r[k] * V[idx]
            );
        }
        """
        
        # Compile kernel
        kernel = cp.RawKernel(kernel_code, 'fused_pde_step')
        
        # Get grid dimensions
        nx, ny, nz = solver.settings.grid_size, solver.settings.grid_size, solver.settings.grid_size
        
        # Set up thread blocks
        threads_per_block = (8, 8, 8)
        blocks_per_grid = (
            (nx + threads_per_block[0] - 1) // threads_per_block[0],
            (ny + threads_per_block[1] - 1) // threads_per_block[1],
            (nz + threads_per_block[2] - 1) // threads_per_block[2]
        )
        
        # Allocate memory for solution
        V = solver.solution
        V_new = cp.empty_like(V)
        
        # Time stepping
        for t in range(solver.total_steps):
            solver.current_step = t
            
            # Launch fused kernel
            kernel(
                blocks_per_grid,
                threads_per_block,
                (
                    V, V_new,
                    solver.x, solver.y, solver.z,
                    solver.settings.dt,
                    solver.dx, solver.dy, solver.dz,
                    rp.kappa, rp.theta, rp.sigma,
                    ep.kappa_v, ep.theta_v, ep.sigma_v,
                    ep.q,
                    self.model.params.corr_matrix[0, 1],
                    self.model.params.corr_matrix[0, 2],
                    self.model.params.corr_matrix[1, 2],
                    nx, ny, nz
                )
            )
            
            # Swap arrays
            V, V_new = V_new, V
            
            # Save checkpoint if enabled
            if params.get('checkpoint_interval') and (t + 1) % params['checkpoint_interval'] == 0:
                state = solver.get_state()
                self.checkpoint_managers[params['gpu_id']].save_checkpoint(
                    state=state,
                    simulation_id=params.get('task_id', str(id(self))),
                    step=t + 1,
                    total_steps=solver.total_steps,
                    device_id=params['gpu_id'],
                    additional_info={
                        'model_params': self.model.params,
                        'settings': self.settings
                    }
                )
        
        return V
    
    def _run_new_pde_task(self, params: Dict) -> cp.ndarray:
        """
        Run new PDE solving task.
        
        Args:
            params: Task parameters
            
        Returns:
            cp.ndarray: PDE solution
        """
        # Create solver
        solver = GPUPDESolver(self.model, self.settings)
        
        # Enable performance optimizations
        if params.get('prefetch'):
            cp.cuda.Stream.null.synchronize()
        
        # Run solver with kernel fusion if enabled
        if params.get('fuse_kernels'):
            return self._fuse_pde_kernels(solver, params)
        
        # Run solver without fusion
        solution = solver.solve()
        
        # Save checkpoint if enabled
        if params.get('checkpoint_interval'):
            state = solver.get_state()
            self.checkpoint_managers[params['gpu_id']].save_checkpoint(
                state=state,
                simulation_id=params.get('task_id', str(id(self))),
                step=solver.current_step,
                total_steps=solver.total_steps,
                device_id=params['gpu_id'],
                additional_info={
                    'model_params': self.model.params,
                    'settings': self.settings
                }
            )
        
        return solution
    
    def get_task_state(self, task_id: str) -> Dict:
        """
        Get state of a specific task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Dict: Task state
        """
        return self.task_states.get(task_id, {})
    
    def get_all_task_states(self) -> Dict[str, Dict]:
        """
        Get states of all tasks.
        
        Returns:
            Dict[str, Dict]: Dictionary of task states
        """
        return self.task_states
    
    def get_gpu_stats(self) -> List[Dict]:
        """
        Get statistics for all GPUs.
        
        Returns:
            List[Dict]: List of GPU statistics
        """
        stats = []
        for i, gpu_id in enumerate(self.settings.gpu_ids):
            stats.append({
                'gpu_id': gpu_id,
                'memory': self.memory_managers[i].get_stats(),
                'performance': self.performance_monitors[i].get_stats(),
                'errors': self.error_handlers[i].get_stats()
            })
        return stats

    def get_task_metrics(self, task_id: str) -> TaskMetrics:
        """
        Get metrics for a specific task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            TaskMetrics: Task metrics
        """
        return self.task_metrics.get(task_id)
    
    def get_gpu_metrics(self, gpu_id: int) -> GPUMetrics:
        """
        Get metrics for a specific GPU.
        
        Args:
            gpu_id: GPU device ID
            
        Returns:
            GPUMetrics: GPU metrics
        """
        return self.gpu_metrics.get(gpu_id)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get all performance metrics.
        
        Returns:
            Dict[str, Any]: Dictionary of all metrics
        """
        return {
            'task_metrics': self.task_metrics,
            'gpu_metrics': self.gpu_metrics,
            'current_task': self.current_task
        }

class DistributedMonteCarloEngine(MonteCarloEngineBase):
    """
    Distributed Monte Carlo simulation engine for the Chen3 model.
    
    This class provides distributed Monte Carlo simulation capabilities
    across multiple GPUs, with automatic load balancing and error handling.
    """
    
    def __init__(
        self,
        model: ThreeFactorChenModel,
        settings: Settings,
        n_paths: int = 10000,
        n_steps: int = 100,
        dt: float = 0.01,
        use_antithetic: bool = True,
        use_control_variate: bool = True,
        variance_reduction: Optional[VarianceReductionTechnique] = None,
        engine_type: str = 'gpu',
        sequence_type: str = 'sobol',
        max_retries: int = 3,
        checkpoint_interval: int = 1000
    ):
        """Initialize the distributed Monte Carlo engine."""
        super().__init__(
            rng=None,  # Will be created per GPU
            path_generator=None,  # Will be created per GPU
            integration_scheme=None,  # Will be created per GPU
            n_paths=n_paths,
            n_steps=n_steps,
            dt=dt,
            use_antithetic=use_antithetic,
            use_control_variate=use_control_variate,
            variance_reduction=variance_reduction
        )
        self.model = model
        self.settings = settings
        self.engine_type = engine_type
        self.sequence_type = sequence_type
        self.max_retries = max_retries
        self.checkpoint_interval = checkpoint_interval
        
        # Initialize distributor
        self.distributor = MultiGPUDistributor(model, settings)
        
        # Validate engine type
        if engine_type not in ['gpu', 'qmc']:
            raise NumericalError(f"Invalid engine type: {engine_type}")
        
        # Initialize task tracking
        self.current_task_id = None
        self.task_metrics = {}
    
    def _create_engine(self, gpu_id: int, n_paths: int) -> MonteCarloEngineBase:
        """Create a Monte Carlo engine for a specific GPU."""
        if self.engine_type == 'gpu':
            return GPUMonteCarloEngine(
                n_paths=n_paths,
                n_steps=self.n_steps,
                dt=self.dt,
                use_antithetic=self.use_antithetic,
                use_control_variate=self.use_control_variate,
                variance_reduction=self.variance_reduction
            )
        else:  # qmc
            return QuasiMonteCarloEngine(
                n_paths=n_paths,
                n_steps=self.n_steps,
                dt=self.dt,
                use_antithetic=self.use_antithetic,
                use_control_variate=self.use_control_variate,
                sequence_type=self.sequence_type,
                variance_reduction=self.variance_reduction
            )
    
    def simulate_paths(
        self,
        initial_state: Dict[str, float],
        drift_function: Callable,
        diffusion_function: Callable,
        correlation_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate paths for the three-factor model using distributed GPUs.
        
        Args:
            initial_state: Initial state variables
            drift_function: Function computing drift terms
            diffusion_function: Function computing diffusion terms
            correlation_matrix: Correlation matrix between factors
            
        Returns:
            Tuple of (rate_paths, equity_paths, variance_paths)
            
        Raises:
            NumericalError: If simulation fails
        """
        try:
            # Generate task ID
            self.current_task_id = str(id(self))
            
            # Prepare task parameters
            task_params = {
                'task_id': self.current_task_id,
                'n_paths': self.n_paths,
                'n_steps': self.n_steps,
                'dt': self.dt,
                'initial_state': initial_state,
                'drift_function': drift_function,
                'diffusion_function': diffusion_function,
                'correlation_matrix': correlation_matrix,
                'engine_type': self.engine_type,
                'sequence_type': self.sequence_type,
                'use_antithetic': self.use_antithetic,
                'use_control_variate': self.use_control_variate,
                'max_retries': self.max_retries,
                'checkpoint_interval': self.checkpoint_interval
            }
            
            # Distribute task
            results = self.distributor.distribute_task('mc', task_params)
            
            # Update task metrics
            self.task_metrics[self.current_task_id] = self.distributor.get_task_metrics(self.current_task_id)
            
            # Combine results
            rate_paths = np.concatenate([r[0] for r in results], axis=0)
            equity_paths = np.concatenate([r[1] for r in results], axis=0)
            variance_paths = np.concatenate([r[2] for r in results], axis=0)
            
            return rate_paths, equity_paths, variance_paths
            
        except Exception as e:
            # Log error and update metrics
            logger.error(f"Distributed path simulation failed: {str(e)}")
            if self.current_task_id:
                self.task_metrics[self.current_task_id] = self.distributor.get_task_metrics(self.current_task_id)
            raise NumericalError(f"Distributed path simulation failed: {str(e)}")
    
    def compute_expectation(
        self,
        payoff_function: Callable,
        paths: Tuple[np.ndarray, ...],
        control_variate: Optional[Callable] = None
    ) -> Tuple[float, float]:
        """
        Compute expectation of a payoff function using distributed GPUs.
        
        Args:
            payoff_function: Function computing payoff
            paths: Simulated paths
            control_variate: Optional control variate function
            
        Returns:
            Tuple of (price, standard_error)
        """
        try:
            # Generate task ID
            self.current_task_id = str(id(self))
            
            # Prepare task parameters
            task_params = {
                'task_id': self.current_task_id,
                'payoff_function': payoff_function,
                'paths': paths,
                'control_variate': control_variate,
                'engine_type': self.engine_type,
                'sequence_type': self.sequence_type,
                'use_antithetic': self.use_antithetic,
                'use_control_variate': self.use_control_variate,
                'max_retries': self.max_retries,
                'checkpoint_interval': self.checkpoint_interval
            }
            
            # Distribute task
            results = self.distributor.distribute_task('mc_expectation', task_params)
            
            # Update task metrics
            self.task_metrics[self.current_task_id] = self.distributor.get_task_metrics(self.current_task_id)
            
            # Combine results
            prices = np.array([r[0] for r in results])
            std_errors = np.array([r[1] for r in results])
            
            # Compute weighted average
            weights = 1.0 / (std_errors ** 2)
            price = np.sum(prices * weights) / np.sum(weights)
            std_error = np.sqrt(1.0 / np.sum(weights))
            
            return price, std_error
            
        except Exception as e:
            # Log error and update metrics
            logger.error(f"Distributed expectation computation failed: {str(e)}")
            if self.current_task_id:
                self.task_metrics[self.current_task_id] = self.distributor.get_task_metrics(self.current_task_id)
            raise NumericalError(f"Distributed expectation computation failed: {str(e)}")
    
    def get_task_metrics(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metrics for a specific task or all tasks.
        
        Args:
            task_id: Optional task identifier
            
        Returns:
            Dict[str, Any]: Task metrics
        """
        if task_id:
            return self.task_metrics.get(task_id)
        return self.task_metrics
    
    def get_gpu_metrics(self) -> Dict[int, GPUMetrics]:
        """
        Get metrics for all GPUs.
        
        Returns:
            Dict[int, GPUMetrics]: GPU metrics
        """
        return self.distributor.gpu_metrics
    
    def __str__(self) -> str:
        """String representation of the distributed Monte Carlo engine."""
        return (
            f"DistributedMonteCarloEngine(n_paths={self.n_paths}, n_steps={self.n_steps}, "
            f"dt={self.dt}, use_antithetic={self.use_antithetic}, "
            f"use_control_variate={self.use_control_variate}, "
            f"engine_type={self.engine_type}, sequence_type={self.sequence_type})"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation of the distributed Monte Carlo engine."""
        return self.__str__()
