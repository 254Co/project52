"""
GPU Checkpoint Management Module

This module provides comprehensive checkpoint management for GPU simulations including:
- State saving and loading
- Distributed checkpointing
- Checkpoint storage management
- Automatic checkpoint scheduling
"""

import hashlib
import json
import logging
import os
import pickle
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple

import cupy as cp
import numpy as np


@dataclass
class CheckpointMetadata:
    """Metadata for a simulation checkpoint."""

    timestamp: datetime
    checkpoint_id: str
    simulation_id: str
    step: int
    total_steps: int
    progress: float
    state_size: int
    checksum: str
    device_id: int
    additional_info: Dict[str, Any]


class GPUCheckpointManager:
    """
    Manages GPU simulation checkpoints.

    This class provides comprehensive checkpoint management for GPU simulations,
    including state saving, loading, and distributed checkpointing. It supports
    automatic checkpoint scheduling and efficient storage management.

    Attributes:
        checkpoint_dir (Path): Directory for storing checkpoints
        metadata_file (Path): File for storing checkpoint metadata
        checkpoints (Dict[str, CheckpointMetadata]): Active checkpoints
        checkpoint_queue (Queue): Queue for checkpoint operations
        logger (logging.Logger): Logger instance
        max_checkpoints (int): Maximum number of checkpoints to keep
    """

    def __init__(self, checkpoint_dir: str = "checkpoints", max_checkpoints: int = 10):
        """
        Initialize the GPU checkpoint manager.

        Args:
            checkpoint_dir (str): Directory for storing checkpoints
            max_checkpoints (int): Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.metadata_file = self.checkpoint_dir / "metadata.json"
        self.checkpoints: Dict[str, CheckpointMetadata] = {}
        self.checkpoint_queue: Queue = Queue()
        self.max_checkpoints = max_checkpoints
        self.logger = logging.getLogger(__name__)

        # Create checkpoint directory if it doesn't exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # Load existing metadata
        self._load_metadata()

    def save_checkpoint(
        self,
        state: Dict[str, cp.ndarray],
        simulation_id: str,
        step: int,
        total_steps: int,
        device_id: int = 0,
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save a simulation checkpoint.

        Args:
            state (Dict[str, cp.ndarray]): Simulation state to save
            simulation_id (str): Unique identifier for the simulation
            step (int): Current simulation step
            total_steps (int): Total number of steps
            device_id (int): GPU device ID
            additional_info (Optional[Dict[str, Any]]): Additional metadata

        Returns:
            str: Checkpoint ID

        Raises:
            RuntimeError: If checkpoint saving fails
        """
        try:
            # Generate checkpoint ID
            checkpoint_id = self._generate_checkpoint_id(simulation_id, step, device_id)

            # Create checkpoint directory
            checkpoint_path = self.checkpoint_dir / checkpoint_id
            checkpoint_path.mkdir(exist_ok=True)

            # Save state arrays
            state_size = 0
            for key, array in state.items():
                array_path = checkpoint_path / f"{key}.npy"
                np.save(array_path, cp.asnumpy(array))
                state_size += array.nbytes

            # Calculate checksum
            checksum = self._calculate_checksum(state)

            # Create metadata
            metadata = CheckpointMetadata(
                timestamp=datetime.now(),
                checkpoint_id=checkpoint_id,
                simulation_id=simulation_id,
                step=step,
                total_steps=total_steps,
                progress=step / total_steps,
                state_size=state_size,
                checksum=checksum,
                device_id=device_id,
                additional_info=additional_info or {},
            )

            # Save metadata
            self.checkpoints[checkpoint_id] = metadata
            self._save_metadata()

            # Clean up old checkpoints
            self._cleanup_old_checkpoints()

            return checkpoint_id

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {str(e)}")
            raise RuntimeError(f"Checkpoint saving failed: {str(e)}")

    def load_checkpoint(
        self, checkpoint_id: str, device_id: int = 0
    ) -> Tuple[Dict[str, cp.ndarray], CheckpointMetadata]:
        """
        Load a simulation checkpoint.

        Args:
            checkpoint_id (str): Checkpoint ID to load
            device_id (int): GPU device ID

        Returns:
            Tuple[Dict[str, cp.ndarray], CheckpointMetadata]: Loaded state and metadata

        Raises:
            RuntimeError: If checkpoint loading fails
        """
        try:
            # Load metadata
            if checkpoint_id not in self.checkpoints:
                raise ValueError(f"Checkpoint {checkpoint_id} not found")

            metadata = self.checkpoints[checkpoint_id]
            checkpoint_path = self.checkpoint_dir / checkpoint_id

            # Load state arrays
            state = {}
            for array_file in checkpoint_path.glob("*.npy"):
                key = array_file.stem
                array = np.load(array_file)
                state[key] = cp.asarray(array, device=device_id)

            # Verify checksum
            if self._calculate_checksum(state) != metadata.checksum:
                raise RuntimeError("Checkpoint checksum verification failed")

            return state, metadata

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {str(e)}")
            raise RuntimeError(f"Checkpoint loading failed: {str(e)}")

    def _generate_checkpoint_id(
        self, simulation_id: str, step: int, device_id: int
    ) -> str:
        """
        Generate a unique checkpoint ID.

        Args:
            simulation_id (str): Simulation ID
            step (int): Simulation step
            device_id (int): GPU device ID

        Returns:
            str: Unique checkpoint ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{simulation_id}_step{step}_device{device_id}_{timestamp}"

    def _calculate_checksum(self, state: Dict[str, cp.ndarray]) -> str:
        """
        Calculate checksum for state arrays.

        Args:
            state (Dict[str, cp.ndarray]): State arrays

        Returns:
            str: Checksum
        """
        hasher = hashlib.sha256()
        for key in sorted(state.keys()):
            array = cp.asnumpy(state[key])
            hasher.update(array.tobytes())
        return hasher.hexdigest()

    def _load_metadata(self) -> None:
        """Load checkpoint metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    data = json.load(f)
                    for checkpoint_id, metadata in data.items():
                        self.checkpoints[checkpoint_id] = CheckpointMetadata(
                            timestamp=datetime.fromisoformat(metadata["timestamp"]),
                            checkpoint_id=metadata["checkpoint_id"],
                            simulation_id=metadata["simulation_id"],
                            step=metadata["step"],
                            total_steps=metadata["total_steps"],
                            progress=metadata["progress"],
                            state_size=metadata["state_size"],
                            checksum=metadata["checksum"],
                            device_id=metadata["device_id"],
                            additional_info=metadata["additional_info"],
                        )
            except Exception as e:
                self.logger.error(f"Failed to load metadata: {str(e)}")

    def _save_metadata(self) -> None:
        """Save checkpoint metadata to file."""
        try:
            data = {
                checkpoint_id: {
                    "timestamp": metadata.timestamp.isoformat(),
                    "checkpoint_id": metadata.checkpoint_id,
                    "simulation_id": metadata.simulation_id,
                    "step": metadata.step,
                    "total_steps": metadata.total_steps,
                    "progress": metadata.progress,
                    "state_size": metadata.state_size,
                    "checksum": metadata.checksum,
                    "device_id": metadata.device_id,
                    "additional_info": metadata.additional_info,
                }
                for checkpoint_id, metadata in self.checkpoints.items()
            }

            with open(self.metadata_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save metadata: {str(e)}")

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints to maintain maximum count."""
        if len(self.checkpoints) <= self.max_checkpoints:
            return

        # Sort checkpoints by timestamp
        sorted_checkpoints = sorted(
            self.checkpoints.items(), key=lambda x: x[1].timestamp
        )

        # Remove oldest checkpoints
        for checkpoint_id, _ in sorted_checkpoints[: -self.max_checkpoints]:
            self._remove_checkpoint(checkpoint_id)

    def _remove_checkpoint(self, checkpoint_id: str) -> None:
        """
        Remove a checkpoint.

        Args:
            checkpoint_id (str): Checkpoint ID to remove
        """
        try:
            # Remove checkpoint directory
            checkpoint_path = self.checkpoint_dir / checkpoint_id
            if checkpoint_path.exists():
                for file in checkpoint_path.glob("*"):
                    file.unlink()
                checkpoint_path.rmdir()

            # Remove from metadata
            if checkpoint_id in self.checkpoints:
                del self.checkpoints[checkpoint_id]
                self._save_metadata()

        except Exception as e:
            self.logger.error(f"Failed to remove checkpoint: {str(e)}")

    def get_checkpoint_info(self, checkpoint_id: str) -> Optional[CheckpointMetadata]:
        """
        Get information about a checkpoint.

        Args:
            checkpoint_id (str): Checkpoint ID

        Returns:
            Optional[CheckpointMetadata]: Checkpoint metadata if found
        """
        return self.checkpoints.get(checkpoint_id)

    def list_checkpoints(
        self, simulation_id: Optional[str] = None
    ) -> List[CheckpointMetadata]:
        """
        List available checkpoints.

        Args:
            simulation_id (Optional[str]): Filter by simulation ID

        Returns:
            List[CheckpointMetadata]: List of checkpoint metadata
        """
        if simulation_id is None:
            return list(self.checkpoints.values())

        return [
            metadata
            for metadata in self.checkpoints.values()
            if metadata.simulation_id == simulation_id
        ]

    def get_latest_checkpoint(self, simulation_id: str) -> Optional[CheckpointMetadata]:
        """
        Get the latest checkpoint for a simulation.

        Args:
            simulation_id (str): Simulation ID

        Returns:
            Optional[CheckpointMetadata]: Latest checkpoint metadata if found
        """
        checkpoints = self.list_checkpoints(simulation_id)
        if not checkpoints:
            return None

        return max(checkpoints, key=lambda x: x.timestamp)
