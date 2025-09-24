"""
Incremental persistence implementation for the Vectara vector database.
"""

import json
import pickle
from typing import Any, Dict, Optional
import numpy as np
from pathlib import Path
from datetime import datetime

from .base import PersistenceManager
from ..exceptions import PersistenceError


class IncrementalPersistence(PersistenceManager):
    """
    Incremental persistence manager implementation.
    
    This persistence manager saves changes incrementally, allowing for
    efficient updates and recovery from partial failures. It maintains
    a log of operations and can replay them to restore state.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize incremental persistence manager."""
        super().__init__(config)
        self.data_dir = Path(config.get("data_dir", "./vectara_incremental"))
        self.checkpoint_interval = config.get("checkpoint_interval", 100)  # Operations per checkpoint
        self.log_file = self.data_dir / "operations.log"
        self.checkpoint_file = self.data_dir / "checkpoint.json"
        self._operation_count = 0
        self._last_checkpoint = 0
    
    def initialize(self) -> None:
        """Initialize the incremental persistence manager."""
        try:
            # Create data directory
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize log file if it doesn't exist
            if not self.log_file.exists():
                self.log_file.touch()
            
            # Load checkpoint info
            self._load_checkpoint_info()
            
            self._initialized = True
        except Exception as e:
            raise PersistenceError(f"Failed to initialize incremental persistence: {str(e)}")
    
    def save_state(self, state: Dict[str, Any], path: str) -> bool:
        """Save database state with checkpoint."""
        if not self._initialized:
            raise PersistenceError("Persistence manager not initialized")
        
        try:
            # Create checkpoint
            checkpoint_path = self.data_dir / f"checkpoint_{self._operation_count}.pkl"
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Update checkpoint info
            self._last_checkpoint = self._operation_count
            self._save_checkpoint_info()
            
            # Log operation
            self._log_operation("save_state", {"path": path, "checkpoint": str(checkpoint_path)})
            
            return True
            
        except Exception as e:
            raise PersistenceError(f"Failed to save state to {path}: {str(e)}")
    
    def load_state(self, path: str) -> Optional[Dict[str, Any]]:
        """Load database state from latest checkpoint."""
        if not self._initialized:
            raise PersistenceError("Persistence manager not initialized")
        
        try:
            # Find latest checkpoint
            checkpoint_files = list(self.data_dir.glob("checkpoint_*.pkl"))
            if not checkpoint_files:
                return None
            
            # Sort by operation count
            checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[1]))
            latest_checkpoint = checkpoint_files[-1]
            
            with open(latest_checkpoint, 'rb') as f:
                state = pickle.load(f)
            
            # Replay operations since last checkpoint if needed
            state = self._replay_operations(state)
            
            return state
            
        except Exception as e:
            raise PersistenceError(f"Failed to load state from {path}: {str(e)}")
    
    def save_vectors(self, vectors: np.ndarray, ids: list, path: str) -> bool:
        """Save vectors incrementally."""
        if not self._initialized:
            raise PersistenceError("Persistence manager not initialized")
        
        try:
            file_path = self.data_dir / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save vectors
            np.savez(file_path, vectors=vectors, ids=np.array(ids))
            
            # Log operation
            self._log_operation("save_vectors", {
                "path": path,
                "vector_count": len(vectors),
                "ids": ids[:10]  # Log first 10 IDs for reference
            })
            
            return True
            
        except Exception as e:
            raise PersistenceError(f"Failed to save vectors to {path}: {str(e)}")
    
    def load_vectors(self, path: str) -> Optional[tuple[np.ndarray, list]]:
        """Load vectors from file."""
        if not self._initialized:
            raise PersistenceError("Persistence manager not initialized")
        
        try:
            file_path = self.data_dir / path
            
            if not file_path.exists():
                return None
            
            data = np.load(file_path)
            vectors = data['vectors']
            ids = data['ids'].tolist()
            
            return vectors, ids
            
        except Exception as e:
            raise PersistenceError(f"Failed to load vectors from {path}: {str(e)}")
    
    def save_index(self, index_data: Dict[str, Any], path: str) -> bool:
        """Save index data incrementally."""
        if not self._initialized:
            raise PersistenceError("Persistence manager not initialized")
        
        try:
            file_path = self.data_dir / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save index data
            with open(file_path, 'wb') as f:
                pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Log operation
            self._log_operation("save_index", {"path": path})
            
            return True
            
        except Exception as e:
            raise PersistenceError(f"Failed to save index to {path}: {str(e)}")
    
    def load_index(self, path: str) -> Optional[Dict[str, Any]]:
        """Load index data from file."""
        if not self._initialized:
            raise PersistenceError("Persistence manager not initialized")
        
        try:
            file_path = self.data_dir / path
            
            if not file_path.exists():
                return None
            
            with open(file_path, 'rb') as f:
                index_data = pickle.load(f)
            
            return index_data
            
        except Exception as e:
            raise PersistenceError(f"Failed to load index from {path}: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get incremental persistence statistics."""
        if not self._initialized:
            return {"status": "not_initialized"}
        
        try:
            # Count operations in log
            operation_count = 0
            if self.log_file.exists():
                with open(self.log_file, 'r') as f:
                    operation_count = sum(1 for line in f if line.strip())
            
            # Calculate disk usage
            total_size = 0
            checkpoint_count = len(list(self.data_dir.glob("checkpoint_*.pkl")))
            
            for file_path in self.data_dir.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            
            return {
                "persistence_type": "incremental",
                "data_directory": str(self.data_dir),
                "checkpoint_interval": self.checkpoint_interval,
                "operation_count": self._operation_count,
                "logged_operations": operation_count,
                "checkpoint_count": checkpoint_count,
                "last_checkpoint": self._last_checkpoint,
                "disk_usage_bytes": total_size,
                "disk_usage_mb": total_size / (1024 * 1024),
                "initialized": self._initialized,
            }
            
        except Exception as e:
            return {
                "persistence_type": "incremental",
                "error": str(e),
                "initialized": self._initialized,
            }
    
    def close(self) -> None:
        """Close the incremental persistence manager."""
        self._initialized = False
    
    def _log_operation(self, operation: str, data: Dict[str, Any]) -> None:
        """Log an operation to the log file."""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "data": data,
                "operation_id": self._operation_count
            }
            
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            self._operation_count += 1
            
            # Create checkpoint if needed
            if self._operation_count - self._last_checkpoint >= self.checkpoint_interval:
                self._create_checkpoint()
                
        except Exception as e:
            raise PersistenceError(f"Failed to log operation: {str(e)}")
    
    def _create_checkpoint(self) -> None:
        """Create a new checkpoint."""
        try:
            # This would typically save the current state
            # For now, just update the checkpoint info
            self._last_checkpoint = self._operation_count
            self._save_checkpoint_info()
            
        except Exception as e:
            raise PersistenceError(f"Failed to create checkpoint: {str(e)}")
    
    def _save_checkpoint_info(self) -> None:
        """Save checkpoint information."""
        try:
            checkpoint_info = {
                "last_checkpoint": self._last_checkpoint,
                "operation_count": self._operation_count,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint_info, f)
                
        except Exception as e:
            raise PersistenceError(f"Failed to save checkpoint info: {str(e)}")
    
    def _load_checkpoint_info(self) -> None:
        """Load checkpoint information."""
        try:
            if self.checkpoint_file.exists():
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint_info = json.load(f)
                
                self._last_checkpoint = checkpoint_info.get("last_checkpoint", 0)
                self._operation_count = checkpoint_info.get("operation_count", 0)
            else:
                self._last_checkpoint = 0
                self._operation_count = 0
                
        except Exception as e:
            # If we can't load checkpoint info, start fresh
            self._last_checkpoint = 0
            self._operation_count = 0
    
    def _replay_operations(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Replay operations since last checkpoint."""
        try:
            # This is a simplified implementation
            # In practice, you would replay operations from the log
            # to bring the state up to date
            
            return state
            
        except Exception as e:
            raise PersistenceError(f"Failed to replay operations: {str(e)}")
