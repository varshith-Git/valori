"""
Tensor-based persistence implementation for the Vectara vector database.
"""

import pickle
from typing import Any, Dict, Optional
import numpy as np
from pathlib import Path

from .base import PersistenceManager
from ..exceptions import PersistenceError


class TensorPersistence(PersistenceManager):
    """
    Tensor-based persistence manager implementation.
    
    This persistence manager saves vectors and index data using numpy's
    native tensor formats (npy/npz) for efficient storage and loading.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize tensor persistence manager."""
        super().__init__(config)
        self.data_dir = Path(config.get("data_dir", "./vectara_persistence"))
        self.compression = config.get("compression", False)
        self._save_count = 0
        self._load_count = 0
    
    def initialize(self) -> None:
        """Initialize the tensor persistence manager."""
        try:
            # Create data directory
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self._initialized = True
        except Exception as e:
            raise PersistenceError(f"Failed to initialize tensor persistence: {str(e)}")
    
    def save_state(self, state: Dict[str, Any], path: str) -> bool:
        """Save database state using pickle."""
        if not self._initialized:
            raise PersistenceError("Persistence manager not initialized")
        
        try:
            file_path = self.data_dir / path
            
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save state using pickle
            with open(file_path, 'wb') as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            self._save_count += 1
            return True
            
        except Exception as e:
            raise PersistenceError(f"Failed to save state to {path}: {str(e)}")
    
    def load_state(self, path: str) -> Optional[Dict[str, Any]]:
        """Load database state using pickle."""
        if not self._initialized:
            raise PersistenceError("Persistence manager not initialized")
        
        try:
            file_path = self.data_dir / path
            
            if not file_path.exists():
                return None
            
            with open(file_path, 'rb') as f:
                state = pickle.load(f)
            
            self._load_count += 1
            return state
            
        except Exception as e:
            raise PersistenceError(f"Failed to load state from {path}: {str(e)}")
    
    def save_vectors(self, vectors: np.ndarray, ids: list, path: str) -> bool:
        """Save vectors using numpy's native format."""
        if not self._initialized:
            raise PersistenceError("Persistence manager not initialized")
        
        try:
            file_path = self.data_dir / path
            
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if self.compression:
                # Save with compression
                np.savez_compressed(file_path, vectors=vectors, ids=np.array(ids))
            else:
                # Save without compression
                np.savez(file_path, vectors=vectors, ids=np.array(ids))
            
            self._save_count += 1
            return True
            
        except Exception as e:
            raise PersistenceError(f"Failed to save vectors to {path}: {str(e)}")
    
    def load_vectors(self, path: str) -> Optional[tuple[np.ndarray, list]]:
        """Load vectors from numpy format."""
        if not self._initialized:
            raise PersistenceError("Persistence manager not initialized")
        
        try:
            file_path = self.data_dir / path
            
            if not file_path.exists():
                return None
            
            # Load vectors and IDs
            data = np.load(file_path)
            vectors = data['vectors']
            ids = data['ids'].tolist()
            
            self._load_count += 1
            return vectors, ids
            
        except Exception as e:
            raise PersistenceError(f"Failed to load vectors from {path}: {str(e)}")
    
    def save_index(self, index_data: Dict[str, Any], path: str) -> bool:
        """Save index data using pickle."""
        if not self._initialized:
            raise PersistenceError("Persistence manager not initialized")
        
        try:
            file_path = self.data_dir / path
            
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'wb') as f:
                pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            self._save_count += 1
            return True
            
        except Exception as e:
            raise PersistenceError(f"Failed to save index to {path}: {str(e)}")
    
    def load_index(self, path: str) -> Optional[Dict[str, Any]]:
        """Load index data using pickle."""
        if not self._initialized:
            raise PersistenceError("Persistence manager not initialized")
        
        try:
            file_path = self.data_dir / path
            
            if not file_path.exists():
                return None
            
            with open(file_path, 'rb') as f:
                index_data = pickle.load(f)
            
            self._load_count += 1
            return index_data
            
        except Exception as e:
            raise PersistenceError(f"Failed to load index from {path}: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tensor persistence statistics."""
        if not self._initialized:
            return {"status": "not_initialized"}
        
        try:
            # Calculate total disk usage
            total_size = 0
            file_count = 0
            
            for file_path in self.data_dir.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1
            
            return {
                "persistence_type": "tensor",
                "data_directory": str(self.data_dir),
                "compression_enabled": self.compression,
                "save_count": self._save_count,
                "load_count": self._load_count,
                "file_count": file_count,
                "disk_usage_bytes": total_size,
                "disk_usage_mb": total_size / (1024 * 1024),
                "initialized": self._initialized,
            }
            
        except Exception as e:
            return {
                "persistence_type": "tensor",
                "error": str(e),
                "initialized": self._initialized,
            }
    
    def close(self) -> None:
        """Close the tensor persistence manager."""
        self._initialized = False
