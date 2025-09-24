"""
Disk-based storage backend for the Vectara vector database.
"""

import os
import pickle
from typing import Any, Dict, List, Optional
import numpy as np
from pathlib import Path

from .base import StorageBackend
from ..exceptions import StorageError


class DiskStorage(StorageBackend):
    """
    Disk-based storage backend implementation.
    
    This backend stores vectors and metadata on disk using pickle files
    and numpy's native serialization format.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize disk storage backend."""
        super().__init__(config)
        self.data_dir = Path(config.get("data_dir", "./vectara_data"))
        self.vectors_dir = self.data_dir / "vectors"
        self.metadata_dir = self.data_dir / "metadata"
        self._vector_count = 0
    
    def initialize(self) -> None:
        """Initialize the disk storage backend."""
        try:
            # Create directories
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.vectors_dir.mkdir(parents=True, exist_ok=True)
            self.metadata_dir.mkdir(parents=True, exist_ok=True)
            
            # Count existing vectors
            self._vector_count = len(list(self.vectors_dir.glob("*.npy")))
            self._initialized = True
            
        except Exception as e:
            raise StorageError(f"Failed to initialize disk storage: {str(e)}")
    
    def store_vector(self, id: str, vector: np.ndarray, metadata: Optional[Dict] = None) -> bool:
        """Store a vector on disk."""
        if not self._initialized:
            raise StorageError("Storage backend not initialized")
        
        try:
            # Sanitize ID for filesystem
            safe_id = self._sanitize_id(id)
            
            # Store vector
            vector_path = self.vectors_dir / f"{safe_id}.npy"
            np.save(vector_path, vector)
            
            # Store metadata
            if metadata is not None:
                metadata_path = self.metadata_dir / f"{safe_id}.pkl"
                with open(metadata_path, 'wb') as f:
                    pickle.dump(metadata, f)
            else:
                # Remove metadata file if exists
                metadata_path = self.metadata_dir / f"{safe_id}.pkl"
                if metadata_path.exists():
                    metadata_path.unlink()
            
            self._vector_count = len(list(self.vectors_dir.glob("*.npy")))
            return True
            
        except Exception as e:
            raise StorageError(f"Failed to store vector {id}: {str(e)}")
    
    def retrieve_vector(self, id: str) -> Optional[tuple[np.ndarray, Optional[Dict]]]:
        """Retrieve a vector from disk."""
        if not self._initialized:
            raise StorageError("Storage backend not initialized")
        
        try:
            safe_id = self._sanitize_id(id)
            
            # Load vector
            vector_path = self.vectors_dir / f"{safe_id}.npy"
            if not vector_path.exists():
                return None
            
            vector = np.load(vector_path)
            
            # Load metadata
            metadata = None
            metadata_path = self.metadata_dir / f"{safe_id}.pkl"
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
            
            return vector, metadata
            
        except Exception as e:
            raise StorageError(f"Failed to retrieve vector {id}: {str(e)}")
    
    def delete_vector(self, id: str) -> bool:
        """Delete a vector from disk."""
        if not self._initialized:
            raise StorageError("Storage backend not initialized")
        
        try:
            safe_id = self._sanitize_id(id)
            
            # Delete vector file
            vector_path = self.vectors_dir / f"{safe_id}.npy"
            vector_deleted = False
            if vector_path.exists():
                vector_path.unlink()
                vector_deleted = True
            
            # Delete metadata file
            metadata_path = self.metadata_dir / f"{safe_id}.pkl"
            if metadata_path.exists():
                metadata_path.unlink()
            
            if vector_deleted:
                self._vector_count = len(list(self.vectors_dir.glob("*.npy")))
                return True
            return False
            
        except Exception as e:
            raise StorageError(f"Failed to delete vector {id}: {str(e)}")
    
    def update_vector(self, id: str, vector: np.ndarray, metadata: Optional[Dict] = None) -> bool:
        """Update a vector on disk."""
        if not self._initialized:
            raise StorageError("Storage backend not initialized")
        
        try:
            safe_id = self._sanitize_id(id)
            
            # Check if vector exists
            vector_path = self.vectors_dir / f"{safe_id}.npy"
            if not vector_path.exists():
                return False
            
            # Update vector
            np.save(vector_path, vector)
            
            # Update metadata
            if metadata is not None:
                metadata_path = self.metadata_dir / f"{safe_id}.pkl"
                with open(metadata_path, 'wb') as f:
                    pickle.dump(metadata, f)
            else:
                # Remove metadata file
                metadata_path = self.metadata_dir / f"{safe_id}.pkl"
                if metadata_path.exists():
                    metadata_path.unlink()
            
            return True
            
        except Exception as e:
            raise StorageError(f"Failed to update vector {id}: {str(e)}")
    
    def list_vectors(self, limit: Optional[int] = None) -> List[str]:
        """List all vector IDs on disk."""
        if not self._initialized:
            raise StorageError("Storage backend not initialized")
        
        try:
            vector_files = list(self.vectors_dir.glob("*.npy"))
            vector_ids = [f.stem for f in vector_files]
            
            if limit is not None:
                vector_ids = vector_ids[:limit]
            
            return vector_ids
            
        except Exception as e:
            raise StorageError(f"Failed to list vectors: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get disk storage statistics."""
        if not self._initialized:
            return {"status": "not_initialized"}
        
        try:
            # Calculate disk usage
            total_size = 0
            for file_path in self.data_dir.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            
            return {
                "backend_type": "disk",
                "vector_count": self._vector_count,
                "data_directory": str(self.data_dir),
                "disk_usage_bytes": total_size,
                "disk_usage_mb": total_size / (1024 * 1024),
                "initialized": self._initialized,
            }
            
        except Exception as e:
            return {
                "backend_type": "disk",
                "vector_count": self._vector_count,
                "error": str(e),
                "initialized": self._initialized,
            }
    
    def close(self) -> None:
        """Close the disk storage backend."""
        self._initialized = False
    
    def _sanitize_id(self, id: str) -> str:
        """Sanitize ID for use as filename."""
        # Replace invalid characters with underscores
        safe_id = "".join(c if c.isalnum() or c in "._-" else "_" for c in id)
        return safe_id
