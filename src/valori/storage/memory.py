"""
In-memory storage backend for the Vectara vector database.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from threading import RLock

from .base import StorageBackend
from ..exceptions import StorageError


class MemoryStorage(StorageBackend):
    """
    In-memory storage backend implementation.
    
    This backend stores vectors and metadata in memory using Python dictionaries
    and numpy arrays. It's fast but not persistent across restarts.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize memory storage backend."""
        super().__init__(config)
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict] = {}
        self.lock = RLock()
        self._vector_count = 0
        self._memory_usage = 0
    
    def initialize(self) -> None:
        """Initialize the memory storage backend."""
        with self.lock:
            self.vectors.clear()
            self.metadata.clear()
            self._vector_count = 0
            self._memory_usage = 0
            self._initialized = True
    
    def store_vector(self, id: str, vector: np.ndarray, metadata: Optional[Dict] = None) -> bool:
        """Store a vector in memory."""
        if not self._initialized:
            raise StorageError("Storage backend not initialized")
        
        try:
            with self.lock:
                # Store vector
                self.vectors[id] = vector.copy()
                
                # Store metadata
                if metadata is not None:
                    self.metadata[id] = metadata.copy()
                else:
                    self.metadata.pop(id, None)  # Remove if exists
                
                # Update stats
                self._vector_count = len(self.vectors)
                self._memory_usage = sum(
                    vector.nbytes for vector in self.vectors.values()
                ) + sum(
                    len(str(meta).encode('utf-8')) for meta in self.metadata.values()
                )
                
                return True
        except Exception as e:
            raise StorageError(f"Failed to store vector {id}: {str(e)}")
    
    def retrieve_vector(self, id: str) -> Optional[tuple[np.ndarray, Optional[Dict]]]:
        """Retrieve a vector from memory."""
        if not self._initialized:
            raise StorageError("Storage backend not initialized")
        
        with self.lock:
            if id not in self.vectors:
                return None
            
            vector = self.vectors[id].copy()
            metadata = self.metadata.get(id)
            return vector, metadata
    
    def delete_vector(self, id: str) -> bool:
        """Delete a vector from memory."""
        if not self._initialized:
            raise StorageError("Storage backend not initialized")
        
        try:
            with self.lock:
                if id in self.vectors:
                    del self.vectors[id]
                    self.metadata.pop(id, None)
                    
                    # Update stats
                    self._vector_count = len(self.vectors)
                    self._memory_usage = sum(
                        vector.nbytes for vector in self.vectors.values()
                    ) + sum(
                        len(str(meta).encode('utf-8')) for meta in self.metadata.values()
                    )
                    
                    return True
                return False
        except Exception as e:
            raise StorageError(f"Failed to delete vector {id}: {str(e)}")
    
    def update_vector(self, id: str, vector: np.ndarray, metadata: Optional[Dict] = None) -> bool:
        """Update a vector in memory."""
        if not self._initialized:
            raise StorageError("Storage backend not initialized")
        
        try:
            with self.lock:
                if id not in self.vectors:
                    return False
                
                # Update vector
                self.vectors[id] = vector.copy()
                
                # Update metadata
                if metadata is not None:
                    self.metadata[id] = metadata.copy()
                else:
                    self.metadata.pop(id, None)
                
                # Update stats
                self._memory_usage = sum(
                    vector.nbytes for vector in self.vectors.values()
                ) + sum(
                    len(str(meta).encode('utf-8')) for meta in self.metadata.values()
                )
                
                return True
        except Exception as e:
            raise StorageError(f"Failed to update vector {id}: {str(e)}")
    
    def list_vectors(self, limit: Optional[int] = None) -> List[str]:
        """List all vector IDs in memory."""
        if not self._initialized:
            raise StorageError("Storage backend not initialized")
        
        with self.lock:
            vector_ids = list(self.vectors.keys())
            if limit is not None:
                vector_ids = vector_ids[:limit]
            return vector_ids
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory storage statistics."""
        if not self._initialized:
            return {"status": "not_initialized"}
        
        with self.lock:
            return {
                "backend_type": "memory",
                "vector_count": self._vector_count,
                "memory_usage_bytes": self._memory_usage,
                "memory_usage_mb": self._memory_usage / (1024 * 1024),
                "initialized": self._initialized,
            }
    
    def close(self) -> None:
        """Close the memory storage backend."""
        with self.lock:
            self.vectors.clear()
            self.metadata.clear()
            self._vector_count = 0
            self._memory_usage = 0
            self._initialized = False
