"""
Hybrid storage backend for the Vectara vector database.

Combines memory and disk storage for optimal performance and persistence.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from threading import RLock

from .base import StorageBackend
from .memory import MemoryStorage
from .disk import DiskStorage
from ..exceptions import StorageError


class HybridStorage(StorageBackend):
    """
    Hybrid storage backend that combines memory and disk storage.
    
    This backend uses memory for frequently accessed vectors and disk
    for persistence. It automatically manages data between the two layers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize hybrid storage backend."""
        super().__init__(config)
        
        # Initialize memory and disk backends
        memory_config = config.get("memory", {})
        disk_config = config.get("disk", {})
        
        self.memory_storage = MemoryStorage(memory_config)
        self.disk_storage = DiskStorage(disk_config)
        
        # Configuration
        self.memory_limit = config.get("memory_limit", 10000)  # Max vectors in memory
        self.lock = RLock()
    
    def initialize(self) -> None:
        """Initialize both storage backends."""
        try:
            self.memory_storage.initialize()
            self.disk_storage.initialize()
            self._initialized = True
        except Exception as e:
            raise StorageError(f"Failed to initialize hybrid storage: {str(e)}")
    
    def store_vector(self, id: str, vector: np.ndarray, metadata: Optional[Dict] = None) -> bool:
        """Store a vector in both memory and disk."""
        if not self._initialized:
            raise StorageError("Storage backend not initialized")
        
        try:
            with self.lock:
                # Always store in disk for persistence
                disk_success = self.disk_storage.store_vector(id, vector, metadata)
                
                # Store in memory if we have space
                if disk_success:
                    memory_stats = self.memory_storage.get_stats()
                    current_count = memory_stats.get("vector_count", 0)
                    
                    if current_count < self.memory_limit:
                        self.memory_storage.store_vector(id, vector, metadata)
                    else:
                        # Remove oldest vector from memory (simple LRU approximation)
                        self._evict_from_memory()
                        self.memory_storage.store_vector(id, vector, metadata)
                
                return disk_success
                
        except Exception as e:
            raise StorageError(f"Failed to store vector {id}: {str(e)}")
    
    def retrieve_vector(self, id: str) -> Optional[tuple[np.ndarray, Optional[Dict]]]:
        """Retrieve a vector, preferring memory over disk."""
        if not self._initialized:
            raise StorageError("Storage backend not initialized")
        
        try:
            with self.lock:
                # Try memory first
                result = self.memory_storage.retrieve_vector(id)
                if result is not None:
                    return result
                
                # Fall back to disk
                result = self.disk_storage.retrieve_vector(id)
                if result is not None:
                    # Promote to memory if we have space
                    vector, metadata = result
                    memory_stats = self.memory_storage.get_stats()
                    current_count = memory_stats.get("vector_count", 0)
                    
                    if current_count < self.memory_limit:
                        self.memory_storage.store_vector(id, vector, metadata)
                    else:
                        # Evict and promote
                        self._evict_from_memory()
                        self.memory_storage.store_vector(id, vector, metadata)
                
                return result
                
        except Exception as e:
            raise StorageError(f"Failed to retrieve vector {id}: {str(e)}")
    
    def delete_vector(self, id: str) -> bool:
        """Delete a vector from both memory and disk."""
        if not self._initialized:
            raise StorageError("Storage backend not initialized")
        
        try:
            with self.lock:
                # Delete from both
                memory_deleted = self.memory_storage.delete_vector(id)
                disk_deleted = self.disk_storage.delete_vector(id)
                
                return memory_deleted or disk_deleted
                
        except Exception as e:
            raise StorageError(f"Failed to delete vector {id}: {str(e)}")
    
    def update_vector(self, id: str, vector: np.ndarray, metadata: Optional[Dict] = None) -> bool:
        """Update a vector in both memory and disk."""
        if not self._initialized:
            raise StorageError("Storage backend not initialized")
        
        try:
            with self.lock:
                # Update in disk
                disk_success = self.disk_storage.update_vector(id, vector, metadata)
                
                # Update in memory if it exists there
                if disk_success:
                    self.memory_storage.update_vector(id, vector, metadata)
                
                return disk_success
                
        except Exception as e:
            raise StorageError(f"Failed to update vector {id}: {str(e)}")
    
    def list_vectors(self, limit: Optional[int] = None) -> List[str]:
        """List all vector IDs from disk (authoritative source)."""
        if not self._initialized:
            raise StorageError("Storage backend not initialized")
        
        return self.disk_storage.list_vectors(limit)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get hybrid storage statistics."""
        if not self._initialized:
            return {"status": "not_initialized"}
        
        memory_stats = self.memory_storage.get_stats()
        disk_stats = self.disk_storage.get_stats()
        
        return {
            "backend_type": "hybrid",
            "memory_limit": self.memory_limit,
            "memory": memory_stats,
            "disk": disk_stats,
            "initialized": self._initialized,
        }
    
    def close(self) -> None:
        """Close both storage backends."""
        with self.lock:
            self.memory_storage.close()
            self.disk_storage.close()
            self._initialized = False
    
    def _evict_from_memory(self) -> None:
        """Evict a vector from memory (simple implementation)."""
        vector_ids = self.memory_storage.list_vectors(limit=1)
        if vector_ids:
            self.memory_storage.delete_vector(vector_ids[0])
