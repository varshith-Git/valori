"""
High-level client for the Vectara vector database.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np

from .base import VectorDB
from .storage.base import StorageBackend
from .indices.base import Index
from .quantization.base import Quantizer
from .persistence.base import PersistenceManager
from .exceptions import ValoriError


class VectorDBClient:
    """
    High-level client for interacting with the Vectara vector database.
    
    This client provides a convenient interface for common vector database
    operations while abstracting away the complexity of underlying components.
    """
    
    def __init__(
        self,
        storage_backend: StorageBackend,
        index: Index,
        quantizer: Optional[Quantizer] = None,
        persistence_manager: Optional[PersistenceManager] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the vector database client.
        
        Args:
            storage_backend: Storage backend implementation
            index: Index implementation for similarity search
            quantizer: Optional quantizer for vector compression
            persistence_manager: Optional persistence manager
            config: Configuration dictionary
        """
        self.storage_backend = storage_backend
        self.index = index
        self.quantizer = quantizer
        self.persistence_manager = persistence_manager
        self.config = config or {}
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize all components."""
        try:
            self.storage_backend.initialize()
            self.index.initialize(self.storage_backend)
            
            if self.quantizer:
                self.quantizer.initialize()
                
            if self.persistence_manager:
                self.persistence_manager.initialize()
                
            self._initialized = True
        except Exception as e:
            raise ValoriError(f"Failed to initialize client: {str(e)}")
    
    def insert(self, vectors: np.ndarray, metadata: Optional[List[Dict]] = None) -> List[str]:
        """Insert vectors into the database."""
        if not self._initialized:
            raise ValoriError("Client not initialized. Call initialize() first.")
        
        return self.index.insert(vectors, metadata)
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Dict]:
        """Search for similar vectors."""
        if not self._initialized:
            raise ValoriError("Client not initialized. Call initialize() first.")
        
        return self.index.search(query_vector, k)
    
    def delete(self, ids: List[str]) -> bool:
        """Delete vectors by their IDs."""
        if not self._initialized:
            raise ValoriError("Client not initialized. Call initialize() first.")
        
        return self.index.delete(ids)
    
    def update(self, id: str, vector: np.ndarray, metadata: Optional[Dict] = None) -> bool:
        """Update a vector by its ID."""
        if not self._initialized:
            raise ValoriError("Client not initialized. Call initialize() first.")
        
        return self.index.update(id, vector, metadata)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not self._initialized:
            raise ValoriError("Client not initialized. Call initialize() first.")
        
        stats = {
            "storage": self.storage_backend.get_stats(),
            "index": self.index.get_stats(),
        }
        
        if self.quantizer:
            stats["quantization"] = self.quantizer.get_stats()
            
        if self.persistence_manager:
            stats["persistence"] = self.persistence_manager.get_stats()
            
        return stats
    
    def close(self) -> None:
        """Close the client and all components."""
        if self.persistence_manager:
            self.persistence_manager.close()
            
        if self.quantizer:
            self.quantizer.close()
            
        self.index.close()
        self.storage_backend.close()
        self._initialized = False
