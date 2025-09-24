"""
Base index interface for the Vectara vector database.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np

from ..storage.base import StorageBackend


class Index(ABC):
    """
    Abstract base class for vector indices.
    
    Indices are responsible for efficient similarity search over vector data.
    They work with storage backends to persist and retrieve vectors.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the index with configuration."""
        self.config = config
        self.storage_backend: Optional[StorageBackend] = None
        self._initialized = False
    
    @abstractmethod
    def initialize(self, storage_backend: StorageBackend) -> None:
        """
        Initialize the index with a storage backend.
        
        Args:
            storage_backend: Storage backend to use for persistence
        """
        pass
    
    @abstractmethod
    def insert(self, vectors: np.ndarray, metadata: Optional[List[Dict]] = None) -> List[str]:
        """
        Insert vectors into the index.
        
        Args:
            vectors: Array of vectors to insert
            metadata: Optional metadata for each vector
            
        Returns:
            List of IDs for the inserted vectors
        """
        pass
    
    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Dict]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            
        Returns:
            List of search results with distances and metadata
        """
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> bool:
        """
        Delete vectors by their IDs.
        
        Args:
            ids: List of vector IDs to delete
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def update(self, id: str, vector: np.ndarray, metadata: Optional[Dict] = None) -> bool:
        """
        Update a vector by its ID.
        
        Args:
            id: Vector ID to update
            vector: New vector data
            metadata: Optional new metadata
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the index and clean up resources."""
        pass
