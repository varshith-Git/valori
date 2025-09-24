"""
Base storage backend interface for the Vectara vector database.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np


class StorageBackend(ABC):
    """
    Abstract base class for storage backends.
    
    Storage backends are responsible for persisting and retrieving
    vector data and associated metadata.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the storage backend with configuration."""
        self.config = config
        self._initialized = False
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the storage backend."""
        pass
    
    @abstractmethod
    def store_vector(self, id: str, vector: np.ndarray, metadata: Optional[Dict] = None) -> bool:
        """
        Store a vector with its ID and optional metadata.
        
        Args:
            id: Unique identifier for the vector
            vector: Vector data to store
            metadata: Optional metadata dictionary
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def retrieve_vector(self, id: str) -> Optional[tuple[np.ndarray, Optional[Dict]]]:
        """
        Retrieve a vector by its ID.
        
        Args:
            id: Vector identifier
            
        Returns:
            Tuple of (vector, metadata) or None if not found
        """
        pass
    
    @abstractmethod
    def delete_vector(self, id: str) -> bool:
        """
        Delete a vector by its ID.
        
        Args:
            id: Vector identifier
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def update_vector(self, id: str, vector: np.ndarray, metadata: Optional[Dict] = None) -> bool:
        """
        Update a vector by its ID.
        
        Args:
            id: Vector identifier
            vector: New vector data
            metadata: Optional new metadata
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def list_vectors(self, limit: Optional[int] = None) -> List[str]:
        """
        List all vector IDs.
        
        Args:
            limit: Optional limit on number of IDs to return
            
        Returns:
            List of vector IDs
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the storage backend and clean up resources."""
        pass
