"""
Base classes and interfaces for the Valori vector database.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np


class VectorDB(ABC):
    """
    Abstract base class for vector database implementations.
    
    This class defines the core interface that all vector database
    implementations must follow.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the vector database with configuration."""
        self.config = config
        self._initialized = False
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the database and any required resources."""
        pass
    
    @abstractmethod
    def insert(self, vectors: np.ndarray, metadata: Optional[List[Dict]] = None) -> List[str]:
        """
        Insert vectors into the database.
        
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
        """Get database statistics."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the database and clean up resources."""
        pass
