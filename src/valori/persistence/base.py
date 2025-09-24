"""
Base persistence interface for the Vectara vector database.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class PersistenceManager(ABC):
    """
    Abstract base class for persistence managers.
    
    Persistence managers handle saving and loading of vector database
    state, including indices, quantizers, and configuration data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the persistence manager with configuration."""
        self.config = config
        self._initialized = False
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the persistence manager."""
        pass
    
    @abstractmethod
    def save_state(self, state: Dict[str, Any], path: str) -> bool:
        """
        Save database state to a file.
        
        Args:
            state: Database state dictionary
            path: File path to save to
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def load_state(self, path: str) -> Optional[Dict[str, Any]]:
        """
        Load database state from a file.
        
        Args:
            path: File path to load from
            
        Returns:
            Database state dictionary or None if failed
        """
        pass
    
    @abstractmethod
    def save_vectors(self, vectors: np.ndarray, ids: list, path: str) -> bool:
        """
        Save vectors to a file.
        
        Args:
            vectors: Array of vectors
            ids: List of vector IDs
            path: File path to save to
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def load_vectors(self, path: str) -> Optional[tuple[np.ndarray, list]]:
        """
        Load vectors from a file.
        
        Args:
            path: File path to load from
            
        Returns:
            Tuple of (vectors, ids) or None if failed
        """
        pass
    
    @abstractmethod
    def save_index(self, index_data: Dict[str, Any], path: str) -> bool:
        """
        Save index data to a file.
        
        Args:
            index_data: Index data dictionary
            path: File path to save to
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def load_index(self, path: str) -> Optional[Dict[str, Any]]:
        """
        Load index data from a file.
        
        Args:
            path: File path to load from
            
        Returns:
            Index data dictionary or None if failed
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get persistence manager statistics."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the persistence manager and clean up resources."""
        pass
