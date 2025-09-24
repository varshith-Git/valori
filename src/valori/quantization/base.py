"""
Base quantization interface for the Vectara vector database.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import numpy as np


class Quantizer(ABC):
    """
    Abstract base class for vector quantizers.
    
    Quantizers compress vectors to reduce memory usage and improve
    search performance while maintaining reasonable accuracy.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the quantizer with configuration."""
        self.config = config
        self._initialized = False
        self._trained = False
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the quantizer."""
        pass
    
    @abstractmethod
    def train(self, vectors: np.ndarray) -> None:
        """
        Train the quantizer on a set of vectors.
        
        Args:
            vectors: Training vectors
        """
        pass
    
    @abstractmethod
    def quantize(self, vectors: np.ndarray) -> np.ndarray:
        """
        Quantize vectors.
        
        Args:
            vectors: Vectors to quantize
            
        Returns:
            Quantized vectors
        """
        pass
    
    @abstractmethod
    def dequantize(self, quantized_vectors: np.ndarray) -> np.ndarray:
        """
        Dequantize vectors.
        
        Args:
            quantized_vectors: Quantized vectors to dequantize
            
        Returns:
            Dequantized vectors
        """
        pass
    
    @abstractmethod
    def compute_distance(self, query: np.ndarray, quantized_vector: np.ndarray) -> float:
        """
        Compute distance between query and quantized vector efficiently.
        
        Args:
            query: Query vector
            quantized_vector: Quantized vector
            
        Returns:
            Distance value
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get quantizer statistics."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the quantizer and clean up resources."""
        pass
