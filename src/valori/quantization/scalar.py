"""
Scalar quantization implementation for the Vectara vector database.
"""

from typing import Any, Dict
import numpy as np

from .base import Quantizer
from ..exceptions import QuantizationError


class ScalarQuantizer(Quantizer):
    """
    Scalar quantization implementation.
    
    Scalar quantization quantizes each dimension of a vector independently
    to a fixed number of bits. It's simple and fast but may not be as
    efficient as other methods for high-dimensional vectors.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize scalar quantizer."""
        super().__init__(config)
        self.bits = config.get("bits", 8)  # Number of bits per dimension
        self.min_values: Optional[np.ndarray] = None
        self.max_values: Optional[np.ndarray] = None
        self.scale: Optional[np.ndarray] = None
        self.zero_point: Optional[np.ndarray] = None
        self.max_val = (2 ** self.bits) - 1
    
    def initialize(self) -> None:
        """Initialize the scalar quantizer."""
        self.min_values = None
        self.max_values = None
        self.scale = None
        self.zero_point = None
        self._initialized = True
    
    def train(self, vectors: np.ndarray) -> None:
        """Train the scalar quantizer on vectors."""
        if not self._initialized:
            raise QuantizationError("Quantizer not initialized")
        
        if vectors.ndim != 2:
            raise QuantizationError("Vectors must be 2D array")
        
        try:
            # Compute min and max values for each dimension
            self.min_values = np.min(vectors, axis=0)
            self.max_values = np.max(vectors, axis=0)
            
            # Compute scale and zero point for quantization
            self.scale = (self.max_values - self.min_values) / self.max_val
            self.zero_point = np.round(-self.min_values / self.scale).astype(np.int32)
            
            # Clamp zero point to valid range
            self.zero_point = np.clip(self.zero_point, 0, self.max_val)
            
            self._trained = True
            
        except Exception as e:
            raise QuantizationError(f"Failed to train scalar quantizer: {str(e)}")
    
    def quantize(self, vectors: np.ndarray) -> np.ndarray:
        """Quantize vectors using scalar quantization."""
        if not self._initialized:
            raise QuantizationError("Quantizer not initialized")
        
        if not self._trained:
            raise QuantizationError("Quantizer not trained")
        
        try:
            # Apply quantization formula: q = round((x - min) / scale + zero_point)
            quantized = np.round(
                (vectors - self.min_values) / self.scale + self.zero_point
            ).astype(np.int32)
            
            # Clamp to valid range
            quantized = np.clip(quantized, 0, self.max_val)
            
            return quantized
            
        except Exception as e:
            raise QuantizationError(f"Failed to quantize vectors: {str(e)}")
    
    def dequantize(self, quantized_vectors: np.ndarray) -> np.ndarray:
        """Dequantize vectors using scalar quantization."""
        if not self._initialized:
            raise QuantizationError("Quantizer not initialized")
        
        if not self._trained:
            raise QuantizationError("Quantizer not trained")
        
        try:
            # Apply dequantization formula: x = (q - zero_point) * scale + min
            dequantized = (quantized_vectors - self.zero_point) * self.scale + self.min_values
            
            return dequantized
            
        except Exception as e:
            raise QuantizationError(f"Failed to dequantize vectors: {str(e)}")
    
    def compute_distance(self, query: np.ndarray, quantized_vector: np.ndarray) -> float:
        """Compute distance between query and quantized vector efficiently."""
        if not self._initialized:
            raise QuantizationError("Quantizer not initialized")
        
        if not self._trained:
            raise QuantizationError("Quantizer not trained")
        
        try:
            # Dequantize the vector
            dequantized = self.dequantize(quantized_vector.reshape(1, -1))[0]
            
            # Compute cosine distance (assuming cosine similarity)
            dot_product = np.dot(query, dequantized)
            norm_query = np.linalg.norm(query)
            norm_dequantized = np.linalg.norm(dequantized)
            
            if norm_query == 0 or norm_dequantized == 0:
                return 1.0
            
            similarity = dot_product / (norm_query * norm_dequantized)
            return 1 - similarity
            
        except Exception as e:
            raise QuantizationError(f"Failed to compute distance: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scalar quantizer statistics."""
        if not self._initialized:
            return {"status": "not_initialized"}
        
        stats = {
            "quantizer_type": "scalar",
            "bits": self.bits,
            "max_val": self.max_val,
            "trained": self._trained,
            "initialized": self._initialized,
        }
        
        if self._trained:
            stats.update({
                "dimensions": len(self.min_values) if self.min_values is not None else 0,
                "min_values_range": (self.min_values.min(), self.min_values.max()) if self.min_values is not None else None,
                "max_values_range": (self.max_values.min(), self.max_values.max()) if self.max_values is not None else None,
                "compression_ratio": self.bits / 32.0,  # Assuming original vectors are 32-bit floats
            })
        
        return stats
    
    def close(self) -> None:
        """Close the scalar quantizer."""
        self.min_values = None
        self.max_values = None
        self.scale = None
        self.zero_point = None
        self._trained = False
        self._initialized = False
