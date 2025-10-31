"""
Scalar quantization implementation for the valori vector database.
"""

from typing import Any, Dict, Optional
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
        self.dim: Optional[int] = None
        self.max_val = (2**self.bits) - 1

    def initialize(self) -> None:
        """Initialize the scalar quantizer."""
        self.min_values = None
        self.max_values = None
        self.scale = None
        self.zero_point = None
        self.dim = None
        self._initialized = True

    def train(self, vectors: np.ndarray) -> None:
        """Train the scalar quantizer on vectors."""
        if not self._initialized:
            raise QuantizationError("Quantizer not initialized")

        if vectors.ndim != 2:
            raise QuantizationError("Vectors must be 2D array")

        try:
            # Set dimension and compute min and max values for each dimension
            n_vectors, dim = vectors.shape
            self.dim = int(dim)
            min_vals = np.min(vectors, axis=0)
            max_vals = np.max(vectors, axis=0)

            # Compute scale and zero point for quantization using local vars
            denom = float((1 << int(self.bits)) - 1)
            scales = (max_vals - min_vals) / denom
            # Avoid zero scale values per-dimension
            scales = np.where(scales == 0, 1.0, scales)
            zero_pt = np.round(-min_vals / scales).astype(np.int32)

            # Clamp zero point to valid range
            zero_pt = np.clip(zero_pt, 0, int(self.max_val))

            # Assign back to attributes (now narrowed)
            self.min_values = min_vals
            self.max_values = max_vals
            self.scale = scales
            self.zero_point = zero_pt

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
            assert self.dim is not None and self.min_values is not None and self.scale is not None and self.zero_point is not None
            n_vectors = int(vectors.shape[0])
            quantized = np.zeros((n_vectors, int(self.dim)), dtype=np.int32)

            for i in range(int(self.dim)):
                val = np.round((vectors[:, i] - self.min_values[i]) / self.scale[i] + self.zero_point[i])
                clipped = np.clip(val, 0, int(self.max_val))
                quantized[:, i] = clipped.astype(np.int32)

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
            assert self.dim is not None and self.scale is not None and self.min_values is not None and self.zero_point is not None
            n_vectors = int(quantized_vectors.shape[0])
            dequantized = np.zeros((n_vectors, int(self.dim)), dtype=float)

            for i in range(int(self.dim)):
                dequantized[:, i] = (quantized_vectors[:, i].astype(np.float32) - self.zero_point[i]) * self.scale[i] + self.min_values[i]

            return dequantized

        except Exception as e:
            raise QuantizationError(f"Failed to dequantize vectors: {str(e)}")

    def compute_distance(
        self, query: np.ndarray, quantized_vector: np.ndarray
    ) -> float:
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
            stats.update(
                {
                    "dimensions": int(self.dim) if self.dim is not None else 0,
                    "min_values_range": (
                        (self.min_values.min(), self.min_values.max()) if self.min_values is not None else None
                    ),
                    "max_values_range": (
                        (self.max_values.min(), self.max_values.max()) if self.max_values is not None else None
                    ),
                    "compression_ratio": float(self.bits) / 32.0,  # Assuming original vectors are 32-bit floats
                }
            )

        return stats

    def close(self) -> None:
        """Close the scalar quantizer."""
        self.min_values = None
        self.max_values = None
        self.scale = None
        self.zero_point = None
        self._trained = False
        self._initialized = False
        self.dim = None
