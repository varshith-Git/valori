"""
Scalar quantization implementation for the valori vector database.
"""

from typing import Any, Dict, Optional

import numpy as np

from ..exceptions import QuantizationError
from .base import Quantizer


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
            # Use a capped denom to avoid floating-point overflow for very large bit counts
            raw_denom = (1 << int(self.bits)) - 1
            denom_cap = float(min(raw_denom, np.iinfo(np.int64).max))
            scales = (max_vals - min_vals) / denom_cap
            # Avoid zero scale values per-dimension
            scales = np.where(scales == 0, 1.0, scales)
            # For range-based uniform quantization we use zero_point = 0
            zero_pt = np.zeros_like(min_vals, dtype=np.int32)

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
            # Expect 2D input (n_vectors, dim)
            if vectors.ndim != 2:
                raise QuantizationError("Vectors must be 2D array")
            # Apply quantization formula: q = round((x - min) / scale + zero_point)
            assert (
                self.dim is not None
                and self.min_values is not None
                and self.scale is not None
                and self.zero_point is not None
            )
            n_vectors = int(vectors.shape[0])
            # Choose integer dtype based on required range to avoid overflow
            if self.max_val <= np.iinfo(np.int32).max:
                itype = np.int32
            else:
                itype = np.int64

            quantized = np.zeros((n_vectors, int(self.dim)), dtype=itype)

            # Compute per-dimension quantized values in a vectorized manner
            # normalized = (vectors - min) / scale
            with np.errstate(invalid="ignore", divide="ignore"):
                normalized_all = (vectors - self.min_values) / self.scale

            # Round and clip to valid range
            rounded = np.rint(normalized_all)
            # Clip to the maximum representable for the chosen integer type
            max_for_type = np.iinfo(itype).max
            upper_bound = int(
                self.max_val if self.max_val <= max_for_type else max_for_type
            )
            clipped = np.clip(rounded, 0, upper_bound)
            clipped = np.nan_to_num(clipped, nan=0.0, posinf=upper_bound, neginf=0.0)

            # Safe cast to integer type
            quantized[:, :] = clipped.astype(itype)

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
            assert (
                self.dim is not None
                and self.scale is not None
                and self.min_values is not None
                and self.zero_point is not None
            )
            # Accept 1D or 2D quantized input; convert to 2D for processing
            if quantized_vectors.ndim == 1:
                quantized_vectors = quantized_vectors.reshape(1, -1)

            if quantized_vectors.ndim != 2:
                raise QuantizationError("Quantized vectors must be 2D array")

            n_vectors = int(quantized_vectors.shape[0])
            dequantized = np.zeros((n_vectors, int(self.dim)), dtype=np.float32)

            # Vectorized dequantization
            q_float = quantized_vectors.astype(np.float32)
            dequantized = q_float * self.scale + self.min_values

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
            # Align query and codes dimensions if needed
            assert self.dim is not None
            q = query.astype(np.float32)
            if q.ndim != 1:
                q = q.reshape(-1)
            # Truncate or pad query to match trained dimension
            if q.shape[0] > int(self.dim):
                q = q[: int(self.dim)]
            elif q.shape[0] < int(self.dim):
                pad_len = int(self.dim) - q.shape[0]
                q = np.pad(q, (0, pad_len), mode="constant")

            # Dequantize the vector
            dequantized = self.dequantize(quantized_vector.reshape(1, -1))[0]

            # Compute cosine distance (assuming cosine similarity)
            dot_product = np.dot(q, dequantized)
            norm_query = np.linalg.norm(q)
            norm_dequantized = np.linalg.norm(dequantized)

            if norm_query == 0 or norm_dequantized == 0:
                return 1.0

            similarity = dot_product / (norm_query * norm_dequantized)
            return float(1.0 - similarity)

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
                        (self.min_values.min(), self.min_values.max())
                        if self.min_values is not None
                        else None
                    ),
                    "max_values_range": (
                        (self.max_values.min(), self.max_values.max())
                        if self.max_values is not None
                        else None
                    ),
                    # Compression ratio = original bits per value / quantized bits
                    "compression_ratio": 32.0 / float(self.bits),
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
