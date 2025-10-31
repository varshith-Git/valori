"""
Product quantization implementation for the valori vector database.
"""

from typing import Any, Dict, Optional
import numpy as np
from sklearn.cluster import KMeans

from .base import Quantizer
from ..exceptions import QuantizationError


class ProductQuantizer(Quantizer):
    """
    Product quantization implementation.

    Product quantization divides vectors into subvectors and quantizes
    each subvector independently. It's more efficient than scalar quantization
    for high-dimensional vectors and provides better compression ratios.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize product quantizer."""
        super().__init__(config)
        self.m = config.get("m", 8)  # Number of subvectors
        self.k = config.get("k", 256)  # Number of centroids per subvector
        self.dim: Optional[int] = None
        self.subvector_dim: Optional[int] = None
        self.centroids: Optional[np.ndarray] = None  # Shape: (m, k, subvector_dim)
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the product quantizer."""
        self.centroids = None
        self.dim = None
        self.subvector_dim = None
        self._initialized = True

    def train(self, vectors: np.ndarray) -> None:
        """Train the product quantizer on vectors."""
        if not self._initialized:
            raise QuantizationError("Quantizer not initialized")

        if vectors.ndim != 2:
            raise QuantizationError("Vectors must be 2D array")

        try:
            n_vectors, dim = vectors.shape
            self.dim = int(dim)

            # Check if dimension is divisible by m
            if self.dim % self.m != 0:
                raise QuantizationError(
                    f"Vector dimension {self.dim} must be divisible by m={self.m}"
                )

            self.subvector_dim = int(self.dim // self.m)

            # Initialize centroids array
            self.centroids = np.zeros((self.m, self.k, self.subvector_dim), dtype=float)

            # Train each subvector separately
            for i in range(self.m):
                assert self.subvector_dim is not None
                start_idx = i * self.subvector_dim
                end_idx = (i + 1) * self.subvector_dim
                subvectors = vectors[:, start_idx:end_idx]

                # Perform K-means clustering on subvectors
                kmeans = KMeans(n_clusters=self.k, random_state=42, n_init=10)
                kmeans.fit(subvectors)

                # Store centroids
                self.centroids[i] = kmeans.cluster_centers_

            self._trained = True

        except Exception as e:
            raise QuantizationError(f"Failed to train product quantizer: {str(e)}")

    def quantize(self, vectors: np.ndarray) -> np.ndarray:
        """Quantize vectors using product quantization."""
        if not self._initialized:
            raise QuantizationError("Quantizer not initialized")

        if not self._trained:
            raise QuantizationError("Quantizer not trained")

        try:
            n_vectors = int(vectors.shape[0])
            quantized = np.zeros((n_vectors, int(self.m)), dtype=np.uint8)

            # Quantize each subvector
            assert self.subvector_dim is not None and self.centroids is not None
            for i in range(int(self.m)):
                start_idx = i * self.subvector_dim
                end_idx = (i + 1) * self.subvector_dim
                subvectors = vectors[:, start_idx:end_idx]

                # Find closest centroids
                centroids_i = self.centroids[i]
                distances = np.sum((subvectors[:, np.newaxis, :] - centroids_i[np.newaxis, :, :]) ** 2, axis=2)
                closest_centroids = np.argmin(distances, axis=1)

                quantized[:, i] = closest_centroids

            return quantized

        except Exception as e:
            raise QuantizationError(f"Failed to quantize vectors: {str(e)}")

    def dequantize(self, quantized_vectors: np.ndarray) -> np.ndarray:
        """Dequantize vectors using product quantization."""
        if not self._initialized:
            raise QuantizationError("Quantizer not initialized")

        if not self._trained:
            raise QuantizationError("Quantizer not trained")

        try:
            n_vectors = int(quantized_vectors.shape[0])
            assert self.dim is not None
            dequantized = np.zeros((n_vectors, int(self.dim)), dtype=float)

            # Dequantize each subvector
            assert self.subvector_dim is not None and self.centroids is not None
            for i in range(int(self.m)):
                start_idx = i * self.subvector_dim
                end_idx = (i + 1) * self.subvector_dim

                # Get centroids for this subvector
                centroid_indices = quantized_vectors[:, i].astype(int)
                centroids_i = self.centroids[i]
                dequantized[:, start_idx:end_idx] = centroids_i[centroid_indices]

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
            # Compute distance using lookup table for efficiency
            distance = 0.0

            assert self.subvector_dim is not None and self.centroids is not None and self.dim is not None
            for i in range(int(self.m)):
                start_idx = i * self.subvector_dim
                end_idx = (i + 1) * self.subvector_dim
                query_subvector = query[start_idx:end_idx]

                centroid_idx = int(quantized_vector[i])
                centroid = self.centroids[i, centroid_idx]

                # Compute squared Euclidean distance for this subvector
                subvector_distance: float = float(np.sum((query_subvector - centroid) ** 2))
                distance += subvector_distance

            # Return Euclidean distance
            return np.sqrt(distance)

        except Exception as e:
            raise QuantizationError(f"Failed to compute distance: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """Get product quantizer statistics."""
        if not self._initialized:
            return {"status": "not_initialized"}

        stats = {
            "quantizer_type": "product",
            "m": self.m,
            "k": self.k,
            "trained": self._trained,
            "initialized": self._initialized,
        }

        if self._trained:
            bits_per_subvector = float(np.log2(self.k))
            total_bits = float(self.m) * bits_per_subvector

            stats.update(
                {
                    "dimensions": int(self.dim) if self.dim is not None else None,
                    "subvector_dimensions": int(self.subvector_dim) if self.subvector_dim is not None else None,
                    "bits_per_subvector": bits_per_subvector,
                    "total_bits": total_bits,
                    "compression_ratio": (total_bits / (32 * float(self.dim))) if self.dim is not None else None,
                    "centroids_shape": (self.centroids.shape if self.centroids is not None else None),
                }
            )

        return stats

    def close(self) -> None:
        """Close the product quantizer."""
        self.centroids = None
        self.dim = None
        self.subvector_dim = None
        self._trained = False
        self._initialized = False
