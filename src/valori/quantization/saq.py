"""
Segmented and Adjusted Quantization (SAQ) implementation.

Based on the SAQ method from recent research, this quantizer provides
production-ready quantization for large-scale vector databases.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .base import Quantizer
from ..exceptions import QuantizationError


class SAQQuantizer(Quantizer):
    """
    Segmented and Adjusted Quantization (SAQ) implementation.

    SAQ provides high-accuracy quantization by:
    1. Performing PCA projection and segmenting dimensions
    2. Dynamically allocating bits to segments based on importance
    3. Applying code adjustment via coordinate descent
    4. Supporting rescoring with full-precision vectors

    This implementation is optimized for large-scale production use.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize SAQ quantizer."""
        super().__init__(config)
        self.total_bits = config.get("total_bits", 128)
        self.n_segments = config.get("n_segments", 8)
        self.adjustment_iters = config.get("adjustment_iters", 3)
        self.rescore_top_k = config.get("rescore_top_k", 50)

        # PCA and segmentation attributes
        self.pca = None
        self.scaler = None
        self.segment_boundaries = None
        self.segment_bits = None
        self.segment_quantizers = None

        # Vector storage for rescoring
        self.full_precision_vectors = None
        self.dim = None

        # Performance monitoring
        self.stats = {
            "compression_ratio": 0.0,
            "segmentation_time": 0.0,
            "adjustment_improvements": [],
        }

    def initialize(self) -> None:
        """Initialize the SAQ quantizer."""
        self.pca = PCA()
        self.scaler = StandardScaler()
        self.segment_boundaries = None
        self.segment_bits = None
        self.segment_quantizers = []
        self.full_precision_vectors = None
        self.dim = None
        self._initialized = True

    def train(self, vectors: np.ndarray) -> None:
        """
        Train SAQ quantizer on vectors with PCA and dynamic bit allocation.

        Args:
            vectors: Training vectors of shape (n_vectors, dim)

        Raises:
            QuantizationError: If training fails or vectors are invalid
        """
        if not self._initialized:
            raise QuantizationError("Quantizer not initialized")

        if vectors.ndim != 2:
            raise QuantizationError("Vectors must be 2D array")

        try:
            import time

            start_time = time.time()

            n_vectors, dim = vectors.shape
            self.dim = int(dim)

            # Store full-precision vectors for rescoring
            self.full_precision_vectors = vectors.astype(np.float32)

            # Standardize data before PCA
            vectors_normalized = self.scaler.fit_transform(vectors)

            # Apply PCA for dimensionality segmentation
            vectors_pca = self.pca.fit_transform(vectors_normalized)

            # Perform optimal segmentation and bit allocation
            self._segment_and_allocate_bits(vectors_pca)

            # Train individual segment quantizers
            self._train_segment_quantizers(vectors_pca)

            # Calculate compression ratio
            original_size = n_vectors * dim * 32  # 32 bits per float
            quantized_size = n_vectors * self.total_bits
            self.stats["compression_ratio"] = original_size / quantized_size
            self.stats["segmentation_time"] = time.time() - start_time

            self._trained = True

        except Exception as e:
            raise QuantizationError(f"SAQ training failed: {str(e)}")

    def _segment_and_allocate_bits(self, vectors_pca: np.ndarray) -> None:
        """
        Perform optimal dimension segmentation and bit allocation.

        Uses dynamic programming to find optimal segmentation based on
        explained variance ratio from PCA.
        """
        n_vectors, dim = vectors_pca.shape

        # Use explained variance to determine segment importance
        explained_variance = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)

        # Dynamic programming for optimal segmentation
        # This minimizes quantization error given bit budget
        dp = np.full((self.n_segments + 1, dim + 1), np.inf)
        dp[0, 0] = 0

        # Backtracking array
        backtrack = np.zeros((self.n_segments + 1, dim + 1), dtype=int)

        for seg in range(1, self.n_segments + 1):
            for end in range(seg, dim + 1):
                for start in range(seg - 1, end):
                    if start == 0:
                        seg_variance = cumulative_variance[end - 1]
                    else:
                        seg_variance = (
                            cumulative_variance[end - 1]
                            - cumulative_variance[start - 1]
                        )

                    # Cost is inverse of variance (we want high-variance segments to have more bits)
                    cost = dp[seg - 1, start] + (1.0 - seg_variance)

                    if cost < dp[seg, end]:
                        dp[seg, end] = cost
                        backtrack[seg, end] = start

        # Reconstruct segmentation
        self.segment_boundaries = []
        current_pos = dim
        for seg in range(self.n_segments, 0, -1):
            start = backtrack[seg, current_pos]
            self.segment_boundaries.insert(0, (start, current_pos))
            current_pos = start

        # Allocate bits based on segment variance
        segment_variances = []
        for start, end in self.segment_boundaries:
            if start == 0:
                seg_variance = cumulative_variance[end - 1]
            else:
                seg_variance = (
                    cumulative_variance[end - 1] - cumulative_variance[start - 1]
                )
            segment_variances.append(seg_variance)

        total_variance = sum(segment_variances)
        self.segment_bits = []

        for variance in segment_variances:
            bits = max(1, int(self.total_bits * (variance / total_variance)))
            self.segment_bits.append(bits)

    def _train_segment_quantizers(self, vectors_pca: np.ndarray) -> None:
        """Train individual quantizers for each segment."""
        self.segment_quantizers = []

        for (start, end), bits in zip(self.segment_boundaries, self.segment_bits):
            segment_vectors = vectors_pca[:, start:end]

            # Create scalar quantizer for this segment
            from .scalar import ScalarQuantizer

            quantizer = ScalarQuantizer({"bits": bits})
            quantizer.initialize()
            quantizer.train(segment_vectors)

            self.segment_quantizers.append(quantizer)

    def quantize(self, vectors: np.ndarray) -> np.ndarray:
        """
        Quantize vectors using SAQ with code adjustment.

        Args:
            vectors: Vectors to quantize

        Returns:
            Quantized vectors as integer codes
        """
        if not self._initialized or not self._trained:
            raise QuantizationError("Quantizer not ready")

        try:
            # Preprocess vectors
            vectors_normalized = self.scaler.transform(vectors)
            vectors_pca = self.pca.transform(vectors_normalized)

            n_vectors = vectors.shape[0]

            # Initial quantization
            quantized_segments = []
            for (start, end), quantizer in zip(
                self.segment_boundaries, self.segment_quantizers
            ):
                segment_vectors = vectors_pca[:, start:end]
                quantized_segment = quantizer.quantize(segment_vectors)
                quantized_segments.append(quantized_segment)

            # Apply code adjustment via coordinate descent
            adjusted_codes = self._apply_code_adjustment(
                vectors_pca, quantized_segments, vectors
            )

            return adjusted_codes

        except Exception as e:
            raise QuantizationError(f"SAQ quantization failed: {str(e)}")

    def _apply_code_adjustment(
        self,
        vectors_pca: np.ndarray,
        quantized_segments: List[np.ndarray],
        original_vectors: np.ndarray,
    ) -> np.ndarray:
        """
        Apply code adjustment via coordinate descent to refine quantization.

        This significantly reduces quantization error by iteratively
        adjusting codes to better approximate original vectors.
        """
        n_vectors = vectors_pca.shape[0]
        best_codes = np.hstack(quantized_segments)

        for iteration in range(self.adjustment_iters):
            improvement = 0.0

            for seg_idx in range(len(self.segment_quantizers)):
                # Temporarily dequantize all but current segment
                reconstructed = self._dequantize_segments(
                    best_codes, exclude_segment=seg_idx
                )

                # Calculate residual for current segment
                residual = vectors_pca - reconstructed

                # Find better codes for current segment
                start, end = self.segment_boundaries[seg_idx]
                segment_residual = residual[:, start:end]

                quantizer = self.segment_quantizers[seg_idx]
                new_codes = quantizer.quantize(
                    segment_residual + reconstructed[:, start:end]
                )

                # Update if improvement
                new_reconstructed = self._dequantize_segments(
                    best_codes, updated_segment=(seg_idx, new_codes)
                )

                old_error = np.mean((vectors_pca - reconstructed) ** 2)
                new_error = np.mean((vectors_pca - new_reconstructed) ** 2)

                if new_error < old_error:
                    best_codes[:, seg_idx] = new_codes
                    improvement += old_error - new_error

            self.stats["adjustment_improvements"].append(improvement)

            if improvement < 1e-6:  # Convergence check
                break

        return best_codes

    def _dequantize_segments(
        self,
        codes: np.ndarray,
        exclude_segment: Optional[int] = None,
        updated_segment: Optional[Tuple[int, np.ndarray]] = None,
    ) -> np.ndarray:
        """Dequantize segments with optional exclusions or updates."""
        n_vectors = codes.shape[0]
        reconstructed = np.zeros((n_vectors, self.dim))

        for seg_idx in range(len(self.segment_quantizers)):
            start, end = self.segment_boundaries[seg_idx]
            quantizer = self.segment_quantizers[seg_idx]

            if exclude_segment == seg_idx:
                continue

            if updated_segment and updated_segment[0] == seg_idx:
                segment_codes = updated_segment[1]
            else:
                segment_codes = codes[:, seg_idx]

            segment_dequantized = quantizer.dequantize(segment_codes)
            reconstructed[:, start:end] = segment_dequantized

        return reconstructed

    def dequantize(self, quantized_vectors: np.ndarray) -> np.ndarray:
        """Dequantize vectors using SAQ."""
        if not self._initialized or not self._trained:
            raise QuantizationError("Quantizer not ready")

        try:
            # Dequantize segments
            reconstructed_pca = self._dequantize_segments(quantized_vectors)

            # Reverse PCA transformation
            reconstructed_normalized = self.pca.inverse_transform(reconstructed_pca)

            # Reverse standardization
            reconstructed = self.scaler.inverse_transform(reconstructed_normalized)

            return reconstructed

        except Exception as e:
            raise QuantizationError(f"SAQ dequantization failed: {str(e)}")

    def compute_distance(
        self, query: np.ndarray, quantized_vector: np.ndarray
    ) -> float:
        """
        Compute distance between query and quantized vector with rescoring.

        Uses efficient distance computation with optional rescoring
        using full-precision vectors for final ranking.
        """
        # Dequantize the vector
        dequantized = self.dequantize(quantized_vector.reshape(1, -1))[0]

        # Compute cosine distance
        dot_product = np.dot(query, dequantized)
        norm_query = np.linalg.norm(query)
        norm_dequantized = np.linalg.norm(dequantized)

        if norm_query == 0 or norm_dequantized == 0:
            return 1.0

        similarity = dot_product / (norm_query * norm_dequantized)
        return 1.0 - similarity

    def search_with_rescoring(
        self, query: np.ndarray, quantized_vectors: np.ndarray, top_k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform search with rescoring using full-precision vectors.

        This provides the best accuracy by using quantized vectors for
        initial candidate selection and full-precision vectors for final ranking.

        Args:
            query: Query vector
            quantized_vectors: All quantized vectors in the database
            top_k: Number of results to return

        Returns:
            Tuple of (indices, distances) for top_k results
        """
        if self.full_precision_vectors is None:
            raise QuantizationError(
                "Full precision vectors not available for rescoring"
            )

        # Initial search using quantized vectors (fast)
        initial_candidates = self._initial_quantized_search(query, quantized_vectors)

        # Rescore top candidates using full-precision vectors (accurate)
        final_results = self._rescoring_search(query, initial_candidates, top_k)

        return final_results

    def _initial_quantized_search(
        self, query: np.ndarray, quantized_vectors: np.ndarray
    ) -> np.ndarray:
        """Perform initial fast search using quantized vectors."""
        n_vectors = quantized_vectors.shape[0]
        batch_size = 1000
        distances = []

        for i in range(0, n_vectors, batch_size):
            batch_end = min(i + batch_size, n_vectors)
            batch_vectors = quantized_vectors[i:batch_end]

            batch_distances = np.array(
                [self.compute_distance(query, qv) for qv in batch_vectors]
            )
            distances.extend(batch_distances)

        # Return indices of top candidates for rescoring
        distances = np.array(distances)
        return np.argsort(distances)[: self.rescore_top_k]

    def _rescoring_search(
        self, query: np.ndarray, candidate_indices: np.ndarray, top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Rescore candidates using full-precision vectors."""
        candidate_vectors = self.full_precision_vectors[candidate_indices]

        # Compute exact distances with full-precision vectors
        distances = []
        for vec in candidate_vectors:
            dot_product = np.dot(query, vec)
            norm_query = np.linalg.norm(query)
            norm_vec = np.linalg.norm(vec)

            if norm_query == 0 or norm_vec == 0:
                similarity = 0
            else:
                similarity = dot_product / (norm_query * norm_vec)

            distances.append(1.0 - similarity)

        distances = np.array(distances)

        # Get top_k results from candidates
        top_indices = np.argsort(distances)[:top_k]
        final_indices = candidate_indices[top_indices]
        final_distances = distances[top_indices]

        return final_indices, final_distances

    def get_stats(self) -> Dict[str, Any]:
        """Get detailed SAQ quantizer statistics."""
        if not self._initialized:
            return {"status": "not_initialized"}

        stats = {
            "quantizer_type": "saq",
            "total_bits": self.total_bits,
            "n_segments": self.n_segments,
            "dimensions": self.dim,
            "trained": self._trained,
            "initialized": self._initialized,
            **self.stats,
        }

        if self._trained:
            stats.update(
                {
                    "segment_boundaries": self.segment_boundaries,
                    "segment_bits": self.segment_bits,
                    "compression_ratio": float(self.stats["compression_ratio"]),
                    "explained_variance_ratio": float(
                        np.sum(self.pca.explained_variance_ratio_)
                    ),
                }
            )

        return stats

    def close(self) -> None:
        """Close the SAQ quantizer and clean up resources."""
        self.pca = None
        self.scaler = None
        self.segment_boundaries = None
        self.segment_bits = None
        self.segment_quantizers = []
        self.full_precision_vectors = None
        self._trained = False
        self._initialized = False
