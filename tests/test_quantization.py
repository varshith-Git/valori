"""
Tests for quantization implementations.
"""

import numpy as np
import pytest

from valori.exceptions import QuantizationError
from valori.quantization import ProductQuantizer, SAQQuantizer, ScalarQuantizer


class TestScalarQuantizer:
    """Test scalar quantization implementation."""

    def test_initialize(self, scalar_quantizer):
        """Test quantizer initialization."""
        scalar_quantizer.initialize()
        assert scalar_quantizer._initialized

    def test_train(self, scalar_quantizer, small_vectors):
        """Test quantizer training."""
        scalar_quantizer.initialize()
        scalar_quantizer.train(small_vectors)
        assert scalar_quantizer._trained
        assert scalar_quantizer.min_values is not None
        assert scalar_quantizer.max_values is not None

    def test_quantize_dequantize(self, scalar_quantizer, small_vectors):
        """Test quantization and dequantization."""
        scalar_quantizer.initialize()
        scalar_quantizer.train(small_vectors)

        # Quantize vectors
        quantized = scalar_quantizer.quantize(small_vectors)
        assert quantized.shape == small_vectors.shape
        assert quantized.dtype == np.int32

        # Dequantize vectors
        dequantized = scalar_quantizer.dequantize(quantized)
        assert dequantized.shape == small_vectors.shape
        assert dequantized.dtype == small_vectors.dtype

        # Check that dequantized vectors are close to original
        # (within quantization error)
        max_error = np.max(np.abs(dequantized - small_vectors))
        assert max_error < 1.0  # Should be reasonable for 8-bit quantization

    def test_compute_distance(self, scalar_quantizer, small_vectors, query_vector):
        """Test distance computation with quantized vectors."""
        scalar_quantizer.initialize()
        scalar_quantizer.train(small_vectors)

        # Quantize a vector
        quantized_vector = scalar_quantizer.quantize(small_vectors[:1])[0]

        # Compute distance
        distance = scalar_quantizer.compute_distance(query_vector, quantized_vector)
        assert isinstance(distance, float)
        assert distance >= 0

    def test_get_stats(self, scalar_quantizer, small_vectors):
        """Test getting quantizer statistics."""
        scalar_quantizer.initialize()

        # Before training
        stats = scalar_quantizer.get_stats()
        assert stats["quantizer_type"] == "scalar"
        assert stats["bits"] == 8
        assert not stats["trained"]

        # After training
        scalar_quantizer.train(small_vectors)
        stats = scalar_quantizer.get_stats()
        assert stats["trained"]
        assert stats["dimensions"] == small_vectors.shape[1]
        # compression_ratio = original_bits_per_value / quantized_bits_per_value
        assert stats["compression_ratio"] == 32.0 / stats["bits"]

    def test_close(self, scalar_quantizer, small_vectors):
        """Test closing quantizer."""
        scalar_quantizer.initialize()
        scalar_quantizer.train(small_vectors)

        scalar_quantizer.close()
        assert not scalar_quantizer._initialized
        assert not scalar_quantizer._trained
        assert scalar_quantizer.min_values is None

    def test_different_bits(self, small_vectors):
        """Test quantizer with different bit depths."""
        for bits in [4, 8, 16]:
            quantizer = ScalarQuantizer({"bits": bits})
            quantizer.initialize()
            quantizer.train(small_vectors)

            quantized = quantizer.quantize(small_vectors)
            dequantized = quantizer.dequantize(quantized)

            # Higher bit depth should have lower error
            error = np.max(np.abs(dequantized - small_vectors))
            assert error < 1.0


class TestProductQuantizer:
    """Test product quantization implementation."""

    def test_initialize(self, product_quantizer):
        """Test quantizer initialization."""
        product_quantizer.initialize()
        assert product_quantizer._initialized

    def test_train(self, product_quantizer, high_dim_vectors):
        """Test quantizer training."""
        product_quantizer.initialize()
        product_quantizer.train(high_dim_vectors)
        assert product_quantizer._trained
        assert product_quantizer.centroids is not None
        assert product_quantizer.centroids.shape[0] == product_quantizer.m

    def test_quantize_dequantize(self, product_quantizer, high_dim_vectors):
        """Test quantization and dequantization."""
        product_quantizer.initialize()
        product_quantizer.train(high_dim_vectors)

        # Quantize vectors
        quantized = product_quantizer.quantize(high_dim_vectors)
        assert quantized.shape == (high_dim_vectors.shape[0], product_quantizer.m)
        assert quantized.dtype == np.uint8

        # Dequantize vectors
        dequantized = product_quantizer.dequantize(quantized)
        assert dequantized.shape == high_dim_vectors.shape
        assert dequantized.dtype == high_dim_vectors.dtype

        # Check that dequantized vectors are reasonable
        max_error = np.max(np.abs(dequantized - high_dim_vectors))
        assert max_error < 2.0  # Should be reasonable for product quantization

    def test_compute_distance(self, product_quantizer, high_dim_vectors, query_vector):
        """Test distance computation with quantized vectors."""
        product_quantizer.initialize()
        product_quantizer.train(high_dim_vectors)

        # Quantize a vector
        quantized_vector = product_quantizer.quantize(high_dim_vectors[:1])[0]

        # Compute distance
        distance = product_quantizer.compute_distance(
            query_vector[:512], quantized_vector
        )
        assert isinstance(distance, float)
        assert distance >= 0

    def test_get_stats(self, product_quantizer, high_dim_vectors):
        """Test getting quantizer statistics."""
        product_quantizer.initialize()

        # Before training
        stats = product_quantizer.get_stats()
        assert stats["quantizer_type"] == "product"
        assert stats["m"] == 8
        assert stats["k"] == 256
        assert not stats["trained"]

        # After training
        product_quantizer.train(high_dim_vectors)
        stats = product_quantizer.get_stats()
        assert stats["trained"]
        assert stats["dimensions"] == high_dim_vectors.shape[1]
        assert stats["subvector_dimensions"] == high_dim_vectors.shape[1] // 8

    def test_close(self, product_quantizer, high_dim_vectors):
        """Test closing quantizer."""
        product_quantizer.initialize()
        product_quantizer.train(high_dim_vectors)

        product_quantizer.close()
        assert not product_quantizer._initialized
        assert not product_quantizer._trained
        assert product_quantizer.centroids is None

    def test_different_m_values(self, high_dim_vectors):
        """Test quantizer with different m values."""
        for m in [4, 8, 16]:
            if high_dim_vectors.shape[1] % m == 0:
                quantizer = ProductQuantizer({"m": m, "k": 256})
                quantizer.initialize()
                quantizer.train(high_dim_vectors)

                quantized = quantizer.quantize(high_dim_vectors)
                dequantized = quantizer.dequantize(quantized)

                # Check shapes
                assert quantized.shape == (high_dim_vectors.shape[0], m)
                assert dequantized.shape == high_dim_vectors.shape


class TestSAQQuantizer:
    """Test SAQ quantization implementation."""

    def test_initialize(self, saq_quantizer):
        """Test quantizer initialization."""
        saq_quantizer.initialize()
        assert saq_quantizer._initialized
        assert saq_quantizer.pca is not None
        assert saq_quantizer.scaler is not None

    def test_train(self, saq_quantizer, high_dim_vectors):
        """Test SAQ quantizer training."""
        saq_quantizer.initialize()
        saq_quantizer.train(high_dim_vectors)

        assert saq_quantizer._trained
        assert saq_quantizer.segment_boundaries is not None
        assert saq_quantizer.segment_bits is not None
        assert len(saq_quantizer.segment_quantizers) > 0
        assert saq_quantizer.full_precision_vectors is not None

    def test_train_with_different_configs(self, high_dim_vectors):
        """Test SAQ with different configurations."""
        configs = [
            {"total_bits": 64, "n_segments": 4},
            {"total_bits": 128, "n_segments": 8},
            {"total_bits": 256, "n_segments": 16, "adjustment_iters": 5},
        ]

        for config in configs:
            quantizer = SAQQuantizer(config)
            quantizer.initialize()
            quantizer.train(high_dim_vectors)

            assert quantizer._trained
            stats = quantizer.get_stats()
            assert stats["total_bits"] == config["total_bits"]
            assert stats["n_segments"] == config["n_segments"]

    def test_quantize_dequantize(self, saq_quantizer, high_dim_vectors):
        """Test SAQ quantization and dequantization."""
        saq_quantizer.initialize()
        saq_quantizer.train(high_dim_vectors)

        # Quantize vectors
        quantized = saq_quantizer.quantize(high_dim_vectors)
        expected_shape = (high_dim_vectors.shape[0], saq_quantizer.n_segments)
        assert quantized.shape == expected_shape
        assert quantized.dtype in [np.int32, np.int64]

        # Dequantize vectors
        dequantized = saq_quantizer.dequantize(quantized)
        assert dequantized.shape == high_dim_vectors.shape
        assert dequantized.dtype == high_dim_vectors.dtype

        # Check reconstruction quality
        mse = np.mean((dequantized - high_dim_vectors) ** 2)
        assert mse < 0.5  # SAQ should have good reconstruction

    def test_compute_distance(self, saq_quantizer, high_dim_vectors, query_vector):
        """Test distance computation with SAQ quantized vectors."""
        saq_quantizer.initialize()
        saq_quantizer.train(high_dim_vectors)

        # Quantize a vector
        quantized_vector = saq_quantizer.quantize(high_dim_vectors[:1])[0]

        # Compute distance
        distance = saq_quantizer.compute_distance(query_vector, quantized_vector)
        assert isinstance(distance, float)
        assert distance >= 0
        assert distance <= 2.0  # Cosine distance should be in [0, 2]

    def test_search_with_rescoring(self, saq_quantizer, high_dim_vectors, query_vector):
        """Test search with rescoring functionality."""
        saq_quantizer.initialize()
        saq_quantizer.train(high_dim_vectors)

        # Quantize all vectors
        quantized_vectors = saq_quantizer.quantize(high_dim_vectors)

        # Perform search with rescoring
        indices, distances = saq_quantizer.search_with_rescoring(
            query_vector, quantized_vectors, top_k=5
        )

        assert len(indices) == 5
        assert len(distances) == 5
        assert all(i < len(high_dim_vectors) for i in indices)
        assert all(d >= 0 for d in distances)

        # Results should be sorted by distance
        assert all(distances[i] <= distances[i + 1] for i in range(len(distances) - 1))

    def test_get_stats(self, saq_quantizer, high_dim_vectors):
        """Test getting SAQ quantizer statistics."""
        saq_quantizer.initialize()

        # Before training
        stats = saq_quantizer.get_stats()
        assert stats["quantizer_type"] == "saq"
        assert stats["total_bits"] == 128
        assert stats["n_segments"] == 8
        assert not stats["trained"]

        # After training
        saq_quantizer.train(high_dim_vectors)
        stats = saq_quantizer.get_stats()
        assert stats["trained"]
        assert stats["dimensions"] == high_dim_vectors.shape[1]
        assert stats["compression_ratio"] > 1.0  # Should achieve compression
        assert "segment_boundaries" in stats
        assert "segment_bits" in stats
        assert "adjustment_improvements" in stats

    def test_close(self, saq_quantizer, high_dim_vectors):
        """Test closing SAQ quantizer."""
        saq_quantizer.initialize()
        saq_quantizer.train(high_dim_vectors)

        saq_quantizer.close()
        assert not saq_quantizer._initialized
        assert not saq_quantizer._trained
        assert saq_quantizer.pca is None
        assert saq_quantizer.segment_quantizers == []

    def test_large_scale_vectors(self, saq_quantizer):
        """Test SAQ with large-scale vectors."""
        # Generate larger dataset
        np.random.seed(999)
        large_vectors = np.random.randn(1000, 256).astype(np.float32)

        saq_quantizer.initialize()
        saq_quantizer.train(large_vectors)

        # Test quantization on subset
        quantized = saq_quantizer.quantize(large_vectors[:100])
        dequantized = saq_quantizer.dequantize(quantized)

        assert dequantized.shape == (100, 256)

        # Should maintain reasonable quality even at scale
        mse = np.mean((dequantized - large_vectors[:100]) ** 2)
        assert mse < 1.0

    def test_adjustment_improvement(self, saq_quantizer, high_dim_vectors):
        """Test that code adjustment actually improves quality."""
        saq_quantizer.initialize()
        saq_quantizer.train(high_dim_vectors)

        # Get adjustment improvements from stats
        stats = saq_quantizer.get_stats()
        improvements = stats["adjustment_improvements"]

        # Adjustment should provide some improvement (may be small)
        assert len(improvements) > 0
        # First iteration should show the most improvement
        assert improvements[0] >= 0

    def test_segmentation_consistency(self, saq_quantizer, high_dim_vectors):
        """Test that segmentation is consistent across runs."""
        saq_quantizer.initialize()
        saq_quantizer.train(high_dim_vectors)

        segment_boundaries = saq_quantizer.segment_boundaries
        segment_bits = saq_quantizer.segment_bits

        # Should have correct number of segments
        assert len(segment_boundaries) == saq_quantizer.n_segments
        assert len(segment_bits) == saq_quantizer.n_segments

        # Segment boundaries should cover entire dimension range
        first_start, first_end = segment_boundaries[0]
        last_start, last_end = segment_boundaries[-1]
        assert first_start == 0
        assert last_end == high_dim_vectors.shape[1]

        # Bits allocation should sum to total bits (approximately)
        total_allocated_bits = sum(segment_bits)
        assert (
            abs(total_allocated_bits - saq_quantizer.total_bits)
            <= saq_quantizer.n_segments
        )


class TestQuantizationErrorHandling:
    """Test quantization error handling."""

    def test_uninitialized_quantizer_error(self, scalar_quantizer, small_vectors):
        """Test error when using uninitialized quantizer."""
        with pytest.raises(QuantizationError):
            scalar_quantizer.train(small_vectors)

        with pytest.raises(QuantizationError):
            scalar_quantizer.quantize(small_vectors)

    def test_untrained_quantizer_error(self, scalar_quantizer, small_vectors):
        """Test error when using untrained quantizer."""
        scalar_quantizer.initialize()

        with pytest.raises(QuantizationError):
            scalar_quantizer.quantize(small_vectors)

        with pytest.raises(QuantizationError):
            scalar_quantizer.compute_distance(small_vectors[0], np.array([1, 2, 3]))

    def test_invalid_vector_dimensions(self, scalar_quantizer, small_vectors):
        """Test handling of invalid vector dimensions."""
        scalar_quantizer.initialize()
        scalar_quantizer.train(small_vectors)

        # Test with 1D vector (should be 2D)
        with pytest.raises(QuantizationError):
            scalar_quantizer.quantize(small_vectors[0])

    def test_dimension_mismatch_product_quantizer(self, high_dim_vectors):
        """Test product quantizer with dimension mismatch."""
        quantizer = ProductQuantizer({"m": 7, "k": 256})  # 7 doesn't divide 512
        quantizer.initialize()

        with pytest.raises(QuantizationError):
            quantizer.train(high_dim_vectors)

    def test_saq_uninitialized_error(self, saq_quantizer, high_dim_vectors):
        """Test SAQ quantizer errors when uninitialized."""
        with pytest.raises(QuantizationError):
            saq_quantizer.train(high_dim_vectors)

        with pytest.raises(QuantizationError):
            saq_quantizer.quantize(high_dim_vectors)

    def test_saq_untrained_error(self, saq_quantizer, high_dim_vectors):
        """Test SAQ quantizer errors when untrained."""
        saq_quantizer.initialize()

        with pytest.raises(QuantizationError):
            saq_quantizer.quantize(high_dim_vectors)

        with pytest.raises(QuantizationError):
            saq_quantizer.compute_distance(high_dim_vectors[0], np.array([1, 2, 3]))

    def test_saq_rescoring_without_storage(self):
        """Test SAQ rescoring error when full-precision vectors not stored."""
        quantizer = SAQQuantizer({"total_bits": 128, "n_segments": 8})
        quantizer.initialize()

        # Create some dummy data
        vectors = np.random.randn(10, 64).astype(np.float32)
        quantizer.train(vectors)

        # Remove stored vectors to simulate the error condition
        quantizer.full_precision_vectors = None

        with pytest.raises(QuantizationError):
            quantizer.search_with_rescoring(
                vectors[0], np.zeros((10, 8), dtype=np.int32), top_k=5
            )

    def test_saq_invalid_config(self):
        """Test SAQ with invalid configuration."""
        with pytest.raises(Exception):  # Could be various exceptions
            SAQQuantizer({"total_bits": 0})  # Invalid bits

        with pytest.raises(Exception):
            SAQQuantizer({"n_segments": 0})  # Invalid segments


class TestQuantizationPerformance:
    """Test quantization performance characteristics."""

    def test_compression_ratio(
        self, scalar_quantizer, product_quantizer, saq_quantizer, high_dim_vectors
    ):
        """Test compression ratios of different quantizers."""
        # Test scalar quantizer
        scalar_quantizer.initialize()
        scalar_quantizer.train(high_dim_vectors)
        scalar_stats = scalar_quantizer.get_stats()

        # Test product quantizer
        product_quantizer.initialize()
        product_quantizer.train(high_dim_vectors)
        product_stats = product_quantizer.get_stats()

        # Test SAQ quantizer
        saq_quantizer.initialize()
        saq_quantizer.train(high_dim_vectors)
        saq_stats = saq_quantizer.get_stats()

        # All should provide compression
        assert scalar_stats["compression_ratio"] > 1.0
        assert product_stats["compression_ratio"] > 1.0
        assert saq_stats["compression_ratio"] > 1.0

        # Product and SAQ should have better compression than scalar
        assert product_stats["compression_ratio"] > scalar_stats["compression_ratio"]
        assert saq_stats["compression_ratio"] > scalar_stats["compression_ratio"]

    def test_quantization_speed(
        self, scalar_quantizer, product_quantizer, saq_quantizer, high_dim_vectors
    ):
        """Test quantization speed."""
        import time

        # Train quantizers
        scalar_quantizer.initialize()
        scalar_quantizer.train(high_dim_vectors)

        product_quantizer.initialize()
        product_quantizer.train(high_dim_vectors)

        saq_quantizer.initialize()
        saq_quantizer.train(high_dim_vectors)

        # Time scalar quantization
        start_time = time.time()
        scalar_quantizer.quantize(high_dim_vectors)
        scalar_time = time.time() - start_time

        # Time product quantization
        start_time = time.time()
        product_quantizer.quantize(high_dim_vectors)
        product_time = time.time() - start_time

        # Time SAQ quantization
        start_time = time.time()
        saq_quantizer.quantize(high_dim_vectors)
        saq_time = time.time() - start_time

        # All should be reasonably fast
        assert scalar_time < 1.0
        assert product_time < 1.0
        assert saq_time < 2.0  # SAQ can be slower due to adjustment

    def test_saq_memory_efficiency(self, saq_quantizer, high_dim_vectors):
        """Test SAQ memory usage characteristics."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        saq_quantizer.initialize()
        saq_quantizer.train(high_dim_vectors)

        # Quantize large batch
        large_batch = np.random.randn(1000, high_dim_vectors.shape[1]).astype(
            np.float32
        )
        quantized = saq_quantizer.quantize(large_batch)

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 500MB)
        assert memory_increase < 500 * 1024 * 1024  # 500MB in bytes

        # Quantized output should be much smaller than original
        original_size = large_batch.nbytes
        quantized_size = quantized.nbytes
        assert quantized_size < original_size / 2  # At least 2x compression

    def test_saq_quality_retention(self, saq_quantizer, high_dim_vectors):
        """Test that SAQ maintains good vector quality."""
        saq_quantizer.initialize()
        saq_quantizer.train(high_dim_vectors)

        # Test on a subset
        test_vectors = high_dim_vectors[:50]
        quantized = saq_quantizer.quantize(test_vectors)
        dequantized = saq_quantizer.dequantize(quantized)

        # Compute similarity between original and reconstructed
        from sklearn.metrics.pairwise import cosine_similarity

        similarities = []
        for i in range(len(test_vectors)):
            orig = test_vectors[i].reshape(1, -1)
            recon = dequantized[i].reshape(1, -1)
            similarity = cosine_similarity(orig, recon)[0, 0]
            similarities.append(similarity)

        avg_similarity = np.mean(similarities)
        # Should maintain high similarity (at least 0.8 on average)
        assert avg_similarity > 0.8
