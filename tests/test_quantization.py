"""
Tests for quantization implementations.
"""

import pytest
import numpy as np
from valori.quantization import ScalarQuantizer, ProductQuantizer
from valori.exceptions import QuantizationError


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
        assert stats["compression_ratio"] == 8 / 32  # 8 bits vs 32-bit floats
    
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
        distance = product_quantizer.compute_distance(query_vector[:512], quantized_vector)
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


class TestQuantizationPerformance:
    """Test quantization performance characteristics."""
    
    def test_compression_ratio(self, scalar_quantizer, product_quantizer, high_dim_vectors):
        """Test compression ratios of different quantizers."""
        # Test scalar quantizer
        scalar_quantizer.initialize()
        scalar_quantizer.train(high_dim_vectors)
        scalar_stats = scalar_quantizer.get_stats()
        
        # Test product quantizer
        product_quantizer.initialize()
        product_quantizer.train(high_dim_vectors)
        product_stats = product_quantizer.get_stats()
        
        # Product quantization should have better compression
        assert product_stats["compression_ratio"] < scalar_stats["compression_ratio"]
    
    def test_quantization_speed(self, scalar_quantizer, product_quantizer, high_dim_vectors):
        """Test quantization speed."""
        import time
        
        # Train quantizers
        scalar_quantizer.initialize()
        scalar_quantizer.train(high_dim_vectors)
        
        product_quantizer.initialize()
        product_quantizer.train(high_dim_vectors)
        
        # Time scalar quantization
        start_time = time.time()
        scalar_quantizer.quantize(high_dim_vectors)
        scalar_time = time.time() - start_time
        
        # Time product quantization
        start_time = time.time()
        product_quantizer.quantize(high_dim_vectors)
        product_time = time.time() - start_time
        
        # Both should be reasonably fast
        assert scalar_time < 1.0
        assert product_time < 1.0
