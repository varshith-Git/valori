"""
Tests for index implementations.
"""

import pytest
import numpy as np
from valori.indices import FlatIndex, HNSWIndex, IVFIndex
from valori.exceptions import ValoriValoriIndexError


class TestFlatIndex:
    """Test flat index implementation."""
    
    def test_initialize(self, flat_index, memory_storage):
        """Test index initialization."""
        flat_index.initialize(memory_storage)
        assert flat_index._initialized
    
    def test_insert_vectors(self, flat_index, memory_storage, small_vectors):
        """Test inserting vectors."""
        flat_index.initialize(memory_storage)
        
        inserted_ids = flat_index.insert(small_vectors)
        assert len(inserted_ids) == len(small_vectors)
        assert all(isinstance(id, str) for id in inserted_ids)
    
    def test_search_vectors(self, flat_index, memory_storage, small_vectors, query_vector):
        """Test searching vectors."""
        flat_index.initialize(memory_storage)
        
        # Insert vectors
        flat_index.insert(small_vectors)
        
        # Search with query vector
        results = flat_index.search(query_vector, k=5)
        assert len(results) <= 5
        assert all("id" in result for result in results)
        assert all("distance" in result for result in results)
        assert all("metadata" in result for result in results)
    
    def test_search_with_metadata(self, flat_index, memory_storage, small_vectors, sample_metadata):
        """Test searching with metadata."""
        flat_index.initialize(memory_storage)
        
        # Insert vectors with metadata
        flat_index.insert(small_vectors, sample_metadata[:len(small_vectors)])
        
        # Search
        results = flat_index.search(small_vectors[0], k=3)
        assert len(results) <= 3
        
        # Check that metadata is preserved
        for result in results:
            assert result["metadata"] is not None
    
    def test_delete_vectors(self, flat_index, memory_storage, small_vectors):
        """Test deleting vectors."""
        flat_index.initialize(memory_storage)
        
        # Insert vectors
        inserted_ids = flat_index.insert(small_vectors)
        
        # Delete some vectors
        ids_to_delete = inserted_ids[:3]
        success = flat_index.delete(ids_to_delete)
        assert success
        
        # Search should return fewer results
        results = flat_index.search(small_vectors[0], k=10)
        assert len(results) <= len(small_vectors) - 3
    
    def test_update_vector(self, flat_index, memory_storage, small_vectors):
        """Test updating vectors."""
        flat_index.initialize(memory_storage)
        
        # Insert vectors
        inserted_ids = flat_index.insert(small_vectors)
        vector_id = inserted_ids[0]
        
        # Update vector
        new_vector = small_vectors[1]
        new_metadata = {"updated": True}
        success = flat_index.update(vector_id, new_vector, new_metadata)
        assert success
        
        # Search for updated vector
        results = flat_index.search(new_vector, k=1)
        assert len(results) > 0
        assert results[0]["id"] == vector_id
    
    def test_get_stats(self, flat_index, memory_storage, small_vectors):
        """Test getting index statistics."""
        flat_index.initialize(memory_storage)
        
        # Insert vectors
        flat_index.insert(small_vectors)
        
        # Get stats
        stats = flat_index.get_stats()
        assert stats["index_type"] == "flat"
        assert stats["vector_count"] == len(small_vectors)
        assert stats["initialized"] is True
    
    def test_close(self, flat_index, memory_storage, small_vectors):
        """Test closing index."""
        flat_index.initialize(memory_storage)
        flat_index.insert(small_vectors)
        
        flat_index.close()
        assert not flat_index._initialized
        assert len(flat_index.vector_ids) == 0


class TestHNSWIndex:
    """Test HNSW index implementation."""
    
    def test_initialize(self, hnsw_index, memory_storage):
        """Test index initialization."""
        hnsw_index.initialize(memory_storage)
        assert hnsw_index._initialized
    
    def test_insert_vectors(self, hnsw_index, memory_storage, small_vectors):
        """Test inserting vectors."""
        hnsw_index.initialize(memory_storage)
        
        inserted_ids = hnsw_index.insert(small_vectors)
        assert len(inserted_ids) == len(small_vectors)
        assert all(isinstance(id, str) for id in inserted_ids)
    
    def test_search_vectors(self, hnsw_index, memory_storage, small_vectors, query_vector):
        """Test searching vectors."""
        hnsw_index.initialize(memory_storage)
        
        # Insert vectors
        hnsw_index.insert(small_vectors)
        
        # Search with query vector
        results = hnsw_index.search(query_vector, k=5)
        assert len(results) <= 5
        assert all("id" in result for result in results)
        assert all("distance" in result for result in results)
    
    def test_get_stats(self, hnsw_index, memory_storage, small_vectors):
        """Test getting index statistics."""
        hnsw_index.initialize(memory_storage)
        
        # Insert vectors
        hnsw_index.insert(small_vectors)
        
        # Get stats
        stats = hnsw_index.get_stats()
        assert stats["index_type"] == "hnsw"
        assert stats["vector_count"] == len(small_vectors)
        assert stats["initialized"] is True
        assert "max_level" in stats
        assert "levels" in stats


class TestIVFIndex:
    """Test IVF index implementation."""
    
    def test_initialize(self, ivf_index, memory_storage):
        """Test index initialization."""
        ivf_index.initialize(memory_storage)
        assert ivf_index._initialized
    
    def test_insert_vectors(self, ivf_index, memory_storage, small_vectors):
        """Test inserting vectors."""
        ivf_index.initialize(memory_storage)
        
        inserted_ids = ivf_index.insert(small_vectors)
        assert len(inserted_ids) == len(small_vectors)
        assert all(isinstance(id, str) for id in inserted_ids)
    
    def test_search_vectors(self, ivf_index, memory_storage, small_vectors, query_vector):
        """Test searching vectors."""
        ivf_index.initialize(memory_storage)
        
        # Insert vectors
        ivf_index.insert(small_vectors)
        
        # Search with query vector
        results = ivf_index.search(query_vector, k=5)
        assert len(results) <= 5
        assert all("id" in result for result in results)
        assert all("distance" in result for result in results)
    
    def test_get_stats(self, ivf_index, memory_storage, small_vectors):
        """Test getting index statistics."""
        ivf_index.initialize(memory_storage)
        
        # Insert vectors
        ivf_index.insert(small_vectors)
        
        # Get stats
        stats = ivf_index.get_stats()
        assert stats["index_type"] == "ivf"
        assert stats["vector_count"] == len(small_vectors)
        assert stats["initialized"] is True
        assert "n_clusters" in stats
        assert "trained" in stats


class TestValoriIndexErrorHandling:
    """Test index error handling."""
    
    def test_uninitialized_index_error(self, flat_index, small_vectors):
        """Test error when using uninitialized index."""
        with pytest.raises(ValoriIndexError):
            flat_index.insert(small_vectors)
        
        with pytest.raises(ValoriIndexError):
            flat_index.search(small_vectors[0])
    
    def test_invalid_vector_dimensions(self, flat_index, memory_storage, sample_vectors):
        """Test handling of invalid vector dimensions."""
        flat_index.initialize(memory_storage)
        
        # Test with 1D vector (should be 2D)
        with pytest.raises(ValoriIndexError):
            flat_index.insert(sample_vectors[0])
    
    def test_invalid_query_vector(self, flat_index, memory_storage, small_vectors, query_vector):
        """Test handling of invalid query vectors."""
        flat_index.initialize(memory_storage)
        flat_index.insert(small_vectors)
        
        # Test with 2D query vector (should be 1D)
        with pytest.raises(ValoriIndexError):
            flat_index.search(small_vectors, k=5)
    
    def test_search_empty_index(self, flat_index, memory_storage, query_vector):
        """Test searching empty index."""
        flat_index.initialize(memory_storage)
        
        results = flat_index.search(query_vector, k=5)
        assert len(results) == 0


class TestIndexMetrics:
    """Test different similarity metrics."""
    
    @pytest.mark.parametrize("metric", ["cosine", "euclidean"])
    def test_different_metrics(self, metric, memory_storage, small_vectors, query_vector):
        """Test indices with different similarity metrics."""
        index = FlatIndex({"metric": metric})
        index.initialize(memory_storage)
        
        # Insert vectors
        index.insert(small_vectors)
        
        # Search
        results = index.search(query_vector, k=3)
        assert len(results) <= 3
        
        # Check that distances are reasonable
        for result in results:
            assert result["distance"] >= 0
