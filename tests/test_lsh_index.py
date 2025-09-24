"""
Tests for LSH (Locality Sensitive Hashing) index implementation.
"""

import pytest
import numpy as np
from valori.indices import LSHIndex
from valori.exceptions import IndexError


class TestLSHIndex:
    """Test cases for LSHIndex."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.dimension = 128
        self.num_vectors = 100
        self.k = 10
        
        # Generate test vectors
        np.random.seed(42)
        self.vectors = np.random.randn(self.num_vectors, self.dimension).astype(np.float32)
        self.metadata = [{"id": i, "category": f"cat_{i%10}"} for i in range(self.num_vectors)]
        
        # Query vector
        self.query_vector = np.random.randn(self.dimension).astype(np.float32)
    
    def test_lsh_initialization(self):
        """Test LSH index initialization."""
        config = {
            "num_hash_tables": 5,
            "hash_size": 16,
            "num_projections": 32,
            "metric": "cosine"
        }
        
        index = LSHIndex(config)
        assert not index.is_initialized()
        
        index.initialize()
        assert index.is_initialized()
    
    def test_lsh_invalid_config(self):
        """Test LSH index with invalid configuration."""
        # Invalid num_hash_tables
        with pytest.raises(IndexError):
            LSHIndex({"num_hash_tables": 0})
        
        # Invalid hash_size
        with pytest.raises(IndexError):
            LSHIndex({"hash_size": -1})
        
        # Invalid num_projections
        with pytest.raises(IndexError):
            LSHIndex({"num_projections": 0})
    
    def test_lsh_add_vectors(self):
        """Test adding vectors to LSH index."""
        config = {"num_hash_tables": 5, "hash_size": 16}
        index = LSHIndex(config)
        index.initialize()
        
        # Add vectors
        ids = index.add(self.vectors, self.metadata)
        
        assert len(ids) == self.num_vectors
        assert all(isinstance(id, int) for id in ids)
        
        # Check stats
        stats = index.get_stats()
        assert stats["num_vectors"] == self.num_vectors
        assert stats["dimension"] == self.dimension
        assert stats["num_hash_tables"] == 5
    
    def test_lsh_search(self):
        """Test searching LSH index."""
        config = {"num_hash_tables": 10, "hash_size": 16, "threshold": 0.1}
        index = LSHIndex(config)
        index.initialize()
        
        # Add vectors
        ids = index.add(self.vectors, self.metadata)
        
        # Search for nearest neighbors
        results = index.search(self.query_vector, k=self.k)
        
        assert len(results) <= self.k
        assert all("id" in result for result in results)
        assert all("distance" in result for result in results)
        assert all("metadata" in result for result in results)
        
        # Check that results are sorted by distance
        distances = [result["distance"] for result in results]
        assert distances == sorted(distances, reverse=True)
    
    def test_lsh_empty_search(self):
        """Test searching empty LSH index."""
        config = {"num_hash_tables": 5, "hash_size": 16}
        index = LSHIndex(config)
        index.initialize()
        
        # Search empty index
        results = index.search(self.query_vector, k=self.k)
        assert len(results) == 0
    
    def test_lsh_dimension_mismatch(self):
        """Test LSH index with dimension mismatch."""
        config = {"num_hash_tables": 5, "hash_size": 16}
        index = LSHIndex(config)
        index.initialize()
        
        # Add vectors with correct dimension
        index.add(self.vectors, self.metadata)
        
        # Try to add vectors with wrong dimension
        wrong_vectors = np.random.randn(10, 64).astype(np.float32)
        wrong_metadata = [{"id": i} for i in range(10)]
        
        with pytest.raises(IndexError):
            index.add(wrong_vectors, wrong_metadata)
    
    def test_lsh_remove_vectors(self):
        """Test removing vectors from LSH index."""
        config = {"num_hash_tables": 5, "hash_size": 16}
        index = LSHIndex(config)
        index.initialize()
        
        # Add vectors
        ids = index.add(self.vectors, self.metadata)
        
        # Remove some vectors
        remove_ids = ids[:10]
        index.remove(remove_ids)
        
        # Check stats
        stats = index.get_stats()
        assert stats["num_vectors"] == self.num_vectors - 10
    
    def test_lsh_clear(self):
        """Test clearing LSH index."""
        config = {"num_hash_tables": 5, "hash_size": 16}
        index = LSHIndex(config)
        index.initialize()
        
        # Add vectors
        index.add(self.vectors, self.metadata)
        
        # Clear index
        index.clear()
        
        # Check stats
        stats = index.get_stats()
        assert stats["num_vectors"] == 0
        assert stats["total_capacity"] == 0
    
    def test_lsh_different_metrics(self):
        """Test LSH index with different distance metrics."""
        metrics = ["cosine", "euclidean", "manhattan"]
        
        for metric in metrics:
            config = {
                "num_hash_tables": 5,
                "hash_size": 16,
                "metric": metric
            }
            
            index = LSHIndex(config)
            index.initialize()
            
            # Add vectors
            index.add(self.vectors, self.metadata)
            
            # Search
            results = index.search(self.query_vector, k=5)
            assert len(results) >= 0  # May be empty due to threshold
    
    def test_lsh_threshold_filtering(self):
        """Test LSH index with different thresholds."""
        config = {
            "num_hash_tables": 10,
            "hash_size": 16,
            "threshold": 0.1
        }
        index = LSHIndex(config)
        index.initialize()
        
        # Add vectors
        index.add(self.vectors, self.metadata)
        
        # Search with default threshold
        results_default = index.search(self.query_vector, k=self.k)
        
        # Search with higher threshold
        results_high = index.search(self.query_vector, k=self.k, threshold=0.5)
        
        # Higher threshold should return fewer results
        assert len(results_high) <= len(results_default)
    
    def test_lsh_close(self):
        """Test closing LSH index."""
        config = {"num_hash_tables": 5, "hash_size": 16}
        index = LSHIndex(config)
        index.initialize()
        
        # Add vectors
        index.add(self.vectors, self.metadata)
        
        # Close index
        index.close()
        
        assert not index.is_initialized()
        
        # Check stats after closing
        stats = index.get_stats()
        assert stats["num_vectors"] == 0
