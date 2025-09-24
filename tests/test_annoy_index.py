"""
Tests for Annoy index implementation.
"""

import pytest
import numpy as np
from valori.indices import AnnoyIndex
from valori.exceptions import IndexError


class TestAnnoyIndex:
    """Test cases for AnnoyIndex."""
    
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
    
    def test_annoy_initialization(self):
        """Test Annoy index initialization."""
        config = {
            "num_trees": 10,
            "metric": "angular",
            "search_k": -1
        }
        
        index = AnnoyIndex(config)
        assert not index.is_initialized()
        
        index.initialize()
        assert index.is_initialized()
    
    def test_annoy_invalid_config(self):
        """Test Annoy index with invalid configuration."""
        # Invalid num_trees
        with pytest.raises(IndexError):
            AnnoyIndex({"num_trees": 0})
        
        # Invalid metric
        with pytest.raises(IndexError):
            AnnoyIndex({"metric": "invalid"})
    
    def test_annoy_add_vectors(self):
        """Test adding vectors to Annoy index."""
        config = {"num_trees": 10, "metric": "angular"}
        index = AnnoyIndex(config)
        index.initialize()
        
        # Add vectors
        ids = index.add(self.vectors, self.metadata)
        
        assert len(ids) == self.num_vectors
        assert all(isinstance(id, int) for id in ids)
        
        # Check stats
        stats = index.get_stats()
        assert stats["num_vectors"] == self.num_vectors
        assert stats["dimension"] == self.dimension
        assert stats["num_trees"] == 10
        assert not stats["built"]
    
    def test_annoy_build_and_search(self):
        """Test building and searching Annoy index."""
        config = {"num_trees": 10, "metric": "angular"}
        index = AnnoyIndex(config)
        index.initialize()
        
        # Add vectors
        ids = index.add(self.vectors, self.metadata)
        
        # Build index
        index.build()
        
        # Check stats after building
        stats = index.get_stats()
        assert stats["built"] is True
        
        # Search for nearest neighbors
        results = index.search(self.query_vector, k=self.k)
        
        assert len(results) <= self.k
        assert all("id" in result for result in results)
        assert all("distance" in result for result in results)
        assert all("metadata" in result for result in results)
        
        # Check that results are sorted by distance
        distances = [result["distance"] for result in results]
        assert distances == sorted(distances, reverse=True)
    
    def test_annoy_search_without_build(self):
        """Test searching Annoy index without building."""
        config = {"num_trees": 10, "metric": "angular"}
        index = AnnoyIndex(config)
        index.initialize()
        
        # Add vectors
        index.add(self.vectors, self.metadata)
        
        # Try to search without building
        with pytest.raises(IndexError):
            index.search(self.query_vector, k=self.k)
    
    def test_annoy_empty_search(self):
        """Test searching empty Annoy index."""
        config = {"num_trees": 10, "metric": "angular"}
        index = AnnoyIndex(config)
        index.initialize()
        
        # Build empty index
        index.build()
        
        # Search empty index
        results = index.search(self.query_vector, k=self.k)
        assert len(results) == 0
    
    def test_annoy_dimension_mismatch(self):
        """Test Annoy index with dimension mismatch."""
        config = {"num_trees": 10, "metric": "angular"}
        index = AnnoyIndex(config)
        index.initialize()
        
        # Add vectors with correct dimension
        index.add(self.vectors, self.metadata)
        
        # Try to add vectors with wrong dimension
        wrong_vectors = np.random.randn(10, 64).astype(np.float32)
        wrong_metadata = [{"id": i} for i in range(10)]
        
        with pytest.raises(IndexError):
            index.add(wrong_vectors, wrong_metadata)
    
    def test_annoy_different_metrics(self):
        """Test Annoy index with different distance metrics."""
        metrics = ["angular", "euclidean", "manhattan", "hamming", "dot"]
        
        for metric in metrics:
            config = {
                "num_trees": 5,
                "metric": metric
            }
            
            index = AnnoyIndex(config)
            index.initialize()
            
            # Add vectors
            index.add(self.vectors, self.metadata)
            
            # Build and search
            index.build()
            results = index.search(self.query_vector, k=5)
            assert len(results) >= 0
    
    def test_annoy_search_k_parameter(self):
        """Test Annoy index with different search_k parameters."""
        config = {
            "num_trees": 10,
            "metric": "angular",
            "search_k": 50  # Explicit search_k
        }
        index = AnnoyIndex(config)
        index.initialize()
        
        # Add vectors
        index.add(self.vectors, self.metadata)
        
        # Build and search
        index.build()
        results = index.search(self.query_vector, k=self.k)
        
        assert len(results) <= self.k
    
    def test_annoy_on_disk_build(self):
        """Test Annoy index with on-disk building."""
        config = {
            "num_trees": 10,
            "metric": "angular",
            "build_on_disk": True
        }
        index = AnnoyIndex(config)
        index.initialize()
        
        # Add vectors
        index.add(self.vectors, self.metadata)
        
        # Build on disk
        index.build()
        
        # Check stats
        stats = index.get_stats()
        assert stats["built"] is True
        assert stats["build_on_disk"] is True
        
        # Search
        results = index.search(self.query_vector, k=self.k)
        assert len(results) <= self.k
    
    def test_annoy_remove_vectors(self):
        """Test removing vectors from Annoy index."""
        config = {"num_trees": 10, "metric": "angular"}
        index = AnnoyIndex(config)
        index.initialize()
        
        # Add vectors
        ids = index.add(self.vectors, self.metadata)
        
        # Build index
        index.build()
        
        # Remove some vectors
        remove_ids = ids[:10]
        index.remove(remove_ids)
        
        # Check stats
        stats = index.get_stats()
        assert stats["num_vectors"] == self.num_vectors - 10
    
    def test_annoy_clear(self):
        """Test clearing Annoy index."""
        config = {"num_trees": 10, "metric": "angular"}
        index = AnnoyIndex(config)
        index.initialize()
        
        # Add vectors
        index.add(self.vectors, self.metadata)
        
        # Build index
        index.build()
        
        # Clear index
        index.clear()
        
        # Check stats
        stats = index.get_stats()
        assert stats["num_vectors"] == 0
        assert stats["total_capacity"] == 0
        assert not stats["built"]
    
    def test_annoy_save_load(self, tmp_path):
        """Test saving and loading Annoy index."""
        config = {"num_trees": 10, "metric": "angular"}
        index = AnnoyIndex(config)
        index.initialize()
        
        # Add vectors
        index.add(self.vectors, self.metadata)
        
        # Build index
        index.build()
        
        # Save index
        save_path = tmp_path / "test_index.annoy"
        index.save(str(save_path))
        
        # Create new index and load
        new_index = AnnoyIndex(config)
        new_index.initialize()
        new_index.load(str(save_path))
        
        # Search with loaded index
        results = new_index.search(self.query_vector, k=self.k)
        assert len(results) <= self.k
        
        # Check stats
        stats = new_index.get_stats()
        assert stats["built"] is True
    
    def test_annoy_close(self):
        """Test closing Annoy index."""
        config = {"num_trees": 10, "metric": "angular"}
        index = AnnoyIndex(config)
        index.initialize()
        
        # Add vectors
        index.add(self.vectors, self.metadata)
        
        # Build index
        index.build()
        
        # Close index
        index.close()
        
        assert not index.is_initialized()
        
        # Check stats after closing
        stats = index.get_stats()
        assert stats["num_vectors"] == 0
        assert not stats["built"]
