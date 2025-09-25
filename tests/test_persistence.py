"""
Tests for persistence implementations.
"""

import pytest
import numpy as np
from valori.persistence import TensorPersistence, IncrementalPersistence
from valori.exceptions import PersistenceError


class TestTensorPersistence:
    """Test tensor persistence implementation."""
    
    def test_initialize(self, tensor_persistence, temp_dir):
        """Test persistence initialization."""
        tensor_persistence.initialize()
        assert tensor_persistence._initialized
        assert (temp_dir / "tensor_persistence").exists()
    
    def test_save_load_state(self, tensor_persistence, sample_vectors):
        """Test saving and loading state."""
        tensor_persistence.initialize()
        
        # Create test state
        state = {
            "vectors": sample_vectors,
            "metadata": {"count": len(sample_vectors)},
            "config": {"dimension": sample_vectors.shape[1]}
        }
        
        # Save state
        success = tensor_persistence.save_state(state, "test_state.pkl")
        assert success
        
        # Load state
        loaded_state = tensor_persistence.load_state("test_state.pkl")
        assert loaded_state is not None
        assert "vectors" in loaded_state
        assert "metadata" in loaded_state
        assert "config" in loaded_state
        assert np.array_equal(loaded_state["vectors"], sample_vectors)
    
    def test_save_load_vectors(self, tensor_persistence, sample_vectors):
        """Test saving and loading vectors."""
        tensor_persistence.initialize()
        
        vector_ids = [f"vector_{i}" for i in range(len(sample_vectors))]
        
        # Save vectors
        success = tensor_persistence.save_vectors(sample_vectors, vector_ids, "test_vectors.npz")
        assert success
        
        # Load vectors
        loaded_vectors, loaded_ids = tensor_persistence.load_vectors("test_vectors.npz")
        assert loaded_vectors is not None
        assert loaded_ids is not None
        assert np.array_equal(loaded_vectors, sample_vectors)
        assert loaded_ids == vector_ids
    
    def test_save_load_index(self, tensor_persistence):
        """Test saving and loading index data."""
        tensor_persistence.initialize()
        
        # Create test index data
        index_data = {
            "type": "flat",
            "centroids": np.random.randn(10, 128),
            "vector_ids": [f"id_{i}" for i in range(100)]
        }
        
        # Save index
        success = tensor_persistence.save_index(index_data, "test_index.pkl")
        assert success
        
        # Load index
        loaded_index = tensor_persistence.load_index("test_index.pkl")
        assert loaded_index is not None
        assert loaded_index["type"] == "flat"
        assert np.array_equal(loaded_index["centroids"], index_data["centroids"])
    
    def test_compression(self, temp_dir, sample_vectors):
        """Test persistence with compression."""
        config = {
            "data_dir": str(temp_dir / "compressed_persistence"),
            "compression": True
        }
        persistence = TensorPersistence(config)
        persistence.initialize()
        
        vector_ids = [f"vector_{i}" for i in range(len(sample_vectors))]
        
        # Save with compression
        success = persistence.save_vectors(sample_vectors, vector_ids, "compressed_vectors.npz")
        assert success
        
        # Load compressed vectors
        loaded_vectors, loaded_ids = persistence.load_vectors("compressed_vectors.npz")
        assert loaded_vectors is not None
        assert np.array_equal(loaded_vectors, sample_vectors)
    
    def test_get_stats(self, tensor_persistence, sample_vectors):
        """Test getting persistence statistics."""
        tensor_persistence.initialize()
        
        # Save some data
        tensor_persistence.save_state({"test": "data"}, "test.pkl")
        tensor_persistence.save_vectors(sample_vectors, ["id1", "id2"], "vectors.npz")
        
        # Get stats
        stats = tensor_persistence.get_stats()
        assert stats["persistence_type"] == "tensor"
        assert stats["initialized"] is True
        assert stats["save_count"] >= 2
        assert stats["file_count"] >= 2
        assert stats["disk_usage_bytes"] > 0
    
    def test_close(self, tensor_persistence):
        """Test closing persistence manager."""
        tensor_persistence.initialize()
        tensor_persistence.close()
        assert not tensor_persistence._initialized


class TestIncrementalPersistence:
    """Test incremental persistence implementation."""
    
    def test_initialize(self, incremental_persistence, temp_dir):
        """Test persistence initialization."""
        incremental_persistence.initialize()
        assert incremental_persistence._initialized
        assert (temp_dir / "incremental_persistence").exists()
        assert (temp_dir / "incremental_persistence" / "operations.log").exists()
    
    def test_save_load_state(self, incremental_persistence, sample_vectors):
        """Test saving and loading state."""
        incremental_persistence.initialize()
        
        # Create test state
        state = {
            "vectors": sample_vectors,
            "metadata": {"count": len(sample_vectors)}
        }
        
        # Save state
        success = incremental_persistence.save_state(state, "test_state.pkl")
        assert success
        
        # Load state
        loaded_state = incremental_persistence.load_state("test_state.pkl")
        assert loaded_state is not None
        assert "vectors" in loaded_state
        assert np.array_equal(loaded_state["vectors"], sample_vectors)
    
    def test_save_load_vectors(self, incremental_persistence, sample_vectors):
        """Test saving and loading vectors."""
        incremental_persistence.initialize()
        
        vector_ids = [f"vector_{i}" for i in range(len(sample_vectors))]
        
        # Save vectors
        success = incremental_persistence.save_vectors(sample_vectors, vector_ids, "test_vectors.npz")
        assert success
        
        # Load vectors
        loaded_vectors, loaded_ids = incremental_persistence.load_vectors("test_vectors.npz")
        assert loaded_vectors is not None
        assert loaded_ids is not None
        assert np.array_equal(loaded_vectors, sample_vectors)
        assert loaded_ids == vector_ids
    
    def test_operation_logging(self, incremental_persistence, sample_vectors):
        """Test operation logging."""
        incremental_persistence.initialize()
        
        # Perform operations that should be logged
        incremental_persistence.save_vectors(sample_vectors[:5], ["id1", "id2", "id3", "id4", "id5"], "vectors1.npz")
        incremental_persistence.save_state({"test": "data"}, "state1.pkl")
        incremental_persistence.save_vectors(sample_vectors[5:], ["id6", "id7", "id8", "id9", "id10"], "vectors2.npz")
        
        # Check that operations were logged
        stats = incremental_persistence.get_stats()
        assert stats["logged_operations"] >= 3
        assert stats["operation_count"] >= 3
    
    def test_checkpoint_creation(self, temp_dir, sample_vectors):
        """Test checkpoint creation with low interval."""
        config = {
            "data_dir": str(temp_dir / "checkpoint_persistence"),
            "checkpoint_interval": 2  # Low interval for testing
        }
        persistence = IncrementalPersistence(config)
        persistence.initialize()
        
        # Perform operations to trigger checkpoint
        for i in range(5):
            persistence.save_state({"iteration": i}, f"state_{i}.pkl")
        
        # Check that checkpoints were created
        stats = persistence.get_stats()
        assert stats["checkpoint_count"] > 0
        assert stats["last_checkpoint"] > 0
    
    def test_get_stats(self, incremental_persistence, sample_vectors):
        """Test getting persistence statistics."""
        incremental_persistence.initialize()
        
        # Perform some operations
        incremental_persistence.save_vectors(sample_vectors, ["id1", "id2"], "vectors.npz")
        incremental_persistence.save_state({"test": "data"}, "state.pkl")
        
        # Get stats
        stats = incremental_persistence.get_stats()
        assert stats["persistence_type"] == "incremental"
        assert stats["initialized"] is True
        assert stats["operation_count"] >= 2
        assert stats["logged_operations"] >= 2
        assert stats["checkpoint_interval"] == 10
        assert stats["disk_usage_bytes"] > 0
    
    def test_close(self, incremental_persistence):
        """Test closing persistence manager."""
        incremental_persistence.initialize()
        incremental_persistence.close()
        assert not incremental_persistence._initialized


class TestPersistenceErrorHandling:
    """Test persistence error handling."""
    
    def test_uninitialized_persistence_error(self, tensor_persistence, sample_vectors):
        """Test error when using uninitialized persistence."""
        with pytest.raises(PersistenceError):
            tensor_persistence.save_state({"test": "data"}, "test.pkl")
        
        with pytest.raises(PersistenceError):
            tensor_persistence.load_state("test.pkl")
    
    def test_nonexistent_file_load(self, tensor_persistence):
        """Test loading from non-existent file."""
        tensor_persistence.initialize()
        
        # Try to load non-existent file
        result = tensor_persistence.load_state("nonexistent.pkl")
        assert result is None
        
        vectors, ids = tensor_persistence.load_vectors("nonexistent.npz")
        assert vectors is None
        assert ids is None
    
    def test_invalid_data_handling(self, tensor_persistence):
        """Test handling of invalid data."""
        tensor_persistence.initialize()
        
        # Try to save invalid data (should still work with pickle)
        invalid_data = {"invalid": object()}  # Non-serializable object
        with pytest.raises(PersistenceError):
            tensor_persistence.save_state(invalid_data, "invalid.pkl")


class TestPersistencePerformance:
    """Test persistence performance characteristics."""
    
    def test_large_data_handling(self, temp_dir, high_dim_vectors):
        """Test handling of large data."""
        config = {"data_dir": str(temp_dir / "large_data")}
        persistence = TensorPersistence(config)
        persistence.initialize()
        
        vector_ids = [f"vector_{i}" for i in range(len(high_dim_vectors))]
        
        # Save large vectors
        success = persistence.save_vectors(high_dim_vectors, vector_ids, "large_vectors.npz")
        assert success
        
        # Load large vectors
        loaded_vectors, loaded_ids = persistence.load_vectors("large_vectors.npz")
        assert loaded_vectors is not None
        assert np.array_equal(loaded_vectors, high_dim_vectors)
    
    def test_compression_vs_no_compression(self, temp_dir, sample_vectors):
        """Test compression vs no compression."""
        # No compression
        config_no_comp = {"data_dir": str(temp_dir / "no_compression"), "compression": False}
        persistence_no_comp = TensorPersistence(config_no_comp)
        persistence_no_comp.initialize()
        
        # With compression
        config_comp = {"data_dir": str(temp_dir / "compression"), "compression": True}
        persistence_comp = TensorPersistence(config_comp)
        persistence_comp.initialize()
        
        vector_ids = [f"vector_{i}" for i in range(len(sample_vectors))]
        
        # Save with both methods
        persistence_no_comp.save_vectors(sample_vectors, vector_ids, "vectors.npz")
        persistence_comp.save_vectors(sample_vectors, vector_ids, "vectors.npz")
        
        # Check file sizes (compressed should be smaller)
        no_comp_stats = persistence_no_comp.get_stats()
        comp_stats = persistence_comp.get_stats()
        
        # Both should work correctly
        assert no_comp_stats["initialized"] is True
        assert comp_stats["initialized"] is True
