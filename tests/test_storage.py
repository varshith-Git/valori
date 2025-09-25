"""
Tests for storage backends.
"""

import pytest
import numpy as np
from valori.storage import MemoryStorage, DiskStorage, HybridStorage
from valori.exceptions import StorageError


class TestMemoryStorage:
    """Test memory storage backend."""
    
    def test_initialize(self, memory_storage):
        """Test storage initialization."""
        memory_storage.initialize()
        assert memory_storage._initialized
    
    def test_store_retrieve_vector(self, memory_storage, sample_vectors):
        """Test storing and retrieving vectors."""
        memory_storage.initialize()
        
        vector = sample_vectors[0]
        vector_id = "test_vector_1"
        metadata = {"category": "test"}
        
        # Store vector
        success = memory_storage.store_vector(vector_id, vector, metadata)
        assert success
        
        # Retrieve vector
        retrieved_vector, retrieved_metadata = memory_storage.retrieve_vector(vector_id)
        assert retrieved_vector is not None
        assert retrieved_metadata is not None
        assert np.array_equal(retrieved_vector, vector)
        assert retrieved_metadata == metadata
    
    def test_store_retrieve_without_metadata(self, memory_storage, sample_vectors):
        """Test storing and retrieving vectors without metadata."""
        memory_storage.initialize()
        
        vector = sample_vectors[0]
        vector_id = "test_vector_2"
        
        # Store vector without metadata
        success = memory_storage.store_vector(vector_id, vector)
        assert success
        
        # Retrieve vector
        retrieved_vector, retrieved_metadata = memory_storage.retrieve_vector(vector_id)
        assert retrieved_vector is not None
        assert retrieved_metadata is None
        assert np.array_equal(retrieved_vector, vector)
    
    def test_delete_vector(self, memory_storage, sample_vectors):
        """Test deleting vectors."""
        memory_storage.initialize()
        
        vector = sample_vectors[0]
        vector_id = "test_vector_3"
        
        # Store vector
        memory_storage.store_vector(vector_id, vector)
        
        # Delete vector
        success = memory_storage.delete_vector(vector_id)
        assert success
        
        # Try to retrieve deleted vector
        result = memory_storage.retrieve_vector(vector_id)
        assert result is None
    
    def test_update_vector(self, memory_storage, sample_vectors):
        """Test updating vectors."""
        memory_storage.initialize()
        
        vector = sample_vectors[0]
        updated_vector = sample_vectors[1]
        vector_id = "test_vector_4"
        metadata = {"category": "original"}
        updated_metadata = {"category": "updated"}
        
        # Store original vector
        memory_storage.store_vector(vector_id, vector, metadata)
        
        # Update vector
        success = memory_storage.update_vector(vector_id, updated_vector, updated_metadata)
        assert success
        
        # Retrieve updated vector
        retrieved_vector, retrieved_metadata = memory_storage.retrieve_vector(vector_id)
        assert np.array_equal(retrieved_vector, updated_vector)
        assert retrieved_metadata == updated_metadata
    
    def test_list_vectors(self, memory_storage, sample_vectors):
        """Test listing vectors."""
        memory_storage.initialize()
        
        # Store multiple vectors
        vector_ids = []
        for i in range(5):
            vector_id = f"test_vector_{i}"
            memory_storage.store_vector(vector_id, sample_vectors[i])
            vector_ids.append(vector_id)
        
        # List vectors
        listed_ids = memory_storage.list_vectors()
        assert len(listed_ids) == 5
        assert set(listed_ids) == set(vector_ids)
    
    def test_get_stats(self, memory_storage, sample_vectors):
        """Test getting storage statistics."""
        memory_storage.initialize()
        
        # Store a vector
        memory_storage.store_vector("test_vector", sample_vectors[0])
        
        # Get stats
        stats = memory_storage.get_stats()
        assert stats["backend_type"] == "memory"
        assert stats["vector_count"] == 1
        assert stats["initialized"] is True
        assert stats["memory_usage_bytes"] > 0
    
    def test_close(self, memory_storage, sample_vectors):
        """Test closing storage."""
        memory_storage.initialize()
        memory_storage.store_vector("test_vector", sample_vectors[0])
        
        memory_storage.close()
        assert not memory_storage._initialized
        assert len(memory_storage.vectors) == 0


class TestDiskStorage:
    """Test disk storage backend."""
    
    def test_initialize(self, disk_storage, temp_dir):
        """Test storage initialization."""
        disk_storage.initialize()
        assert disk_storage._initialized
        assert (temp_dir / "disk_data").exists()
    
    def test_store_retrieve_vector(self, disk_storage, sample_vectors):
        """Test storing and retrieving vectors."""
        disk_storage.initialize()
        
        vector = sample_vectors[0]
        vector_id = "test_vector_1"
        metadata = {"category": "test"}
        
        # Store vector
        success = disk_storage.store_vector(vector_id, vector, metadata)
        assert success
        
        # Retrieve vector
        retrieved_vector, retrieved_metadata = disk_storage.retrieve_vector(vector_id)
        assert retrieved_vector is not None
        assert retrieved_metadata is not None
        assert np.array_equal(retrieved_vector, vector)
        assert retrieved_metadata == metadata
    
    def test_delete_vector(self, disk_storage, sample_vectors):
        """Test deleting vectors."""
        disk_storage.initialize()
        
        vector = sample_vectors[0]
        vector_id = "test_vector_3"
        
        # Store vector
        disk_storage.store_vector(vector_id, vector)
        
        # Delete vector
        success = disk_storage.delete_vector(vector_id)
        assert success
        
        # Try to retrieve deleted vector
        result = disk_storage.retrieve_vector(vector_id)
        assert result is None
    
    def test_get_stats(self, disk_storage, sample_vectors):
        """Test getting storage statistics."""
        disk_storage.initialize()
        
        # Store a vector
        disk_storage.store_vector("test_vector", sample_vectors[0])
        
        # Get stats
        stats = disk_storage.get_stats()
        assert stats["backend_type"] == "disk"
        assert stats["vector_count"] == 1
        assert stats["initialized"] is True
        assert stats["disk_usage_bytes"] > 0


class TestHybridStorage:
    """Test hybrid storage backend."""
    
    def test_initialize(self, hybrid_storage):
        """Test storage initialization."""
        hybrid_storage.initialize()
        assert hybrid_storage._initialized
        assert hybrid_storage.memory_storage._initialized
        assert hybrid_storage.disk_storage._initialized
    
    def test_store_retrieve_vector(self, hybrid_storage, sample_vectors):
        """Test storing and retrieving vectors."""
        hybrid_storage.initialize()
        
        vector = sample_vectors[0]
        vector_id = "test_vector_1"
        metadata = {"category": "test"}
        
        # Store vector
        success = hybrid_storage.store_vector(vector_id, vector, metadata)
        assert success
        
        # Retrieve vector
        retrieved_vector, retrieved_metadata = hybrid_storage.retrieve_vector(vector_id)
        assert retrieved_vector is not None
        assert retrieved_metadata is not None
        assert np.array_equal(retrieved_vector, vector)
        assert retrieved_metadata == metadata
    
    def test_memory_limit_eviction(self, hybrid_storage, sample_vectors):
        """Test memory limit and eviction."""
        hybrid_storage.initialize()
        
        # Store vectors beyond memory limit
        for i in range(60):  # More than memory_limit of 50
            vector_id = f"test_vector_{i}"
            hybrid_storage.store_vector(vector_id, sample_vectors[i % len(sample_vectors)])
        
        # Check that some vectors are in memory and some on disk
        memory_stats = hybrid_storage.memory_storage.get_stats()
        disk_stats = hybrid_storage.disk_storage.get_stats()
        
        assert memory_stats["vector_count"] <= hybrid_storage.memory_limit
        assert disk_stats["vector_count"] == 60
    
    def test_get_stats(self, hybrid_storage, sample_vectors):
        """Test getting storage statistics."""
        hybrid_storage.initialize()
        
        # Store a vector
        hybrid_storage.store_vector("test_vector", sample_vectors[0])
        
        # Get stats
        stats = hybrid_storage.get_stats()
        assert stats["backend_type"] == "hybrid"
        assert stats["memory"]["vector_count"] >= 0
        assert stats["disk"]["vector_count"] >= 0
        assert stats["initialized"] is True


class TestStorageErrorHandling:
    """Test storage error handling."""
    
    def test_uninitialized_storage_error(self, memory_storage, sample_vectors):
        """Test error when using uninitialized storage."""
        vector = sample_vectors[0]
        
        with pytest.raises(StorageError):
            memory_storage.store_vector("test", vector)
        
        with pytest.raises(StorageError):
            memory_storage.retrieve_vector("test")
    
    def test_invalid_id_handling(self, memory_storage, sample_vectors):
        """Test handling of invalid IDs."""
        memory_storage.initialize()
        
        # Test with empty ID
        with pytest.raises(StorageError):
            memory_storage.store_vector("", sample_vectors[0])
    
    def test_nonexistent_vector_retrieval(self, memory_storage):
        """Test retrieving non-existent vector."""
        memory_storage.initialize()
        
        result = memory_storage.retrieve_vector("nonexistent")
        assert result is None
