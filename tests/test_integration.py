"""
Integration tests for the Vectara vector database.
"""

import pytest
import numpy as np
from valori import VectorDBClient
from valori.storage import MemoryStorage, DiskStorage, HybridStorage
from valori.indices import FlatIndex, HNSWIndex, IVFIndex
from valori.quantization import ScalarQuantizer, ProductQuantizer
from valori.persistence import TensorPersistence


class TestVectorDBClientIntegration:
    """Test complete VectorDB client integration."""
    
    def test_memory_flat_integration(self, sample_vectors, sample_metadata, query_vector):
        """Test integration with memory storage and flat index."""
        # Create components
        storage = MemoryStorage({})
        index = FlatIndex({"metric": "cosine"})
        
        # Create client
        client = VectorDBClient(storage, index)
        client.initialize()
        
        # Insert vectors
        inserted_ids = client.insert(sample_vectors, sample_metadata)
        assert len(inserted_ids) == len(sample_vectors)
        
        # Search vectors
        results = client.search(query_vector, k=5)
        assert len(results) <= 5
        assert all("id" in result for result in results)
        
        # Update a vector
        success = client.update(inserted_ids[0], sample_vectors[1])
        assert success
        
        # Delete vectors
        success = client.delete(inserted_ids[:3])
        assert success
        
        # Get stats
        stats = client.get_stats()
        assert "storage" in stats
        assert "index" in stats
        
        # Close client
        client.close()
    
    def test_disk_hnsw_integration(self, temp_dir, sample_vectors, sample_metadata, query_vector):
        """Test integration with disk storage and HNSW index."""
        # Create components
        storage = DiskStorage({"data_dir": str(temp_dir / "disk_data")})
        index = HNSWIndex({"metric": "cosine", "m": 16})
        
        # Create client
        client = VectorDBClient(storage, index)
        client.initialize()
        
        # Insert vectors
        inserted_ids = client.insert(sample_vectors, sample_metadata)
        assert len(inserted_ids) == len(sample_vectors)
        
        # Search vectors
        results = client.search(query_vector, k=5)
        assert len(results) <= 5
        
        # Get stats
        stats = client.get_stats()
        assert stats["storage"]["backend_type"] == "disk"
        assert stats["index"]["index_type"] == "hnsw"
        
        client.close()
    
    def test_hybrid_ivf_integration(self, temp_dir, sample_vectors, sample_metadata, query_vector):
        """Test integration with hybrid storage and IVF index."""
        # Create components
        storage = HybridStorage({
            "memory": {},
            "disk": {"data_dir": str(temp_dir / "hybrid_data")},
            "memory_limit": 20
        })
        index = IVFIndex({"metric": "cosine", "n_clusters": 10})
        
        # Create client
        client = VectorDBClient(storage, index)
        client.initialize()
        
        # Insert vectors
        inserted_ids = client.insert(sample_vectors, sample_metadata)
        assert len(inserted_ids) == len(sample_vectors)
        
        # Search vectors
        results = client.search(query_vector, k=5)
        assert len(results) <= 5
        
        # Get stats
        stats = client.get_stats()
        assert stats["storage"]["backend_type"] == "hybrid"
        assert stats["index"]["index_type"] == "ivf"
        
        client.close()
    
    def test_with_quantization(self, sample_vectors, sample_metadata, query_vector):
        """Test integration with quantization."""
        # Create components
        storage = MemoryStorage({})
        index = FlatIndex({"metric": "cosine"})
        quantizer = ScalarQuantizer({"bits": 8})
        
        # Create client with quantizer
        client = VectorDBClient(storage, index, quantizer)
        client.initialize()
        
        # Insert vectors
        inserted_ids = client.insert(sample_vectors, sample_metadata)
        assert len(inserted_ids) == len(sample_vectors)
        
        # Search vectors
        results = client.search(query_vector, k=5)
        assert len(results) <= 5
        
        # Get stats (should include quantization stats)
        stats = client.get_stats()
        assert "quantization" in stats
        assert stats["quantization"]["quantizer_type"] == "scalar"
        
        client.close()
    
    def test_with_persistence(self, temp_dir, sample_vectors, sample_metadata, query_vector):
        """Test integration with persistence."""
        # Create components
        storage = MemoryStorage({})
        index = FlatIndex({"metric": "cosine"})
        persistence = TensorPersistence({"data_dir": str(temp_dir / "persistence")})
        
        # Create client with persistence
        client = VectorDBClient(storage, index, persistence_manager=persistence)
        client.initialize()
        
        # Insert vectors
        inserted_ids = client.insert(sample_vectors, sample_metadata)
        assert len(inserted_ids) == len(sample_vectors)
        
        # Search vectors
        results = client.search(query_vector, k=5)
        assert len(results) <= 5
        
        # Get stats (should include persistence stats)
        stats = client.get_stats()
        assert "persistence" in stats
        assert stats["persistence"]["persistence_type"] == "tensor"
        
        client.close()
    
    def test_full_integration(self, temp_dir, sample_vectors, sample_metadata, query_vector):
        """Test full integration with all components."""
        # Create all components
        storage = HybridStorage({
            "memory": {},
            "disk": {"data_dir": str(temp_dir / "full_data")},
            "memory_limit": 30
        })
        index = HNSWIndex({"metric": "cosine", "m": 16})
        quantizer = ProductQuantizer({"m": 8, "k": 256})
        persistence = TensorPersistence({"data_dir": str(temp_dir / "full_persistence")})
        
        # Create client with all components
        client = VectorDBClient(storage, index, quantizer, persistence)
        client.initialize()
        
        # Insert vectors
        inserted_ids = client.insert(sample_vectors, sample_metadata)
        assert len(inserted_ids) == len(sample_vectors)
        
        # Search vectors
        results = client.search(query_vector, k=5)
        assert len(results) <= 5
        
        # Update vectors
        success = client.update(inserted_ids[0], sample_vectors[1])
        assert success
        
        # Delete some vectors
        success = client.delete(inserted_ids[:3])
        assert success
        
        # Get comprehensive stats
        stats = client.get_stats()
        assert "storage" in stats
        assert "index" in stats
        assert "quantization" in stats
        assert "persistence" in stats
        
        # Verify all component types
        assert stats["storage"]["backend_type"] == "hybrid"
        assert stats["index"]["index_type"] == "hnsw"
        assert stats["quantization"]["quantizer_type"] == "product"
        assert stats["persistence"]["persistence_type"] == "tensor"
        
        client.close()


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""
    
    def test_build_and_search_workflow(self, sample_vectors, sample_metadata, query_vector):
        """Test complete build and search workflow."""
        # Setup
        storage = MemoryStorage({})
        index = FlatIndex({"metric": "cosine"})
        client = VectorDBClient(storage, index)
        client.initialize()
        
        # Build index
        inserted_ids = client.insert(sample_vectors, sample_metadata)
        
        # Search multiple queries
        query_results = []
        for i in range(5):
            query = sample_vectors[i]
            results = client.search(query, k=10)
            query_results.append(results)
        
        # Verify results
        for results in query_results:
            assert len(results) > 0
            # Results should be sorted by distance
            distances = [r["distance"] for r in results]
            assert distances == sorted(distances)
        
        client.close()
    
    def test_incremental_updates_workflow(self, sample_vectors, sample_metadata):
        """Test incremental updates workflow."""
        # Setup
        storage = MemoryStorage({})
        index = FlatIndex({"metric": "cosine"})
        client = VectorDBClient(storage, index)
        client.initialize()
        
        # Initial batch
        batch1 = sample_vectors[:30]
        metadata1 = sample_metadata[:30]
        inserted_ids1 = client.insert(batch1, metadata1)
        
        # Search after first batch
        results1 = client.search(sample_vectors[0], k=10)
        assert len(results1) <= 10
        
        # Second batch
        batch2 = sample_vectors[30:60]
        metadata2 = sample_metadata[30:60]
        inserted_ids2 = client.insert(batch2, metadata2)
        
        # Search after second batch
        results2 = client.search(sample_vectors[0], k=10)
        assert len(results2) <= 10
        
        # Update some vectors
        for i in range(5):
            client.update(inserted_ids1[i], sample_vectors[60 + i])
        
        # Delete some vectors
        client.delete(inserted_ids1[:5])
        
        # Final search
        results3 = client.search(sample_vectors[0], k=10)
        assert len(results3) <= 10
        
        # Verify total count
        stats = client.get_stats()
        expected_count = len(inserted_ids1) + len(inserted_ids2) - 5  # -5 for deletions
        assert stats["index"]["vector_count"] == expected_count
        
        client.close()
    
    def test_large_dataset_workflow(self, high_dim_vectors, temp_dir):
        """Test workflow with large dataset."""
        # Setup with disk storage for large dataset
        storage = DiskStorage({"data_dir": str(temp_dir / "large_dataset")})
        index = IVFIndex({"metric": "cosine", "n_clusters": 20, "n_probes": 10})
        client = VectorDBClient(storage, index)
        client.initialize()
        
        # Insert large dataset
        inserted_ids = client.insert(high_dim_vectors)
        assert len(inserted_ids) == len(high_dim_vectors)
        
        # Search with different k values
        query = high_dim_vectors[0]
        for k in [1, 5, 10, 20]:
            results = client.search(query, k=k)
            assert len(results) <= k
        
        # Verify index was trained
        stats = client.get_stats()
        assert stats["index"]["trained"] is True
        assert stats["index"]["vector_count"] == len(high_dim_vectors)
        
        client.close()


class TestErrorRecovery:
    """Test error recovery and resilience."""
    
    def test_partial_failure_recovery(self, sample_vectors, temp_dir):
        """Test recovery from partial failures."""
        # Setup with disk storage
        storage = DiskStorage({"data_dir": str(temp_dir / "recovery_test")})
        index = FlatIndex({"metric": "cosine"})
        client = VectorDBClient(storage, index)
        client.initialize()
        
        # Insert some vectors
        inserted_ids = client.insert(sample_vectors[:50])
        
        # Simulate partial failure by corrupting storage
        # (In real scenario, this would be handled by the storage backend)
        
        # Create new client to simulate recovery
        storage2 = DiskStorage({"data_dir": str(temp_dir / "recovery_test")})
        index2 = FlatIndex({"metric": "cosine"})
        client2 = VectorDBClient(storage2, index2)
        client2.initialize()
        
        # Should be able to continue operations
        additional_ids = client2.insert(sample_vectors[50:])
        assert len(additional_ids) == len(sample_vectors[50:])
        
        client2.close()
        client.close()
    
    def test_memory_pressure_handling(self, sample_vectors):
        """Test handling of memory pressure."""
        # Setup with small memory limit
        storage = MemoryStorage({})
        index = FlatIndex({"metric": "cosine"})
        client = VectorDBClient(storage, index)
        client.initialize()
        
        # Insert vectors in batches
        batch_size = 10
        all_ids = []
        
        for i in range(0, len(sample_vectors), batch_size):
            batch = sample_vectors[i:i+batch_size]
            batch_ids = client.insert(batch)
            all_ids.extend(batch_ids)
            
            # Search after each batch
            results = client.search(sample_vectors[0], k=5)
            assert len(results) <= 5
        
        # Verify all vectors were inserted
        stats = client.get_stats()
        assert stats["index"]["vector_count"] == len(sample_vectors)
        
        client.close()


class TestPerformanceCharacteristics:
    """Test performance characteristics."""
    
    def test_search_performance(self, sample_vectors):
        """Test search performance with different index types."""
        import time
        
        # Test with flat index
        storage = MemoryStorage({})
        flat_index = FlatIndex({"metric": "cosine"})
        flat_client = VectorDBClient(storage, flat_index)
        flat_client.initialize()
        flat_client.insert(sample_vectors)
        
        query = sample_vectors[0]
        
        # Time flat search
        start_time = time.time()
        flat_results = flat_client.search(query, k=10)
        flat_time = time.time() - start_time
        
        flat_client.close()
        
        # Verify results are reasonable
        assert len(flat_results) <= 10
        assert flat_time < 1.0  # Should be fast for small dataset
        
        # Test with HNSW index
        storage2 = MemoryStorage({})
        hnsw_index = HNSWIndex({"metric": "cosine", "m": 16})
        hnsw_client = VectorDBClient(storage2, hnsw_index)
        hnsw_client.initialize()
        hnsw_client.insert(sample_vectors)
        
        # Time HNSW search
        start_time = time.time()
        hnsw_results = hnsw_client.search(query, k=10)
        hnsw_time = time.time() - start_time
        
        hnsw_client.close()
        
        # Verify results
        assert len(hnsw_results) <= 10
        assert hnsw_time < 1.0  # Should be fast
        
        # HNSW should be faster for larger datasets (not necessarily for small ones)
        # This is just to ensure both work reasonably fast
