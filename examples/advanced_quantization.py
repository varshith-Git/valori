"""
Advanced quantization example for the Vectara vector database.

This example demonstrates the use of quantization for compressing vectors
and improving search performance while maintaining accuracy.
"""

import numpy as np
from valori import VectorDBClient
from valori.storage import MemoryStorage, DiskStorage
from valori.indices import FlatIndex, HNSWIndex
from valori.quantization import ScalarQuantizer, ProductQuantizer


def compare_quantization_methods(vectors, query_vector, metadata):
    """Compare different quantization methods."""
    print("\nComparing Quantization Methods")
    print("-" * 40)
    
    # 1. No quantization (baseline)
    print("\n1. Baseline (No Quantization):")
    storage1 = MemoryStorage({})
    index1 = FlatIndex({"metric": "cosine"})
    client1 = VectorDBClient(storage1, index1)
    client1.initialize()
    
    client1.insert(vectors, metadata)
    results1 = client1.search(query_vector, k=5)
    
    stats1 = client1.get_stats()
    print(f"   Memory usage: {stats1['storage']['memory_usage_mb']:.2f} MB")
    print(f"   Search results: {len(results1)}")
    
    # 2. Scalar quantization
    print("\n2. Scalar Quantization (8-bit):")
    storage2 = MemoryStorage({})
    index2 = FlatIndex({"metric": "cosine"})
    quantizer2 = ScalarQuantizer({"bits": 8})
    client2 = VectorDBClient(storage2, index2, quantizer2)
    client2.initialize()
    
    client2.insert(vectors, metadata)
    results2 = client2.search(query_vector, k=5)
    
    stats2 = client2.get_stats()
    print(f"   Memory usage: {stats2['storage']['memory_usage_mb']:.2f} MB")
    print(f"   Compression ratio: {stats2['quantization']['compression_ratio']:.2f}")
    print(f"   Search results: {len(results2)}")
    
    # 3. Product quantization
    print("\n3. Product Quantization (8 subvectors, 256 centroids each):")
    storage3 = MemoryStorage({})
    index3 = FlatIndex({"metric": "cosine"})
    quantizer3 = ProductQuantizer({"m": 8, "k": 256})
    client3 = VectorDBClient(storage3, index3, quantizer3)
    client3.initialize()
    
    client3.insert(vectors, metadata)
    results3 = client3.search(query_vector, k=5)
    
    stats3 = client3.get_stats()
    print(f"   Memory usage: {stats3['storage']['memory_usage_mb']:.2f} MB")
    print(f"   Compression ratio: {stats3['quantization']['compression_ratio']:.2f}")
    print(f"   Search results: {len(results3)}")
    
    # Compare search quality
    print("\n4. Search Quality Comparison:")
    print("   Baseline top 5 results:")
    for i, result in enumerate(results1):
        print(f"     {i+1}. ID: {result['id']}, Distance: {result['distance']:.4f}")
    
    print("   Scalar quantization top 5 results:")
    for i, result in enumerate(results2):
        print(f"     {i+1}. ID: {result['id']}, Distance: {result['distance']:.4f}")
    
    print("   Product quantization top 5 results:")
    for i, result in enumerate(results3):
        print(f"     {i+1}. ID: {result['id']}, Distance: {result['distance']:.4f}")
    
    # Cleanup
    client1.close()
    client2.close()
    client3.close()


def demonstrate_quantization_with_different_indices(vectors, query_vector, metadata):
    """Demonstrate quantization with different index types."""
    print("\nQuantization with Different Index Types")
    print("-" * 40)
    
    # Flat index with scalar quantization
    print("\n1. Flat Index + Scalar Quantization:")
    storage1 = MemoryStorage({})
    index1 = FlatIndex({"metric": "cosine"})
    quantizer1 = ScalarQuantizer({"bits": 8})
    client1 = VectorDBClient(storage1, index1, quantizer1)
    client1.initialize()
    
    client1.insert(vectors, metadata)
    results1 = client1.search(query_vector, k=5)
    
    stats1 = client1.get_stats()
    print(f"   Index type: {stats1['index']['index_type']}")
    print(f"   Quantizer type: {stats1['quantization']['quantizer_type']}")
    print(f"   Vector count: {stats1['index']['vector_count']}")
    
    # HNSW index with product quantization
    print("\n2. HNSW Index + Product Quantization:")
    storage2 = MemoryStorage({})
    index2 = HNSWIndex({"metric": "cosine", "m": 16, "ef_construction": 200})
    quantizer2 = ProductQuantizer({"m": 8, "k": 256})
    client2 = VectorDBClient(storage2, index2, quantizer2)
    client2.initialize()
    
    client2.insert(vectors, metadata)
    results2 = client2.search(query_vector, k=5)
    
    stats2 = client2.get_stats()
    print(f"   Index type: {stats2['index']['index_type']}")
    print(f"   Quantizer type: {stats2['quantization']['quantizer_type']}")
    print(f"   Vector count: {stats2['index']['vector_count']}")
    print(f"   HNSW levels: {stats2['index']['levels']}")
    
    # Compare search performance
    import time
    
    print("\n3. Search Performance Comparison:")
    
    # Time flat + scalar
    start_time = time.time()
    for _ in range(10):
        client1.search(query_vector, k=5)
    flat_scalar_time = time.time() - start_time
    
    # Time HNSW + product
    start_time = time.time()
    for _ in range(10):
        client2.search(query_vector, k=5)
    hnsw_product_time = time.time() - start_time
    
    print(f"   Flat + Scalar (10 searches): {flat_scalar_time:.4f} seconds")
    print(f"   HNSW + Product (10 searches): {hnsw_product_time:.4f} seconds")
    
    # Cleanup
    client1.close()
    client2.close()


def demonstrate_quantization_accuracy(vectors, query_vector):
    """Demonstrate quantization accuracy with different bit depths."""
    print("\nQuantization Accuracy Analysis")
    print("-" * 40)
    
    bit_depths = [4, 8, 16]
    
    for bits in bit_depths:
        print(f"\n{bits}-bit Scalar Quantization:")
        
        storage = MemoryStorage({})
        index = FlatIndex({"metric": "cosine"})
        quantizer = ScalarQuantizer({"bits": bits})
        client = VectorDBClient(storage, index, quantizer)
        client.initialize()
        
        # Insert vectors
        client.insert(vectors)
        
        # Search
        results = client.search(query_vector, k=5)
        
        # Get quantization stats
        stats = client.get_stats()
        compression_ratio = stats['quantization']['compression_ratio']
        
        print(f"   Compression ratio: {compression_ratio:.3f}")
        print(f"   Top result distance: {results[0]['distance']:.4f}")
        print(f"   Memory usage: {stats['storage']['memory_usage_mb']:.2f} MB")
        
        client.close()


def main():
    """Run advanced quantization example."""
    print("Vectara Vector Database - Advanced Quantization Example")
    print("=" * 60)
    
    # Generate sample data
    print("\nGenerating sample data...")
    np.random.seed(42)
    
    # Create 500 random vectors of dimension 256
    vectors = np.random.randn(500, 256).astype(np.float32)
    
    # Create metadata
    metadata = [
        {"id": i, "category": f"category_{i % 20}", "value": float(i)}
        for i in range(500)
    ]
    
    # Create query vector
    query_vector = vectors[0]
    
    print(f"Generated {len(vectors)} vectors of dimension {vectors.shape[1]}")
    
    # Run comparisons
    compare_quantization_methods(vectors, query_vector, metadata)
    demonstrate_quantization_with_different_indices(vectors, query_vector, metadata)
    demonstrate_quantization_accuracy(vectors, query_vector)
    
    print("\n" + "=" * 60)
    print("Advanced quantization example completed!")


if __name__ == "__main__":
    main()
