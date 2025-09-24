"""
Advanced indexing algorithms example for the Vectara vector database.

This example demonstrates the use of different indexing algorithms including
LSH (Locality Sensitive Hashing) and Annoy for various use cases and
performance characteristics.
"""

import time
import numpy as np
from pathlib import Path
from valori import VectorDBClient
from valori.storage import MemoryStorage
from valori.indices import FlatIndex, HNSWIndex, IVFIndex, LSHIndex, AnnoyIndex
from valori.processors import ProcessingPipeline


def generate_test_data(num_vectors=10000, dimension=128, num_queries=100):
    """Generate test data for benchmarking."""
    print(f"Generating test data: {num_vectors} vectors of dimension {dimension}")
    
    # Generate random vectors
    np.random.seed(42)
    vectors = np.random.randn(num_vectors, dimension).astype(np.float32)
    
    # Normalize vectors for cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms
    
    # Generate metadata
    metadata = [
        {
            "id": i,
            "category": f"cat_{i % 100}",
            "timestamp": i,
            "feature_vector": vectors[i].tolist()
        }
        for i in range(num_vectors)
    ]
    
    # Generate query vectors
    query_vectors = np.random.randn(num_queries, dimension).astype(np.float32)
    query_norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
    query_vectors = query_vectors / query_norms
    
    return vectors, metadata, query_vectors


def benchmark_index(index, vectors, metadata, query_vectors, k=10, name=""):
    """Benchmark an index implementation."""
    print(f"\n{'='*60}")
    print(f"Benchmarking {name}")
    print(f"{'='*60}")
    
    # Initialize index
    start_time = time.time()
    index.initialize()
    init_time = time.time() - start_time
    print(f"Initialization time: {init_time:.4f} seconds")
    
    # Add vectors
    start_time = time.time()
    if hasattr(index, 'build'):
        # For Annoy, we need to build after adding
        ids = index.add(vectors, metadata)
        index.build()
    else:
        ids = index.add(vectors, metadata)
    add_time = time.time() - start_time
    print(f"Add time: {add_time:.4f} seconds ({len(vectors)/add_time:.0f} vectors/sec)")
    
    # Get index statistics
    stats = index.get_stats()
    print(f"Index stats: {stats}")
    
    # Search benchmark
    search_times = []
    total_results = 0
    
    start_time = time.time()
    for query_vector in query_vectors:
        search_start = time.time()
        results = index.search(query_vector, k=k)
        search_times.append(time.time() - search_start)
        total_results += len(results)
    
    total_search_time = time.time() - start_time
    avg_search_time = np.mean(search_times)
    
    print(f"Search time: {total_search_time:.4f} seconds total")
    print(f"Average search time: {avg_search_time*1000:.2f} ms per query")
    print(f"Search throughput: {len(query_vectors)/total_search_time:.0f} queries/sec")
    print(f"Average results per query: {total_results/len(query_vectors):.1f}")
    
    return {
        "name": name,
        "init_time": init_time,
        "add_time": add_time,
        "add_throughput": len(vectors)/add_time,
        "search_time": total_search_time,
        "avg_search_time": avg_search_time,
        "search_throughput": len(query_vectors)/total_search_time,
        "stats": stats
    }


def compare_indexing_algorithms():
    """Compare different indexing algorithms."""
    print("Valori Advanced Indexing Algorithms Comparison")
    print("=" * 60)
    
    # Generate test data
    vectors, metadata, query_vectors = generate_test_data(
        num_vectors=5000,  # Reduced for faster demo
        dimension=128,
        num_queries=50
    )
    
    # Define index configurations
    indices = [
        (FlatIndex({"metric": "cosine"}), "Flat Index (Exact)"),
        (HNSWIndex({
            "metric": "cosine",
            "M": 16,
            "ef_construction": 200,
            "ef_search": 50
        }), "HNSW Index"),
        (IVFIndex({
            "metric": "cosine",
            "n_clusters": 100,
            "n_probes": 10
        }), "IVF Index"),
        (LSHIndex({
            "metric": "cosine",
            "num_hash_tables": 10,
            "hash_size": 16,
            "num_projections": 64,
            "threshold": 0.3
        }), "LSH Index"),
        (AnnoyIndex({
            "metric": "angular",
            "num_trees": 10,
            "search_k": -1
        }), "Annoy Index")
    ]
    
    # Benchmark each index
    results = []
    for index, name in indices:
        try:
            result = benchmark_index(index, vectors, metadata, query_vectors, k=10, name=name)
            results.append(result)
        except Exception as e:
            print(f"Error benchmarking {name}: {e}")
            continue
        finally:
            # Clean up
            if hasattr(index, 'close'):
                index.close()
    
    # Print comparison summary
    print(f"\n{'='*80}")
    print("PERFORMANCE COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Index Name':<20} {'Add (vec/s)':<12} {'Search (ms)':<12} {'Throughput (q/s)':<16}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['name']:<20} "
              f"{result['add_throughput']:<12.0f} "
              f"{result['avg_search_time']*1000:<12.2f} "
              f"{result['search_throughput']:<16.0f}")
    
    return results


def demonstrate_lsh_features():
    """Demonstrate LSH-specific features."""
    print(f"\n{'='*60}")
    print("LSH (Locality Sensitive Hashing) Features")
    print(f"{'='*60}")
    
    # Generate test data
    vectors, metadata, query_vectors = generate_test_data(
        num_vectors=1000,
        dimension=64,
        num_queries=10
    )
    
    # Test different LSH configurations
    lsh_configs = [
        {
            "name": "High Precision LSH",
            "config": {
                "metric": "cosine",
                "num_hash_tables": 20,
                "hash_size": 20,
                "num_projections": 100,
                "threshold": 0.5
            }
        },
        {
            "name": "High Recall LSH", 
            "config": {
                "metric": "cosine",
                "num_hash_tables": 5,
                "hash_size": 10,
                "num_projections": 32,
                "threshold": 0.1
            }
        },
        {
            "name": "Fast LSH",
            "config": {
                "metric": "cosine",
                "num_hash_tables": 3,
                "hash_size": 8,
                "num_projections": 16,
                "threshold": 0.2
            }
        }
    ]
    
    for lsh_config in lsh_configs:
        print(f"\n--- {lsh_config['name']} ---")
        
        index = LSHIndex(lsh_config['config'])
        index.initialize()
        index.add(vectors, metadata)
        
        # Test search with different thresholds
        query_vector = query_vectors[0]
        
        for threshold in [0.1, 0.3, 0.5, 0.7]:
            results = index.search(query_vector, k=10, threshold=threshold)
            print(f"Threshold {threshold}: {len(results)} results")
        
        index.close()


def demonstrate_annoy_features():
    """Demonstrate Annoy-specific features."""
    print(f"\n{'='*60}")
    print("Annoy (Approximate Nearest Neighbors) Features")
    print(f"{'='*60}")
    
    # Generate test data
    vectors, metadata, query_vectors = generate_test_data(
        num_vectors=1000,
        dimension=64,
        num_queries=10
    )
    
    # Test different Annoy configurations
    annoy_configs = [
        {
            "name": "High Quality Annoy",
            "config": {
                "metric": "angular",
                "num_trees": 50,
                "search_k": 100
            }
        },
        {
            "name": "Fast Annoy",
            "config": {
                "metric": "angular", 
                "num_trees": 5,
                "search_k": 10
            }
        },
        {
            "name": "On-Disk Annoy",
            "config": {
                "metric": "euclidean",
                "num_trees": 20,
                "build_on_disk": True
            }
        }
    ]
    
    for annoy_config in annoy_configs:
        print(f"\n--- {annoy_config['name']} ---")
        
        index = AnnoyIndex(annoy_config['config'])
        index.initialize()
        index.add(vectors, metadata)
        index.build()
        
        # Test search
        query_vector = query_vectors[0]
        results = index.search(query_vector, k=10)
        
        print(f"Search results: {len(results)}")
        if results:
            print(f"Best match distance: {results[0]['distance']:.4f}")
            print(f"Worst match distance: {results[-1]['distance']:.4f}")
        
        # Test save/load functionality
        if not annoy_config['config'].get('build_on_disk', False):
            save_path = "test_annoy_index.annoy"
            index.save(save_path)
            print(f"Index saved to {save_path}")
            
            # Clean up
            import os
            if os.path.exists(save_path):
                os.remove(save_path)
        
        index.close()


def demonstrate_use_cases():
    """Demonstrate different use cases for each index type."""
    print(f"\n{'='*60}")
    print("Index Selection Guide")
    print(f"{'='*60}")
    
    use_cases = [
        {
            "scenario": "Small Dataset (< 1K vectors)",
            "recommendation": "FlatIndex",
            "reason": "Exact results, simple implementation"
        },
        {
            "scenario": "Medium Dataset (1K - 100K vectors)",
            "recommendation": "HNSWIndex or AnnoyIndex", 
            "reason": "Good balance of speed and accuracy"
        },
        {
            "scenario": "Large Dataset (> 100K vectors)",
            "recommendation": "IVFIndex or AnnoyIndex",
            "reason": "Scalable to millions of vectors"
        },
        {
            "scenario": "High-Dimensional Data (> 1000D)",
            "recommendation": "LSHIndex",
            "reason": "Efficient for curse of dimensionality"
        },
        {
            "scenario": "Real-time Search Requirements",
            "recommendation": "AnnoyIndex or LSHIndex",
            "reason": "Fast approximate search"
        },
        {
            "scenario": "Memory-Constrained Environment",
            "recommendation": "IVFIndex with quantization",
            "reason": "Memory-efficient clustering approach"
        },
        {
            "scenario": "Batch Processing",
            "recommendation": "FlatIndex or HNSWIndex",
            "reason": "Good throughput for batch operations"
        }
    ]
    
    print(f"{'Scenario':<30} {'Recommended Index':<20} {'Reason'}")
    print("-" * 80)
    
    for use_case in use_cases:
        print(f"{use_case['scenario']:<30} "
              f"{use_case['recommendation']:<20} "
              f"{use_case['reason']}")


def main():
    """Run the advanced indexing example."""
    print("Vectara Advanced Indexing Algorithms Example")
    print("=" * 60)
    
    try:
        # Compare all indexing algorithms
        results = compare_indexing_algorithms()
        
        # Demonstrate LSH features
        demonstrate_lsh_features()
        
        # Demonstrate Annoy features  
        demonstrate_annoy_features()
        
        # Show use case recommendations
        demonstrate_use_cases()
        
        print(f"\n{'='*60}")
        print("Example completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error running example: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
