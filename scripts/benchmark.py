#!/usr/bin/env python3
"""
Benchmark script for Vectara vector database.

This script performs comprehensive benchmarks to evaluate the performance
of different configurations and components.
"""

import argparse
import time
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any

from valori import VectorDBClient
from valori.storage import MemoryStorage, DiskStorage, HybridStorage
from valori.indices import FlatIndex, HNSWIndex, IVFIndex
from valori.quantization import ScalarQuantizer, ProductQuantizer
from valori.persistence import TensorPersistence


class BenchmarkSuite:
    """Comprehensive benchmark suite for Vectara."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """Initialize benchmark suite."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
    
    def generate_data(self, num_vectors: int, dimension: int) -> tuple:
        """Generate test data."""
        np.random.seed(42)
        vectors = np.random.randn(num_vectors, dimension).astype(np.float32)
        metadata = [{"id": i, "category": f"cat_{i%10}"} for i in range(num_vectors)]
        return vectors, metadata
    
    def benchmark_storage_backends(self, vectors: np.ndarray, metadata: List[Dict]) -> Dict[str, Any]:
        """Benchmark different storage backends."""
        print("Benchmarking storage backends...")
        
        results = {}
        backends = {
            "memory": MemoryStorage({}),
            "disk": DiskStorage({"data_dir": "./benchmark_disk"}),
            "hybrid": HybridStorage({
                "memory": {},
                "disk": {"data_dir": "./benchmark_hybrid"},
                "memory_limit": 1000
            })
        }
        
        for name, storage in backends.items():
            print(f"  Testing {name} storage...")
            
            index = FlatIndex({"metric": "cosine"})
            client = VectorDBClient(storage, index)
            client.initialize()
            
            # Insert benchmark
            start_time = time.time()
            inserted_ids = client.insert(vectors, metadata)
            insert_time = time.time() - start_time
            
            # Search benchmark
            query_vector = vectors[0]
            start_time = time.time()
            for _ in range(100):
                client.search(query_vector, k=10)
            search_time = time.time() - start_time
            
            # Get stats
            stats = client.get_stats()
            
            results[name] = {
                "insert_time": insert_time,
                "insert_rate": len(vectors) / insert_time,
                "search_time": search_time / 100,  # Average per search
                "search_rate": 100 / search_time,
                "memory_usage_mb": stats["storage"].get("memory_usage_mb", 0),
                "disk_usage_mb": stats["storage"].get("disk_usage_mb", 0),
                "vector_count": stats["index"]["vector_count"]
            }
            
            client.close()
        
        return results
    
    def benchmark_index_types(self, vectors: np.ndarray, metadata: List[Dict]) -> Dict[str, Any]:
        """Benchmark different index types."""
        print("Benchmarking index types...")
        
        results = {}
        indices = {
            "flat": FlatIndex({"metric": "cosine"}),
            "hnsw": HNSWIndex({"metric": "cosine", "m": 16, "ef_construction": 200}),
            "ivf": IVFIndex({"metric": "cosine", "n_clusters": 50, "n_probes": 10})
        }
        
        storage = MemoryStorage({})
        
        for name, index in indices.items():
            print(f"  Testing {name} index...")
            
            client = VectorDBClient(storage, index)
            client.initialize()
            
            # Insert benchmark
            start_time = time.time()
            inserted_ids = client.insert(vectors, metadata)
            insert_time = time.time() - start_time
            
            # Search benchmark
            query_vector = vectors[0]
            start_time = time.time()
            for _ in range(100):
                client.search(query_vector, k=10)
            search_time = time.time() - start_time
            
            # Get stats
            stats = client.get_stats()
            
            results[name] = {
                "insert_time": insert_time,
                "insert_rate": len(vectors) / insert_time,
                "search_time": search_time / 100,  # Average per search
                "search_rate": 100 / search_time,
                "vector_count": stats["index"]["vector_count"],
                "index_specific": {k: v for k, v in stats["index"].items() 
                                 if k not in ["index_type", "vector_count", "initialized"]}
            }
            
            client.close()
        
        return results
    
    def benchmark_quantization(self, vectors: np.ndarray, metadata: List[Dict]) -> Dict[str, Any]:
        """Benchmark quantization methods."""
        print("Benchmarking quantization...")
        
        results = {}
        storage = MemoryStorage({})
        index = FlatIndex({"metric": "cosine"})
        
        # No quantization baseline
        print("  Testing baseline (no quantization)...")
        client = VectorDBClient(storage, index)
        client.initialize()
        
        start_time = time.time()
        client.insert(vectors, metadata)
        insert_time = time.time() - start_time
        
        query_vector = vectors[0]
        start_time = time.time()
        for _ in range(100):
            client.search(query_vector, k=10)
        search_time = time.time() - start_time
        
        stats = client.get_stats()
        results["baseline"] = {
            "insert_time": insert_time,
            "search_time": search_time / 100,
            "memory_usage_mb": stats["storage"]["memory_usage_mb"],
            "compression_ratio": 1.0
        }
        
        client.close()
        
        # Scalar quantization
        quantizers = {
            "scalar_4bit": ScalarQuantizer({"bits": 4}),
            "scalar_8bit": ScalarQuantizer({"bits": 8}),
            "scalar_16bit": ScalarQuantizer({"bits": 16}),
            "product": ProductQuantizer({"m": 8, "k": 256})
        }
        
        for name, quantizer in quantizers.items():
            print(f"  Testing {name} quantization...")
            
            client = VectorDBClient(storage, index, quantizer)
            client.initialize()
            
            start_time = time.time()
            client.insert(vectors, metadata)
            insert_time = time.time() - start_time
            
            start_time = time.time()
            for _ in range(100):
                client.search(query_vector, k=10)
            search_time = time.time() - start_time
            
            stats = client.get_stats()
            results[name] = {
                "insert_time": insert_time,
                "search_time": search_time / 100,
                "memory_usage_mb": stats["storage"]["memory_usage_mb"],
                "compression_ratio": stats["quantization"]["compression_ratio"]
            }
            
            client.close()
        
        return results
    
    def benchmark_scalability(self, dimensions: List[int], vector_counts: List[int]) -> Dict[str, Any]:
        """Benchmark scalability with different dimensions and vector counts."""
        print("Benchmarking scalability...")
        
        results = {}
        
        # Test different dimensions
        print("  Testing different dimensions...")
        dim_results = {}
        for dim in dimensions:
            print(f"    Dimension {dim}...")
            vectors, metadata = self.generate_data(1000, dim)
            
            storage = MemoryStorage({})
            index = FlatIndex({"metric": "cosine"})
            client = VectorDBClient(storage, index)
            client.initialize()
            
            start_time = time.time()
            client.insert(vectors, metadata)
            insert_time = time.time() - start_time
            
            query_vector = vectors[0]
            start_time = time.time()
            for _ in range(50):
                client.search(query_vector, k=10)
            search_time = time.time() - start_time
            
            stats = client.get_stats()
            dim_results[dim] = {
                "insert_time": insert_time,
                "search_time": search_time / 50,
                "memory_usage_mb": stats["storage"]["memory_usage_mb"]
            }
            
            client.close()
        
        results["dimensions"] = dim_results
        
        # Test different vector counts
        print("  Testing different vector counts...")
        count_results = {}
        for count in vector_counts:
            print(f"    Vector count {count}...")
            vectors, metadata = self.generate_data(count, 128)
            
            storage = MemoryStorage({})
            index = FlatIndex({"metric": "cosine"})
            client = VectorDBClient(storage, index)
            client.initialize()
            
            start_time = time.time()
            client.insert(vectors, metadata)
            insert_time = time.time() - start_time
            
            query_vector = vectors[0]
            start_time = time.time()
            for _ in range(min(50, count)):
                client.search(query_vector, k=10)
            search_time = time.time() - start_time
            
            stats = client.get_stats()
            count_results[count] = {
                "insert_time": insert_time,
                "search_time": search_time / min(50, count),
                "memory_usage_mb": stats["storage"]["memory_usage_mb"]
            }
            
            client.close()
        
        results["vector_counts"] = count_results
        
        return results
    
    def run_benchmarks(self, config: Dict[str, Any]):
        """Run all benchmarks."""
        print("Starting Vectara benchmarks...")
        print("=" * 50)
        
        # Generate test data
        vectors, metadata = self.generate_data(
            config["num_vectors"], 
            config["dimension"]
        )
        
        print(f"Test data: {len(vectors)} vectors of dimension {vectors.shape[1]}")
        print()
        
        # Run benchmarks
        all_results = {
            "config": config,
            "timestamp": time.time(),
            "storage_backends": self.benchmark_storage_backends(vectors, metadata),
            "index_types": self.benchmark_index_types(vectors, metadata),
            "quantization": self.benchmark_quantization(vectors, metadata),
            "scalability": self.benchmark_scalability(
                config["dimensions"], 
                config["vector_counts"]
            )
        }
        
        # Save results
        results_file = self.output_dir / f"benchmark_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nBenchmark results saved to: {results_file}")
        
        # Print summary
        self.print_summary(all_results)
        
        return all_results
    
    def print_summary(self, results: Dict[str, Any]):
        """Print benchmark summary."""
        print("\nBenchmark Summary")
        print("=" * 50)
        
        # Storage backends
        print("\nStorage Backends:")
        for name, data in results["storage_backends"].items():
            print(f"  {name}: {data['insert_rate']:.1f} vectors/sec insert, "
                  f"{data['search_rate']:.1f} searches/sec")
        
        # Index types
        print("\nIndex Types:")
        for name, data in results["index_types"].items():
            print(f"  {name}: {data['insert_rate']:.1f} vectors/sec insert, "
                  f"{data['search_rate']:.1f} searches/sec")
        
        # Quantization
        print("\nQuantization:")
        for name, data in results["quantization"].items():
            print(f"  {name}: {data['compression_ratio']:.3f}x compression, "
                  f"{data['memory_usage_mb']:.1f} MB")


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description="Benchmark Vectara vector database")
    parser.add_argument("--vectors", type=int, default=1000, 
                       help="Number of vectors to test with")
    parser.add_argument("--dimension", type=int, default=128,
                       help="Dimension of test vectors")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                       help="Output directory for results")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick benchmarks with smaller datasets")
    
    args = parser.parse_args()
    
    # Configuration
    if args.quick:
        config = {
            "num_vectors": 100,
            "dimension": 64,
            "dimensions": [32, 64, 128],
            "vector_counts": [100, 500, 1000]
        }
    else:
        config = {
            "num_vectors": args.vectors,
            "dimension": args.dimension,
            "dimensions": [64, 128, 256, 512],
            "vector_counts": [100, 500, 1000, 5000, 10000]
        }
    
    # Run benchmarks
    suite = BenchmarkSuite(args.output_dir)
    suite.run_benchmarks(config)


if __name__ == "__main__":
    main()
