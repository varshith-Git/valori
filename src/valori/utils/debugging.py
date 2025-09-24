"""
Debugging and monitoring utilities for the Vectara vector database.

This module provides tools for debugging, monitoring, and analyzing
vector database performance and behavior.
"""

import time
import numpy as np
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path
import json
import logging
from contextlib import contextmanager
import functools
import inspect

from ..exceptions import ValidationError


class PerformanceProfiler:
    """Profile performance of vector database operations."""
    
    def __init__(self):
        self.measurements = {}
        self.current_measurements = {}
    
    @contextmanager
    def measure(self, operation_name: str):
        """Context manager to measure operation time."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            if operation_name not in self.measurements:
                self.measurements[operation_name] = []
            
            self.measurements[operation_name].append({
                "duration": duration,
                "memory_delta": memory_delta,
                "timestamp": end_time
            })
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        summary = {}
        
        for operation, measurements in self.measurements.items():
            durations = [m["duration"] for m in measurements]
            memory_deltas = [m["memory_delta"] for m in measurements]
            
            summary[operation] = {
                "count": len(measurements),
                "duration": {
                    "min": min(durations),
                    "max": max(durations),
                    "avg": sum(durations) / len(durations),
                    "total": sum(durations)
                },
                "memory": {
                    "min": min(memory_deltas),
                    "max": max(memory_deltas),
                    "avg": sum(memory_deltas) / len(memory_deltas),
                    "total": sum(memory_deltas)
                }
            }
        
        return summary
    
    def print_summary(self):
        """Print performance summary."""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("PERFORMANCE PROFILE SUMMARY")
        print("="*60)
        
        for operation, stats in summary.items():
            print(f"\n{operation}:")
            print(f"  Count: {stats['count']}")
            print(f"  Duration: {stats['duration']['avg']:.4f}s avg "
                  f"({stats['duration']['min']:.4f}s - {stats['duration']['max']:.4f}s)")
            print(f"  Memory: {stats['memory']['avg']:.2f} MB avg "
                  f"({stats['memory']['min']:.2f} MB - {stats['memory']['max']:.2f} MB)")
    
    def save_report(self, filepath: Union[str, Path]):
        """Save performance report to file."""
        report = {
            "summary": self.get_summary(),
            "raw_data": self.measurements
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)


class VectorAnalyzer:
    """Analyze vector data for insights and debugging."""
    
    def __init__(self):
        self.analysis_cache = {}
    
    def analyze_vectors(self, vectors: np.ndarray) -> Dict[str, Any]:
        """
        Analyze vector dataset for insights.
        
        Args:
            vectors: Array of vectors to analyze
            
        Returns:
            Analysis results
        """
        if vectors.size == 0:
            return {"error": "Empty vector array"}
        
        analysis = {
            "basic_stats": self._basic_statistics(vectors),
            "distribution": self._distribution_analysis(vectors),
            "similarity": self._similarity_analysis(vectors),
            "quality": self._quality_analysis(vectors)
        }
        
        return analysis
    
    def _basic_statistics(self, vectors: np.ndarray) -> Dict[str, Any]:
        """Compute basic statistics."""
        return {
            "count": len(vectors),
            "dimension": vectors.shape[1],
            "memory_size_mb": vectors.nbytes / 1024 / 1024,
            "mean": float(np.mean(vectors)),
            "std": float(np.std(vectors)),
            "min": float(np.min(vectors)),
            "max": float(np.max(vectors)),
            "sparsity": float(np.count_nonzero(vectors) / vectors.size)
        }
    
    def _distribution_analysis(self, vectors: np.ndarray) -> Dict[str, Any]:
        """Analyze vector distribution."""
        # Per-dimension statistics
        dim_means = np.mean(vectors, axis=0)
        dim_stds = np.std(vectors, axis=0)
        
        # Vector norms
        norms = np.linalg.norm(vectors, axis=1)
        
        return {
            "dimension_stats": {
                "mean_of_means": float(np.mean(dim_means)),
                "std_of_means": float(np.std(dim_means)),
                "mean_of_stds": float(np.mean(dim_stds)),
                "std_of_stds": float(np.std(dim_stds))
            },
            "norms": {
                "mean": float(np.mean(norms)),
                "std": float(np.std(norms)),
                "min": float(np.min(norms)),
                "max": float(np.max(norms))
            }
        }
    
    def _similarity_analysis(self, vectors: np.ndarray, sample_size: int = 1000) -> Dict[str, Any]:
        """Analyze similarity patterns."""
        if len(vectors) > sample_size:
            # Sample vectors for analysis
            indices = np.random.choice(len(vectors), sample_size, replace=False)
            sample_vectors = vectors[indices]
        else:
            sample_vectors = vectors
        
        # Normalize vectors
        normalized = sample_vectors / np.linalg.norm(sample_vectors, axis=1, keepdims=True)
        
        # Compute pairwise similarities
        similarity_matrix = np.dot(normalized, normalized.T)
        
        # Remove diagonal (self-similarities)
        mask = np.ones_like(similarity_matrix, dtype=bool)
        np.fill_diagonal(mask, False)
        similarities = similarity_matrix[mask]
        
        return {
            "similarity_stats": {
                "mean": float(np.mean(similarities)),
                "std": float(np.std(similarities)),
                "min": float(np.min(similarities)),
                "max": float(np.max(similarities))
            },
            "high_similarity_pairs": int(np.sum(similarities > 0.95)),
            "low_similarity_pairs": int(np.sum(similarities < -0.5))
        }
    
    def _quality_analysis(self, vectors: np.ndarray) -> Dict[str, Any]:
        """Analyze vector quality."""
        # Check for NaN or infinite values
        nan_count = np.isnan(vectors).sum()
        inf_count = np.isinf(vectors).sum()
        
        # Check for constant vectors
        constant_vectors = 0
        for vector in vectors:
            if np.all(vector == vector[0]):
                constant_vectors += 1
        
        # Check for zero vectors
        zero_vectors = np.sum(np.all(vectors == 0, axis=1))
        
        return {
            "nan_values": int(nan_count),
            "inf_values": int(inf_count),
            "constant_vectors": constant_vectors,
            "zero_vectors": int(zero_vectors),
            "quality_score": self._compute_quality_score(vectors)
        }
    
    def _compute_quality_score(self, vectors: np.ndarray) -> float:
        """Compute overall quality score (0-1)."""
        score = 1.0
        
        # Penalize for NaN/inf values
        nan_ratio = np.isnan(vectors).sum() / vectors.size
        inf_ratio = np.isinf(vectors).sum() / vectors.size
        score -= (nan_ratio + inf_ratio) * 0.5
        
        # Penalize for constant vectors
        constant_ratio = sum(1 for v in vectors if np.all(v == v[0])) / len(vectors)
        score -= constant_ratio * 0.3
        
        # Penalize for zero vectors
        zero_ratio = np.sum(np.all(vectors == 0, axis=1)) / len(vectors)
        score -= zero_ratio * 0.2
        
        return max(0.0, score)
    
    def print_analysis(self, vectors: np.ndarray):
        """Print vector analysis."""
        analysis = self.analyze_vectors(vectors)
        
        print("\n" + "="*60)
        print("VECTOR ANALYSIS")
        print("="*60)
        
        # Basic stats
        basic = analysis["basic_stats"]
        print(f"\nBasic Statistics:")
        print(f"  Count: {basic['count']:,}")
        print(f"  Dimension: {basic['dimension']}")
        print(f"  Memory: {basic['memory_size_mb']:.2f} MB")
        print(f"  Range: [{basic['min']:.4f}, {basic['max']:.4f}]")
        print(f"  Mean: {basic['mean']:.4f} ± {basic['std']:.4f}")
        print(f"  Sparsity: {basic['sparsity']:.2%}")
        
        # Distribution
        dist = analysis["distribution"]
        print(f"\nDistribution:")
        print(f"  Norms: {dist['norms']['mean']:.4f} ± {dist['norms']['std']:.4f}")
        print(f"  Dimension consistency: {dist['dimension_stats']['std_of_stds']:.4f}")
        
        # Quality
        quality = analysis["quality"]
        print(f"\nQuality:")
        print(f"  Quality Score: {quality['quality_score']:.2f}")
        print(f"  NaN values: {quality['nan_values']:,}")
        print(f"  Inf values: {quality['inf_values']:,}")
        print(f"  Constant vectors: {quality['constant_vectors']:,}")
        print(f"  Zero vectors: {quality['zero_vectors']:,}")


class QueryAnalyzer:
    """Analyze query performance and results."""
    
    def __init__(self):
        self.query_history = []
    
    def analyze_query(self, query_vector: np.ndarray, results: List[Dict[str, Any]], 
                     k: int, search_time: float) -> Dict[str, Any]:
        """Analyze a single query."""
        analysis = {
            "query_stats": {
                "vector_norm": float(np.linalg.norm(query_vector)),
                "vector_mean": float(np.mean(query_vector)),
                "vector_std": float(np.std(query_vector))
            },
            "result_stats": {
                "count": len(results),
                "expected_count": k,
                "search_time": search_time
            }
        }
        
        if results:
            distances = [r["distance"] for r in results]
            analysis["result_stats"].update({
                "distance_mean": float(np.mean(distances)),
                "distance_std": float(np.std(distances)),
                "distance_min": float(min(distances)),
                "distance_max": float(max(distances))
            })
        
        # Store in history
        self.query_history.append(analysis)
        
        return analysis
    
    def analyze_query_patterns(self) -> Dict[str, Any]:
        """Analyze patterns across multiple queries."""
        if not self.query_history:
            return {"error": "No query history"}
        
        # Aggregate statistics
        search_times = [q["result_stats"]["search_time"] for q in self.query_history]
        result_counts = [q["result_stats"]["count"] for q in self.query_history]
        
        return {
            "total_queries": len(self.query_history),
            "search_time": {
                "mean": float(np.mean(search_times)),
                "std": float(np.std(search_times)),
                "min": float(min(search_times)),
                "max": float(max(search_times))
            },
            "result_count": {
                "mean": float(np.mean(result_counts)),
                "std": float(np.std(result_counts)),
                "min": int(min(result_counts)),
                "max": int(max(result_counts))
            }
        }


def debug_function(func):
    """Decorator to add debugging information to functions."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get function info
        func_name = func.__name__
        module = func.__module__
        
        # Log function call
        print(f"\n🔍 DEBUG: Calling {module}.{func_name}")
        print(f"   Args: {len(args)} positional, {len(kwargs)} keyword")
        
        # Time the function
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            print(f"✅ SUCCESS: {func_name} completed in {duration:.4f}s")
            
            # Analyze result if it's a vector array
            if isinstance(result, np.ndarray):
                print(f"   Result: {result.shape} array, dtype={result.dtype}")
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ ERROR: {func_name} failed after {duration:.4f}s")
            print(f"   Error: {str(e)}")
            raise
    
    return wrapper


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging for vector database operations."""
    log_level = getattr(logging, level.upper())
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    logger = logging.getLogger('valori')
    logger.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class IndexInspector:
    """Inspect and debug index internals."""
    
    def __init__(self, index):
        self.index = index
    
    def inspect_index(self) -> Dict[str, Any]:
        """Get detailed index information."""
        info = {
            "type": type(self.index).__name__,
            "stats": self.index.get_stats(),
            "configuration": getattr(self.index, 'config', {}),
            "initialized": self.index.is_initialized()
        }
        
        # Index-specific inspection
        if hasattr(self.index, 'vectors'):
            info["vector_info"] = self._inspect_vectors()
        
        if hasattr(self.index, 'hash_tables'):
            info["hash_info"] = self._inspect_hash_tables()
        
        return info
    
    def _inspect_vectors(self) -> Dict[str, Any]:
        """Inspect stored vectors."""
        vectors = getattr(self.index, 'vectors', [])
        if not vectors:
            return {"count": 0}
        
        # Sample analysis
        sample_size = min(100, len(vectors))
        sample_vectors = vectors[:sample_size]
        
        if isinstance(sample_vectors[0], np.ndarray):
            sample_array = np.array(sample_vectors)
            return {
                "count": len(vectors),
                "dimension": sample_array.shape[1] if sample_array.size > 0 else 0,
                "sample_stats": {
                    "mean": float(np.mean(sample_array)),
                    "std": float(np.std(sample_array))
                }
            }
        
        return {"count": len(vectors)}
    
    def _inspect_hash_tables(self) -> Dict[str, Any]:
        """Inspect LSH hash tables."""
        hash_tables = getattr(self.index, 'hash_tables', [])
        if not hash_tables:
            return {"count": 0}
        
        total_buckets = sum(len(table) for table in hash_tables)
        total_entries = sum(
            sum(len(bucket) for bucket in table.values())
            for table in hash_tables
        )
        
        return {
            "table_count": len(hash_tables),
            "total_buckets": total_buckets,
            "total_entries": total_entries,
            "avg_entries_per_bucket": total_entries / max(total_buckets, 1)
        }
    
    def print_inspection(self):
        """Print index inspection results."""
        info = self.inspect_index()
        
        print("\n" + "="*60)
        print("INDEX INSPECTION")
        print("="*60)
        
        print(f"\nIndex Type: {info['type']}")
        print(f"Initialized: {info['initialized']}")
        
        # Print stats
        stats = info["stats"]
        print(f"\nStatistics:")
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value}")
        
        # Print configuration
        config = info.get("configuration", {})
        if config:
            print(f"\nConfiguration:")
            for key, value in config.items():
                print(f"  {key}: {value}")
        
        # Print vector info
        vector_info = info.get("vector_info", {})
        if vector_info:
            print(f"\nVector Information:")
            for key, value in vector_info.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        print(f"    {sub_key}: {sub_value}")
                else:
                    print(f"  {key}: {value}")
        
        # Print hash info
        hash_info = info.get("hash_info", {})
        if hash_info:
            print(f"\nHash Table Information:")
            for key, value in hash_info.items():
                print(f"  {key}: {value}")
