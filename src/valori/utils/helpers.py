"""
Helper functions and utilities for the Vectara vector database.

This module provides convenient helper functions for common operations,
making the vector database easier to use.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import json
import pickle
import time
from functools import wraps

from .validation import validate_vector
from ..exceptions import ValidationError


def create_vectors_from_text(
    texts: List[str], 
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Create vectors from text using a sentence transformer model.
    
    Args:
        texts: List of text strings to convert to vectors
        model_name: Name of the sentence transformer model
        batch_size: Batch size for processing
        
    Returns:
        Tuple of (vectors, metadata)
    """
    try:
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer(model_name)
        
        # Generate embeddings in batches
        vectors = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        
        # Create metadata
        metadata = [
            {
                "text": text,
                "text_length": len(text),
                "model_used": model_name,
                "embedding_dimension": vectors.shape[1]
            }
            for text in texts
        ]
        
        return vectors.astype(np.float32), metadata
        
    except ImportError:
        raise ValidationError("sentence-transformers not installed. Install with: pip install sentence-transformers")


def create_vectors_from_files(
    file_paths: List[Union[str, Path]],
    parser_name: Optional[str] = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Create vectors from files using document parsing and embedding.
    
    Args:
        file_paths: List of file paths to process
        parser_name: Specific parser to use (optional)
        embedding_model: Embedding model to use
        
    Returns:
        Tuple of (vectors, metadata)
    """
    from ..processors import ProcessingPipeline
    from ..parsers import ParserRegistry
    
    # Setup processing pipeline
    pipeline_config = {
        "parsers": {
            "text": {"chunk_size": 1000},
            "pdf": {"chunk_size": 1000},
            "office": {"chunk_size": 1000}
        },
        "processors": {
            "cleaning": {"normalize_whitespace": True},
            "embedding": {"model_name": embedding_model}
        }
    }
    
    pipeline = ProcessingPipeline(pipeline_config)
    pipeline.initialize()
    
    all_vectors = []
    all_metadata = []
    
    for file_path in file_paths:
        try:
            result = pipeline.process_document(file_path, parser_name)
            
            if "embedding" in result:
                all_vectors.append(result["embedding"])
                all_metadata.append(result["metadata"])
            
        except Exception as e:
            print(f"Warning: Failed to process {file_path}: {e}")
            continue
    
    pipeline.close()
    
    if not all_vectors:
        return np.array([]).reshape(0, -1), []
    
    return np.array(all_vectors), all_metadata


def normalize_vectors(vectors: np.ndarray, method: str = "l2") -> np.ndarray:
    """
    Normalize vectors using various methods.
    
    Args:
        vectors: Array of vectors to normalize
        method: Normalization method ('l2', 'l1', 'max', 'minmax')
        
    Returns:
        Normalized vectors
    """
    if method == "l2":
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / (norms + 1e-8)
    
    elif method == "l1":
        norms = np.sum(np.abs(vectors), axis=1, keepdims=True)
        return vectors / (norms + 1e-8)
    
    elif method == "max":
        max_vals = np.max(np.abs(vectors), axis=1, keepdims=True)
        return vectors / (max_vals + 1e-8)
    
    elif method == "minmax":
        min_vals = np.min(vectors, axis=1, keepdims=True)
        max_vals = np.max(vectors, axis=1, keepdims=True)
        return (vectors - min_vals) / (max_vals - min_vals + 1e-8)
    
    else:
        raise ValidationError(f"Unknown normalization method: {method}")


def compute_similarity_matrix(vectors: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """
    Compute pairwise similarity matrix for vectors.
    
    Args:
        vectors: Array of vectors
        metric: Similarity metric ('cosine', 'euclidean', 'dot')
        
    Returns:
        Similarity matrix
    """
    if metric == "cosine":
        # Normalize vectors
        normalized = normalize_vectors(vectors, "l2")
        return np.dot(normalized, normalized.T)
    
    elif metric == "euclidean":
        # Convert to similarity (1 / (1 + distance))
        distances = np.linalg.norm(vectors[:, np.newaxis] - vectors[np.newaxis, :], axis=2)
        return 1.0 / (1.0 + distances)
    
    elif metric == "dot":
        return np.dot(vectors, vectors.T)
    
    else:
        raise ValidationError(f"Unknown similarity metric: {metric}")


def find_duplicates(
    vectors: np.ndarray, 
    threshold: float = 0.99,
    metric: str = "cosine"
) -> List[List[int]]:
    """
    Find duplicate or near-duplicate vectors.
    
    Args:
        vectors: Array of vectors
        threshold: Similarity threshold for considering duplicates
        metric: Similarity metric to use
        
    Returns:
        List of groups of duplicate indices
    """
    similarity_matrix = compute_similarity_matrix(vectors, metric)
    
    # Find pairs above threshold
    n = len(vectors)
    duplicate_groups = []
    processed = set()
    
    for i in range(n):
        if i in processed:
            continue
        
        group = [i]
        processed.add(i)
        
        for j in range(i + 1, n):
            if j in processed:
                continue
            
            if similarity_matrix[i, j] >= threshold:
                group.append(j)
                processed.add(j)
        
        if len(group) > 1:
            duplicate_groups.append(group)
    
    return duplicate_groups


def batch_search(
    index, 
    query_vectors: np.ndarray, 
    k: int = 10,
    batch_size: int = 100
) -> List[List[Dict[str, Any]]]:
    """
    Perform batch search on multiple query vectors.
    
    Args:
        index: Vector index instance
        query_vectors: Array of query vectors
        k: Number of neighbors to return per query
        batch_size: Batch size for processing
        
    Returns:
        List of search results for each query
    """
    all_results = []
    
    for i in range(0, len(query_vectors), batch_size):
        batch = query_vectors[i:i + batch_size]
        
        for query_vector in batch:
            results = index.search(query_vector, k=k)
            all_results.append(results)
    
    return all_results


def save_vectors_to_file(
    vectors: np.ndarray, 
    metadata: List[Dict[str, Any]], 
    filepath: Union[str, Path],
    format: str = "npz"
) -> None:
    """
    Save vectors and metadata to file.
    
    Args:
        vectors: Array of vectors
        metadata: List of metadata dictionaries
        filepath: Path to save file
        format: File format ('npz', 'pickle', 'json')
    """
    filepath = Path(filepath)
    
    if format == "npz":
        np.savez_compressed(
            filepath,
            vectors=vectors,
            metadata=metadata
        )
    
    elif format == "pickle":
        with open(filepath, 'wb') as f:
            pickle.dump({"vectors": vectors, "metadata": metadata}, f)
    
    elif format == "json":
        # Convert numpy arrays to lists for JSON serialization
        data = {
            "vectors": vectors.tolist(),
            "metadata": metadata,
            "shape": vectors.shape
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    else:
        raise ValidationError(f"Unknown format: {format}")


def load_vectors_from_file(filepath: Union[str, Path]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Load vectors and metadata from file.
    
    Args:
        filepath: Path to load file from
        
    Returns:
        Tuple of (vectors, metadata)
    """
    filepath = Path(filepath)
    
    if filepath.suffix == ".npz":
        data = np.load(filepath, allow_pickle=True)
        vectors = data["vectors"]
        metadata = data["metadata"].tolist()
    
    elif filepath.suffix == ".pkl":
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            vectors = data["vectors"]
            metadata = data["metadata"]
    
    elif filepath.suffix == ".json":
        with open(filepath, 'r') as f:
            data = json.load(f)
            vectors = np.array(data["vectors"])
            metadata = data["metadata"]
    
    else:
        raise ValidationError(f"Unknown file format: {filepath.suffix}")
    
    return vectors, metadata


def timing_decorator(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper


def memory_usage_decorator(func):
    """Decorator to monitor memory usage."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import psutil
            process = psutil.Process()
            
            # Get memory before
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            result = func(*args, **kwargs)
            
            # Get memory after
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            
            print(f"{func.__name__} memory usage: {mem_after - mem_before:.2f} MB")
            return result
            
        except ImportError:
            print("psutil not available for memory monitoring")
            return func(*args, **kwargs)
    
    return wrapper


def get_recommended_index_config(
    num_vectors: int, 
    dimension: int, 
    use_case: str = "balanced"
) -> Dict[str, Any]:
    """
    Get recommended index configuration based on dataset characteristics.
    
    Args:
        num_vectors: Number of vectors in dataset
        dimension: Vector dimension
        use_case: Use case ('speed', 'accuracy', 'memory', 'balanced')
        
    Returns:
        Recommended index configuration
    """
    if num_vectors < 1000:
        # Small dataset - use exact search
        return {
            "type": "flat",
            "config": {"metric": "cosine"}
        }
    
    elif num_vectors < 100000:
        # Medium dataset - use HNSW or Annoy
        if use_case == "speed":
            return {
                "type": "annoy",
                "config": {
                    "metric": "angular",
                    "num_trees": 10,
                    "search_k": -1
                }
            }
        elif use_case == "accuracy":
            return {
                "type": "hnsw",
                "config": {
                    "metric": "cosine",
                    "M": 16,
                    "ef_construction": 200,
                    "ef_search": 50
                }
            }
        else:  # balanced
            return {
                "type": "annoy",
                "config": {
                    "metric": "angular",
                    "num_trees": 20,
                    "search_k": -1
                }
            }
    
    elif dimension > 1000:
        # High-dimensional data - use LSH
        return {
            "type": "lsh",
            "config": {
                "metric": "cosine",
                "num_hash_tables": 15,
                "hash_size": 20,
                "num_projections": 100,
                "threshold": 0.3
            }
        }
    
    else:
        # Large dataset - use IVF
        return {
            "type": "ivf",
            "config": {
                "metric": "cosine",
                "n_clusters": min(1000, num_vectors // 100),
                "n_probes": 10
            }
        }


def validate_index_performance(
    index, 
    test_vectors: np.ndarray,
    test_queries: np.ndarray,
    k: int = 10
) -> Dict[str, Any]:
    """
    Validate index performance with test data.
    
    Args:
        index: Vector index instance
        test_vectors: Test vectors to add
        test_queries: Test query vectors
        k: Number of neighbors to return
        
    Returns:
        Performance metrics
    """
    # Add vectors
    start_time = time.time()
    metadata = [{"id": i} for i in range(len(test_vectors))]
    ids = index.add(test_vectors, metadata)
    
    if hasattr(index, 'build'):
        index.build()
    
    add_time = time.time() - start_time
    
    # Search performance
    search_times = []
    for query in test_queries:
        start_time = time.time()
        results = index.search(query, k=k)
        search_times.append(time.time() - start_time)
    
    # Calculate metrics
    avg_search_time = np.mean(search_times)
    search_throughput = len(test_queries) / sum(search_times)
    
    return {
        "add_time": add_time,
        "add_throughput": len(test_vectors) / add_time,
        "avg_search_time": avg_search_time,
        "search_throughput": search_throughput,
        "index_stats": index.get_stats()
    }
