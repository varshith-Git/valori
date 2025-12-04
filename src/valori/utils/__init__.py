"""
Utility modules for the valori vector database.

This module provides various utility functions for similarity computation,
validation, logging, and other common operations.
"""

from .batch_manager import BatchManager, ProgressTracker, ResourceMonitor
from .debugging import (
    IndexInspector,
    PerformanceProfiler,
    QueryAnalyzer,
    VectorAnalyzer,
    debug_function,
)
from .helpers import (
    batch_search,
    compute_similarity_matrix,
    create_vectors_from_files,
    create_vectors_from_text,
    find_duplicates,
    get_recommended_index_config,
    load_vectors_from_file,
    memory_usage_decorator,
    normalize_vectors,
    save_vectors_to_file,
    timing_decorator,
    validate_index_performance,
)
from .logging import get_logger, setup_logging
from .similarity import cosine_similarity, dot_product, euclidean_distance
from .validation import validate_config, validate_vector, validate_vectors

__all__ = [
    # Similarity and validation
    "cosine_similarity",
    "euclidean_distance",
    "dot_product",
    "validate_vector",
    "validate_vectors",
    "validate_config",
    # Logging
    "get_logger",
    "setup_logging",
    # Helper functions
    "create_vectors_from_text",
    "create_vectors_from_files",
    "normalize_vectors",
    "compute_similarity_matrix",
    "find_duplicates",
    "batch_search",
    "save_vectors_to_file",
    "load_vectors_from_file",
    "timing_decorator",
    "memory_usage_decorator",
    "get_recommended_index_config",
    "validate_index_performance",
    # Batch management
    "BatchManager",
    "ProgressTracker",
    "ResourceMonitor",
    # Debugging and analysis
    "PerformanceProfiler",
    "VectorAnalyzer",
    "QueryAnalyzer",
    "debug_function",
    "IndexInspector",
]
