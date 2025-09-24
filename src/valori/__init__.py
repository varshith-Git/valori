"""
Valori - A high-performance vector database library for Python.

This package provides a comprehensive vector database implementation with support for
various storage backends, indexing algorithms, quantization methods, persistence
strategies, and document parsing capabilities.
"""

__version__ = "0.1.0"
__author__ = "Varshith"
__email__ = "team@valori.com"

from .base import VectorDB
from .client import VectorDBClient
from .exceptions import ValoriError, StorageError, IndexError, QuantizationError, ParsingError, ProcessingError

# Import factory functions for easy usage
from .factory import (
    create_vector_db,
    create_document_db,
    create_semantic_search_db,
    create_image_search_db,
    create_hybrid_search_db,
    create_from_template
)

__all__ = [
    "VectorDB",
    "VectorDBClient", 
    "ValoriError",
    "StorageError",
    "IndexError",
    "QuantizationError",
    "ParsingError",
    "ProcessingError",
    # Factory functions
    "create_vector_db",
    "create_document_db",
    "create_semantic_search_db",
    "create_image_search_db",
    "create_hybrid_search_db",
    "create_from_template",
]
