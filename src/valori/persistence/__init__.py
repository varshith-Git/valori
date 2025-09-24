"""
Persistence implementations for the Vectara vector database.

This module provides various persistence strategies for saving and loading
vector database state, including tensor-based and incremental persistence.
"""

from .base import PersistenceManager
from .tensor import TensorPersistence
from .incremental import IncrementalPersistence

__all__ = [
    "PersistenceManager",
    "TensorPersistence",
    "IncrementalPersistence",
]
