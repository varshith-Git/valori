"""
Quantization implementations for the Vectara vector database.

This module provides various quantization methods for compressing vectors
and reducing memory usage while maintaining search quality.
"""

from .base import Quantizer
from .scalar import ScalarQuantizer
from .product import ProductQuantizer

__all__ = [
    "Quantizer",
    "ScalarQuantizer",
    "ProductQuantizer",
]
