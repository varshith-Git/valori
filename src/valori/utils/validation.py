"""
Validation utilities for the Vectara vector database.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union
from ..exceptions import ValidationError


def validate_vector(vector: np.ndarray, 
                   min_dim: Optional[int] = None,
                   max_dim: Optional[int] = None,
                   dtype: Optional[np.dtype] = None,
                   allow_zero: bool = True) -> bool:
    """
    Validate a single vector.
    
    Args:
        vector: Vector to validate
        min_dim: Minimum dimension
        max_dim: Maximum dimension
        dtype: Expected data type
        allow_zero: Whether to allow zero vectors
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If vector is invalid
    """
    if not isinstance(vector, np.ndarray):
        raise ValidationError("Vector must be a numpy array")
    
    if vector.ndim != 1:
        raise ValidationError("Vector must be 1D")
    
    if len(vector) == 0:
        raise ValidationError("Vector cannot be empty")
    
    if min_dim is not None and len(vector) < min_dim:
        raise ValidationError(f"Vector dimension {len(vector)} is less than minimum {min_dim}")
    
    if max_dim is not None and len(vector) > max_dim:
        raise ValidationError(f"Vector dimension {len(vector)} is greater than maximum {max_dim}")
    
    if dtype is not None and vector.dtype != dtype:
        raise ValidationError(f"Vector dtype {vector.dtype} does not match expected {dtype}")
    
    if not allow_zero and np.all(vector == 0):
        raise ValidationError("Zero vector not allowed")
    
    # Check for NaN or infinite values
    if np.any(np.isnan(vector)) or np.any(np.isinf(vector)):
        raise ValidationError("Vector contains NaN or infinite values")
    
    return True


def validate_vectors(vectors: np.ndarray,
                    min_dim: Optional[int] = None,
                    max_dim: Optional[int] = None,
                    dtype: Optional[np.dtype] = None,
                    allow_zero: bool = True,
                    consistent_dims: bool = True) -> bool:
    """
    Validate a batch of vectors.
    
    Args:
        vectors: Batch of vectors to validate
        min_dim: Minimum dimension
        max_dim: Maximum dimension
        dtype: Expected data type
        allow_zero: Whether to allow zero vectors
        consistent_dims: Whether all vectors must have same dimension
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If vectors are invalid
    """
    if not isinstance(vectors, np.ndarray):
        raise ValidationError("Vectors must be a numpy array")
    
    if vectors.ndim != 2:
        raise ValidationError("Vectors must be 2D array")
    
    if vectors.shape[0] == 0:
        raise ValidationError("Vector batch cannot be empty")
    
    if vectors.shape[1] == 0:
        raise ValidationError("Vector dimension cannot be zero")
    
    if min_dim is not None and vectors.shape[1] < min_dim:
        raise ValidationError(f"Vector dimension {vectors.shape[1]} is less than minimum {min_dim}")
    
    if max_dim is not None and vectors.shape[1] > max_dim:
        raise ValidationError(f"Vector dimension {vectors.shape[1]} is greater than maximum {max_dim}")
    
    if dtype is not None and vectors.dtype != dtype:
        raise ValidationError(f"Vectors dtype {vectors.dtype} does not match expected {dtype}")
    
    if not allow_zero:
        zero_vectors = np.all(vectors == 0, axis=1)
        if np.any(zero_vectors):
            raise ValidationError("Zero vectors not allowed")
    
    # Check for NaN or infinite values
    if np.any(np.isnan(vectors)) or np.any(np.isinf(vectors)):
        raise ValidationError("Vectors contain NaN or infinite values")
    
    return True


def validate_config(config: Dict[str, Any], 
                   required_keys: Optional[List[str]] = None,
                   allowed_keys: Optional[List[str]] = None,
                   value_types: Optional[Dict[str, type]] = None,
                   value_ranges: Optional[Dict[str, tuple]] = None) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        required_keys: List of required keys
        allowed_keys: List of allowed keys (if None, all keys allowed)
        value_types: Dictionary mapping keys to expected types
        value_ranges: Dictionary mapping keys to (min, max) ranges
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If config is invalid
    """
    if not isinstance(config, dict):
        raise ValidationError("Config must be a dictionary")
    
    # Check required keys
    if required_keys:
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValidationError(f"Missing required keys: {missing_keys}")
    
    # Check allowed keys
    if allowed_keys is not None:
        invalid_keys = [key for key in config.keys() if key not in allowed_keys]
        if invalid_keys:
            raise ValidationError(f"Invalid keys: {invalid_keys}")
    
    # Check value types
    if value_types:
        for key, expected_type in value_types.items():
            if key in config:
                if not isinstance(config[key], expected_type):
                    raise ValidationError(f"Key '{key}' must be of type {expected_type}, got {type(config[key])}")
    
    # Check value ranges
    if value_ranges:
        for key, (min_val, max_val) in value_ranges.items():
            if key in config:
                value = config[key]
                if isinstance(value, (int, float)):
                    if value < min_val or value > max_val:
                        raise ValidationError(f"Key '{key}' value {value} is outside range [{min_val}, {max_val}]")
    
    return True


def validate_metric(metric: str) -> bool:
    """
    Validate similarity metric.
    
    Args:
        metric: Metric name to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If metric is invalid
    """
    valid_metrics = ['cosine', 'euclidean', 'dot_product', 'manhattan', 'chebyshev', 'hamming', 'jaccard']
    
    if not isinstance(metric, str):
        raise ValidationError("Metric must be a string")
    
    if metric not in valid_metrics:
        raise ValidationError(f"Invalid metric '{metric}'. Valid metrics: {valid_metrics}")
    
    return True


def validate_k(k: int, max_k: Optional[int] = None) -> bool:
    """
    Validate k parameter for search operations.
    
    Args:
        k: k value to validate
        max_k: Maximum allowed k value
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If k is invalid
    """
    if not isinstance(k, int):
        raise ValidationError("k must be an integer")
    
    if k <= 0:
        raise ValidationError("k must be positive")
    
    if max_k is not None and k > max_k:
        raise ValidationError(f"k ({k}) cannot be greater than max_k ({max_k})")
    
    return True


def validate_id(id: Union[str, int]) -> bool:
    """
    Validate vector ID.
    
    Args:
        id: ID to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If ID is invalid
    """
    if not isinstance(id, (str, int)):
        raise ValidationError("ID must be a string or integer")
    
    if isinstance(id, str) and len(id.strip()) == 0:
        raise ValidationError("ID cannot be empty")
    
    return True


def validate_ids(ids: List[Union[str, int]]) -> bool:
    """
    Validate list of vector IDs.
    
    Args:
        ids: List of IDs to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If IDs are invalid
    """
    if not isinstance(ids, list):
        raise ValidationError("IDs must be a list")
    
    if len(ids) == 0:
        raise ValidationError("IDs list cannot be empty")
    
    for i, id in enumerate(ids):
        try:
            validate_id(id)
        except ValidationError as e:
            raise ValidationError(f"Invalid ID at index {i}: {str(e)}")
    
    # Check for duplicates
    if len(ids) != len(set(ids)):
        raise ValidationError("IDs list contains duplicates")
    
    return True


def validate_metadata(metadata: Optional[Dict[str, Any]]) -> bool:
    """
    Validate metadata dictionary.
    
    Args:
        metadata: Metadata dictionary to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If metadata is invalid
    """
    if metadata is None:
        return True
    
    if not isinstance(metadata, dict):
        raise ValidationError("Metadata must be a dictionary")
    
    # Check for serializable values (basic check)
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise ValidationError(f"Metadata key '{key}' must be a string")
        
        # Check if value is JSON serializable
        try:
            import json
            json.dumps(value)
        except (TypeError, ValueError):
            raise ValidationError(f"Metadata value for key '{key}' is not JSON serializable")
    
    return True
