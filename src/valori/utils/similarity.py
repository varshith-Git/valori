"""
Similarity computation utilities for the Vectara vector database.
"""

import numpy as np
from typing import Union


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity value between -1 and 1
    """
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Vectors must be 1D")
    
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")
    
    # Compute dot product
    dot_product = np.dot(a, b)
    
    # Compute norms
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    # Handle zero vectors
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    # Compute cosine similarity
    similarity = dot_product / (norm_a * norm_b)
    
    # Clamp to valid range to handle numerical precision issues
    return np.clip(similarity, -1.0, 1.0)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine distance between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine distance value between 0 and 2
    """
    similarity = cosine_similarity(a, b)
    return 1 - similarity


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Euclidean distance between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Euclidean distance value
    """
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Vectors must be 1D")
    
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")
    
    return np.linalg.norm(a - b)


def euclidean_distance_squared(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute squared Euclidean distance between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Squared Euclidean distance value
    """
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Vectors must be 1D")
    
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")
    
    diff = a - b
    return np.dot(diff, diff)


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute dot product between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Dot product value
    """
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Vectors must be 1D")
    
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")
    
    return np.dot(a, b)


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Manhattan distance between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Manhattan distance value
    """
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Vectors must be 1D")
    
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")
    
    return np.sum(np.abs(a - b))


def chebyshev_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Chebyshev distance between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Chebyshev distance value
    """
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Vectors must be 1D")
    
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")
    
    return np.max(np.abs(a - b))


def hamming_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Hamming distance between two binary vectors.
    
    Args:
        a: First binary vector
        b: Second binary vector
        
    Returns:
        Hamming distance value
    """
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Vectors must be 1D")
    
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")
    
    # Convert to binary if needed
    a_binary = (a > 0).astype(int)
    b_binary = (b > 0).astype(int)
    
    return np.sum(a_binary != b_binary)


def jaccard_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Jaccard similarity between two binary vectors.
    
    Args:
        a: First binary vector
        b: Second binary vector
        
    Returns:
        Jaccard similarity value between 0 and 1
    """
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Vectors must be 1D")
    
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")
    
    # Convert to binary if needed
    a_binary = (a > 0).astype(bool)
    b_binary = (b > 0).astype(bool)
    
    # Compute intersection and union
    intersection = np.sum(a_binary & b_binary)
    union = np.sum(a_binary | b_binary)
    
    # Handle case where both vectors are all zeros
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


def jaccard_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Jaccard distance between two binary vectors.
    
    Args:
        a: First binary vector
        b: Second binary vector
        
    Returns:
        Jaccard distance value between 0 and 1
    """
    similarity = jaccard_similarity(a, b)
    return 1 - similarity


def normalize_vector(vector: np.ndarray, norm: str = 'l2') -> np.ndarray:
    """
    Normalize a vector to unit length.
    
    Args:
        vector: Vector to normalize
        norm: Type of norm ('l1', 'l2', 'max')
        
    Returns:
        Normalized vector
    """
    if vector.ndim != 1:
        raise ValueError("Vector must be 1D")
    
    if norm == 'l2':
        norm_value = np.linalg.norm(vector)
    elif norm == 'l1':
        norm_value = np.sum(np.abs(vector))
    elif norm == 'max':
        norm_value = np.max(np.abs(vector))
    else:
        raise ValueError(f"Unsupported norm: {norm}")
    
    if norm_value == 0:
        return vector.copy()
    
    return vector / norm_value


def batch_cosine_similarity(vectors_a: np.ndarray, vectors_b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between batches of vectors.
    
    Args:
        vectors_a: First batch of vectors (n_vectors_a, n_dim)
        vectors_b: Second batch of vectors (n_vectors_b, n_dim)
        
    Returns:
        Similarity matrix (n_vectors_a, n_vectors_b)
    """
    if vectors_a.ndim != 2 or vectors_b.ndim != 2:
        raise ValueError("Input arrays must be 2D")
    
    if vectors_a.shape[1] != vectors_b.shape[1]:
        raise ValueError("Vectors must have the same dimension")
    
    # Compute dot products
    dot_products = np.dot(vectors_a, vectors_b.T)
    
    # Compute norms
    norms_a = np.linalg.norm(vectors_a, axis=1, keepdims=True)
    norms_b = np.linalg.norm(vectors_b, axis=1, keepdims=True)
    
    # Avoid division by zero
    norms_a = np.where(norms_a == 0, 1, norms_a)
    norms_b = np.where(norms_b == 0, 1, norms_b)
    
    # Compute similarities
    similarities = dot_products / (norms_a * norms_b.T)
    
    # Clamp to valid range
    return np.clip(similarities, -1.0, 1.0)
