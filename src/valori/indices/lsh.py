"""
LSH (Locality Sensitive Hashing) index for the Vectara vector database.

Implements Locality Sensitive Hashing for approximate nearest neighbor search
with configurable hash functions and parameters.
"""

import numpy as np
import hashlib
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict

from .base import Index
from ..exceptions import ValoriIndexError
from ..utils.validation import validate_vector


class LSHIndex(Index):
    """
    Locality Sensitive Hashing index implementation.
    
    LSH is particularly useful for high-dimensional data where exact search
    becomes computationally expensive. It provides approximate results with
    tunable accuracy/speed trade-offs.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize LSH index."""
        super().__init__(config)
        self.num_hash_tables = config.get("num_hash_tables", 10)
        self.hash_size = config.get("hash_size", 16)
        self.num_projections = config.get("num_projections", 64)
        self.random_seed = config.get("random_seed", 42)
        self.threshold = config.get("threshold", 0.5)
        
        # Validate configuration
        if self.num_hash_tables <= 0:
            raise ValoriIndexError("Number of hash tables must be positive")
        if self.hash_size <= 0:
            raise ValoriIndexError("Hash size must be positive")
        if self.num_projections <= 0:
            raise ValoriIndexError("Number of projections must be positive")
        
        # Initialize random state
        self.rng = np.random.RandomState(self.random_seed)
        
        # Storage structures
        self.hash_tables: List[Dict[str, List[int]]] = []
        self.projections: Optional[np.ndarray] = None
        self.offsets: Optional[np.ndarray] = None
        self.dimension: Optional[int] = None
        self.vectors: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []
        self._next_id = 0
    
    def initialize(self) -> None:
        """Initialize the LSH index."""
        # Initialize hash tables
        self.hash_tables = [defaultdict(list) for _ in range(self.num_hash_tables)]
        self._initialized = True
    
    def add(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> List[int]:
        """
        Add vectors to the LSH index.
        
        Args:
            vectors: Array of vectors to add
            metadata: List of metadata dictionaries
            
        Returns:
            List of assigned IDs
        """
        if not self._initialized:
            raise ValoriIndexError("Index not initialized")
        
        if len(vectors) != len(metadata):
            raise ValoriIndexError("Number of vectors must match number of metadata items")
        
        # Set dimension on first add
        if self.dimension is None:
            self.dimension = vectors.shape[1]
            self._generate_projections()
        
        if vectors.shape[1] != self.dimension:
            raise ValoriIndexError(f"Vector dimension mismatch: expected {self.dimension}, got {vectors.shape[1]}")
        
        # Validate vectors
        for vector in vectors:
            validate_vector(vector, self.dimension, self.dimension)
        
        assigned_ids = []
        
        for i, vector in enumerate(vectors):
            vector_id = self._next_id
            self._next_id += 1
            
            # Store vector and metadata
            self.vectors.append(vector.copy())
            self.metadata.append(metadata[i].copy())
            
            # Generate hash codes for all hash tables
            hash_codes = self._hash_vector(vector)
            
            # Add to each hash table
            for table_idx, hash_code in enumerate(hash_codes):
                self.hash_tables[table_idx][hash_code].append(vector_id)
            
            assigned_ids.append(vector_id)
        
        return assigned_ids
    
    def search(self, query_vector: np.ndarray, k: int = 10, 
               threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Search for nearest neighbors using LSH.
        
        Args:
            query_vector: Query vector
            k: Number of neighbors to return
            threshold: Similarity threshold (optional)
            
        Returns:
            List of search results with 'id', 'distance', and 'metadata'
        """
        if not self._initialized:
            raise ValoriIndexError("Index not initialized")
        
        if self.dimension is None:
            return []
        
        validate_vector(query_vector, self.dimension, self.dimension)
        
        if threshold is None:
            threshold = self.threshold
        
        # Generate hash codes for query
        query_hashes = self._hash_vector(query_vector)
        
        # Find candidate vectors from hash tables
        candidates = set()
        for table_idx, hash_code in enumerate(query_hashes):
            if hash_code in self.hash_tables[table_idx]:
                candidates.update(self.hash_tables[table_idx][hash_code])
        
        # If no candidates found, try with more relaxed matching
        if not candidates:
            candidates = self._relaxed_search(query_hashes)
        
        # Compute exact distances for candidates
        results = []
        for candidate_id in candidates:
            if candidate_id < len(self.vectors):
                vector = self.vectors[candidate_id]
                distance = self._compute_distance(query_vector, vector)
                
                # Apply threshold filter
                if distance >= threshold:
                    results.append({
                        "id": candidate_id,
                        "distance": distance,
                        "metadata": self.metadata[candidate_id]
                    })
        
        # Sort by distance and return top k
        results.sort(key=lambda x: x["distance"], reverse=True)
        return results[:k]
    
    def remove(self, ids: List[int]) -> None:
        """
        Remove vectors from the index.
        
        Args:
            ids: List of IDs to remove
        """
        if not self._initialized:
            raise ValoriIndexError("Index not initialized")
        
        for vector_id in ids:
            if vector_id < len(self.vectors):
                # Remove from hash tables
                vector = self.vectors[vector_id]
                hash_codes = self._hash_vector(vector)
                
                for table_idx, hash_code in enumerate(hash_codes):
                    if hash_code in self.hash_tables[table_idx]:
                        try:
                            self.hash_tables[table_idx][hash_code].remove(vector_id)
                        except ValueError:
                            pass  # Already removed
                
                # Mark as removed (keep indices consistent)
                self.vectors[vector_id] = None
                self.metadata[vector_id] = None
    
    def clear(self) -> None:
        """Clear all vectors from the index."""
        self.vectors.clear()
        self.metadata.clear()
        self.hash_tables = [defaultdict(list) for _ in range(self.num_hash_tables)]
        self._next_id = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        active_vectors = sum(1 for v in self.vectors if v is not None)
        
        # Calculate hash table statistics
        table_sizes = [len(table) for table in self.hash_tables]
        bucket_sizes = []
        for table in self.hash_tables:
            for bucket in table.values():
                bucket_sizes.extend(bucket)
        
        return {
            "index_type": "lsh",
            "dimension": self.dimension,
            "num_vectors": active_vectors,
            "total_capacity": len(self.vectors),
            "num_hash_tables": self.num_hash_tables,
            "hash_size": self.hash_size,
            "num_projections": self.num_projections,
            "avg_bucket_size": np.mean(bucket_sizes) if bucket_sizes else 0,
            "max_bucket_size": max(bucket_sizes) if bucket_sizes else 0,
            "hash_table_sizes": table_sizes,
            "threshold": self.threshold,
            "initialized": self._initialized,
        }
    
    def _generate_projections(self) -> None:
        """Generate random projection vectors for hashing."""
        # Generate random projection vectors
        self.projections = self.rng.normal(0, 1, (self.num_projections, self.dimension))
        
        # Generate random offsets for each projection
        self.offsets = self.rng.uniform(0, 1, self.num_projections)
    
    def _hash_vector(self, vector: np.ndarray) -> List[str]:
        """
        Generate hash codes for a vector across all hash tables.
        
        Args:
            vector: Input vector
            
        Returns:
            List of hash codes (one per hash table)
        """
        hash_codes = []
        
        for table_idx in range(self.num_hash_tables):
            # Select a subset of projections for this hash table
            start_idx = (table_idx * self.hash_size) % self.num_projections
            end_idx = start_idx + self.hash_size
            
            if end_idx > self.num_projections:
                # Wrap around if needed
                proj_indices = list(range(start_idx, self.num_projections))
                proj_indices.extend(range(0, end_idx - self.num_projections))
                selected_projections = self.projections[proj_indices]
            else:
                selected_projections = self.projections[start_idx:end_idx]
            
            # Compute dot products with selected projections
            dot_products = np.dot(selected_projections, vector)
            
            # Apply offsets and threshold to get binary hash
            hash_bits = (dot_products + self.offsets[:len(dot_products)]) >= 0
            
            # Convert to hash string
            hash_string = ''.join(['1' if bit else '0' for bit in hash_bits])
            hash_codes.append(hash_string)
        
        return hash_codes
    
    def _relaxed_search(self, query_hashes: List[str]) -> set:
        """
        Perform relaxed search when no exact matches are found.
        
        Args:
            query_hashes: Hash codes for the query vector
            
        Returns:
            Set of candidate vector IDs
        """
        candidates = set()
        
        # Try searching with Hamming distance tolerance
        for table_idx, query_hash in enumerate(query_hashes):
            for stored_hash, vector_ids in self.hash_tables[table_idx].items():
                # Calculate Hamming distance
                hamming_distance = sum(c1 != c2 for c1, c2 in zip(query_hash, stored_hash))
                
                # If Hamming distance is small, consider as candidate
                if hamming_distance <= 2:  # Allow up to 2 bit differences
                    candidates.update(vector_ids)
        
        return candidates
    
    def _compute_distance(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Compute distance between two vectors."""
        if self.metric == "cosine":
            # Cosine similarity
            dot_product = np.dot(vector1, vector2)
            norms = np.linalg.norm(vector1) * np.linalg.norm(vector2)
            if norms == 0:
                return 0.0
            return dot_product / norms
        elif self.metric == "euclidean":
            # Euclidean distance (converted to similarity)
            distance = np.linalg.norm(vector1 - vector2)
            return 1.0 / (1.0 + distance)
        elif self.metric == "manhattan":
            # Manhattan distance (converted to similarity)
            distance = np.sum(np.abs(vector1 - vector2))
            return 1.0 / (1.0 + distance)
        else:
            raise ValoriIndexError(f"Unsupported metric: {self.metric}")
    
    def close(self) -> None:
        """Close the index and clean up resources."""
        self.clear()
        self._initialized = False
