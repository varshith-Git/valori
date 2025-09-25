"""
Flat (brute force) index implementation for the Vectara vector database.
"""

import uuid
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from .base import Index
from ..exceptions import ValoriIndexError


class FlatIndex(Index):
    """
    Flat index implementation using brute force search.
    
    This index performs exhaustive search over all vectors, making it
    accurate but potentially slow for large datasets. It's suitable for
    small to medium-sized collections.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize flat index."""
        super().__init__(config)
        self.metric = config.get("metric", "cosine")  # cosine, euclidean
        self.vector_ids: List[str] = []
        self.vector_cache: Dict[str, np.ndarray] = {}
        self._vector_count = 0
    
    def initialize(self, storage_backend) -> None:
        """Initialize the flat index with storage backend."""
        self.storage_backend = storage_backend
        self._initialized = True
        
        # Load existing vectors if any
        try:
            existing_ids = self.storage_backend.list_vectors()
            self.vector_ids = existing_ids
            self._vector_count = len(self.vector_ids)
        except Exception as e:
            raise ValoriIndexError(f"Failed to initialize flat index: {str(e)}")
    
    def insert(self, vectors: np.ndarray, metadata: Optional[List[Dict]] = None) -> List[str]:
        """Insert vectors into the flat index."""
        if not self._initialized:
            raise ValoriIndexError("Index not initialized")
        
        if vectors.ndim != 2:
            raise ValoriIndexError("Vectors must be 2D array")
        
        try:
            inserted_ids = []
            
            for i, vector in enumerate(vectors):
                # Generate unique ID
                vector_id = str(uuid.uuid4())
                
                # Get metadata for this vector
                vector_metadata = metadata[i] if metadata and i < len(metadata) else None
                
                # Store in backend
                self.storage_backend.store_vector(vector_id, vector, vector_metadata)
                
                # Add to index
                self.vector_ids.append(vector_id)
                self.vector_cache[vector_id] = vector.copy()
                
                inserted_ids.append(vector_id)
            
            self._vector_count = len(self.vector_ids)
            return inserted_ids
            
        except Exception as e:
            raise ValoriIndexError(f"Failed to insert vectors: {str(e)}")
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Dict]:
        """Search for similar vectors using brute force."""
        if not self._initialized:
            raise ValoriIndexError("Index not initialized")
        
        if query_vector.ndim != 1:
            raise ValoriIndexError("Query vector must be 1D")
        
        try:
            if not self.vector_ids:
                return []
            
            # Limit k to available vectors
            k = min(k, len(self.vector_ids))
            
            # Load vectors from cache or storage
            vectors = []
            valid_ids = []
            
            for vector_id in self.vector_ids:
                if vector_id in self.vector_cache:
                    vector = self.vector_cache[vector_id]
                else:
                    result = self.storage_backend.retrieve_vector(vector_id)
                    if result is None:
                        continue
                    vector, _ = result
                    self.vector_cache[vector_id] = vector
                
                vectors.append(vector)
                valid_ids.append(vector_id)
            
            if not vectors:
                return []
            
            vectors_array = np.array(vectors)
            
            # Compute similarities/distances
            if self.metric == "cosine":
                similarities = cosine_similarity([query_vector], vectors_array)[0]
                # Convert to distances (1 - similarity for sorting)
                distances = 1 - similarities
            elif self.metric == "euclidean":
                distances = euclidean_distances([query_vector], vectors_array)[0]
            else:
                raise ValoriIndexError(f"Unsupported metric: {self.metric}")
            
            # Get top k results
            top_k_indices = np.argsort(distances)[:k]
            
            results = []
            for idx in top_k_indices:
                vector_id = valid_ids[idx]
                
                # Get metadata
                _, metadata = self.storage_backend.retrieve_vector(vector_id)
                
                results.append({
                    "id": vector_id,
                    "distance": float(distances[idx]),
                    "metadata": metadata,
                })
            
            return results
            
        except Exception as e:
            raise ValoriIndexError(f"Search failed: {str(e)}")
    
    def delete(self, ids: List[str]) -> bool:
        """Delete vectors by their IDs."""
        if not self._initialized:
            raise ValoriIndexError("Index not initialized")
        
        try:
            success = True
            
            for vector_id in ids:
                # Remove from storage
                if not self.storage_backend.delete_vector(vector_id):
                    success = False
                
                # Remove from index
                if vector_id in self.vector_ids:
                    self.vector_ids.remove(vector_id)
                
                # Remove from cache
                self.vector_cache.pop(vector_id, None)
            
            self._vector_count = len(self.vector_ids)
            return success
            
        except Exception as e:
            raise ValoriIndexError(f"Failed to delete vectors: {str(e)}")
    
    def update(self, id: str, vector: np.ndarray, metadata: Optional[Dict] = None) -> bool:
        """Update a vector by its ID."""
        if not self._initialized:
            raise ValoriIndexError("Index not initialized")
        
        try:
            # Update in storage
            if not self.storage_backend.update_vector(id, vector, metadata):
                return False
            
            # Update cache
            self.vector_cache[id] = vector.copy()
            
            return True
            
        except Exception as e:
            raise ValoriIndexError(f"Failed to update vector {id}: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get flat index statistics."""
        if not self._initialized:
            return {"status": "not_initialized"}
        
        return {
            "index_type": "flat",
            "metric": self.metric,
            "vector_count": self._vector_count,
            "cache_size": len(self.vector_cache),
            "initialized": self._initialized,
        }
    
    def close(self) -> None:
        """Close the flat index."""
        self.vector_cache.clear()
        self.vector_ids.clear()
        self._vector_count = 0
        self._initialized = False
