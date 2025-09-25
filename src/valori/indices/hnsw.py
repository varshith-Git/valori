"""
HNSW (Hierarchical Navigable Small World) index implementation for the Vectara vector database.
"""

import uuid
from typing import Any, Dict, List, Optional
import numpy as np

from .base import Index
from ..exceptions import ValoriIndexError


class HNSWIndex(Index):
    """
    HNSW index implementation for approximate nearest neighbor search.
    
    HNSW (Hierarchical Navigable Small World) is a graph-based algorithm
    that provides fast approximate search with high recall. It's suitable
    for large-scale vector collections.
    
    Note: This is a simplified implementation. For production use,
    consider using libraries like faiss or hnswlib.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize HNSW index."""
        super().__init__(config)
        self.m = config.get("m", 16)  # Number of bi-directional links
        self.ef_construction = config.get("ef_construction", 200)
        self.ef_search = config.get("ef_search", 50)
        self.metric = config.get("metric", "cosine")
        
        # Graph structure (simplified)
        self.levels: List[Dict[str, List]] = []  # Each level is a dict of node -> neighbors
        self.vectors: Dict[str, np.ndarray] = {}  # Vector data
        self.max_level = 0
        self._vector_count = 0
    
    def initialize(self, storage_backend) -> None:
        """Initialize the HNSW index with storage backend."""
        self.storage_backend = storage_backend
        self._initialized = True
        
        # Initialize level 0
        self.levels = [{}]
        
        # Load existing vectors if any
        try:
            existing_ids = self.storage_backend.list_vectors()
            for vector_id in existing_ids:
                result = self.storage_backend.retrieve_vector(vector_id)
                if result is not None:
                    vector, _ = result
                    self._insert_vector_internal(vector_id, vector)
        except Exception as e:
            raise ValoriIndexError(f"Failed to initialize HNSW index: {str(e)}")
    
    def insert(self, vectors: np.ndarray, metadata: Optional[List[Dict]] = None) -> List[str]:
        """Insert vectors into the HNSW index."""
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
                
                # Insert into HNSW graph
                self._insert_vector_internal(vector_id, vector)
                
                inserted_ids.append(vector_id)
            
            return inserted_ids
            
        except Exception as e:
            raise ValoriIndexError(f"Failed to insert vectors: {str(e)}")
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Dict]:
        """Search for similar vectors using HNSW graph traversal."""
        if not self._initialized:
            raise ValoriIndexError("Index not initialized")
        
        if query_vector.ndim != 1:
            raise ValoriIndexError("Query vector must be 1D")
        
        try:
            if not self.vectors:
                return []
            
            # Limit k to available vectors
            k = min(k, len(self.vectors))
            
            # Start from top level
            current_level = min(self.max_level, len(self.levels) - 1)
            candidates = []
            
            # Find entry point at highest level
            if current_level >= 0 and self.levels[current_level]:
                # Start from a random node at top level (simplified)
                entry_point = next(iter(self.levels[current_level].keys()))
                candidates = [(entry_point, self._compute_distance(query_vector, self.vectors[entry_point]))]
            
            # Traverse down levels
            while current_level > 0:
                new_candidates = []
                for node_id, _ in candidates:
                    if node_id in self.levels[current_level]:
                        for neighbor_id in self.levels[current_level][node_id]:
                            if neighbor_id in self.vectors:
                                distance = self._compute_distance(query_vector, self.vectors[neighbor_id])
                                new_candidates.append((neighbor_id, distance))
                
                # Keep best candidates
                new_candidates.sort(key=lambda x: x[1])
                candidates = new_candidates[:self.ef_search]
                current_level -= 1
            
            # Final search at level 0
            if current_level == 0 and candidates:
                final_candidates = []
                for node_id, _ in candidates:
                    if node_id in self.levels[0]:
                        for neighbor_id in self.levels[0][node_id]:
                            if neighbor_id in self.vectors:
                                distance = self._compute_distance(query_vector, self.vectors[neighbor_id])
                                final_candidates.append((neighbor_id, distance))
                
                candidates.extend(final_candidates)
            
            # Sort and return top k
            candidates.sort(key=lambda x: x[1])
            top_k = candidates[:k]
            
            results = []
            for node_id, distance in top_k:
                _, metadata = self.storage_backend.retrieve_vector(node_id)
                results.append({
                    "id": node_id,
                    "distance": distance,
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
                
                # Remove from graph
                self._remove_vector_internal(vector_id)
            
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
            
            # Update in graph
            if id in self.vectors:
                self.vectors[id] = vector.copy()
            
            return True
            
        except Exception as e:
            raise ValoriIndexError(f"Failed to update vector {id}: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get HNSW index statistics."""
        if not self._initialized:
            return {"status": "not_initialized"}
        
        return {
            "index_type": "hnsw",
            "metric": self.metric,
            "m": self.m,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "max_level": self.max_level,
            "vector_count": self._vector_count,
            "levels": len(self.levels),
            "initialized": self._initialized,
        }
    
    def close(self) -> None:
        """Close the HNSW index."""
        self.vectors.clear()
        self.levels.clear()
        self.max_level = 0
        self._vector_count = 0
        self._initialized = False
    
    def _insert_vector_internal(self, vector_id: str, vector: np.ndarray) -> None:
        """Internal method to insert vector into HNSW graph."""
        self.vectors[vector_id] = vector.copy()
        self._vector_count = len(self.vectors)
        
        # Simplified level assignment (random)
        import random
        level = 0
        while random.random() < 0.5 and level < 10:  # Max 10 levels
            level += 1
        
        # Ensure we have enough levels
        while len(self.levels) <= level:
            self.levels.append({})
        
        # Add to level 0 (all vectors are at level 0)
        if vector_id not in self.levels[0]:
            self.levels[0][vector_id] = []
        
        # Add to higher levels (simplified)
        for l in range(1, min(level + 1, len(self.levels))):
            if vector_id not in self.levels[l]:
                self.levels[l][vector_id] = []
        
        self.max_level = max(self.max_level, level)
        
        # Connect to neighbors (simplified - connect to a few random existing nodes)
        for l in range(min(level + 1, len(self.levels))):
            existing_nodes = list(self.levels[l].keys())
            if existing_nodes:
                # Connect to up to m neighbors
                num_connections = min(self.m, len(existing_nodes))
                neighbors = random.sample(existing_nodes, num_connections)
                
                for neighbor_id in neighbors:
                    if neighbor_id != vector_id:
                        # Bidirectional connection
                        if vector_id not in self.levels[l][neighbor_id]:
                            self.levels[l][neighbor_id].append(vector_id)
                        if neighbor_id not in self.levels[l][vector_id]:
                            self.levels[l][vector_id].append(neighbor_id)
    
    def _remove_vector_internal(self, vector_id: str) -> None:
        """Internal method to remove vector from HNSW graph."""
        if vector_id in self.vectors:
            del self.vectors[vector_id]
            self._vector_count = len(self.vectors)
        
        # Remove from all levels
        for level in self.levels:
            if vector_id in level:
                # Remove connections
                for neighbor_id in level[vector_id]:
                    if neighbor_id in level and vector_id in level[neighbor_id]:
                        level[neighbor_id].remove(vector_id)
                
                del level[vector_id]
    
    def _compute_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute distance between two vectors."""
        if self.metric == "cosine":
            # Cosine distance
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 1.0
            similarity = dot_product / (norm1 * norm2)
            return 1 - similarity
        elif self.metric == "euclidean":
            return np.linalg.norm(vec1 - vec2)
        else:
            raise ValoriIndexError(f"Unsupported metric: {self.metric}")
