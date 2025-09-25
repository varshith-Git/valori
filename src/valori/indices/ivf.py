"""
IVF (Inverted File) index implementation for the Vectara vector database.
"""

import uuid
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.cluster import KMeans

from .base import Index
from ..exceptions import ValoriIndexError


class IVFIndex(Index):
    """
    IVF (Inverted File) index implementation for approximate nearest neighbor search.
    
    IVF partitions the vector space into clusters and maintains an inverted
    file structure for fast retrieval. It's suitable for large-scale collections
    with good balance between accuracy and speed.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize IVF index."""
        super().__init__(config)
        self.n_clusters = config.get("n_clusters", 100)
        self.n_probes = config.get("n_probes", 10)  # Number of clusters to search
        self.metric = config.get("metric", "cosine")
        
        # IVF structure
        self.centroids: Optional[np.ndarray] = None  # Cluster centroids
        self.clusters: Dict[int, List[str]] = {}  # Cluster ID -> vector IDs
        self.vector_ids: List[str] = []
        self.vectors_cache: Dict[str, np.ndarray] = {}
        self._vector_count = 0
        self._trained = False
    
    def initialize(self, storage_backend) -> None:
        """Initialize the IVF index with storage backend."""
        self.storage_backend = storage_backend
        self._initialized = True
        
        # Load existing vectors if any
        try:
            existing_ids = self.storage_backend.list_vectors()
            self.vector_ids = existing_ids
            self._vector_count = len(self.vector_ids)
            
            # If we have enough vectors, retrain the index
            if len(self.vector_ids) >= self.n_clusters:
                self._train_index()
        except Exception as e:
            raise ValoriIndexError(f"Failed to initialize IVF index: {str(e)}")
    
    def insert(self, vectors: np.ndarray, metadata: Optional[List[Dict]] = None) -> List[str]:
        """Insert vectors into the IVF index."""
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
                self.vectors_cache[vector_id] = vector.copy()
                inserted_ids.append(vector_id)
            
            self._vector_count = len(self.vector_ids)
            
            # Retrain if we have enough vectors
            if len(self.vector_ids) >= self.n_clusters and not self._trained:
                self._train_index()
            elif self._trained:
                # Add new vectors to appropriate clusters
                for vector_id in inserted_ids:
                    self._add_to_cluster(vector_id, self.vectors_cache[vector_id])
            
            return inserted_ids
            
        except Exception as e:
            raise ValoriIndexError(f"Failed to insert vectors: {str(e)}")
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Dict]:
        """Search for similar vectors using IVF index."""
        if not self._initialized:
            raise ValoriIndexError("Index not initialized")
        
        if query_vector.ndim != 1:
            raise ValoriIndexError("Query vector must be 1D")
        
        try:
            if not self.vector_ids:
                return []
            
            # If not trained, fall back to brute force
            if not self._trained:
                return self._brute_force_search(query_vector, k)
            
            # Find closest centroids
            centroid_distances = []
            for i, centroid in enumerate(self.centroids):
                distance = self._compute_distance(query_vector, centroid)
                centroid_distances.append((i, distance))
            
            # Sort by distance and take top n_probes
            centroid_distances.sort(key=lambda x: x[1])
            probe_clusters = [cluster_id for cluster_id, _ in centroid_distances[:self.n_probes]]
            
            # Search in selected clusters
            candidates = []
            for cluster_id in probe_clusters:
                if cluster_id in self.clusters:
                    for vector_id in self.clusters[cluster_id]:
                        if vector_id in self.vectors_cache:
                            vector = self.vectors_cache[vector_id]
                        else:
                            result = self.storage_backend.retrieve_vector(vector_id)
                            if result is None:
                                continue
                            vector, _ = result
                            self.vectors_cache[vector_id] = vector
                        
                        distance = self._compute_distance(query_vector, vector)
                        candidates.append((vector_id, distance))
            
            # Sort and return top k
            candidates.sort(key=lambda x: x[1])
            top_k = candidates[:k]
            
            results = []
            for vector_id, distance in top_k:
                _, metadata = self.storage_backend.retrieve_vector(vector_id)
                results.append({
                    "id": vector_id,
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
                
                # Remove from index
                if vector_id in self.vector_ids:
                    self.vector_ids.remove(vector_id)
                
                # Remove from cache
                self.vectors_cache.pop(vector_id, None)
                
                # Remove from clusters
                for cluster_id, vector_list in self.clusters.items():
                    if vector_id in vector_list:
                        vector_list.remove(vector_id)
            
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
            self.vectors_cache[id] = vector.copy()
            
            # If trained, update cluster assignment
            if self._trained:
                self._update_cluster_assignment(id, vector)
            
            return True
            
        except Exception as e:
            raise ValoriIndexError(f"Failed to update vector {id}: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get IVF index statistics."""
        if not self._initialized:
            return {"status": "not_initialized"}
        
        cluster_stats = {}
        for cluster_id, vector_list in self.clusters.items():
            cluster_stats[f"cluster_{cluster_id}"] = len(vector_list)
        
        return {
            "index_type": "ivf",
            "metric": self.metric,
            "n_clusters": self.n_clusters,
            "n_probes": self.n_probes,
            "vector_count": self._vector_count,
            "trained": self._trained,
            "cluster_stats": cluster_stats,
            "cache_size": len(self.vectors_cache),
            "initialized": self._initialized,
        }
    
    def close(self) -> None:
        """Close the IVF index."""
        self.centroids = None
        self.clusters.clear()
        self.vector_ids.clear()
        self.vectors_cache.clear()
        self._vector_count = 0
        self._trained = False
        self._initialized = False
    
    def _train_index(self) -> None:
        """Train the IVF index by clustering existing vectors."""
        if len(self.vector_ids) < self.n_clusters:
            return
        
        try:
            # Load all vectors
            vectors = []
            valid_ids = []
            
            for vector_id in self.vector_ids:
                if vector_id in self.vectors_cache:
                    vector = self.vectors_cache[vector_id]
                else:
                    result = self.storage_backend.retrieve_vector(vector_id)
                    if result is None:
                        continue
                    vector, _ = result
                    self.vectors_cache[vector_id] = vector
                
                vectors.append(vector)
                valid_ids.append(vector_id)
            
            if not vectors:
                return
            
            vectors_array = np.array(vectors)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(vectors_array)
            
            # Store centroids
            self.centroids = kmeans.cluster_centers_
            
            # Build clusters
            self.clusters.clear()
            for i, vector_id in enumerate(valid_ids):
                cluster_id = cluster_labels[i]
                if cluster_id not in self.clusters:
                    self.clusters[cluster_id] = []
                self.clusters[cluster_id].append(vector_id)
            
            self._trained = True
            
        except Exception as e:
            raise ValoriIndexError(f"Failed to train IVF index: {str(e)}")
    
    def _add_to_cluster(self, vector_id: str, vector: np.ndarray) -> None:
        """Add a vector to the appropriate cluster."""
        if not self._trained or self.centroids is None:
            return
        
        # Find closest centroid
        min_distance = float('inf')
        best_cluster = 0
        
        for i, centroid in enumerate(self.centroids):
            distance = self._compute_distance(vector, centroid)
            if distance < min_distance:
                min_distance = distance
                best_cluster = i
        
        # Add to cluster
        if best_cluster not in self.clusters:
            self.clusters[best_cluster] = []
        self.clusters[best_cluster].append(vector_id)
    
    def _update_cluster_assignment(self, vector_id: str, vector: np.ndarray) -> None:
        """Update cluster assignment for a vector."""
        if not self._trained or self.centroids is None:
            return
        
        # Remove from old cluster
        for cluster_id, vector_list in self.clusters.items():
            if vector_id in vector_list:
                vector_list.remove(vector_id)
                break
        
        # Add to new cluster
        self._add_to_cluster(vector_id, vector)
    
    def _brute_force_search(self, query_vector: np.ndarray, k: int) -> List[Dict]:
        """Fallback brute force search when index is not trained."""
        candidates = []
        
        for vector_id in self.vector_ids:
            if vector_id in self.vectors_cache:
                vector = self.vectors_cache[vector_id]
            else:
                result = self.storage_backend.retrieve_vector(vector_id)
                if result is None:
                    continue
                vector, _ = result
                self.vectors_cache[vector_id] = vector
            
            distance = self._compute_distance(query_vector, vector)
            candidates.append((vector_id, distance))
        
        # Sort and return top k
        candidates.sort(key=lambda x: x[1])
        top_k = candidates[:k]
        
        results = []
        for vector_id, distance in top_k:
            _, metadata = self.storage_backend.retrieve_vector(vector_id)
            results.append({
                "id": vector_id,
                "distance": distance,
                "metadata": metadata,
            })
        
        return results
    
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
