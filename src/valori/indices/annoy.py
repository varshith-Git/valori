"""
Annoy index for the Vectara vector database.

Implements Annoy (Approximate Nearest Neighbors Oh Yeah) for fast
approximate nearest neighbor search using random projection trees.
"""

import os
import tempfile
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..exceptions import ValoriIndexError
from ..utils.validation import validate_vector
from .base import Index


class AnnoyIndex(Index):
    """
    Annoy (Approximate Nearest Neighbors Oh Yeah) index implementation.

    Annoy is particularly efficient for high-dimensional data and provides
    good balance between search speed and accuracy. It builds multiple
    random projection trees for robust approximate search.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize Annoy index."""
        super().__init__(config)
        self.num_trees = config.get("num_trees", 10)
        self.metric = config.get(
            "metric", "angular"
        )  # angular, euclidean, manhattan, hamming, dot
        self.search_k = config.get("search_k", -1)  # -1 means use num_trees * n
        self.build_on_disk = config.get("build_on_disk", False)
        self.random_seed = config.get("random_seed", 42)

        # Validate configuration
        if self.num_trees <= 0:
            raise ValoriIndexError("Number of trees must be positive")

        if self.metric not in ["angular", "euclidean", "manhattan", "hamming", "dot"]:
            raise ValoriIndexError(f"Unsupported metric: {self.metric}")

        # Storage structures
        self.vectors: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []
        self.dimension: Optional[int] = None
        self._next_id = 0
        self._annoy_index = None
        self._temp_file: Optional[str] = None
        self._built = False

        # Initialize Annoy
        self._initialize_annoy()

    def _initialize_annoy(self) -> None:
        """Initialize the Annoy library."""
        try:
            import annoy

            self.annoy = annoy
        except ImportError:
            raise ValoriIndexError(
                "annoy library not installed. Install with: pip install annoy"
            )

    def initialize(self, storage_backend=None) -> None:
        """Initialize the Annoy index."""
        if storage_backend is not None:
            self.storage_backend = storage_backend

        if self.build_on_disk:
            # Create temporary file for on-disk building
            fd, self._temp_file = tempfile.mkstemp(suffix=".annoy")
            os.close(fd)

        self._initialized = True

    def insert(
        self, vectors: np.ndarray, metadata: Optional[List[Dict]] = None
    ) -> List[str]:
        """
        Insert vectors into the index (alias for add).

        Args:
            vectors: Array of vectors to insert
            metadata: Optional metadata for each vector

        Returns:
            List of IDs for the inserted vectors
        """
        if metadata is None:
            metadata = [{} for _ in range(len(vectors))]

        assigned_ids = self.add(vectors, metadata)
        return [str(id) for id in assigned_ids]

    def delete(self, ids: List[str]) -> bool:
        """
        Delete vectors by their IDs (alias for remove).

        Args:
            ids: List of vector IDs to delete

        Returns:
            True if successful
        """
        try:
            int_ids = [int(id) for id in ids]
            self.remove(int_ids)
            return True
        except Exception as e:
            raise ValoriIndexError(f"Failed to delete vectors: {str(e)}")

    def update(
        self, id: str, vector: np.ndarray, metadata: Optional[Dict] = None
    ) -> bool:
        """
        Update a vector by its ID.

        Args:
            id: Vector ID to update
            vector: New vector data
            metadata: Optional new metadata

        Returns:
            True if successful
        """
        try:
            int_id = int(id)
            if int_id >= len(self.vectors):
                raise ValoriIndexError(f"Vector ID not found: {id}")

            if metadata is None:
                metadata = self.metadata[int_id]

            # Remove old vector
            self.remove([int_id])

            # Add new vector (but reuse the same ID by temporarily modifying _next_id)
            old_next_id = self._next_id
            self._next_id = int_id
            self.add(vector.reshape(1, -1), [metadata])
            self._next_id = old_next_id

            return True
        except Exception as e:
            raise ValoriIndexError(f"Failed to update vector: {str(e)}")

    def add(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> List[int]:
        """
        Add vectors to the Annoy index.

        Args:
            vectors: Array of vectors to add
            metadata: List of metadata dictionaries

        Returns:
            List of assigned IDs
        """
        if not self._initialized:
            raise ValoriIndexError("Index not initialized")

        if len(vectors) != len(metadata):
            raise ValoriIndexError(
                "Number of vectors must match number of metadata items"
            )

        # Set dimension on first add
        if self.dimension is None:
            self.dimension = vectors.shape[1]
            self._create_annoy_index()

        if vectors.shape[1] != self.dimension:
            raise ValoriIndexError(
                f"Vector dimension mismatch: expected {self.dimension}, got {vectors.shape[1]}"
            )

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

            # Add to Annoy index
            self._annoy_index.add_item(vector_id, vector.astype(np.float32))

            assigned_ids.append(vector_id)

        return assigned_ids

    def build(self) -> None:
        """Build the Annoy index."""
        if not self._initialized:
            raise ValoriIndexError("Index not initialized")

        # Allow building empty index, but need dimension if there are vectors
        if self.dimension is None and len(self.vectors) > 0:
            raise ValoriIndexError("No dimension set for vectors")

        # Only build if there's an annoy index instance
        if self._annoy_index is not None:
            if self.build_on_disk:
                # Build on disk
                self._annoy_index.build(self.num_trees)
                self._annoy_index.save(self._temp_file)

                # Reload from disk
                self._annoy_index.unload()
                self._annoy_index.load(self._temp_file)
            else:
                # Build in memory
                self._annoy_index.build(self.num_trees)

        self._built = True

    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for nearest neighbors using Annoy.

        Args:
            query_vector: Query vector
            k: Number of neighbors to return

        Returns:
            List of search results with 'id', 'distance', and 'metadata'
        """
        if not self._initialized:
            raise ValoriIndexError("Index not initialized")

        if not self._built:
            raise ValoriIndexError(
                "Index not built. Call build() after adding vectors."
            )

        if self.dimension is None:
            return []

        validate_vector(query_vector, self.dimension, self.dimension)

        # Convert query vector to float32
        query_vector = query_vector.astype(np.float32)

        # Determine search_k
        search_k = (
            self.search_k if self.search_k > 0 else self.num_trees * len(self.vectors)
        )

        # Search for nearest neighbors
        neighbor_ids, distances = self._annoy_index.get_nns_by_vector(
            query_vector, k, search_k=search_k, include_distances=True
        )

        # Convert to results format
        results = []
        for neighbor_id, distance in zip(neighbor_ids, distances):
            if neighbor_id < len(self.vectors):
                # Convert distance to similarity based on metric
                similarity = self._distance_to_similarity(distance)

                results.append(
                    {
                        "id": neighbor_id,
                        "distance": similarity,
                        "metadata": self.metadata[neighbor_id],
                    }
                )

        return results

    def remove(self, ids: List[int]) -> None:
        """
        Remove vectors from the index.

        Note: Annoy doesn't support efficient removal, so we mark vectors as removed.

        Args:
            ids: List of IDs to remove
        """
        if not self._initialized:
            raise ValoriIndexError("Index not initialized")

        for vector_id in ids:
            if vector_id < len(self.vectors):
                # Mark as removed (Annoy doesn't support efficient removal)
                self.vectors[vector_id] = None
                self.metadata[vector_id] = None

    def clear(self) -> None:
        """Clear all vectors from the index."""
        self.vectors.clear()
        self.metadata.clear()
        self._next_id = 0
        self._built = False

        if self._annoy_index is not None:
            self._annoy_index.unload()
            self._annoy_index = None

        if self._temp_file and os.path.exists(self._temp_file):
            os.unlink(self._temp_file)
            self._temp_file = None

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        active_vectors = sum(1 for v in self.vectors if v is not None)

        return {
            "index_type": "annoy",
            "dimension": self.dimension,
            "num_vectors": active_vectors,
            "total_capacity": len(self.vectors),
            "num_trees": self.num_trees,
            "metric": self.metric,
            "search_k": self.search_k,
            "build_on_disk": self.build_on_disk,
            "built": self._built,
            "initialized": self._initialized,
        }

    def _create_annoy_index(self) -> None:
        """Create the Annoy index instance."""
        # Annoy library AnnoyIndex constructor only accepts dimension and metric
        self._annoy_index = self.annoy.AnnoyIndex(self.dimension, self.metric)

    def _distance_to_similarity(self, distance: float) -> float:
        """Convert Annoy distance to similarity score."""
        if self.metric == "angular":
            # Angular distance: 0 = identical, 2 = opposite
            # Convert to similarity: 1 - (distance / 2)
            return max(0.0, 1.0 - (distance / 2.0))
        elif self.metric == "euclidean":
            # Euclidean distance: 0 = identical, larger = more different
            # Convert to similarity: 1 / (1 + distance)
            return 1.0 / (1.0 + distance)
        elif self.metric == "manhattan":
            # Manhattan distance: 0 = identical, larger = more different
            # Convert to similarity: 1 / (1 + distance)
            return 1.0 / (1.0 + distance)
        elif self.metric == "hamming":
            # Hamming distance: 0 = identical, 1 = completely different
            # Convert to similarity: 1 - distance
            return 1.0 - distance
        elif self.metric == "dot":
            # Dot product: higher = more similar
            # Normalize to [0, 1] range (assuming normalized vectors)
            return max(0.0, min(1.0, distance))
        else:
            return distance

    def save(self, filepath: str) -> None:
        """Save the index to disk."""
        if not self._built:
            raise ValoriIndexError("Index not built. Call build() before saving.")

        if self._annoy_index is not None:
            # Save the Annoy index
            self._annoy_index.save(filepath)

            # Also save metadata alongside the index
            import json

            metadata_path = filepath + ".meta"
            metadata = {
                "dimension": self.dimension,
                "metric": self.metric,
                "num_trees": self.num_trees,
            }
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)

    def load(self, filepath: str) -> None:
        """Load the index from disk."""
        if not self._initialized:
            raise ValoriIndexError("Index not initialized")

        if not os.path.exists(filepath):
            raise ValoriIndexError(f"Index file not found: {filepath}")

        # Load metadata first
        import json

        metadata_path = filepath + ".meta"
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                self.dimension = metadata.get("dimension")
                self.metric = metadata.get("metric", self.metric)
                # num_trees already set from config during init
        else:
            # Fallback: if no metadata file, try to infer from index properties
            # This is a limitation - we need the dimension to create AnnoyIndex
            if self.dimension is None:
                raise ValoriIndexError(
                    f"Cannot load index without dimension info. "
                    f"Metadata file not found: {metadata_path}"
                )

        self._create_annoy_index()
        self._annoy_index.load(filepath)
        self._built = True

        # Note: When loading from disk, we lose the original vectors and metadata
        # This is a limitation of Annoy - it only stores the tree structure

    def close(self) -> None:
        """Close the index and clean up resources."""
        if self._annoy_index is not None:
            self._annoy_index.unload()
            self._annoy_index = None

        if self._temp_file and os.path.exists(self._temp_file):
            os.unlink(self._temp_file)
            self._temp_file = None

        self.clear()
        self._initialized = False
