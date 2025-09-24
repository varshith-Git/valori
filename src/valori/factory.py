"""
Factory functions for easy Vectara vector database setup.

This module provides convenient factory functions to create pre-configured
vector database instances for common use cases.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from .client import VectorDBClient
from .storage import MemoryStorage, DiskStorage, HybridStorage
from .indices import FlatIndex, HNSWIndex, IVFIndex, LSHIndex, AnnoyIndex
from .processors import ProcessingPipeline
from .parsers import ParserRegistry
from .utils.helpers import get_recommended_index_config


def create_vector_db(
    num_vectors: Optional[int] = None,
    dimension: Optional[int] = None,
    use_case: str = "balanced",
    index_type: Optional[str] = None,
    storage_type: str = "memory",
    **kwargs
) -> VectorDBClient:
    """
    Create a pre-configured vector database client.
    
    Args:
        num_vectors: Expected number of vectors (for index selection)
        dimension: Vector dimension (for index selection)
        use_case: Use case ('speed', 'accuracy', 'memory', 'balanced')
        index_type: Specific index type to use (optional)
        storage_type: Storage type ('memory', 'disk', 'hybrid')
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured VectorDBClient instance
    """
    # Select index configuration
    if index_type:
        index_config = {"type": index_type}
        if "index_config" in kwargs:
            index_config["config"] = kwargs["index_config"]
        else:
            index_config["config"] = {}
    else:
        index_config = get_recommended_index_config(num_vectors, dimension, use_case)
    
    # Create index
    index = _create_index(index_config["type"], index_config["config"])
    
    # Create storage
    storage = _create_storage(storage_type, kwargs.get("storage_config", {}))
    
    # Create client
    client = VectorDBClient(storage, index)
    client.initialize()
    
    return client


def create_document_db(
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    index_type: str = "annoy",
    **kwargs
) -> tuple[VectorDBClient, ProcessingPipeline]:
    """
    Create a vector database optimized for document processing.
    
    Args:
        embedding_model: Embedding model to use
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        index_type: Index type for vector search
        **kwargs: Additional configuration parameters
        
    Returns:
        Tuple of (VectorDBClient, ProcessingPipeline)
    """
    # Create processing pipeline
    pipeline_config = {
        "parsers": {
            "text": {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
            "pdf": {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
            "office": {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
        },
        "processors": {
            "cleaning": {
                "normalize_whitespace": True,
                "remove_html": True,
                "lowercase": False
            },
            "chunking": {
                "strategy": "semantic",
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            },
            "embedding": {
                "model_name": embedding_model,
                "normalize_embeddings": True,
                "cache_embeddings": True
            }
        }
    }
    
    pipeline = ProcessingPipeline(pipeline_config)
    pipeline.initialize()
    
    # Create vector database
    db_config = {
        "index_type": index_type,
        "storage_type": "memory",
        **kwargs
    }
    
    # Default index config for document search
    if index_type == "annoy":
        db_config["index_config"] = {
            "metric": "angular",
            "num_trees": 20,
            "search_k": -1
        }
    elif index_type == "hnsw":
        db_config["index_config"] = {
            "metric": "cosine",
            "M": 16,
            "ef_construction": 200,
            "ef_search": 50
        }
    
    client = create_vector_db(**db_config)
    
    return client, pipeline


def create_semantic_search_db(
    corpus_texts: List[str],
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    index_type: str = "annoy",
    **kwargs
) -> VectorDBClient:
    """
    Create a vector database for semantic search over text corpus.
    
    Args:
        corpus_texts: List of texts to index
        embedding_model: Embedding model to use
        index_type: Index type for vector search
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured VectorDBClient with indexed texts
    """
    from .utils.helpers import create_vectors_from_text
    
    # Create vectors from texts
    vectors, metadata = create_vectors_from_text(corpus_texts, embedding_model)
    
    # Create vector database
    db_config = {
        "index_type": index_type,
        "storage_type": "memory",
        "dimension": vectors.shape[1],
        "num_vectors": len(vectors),
        **kwargs
    }
    
    client = create_vector_db(**db_config)
    
    # Add vectors
    client.insert(vectors, metadata)
    
    return client


def create_image_search_db(
    image_paths: List[Union[str, Path]],
    feature_extractor: str = "clip",
    index_type: str = "hnsw",
    **kwargs
) -> VectorDBClient:
    """
    Create a vector database for image search.
    
    Args:
        image_paths: List of image file paths
        feature_extractor: Feature extraction method ('clip', 'resnet')
        index_type: Index type for vector search
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured VectorDBClient with indexed images
    """
    # Extract features from images
    vectors, metadata = _extract_image_features(image_paths, feature_extractor)
    
    # Create vector database
    db_config = {
        "index_type": index_type,
        "storage_type": "memory",
        "dimension": vectors.shape[1],
        "num_vectors": len(vectors),
        **kwargs
    }
    
    client = create_vector_db(**db_config)
    
    # Add vectors
    client.insert(vectors, metadata)
    
    return client


def create_hybrid_search_db(
    documents: List[Dict[str, Any]],
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    **kwargs
) -> VectorDBClient:
    """
    Create a vector database for hybrid search (text + metadata).
    
    Args:
        documents: List of documents with 'text' and metadata fields
        embedding_model: Embedding model to use
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured VectorDBClient for hybrid search
    """
    from .utils.helpers import create_vectors_from_text
    
    # Extract texts and metadata
    texts = [doc["text"] for doc in documents]
    metadata = [{k: v for k, v in doc.items() if k != "text"} for doc in documents]
    
    # Create vectors from texts
    vectors, text_metadata = create_vectors_from_text(texts, embedding_model)
    
    # Combine metadata
    combined_metadata = []
    for i, (text_meta, doc_meta) in enumerate(zip(text_metadata, metadata)):
        combined_meta = {**text_meta, **doc_meta}
        combined_meta["doc_id"] = i
        combined_metadata.append(combined_meta)
    
    # Create vector database
    db_config = {
        "index_type": "hnsw",  # Good for hybrid search
        "storage_type": "memory",
        "dimension": vectors.shape[1],
        "num_vectors": len(vectors),
        "index_config": {
            "metric": "cosine",
            "M": 16,
            "ef_construction": 200,
            "ef_search": 50
        },
        **kwargs
    }
    
    client = create_vector_db(**db_config)
    
    # Add vectors
    client.insert(vectors, combined_metadata)
    
    return client


def _create_index(index_type: str, config: Dict[str, Any]):
    """Create index instance based on type."""
    if index_type == "flat":
        return FlatIndex(config)
    elif index_type == "hnsw":
        return HNSWIndex(config)
    elif index_type == "ivf":
        return IVFIndex(config)
    elif index_type == "lsh":
        return LSHIndex(config)
    elif index_type == "annoy":
        return AnnoyIndex(config)
    else:
        raise ValueError(f"Unknown index type: {index_type}")


def _create_storage(storage_type: str, config: Dict[str, Any]):
    """Create storage instance based on type."""
    if storage_type == "memory":
        return MemoryStorage(config)
    elif storage_type == "disk":
        return DiskStorage(config)
    elif storage_type == "hybrid":
        return HybridStorage(config)
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")


def _extract_image_features(
    image_paths: List[Union[str, Path]], 
    extractor: str
) -> tuple:
    """Extract features from images."""
    try:
        if extractor == "clip":
            import clip
            import torch
            from PIL import Image
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load("ViT-B/32", device=device)
            
            vectors = []
            metadata = []
            
            for i, image_path in enumerate(image_paths):
                try:
                    image = Image.open(image_path)
                    image_input = preprocess(image).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        image_features = model.encode_image(image_input)
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    
                    vectors.append(image_features.cpu().numpy().flatten())
                    metadata.append({
                        "image_path": str(image_path),
                        "image_id": i,
                        "extractor": "clip"
                    })
                    
                except Exception as e:
                    print(f"Warning: Failed to process {image_path}: {e}")
                    continue
            
            return np.array(vectors), metadata
            
        else:
            raise ValueError(f"Unknown feature extractor: {extractor}")
            
    except ImportError:
        raise ImportError(f"Required dependencies for {extractor} not installed")


# Pre-configured templates
TEMPLATES = {
    "small_dataset": {
        "index_type": "flat",
        "storage_type": "memory"
    },
    "medium_dataset": {
        "index_type": "annoy",
        "storage_type": "memory",
        "index_config": {
            "metric": "angular",
            "num_trees": 10,
            "search_k": -1
        }
    },
    "large_dataset": {
        "index_type": "ivf",
        "storage_type": "disk",
        "index_config": {
            "metric": "cosine",
            "n_clusters": 1000,
            "n_probes": 10
        }
    },
    "high_dimensional": {
        "index_type": "lsh",
        "storage_type": "memory",
        "index_config": {
            "metric": "cosine",
            "num_hash_tables": 15,
            "hash_size": 20,
            "num_projections": 100,
            "threshold": 0.3
        }
    },
    "real_time_search": {
        "index_type": "annoy",
        "storage_type": "memory",
        "index_config": {
            "metric": "angular",
            "num_trees": 5,
            "search_k": 10
        }
    },
    "document_search": {
        "index_type": "hnsw",
        "storage_type": "memory",
        "index_config": {
            "metric": "cosine",
            "M": 16,
            "ef_construction": 200,
            "ef_search": 50
        }
    }
}


def create_from_template(template_name: str, **overrides) -> VectorDBClient:
    """
    Create vector database from pre-configured template.
    
    Args:
        template_name: Name of the template to use
        **overrides: Configuration overrides
        
    Returns:
        Configured VectorDBClient instance
    """
    if template_name not in TEMPLATES:
        raise ValueError(f"Unknown template: {template_name}. Available: {list(TEMPLATES.keys())}")
    
    config = TEMPLATES[template_name].copy()
    config.update(overrides)
    
    return create_vector_db(**config)
