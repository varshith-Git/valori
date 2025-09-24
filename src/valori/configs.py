"""
Pre-configured templates and configurations for the Vectara vector database.

This module provides ready-to-use configurations for common use cases,
making it easy to get started with the vector database.
"""

from typing import Dict, Any, List


# Index configurations for different scenarios
INDEX_CONFIGS = {
    # Small datasets (< 1K vectors)
    "small_dataset": {
        "flat": {
            "metric": "cosine",
            "description": "Exact search for small datasets"
        }
    },
    
    # Medium datasets (1K - 100K vectors)
    "medium_dataset": {
        "annoy_fast": {
            "metric": "angular",
            "num_trees": 10,
            "search_k": 50,
            "description": "Fast approximate search"
        },
        "annoy_accurate": {
            "metric": "angular", 
            "num_trees": 50,
            "search_k": 200,
            "description": "Accurate approximate search"
        },
        "hnsw_balanced": {
            "metric": "cosine",
            "M": 16,
            "ef_construction": 200,
            "ef_search": 50,
            "description": "Balanced speed and accuracy"
        }
    },
    
    # Large datasets (> 100K vectors)
    "large_dataset": {
        "ivf_efficient": {
            "metric": "cosine",
            "n_clusters": 1000,
            "n_probes": 10,
            "description": "Memory-efficient for large datasets"
        },
        "ivf_fast": {
            "metric": "cosine",
            "n_clusters": 500,
            "n_probes": 5,
            "description": "Fast search for large datasets"
        }
    },
    
    # High-dimensional data (> 1000D)
    "high_dimensional": {
        "lsh_balanced": {
            "metric": "cosine",
            "num_hash_tables": 15,
            "hash_size": 20,
            "num_projections": 100,
            "threshold": 0.3,
            "description": "Balanced precision and recall"
        },
        "lsh_precise": {
            "metric": "cosine",
            "num_hash_tables": 20,
            "hash_size": 25,
            "num_projections": 150,
            "threshold": 0.5,
            "description": "High precision, lower recall"
        },
        "lsh_fast": {
            "metric": "cosine",
            "num_hash_tables": 5,
            "hash_size": 10,
            "num_projections": 50,
            "threshold": 0.1,
            "description": "Fast search, higher recall"
        }
    },
    
    # Real-time applications
    "real_time": {
        "annoy_minimal": {
            "metric": "angular",
            "num_trees": 5,
            "search_k": 10,
            "description": "Minimal latency for real-time search"
        },
        "hnsw_fast": {
            "metric": "cosine",
            "M": 8,
            "ef_construction": 100,
            "ef_search": 20,
            "description": "Fast HNSW configuration"
        }
    }
}

# Storage configurations
STORAGE_CONFIGS = {
    "memory_only": {
        "type": "memory",
        "config": {},
        "description": "In-memory storage for small datasets"
    },
    "disk_only": {
        "type": "disk",
        "config": {
            "data_dir": "./vectara_data",
            "compression": True
        },
        "description": "Disk storage for persistence"
    },
    "hybrid": {
        "type": "hybrid",
        "config": {
            "memory_limit_mb": 1000,
            "disk_path": "./vectara_data",
            "cache_size": 10000
        },
        "description": "Hybrid memory/disk storage"
    }
}

# Document processing configurations
DOCUMENT_CONFIGS = {
    "text_processing": {
        "parsers": {
            "text": {
                "encoding": "auto",
                "max_file_size": 10 * 1024 * 1024,  # 10MB
                "chunk_size": 1000,
                "chunk_overlap": 100
            }
        },
        "processors": {
            "cleaning": {
                "remove_html": True,
                "normalize_whitespace": True,
                "remove_special_chars": False,
                "normalize_unicode": True,
                "lowercase": False,
                "remove_stopwords": False
            },
            "chunking": {
                "strategy": "semantic",
                "chunk_size": 1000,
                "chunk_overlap": 100,
                "min_chunk_size": 100,
                "max_chunk_size": 2000
            },
            "embedding": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "model_type": "sentence_transformers",
                "batch_size": 32,
                "normalize_embeddings": True,
                "cache_embeddings": True
            }
        },
        "description": "Basic text processing pipeline"
    },
    
    "pdf_processing": {
        "parsers": {
            "pdf": {
                "extract_tables": True,
                "extract_images": False,
                "chunk_size": 1000,
                "chunk_overlap": 100,
                "max_file_size": 50 * 1024 * 1024  # 50MB
            }
        },
        "processors": {
            "cleaning": {
                "normalize_whitespace": True,
                "remove_html": False,
                "lowercase": False
            },
            "chunking": {
                "strategy": "semantic",
                "chunk_size": 1000,
                "chunk_overlap": 100
            },
            "embedding": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "normalize_embeddings": True
            }
        },
        "description": "PDF document processing"
    },
    
    "office_processing": {
        "parsers": {
            "office": {
                "extract_tables": True,
                "extract_images": False,
                "chunk_size": 800,
                "chunk_overlap": 80
            }
        },
        "processors": {
            "cleaning": {
                "normalize_whitespace": True,
                "remove_html": True,
                "lowercase": False
            },
            "chunking": {
                "strategy": "semantic",
                "chunk_size": 800,
                "chunk_overlap": 80
            },
            "embedding": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "normalize_embeddings": True
            }
        },
        "description": "Office document processing"
    },
    
    "advanced_processing": {
        "parsers": {
            "text": {"chunk_size": 1000},
            "pdf": {"extract_tables": True, "chunk_size": 1000},
            "office": {"extract_tables": True, "chunk_size": 800},
            "docling": {
                "extract_tables": True,
                "extract_figures": True,
                "preserve_layout": True,
                "chunk_size": 1000
            }
        },
        "processors": {
            "cleaning": {
                "remove_html": True,
                "normalize_whitespace": True,
                "remove_special_chars": False,
                "normalize_unicode": True,
                "lowercase": False,
                "remove_stopwords": True,
                "stopwords_language": "english"
            },
            "chunking": {
                "strategy": "semantic",
                "chunk_size": 1000,
                "chunk_overlap": 100,
                "min_chunk_size": 100,
                "max_chunk_size": 2000
            },
            "embedding": {
                "model_name": "sentence-transformers/all-mpnet-base-v2",
                "model_type": "sentence_transformers",
                "batch_size": 16,
                "normalize_embeddings": True,
                "cache_embeddings": True
            }
        },
        "description": "Advanced processing with multiple parsers"
    }
}

# Embedding model configurations
EMBEDDING_CONFIGS = {
    "fast_small": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "description": "Fast, small model for quick prototyping"
    },
    "balanced": {
        "model_name": "sentence-transformers/all-mpnet-base-v2",
        "dimension": 768,
        "description": "Good balance of speed and quality"
    },
    "high_quality": {
        "model_name": "sentence-transformers/all-MiniLM-L12-v2",
        "dimension": 384,
        "description": "High quality embeddings"
    },
    "multilingual": {
        "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "dimension": 384,
        "description": "Multilingual support"
    },
    "domain_specific": {
        "biomedical": {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "description": "For biomedical texts"
        },
        "legal": {
            "model_name": "sentence-transformers/all-mpnet-base-v2",
            "description": "For legal documents"
        },
        "scientific": {
            "model_name": "sentence-transformers/all-MiniLM-L12-v2",
            "description": "For scientific papers"
        }
    }
}

# Complete application configurations
APPLICATION_CONFIGS = {
    "semantic_search": {
        "index": INDEX_CONFIGS["medium_dataset"]["annoy_balanced"],
        "storage": STORAGE_CONFIGS["memory_only"],
        "embedding": EMBEDDING_CONFIGS["balanced"],
        "description": "General-purpose semantic search"
    },
    
    "document_retrieval": {
        "index": INDEX_CONFIGS["medium_dataset"]["hnsw_balanced"],
        "storage": STORAGE_CONFIGS["hybrid"],
        "processing": DOCUMENT_CONFIGS["text_processing"],
        "description": "Document search and retrieval"
    },
    
    "real_time_chat": {
        "index": INDEX_CONFIGS["real_time"]["annoy_minimal"],
        "storage": STORAGE_CONFIGS["memory_only"],
        "embedding": EMBEDDING_CONFIGS["fast_small"],
        "description": "Real-time chat and Q&A"
    },
    
    "research_papers": {
        "index": INDEX_CONFIGS["large_dataset"]["ivf_efficient"],
        "storage": STORAGE_CONFIGS["disk_only"],
        "processing": DOCUMENT_CONFIGS["pdf_processing"],
        "embedding": EMBEDDING_CONFIGS["high_quality"],
        "description": "Academic paper search"
    },
    
    "enterprise_search": {
        "index": INDEX_CONFIGS["large_dataset"]["ivf_fast"],
        "storage": STORAGE_CONFIGS["hybrid"],
        "processing": DOCUMENT_CONFIGS["advanced_processing"],
        "description": "Enterprise document search"
    },
    
    "multilingual_search": {
        "index": INDEX_CONFIGS["medium_dataset"]["hnsw_balanced"],
        "storage": STORAGE_CONFIGS["memory_only"],
        "embedding": EMBEDDING_CONFIGS["multilingual"],
        "description": "Multilingual document search"
    }
}

# Performance optimization configurations
PERFORMANCE_CONFIGS = {
    "cpu_optimized": {
        "batch_size": 64,
        "num_workers": 4,
        "memory_limit_mb": 2000,
        "description": "Optimized for CPU-only environments"
    },
    
    "memory_optimized": {
        "batch_size": 32,
        "memory_limit_mb": 1000,
        "use_quantization": True,
        "description": "Memory-efficient configuration"
    },
    
    "speed_optimized": {
        "batch_size": 128,
        "num_workers": 8,
        "cache_size": 50000,
        "description": "Maximum speed configuration"
    },
    
    "balanced": {
        "batch_size": 64,
        "num_workers": 4,
        "memory_limit_mb": 1500,
        "cache_size": 25000,
        "description": "Balanced performance configuration"
    }
}


def get_config(category: str, name: str) -> Dict[str, Any]:
    """
    Get a specific configuration.
    
    Args:
        category: Configuration category (index, storage, document, etc.)
        name: Configuration name
        
    Returns:
        Configuration dictionary
    """
    config_map = {
        "index": INDEX_CONFIGS,
        "storage": STORAGE_CONFIGS,
        "document": DOCUMENT_CONFIGS,
        "embedding": EMBEDDING_CONFIGS,
        "application": APPLICATION_CONFIGS,
        "performance": PERFORMANCE_CONFIGS
    }
    
    if category not in config_map:
        raise ValueError(f"Unknown category: {category}")
    
    category_configs = config_map[category]
    
    # Handle nested categories
    if category == "index":
        for subcategory, configs in category_configs.items():
            if name in configs:
                return configs[name]
    else:
        if name not in category_configs:
            raise ValueError(f"Unknown {category} config: {name}")
        return category_configs[name]
    
    raise ValueError(f"Unknown {category} config: {name}")


def list_configs(category: Optional[str] = None) -> Dict[str, List[str]]:
    """
    List available configurations.
    
    Args:
        category: Specific category to list (optional)
        
    Returns:
        Dictionary of available configurations
    """
    config_map = {
        "index": INDEX_CONFIGS,
        "storage": STORAGE_CONFIGS,
        "document": DOCUMENT_CONFIGS,
        "embedding": EMBEDDING_CONFIGS,
        "application": APPLICATION_CONFIGS,
        "performance": PERFORMANCE_CONFIGS
    }
    
    if category:
        if category not in config_map:
            raise ValueError(f"Unknown category: {category}")
        
        if category == "index":
            result = {}
            for subcategory, configs in config_map[category].items():
                result[subcategory] = list(configs.keys())
            return result
        else:
            return {category: list(config_map[category].keys())}
    
    # Return all configurations
    result = {}
    for cat, configs in config_map.items():
        if cat == "index":
            result[cat] = {}
            for subcategory, subconfigs in configs.items():
                result[cat][subcategory] = list(subconfigs.keys())
        else:
            result[cat] = list(configs.keys())
    
    return result


def get_recommended_config(
    num_vectors: int,
    dimension: int,
    use_case: str = "balanced",
    storage_type: str = "memory"
) -> Dict[str, Any]:
    """
    Get recommended configuration based on dataset characteristics.
    
    Args:
        num_vectors: Number of vectors
        dimension: Vector dimension
        use_case: Use case type
        storage_type: Storage type preference
        
    Returns:
        Recommended configuration
    """
    # Select index configuration
    if num_vectors < 1000:
        index_config = get_config("index", "small_dataset")
        index_name = "flat"
    elif num_vectors < 100000:
        if use_case == "speed":
            index_config = get_config("index", "medium_dataset")["annoy_fast"]
            index_name = "annoy"
        elif use_case == "accuracy":
            index_config = get_config("index", "medium_dataset")["hnsw_balanced"]
            index_name = "hnsw"
        else:  # balanced
            index_config = get_config("index", "medium_dataset")["annoy_balanced"]
            index_name = "annoy"
    elif dimension > 1000:
        index_config = get_config("index", "high_dimensional")["lsh_balanced"]
        index_name = "lsh"
    else:
        index_config = get_config("index", "large_dataset")["ivf_efficient"]
        index_name = "ivf"
    
    # Select storage configuration
    storage_config = get_config("storage", storage_type)
    
    # Select embedding configuration
    if dimension <= 384:
        embedding_config = get_config("embedding", "fast_small")
    elif dimension <= 768:
        embedding_config = get_config("embedding", "balanced")
    else:
        embedding_config = get_config("embedding", "high_quality")
    
    return {
        "index": {
            "type": index_name,
            "config": index_config
        },
        "storage": storage_config,
        "embedding": embedding_config,
        "metadata": {
            "num_vectors": num_vectors,
            "dimension": dimension,
            "use_case": use_case,
            "storage_type": storage_type
        }
    }
