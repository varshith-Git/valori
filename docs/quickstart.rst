Quick Start Guide
=================

This guide will help you get started with Vectara quickly. We'll cover the basic
concepts and show you how to build your first vector database.

Basic Concepts
--------------

Vectara is built around several key components:

* **Storage Backend**: Handles persistence of vectors and metadata
* **Index**: Provides efficient similarity search algorithms
* **Quantizer**: Optional component for vector compression
* **Persistence Manager**: Handles saving and loading database state
* **Client**: High-level interface that coordinates all components

Installation
------------

Install Vectara using pip:

.. code-block:: bash

    pip install vectara

Or install from source:

.. code-block:: bash

    git clone https://github.com/varshith-Git/valori.git
    cd valori
    pip install -e .

Basic Usage
-----------

Let's start with a simple example:

.. code-block:: python

    import numpy as np
    from vectordb import VectorDBClient
    from vectordb.storage import MemoryStorage
    from vectordb.indices import FlatIndex
    
    # 1. Create components
    storage = MemoryStorage({})
    index = FlatIndex({"metric": "cosine"})
    
    # 2. Create client
    client = VectorDBClient(storage, index)
    client.initialize()
    
    # 3. Generate sample data
    np.random.seed(42)
    vectors = np.random.randn(100, 128).astype(np.float32)
    metadata = [{"id": i, "category": f"cat_{i%10}"} for i in range(100)]
    
    # 4. Insert vectors
    inserted_ids = client.insert(vectors, metadata)
    print(f"Inserted {len(inserted_ids)} vectors")
    
    # 5. Search for similar vectors
    query_vector = vectors[0]
    results = client.search(query_vector, k=5)
    
    for i, result in enumerate(results):
        print(f"{i+1}. ID: {result['id']}, Distance: {result['distance']:.4f}")
    
    # 6. Clean up
    client.close()

Storage Backends
----------------

Vectara provides three storage backends:

Memory Storage
~~~~~~~~~~~~~~

Fast but not persistent across restarts:

.. code-block:: python

    from vectordb.storage import MemoryStorage
    
    storage = MemoryStorage({})

Disk Storage
~~~~~~~~~~~~

Persistent but slower than memory:

.. code-block:: python

    from vectordb.storage import DiskStorage
    
    storage = DiskStorage({"data_dir": "./my_vectordb"})

Hybrid Storage
~~~~~~~~~~~~~~

Combines memory and disk for optimal performance:

.. code-block:: python

    from vectordb.storage import HybridStorage
    
    storage = HybridStorage({
        "memory": {},
        "disk": {"data_dir": "./my_vectordb"},
        "memory_limit": 10000
    })

Index Types
-----------

Vectara supports three index types:

Flat Index
~~~~~~~~~~

Exhaustive search, accurate but slower for large datasets:

.. code-block:: python

    from vectordb.indices import FlatIndex
    
    index = FlatIndex({"metric": "cosine"})  # or "euclidean"

HNSW Index
~~~~~~~~~~

Fast approximate search for large datasets:

.. code-block:: python

    from vectordb.indices import HNSWIndex
    
    index = HNSWIndex({
        "metric": "cosine",
        "m": 16,
        "ef_construction": 200,
        "ef_search": 50
    })

IVF Index
~~~~~~~~~

Clustering-based index for large datasets:

.. code-block:: python

    from vectordb.indices import IVFIndex
    
    index = IVFIndex({
        "metric": "cosine",
        "n_clusters": 100,
        "n_probes": 10
    })

Quantization
------------

Reduce memory usage with vector quantization:

Scalar Quantization
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from vectordb.quantization import ScalarQuantizer
    
    quantizer = ScalarQuantizer({"bits": 8})

Product Quantization
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from vectordb.quantization import ProductQuantizer
    
    quantizer = ProductQuantizer({"m": 8, "k": 256})

Complete Example
----------------

Here's a complete example with all components:

.. code-block:: python

    import numpy as np
    from vectordb import VectorDBClient
    from vectordb.storage import HybridStorage
    from vectordb.indices import HNSWIndex
    from vectordb.quantization import ProductQuantizer
    from vectordb.persistence import TensorPersistence
    
    # Create all components
    storage = HybridStorage({
        "memory": {},
        "disk": {"data_dir": "./vectordb_data"},
        "memory_limit": 10000
    })
    
    index = HNSWIndex({
        "metric": "cosine",
        "m": 32,
        "ef_construction": 400,
        "ef_search": 100
    })
    
    quantizer = ProductQuantizer({
        "m": 16,
        "k": 256
    })
    
    persistence = TensorPersistence({
        "data_dir": "./vectordb_persistence",
        "compression": True
    })
    
    # Create client
    client = VectorDBClient(storage, index, quantizer, persistence)
    client.initialize()
    
    # Generate data
    np.random.seed(42)
    vectors = np.random.randn(1000, 512).astype(np.float32)
    metadata = [{"id": i} for i in range(1000)]
    
    # Insert vectors
    inserted_ids = client.insert(vectors, metadata)
    
    # Search
    query_vector = vectors[0]
    results = client.search(query_vector, k=10)
    
    # Get statistics
    stats = client.get_stats()
    print(f"Vector count: {stats['index']['vector_count']}")
    print(f"Memory usage: {stats['storage']['memory']['memory_usage_mb']:.2f} MB")
    print(f"Compression ratio: {stats['quantization']['compression_ratio']:.3f}")
    
    # Clean up
    client.close()

Next Steps
----------

* Explore the :doc:`api` documentation for detailed API reference
* Check out the examples in the `examples/` directory
* Learn about production deployment in the advanced examples
