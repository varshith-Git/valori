Welcome to Vectara's documentation!
====================================

Vectara is a high-performance vector database library for Python that provides
efficient storage, indexing, and search capabilities for high-dimensional vectors.

Features
--------

* **Multiple Storage Backends**: Memory, disk, and hybrid storage options
* **Advanced Indexing**: Flat, HNSW, and IVF indices for different use cases
* **Vector Quantization**: Scalar and product quantization for memory efficiency
* **Persistence**: Tensor-based and incremental persistence strategies
* **Production Ready**: Comprehensive logging, monitoring, and error handling

Quick Start
-----------

.. code-block:: python

    from vectordb import VectorDBClient
    from vectordb.storage import MemoryStorage
    from vectordb.indices import FlatIndex
    
    # Create components
    storage = MemoryStorage({})
    index = FlatIndex({"metric": "cosine"})
    
    # Create client
    client = VectorDBClient(storage, index)
    client.initialize()
    
    # Insert vectors
    import numpy as np
    vectors = np.random.randn(100, 128).astype(np.float32)
    inserted_ids = client.insert(vectors)
    
    # Search
    query_vector = vectors[0]
    results = client.search(query_vector, k=5)
    
    # Clean up
    client.close()

Installation
------------

.. code-block:: bash

    pip install vectara

Contents
--------

.. toctree::
   :maxdepth: 2

   quickstart
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
