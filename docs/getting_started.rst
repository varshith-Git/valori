Getting Started — Valori
========================

This guide explains how to choose components, how each main function and
algorithm works, real-world trade-offs, and production recommendations. The
author: Varshith (varshith.gudur17@gmail.com) — Team: Valori — Project: Valori

Overview
--------

Valori provides modular building blocks for a vector search system:

- Storage backends (Memory, Disk, Hybrid)
- Indexes (Flat, HNSW, IVF, LSH, Annoy)
- Quantizers (Scalar, Product)
- Parsers and Processing Pipelines
- Persistence strategies

This document focuses on practical guidance and API-level usage notes so you
can pick the right configuration for your workload.

Contract (what this guide describes)
------------------------------------

- Inputs: vectors (numpy arrays, shape (N, D)), optional metadata (list/dict)
- Outputs: insert returns ids; search returns ordered results with id,
  distance/score, and metadata when available
- Error modes: invalid shapes, missing components, I/O errors for disk
- Success: accurate/fast nearest neighbors according to chosen index

Choosing a Storage Backend
--------------------------

1) MemoryStorage

- Use when: development, tests, or very low-latency ephemeral workloads.
- Pros: fastest read/write, simplest setup.
- Cons: non-persistent, limited by process memory.
- Typical usage:

.. code-block:: python

    from valori.storage import MemoryStorage
    storage = MemoryStorage({})

2) DiskStorage

- Use when: persistence across restarts is required and dataset fits local
  disk.
- Pros: durable, simple backups.
- Cons: higher latency, I/O so tune batch sizes.

.. code-block:: python

    from valori.storage import DiskStorage
    storage = DiskStorage({"data_dir": "./data"})

3) HybridStorage

- Use when: large datasets where active set should be in-memory and colder
  vectors persisted to disk.
- Pros: balances speed and capacity.
- Cons: more complex; requires memory limit tuning.

.. code-block:: python

    from valori.storage import HybridStorage
    storage = HybridStorage({"memory": {}, "disk": {"data_dir": "./data"}, "memory_limit": 10000})

Index Selection — Algorithms, Trade-offs and Typical Parameters
-------------------------------------------------------------

Flat (Brute-force)

- When to use: small datasets (tens of thousands of vectors), or when exact
  results are required.
- Complexity: O(N) per query but simple to reason about.
- API notes: no build step required; good for incremental inserts.
- Example:

.. code-block:: python

    from valori.indices import FlatIndex
    index = FlatIndex({"metric": "cosine"})

HNSW (Hierarchical Navigable Small World)

- When to use: medium to very large datasets (hundreds of thousands to
  millions), when low-latency approximate nearest neighbors are required.
- Pros: excellent recall/latency trade-off; widely used in production.
- Tuning knobs:
  - m (connectivity): higher m improves recall at cost of index size
  - ef_construction: higher gives better graph for search but longer build
  - ef_search: tune at query time for recall vs latency
- Typical values: m=16..64, ef_construction=200..1000, ef_search=50..500
- Example:

.. code-block:: python

    from valori.indices import HNSWIndex
    index = HNSWIndex({"metric": "cosine", "m": 32, "ef_construction": 400, "ef_search": 100})

IVF (Inverted File — clustering)

- When to use: very large datasets (millions+) where a coarse quantization
  reduces candidate set size.
- Pros: scales to huge corpora; often combined with product quantization.
- Cons: requires training (clustering), insertion may be more expensive.
- Tuning knobs:
  - n_clusters (aka nlist): more clusters reduces per-query cost but may
    hurt recall if too many.
  - n_probes (aka nprobe): controls how many clusters to search at query
    time; higher n_probes increases recall and latency.
- Example:

.. code-block:: python

    from valori.indices import IVFIndex
    index = IVFIndex({"metric": "cosine", "n_clusters": 1024, "n_probes": 10})

LSH (Locality Sensitive Hashing)

- When to use: high-dimensional sparse vectors or when sub-linear time with
  probabilistic guarantees is acceptable.
- Pros: good for approximate membership and very large datasets; simple to
  parallelize.
- Cons: harder to tune for high recall in dense embedding spaces.

Annoy

- When to use: read-heavy workloads where index build is offline (Annoy
  requires build() and is immutable after build).
- Pros: small on-disk indices, fast reads; useful for embedding stores that
  rarely update.
- Cons: not ideal for frequent inserts/real-time updates.

Quantization
------------

Quantization reduces memory and storage by representing vectors approximately.

Scalar Quantizer

- When to use: simple memory savings with minimal compute.
- Pros: fast, simple.
- Cons: lower fidelity than product quantization for high-dim vectors.

Product Quantizer (PQ)

- When to use: large-scale databases where memory is the limiting factor and
  some loss in accuracy is acceptable.
- Typical config: m (subvector count) and k (codebook size, typically 256)
- Works well when combined with IVF for coarse-to-fine search.

API Reference: Common Patterns
------------------------------

Client lifecycle

1) Create components

.. code-block:: python

    storage = MemoryStorage({})
    index = HNSWIndex({"metric": "cosine"})
    client = VectorDBClient(storage, index)

2) Initialize

.. code-block:: python

    client.initialize()

3) Insert (batching recommended)

.. code-block:: python

    # vectors: numpy array shape (N, D)
    # metadata: list of dicts (optional)
    ids = client.insert(vectors, metadata)

Edge cases: ensure vectors.ndim == 2 and vectors.shape[0] == len(metadata)

4) Search

.. code-block:: python

    # single vector query
    results = client.search(query_vector, k=10)

    # batched queries
    results = client.search_batch(query_vectors, k=10)

Results schema (guaranteed fields)

- id: stored id of the vector
- distance: float distance (or similarity)
- metadata: returned if available

5) Update / Delete

- Use client.update(id, vector, metadata) and client.delete(id). Not all
  indexes support efficient updates — IVF and Annoy typically prefer
  offline/batch updates.

Production Recommendations
--------------------------

- Use HybridStorage for large datasets where portion of active set needs low
  latency.
- For high-throughput online inserts prefer HNSW with smaller m and later
  periodic rebuilds.
- For very large datasets combine IVF + PQ for the index and tune nlist/nprobe
  carefully.
- Monitor ef_search or n_probes to balance recall/latency; expose these as
  runtime knobs in configuration.
- Use persistence (TensorPersistence or IncrementalPersistence) to safely
  checkpoint index and metadata. Keep checkpointing frequency in line with
  expected RPO.

Operational checklist

- Logging: ensure structured logs at INFO level and DEBUG for dev. Use
  `valori.utils.logging.setup_logging`.
- Backups: schedule disk backups and snapshot persistence directories.
- Metrics: expose vector count, memory usage, query latency distribution,
  and recall/precision metrics.
- Testing: regressions for index recall when changing build parameters.

Examples and Recipes
--------------------

1) Low-latency production serving (small-medium dataset)

.. code-block:: python

    storage = HybridStorage({"memory": {}, "disk": {"data_dir": "./data"}, "memory_limit": 20000})
    index = HNSWIndex({"metric": "cosine", "m": 32, "ef_construction": 400})
    client = VectorDBClient(storage, index)
    client.initialize()

2) Massive offline index for semantic search (millions)

.. code-block:: python

    index = IVFIndex({"metric": "cosine", "n_clusters": 65536})
    quantizer = ProductQuantizer({"m": 16, "k": 256})
    # Train quantizer and index on a sample then insert all vectors in bulk

Further reading
---------------

See the `examples/` folder for runnable samples. For a complete API
reference, see :doc:`/api`.

Contact & Support
-----------------

Author: Varshith
Email: varshith.gudur17@gmail.com
Team: Valori
