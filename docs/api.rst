API Reference
=============

This page provides detailed API documentation for Vectara.

Core Classes
------------

.. automodule:: vectordb
   :members:

VectorDBClient
~~~~~~~~~~~~~~

.. autoclass:: vectordb.VectorDBClient
   :members:
   :special-members: __init__

Storage Backends
----------------

.. automodule:: vectordb.storage
   :members:

StorageBackend
~~~~~~~~~~~~~~

.. autoclass:: vectordb.storage.StorageBackend
   :members:
   :special-members: __init__

MemoryStorage
~~~~~~~~~~~~~

.. autoclass:: vectordb.storage.MemoryStorage
   :members:
   :special-members: __init__

DiskStorage
~~~~~~~~~~~

.. autoclass:: vectordb.storage.DiskStorage
   :members:
   :special-members: __init__

HybridStorage
~~~~~~~~~~~~~

.. autoclass:: vectordb.storage.HybridStorage
   :members:
   :special-members: __init__

Index Implementations
---------------------

.. automodule:: vectordb.indices
   :members:

Index
~~~~~

.. autoclass:: vectordb.indices.Index
   :members:
   :special-members: __init__

FlatIndex
~~~~~~~~~

.. autoclass:: vectordb.indices.FlatIndex
   :members:
   :special-members: __init__

HNSWIndex
~~~~~~~~~

.. autoclass:: vectordb.indices.HNSWIndex
   :members:
   :special-members: __init__

IVFIndex
~~~~~~~~

.. autoclass:: vectordb.indices.IVFIndex
   :members:
   :special-members: __init__

Quantization
------------

.. automodule:: vectordb.quantization
   :members:

Quantizer
~~~~~~~~~

.. autoclass:: vectordb.quantization.Quantizer
   :members:
   :special-members: __init__

ScalarQuantizer
~~~~~~~~~~~~~~~

.. autoclass:: vectordb.quantization.ScalarQuantizer
   :members:
   :special-members: __init__

ProductQuantizer
~~~~~~~~~~~~~~~~

.. autoclass:: vectordb.quantization.ProductQuantizer
   :members:
   :special-members: __init__

Persistence
-----------

.. automodule:: vectordb.persistence
   :members:

PersistenceManager
~~~~~~~~~~~~~~~~~~

.. autoclass:: vectordb.persistence.PersistenceManager
   :members:
   :special-members: __init__

TensorPersistence
~~~~~~~~~~~~~~~~~

.. autoclass:: vectordb.persistence.TensorPersistence
   :members:
   :special-members: __init__

IncrementalPersistence
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: vectordb.persistence.IncrementalPersistence
   :members:
   :special-members: __init__

Utilities
---------

.. automodule:: vectordb.utils
   :members:

Similarity Functions
~~~~~~~~~~~~~~~~~~~~

.. automodule:: vectordb.utils.similarity
   :members:

Validation Functions
~~~~~~~~~~~~~~~~~~~~

.. automodule:: vectordb.utils.validation
   :members:

Logging Utilities
~~~~~~~~~~~~~~~~~

.. automodule:: vectordb.utils.logging
   :members:

Exceptions
----------

.. automodule:: vectordb.exceptions
   :members:
