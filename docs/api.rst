API Reference
=============

This page provides detailed API documentation for Valori.

Core Classes
------------

.. automodule:: valori
   :members:

VectorDBClient
~~~~~~~~~~~~~~

.. autoclass:: valori.VectorDBClient
   :members:
   :special-members: __init__

Storage Backends
----------------

.. automodule:: valori.storage
   :members:

StorageBackend
~~~~~~~~~~~~~~

.. autoclass:: valori.storage.StorageBackend
   :members:
   :special-members: __init__

MemoryStorage
~~~~~~~~~~~~~

.. autoclass:: valori.storage.MemoryStorage
   :members:
   :special-members: __init__

DiskStorage
~~~~~~~~~~~

.. autoclass:: valori.storage.DiskStorage
   :members:
   :special-members: __init__

HybridStorage
~~~~~~~~~~~~~

.. autoclass:: valori.storage.HybridStorage
   :members:
   :special-members: __init__

Index Implementations
---------------------

.. automodule:: valori.indices
   :members:

Index
~~~~~

.. autoclass:: valori.indices.Index
   :members:
   :special-members: __init__

FlatIndex
~~~~~~~~~

.. autoclass:: valori.indices.FlatIndex
   :members:
   :special-members: __init__

HNSWIndex
~~~~~~~~~

.. autoclass:: valori.indices.HNSWIndex
   :members:
   :special-members: __init__

IVFIndex
~~~~~~~~

.. autoclass:: valori.indices.IVFIndex
   :members:
   :special-members: __init__

Quantization
------------

.. automodule:: valori.quantization
   :members:

Quantizer
~~~~~~~~~

.. autoclass:: valori.quantization.Quantizer
   :members:
   :special-members: __init__

ScalarQuantizer
~~~~~~~~~~~~~~~

.. autoclass:: valori.quantization.ScalarQuantizer
   :members:
   :special-members: __init__

ProductQuantizer
~~~~~~~~~~~~~~~~

.. autoclass:: valori.quantization.ProductQuantizer
   :members:
   :special-members: __init__

Persistence
-----------

.. automodule:: valori.persistence
   :members:

PersistenceManager
~~~~~~~~~~~~~~~~~~

.. autoclass:: valori.persistence.PersistenceManager
   :members:
   :special-members: __init__

TensorPersistence
~~~~~~~~~~~~~~~~~

.. autoclass:: valori.persistence.TensorPersistence
   :members:
   :special-members: __init__

IncrementalPersistence
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: valori.persistence.IncrementalPersistence
   :members:
   :special-members: __init__

Utilities
---------

.. automodule:: valori.utils
   :members:

Similarity Functions
~~~~~~~~~~~~~~~~~~~~

.. automodule:: valori.utils.similarity
   :members:

Validation Functions
~~~~~~~~~~~~~~~~~~~~

.. automodule:: valori.utils.validation
   :members:

Logging Utilities
~~~~~~~~~~~~~~~~~

.. automodule:: valori.utils.logging
   :members:

Exceptions
----------

.. automodule:: valori.exceptions
   :members:
