"""
Pytest configuration and fixtures for Vectara tests.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from valori.storage import MemoryStorage, DiskStorage, HybridStorage
from valori.indices import FlatIndex, HNSWIndex, IVFIndex
from valori.quantization import ScalarQuantizer, ProductQuantizer
from valori.persistence import TensorPersistence, IncrementalPersistence


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_vectors():
    """Generate sample vectors for testing."""
    np.random.seed(42)
    return np.random.randn(100, 128).astype(np.float32)


@pytest.fixture
def sample_metadata():
    """Generate sample metadata for testing."""
    return [{"id": i, "category": f"category_{i % 10}"} for i in range(100)]


@pytest.fixture
def memory_storage():
    """Create a memory storage backend."""
    return MemoryStorage({})


@pytest.fixture
def disk_storage(temp_dir):
    """Create a disk storage backend."""
    config = {"data_dir": str(temp_dir / "disk_data")}
    return DiskStorage(config)


@pytest.fixture
def hybrid_storage(temp_dir):
    """Create a hybrid storage backend."""
    config = {
        "memory": {},
        "disk": {"data_dir": str(temp_dir / "hybrid_data")},
        "memory_limit": 50
    }
    return HybridStorage(config)


@pytest.fixture
def flat_index():
    """Create a flat index."""
    return FlatIndex({"metric": "cosine"})


@pytest.fixture
def hnsw_index():
    """Create an HNSW index."""
    return HNSWIndex({
        "metric": "cosine",
        "m": 16,
        "ef_construction": 200,
        "ef_search": 50
    })


@pytest.fixture
def ivf_index():
    """Create an IVF index."""
    return IVFIndex({
        "metric": "cosine",
        "n_clusters": 10,
        "n_probes": 5
    })


@pytest.fixture
def scalar_quantizer():
    """Create a scalar quantizer."""
    return ScalarQuantizer({"bits": 8})


@pytest.fixture
def product_quantizer():
    """Create a product quantizer."""
    return ProductQuantizer({"m": 8, "k": 256})


@pytest.fixture
def tensor_persistence(temp_dir):
    """Create a tensor persistence manager."""
    config = {"data_dir": str(temp_dir / "tensor_persistence")}
    return TensorPersistence(config)


@pytest.fixture
def incremental_persistence(temp_dir):
    """Create an incremental persistence manager."""
    config = {
        "data_dir": str(temp_dir / "incremental_persistence"),
        "checkpoint_interval": 10
    }
    return IncrementalPersistence(config)


@pytest.fixture
def query_vector():
    """Generate a query vector for testing."""
    np.random.seed(123)
    return np.random.randn(128).astype(np.float32)


@pytest.fixture
def small_vectors():
    """Generate small sample vectors for testing."""
    np.random.seed(456)
    return np.random.randn(10, 32).astype(np.float32)


@pytest.fixture
def high_dim_vectors():
    """Generate high-dimensional vectors for testing."""
    np.random.seed(789)
    return np.random.randn(50, 512).astype(np.float32)
