# Valori

[![PyPI version](https://badge.fury.io/py/valori.svg)](https://badge.fury.io/py/valori)
[![Python versions](https://img.shields.io/pypi/pyversions/valori.svg)](https://pypi.org/project/valori/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/varshith-Git/valori/workflows/Tests/badge.svg)](https://github.com/varshith-Git/valori/actions)

A high-performance vector database library for Python that provides efficient storage, indexing, and search capabilities for high-dimensional vectors.

## Features

- **🚀 High Performance**: Optimized for speed with multiple indexing algorithms
- **📄 Document Parsing**: Support for PDF, Office, text, and advanced parsing with Docling/LlamaParse
- **🔄 Processing Pipeline**: Complete document processing with cleaning, chunking, and embedding
- **💾 Multiple Storage Backends**: Memory, disk, and hybrid storage options
- **🔍 Advanced Indexing**: Flat, HNSW, and IVF indices for different use cases
- **🗜️ Vector Quantization**: Scalar and product quantization for memory efficiency
- **💾 Persistence**: Tensor-based and incremental persistence strategies
- **🏭 Production Ready**: Comprehensive logging, monitoring, and error handling
- **🐍 Python Native**: Pure Python implementation with NumPy integration
- **📊 Extensible**: Plugin architecture for custom components

## Installation

Install Valori using pip:

```bash
pip install valori
```

Or install from source:

```bash
git clone https://github.com/varshith-Git/valori.git
cd valori
pip install -e .
```

## Quick Start

```python
import numpy as np
from valori import VectorDBClient
from valori.storage import MemoryStorage
from valori.indices import FlatIndex
from valori.processors import ProcessingPipeline

# Create components
storage = MemoryStorage({})
index = FlatIndex({"metric": "cosine"})

# Create client
client = VectorDBClient(storage, index)
client.initialize()

# Process documents
pipeline_config = {
    "parsers": {"text": {"chunk_size": 1000}},
    "processors": {
        "cleaning": {"normalize_whitespace": True},
        "chunking": {"strategy": "semantic"},
        "embedding": {"model_name": "sentence-transformers/all-MiniLM-L6-v2"}
    }
}

pipeline = ProcessingPipeline(pipeline_config)
pipeline.initialize()

# Process a document
result = pipeline.process_document("document.pdf")
embedding = np.array(result["embedding"]).reshape(1, -1)

# Store in vector database
inserted_ids = client.insert(embedding, [result["metadata"]])

# Search for similar documents
query_text = "machine learning"
query_result = pipeline.process_text(query_text)
query_embedding = np.array(query_result["embedding"])

results = client.search(query_embedding, k=5)
for i, result in enumerate(results):
    print(f"{i+1}. Document: {result['metadata']['file_name']}")

# Clean up
client.close()
pipeline.close()
```

## Components

### Storage Backends

**Memory Storage**: Fast but not persistent
```python
from valori.storage import MemoryStorage
storage = MemoryStorage({})
```

**Disk Storage**: Persistent but slower
```python
from valori.storage import DiskStorage
storage = DiskStorage({"data_dir": "./my_vectordb"})
```

**Hybrid Storage**: Combines memory and disk for optimal performance
```python
from valori.storage import HybridStorage
storage = HybridStorage({
    "memory": {},
    "disk": {"data_dir": "./my_vectordb"},
    "memory_limit": 10000
})
```

### Index Types

**Flat Index**: Exhaustive search, accurate but slower for large datasets
```python
from valori.indices import FlatIndex
index = FlatIndex({"metric": "cosine"})  # or "euclidean"
```

**HNSW Index**: Fast approximate search for large datasets
```python
from valori.indices import HNSWIndex
index = HNSWIndex({
    "metric": "cosine",
    "m": 16,
    "ef_construction": 200,
    "ef_search": 50
})
```

**IVF Index**: Clustering-based index for large datasets
```python
from valori.indices import IVFIndex
index = IVFIndex({
    "metric": "cosine",
    "n_clusters": 100,
    "n_probes": 10
})
```

**LSH Index**: Locality sensitive hashing for high-dimensional data
```python
from valori.indices import LSHIndex
index = LSHIndex({
    "metric": "cosine",
    "num_hash_tables": 10,
    "hash_size": 16,
    "num_projections": 64,
    "threshold": 0.3
})
```

**Annoy Index**: Approximate nearest neighbors with random projection trees
```python
from valori.indices import AnnoyIndex
index = AnnoyIndex({
    "metric": "angular",
    "num_trees": 10,
    "search_k": -1
})

# Add vectors, then build
index.add(vectors, metadata)
index.build()  # Required for Annoy
```

### Document Parsing

Parse various document formats:

**Text and PDF Parsing**:
```python
from valori.parsers import TextParser, PDFParser

# Parse text files
text_parser = TextParser({"encoding": "auto", "chunk_size": 1000})
result = text_parser.parse("document.txt")

# Parse PDF files
pdf_parser = PDFParser({"extract_tables": True, "chunk_size": 1000})
result = pdf_parser.parse("document.pdf")
```

**Advanced Parsing with Docling/LlamaParse**:
```python
from valori.parsers import DoclingParser, LlamaParser

# Microsoft Docling for advanced parsing
docling_parser = DoclingParser({"extract_tables": True, "preserve_layout": True})

# LlamaParse for AI-powered parsing
llama_parser = LlamaParser({"api_key": "your_api_key", "result_type": "text"})
```

### Document Processing Pipeline

**Complete Processing Pipeline**:
```python
from valori.processors import ProcessingPipeline

pipeline_config = {
    "parsers": {"text": {"chunk_size": 1000}},
    "processors": {
        "cleaning": {"normalize_whitespace": True, "remove_html": True},
        "chunking": {"strategy": "semantic", "chunk_size": 1000},
        "embedding": {"model_name": "sentence-transformers/all-MiniLM-L6-v2"}
    }
}

pipeline = ProcessingPipeline(pipeline_config)
pipeline.initialize()

# Process document end-to-end
result = pipeline.process_document("document.pdf")
```

### Quantization

Reduce memory usage with vector quantization:

**Scalar Quantization**:
```python
from valori.quantization import ScalarQuantizer
quantizer = ScalarQuantizer({"bits": 8})
```

**Product Quantization**:
```python
from valori.quantization import ProductQuantizer
quantizer = ProductQuantizer({"m": 8, "k": 256})
```

## Advanced Usage

### Complete Setup with All Components

```python
import numpy as np
from valori import VectorDBClient
from valori.storage import HybridStorage
from valori.indices import HNSWIndex
from valori.quantization import ProductQuantizer
from valori.persistence import TensorPersistence

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

# Your vector operations here...
client.close()
```

### Production Setup

```python
import json
from valori.utils.logging import setup_logging

# Setup logging
setup_logging({
    "level": "INFO",
    "log_to_file": True,
    "log_file": "vectara.log"
})

# Load configuration
with open("config.json", "r") as f:
    config = json.load(f)

# Initialize with production config
client = VectorDBClient.from_config(config)
client.initialize()

# Your production code here...
client.close()
```

## Examples

Check out the `examples/` directory for comprehensive examples:

- `basic_usage.py` - Basic operations and concepts
- `document_processing.py` - Complete document parsing and processing workflow
- `advanced_indexing.py` - LSH and Annoy indexing algorithms comparison
- `advanced_quantization.py` - Quantization techniques and performance
- `production_setup.py` - Production deployment and monitoring

## Documentation

Full documentation is available at [https://github.com/varshith-Git/valori](https://github.com/varshith-Git/valori).

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/varshith-Git/valori.git
cd valori

# Setup development environment
bash scripts/install_dev.sh

# Activate virtual environment
source venv/bin/activate
```

### Running Tests

```bash
# Run all tests
bash scripts/run_tests.sh

# Run with coverage
bash scripts/run_tests.sh --coverage

# Run specific tests
bash scripts/run_tests.sh tests/test_storage.py
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/

# Security checks
safety check
bandit -r src/
```

### Building Documentation

```bash
cd docs
make html
```

### Benchmarking

```bash
# Run benchmarks
python scripts/benchmark.py

# Quick benchmarks
python scripts/benchmark.py --quick
```

## Performance

Vectara is designed for high performance:

- **Memory Efficiency**: Up to 75% memory reduction with quantization
- **Search Speed**: Sub-millisecond search times for small datasets
- **Scalability**: Handles millions of vectors with appropriate indexing
- **Flexibility**: Choose the right components for your use case

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- 📚 [Documentation](https://github.com/varshith-Git/valori)
- 🐛 [Issue Tracker](https://github.com/varshith-Git/valori/issues)
- 💬 [Discussions](https://github.com/varshith-Git/valori/discussions)
- 📧 [Email Support](mailto:team@valori.com)

## Roadmap

- [ ] GPU acceleration support
- [ ] Distributed deployment
- [ ] More indexing algorithms (LSH, Annoy)
- [ ] REST API server
- [ ] Web UI for database management
- [ ] Integration with popular ML frameworks

## Citation

If you use Valori in your research, please cite:

```bibtex
@software{valori2024,
  title={Valori: A High-Performance Vector Database for Python},
  author={Valori Team},
  year={2024},
  url={https://github.com/varshith-Git/valori}
}
```

---

Made with ❤️ by the Vectara Team
