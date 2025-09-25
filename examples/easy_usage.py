"""
Easy usage examples for the Vectara vector database.

This example demonstrates the various supporting functions and utilities
that make the vector database easy to use for common tasks.
"""

import numpy as np
from pathlib import Path
from valori import VectorDBClient
from valori.utils import (
    create_vectors_from_text,
    normalize_vectors,
    find_duplicates,
    save_vectors_to_file,
    load_vectors_from_file,
    timing_decorator,
    BatchManager,
    ProgressTracker,
    VectorAnalyzer,
    PerformanceProfiler,
    IndexInspector
)
from valori.factory import (
    create_vector_db,
    create_document_db,
    create_semantic_search_db,
    create_from_template
)
from valori.configs import get_config, get_recommended_config, list_configs


def demo_helper_functions():
    """Demonstrate helper functions for common operations."""
    print("=" * 60)
    print("HELPER FUNCTIONS DEMO")
    print("=" * 60)
    
    # Create vectors from text
    print("\n1. Creating vectors from text...")
    texts = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing deals with text and speech",
        "Computer vision focuses on image and video analysis"
    ]
    
    vectors, metadata = create_vectors_from_text(texts)
    print(f"Created {len(vectors)} vectors of dimension {vectors.shape[1]}")
    
    # Normalize vectors
    print("\n2. Normalizing vectors...")
    normalized_vectors = normalize_vectors(vectors, method="l2")
    print(f"Normalized vectors: mean norm = {np.mean(np.linalg.norm(normalized_vectors, axis=1)):.4f}")
    
    # Find duplicates
    print("\n3. Finding duplicate vectors...")
    # Add some duplicates for demo
    extended_vectors = np.vstack([vectors, vectors[:2]])  # Add first 2 vectors again
    duplicate_groups = find_duplicates(extended_vectors, threshold=0.99)
    print(f"Found {len(duplicate_groups)} groups of duplicate vectors")
    
    # Save and load vectors
    print("\n4. Saving and loading vectors...")
    save_path = "demo_vectors.npz"
    save_vectors_to_file(vectors, metadata, save_path)
    
    loaded_vectors, loaded_metadata = load_vectors_from_file(save_path)
    print(f"Loaded {len(loaded_vectors)} vectors from file")
    
    # Clean up
    Path(save_path).unlink(missing_ok=True)
    
    return vectors, metadata


def demo_factory_functions():
    """Demonstrate factory functions for easy setup."""
    print("\n" + "=" * 60)
    print("FACTORY FUNCTIONS DEMO")
    print("=" * 60)
    
    # Create vector database from template
    print("\n1. Creating database from template...")
    client = create_from_template("small_dataset")
    print(f"Created {type(client.index).__name__} index")
    
    # Add some test data
    vectors, metadata = demo_helper_functions()
    client.insert(vectors, metadata)
    
    # Search
    query_vector = vectors[0]
    results = client.search(query_vector, k=3)
    print(f"Search returned {len(results)} results")
    
    client.close()
    
    # Create semantic search database
    print("\n2. Creating semantic search database...")
    texts = [
        "Python is a programming language",
        "Machine learning uses algorithms",
        "Data science combines statistics and programming",
        "Artificial intelligence mimics human intelligence"
    ]
    
    search_db = create_semantic_search_db(texts)
    print(f"Created semantic search database with {len(texts)} documents")
    
    # Search in semantic database
    query_text = "What is programming?"
    query_vector, _ = create_vectors_from_text([query_text])
    results = search_db.search(query_vector[0], k=3)
    print(f"Semantic search returned {len(results)} results")
    
    search_db.close()


def demo_batch_operations():
    """Demonstrate batch operations and progress tracking."""
    print("\n" + "=" * 60)
    print("BATCH OPERATIONS DEMO")
    print("=" * 60)
    
    # Create test data
    print("\n1. Creating test data...")
    num_vectors = 1000
    dimension = 128
    
    np.random.seed(42)
    vectors = np.random.randn(num_vectors, dimension).astype(np.float32)
    metadata = [{"id": i, "category": f"cat_{i%10}"} for i in range(num_vectors)]
    
    # Create database
    client = create_from_template("medium_dataset")
    
    # Batch manager with progress tracking
    print("\n2. Batch insertion with progress tracking...")
    batch_manager = BatchManager(batch_size=100)
    
    def progress_callback(processed, total):
        if processed % 500 == 0:  # Print every 500 items
            print(f"Progress: {processed}/{total} ({processed/total*100:.1f}%)")
    
    batch_manager.set_progress_callback(progress_callback)
    
    # Insert in batches
    inserted_ids = batch_manager.batch_insert(client, vectors, metadata, show_progress=True)
    print(f"\nInserted {len(inserted_ids)} vectors")
    
    # Batch search
    print("\n3. Batch search...")
    query_vectors = vectors[:10]  # Search first 10 vectors
    search_results = batch_manager.batch_search(client, query_vectors, k=5)
    print(f"Batch search completed for {len(search_results)} queries")
    
    # Show batch statistics
    stats = batch_manager.get_stats()
    print(f"\nBatch statistics:")
    print(f"  Total items processed: {stats['total_items']}")
    print(f"  Processing rate: {stats.get('items_per_second', 0):.1f} items/sec")
    
    client.close()


def demo_performance_monitoring():
    """Demonstrate performance monitoring and analysis."""
    print("\n" + "=" * 60)
    print("PERFORMANCE MONITORING DEMO")
    print("=" * 60)
    
    # Create test data
    vectors, metadata = demo_helper_functions()
    
    # Performance profiler
    print("\n1. Performance profiling...")
    profiler = PerformanceProfiler()
    
    # Create database with profiling
    with profiler.measure("database_creation"):
        client = create_from_template("small_dataset")
    
    # Insert with profiling
    with profiler.measure("vector_insertion"):
        client.insert(vectors, metadata)
    
    # Search with profiling
    with profiler.measure("vector_search"):
        results = client.search(vectors[0], k=3)
    
    # Show performance summary
    profiler.print_summary()
    
    # Vector analyzer
    print("\n2. Vector analysis...")
    analyzer = VectorAnalyzer()
    analyzer.print_analysis(vectors)
    
    # Index inspector
    print("\n3. Index inspection...")
    inspector = IndexInspector(client.index)
    inspector.print_inspection()
    
    client.close()


def demo_configuration_templates():
    """Demonstrate configuration templates."""
    print("\n" + "=" * 60)
    print("CONFIGURATION TEMPLATES DEMO")
    print("=" * 60)
    
    # List available configurations
    print("\n1. Available configurations:")
    configs = list_configs()
    for category, names in configs.items():
        print(f"  {category}: {len(names) if isinstance(names, list) else 'nested'}")
    
    # Get specific configuration
    print("\n2. Getting specific configurations:")
    
    # Index configuration
    index_config = get_config("index", "medium_dataset")["annoy_fast"]
    print(f"Fast Annoy config: {index_config}")
    
    # Storage configuration
    storage_config = get_config("storage", "hybrid")
    print(f"Hybrid storage config: {storage_config['description']}")
    
    # Document processing configuration
    doc_config = get_config("document", "text_processing")
    print(f"Text processing config: {doc_config['description']}")
    
    # Get recommended configuration
    print("\n3. Recommended configurations:")
    
    # Small dataset
    small_config = get_recommended_config(500, 128, "balanced")
    print(f"Small dataset (500 vectors): {small_config['index']['type']}")
    
    # Large dataset
    large_config = get_recommended_config(100000, 768, "speed")
    print(f"Large dataset (100K vectors): {large_config['index']['type']}")
    
    # High-dimensional data
    high_dim_config = get_recommended_config(10000, 2048, "balanced")
    print(f"High-dimensional (10K x 2048): {high_dim_config['index']['type']}")


def demo_easy_document_processing():
    """Demonstrate easy document processing workflow."""
    print("\n" + "=" * 60)
    print("EASY DOCUMENT PROCESSING DEMO")
    print("=" * 60)
    
    # Create sample documents
    sample_dir = Path("sample_docs")
    sample_dir.mkdir(exist_ok=True)
    
    # Create sample text files
    documents = {
        "ml_basics.txt": """
        Machine Learning Fundamentals
        
        Machine learning is a subset of artificial intelligence that focuses on
        algorithms that can learn from data. There are three main types:
        
        1. Supervised Learning: Learning with labeled examples
        2. Unsupervised Learning: Finding patterns in unlabeled data  
        3. Reinforcement Learning: Learning through interaction
        
        Supervised learning is the most common and includes algorithms like
        linear regression, decision trees, and neural networks.
        """,
        
        "python_guide.txt": """
        Python Programming Guide
        
        Python is a high-level programming language known for its simplicity.
        Key features include:
        
        - Easy to learn and use
        - Extensive library ecosystem
        - Cross-platform compatibility
        - Strong community support
        
        Common use cases include web development, data science, machine learning,
        and automation scripting.
        """
    }
    
    # Write sample files
    for filename, content in documents.items():
        with open(sample_dir / filename, 'w') as f:
            f.write(content.strip())
    
    print("\n1. Created sample documents")
    
    # Create document database
    print("\n2. Creating document database...")
    client, pipeline = create_document_db()
    
    # Process documents
    print("\n3. Processing documents...")
    processed_docs = []
    
    for doc_path in sample_dir.glob("*.txt"):
        print(f"Processing: {doc_path.name}")
        
        try:
            result = pipeline.process_document(doc_path)
            
            if "embedding" in result:
                embedding = np.array(result["embedding"]).reshape(1, -1)
                client.insert(embedding, [result["metadata"]])
                processed_docs.append(result)
                print(f"  ✓ Processed successfully")
            else:
                print(f"  ✗ No embedding generated")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # Search documents
    print("\n4. Searching documents...")
    query_text = "What is machine learning?"
    query_result = pipeline.process_text(query_text)
    query_embedding = np.array(query_result["embedding"])
    
    results = client.search(query_embedding, k=3)
    
    print(f"Query: '{query_text}'")
    print(f"Found {len(results)} relevant documents:")
    
    for i, result in enumerate(results):
        metadata = result["metadata"]
        print(f"\n{i+1}. {metadata.get('file_name', 'Unknown')}")
        print(f"   Similarity: {result['distance']:.4f}")
        print(f"   Text length: {metadata.get('text_length', 'Unknown')} characters")
    
    # Clean up
    client.close()
    pipeline.close()
    
    # Remove sample files
    import shutil
    shutil.rmtree(sample_dir, ignore_errors=True)
    print("\n5. Cleaned up sample files")


def demo_timing_decorators():
    """Demonstrate timing and memory decorators."""
    print("\n" + "=" * 60)
    print("TIMING DECORATORS DEMO")
    print("=" * 60)
    
    @timing_decorator
    def slow_operation():
        """Simulate a slow operation."""
        import time
        time.sleep(0.1)
        return "Operation completed"
    
    # Run timed operation
    print("\n1. Running timed operation...")
    result = slow_operation()
    print(f"Result: {result}")
    
    # Memory usage decorator (if psutil is available)
    try:
        from valori.utils import memory_usage_decorator
        
        @memory_usage_decorator
        def memory_intensive_operation():
            """Create some memory usage."""
            data = np.random.randn(1000, 1000)
            return data.sum()
        
        print("\n2. Running memory-intensive operation...")
        result = memory_intensive_operation()
        print(f"Result: {result}")
        
    except ImportError:
        print("\n2. Memory monitoring requires psutil (not installed)")


def main():
    """Run all easy usage demonstrations."""
    print("Valori Vector Database - Easy Usage Examples")
    print("=" * 60)
    
    try:
        # Run all demonstrations
        demo_helper_functions()
        demo_factory_functions()
        demo_batch_operations()
        demo_performance_monitoring()
        demo_configuration_templates()
        demo_easy_document_processing()
        demo_timing_decorators()
        
        print("\n" + "=" * 60)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print("\nKey takeaways:")
        print("• Helper functions make common operations simple")
        print("• Factory functions provide easy database setup")
        print("• Batch operations handle large datasets efficiently")
        print("• Performance monitoring helps optimize your code")
        print("• Configuration templates cover common use cases")
        print("• Document processing is streamlined and automated")
        
    except Exception as e:
        print(f"\nError running demonstrations: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
