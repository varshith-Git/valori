"""
Document processing example for the Vectara vector database.

This example demonstrates how to use the document parsing and processing
pipeline to extract, clean, chunk, and embed documents for vector database storage.
"""

import os
import numpy as np
from pathlib import Path
from valori import VectorDBClient
from valori.storage import MemoryStorage
from valori.indices import FlatIndex
from valori.processors import ProcessingPipeline
from valori.parsers import ParserRegistry


def create_sample_documents():
    """Create sample documents for demonstration."""
    # Create sample directory
    sample_dir = Path("sample_documents")
    sample_dir.mkdir(exist_ok=True)
    
    # Create sample text file
    text_content = """
    Machine Learning Fundamentals
    
    Machine learning is a subset of artificial intelligence that focuses on algorithms
    that can learn from data. There are three main types of machine learning:
    
    1. Supervised Learning: Learning with labeled examples
    2. Unsupervised Learning: Finding patterns in unlabeled data
    3. Reinforcement Learning: Learning through interaction with environment
    
    Supervised learning is the most common type and includes algorithms like
    linear regression, decision trees, and neural networks.
    """
    
    with open(sample_dir / "ml_fundamentals.txt", "w", encoding="utf-8") as f:
        f.write(text_content.strip())
    
    # Create sample markdown file
    markdown_content = """
    # Python Programming Guide
    
    ## Introduction
    Python is a high-level programming language known for its simplicity and readability.
    
    ## Key Features
    - Easy to learn and use
    - Extensive library ecosystem
    - Cross-platform compatibility
    - Strong community support
    
    ## Common Use Cases
    1. Web development
    2. Data science and analytics
    3. Machine learning and AI
    4. Automation and scripting
    
    ## Getting Started
    To start programming in Python, you need to install Python from python.org
    and choose a good IDE or text editor.
    """
    
    with open(sample_dir / "python_guide.md", "w", encoding="utf-8") as f:
        f.write(markdown_content.strip())
    
    print(f"Created sample documents in {sample_dir}")
    return sample_dir


def setup_processing_pipeline():
    """Setup the document processing pipeline."""
    print("Setting up document processing pipeline...")
    
    # Configuration for the processing pipeline
    pipeline_config = {
        "parsers": {
            "text": {
                "encoding": "auto",
                "max_file_size": 10 * 1024 * 1024,  # 10MB
                "chunk_size": 500,
                "chunk_overlap": 50
            },
            "pdf": {
                "extract_tables": True,
                "chunk_size": 1000,
                "chunk_overlap": 100
            },
            "office": {
                "extract_tables": True,
                "chunk_size": 800,
                "chunk_overlap": 80
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
                "batch_size": 16,
                "normalize_embeddings": True,
                "cache_embeddings": True
            }
        }
    }
    
    # Create and initialize pipeline
    pipeline = ProcessingPipeline(pipeline_config)
    pipeline.initialize()
    
    print("Pipeline initialized successfully!")
    return pipeline


def process_documents(pipeline, sample_dir):
    """Process documents using the pipeline."""
    print(f"\nProcessing documents from {sample_dir}...")
    
    # Get all document files
    document_files = []
    for ext in ["*.txt", "*.md", "*.pdf", "*.docx"]:
        document_files.extend(sample_dir.glob(ext))
    
    if not document_files:
        print("No documents found to process!")
        return []
    
    print(f"Found {len(document_files)} documents to process")
    
    # Process each document
    processed_documents = []
    for file_path in document_files:
        print(f"Processing: {file_path.name}")
        
        try:
            # Process document through pipeline
            result = pipeline.process_document(file_path)
            
            # Extract key information
            doc_info = {
                "file_name": file_path.name,
                "text_length": len(result.get("text", "")),
                "num_chunks": len(result.get("chunks", [])),
                "has_embedding": "embedding" in result,
                "has_chunk_embeddings": "chunk_embeddings" in result,
                "metadata": result.get("metadata", {}),
                "pipeline_metadata": result.get("pipeline_metadata", {})
            }
            
            processed_documents.append(doc_info)
            print(f"  ✓ Processed successfully")
            print(f"    Text length: {doc_info['text_length']} characters")
            print(f"    Chunks created: {doc_info['num_chunks']}")
            print(f"    Embeddings generated: {doc_info['has_embedding']}")
            
        except Exception as e:
            print(f"  ✗ Failed to process: {str(e)}")
    
    return processed_documents


def store_in_vector_db(processed_documents, pipeline, sample_dir):
    """Store processed documents in vector database."""
    print("\nStoring processed documents in vector database...")
    
    # Setup vector database
    storage = MemoryStorage({})
    index = FlatIndex({"metric": "cosine"})
    client = VectorDBClient(storage, index)
    client.initialize()
    
    # Get documents and their embeddings
    document_files = list(sample_dir.glob("*.txt")) + list(sample_dir.glob("*.md"))
    
    for file_path in document_files:
        try:
            # Process document to get embeddings
            result = pipeline.process_document(file_path)
            
            if "embedding" in result:
                # Convert embedding to numpy array
                embedding = np.array(result["embedding"]).reshape(1, -1)
                
                # Create metadata
                metadata = {
                    "file_name": file_path.name,
                    "file_type": file_path.suffix,
                    "text_length": len(result.get("text", "")),
                    "num_chunks": len(result.get("chunks", [])),
                    "parser_used": result.get("pipeline_metadata", {}).get("parser_used", "unknown")
                }
                
                # Store in vector database
                inserted_ids = client.insert(embedding, [metadata])
                print(f"✓ Stored {file_path.name} with ID: {inserted_ids[0]}")
            
        except Exception as e:
            print(f"✗ Failed to store {file_path.name}: {str(e)}")
    
    return client


def demonstrate_search(client):
    """Demonstrate vector search capabilities."""
    print("\nDemonstrating vector search...")
    
    # Create a search query
    query_text = "What is machine learning?"
    
    # Process query text to get embedding
    pipeline_config = {
        "processors": {
            "embedding": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "model_type": "sentence_transformers",
                "normalize_embeddings": True
            }
        }
    }
    
    pipeline = ProcessingPipeline(pipeline_config)
    pipeline.initialize()
    
    query_result = pipeline.process_text(query_text)
    query_embedding = np.array(query_result["embedding"]).reshape(1, -1)
    
    # Search for similar documents
    results = client.search(query_embedding[0], k=3)
    
    print(f"Query: '{query_text}'")
    print(f"Found {len(results)} similar documents:")
    
    for i, result in enumerate(results):
        print(f"\n{i+1}. Document: {result['metadata']['file_name']}")
        print(f"   Distance: {result['distance']:.4f}")
        print(f"   File type: {result['metadata']['file_type']}")
        print(f"   Text length: {result['metadata']['text_length']} characters")
    
    return results


def main():
    """Run the document processing example."""
    print("Valori Document Processing Example")
    print("=" * 50)
    
    # Step 1: Create sample documents
    sample_dir = create_sample_documents()
    
    # Step 2: Setup processing pipeline
    pipeline = setup_processing_pipeline()
    
    # Step 3: Process documents
    processed_docs = process_documents(pipeline, sample_dir)
    
    # Step 4: Store in vector database
    client = store_in_vector_db(processed_docs, pipeline, sample_dir)
    
    # Step 5: Demonstrate search
    search_results = demonstrate_search(client)
    
    # Step 6: Show pipeline statistics
    print("\nPipeline Statistics:")
    print("-" * 30)
    pipeline_info = pipeline.get_pipeline_info()
    
    for processor in pipeline_info.get("processors", []):
        print(f"\n{processor['name']}:")
        stats = processor["stats"]
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value}")
    
    # Cleanup
    print("\nCleaning up...")
    client.close()
    pipeline.close()
    
    # Remove sample documents
    import shutil
    shutil.rmtree(sample_dir, ignore_errors=True)
    
    print("\n" + "=" * 50)
    print("Document processing example completed!")


if __name__ == "__main__":
    main()
