"""
Basic usage example for the Vectara vector database.

This example demonstrates the fundamental operations of the vector database:
- Creating a client with storage and index
- Inserting vectors
- Searching for similar vectors
- Updating and deleting vectors
"""

import numpy as np
from valori import VectorDBClient
from valori.storage import MemoryStorage
from valori.indices import FlatIndex


def main():
    """Run basic usage example."""
    print("Vectara Vector Database - Basic Usage Example")
    print("=" * 50)
    
    # 1. Create components
    print("\n1. Creating components...")
    storage = MemoryStorage({})
    index = FlatIndex({"metric": "cosine"})
    
    # 2. Create client
    print("2. Creating client...")
    client = VectorDBClient(storage, index)
    client.initialize()
    
    # 3. Generate sample data
    print("3. Generating sample data...")
    np.random.seed(42)
    
    # Create 100 random vectors of dimension 128
    vectors = np.random.randn(100, 128).astype(np.float32)
    
    # Create metadata for each vector
    metadata = [
        {"id": i, "category": f"category_{i % 10}", "value": float(i)}
        for i in range(100)
    ]
    
    print(f"   Generated {len(vectors)} vectors of dimension {vectors.shape[1]}")
    
    # 4. Insert vectors
    print("4. Inserting vectors...")
    inserted_ids = client.insert(vectors, metadata)
    print(f"   Inserted {len(inserted_ids)} vectors")
    
    # 5. Search for similar vectors
    print("5. Searching for similar vectors...")
    query_vector = vectors[0]  # Use first vector as query
    
    # Search for top 5 similar vectors
    results = client.search(query_vector, k=5)
    
    print(f"   Found {len(results)} similar vectors:")
    for i, result in enumerate(results):
        print(f"   {i+1}. ID: {result['id']}, Distance: {result['distance']:.4f}")
        print(f"      Metadata: {result['metadata']}")
    
    # 6. Update a vector
    print("\n6. Updating a vector...")
    vector_id = inserted_ids[0]
    new_vector = np.random.randn(128).astype(np.float32)
    new_metadata = {"updated": True, "timestamp": "2024-01-01"}
    
    success = client.update(vector_id, new_vector, new_metadata)
    print(f"   Update successful: {success}")
    
    # Search again to see the update
    results = client.search(query_vector, k=3)
    print(f"   Top 3 results after update:")
    for i, result in enumerate(results):
        print(f"   {i+1}. ID: {result['id']}, Distance: {result['distance']:.4f}")
    
    # 7. Delete some vectors
    print("\n7. Deleting vectors...")
    ids_to_delete = inserted_ids[:5]
    success = client.delete(ids_to_delete)
    print(f"   Deletion successful: {success}")
    
    # Search again to see the effect
    results = client.search(query_vector, k=3)
    print(f"   Top 3 results after deletion:")
    for i, result in enumerate(results):
        print(f"   {i+1}. ID: {result['id']}, Distance: {result['distance']:.4f}")
    
    # 8. Get database statistics
    print("\n8. Database statistics:")
    stats = client.get_stats()
    print(f"   Storage type: {stats['storage']['backend_type']}")
    print(f"   Index type: {stats['index']['index_type']}")
    print(f"   Vector count: {stats['index']['vector_count']}")
    print(f"   Memory usage: {stats['storage']['memory_usage_mb']:.2f} MB")
    
    # 9. Close the client
    print("\n9. Closing client...")
    client.close()
    print("   Client closed successfully")
    
    print("\n" + "=" * 50)
    print("Basic usage example completed!")


if __name__ == "__main__":
    main()
