"""
Production setup example for the Vectara vector database.

This example demonstrates how to set up the vector database for production use
with proper configuration, persistence, monitoring, and error handling.
"""

import numpy as np
import json
import time
from pathlib import Path
from valori import VectorDBClient
from valori.storage import HybridStorage
from valori.indices import HNSWIndex
from valori.quantization import ProductQuantizer
from valori.persistence import TensorPersistence
from valori.utils.logging import setup_logging, get_logger


class ProductionVectorDB:
    """Production-ready vector database wrapper."""
    
    def __init__(self, config_path: str):
        """Initialize production vector database."""
        self.config = self._load_config(config_path)
        self.client = None
        self.logger = get_logger(__name__)
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from file."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def initialize(self):
        """Initialize the vector database with production configuration."""
        try:
            self.logger.info("Initializing production vector database...")
            
            # Setup logging
            setup_logging(self.config.get("logging", {}))
            
            # Create components based on config
            storage_config = self.config["storage"]
            storage = HybridStorage(storage_config)
            
            index_config = self.config["index"]
            index = HNSWIndex(index_config)
            
            quantizer_config = self.config.get("quantization")
            quantizer = None
            if quantizer_config:
                quantizer = ProductQuantizer(quantizer_config)
            
            persistence_config = self.config.get("persistence")
            persistence = None
            if persistence_config:
                persistence = TensorPersistence(persistence_config)
            
            # Create client
            self.client = VectorDBClient(storage, index, quantizer, persistence)
            self.client.initialize()
            
            self.logger.info("Vector database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vector database: {str(e)}")
            raise
    
    def insert_vectors(self, vectors: np.ndarray, metadata: list = None) -> list:
        """Insert vectors with error handling and logging."""
        try:
            start_time = time.time()
            
            self.logger.info(f"Inserting {len(vectors)} vectors...")
            inserted_ids = self.client.insert(vectors, metadata)
            
            duration = time.time() - start_time
            self.logger.info(f"Inserted {len(inserted_ids)} vectors in {duration:.2f} seconds")
            
            return inserted_ids
            
        except Exception as e:
            self.logger.error(f"Failed to insert vectors: {str(e)}")
            raise
    
    def search_vectors(self, query_vector: np.ndarray, k: int = 10) -> list:
        """Search vectors with error handling and logging."""
        try:
            start_time = time.time()
            
            self.logger.debug(f"Searching for {k} similar vectors...")
            results = self.client.search(query_vector, k)
            
            duration = time.time() - start_time
            self.logger.info(f"Search completed in {duration:.4f} seconds, found {len(results)} results")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}")
            raise
    
    def update_vector(self, vector_id: str, vector: np.ndarray, metadata: dict = None) -> bool:
        """Update vector with error handling."""
        try:
            self.logger.info(f"Updating vector {vector_id}")
            success = self.client.update(vector_id, vector, metadata)
            
            if success:
                self.logger.info(f"Successfully updated vector {vector_id}")
            else:
                self.logger.warning(f"Failed to update vector {vector_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to update vector {vector_id}: {str(e)}")
            raise
    
    def delete_vectors(self, vector_ids: list) -> bool:
        """Delete vectors with error handling."""
        try:
            self.logger.info(f"Deleting {len(vector_ids)} vectors")
            success = self.client.delete(vector_ids)
            
            if success:
                self.logger.info(f"Successfully deleted {len(vector_ids)} vectors")
            else:
                self.logger.warning(f"Failed to delete some vectors")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to delete vectors: {str(e)}")
            raise
    
    def get_health_status(self) -> dict:
        """Get health status of the vector database."""
        try:
            stats = self.client.get_stats()
            
            health = {
                "status": "healthy",
                "timestamp": time.time(),
                "vector_count": stats["index"]["vector_count"],
                "memory_usage_mb": stats["storage"]["memory"]["memory_usage_mb"],
                "disk_usage_mb": stats["storage"]["disk"]["disk_usage_mb"],
                "index_trained": stats["index"].get("trained", True),
                "quantization_enabled": "quantization" in stats,
                "persistence_enabled": "persistence" in stats
            }
            
            # Check for potential issues
            if health["memory_usage_mb"] > 1000:  # 1GB
                health["warnings"] = health.get("warnings", [])
                health["warnings"].append("High memory usage")
            
            if not health["index_trained"]:
                health["warnings"] = health.get("warnings", [])
                health["warnings"].append("Index not trained")
            
            return health
            
        except Exception as e:
            self.logger.error(f"Failed to get health status: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def backup_database(self, backup_path: str) -> bool:
        """Backup database state."""
        try:
            self.logger.info(f"Creating backup at {backup_path}")
            
            # Get current state
            stats = self.client.get_stats()
            backup_data = {
                "timestamp": time.time(),
                "stats": stats,
                "config": self.config
            }
            
            # Save backup
            backup_file = Path(backup_path)
            backup_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(backup_file, 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            self.logger.info("Backup created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create backup: {str(e)}")
            return False
    
    def close(self):
        """Close the vector database."""
        try:
            if self.client:
                self.client.close()
                self.logger.info("Vector database closed successfully")
        except Exception as e:
            self.logger.error(f"Error closing vector database: {str(e)}")


def create_production_config(config_path: str, data_dir: str):
    """Create a production configuration file."""
    config = {
        "storage": {
            "memory": {},
            "disk": {
                "data_dir": f"{data_dir}/disk_storage"
            },
            "memory_limit": 10000
        },
        "index": {
            "metric": "cosine",
            "m": 32,
            "ef_construction": 400,
            "ef_search": 100
        },
        "quantization": {
            "m": 16,
            "k": 256
        },
        "persistence": {
            "data_dir": f"{data_dir}/persistence",
            "compression": True
        },
        "logging": {
            "level": "INFO",
            "log_to_file": True,
            "log_file": f"{data_dir}/vectara.log"
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Production configuration created at {config_path}")


def main():
    """Run production setup example."""
    print("Vectara Vector Database - Production Setup Example")
    print("=" * 55)
    
    # Setup paths
    data_dir = "./production_data"
    config_path = f"{data_dir}/config.json"
    
    # Create production configuration
    print("\n1. Creating production configuration...")
    Path(data_dir).mkdir(exist_ok=True)
    create_production_config(config_path, data_dir)
    
    # Initialize production database
    print("\n2. Initializing production database...")
    db = ProductionVectorDB(config_path)
    db.initialize()
    
    # Generate and insert sample data
    print("\n3. Generating and inserting sample data...")
    np.random.seed(42)
    
    # Create 1000 vectors of dimension 512
    vectors = np.random.randn(1000, 512).astype(np.float32)
    
    # Create metadata
    metadata = [
        {
            "id": i,
            "category": f"category_{i % 50}",
            "timestamp": time.time(),
            "source": "production_example"
        }
        for i in range(1000)
    ]
    
    # Insert in batches
    batch_size = 100
    all_inserted_ids = []
    
    for i in range(0, len(vectors), batch_size):
        batch_vectors = vectors[i:i+batch_size]
        batch_metadata = metadata[i:i+batch_size]
        
        batch_ids = db.insert_vectors(batch_vectors, batch_metadata)
        all_inserted_ids.extend(batch_ids)
        
        print(f"   Inserted batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")
    
    # Perform searches
    print("\n4. Performing searches...")
    query_vector = vectors[0]
    
    # Search with different k values
    for k in [1, 5, 10, 20]:
        results = db.search_vectors(query_vector, k=k)
        print(f"   Top {k} results: {len(results)} found")
    
    # Update some vectors
    print("\n5. Updating vectors...")
    for i in range(5):
        vector_id = all_inserted_ids[i]
        new_vector = np.random.randn(512).astype(np.float32)
        new_metadata = {"updated": True, "update_time": time.time()}
        
        success = db.update_vector(vector_id, new_vector, new_metadata)
        print(f"   Updated vector {vector_id}: {success}")
    
    # Delete some vectors
    print("\n6. Deleting vectors...")
    ids_to_delete = all_inserted_ids[:10]
    success = db.delete_vectors(ids_to_delete)
    print(f"   Deleted {len(ids_to_delete)} vectors: {success}")
    
    # Check health status
    print("\n7. Checking health status...")
    health = db.get_health_status()
    print(f"   Status: {health['status']}")
    print(f"   Vector count: {health['vector_count']}")
    print(f"   Memory usage: {health['memory_usage_mb']:.2f} MB")
    print(f"   Disk usage: {health['disk_usage_mb']:.2f} MB")
    
    if health.get('warnings'):
        print(f"   Warnings: {health['warnings']}")
    
    # Create backup
    print("\n8. Creating backup...")
    backup_path = f"{data_dir}/backup_{int(time.time())}.json"
    backup_success = db.backup_database(backup_path)
    print(f"   Backup created: {backup_success}")
    
    # Performance test
    print("\n9. Running performance test...")
    start_time = time.time()
    
    for _ in range(100):
        query = np.random.randn(512).astype(np.float32)
        db.search_vectors(query, k=10)
    
    duration = time.time() - start_time
    print(f"   100 searches completed in {duration:.2f} seconds")
    print(f"   Average search time: {duration/100*1000:.2f} ms")
    
    # Close database
    print("\n10. Closing database...")
    db.close()
    
    print("\n" + "=" * 55)
    print("Production setup example completed!")
    print(f"Data directory: {data_dir}")
    print(f"Log file: {data_dir}/vectara.log")


if __name__ == "__main__":
    main()
