"""
Batch operations manager for the Vectara vector database.

This module provides utilities for efficient batch operations,
progress tracking, and resource management.
"""

import time
import numpy as np
from typing import Any, Dict, List, Optional, Union, Callable, Iterator
from pathlib import Path
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from .helpers import timing_decorator
from ..exceptions import ValidationError


class BatchManager:
    """
    Manages batch operations for vector database operations.
    
    Provides efficient batching, progress tracking, and error handling
    for large-scale operations.
    """
    
    def __init__(self, batch_size: int = 1000, max_workers: int = 4):
        """
        Initialize batch manager.
        
        Args:
            batch_size: Size of each batch
            max_workers: Maximum number of worker threads
        """
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.progress_callback: Optional[Callable] = None
        self.error_callback: Optional[Callable] = None
        self._stats = {
            "total_items": 0,
            "processed_items": 0,
            "failed_items": 0,
            "start_time": None,
            "end_time": None
        }
    
    def set_progress_callback(self, callback: Callable[[int, int], None]):
        """Set callback for progress updates."""
        self.progress_callback = callback
    
    def set_error_callback(self, callback: Callable[[Exception, Any], None]):
        """Set callback for error handling."""
        self.error_callback = callback
    
    def batch_insert(
        self, 
        client, 
        vectors: np.ndarray, 
        metadata: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> List[int]:
        """
        Insert vectors in batches with progress tracking.
        
        Args:
            client: VectorDBClient instance
            vectors: Array of vectors to insert
            metadata: List of metadata dictionaries
            show_progress: Whether to show progress
            
        Returns:
            List of all inserted IDs
        """
        self._stats["total_items"] = len(vectors)
        self._stats["start_time"] = time.time()
        
        all_ids = []
        
        try:
            for i in range(0, len(vectors), self.batch_size):
                batch_vectors = vectors[i:i + self.batch_size]
                batch_metadata = metadata[i:i + self.batch_size]
                
                try:
                    batch_ids = client.insert(batch_vectors, batch_metadata)
                    all_ids.extend(batch_ids)
                    
                    self._stats["processed_items"] += len(batch_vectors)
                    
                    if show_progress:
                        self._print_progress()
                    
                    if self.progress_callback:
                        self.progress_callback(
                            self._stats["processed_items"],
                            self._stats["total_items"]
                        )
                
                except Exception as e:
                    self._stats["failed_items"] += len(batch_vectors)
                    
                    if self.error_callback:
                        self.error_callback(e, batch_vectors)
                    else:
                        print(f"Error processing batch {i//self.batch_size + 1}: {e}")
        
        finally:
            self._stats["end_time"] = time.time()
        
        return all_ids
    
    def batch_search(
        self,
        client,
        query_vectors: np.ndarray,
        k: int = 10,
        show_progress: bool = True
    ) -> List[List[Dict[str, Any]]]:
        """
        Search multiple queries in batches.
        
        Args:
            client: VectorDBClient instance
            query_vectors: Array of query vectors
            k: Number of neighbors to return
            show_progress: Whether to show progress
            
        Returns:
            List of search results for each query
        """
        self._stats["total_items"] = len(query_vectors)
        self._stats["start_time"] = time.time()
        
        all_results = []
        
        try:
            for i in range(0, len(query_vectors), self.batch_size):
                batch_queries = query_vectors[i:i + self.batch_size]
                
                batch_results = []
                for query in batch_queries:
                    try:
                        results = client.search(query, k=k)
                        batch_results.append(results)
                        
                    except Exception as e:
                        batch_results.append([])
                        self._stats["failed_items"] += 1
                        
                        if self.error_callback:
                            self.error_callback(e, query)
                
                all_results.extend(batch_results)
                self._stats["processed_items"] += len(batch_queries)
                
                if show_progress:
                    self._print_progress()
                
                if self.progress_callback:
                    self.progress_callback(
                        self._stats["processed_items"],
                        self._stats["total_items"]
                    )
        
        finally:
            self._stats["end_time"] = time.time()
        
        return all_results
    
    def parallel_batch_insert(
        self,
        client,
        vectors: np.ndarray,
        metadata: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> List[int]:
        """
        Insert vectors in parallel batches.
        
        Args:
            client: VectorDBClient instance
            vectors: Array of vectors to insert
            metadata: List of metadata dictionaries
            show_progress: Whether to show progress
            
        Returns:
            List of all inserted IDs
        """
        self._stats["total_items"] = len(vectors)
        self._stats["start_time"] = time.time()
        
        all_ids = []
        
        # Create batches
        batches = []
        for i in range(0, len(vectors), self.batch_size):
            batches.append({
                "vectors": vectors[i:i + self.batch_size],
                "metadata": metadata[i:i + self.batch_size],
                "batch_id": i // self.batch_size
            })
        
        def process_batch(batch_data):
            """Process a single batch."""
            try:
                batch_ids = client.insert(
                    batch_data["vectors"],
                    batch_data["metadata"]
                )
                return batch_data["batch_id"], batch_ids, None
            except Exception as e:
                return batch_data["batch_id"], [], e
        
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all batches
                future_to_batch = {
                    executor.submit(process_batch, batch): batch
                    for batch in batches
                }
                
                # Collect results
                for future in as_completed(future_to_batch):
                    batch_id, batch_ids, error = future.result()
                    
                    if error:
                        self._stats["failed_items"] += len(batches[batch_id]["vectors"])
                        if self.error_callback:
                            self.error_callback(error, batches[batch_id])
                    else:
                        all_ids.extend(batch_ids)
                        self._stats["processed_items"] += len(batch_ids)
                    
                    if show_progress:
                        self._print_progress()
                    
                    if self.progress_callback:
                        self.progress_callback(
                            self._stats["processed_items"],
                            self._stats["total_items"]
                        )
        
        finally:
            self._stats["end_time"] = time.time()
        
        return all_ids
    
    def _print_progress(self):
        """Print progress information."""
        if self._stats["total_items"] == 0:
            return
        
        progress = self._stats["processed_items"] / self._stats["total_items"]
        percentage = progress * 100
        
        elapsed = time.time() - self._stats["start_time"]
        if self._stats["processed_items"] > 0:
            eta = (elapsed / self._stats["processed_items"]) * (
                self._stats["total_items"] - self._stats["processed_items"]
            )
        else:
            eta = 0
        
        print(f"\rProgress: {percentage:.1f}% ({self._stats['processed_items']}/{self._stats['total_items']}) "
              f"| Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s | Failed: {self._stats['failed_items']}", 
              end="", flush=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics."""
        stats = self._stats.copy()
        
        if stats["start_time"] and stats["end_time"]:
            stats["total_time"] = stats["end_time"] - stats["start_time"]
            if stats["processed_items"] > 0:
                stats["items_per_second"] = stats["processed_items"] / stats["total_time"]
            else:
                stats["items_per_second"] = 0
        else:
            stats["total_time"] = 0
            stats["items_per_second"] = 0
        
        return stats
    
    def reset_stats(self):
        """Reset processing statistics."""
        self._stats = {
            "total_items": 0,
            "processed_items": 0,
            "failed_items": 0,
            "start_time": None,
            "end_time": None
        }


class ProgressTracker:
    """Track progress of long-running operations."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
        self.last_update = time.time()
        self.update_interval = 1.0  # Update every second
    
    def update(self, increment: int = 1):
        """Update progress."""
        self.current += increment
        current_time = time.time()
        
        if current_time - self.last_update >= self.update_interval:
            self._print_progress()
            self.last_update = current_time
    
    def finish(self):
        """Mark as finished."""
        self.current = self.total
        self._print_progress()
        print()  # New line
    
    def _print_progress(self):
        """Print progress information."""
        progress = self.current / self.total
        percentage = progress * 100
        
        elapsed = time.time() - self.start_time
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
        else:
            eta = 0
        
        # Create progress bar
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = "█" * filled_length + "-" * (bar_length - filled_length)
        
        print(f"\r{self.description}: |{bar}| {percentage:.1f}% "
              f"({self.current}/{self.total}) "
              f"Elapsed: {elapsed:.1f}s ETA: {eta:.1f}s", 
              end="", flush=True)


class ResourceMonitor:
    """Monitor resource usage during operations."""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.resource_data = []
        self.start_time = None
    
    def start_monitoring(self, interval: float = 1.0):
        """Start resource monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self, interval: float):
        """Resource monitoring loop."""
        try:
            import psutil
            process = psutil.Process()
            
            while self.monitoring:
                try:
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    cpu_percent = process.cpu_percent()
                    
                    self.resource_data.append({
                        "timestamp": time.time(),
                        "memory_mb": memory_mb,
                        "cpu_percent": cpu_percent
                    })
                    
                    time.sleep(interval)
                    
                except Exception:
                    break
        except ImportError:
            print("psutil not available for resource monitoring")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get resource usage summary."""
        if not self.resource_data:
            return {}
        
        memory_values = [d["memory_mb"] for d in self.resource_data]
        cpu_values = [d["cpu_percent"] for d in self.resource_data]
        
        return {
            "duration": self.resource_data[-1]["timestamp"] - self.resource_data[0]["timestamp"],
            "memory": {
                "min": min(memory_values),
                "max": max(memory_values),
                "avg": sum(memory_values) / len(memory_values)
            },
            "cpu": {
                "min": min(cpu_values),
                "max": max(cpu_values),
                "avg": sum(cpu_values) / len(cpu_values)
            }
        }
    
    def save_report(self, filepath: Union[str, Path]):
        """Save resource monitoring report."""
        report = {
            "summary": self.get_summary(),
            "data": self.resource_data
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
