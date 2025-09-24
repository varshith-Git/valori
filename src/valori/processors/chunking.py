"""
Text chunking processor for the Vectara vector database.
"""

import re
from typing import Any, Dict, List, Optional
import numpy as np

from .base import DocumentProcessor
from ..exceptions import ProcessingError


class ChunkingProcessor(DocumentProcessor):
    """
    Processor for splitting text into chunks for vector database indexing.
    
    Supports various chunking strategies including fixed-size, semantic,
    and sliding window approaches.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize chunking processor."""
        super().__init__(config)
        self.strategy = config.get("strategy", "fixed_size")  # fixed_size, semantic, sliding_window
        self.chunk_size = config.get("chunk_size", 1000)
        self.chunk_overlap = config.get("chunk_overlap", 100)
        self.min_chunk_size = config.get("min_chunk_size", 50)
        self.max_chunk_size = config.get("max_chunk_size", 2000)
        self.separators = config.get("separators", ["\n\n", "\n", ". ", "! ", "? ", " "])
        self.preserve_whitespace = config.get("preserve_whitespace", False)
        self._chunks_created = 0
        self._total_characters_processed = 0
    
    def initialize(self) -> None:
        """Initialize the chunking processor."""
        if self.chunk_size <= 0:
            raise ProcessingError("Chunk size must be positive")
        
        if self.chunk_overlap >= self.chunk_size:
            raise ProcessingError("Chunk overlap must be less than chunk size")
        
        self._initialized = True
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process document data and create chunks.
        
        Args:
            data: Document data containing 'text' and optionally 'chunks'
            
        Returns:
            Document data with chunks
        """
        if not self._initialized:
            raise ProcessingError("Processor not initialized")
        
        if "text" not in data:
            raise ProcessingError("Input data must contain 'text' field")
        
        text = data["text"]
        self._total_characters_processed += len(text)
        
        # Create chunks based on strategy
        if self.strategy == "fixed_size":
            chunks = self._fixed_size_chunking(text)
        elif self.strategy == "semantic":
            chunks = self._semantic_chunking(text)
        elif self.strategy == "sliding_window":
            chunks = self._sliding_window_chunking(text)
        else:
            raise ProcessingError(f"Unknown chunking strategy: {self.strategy}")
        
        # Filter chunks by size
        filtered_chunks = self._filter_chunks_by_size(chunks)
        self._chunks_created += len(filtered_chunks)
        
        # Update data with chunks
        result_data = data.copy()
        result_data["chunks"] = filtered_chunks
        
        return result_data
    
    def _fixed_size_chunking(self, text: str) -> List[Dict[str, Any]]:
        """Create fixed-size chunks."""
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # Try to break at separator boundary
            if end < len(text):
                best_break = self._find_best_break(text, start, end)
                if best_break > start + self.min_chunk_size:
                    end = best_break
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "start_pos": start,
                    "end_pos": end,
                    "length": len(chunk_text),
                    "metadata": {
                        "chunk_type": "fixed_size",
                        "strategy": self.strategy
                    }
                })
                chunk_id += 1
            
            # Move start position with overlap
            start = max(end - self.chunk_overlap, start + 1)
        
        return chunks
    
    def _semantic_chunking(self, text: str) -> List[Dict[str, Any]]:
        """Create semantic chunks based on content boundaries."""
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        current_chunk = ""
        current_start = 0
        chunk_id = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If adding this paragraph would exceed chunk size, finalize current chunk
            if current_chunk and len(current_chunk) + len(paragraph) > self.chunk_size:
                if current_chunk:
                    chunks.append({
                        "id": chunk_id,
                        "text": current_chunk.strip(),
                        "start_pos": current_start,
                        "end_pos": current_start + len(current_chunk),
                        "length": len(current_chunk.strip()),
                        "metadata": {
                            "chunk_type": "semantic",
                            "strategy": self.strategy
                        }
                    })
                    chunk_id += 1
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else ""
                current_chunk = overlap_text + paragraph
                current_start = max(0, current_start + len(current_chunk) - self.chunk_overlap)
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                "id": chunk_id,
                "text": current_chunk.strip(),
                "start_pos": current_start,
                "end_pos": current_start + len(current_chunk),
                "length": len(current_chunk.strip()),
                "metadata": {
                    "chunk_type": "semantic",
                    "strategy": self.strategy
                }
            })
        
        return chunks
    
    def _sliding_window_chunking(self, text: str) -> List[Dict[str, Any]]:
        """Create chunks using sliding window approach."""
        chunks = []
        window_size = self.chunk_size
        step_size = window_size - self.chunk_overlap
        
        for i in range(0, len(text), step_size):
            end = min(i + window_size, len(text))
            chunk_text = text[i:end].strip()
            
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append({
                    "id": len(chunks),
                    "text": chunk_text,
                    "start_pos": i,
                    "end_pos": end,
                    "length": len(chunk_text),
                    "metadata": {
                        "chunk_type": "sliding_window",
                        "strategy": self.strategy,
                        "window_position": i
                    }
                })
        
        return chunks
    
    def _find_best_break(self, text: str, start: int, end: int) -> int:
        """Find the best break point within the given range."""
        best_break = end
        
        for separator in self.separators:
            # Look for separator from the end backwards
            break_pos = text.rfind(separator, start, end)
            if break_pos > start + self.min_chunk_size:
                best_break = break_pos + len(separator)
                break
        
        return best_break
    
    def _filter_chunks_by_size(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter chunks by size constraints."""
        filtered_chunks = []
        
        for chunk in chunks:
            if self.min_chunk_size <= len(chunk["text"]) <= self.max_chunk_size:
                filtered_chunks.append(chunk)
        
        return filtered_chunks
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chunking processor statistics."""
        return {
            "processor_type": "chunking",
            "strategy": self.strategy,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "min_chunk_size": self.min_chunk_size,
            "max_chunk_size": self.max_chunk_size,
            "chunks_created": self._chunks_created,
            "total_characters_processed": self._total_characters_processed,
            "average_chunk_size": self._total_characters_processed / max(self._chunks_created, 1),
            "initialized": self._initialized,
        }
