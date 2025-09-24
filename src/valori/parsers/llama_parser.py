"""
LlamaParse parser for the Vectara vector database.

Uses LlamaIndex's LlamaParse for advanced document parsing with AI.
"""

from typing import Any, Dict, List, Union, Optional
from pathlib import Path

from .base import DocumentParser
from ..exceptions import ParsingError


class LlamaParser(DocumentParser):
    """
    Advanced document parser using LlamaIndex's LlamaParse.
    
    Provides AI-powered parsing with better understanding of document
    structure, tables, and content extraction.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize LlamaParse parser."""
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.result_type = config.get("result_type", "text")  # "text", "markdown", "pdf"
        self.use_cache = config.get("use_cache", True)
        self.chunk_size = config.get("chunk_size", 1000)
        self.chunk_overlap = config.get("chunk_overlap", 100)
        self.max_file_size = config.get("max_file_size", 100 * 1024 * 1024)  # 100MB
        self.timeout = config.get("timeout", 2000)  # 2000 seconds
        self.check_interval = config.get("check_interval", 1)  # 1 second
    
    def initialize(self) -> None:
        """Initialize the LlamaParse parser."""
        if not self.api_key:
            raise ParsingError("LlamaParse API key not provided")
        
        try:
            # Try to import llama-parse
            from llama_parse import LlamaParse
            
            self.parser = LlamaParse(
                api_key=self.api_key,
                result_type=self.result_type,
                use_cache=self.use_cache,
                timeout=self.timeout,
                check_interval=self.check_interval
            )
            self._initialized = True
        except ImportError:
            raise ParsingError("llama-parse not installed. Install with: pip install llama-parse")
        except Exception as e:
            raise ParsingError(f"Failed to initialize LlamaParse: {str(e)}")
    
    def parse(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Parse a document using LlamaParse."""
        if not self._initialized:
            raise ParsingError("Parser not initialized")
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ParsingError(f"File not found: {file_path}")
        
        if not self.can_parse(file_path):
            raise ParsingError(f"Unsupported file format: {file_path.suffix}")
        
        try:
            # Check file size
            file_size = file_path.stat().st_size
            if file_size > self.max_file_size:
                raise ParsingError(f"File too large: {file_size} bytes")
            
            # Parse document using LlamaParse
            documents = self.parser.load_data(file_path)
            
            if not documents:
                raise ParsingError("No content extracted from document")
            
            # Combine all document text
            text_parts = []
            metadata_parts = []
            
            for i, doc in enumerate(documents):
                if hasattr(doc, 'text') and doc.text.strip():
                    text_parts.append(doc.text.strip())
                
                # Extract metadata from document
                doc_metadata = {
                    "document_id": i,
                    "text_length": len(doc.text) if hasattr(doc, 'text') else 0,
                    "metadata": getattr(doc, 'metadata', {}),
                    "extra_info": getattr(doc, 'extra_info', {})
                }
                metadata_parts.append(doc_metadata)
            
            full_text = "\n\n".join(text_parts)
            
            # Create comprehensive metadata
            metadata = {
                "file_name": file_path.name,
                "file_size": file_size,
                "file_type": "llamaparse_parsed",
                "original_format": file_path.suffix,
                "num_documents": len(documents),
                "documents_metadata": metadata_parts,
                "parsing_method": "llamaparse",
                "result_type": self.result_type,
                "total_text_length": len(full_text)
            }
            
            # Create chunks
            chunks = self._create_chunks(full_text, metadata_parts)
            
            # Document structure
            structure = {
                "type": "llamaparse_parsed",
                "num_documents": len(documents),
                "ai_enhanced": True,
                "structure_preserved": True,
                "estimated_reading_time": len(full_text.split()) // 200
            }
            
            return {
                "text": full_text,
                "metadata": metadata,
                "chunks": chunks,
                "structure": structure
            }
            
        except Exception as e:
            raise ParsingError(f"Failed to parse document with LlamaParse {file_path}: {str(e)}")
    
    def _create_chunks(self, text: str, documents_metadata: List[Dict]) -> List[Dict[str, Any]]:
        """Create chunks with document structure information."""
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # Try to break at paragraph boundary
            if end < len(text):
                last_newline = text.rfind('\n\n', start, end)
                if last_newline > start + self.chunk_size // 2:
                    end = last_newline + 2
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                # Determine which document this chunk belongs to
                doc_info = self._get_document_info(start, documents_metadata)
                
                chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "start_pos": start,
                    "end_pos": end,
                    "document_id": doc_info.get("document_id", 0),
                    "metadata": {
                        "chunk_type": "llamaparse_segment",
                        "ai_enhanced": True,
                        "document_metadata": doc_info
                    }
                })
                chunk_id += 1
            
            # Move start position with overlap
            start = max(end - self.chunk_overlap, start + 1)
        
        return chunks
    
    def _get_document_info(self, position: int, documents_metadata: List[Dict]) -> Dict[str, Any]:
        """Get document information for a text position."""
        current_pos = 0
        for doc_info in documents_metadata:
            if position < current_pos + doc_info["text_length"]:
                return doc_info
            current_pos += doc_info["text_length"]
        return documents_metadata[-1] if documents_metadata else {}
    
    def get_supported_formats(self) -> List[str]:
        """Get supported file formats."""
        return [".pdf", ".docx", ".pptx", ".txt", ".html", ".md"]
    
    def can_parse(self, file_path: Union[str, Path]) -> bool:
        """Check if file can be parsed."""
        file_path = Path(file_path)
        return file_path.suffix.lower() in self.get_supported_formats()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get parser statistics."""
        stats = super().get_stats()
        stats.update({
            "api_key_configured": bool(self.api_key),
            "result_type": self.result_type,
            "use_cache": self.use_cache,
            "timeout": self.timeout
        })
        return stats
