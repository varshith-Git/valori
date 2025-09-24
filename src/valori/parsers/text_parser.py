"""
Text file parser for the Vectara vector database.
"""

import chardet
from typing import Any, Dict, List, Union
from pathlib import Path

from .base import DocumentParser
from ..exceptions import ParsingError


class TextParser(DocumentParser):
    """
    Parser for plain text files.
    
    Supports various text encodings and provides basic text extraction
    with metadata.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize text parser."""
        super().__init__(config)
        self.encoding = config.get("encoding", "auto")
        self.max_file_size = config.get("max_file_size", 100 * 1024 * 1024)  # 100MB
        self.chunk_size = config.get("chunk_size", 1000)
        self.chunk_overlap = config.get("chunk_overlap", 100)
    
    def initialize(self) -> None:
        """Initialize the text parser."""
        self._initialized = True
    
    def parse(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Parse a text file."""
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
            
            # Read file content
            if self.encoding == "auto":
                # Detect encoding
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                    detected = chardet.detect(raw_data)
                    encoding = detected.get('encoding', 'utf-8')
            else:
                encoding = self.encoding
            
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                text = f.read()
            
            # Create metadata
            metadata = {
                "file_name": file_path.name,
                "file_size": file_size,
                "encoding": encoding,
                "file_type": "text",
                "language": self._detect_language(text[:1000]) if len(text) > 100 else "unknown"
            }
            
            # Create chunks if text is long enough
            chunks = []
            if len(text) > self.chunk_size:
                chunks = self._create_chunks(text)
            
            return {
                "text": text,
                "metadata": metadata,
                "chunks": chunks,
                "structure": {"type": "plain_text", "paragraphs": len(text.split('\n\n'))}
            }
            
        except Exception as e:
            raise ParsingError(f"Failed to parse text file {file_path}: {str(e)}")
    
    def get_supported_formats(self) -> List[str]:
        """Get supported text file formats."""
        return [".txt", ".md", ".rst", ".log", ".csv"]
    
    def can_parse(self, file_path: Union[str, Path]) -> bool:
        """Check if file can be parsed."""
        file_path = Path(file_path)
        return file_path.suffix.lower() in self.get_supported_formats()
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection based on common words."""
        # This is a very basic implementation
        # In production, you'd use a proper language detection library
        english_words = ["the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"]
        text_lower = text.lower()
        
        english_count = sum(1 for word in english_words if word in text_lower)
        if english_count > 3:
            return "en"
        return "unknown"
    
    def _create_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Create text chunks with overlap."""
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = text.rfind('.', start, end)
                last_newline = text.rfind('\n', start, end)
                break_point = max(last_period, last_newline)
                
                if break_point > start + self.chunk_size // 2:
                    end = break_point + 1
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "start_pos": start,
                    "end_pos": end,
                    "metadata": {"chunk_type": "text_segment"}
                })
                chunk_id += 1
            
            # Move start position with overlap
            start = max(end - self.chunk_overlap, start + 1)
        
        return chunks
