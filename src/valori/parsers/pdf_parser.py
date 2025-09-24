"""
PDF parser for the Vectara vector database.
"""

from typing import Any, Dict, List, Union
from pathlib import Path

from .base import DocumentParser
from ..exceptions import ParsingError


class PDFParser(DocumentParser):
    """
    Parser for PDF documents.
    
    Extracts text, metadata, and structure from PDF files using PyPDF2.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize PDF parser."""
        super().__init__(config)
        self.extract_images = config.get("extract_images", False)
        self.extract_tables = config.get("extract_tables", False)
        self.chunk_size = config.get("chunk_size", 1000)
        self.chunk_overlap = config.get("chunk_overlap", 100)
        self.max_file_size = config.get("max_file_size", 50 * 1024 * 1024)  # 50MB
    
    def initialize(self) -> None:
        """Initialize the PDF parser."""
        try:
            # Try to import required libraries
            import PyPDF2
            self._initialized = True
        except ImportError:
            raise ParsingError("PyPDF2 not installed. Install with: pip install PyPDF2")
    
    def parse(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Parse a PDF file."""
        if not self._initialized:
            raise ParsingError("Parser not initialized")
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ParsingError(f"File not found: {file_path}")
        
        if not self.can_parse(file_path):
            raise ParsingError(f"Unsupported file format: {file_path.suffix}")
        
        try:
            import PyPDF2
            
            # Check file size
            file_size = file_path.stat().st_size
            if file_size > self.max_file_size:
                raise ParsingError(f"File too large: {file_size} bytes")
            
            # Read PDF
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract text from all pages
                text_parts = []
                page_metadata = []
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_parts.append(page_text)
                            page_metadata.append({
                                "page_number": page_num + 1,
                                "text_length": len(page_text),
                                "has_images": len(page.get_images()) > 0 if hasattr(page, 'get_images') else False
                            })
                    except Exception as e:
                        # Skip pages that can't be processed
                        continue
                
                full_text = "\n\n".join(text_parts)
                
                # Extract metadata
                metadata = {
                    "file_name": file_path.name,
                    "file_size": file_size,
                    "file_type": "pdf",
                    "num_pages": len(pdf_reader.pages),
                    "num_pages_with_text": len(page_metadata),
                    "title": pdf_reader.metadata.get("/Title", "") if pdf_reader.metadata else "",
                    "author": pdf_reader.metadata.get("/Author", "") if pdf_reader.metadata else "",
                    "subject": pdf_reader.metadata.get("/Subject", "") if pdf_reader.metadata else "",
                    "creator": pdf_reader.metadata.get("/Creator", "") if pdf_reader.metadata else "",
                    "producer": pdf_reader.metadata.get("/Producer", "") if pdf_reader.metadata else "",
                    "creation_date": str(pdf_reader.metadata.get("/CreationDate", "")) if pdf_reader.metadata else "",
                    "modification_date": str(pdf_reader.metadata.get("/ModDate", "")) if pdf_reader.metadata else "",
                    "page_metadata": page_metadata
                }
                
                # Create chunks if text is long enough
                chunks = []
                if len(full_text) > self.chunk_size:
                    chunks = self._create_chunks(full_text, page_metadata)
                
                # Document structure
                structure = {
                    "type": "pdf",
                    "pages": len(pdf_reader.pages),
                    "pages_with_text": len(page_metadata),
                    "estimated_reading_time": len(full_text.split()) // 200  # ~200 words per minute
                }
                
                return {
                    "text": full_text,
                    "metadata": metadata,
                    "chunks": chunks,
                    "structure": structure
                }
                
        except Exception as e:
            raise ParsingError(f"Failed to parse PDF file {file_path}: {str(e)}")
    
    def get_supported_formats(self) -> List[str]:
        """Get supported PDF file formats."""
        return [".pdf"]
    
    def can_parse(self, file_path: Union[str, Path]) -> bool:
        """Check if file can be parsed."""
        file_path = Path(file_path)
        return file_path.suffix.lower() in self.get_supported_formats()
    
    def _create_chunks(self, text: str, page_metadata: List[Dict]) -> List[Dict[str, Any]]:
        """Create text chunks with page information."""
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
                # Determine which page this chunk belongs to
                page_num = self._get_page_for_position(start, page_metadata)
                
                chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "start_pos": start,
                    "end_pos": end,
                    "page_number": page_num,
                    "metadata": {
                        "chunk_type": "pdf_segment",
                        "source_page": page_num
                    }
                })
                chunk_id += 1
            
            # Move start position with overlap
            start = max(end - self.chunk_overlap, start + 1)
        
        return chunks
    
    def _get_page_for_position(self, position: int, page_metadata: List[Dict]) -> int:
        """Determine which page a text position belongs to."""
        current_pos = 0
        for page_info in page_metadata:
            if position < current_pos + page_info["text_length"]:
                return page_info["page_number"]
            current_pos += page_info["text_length"]
        return page_metadata[-1]["page_number"] if page_metadata else 1
