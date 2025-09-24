"""
Docling parser for the Vectara vector database.

Uses Microsoft's Docling library for advanced document parsing.
"""

from typing import Any, Dict, List, Union
from pathlib import Path

from .base import DocumentParser
from ..exceptions import ParsingError


class DoclingParser(DocumentParser):
    """
    Advanced document parser using Microsoft's Docling library.
    
    Provides high-quality parsing for complex documents with better
    structure preservation and table extraction.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Docling parser."""
        super().__init__(config)
        self.extract_tables = config.get("extract_tables", True)
        self.extract_figures = config.get("extract_figures", True)
        self.preserve_layout = config.get("preserve_layout", True)
        self.chunk_size = config.get("chunk_size", 1000)
        self.chunk_overlap = config.get("chunk_overlap", 100)
        self.max_file_size = config.get("max_file_size", 200 * 1024 * 1024)  # 200MB
    
    def initialize(self) -> None:
        """Initialize the Docling parser."""
        try:
            # Try to import docling
            import docling
            from docling.document_converter import DocumentConverter
            from docling.datamodel.base_models import InputFormat
            
            self.docling = docling
            self.converter = DocumentConverter()
            self._initialized = True
        except ImportError:
            raise ParsingError("Docling not installed. Install with: pip install docling")
    
    def parse(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Parse a document using Docling."""
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
            
            # Convert document using Docling
            doc = self.converter.convert(file_path)
            
            # Extract text content
            text_content = []
            tables_content = []
            figures_content = []
            
            # Process document elements
            for element in doc.iterate_items():
                if hasattr(element, 'text') and element.text.strip():
                    text_content.append(element.text.strip())
                
                # Extract tables if requested
                if self.extract_tables and hasattr(element, 'label') and 'table' in element.label.lower():
                    table_data = self._extract_table_data(element)
                    if table_data:
                        tables_content.append(table_data)
                
                # Extract figures if requested
                if self.extract_figures and hasattr(element, 'label') and 'figure' in element.label.lower():
                    figure_data = self._extract_figure_data(element)
                    if figure_data:
                        figures_content.append(figure_data)
            
            full_text = "\n".join(text_content)
            
            # Create metadata
            metadata = {
                "file_name": file_path.name,
                "file_size": file_size,
                "file_type": "docling_parsed",
                "original_format": file_path.suffix,
                "num_text_elements": len(text_content),
                "num_tables": len(tables_content),
                "num_figures": len(figures_content),
                "tables": tables_content,
                "figures": figures_content
            }
            
            # Create chunks with structure preservation
            chunks = self._create_structured_chunks(text_content, tables_content)
            
            # Document structure
            structure = {
                "type": "docling_parsed",
                "text_elements": len(text_content),
                "tables": len(tables_content),
                "figures": len(figures_content),
                "layout_preserved": self.preserve_layout,
                "estimated_reading_time": len(full_text.split()) // 200
            }
            
            return {
                "text": full_text,
                "metadata": metadata,
                "chunks": chunks,
                "structure": structure
            }
            
        except Exception as e:
            raise ParsingError(f"Failed to parse document with Docling {file_path}: {str(e)}")
    
    def _extract_table_data(self, table_element) -> Dict[str, Any]:
        """Extract table data from Docling element."""
        try:
            # This is a simplified implementation
            # In practice, you'd extract the actual table structure
            return {
                "table_id": getattr(table_element, 'id', 'unknown'),
                "rows": getattr(table_element, 'rows', 0),
                "columns": getattr(table_element, 'columns', 0),
                "data": getattr(table_element, 'data', [])
            }
        except Exception:
            return {}
    
    def _extract_figure_data(self, figure_element) -> Dict[str, Any]:
        """Extract figure data from Docling element."""
        try:
            return {
                "figure_id": getattr(figure_element, 'id', 'unknown'),
                "caption": getattr(figure_element, 'caption', ''),
                "type": getattr(figure_element, 'type', 'unknown')
            }
        except Exception:
            return {}
    
    def _create_structured_chunks(self, text_elements: List[str], tables: List[Dict]) -> List[Dict[str, Any]]:
        """Create chunks preserving document structure."""
        chunks = []
        chunk_id = 0
        current_text = ""
        
        for i, text_element in enumerate(text_elements):
            current_text += text_element + "\n"
            
            # Create chunk when size limit is reached
            if len(current_text) >= self.chunk_size:
                chunks.append({
                    "id": chunk_id,
                    "text": current_text.strip(),
                    "start_element": chunk_id * self.chunk_size,
                    "end_element": i,
                    "metadata": {
                        "chunk_type": "docling_segment",
                        "structure_preserved": True
                    }
                })
                chunk_id += 1
                current_text = ""
        
        # Add remaining text
        if current_text.strip():
            chunks.append({
                "id": chunk_id,
                "text": current_text.strip(),
                "start_element": chunk_id * self.chunk_size,
                "end_element": len(text_elements) - 1,
                "metadata": {
                    "chunk_type": "docling_segment",
                    "structure_preserved": True
                }
            })
        
        return chunks
    
    def get_supported_formats(self) -> List[str]:
        """Get supported file formats."""
        return [".pdf", ".docx", ".pptx", ".html", ".xml", ".txt"]
    
    def can_parse(self, file_path: Union[str, Path]) -> bool:
        """Check if file can be parsed."""
        file_path = Path(file_path)
        return file_path.suffix.lower() in self.get_supported_formats()
