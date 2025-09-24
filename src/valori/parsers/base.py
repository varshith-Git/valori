"""
Base document parser interface for the Vectara vector database.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


class DocumentParser(ABC):
    """
    Abstract base class for document parsers.
    
    Document parsers extract text and metadata from various document formats
    for vector database indexing and search.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the document parser with configuration."""
        self.config = config
        self._initialized = False
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the parser and any required resources."""
        pass
    
    @abstractmethod
    def parse(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse a document and extract text and metadata.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing:
                - text: Extracted text content
                - metadata: Document metadata
                - chunks: List of text chunks (optional)
                - structure: Document structure (optional)
        """
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported file formats.
        
        Returns:
            List of supported file extensions
        """
        pass
    
    @abstractmethod
    def can_parse(self, file_path: Union[str, Path]) -> bool:
        """
        Check if the parser can handle the given file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file can be parsed
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get parser statistics."""
        return {
            "parser_type": self.__class__.__name__,
            "supported_formats": self.get_supported_formats(),
            "initialized": self._initialized,
        }
    
    def close(self) -> None:
        """Close the parser and clean up resources."""
        self._initialized = False
