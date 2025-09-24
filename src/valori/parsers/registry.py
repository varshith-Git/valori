"""
Parser registry for the Vectara vector database.

Manages available document parsers and provides automatic parser selection.
"""

from typing import Any, Dict, List, Optional, Union, Type
from pathlib import Path

from .base import DocumentParser
from .text_parser import TextParser
from .pdf_parser import PDFParser
from .office_parser import OfficeParser
from .docling_parser import DoclingParser
from .llama_parser import LlamaParser
from ..exceptions import ParsingError


class ParserRegistry:
    """
    Registry for managing document parsers.
    
    Provides automatic parser selection based on file type and
    manages parser instances with configuration.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize parser registry."""
        self.config = config
        self.parsers: Dict[str, DocumentParser] = {}
        self.parser_classes = {
            "text": TextParser,
            "pdf": PDFParser,
            "office": OfficeParser,
            "docling": DoclingParser,
            "llama": LlamaParser,
        }
        
        # Default parser priorities (higher number = higher priority)
        self.parser_priorities = {
            "text": 1,
            "pdf": 2,
            "office": 3,
            "docling": 4,
            "llama": 5
        }
        
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize the parser registry."""
        # Initialize parsers based on configuration
        for parser_name, parser_config in self.config.get("parsers", {}).items():
            if parser_name in self.parser_classes:
                try:
                    parser = self.parser_classes[parser_name](parser_config)
                    parser.initialize()
                    self.parsers[parser_name] = parser
                except Exception as e:
                    # Skip parsers that can't be initialized
                    continue
        
        self._initialized = True
    
    def get_parser(self, file_path: Union[str, Path], parser_name: Optional[str] = None) -> DocumentParser:
        """
        Get appropriate parser for a file.
        
        Args:
            file_path: Path to the file to parse
            parser_name: Specific parser to use (optional)
            
        Returns:
            Appropriate parser instance
            
        Raises:
            ParsingError: If no suitable parser is found
        """
        if not self._initialized:
            raise ParsingError("Parser registry not initialized")
        
        file_path = Path(file_path)
        
        # Use specific parser if requested
        if parser_name:
            if parser_name in self.parsers:
                return self.parsers[parser_name]
            else:
                raise ParsingError(f"Parser '{parser_name}' not available")
        
        # Find best parser for the file
        suitable_parsers = []
        
        for parser_name, parser in self.parsers.items():
            if parser.can_parse(file_path):
                priority = self.parser_priorities.get(parser_name, 0)
                suitable_parsers.append((priority, parser_name, parser))
        
        if not suitable_parsers:
            raise ParsingError(f"No parser available for file: {file_path}")
        
        # Return parser with highest priority
        suitable_parsers.sort(key=lambda x: x[0], reverse=True)
        return suitable_parsers[0][2]
    
    def parse_document(self, file_path: Union[str, Path], parser_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse a document using the appropriate parser.
        
        Args:
            file_path: Path to the document
            parser_name: Specific parser to use (optional)
            
        Returns:
            Parsed document data
        """
        parser = self.get_parser(file_path, parser_name)
        return parser.parse(file_path)
    
    def get_supported_formats(self) -> List[str]:
        """Get all supported file formats."""
        if not self._initialized:
            return []
        
        all_formats = set()
        for parser in self.parsers.values():
            all_formats.update(parser.get_supported_formats())
        
        return sorted(list(all_formats))
    
    def get_parser_info(self) -> Dict[str, Any]:
        """Get information about available parsers."""
        if not self._initialized:
            return {"parsers": {}}
        
        parser_info = {}
        for name, parser in self.parsers.items():
            parser_info[name] = {
                "available": True,
                "supported_formats": parser.get_supported_formats(),
                "stats": parser.get_stats()
            }
        
        # Add unavailable parsers
        for name in self.parser_classes:
            if name not in parser_info:
                parser_info[name] = {
                    "available": False,
                    "supported_formats": [],
                    "stats": {}
                }
        
        return {
            "parsers": parser_info,
            "supported_formats": self.get_supported_formats(),
            "initialized": self._initialized
        }
    
    def add_parser(self, name: str, parser_class: Type[DocumentParser], config: Dict[str, Any]) -> None:
        """
        Add a custom parser to the registry.
        
        Args:
            name: Parser name
            parser_class: Parser class
            config: Parser configuration
        """
        try:
            parser = parser_class(config)
            parser.initialize()
            self.parsers[name] = parser
            self.parser_classes[name] = parser_class
        except Exception as e:
            raise ParsingError(f"Failed to add parser '{name}': {str(e)}")
    
    def remove_parser(self, name: str) -> None:
        """
        Remove a parser from the registry.
        
        Args:
            name: Parser name to remove
        """
        if name in self.parsers:
            parser = self.parsers[name]
            parser.close()
            del self.parsers[name]
        
        if name in self.parser_classes:
            del self.parser_classes[name]
    
    def close(self) -> None:
        """Close all parsers and clean up resources."""
        for parser in self.parsers.values():
            parser.close()
        
        self.parsers.clear()
        self._initialized = False
