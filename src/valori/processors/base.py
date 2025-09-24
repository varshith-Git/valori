"""
Base document processor interface for the Vectara vector database.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np


class DocumentProcessor(ABC):
    """
    Abstract base class for document processors.
    
    Document processors handle various preprocessing tasks such as
    cleaning, chunking, embedding, and other transformations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the document processor with configuration."""
        self.config = config
        self._initialized = False
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the processor and any required resources."""
        pass
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        """
        Process the input data.
        
        Args:
            data: Input data to process
            
        Returns:
            Processed data
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        pass
    
    def close(self) -> None:
        """Close the processor and clean up resources."""
        self._initialized = False
