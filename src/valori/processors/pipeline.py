"""
Document processing pipeline for the Vectara vector database.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from .base import DocumentProcessor
from .chunking import ChunkingProcessor
from .cleaning import CleaningProcessor
from .embedding import EmbeddingProcessor
from ..parsers.registry import ParserRegistry
from ..exceptions import ProcessingError


class ProcessingPipeline:
    """
    Document processing pipeline that combines parsing, cleaning,
    chunking, and embedding generation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize processing pipeline."""
        self.config = config
        self.parser_registry = None
        self.processors: List[DocumentProcessor] = []
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize the processing pipeline."""
        try:
            # Initialize parser registry
            parser_config = self.config.get("parsers", {})
            self.parser_registry = ParserRegistry(parser_config)
            self.parser_registry.initialize()
            
            # Initialize processors in order
            processor_configs = self.config.get("processors", {})
            
            # Add cleaning processor if configured
            if "cleaning" in processor_configs:
                cleaning_processor = CleaningProcessor(processor_configs["cleaning"])
                cleaning_processor.initialize()
                self.processors.append(cleaning_processor)
            
            # Add chunking processor if configured
            if "chunking" in processor_configs:
                chunking_processor = ChunkingProcessor(processor_configs["chunking"])
                chunking_processor.initialize()
                self.processors.append(chunking_processor)
            
            # Add embedding processor if configured
            if "embedding" in processor_configs:
                embedding_processor = EmbeddingProcessor(processor_configs["embedding"])
                embedding_processor.initialize()
                self.processors.append(embedding_processor)
            
            self._initialized = True
            
        except Exception as e:
            raise ProcessingError(f"Failed to initialize processing pipeline: {str(e)}")
    
    def process_document(self, file_path: Union[str, Path], parser_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a document through the complete pipeline.
        
        Args:
            file_path: Path to the document
            parser_name: Specific parser to use (optional)
            
        Returns:
            Processed document data with embeddings and chunks
        """
        if not self._initialized:
            raise ProcessingError("Pipeline not initialized")
        
        try:
            # Step 1: Parse document
            document_data = self.parser_registry.parse_document(file_path, parser_name)
            
            # Step 2: Process through pipeline
            processed_data = document_data
            for processor in self.processors:
                processed_data = processor.process(processed_data)
            
            # Add pipeline metadata
            processed_data["pipeline_metadata"] = {
                "file_path": str(file_path),
                "parser_used": self._get_parser_name(file_path, parser_name),
                "processors_used": [p.__class__.__name__ for p in self.processors],
                "processing_complete": True
            }
            
            return processed_data
            
        except Exception as e:
            raise ProcessingError(f"Failed to process document {file_path}: {str(e)}")
    
    def process_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process raw text through the pipeline (skipping parsing step).
        
        Args:
            text: Raw text to process
            metadata: Optional metadata for the text
            
        Returns:
            Processed text data with embeddings and chunks
        """
        if not self._initialized:
            raise ProcessingError("Pipeline not initialized")
        
        try:
            # Start with text data
            document_data = {
                "text": text,
                "metadata": metadata or {},
                "structure": {"type": "raw_text"}
            }
            
            # Process through pipeline
            processed_data = document_data
            for processor in self.processors:
                processed_data = processor.process(processed_data)
            
            # Add pipeline metadata
            processed_data["pipeline_metadata"] = {
                "input_type": "raw_text",
                "processors_used": [p.__class__.__name__ for p in self.processors],
                "processing_complete": True
            }
            
            return processed_data
            
        except Exception as e:
            raise ProcessingError(f"Failed to process text: {str(e)}")
    
    def batch_process_documents(self, file_paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
        """
        Process multiple documents in batch.
        
        Args:
            file_paths: List of document paths
            
        Returns:
            List of processed document data
        """
        if not self._initialized:
            raise ProcessingError("Pipeline not initialized")
        
        results = []
        
        for file_path in file_paths:
            try:
                result = self.process_document(file_path)
                results.append(result)
            except Exception as e:
                # Log error but continue with other documents
                error_result = {
                    "error": str(e),
                    "file_path": str(file_path),
                    "pipeline_metadata": {
                        "file_path": str(file_path),
                        "processing_failed": True,
                        "error_message": str(e)
                    }
                }
                results.append(error_result)
        
        return results
    
    def get_supported_formats(self) -> List[str]:
        """Get all supported document formats."""
        if not self._initialized or not self.parser_registry:
            return []
        
        return self.parser_registry.get_supported_formats()
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the processing pipeline."""
        if not self._initialized:
            return {"initialized": False}
        
        processor_info = []
        for processor in self.processors:
            processor_info.append({
                "name": processor.__class__.__name__,
                "stats": processor.get_stats()
            })
        
        return {
            "initialized": self._initialized,
            "processors": processor_info,
            "parser_info": self.parser_registry.get_parser_info() if self.parser_registry else {},
            "supported_formats": self.get_supported_formats()
        }
    
    def _get_parser_name(self, file_path: Union[str, Path], parser_name: Optional[str] = None) -> str:
        """Get the name of the parser that will be used."""
        if parser_name:
            return parser_name
        
        if self.parser_registry:
            try:
                parser = self.parser_registry.get_parser(file_path)
                return parser.__class__.__name__
            except:
                return "unknown"
        
        return "unknown"
    
    def close(self) -> None:
        """Close the processing pipeline and clean up resources."""
        # Close all processors
        for processor in self.processors:
            processor.close()
        
        # Close parser registry
        if self.parser_registry:
            self.parser_registry.close()
        
        self.processors.clear()
        self.parser_registry = None
        self._initialized = False
