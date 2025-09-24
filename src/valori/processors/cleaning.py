"""
Text cleaning processor for the Vectara vector database.
"""

import re
import html
from typing import Any, Dict, List, Optional
import unicodedata

from .base import DocumentProcessor
from ..exceptions import ProcessingError


class CleaningProcessor(DocumentProcessor):
    """
    Processor for cleaning and normalizing text data.
    
    Handles various text cleaning tasks including HTML decoding,
    whitespace normalization, special character removal, and more.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize cleaning processor."""
        super().__init__(config)
        self.remove_html = config.get("remove_html", True)
        self.normalize_whitespace = config.get("normalize_whitespace", True)
        self.remove_special_chars = config.get("remove_special_chars", False)
        self.special_chars_pattern = config.get("special_chars_pattern", r'[^\w\s\-.,!?;:]')
        self.normalize_unicode = config.get("normalize_unicode", True)
        self.remove_extra_spaces = config.get("remove_extra_spaces", True)
        self.lowercase = config.get("lowercase", False)
        self.remove_stopwords = config.get("remove_stopwords", False)
        self.stopwords_language = config.get("stopwords_language", "english")
        self.min_word_length = config.get("min_word_length", 1)
        self.max_word_length = config.get("max_word_length", 50)
        self._documents_processed = 0
        self._total_characters_before = 0
        self._total_characters_after = 0
    
    def initialize(self) -> None:
        """Initialize the cleaning processor."""
        # Load stopwords if needed
        self._stopwords = set()
        if self.remove_stopwords:
            try:
                import nltk
                from nltk.corpus import stopwords
                self._stopwords = set(stopwords.words(self.stopwords_language))
            except ImportError:
                # Fallback to basic English stopwords
                self._stopwords = {
                    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
                    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
                    'to', 'was', 'will', 'with'
                }
        
        self._initialized = True
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process document data and clean text.
        
        Args:
            data: Document data containing 'text' and optionally 'chunks'
            
        Returns:
            Document data with cleaned text and chunks
        """
        if not self._initialized:
            raise ProcessingError("Processor not initialized")
        
        result_data = data.copy()
        
        # Clean main text
        if "text" in data:
            original_text = data["text"]
            self._total_characters_before += len(original_text)
            
            cleaned_text = self._clean_text(original_text)
            self._total_characters_after += len(cleaned_text)
            
            result_data["text"] = cleaned_text
        
        # Clean chunks if they exist
        if "chunks" in data:
            cleaned_chunks = []
            for chunk in data["chunks"]:
                cleaned_chunk = chunk.copy()
                if "text" in chunk:
                    cleaned_chunk["text"] = self._clean_text(chunk["text"])
                cleaned_chunks.append(cleaned_chunk)
            
            result_data["chunks"] = cleaned_chunks
        
        self._documents_processed += 1
        return result_data
    
    def _clean_text(self, text: str) -> str:
        """Clean a single text string."""
        if not text:
            return text
        
        # Remove HTML tags and decode HTML entities
        if self.remove_html:
            text = html.unescape(text)
            text = re.sub(r'<[^>]+>', '', text)
        
        # Normalize Unicode characters
        if self.normalize_unicode:
            text = unicodedata.normalize('NFKD', text)
        
        # Remove special characters
        if self.remove_special_chars:
            text = re.sub(self.special_chars_pattern, ' ', text)
        
        # Normalize whitespace
        if self.normalize_whitespace:
            # Replace multiple whitespace characters with single space
            text = re.sub(r'\s+', ' ', text)
            # Remove leading/trailing whitespace
            text = text.strip()
        
        # Remove extra spaces
        if self.remove_extra_spaces:
            text = re.sub(r' +', ' ', text)
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove stopwords
        if self.remove_stopwords and self._stopwords:
            words = text.split()
            filtered_words = [
                word for word in words 
                if word.lower() not in self._stopwords and 
                   self.min_word_length <= len(word) <= self.max_word_length
            ]
            text = ' '.join(filtered_words)
        
        return text
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cleaning processor statistics."""
        compression_ratio = 0
        if self._total_characters_before > 0:
            compression_ratio = self._total_characters_after / self._total_characters_before
        
        return {
            "processor_type": "cleaning",
            "documents_processed": self._documents_processed,
            "total_characters_before": self._total_characters_before,
            "total_characters_after": self._total_characters_after,
            "compression_ratio": compression_ratio,
            "characters_removed": self._total_characters_before - self._total_characters_after,
            "remove_html": self.remove_html,
            "normalize_whitespace": self.normalize_whitespace,
            "remove_special_chars": self.remove_special_chars,
            "normalize_unicode": self.normalize_unicode,
            "lowercase": self.lowercase,
            "remove_stopwords": self.remove_stopwords,
            "initialized": self._initialized,
        }
