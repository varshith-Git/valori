"""
Embedding processor for the Vectara vector database.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union
import hashlib

from .base import DocumentProcessor
from ..exceptions import ProcessingError


class EmbeddingProcessor(DocumentProcessor):
    """
    Processor for generating embeddings from text data.
    
    Supports various embedding models and provides caching and
    batch processing capabilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize embedding processor."""
        super().__init__(config)
        self.model_name = config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        self.model_type = config.get("model_type", "sentence_transformers")  # sentence_transformers, openai, huggingface
        self.batch_size = config.get("batch_size", 32)
        self.max_length = config.get("max_length", 512)
        self.cache_embeddings = config.get("cache_embeddings", True)
        self.normalize_embeddings = config.get("normalize_embeddings", True)
        self.device = config.get("device", "cpu")  # cpu, cuda, auto
        self._model = None
        self._cache = {}
        self._texts_processed = 0
        self._embeddings_generated = 0
        self._cache_hits = 0
    
    def initialize(self) -> None:
        """Initialize the embedding processor."""
        try:
            if self.model_type == "sentence_transformers":
                self._initialize_sentence_transformers()
            elif self.model_type == "openai":
                self._initialize_openai()
            elif self.model_type == "huggingface":
                self._initialize_huggingface()
            else:
                raise ProcessingError(f"Unsupported model type: {self.model_type}")
            
            self._initialized = True
        except Exception as e:
            raise ProcessingError(f"Failed to initialize embedding processor: {str(e)}")
    
    def _initialize_sentence_transformers(self) -> None:
        """Initialize SentenceTransformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            
            # Set device
            if self.device == "auto":
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self._model = self._model.to(self.device)
            
        except ImportError:
            raise ProcessingError("sentence-transformers not installed. Install with: pip install sentence-transformers")
    
    def _initialize_openai(self) -> None:
        """Initialize OpenAI embeddings."""
        try:
            import openai
            self._model = openai
            # API key should be set in environment or config
        except ImportError:
            raise ProcessingError("openai not installed. Install with: pip install openai")
    
    def _initialize_huggingface(self) -> None:
        """Initialize HuggingFace transformers model."""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self._model = self._model.to(self.device)
            
        except ImportError:
            raise ProcessingError("transformers not installed. Install with: pip install transformers")
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process document data and generate embeddings.
        
        Args:
            data: Document data containing 'text' and optionally 'chunks'
            
        Returns:
            Document data with embeddings
        """
        if not self._initialized:
            raise ProcessingError("Processor not initialized")
        
        result_data = data.copy()
        
        # Generate embedding for main text
        if "text" in data:
            text_embedding = self._generate_embedding(data["text"])
            result_data["embedding"] = text_embedding
        
        # Generate embeddings for chunks
        if "chunks" in data:
            chunk_embeddings = []
            for chunk in data["chunks"]:
                if "text" in chunk:
                    chunk_embedding = self._generate_embedding(chunk["text"])
                    chunk_embeddings.append(chunk_embedding)
            
            result_data["chunk_embeddings"] = chunk_embeddings
        
        return result_data
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        if not text or not text.strip():
            # Return zero vector for empty text
            return np.zeros(384)  # Default dimension
        
        # Check cache
        if self.cache_embeddings:
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                self._cache_hits += 1
                return self._cache[cache_key]
        
        # Generate embedding
        if self.model_type == "sentence_transformers":
            embedding = self._generate_sentence_transformer_embedding(text)
        elif self.model_type == "openai":
            embedding = self._generate_openai_embedding(text)
        elif self.model_type == "huggingface":
            embedding = self._generate_huggingface_embedding(text)
        else:
            raise ProcessingError(f"Unknown model type: {self.model_type}")
        
        # Normalize embedding
        if self.normalize_embeddings:
            embedding = embedding / np.linalg.norm(embedding)
        
        # Cache embedding
        if self.cache_embeddings:
            cache_key = self._get_cache_key(text)
            self._cache[cache_key] = embedding
        
        self._texts_processed += 1
        self._embeddings_generated += 1
        
        return embedding
    
    def _generate_sentence_transformer_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using SentenceTransformers."""
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding
    
    def _generate_openai_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using OpenAI API."""
        try:
            response = self._model.Embedding.create(
                input=text,
                model="text-embedding-ada-002"
            )
            embedding = np.array(response['data'][0]['embedding'])
            return embedding
        except Exception as e:
            raise ProcessingError(f"OpenAI embedding generation failed: {str(e)}")
    
    def _generate_huggingface_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using HuggingFace transformers."""
        import torch
        
        # Tokenize text
        inputs = self._tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.max_length,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embedding
        with torch.no_grad():
            outputs = self._model(**inputs)
            # Use mean pooling of last hidden states
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        
        return embedding
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        # Use hash of text + model name for cache key
        content = f"{text}_{self.model_name}_{self.model_type}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def batch_process(self, texts: List[str]) -> List[np.ndarray]:
        """
        Process multiple texts in batches for efficiency.
        
        Args:
            texts: List of texts to process
            
        Returns:
            List of embeddings
        """
        if not self._initialized:
            raise ProcessingError("Processor not initialized")
        
        embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            if self.model_type == "sentence_transformers":
                batch_embeddings = self._model.encode(batch_texts, convert_to_numpy=True)
                if self.normalize_embeddings:
                    batch_embeddings = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
            else:
                # Process individually for other model types
                batch_embeddings = []
                for text in batch_texts:
                    embedding = self._generate_embedding(text)
                    batch_embeddings.append(embedding)
                batch_embeddings = np.array(batch_embeddings)
            
            embeddings.extend(batch_embeddings)
        
        self._texts_processed += len(texts)
        self._embeddings_generated += len(texts)
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this processor."""
        if not self._initialized:
            return 0
        
        # Generate a test embedding to determine dimension
        test_embedding = self._generate_embedding("test")
        return len(test_embedding)
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
        self._cache_hits = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get embedding processor statistics."""
        cache_hit_rate = 0
        if self._embeddings_generated > 0:
            cache_hit_rate = self._cache_hits / self._embeddings_generated
        
        return {
            "processor_type": "embedding",
            "model_name": self.model_name,
            "model_type": self.model_type,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "device": self.device,
            "normalize_embeddings": self.normalize_embeddings,
            "cache_embeddings": self.cache_embeddings,
            "texts_processed": self._texts_processed,
            "embeddings_generated": self._embeddings_generated,
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "embedding_dimension": self.get_embedding_dimension(),
            "initialized": self._initialized,
        }
