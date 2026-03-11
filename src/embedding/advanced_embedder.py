import os
import logging
from typing import List, Union
import numpy as np
import openai
import google.generativeai as genai
from dataclasses import dataclass

from ..chunking.llm_chunker import Chunk
from ..chunking.semantic_chunker import SemanticChunk

@dataclass
class EmbeddingConfig:
    """Configuration for embedding models"""
    provider: str  # "openai" or "gemini"
    model_name: str
    api_key: str = None
    embedding_dimension: int = None

class AdvancedEmbedder:
    """Advanced embedding system supporting OpenAI and Gemini embeddings"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize based on provider
        if config.provider.lower() == "openai":
            self._init_openai()
        elif config.provider.lower() == "gemini":
            self._init_gemini()
        else:
            raise ValueError(f"Unsupported embedding provider: {config.provider}")
    
    def _init_openai(self):
        """Initialize OpenAI embeddings"""
        try:
            self.api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key not provided")
            
            openai.api_key = self.api_key
            self.client = openai.OpenAI(api_key=self.api_key)
            
            # Set model and dimensions
            self.model = self.config.model_name or "text-embedding-3-small"
            self.embedding_dim = self.config.embedding_dimension or 1536
            
            self.logger.info(f"Initialized OpenAI embeddings with model: {self.model}")
            
        except Exception as e:
            self.logger.error(f"Error initializing OpenAI embeddings: {e}")
            raise
    
    def _init_gemini(self):
        """Initialize Gemini embeddings"""
        try:
            self.api_key = self.config.api_key or os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError("Gemini API key not provided")
            
            genai.configure(api_key=self.api_key)
            
            # Set model and dimensions
            self.model = self.config.model_name or "text-embedding-004"
            self.embedding_dim = self.config.embedding_dimension or 768
            
            self.logger.info(f"Initialized Gemini embeddings with model: {self.model}")
            
        except Exception as e:
            self.logger.error(f"Error initializing Gemini embeddings: {e}")
            raise
    
    def embed_chunks(self, chunks: List[Union[Chunk, SemanticChunk]], batch_size: int = 100) -> List[np.ndarray]:
        """
        Generate embeddings for a list of chunks
        
        Args:
            chunks: List of Chunk or SemanticChunk objects
            batch_size: Batch size for processing
            
        Returns:
            List of numpy arrays representing embeddings
        """
        if not chunks:
            return []
        
        self.logger.info(f"Generating embeddings for {len(chunks)} chunks using {self.config.provider}")
        
        # Extract text content from chunks
        texts = []
        for chunk in chunks:
            if hasattr(chunk, 'content'):
                texts.append(chunk.content)
            else:
                texts.append(str(chunk))
        
        try:
            embeddings = []
            
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                if self.config.provider.lower() == "openai":
                    batch_embeddings = self._embed_openai_batch(batch_texts)
                elif self.config.provider.lower() == "gemini":
                    batch_embeddings = self._embed_gemini_batch(batch_texts)
                
                embeddings.extend(batch_embeddings)
                
                self.logger.debug(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            self.logger.info(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            raise
    
    def _embed_openai_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings using OpenAI"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            
            embeddings = [np.array(data.embedding, dtype=np.float32) for data in response.data]
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error in OpenAI embedding batch: {e}")
            raise
    
    def _embed_gemini_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings using Gemini"""
        try:
            embeddings = []
            
            for text in texts:
                result = genai.embed_content(
                    model=self.model,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(np.array(result['embedding'], dtype=np.float32))
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error in Gemini embedding batch: {e}")
            raise
    
    def embed_single_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array representing the embedding
        """
        try:
            if self.config.provider.lower() == "openai":
                response = self.client.embeddings.create(
                    model=self.model,
                    input=[text]
                )
                return np.array(response.data[0].embedding, dtype=np.float32)
            
            elif self.config.provider.lower() == "gemini":
                result = genai.embed_content(
                    model=self.model,
                    content=text,
                    task_type="retrieval_query"
                )
                return np.array(result['embedding'], dtype=np.float32)
                
        except Exception as e:
            self.logger.error(f"Error embedding single text: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self.embedding_dim
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Compute cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Error computing similarity: {e}")
            return 0.0
    
    def normalize_embeddings(self, embeddings: List[np.ndarray]) -> List[np.ndarray]:
        """Normalize embeddings to unit length"""
        normalized = []
        for embedding in embeddings:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                normalized.append(embedding / norm)
            else:
                normalized.append(embedding)
        return normalized
    
    def get_provider_info(self) -> dict:
        """Get information about the embedding provider"""
        return {
            "provider": self.config.provider,
            "model": self.model,
            "embedding_dimension": self.embedding_dim,
            "api_configured": bool(self.api_key)
        }

class EmbeddingFactory:
    """Factory for creating embedding instances"""
    
    @staticmethod
    def create_openai_embedder(api_key: str = None, model: str = "text-embedding-3-small") -> AdvancedEmbedder:
        """Create OpenAI embedder"""
        config = EmbeddingConfig(
            provider="openai",
            model_name=model,
            api_key=api_key,
            embedding_dimension=1536 if "small" in model else 3072
        )
        return AdvancedEmbedder(config)
    
    @staticmethod
    def create_gemini_embedder(api_key: str = None, model: str = "text-embedding-004") -> AdvancedEmbedder:
        """Create Gemini embedder"""
        config = EmbeddingConfig(
            provider="gemini",
            model_name=model,
            api_key=api_key,
            embedding_dimension=768
        )
        return AdvancedEmbedder(config)
    
    @staticmethod
    def create_from_env(provider: str = None) -> AdvancedEmbedder:
        """Create embedder from environment variables"""
        if not provider:
            provider = os.getenv("EMBEDDING_PROVIDER", "openai")
        
        if provider.lower() == "openai":
            model = os.getenv("OPENAI_MODEL", "text-embedding-3-small")
            return EmbeddingFactory.create_openai_embedder(model=model)
        elif provider.lower() == "gemini":
            model = os.getenv("GEMINI_MODEL", "text-embedding-004")
            return EmbeddingFactory.create_gemini_embedder(model=model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
