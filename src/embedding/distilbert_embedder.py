import os
import logging
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
import torch
from .llm_chunker import Chunk

class DistilBERTEmbedder:
    """DistilBERT-based text embedding generator"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            self.logger.info(f"Loaded DistilBERT model: {model_name} on {self.device}")
        except Exception as e:
            self.logger.error(f"Error loading model {model_name}: {e}")
            raise
    
    def embed_chunks(self, chunks: List[Chunk], batch_size: int = 32) -> List[np.ndarray]:
        """
        Convert chunks to vector embeddings using DistilBERT
        
        Args:
            chunks: List of Chunk objects to embed
            batch_size: Batch size for processing
            
        Returns:
            List of numpy arrays representing embeddings
        """
        if not chunks:
            return []
        
        self.logger.info(f"Embedding {len(chunks)} chunks using DistilBERT")
        
        # Extract text content from chunks
        texts = [chunk.content for chunk in chunks]
        
        try:
            # Generate embeddings in batches
            embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                with torch.no_grad():
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        batch_size=batch_size,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )
                    embeddings.extend(batch_embeddings)
                
                self.logger.debug(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            self.logger.info(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            raise
    
    def embed_single_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array representing the embedding
        """
        try:
            with torch.no_grad():
                embedding = self.model.encode(
                    text,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
            return embedding
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
            # Compute cosine similarity
            similarity = np.dot(embedding1, embedding2)
            return float(similarity)
        except Exception as e:
            self.logger.error(f"Error computing similarity: {e}")
            return 0.0
