import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import requests
from datetime import datetime

from ..chunking.llm_chunker import Chunk

class EndeeVectorDatabase:
    """Endee Vector Database HTTP client for communicating with Endee server"""
    
    def __init__(self, database_path: str = None):
        """
        Initialize Endee Vector Database HTTP client
        
        Args:
            database_path: Deprecated parameter (kept for backward compatibility)
        """
        self.logger = logging.getLogger(__name__)
        
        # Load configuration from environment
        self.server_url = os.getenv("ENDEE_SERVER_URL", "http://localhost:8080")
        self.auth_token = os.getenv("ENDEE_AUTH_TOKEN", "")
        self.index_name = os.getenv("ENDEE_INDEX_NAME", "event_ops_vectors")
        self.dimension = int(os.getenv("ENDEE_DIMENSION", "384"))
        self.metric = os.getenv("ENDEE_METRIC", "cosine")
        
        # Initialize HTTP session
        self.session = requests.Session()
        
        # Add authentication header if token is provided
        if self.auth_token:
            self.session.headers.update({"Authorization": self.auth_token})
        
        # Verify Endee server is reachable
        self.health_check()
        
        # Create index if it doesn't exist (idempotent)
        self._create_index()
        
        self.logger.info(f"Initialized Endee Vector Database client connected to: {self.server_url}")
    
    def health_check(self):
        """Check if Endee server is reachable"""
        try:
            response = self.session.get(f"{self.server_url}/api/v1/health", timeout=5)
            response.raise_for_status()
            self.logger.info("Endee server health check passed")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Endee server health check failed: {e}")
            raise ConnectionError(f"Cannot connect to Endee server at {self.server_url}: {e}")
    
    def _create_index(self):
        """Create vector index (idempotent operation)"""
        try:
            payload = {
                "index_name": self.index_name,
                "dimension": self.dimension,
                "metric": self.metric
            }
            response = self.session.post(
                f"{self.server_url}/api/v1/index/create",
                json=payload,
                timeout=10
            )
            
            # Index creation is idempotent - 200 or 409 (already exists) are both OK
            if response.status_code in [200, 201, 409]:
                self.logger.info(f"Index '{self.index_name}' is ready")
            else:
                response.raise_for_status()
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error creating index: {e}")
            raise
    
    def store_chunks_with_embeddings(self, chunks: List[Chunk], embeddings: List[np.ndarray]) -> List[str]:
        """
        Store chunks and their embeddings in the database
        
        Args:
            chunks: List of Chunk objects
            embeddings: List of numpy arrays representing embeddings
            
        Returns:
            List of stored chunk IDs
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match")
        
        stored_ids = []
        
        try:
            # Transform chunks and embeddings to Endee API format
            vectors = []
            for chunk, embedding in zip(chunks, embeddings):
                chunk_id = chunk.chunk_id
                
                # Convert numpy array to list
                embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
                
                # Prepare metadata including chunk content and attributes
                metadata = {
                    'content': chunk.content,
                    'source': chunk.source,
                    'start_index': chunk.start_index,
                    'end_index': chunk.end_index,
                    'created_at': datetime.now().isoformat()
                }
                
                # Merge with chunk's existing metadata
                if chunk.metadata:
                    metadata.update(chunk.metadata)
                
                vectors.append({
                    'id': chunk_id,
                    'values': embedding_list,
                    'metadata': metadata
                })
                
                stored_ids.append(chunk_id)
            
            # Call Endee API to insert vectors
            payload = {'vectors': vectors}
            response = self.session.post(
                f"{self.server_url}/api/v1/index/{self.index_name}/vector/insert",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            self.logger.info(f"Successfully stored {len(stored_ids)} chunks")
            return stored_ids
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error storing chunks: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error storing chunks: {e}")
            raise
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a chunk by its ID"""
        try:
            payload = {'id': chunk_id}
            response = self.session.post(
                f"{self.server_url}/api/v1/index/{self.index_name}/vector/get",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 404:
                return None
            
            response.raise_for_status()
            data = response.json()
            
            # Transform Endee response to expected format
            if data and 'vector' in data:
                vector_data = data['vector']
                chunk_data = vector_data.get('metadata', {}).copy()
                chunk_data['embedding'] = vector_data.get('values')
                chunk_data['chunk_id'] = vector_data.get('id', chunk_id)
                return chunk_data
            
            return None
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error retrieving chunk {chunk_id}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error retrieving chunk {chunk_id}: {e}")
            return None
    
    def get_chunks_by_source(self, source: str) -> List[Dict[str, Any]]:
        """Retrieve all chunks from a specific source"""
        try:
            # Use search with metadata filter to find chunks by source
            # Create a dummy embedding for filtering (won't affect results with proper filter)
            dummy_embedding = [0.0] * self.dimension
            
            payload = {
                'vector': dummy_embedding,
                'top_k': 10000,  # Large number to get all matching chunks
                'filter': {'source': source}
            }
            
            response = self.session.post(
                f"{self.server_url}/api/v1/index/{self.index_name}/search",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            chunks = []
            if data and 'results' in data:
                for result in data['results']:
                    chunk_data = result.get('metadata', {}).copy()
                    chunk_data['chunk_id'] = result.get('id')
                    chunk_data['embedding'] = result.get('values')
                    chunks.append(chunk_data)
            
            return chunks
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error retrieving chunks from source {source}: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Error retrieving chunks from source {source}: {e}")
            return []
    
    def search_similar_chunks(self, query_embedding: np.ndarray, limit: int = 10, 
                            source_filter: str = None, similarity_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using Endee server
        
        Args:
            query_embedding: Query embedding as numpy array
            limit: Maximum number of results
            source_filter: Optional source filter
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of dictionaries with chunk info and similarity scores
        """
        try:
            # Convert numpy array to list
            query_vector = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
            
            # Prepare search payload
            payload = {
                'vector': query_vector,
                'top_k': limit
            }
            
            # Add metadata filter if source_filter is provided
            if source_filter:
                payload['filter'] = {'source': source_filter}
            
            # Call Endee search endpoint
            response = self.session.post(
                f"{self.server_url}/api/v1/index/{self.index_name}/search",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            results = []
            if data and 'results' in data:
                for result in data['results']:
                    # Get similarity score (Endee returns distance, convert to similarity)
                    similarity = result.get('score', 0.0)
                    
                    # Apply threshold
                    if similarity >= similarity_threshold:
                        chunk_data = result.get('metadata', {}).copy()
                        chunk_data['chunk_id'] = result.get('id')
                        chunk_data['similarity'] = float(similarity)
                        results.append(chunk_data)
            
            return results
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error searching similar chunks: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Error searching similar chunks: {e}")
            return []
    
    def delete_chunks_by_source(self, source: str) -> int:
        """Delete all chunks from a specific source"""
        try:
            # First, retrieve all chunk IDs from this source
            chunks = self.get_chunks_by_source(source)
            deleted_count = 0
            
            # Delete each chunk individually
            for chunk in chunks:
                chunk_id = chunk.get('chunk_id')
                if chunk_id:
                    try:
                        response = self.session.delete(
                            f"{self.server_url}/api/v1/index/{self.index_name}/vector/{chunk_id}/delete",
                            timeout=10
                        )
                        
                        if response.status_code in [200, 204, 404]:
                            deleted_count += 1
                        else:
                            response.raise_for_status()
                            
                    except requests.exceptions.RequestException as e:
                        self.logger.warning(f"Error deleting chunk {chunk_id}: {e}")
            
            self.logger.info(f"Deleted {deleted_count} chunks from source: {source}")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Error deleting chunks from source {source}: {e}")
            return 0
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            # Call Endee index info endpoint
            response = self.session.get(
                f"{self.server_url}/api/v1/index/{self.index_name}/info",
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            # Transform Endee response to expected stats format
            total_chunks = data.get('vector_count', 0)
            
            # Get source breakdown by searching all vectors
            # Note: This is a simplified approach - in production, you might want to cache this
            source_breakdown = {}
            
            return {
                'total_chunks': total_chunks,
                'total_sources': len(source_breakdown),
                'source_breakdown': source_breakdown,
                'embedding_dimension': data.get('dimension', self.dimension),
                'index_name': self.index_name,
                'metric': data.get('metric', self.metric)
            }
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error getting database stats: {e}")
            return {}
        except Exception as e:
            self.logger.error(f"Error getting database stats: {e}")
            return {}
    
    def rebuild_index(self):
        """Rebuild the search index (delegated to Endee server)"""
        try:
            self.logger.info("Index rebuilding is managed by Endee server")
            # Endee server handles index optimization internally
            # This method is kept for backward compatibility
            
        except Exception as e:
            self.logger.error(f"Error rebuilding index: {e}")
    
    def export_to_format(self, format_type: str = "json", output_path: str = None) -> str:
        """Export database to different formats"""
        try:
            if format_type.lower() == "json":
                # Get all vectors from Endee server
                # Note: This is a simplified implementation
                # In production, you might want to implement pagination
                
                stats = self.get_database_stats()
                
                data = {
                    'index_name': self.index_name,
                    'dimension': self.dimension,
                    'metric': self.metric,
                    'total_chunks': stats.get('total_chunks', 0),
                    'exported_at': datetime.now().isoformat()
                }
                
                output_path = output_path or f"{self.index_name}_export.json"
                
                import json
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                return output_path
            
            # Add other export formats as needed
            raise ValueError(f"Unsupported export format: {format_type}")
            
        except Exception as e:
            self.logger.error(f"Error exporting database: {e}")
            raise
