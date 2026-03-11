import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from .chunking.llm_chunker import OllamaChunker
from .embedding.distilbert_embedder import DistilBERTEmbedder
from .database.endee_vector_db import EndeeVectorDatabase

class IngestionPipeline:
    """Main ingestion pipeline for processing documents and storing in Endee database"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the ingestion pipeline
        
        Args:
            config: Configuration dictionary with pipeline settings
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize all pipeline components"""
        try:
            # Initialize chunker
            self.chunker = OllamaChunker(
                base_url=self.config.get("ollama_base_url"),
                model=self.config.get("ollama_model", "tinyllama")
            )
            
            # Initialize embedder
            self.embedder = DistilBERTEmbedder(
                model_name=self.config.get("embedding_model", "distilbert-base-uncased")
            )
            
            # Initialize database
            self.database = EndeeVectorDatabase(
                database_path=self.config.get("endeedb_path")
            )
            
            self.logger.info("All pipeline components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing pipeline components: {e}")
            raise
    
    def process_text(self, text: str, source: str, **kwargs) -> Dict[str, Any]:
        """
        Process text through the complete ingestion pipeline
        
        Args:
            text: Text to process
            source: Source identifier
            **kwargs: Additional parameters for chunking
            
        Returns:
            Dictionary with processing results
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        self.logger.info(f"Starting ingestion pipeline for source: {source}")
        
        try:
            # Step 1: Chunk the text
            chunk_size = kwargs.get("chunk_size", int(os.getenv("CHUNK_SIZE", 1000)))
            overlap = kwargs.get("overlap", int(os.getenv("CHUNK_OVERLAP", 200)))
            
            chunks = self.chunker.chunk_text(
                text=text,
                source=source,
                chunk_size=chunk_size,
                overlap=overlap
            )
            
            if not chunks:
                raise ValueError("No chunks were created from the text")
            
            self.logger.info(f"Created {len(chunks)} chunks")
            
            # Step 2: Generate embeddings
            batch_size = kwargs.get("batch_size", 32)
            embeddings = self.embedder.embed_chunks(chunks, batch_size=batch_size)
            
            if len(embeddings) != len(chunks):
                raise ValueError("Number of embeddings doesn't match number of chunks")
            
            self.logger.info(f"Generated {len(embeddings)} embeddings")
            
            # Step 3: Store in database
            stored_ids = self.database.store_chunks_with_embeddings(chunks, embeddings)
            
            self.logger.info(f"Stored {len(stored_ids)} chunks in database")
            
            return {
                "status": "success",
                "source": source,
                "chunks_created": len(chunks),
                "embeddings_generated": len(embeddings),
                "chunks_stored": len(stored_ids),
                "stored_ids": stored_ids,
                "embedding_dimension": self.embedder.get_embedding_dimension(),
                "chunk_ids": [chunk.chunk_id for chunk in chunks]
            }
            
        except Exception as e:
            self.logger.error(f"Error in ingestion pipeline: {e}")
            return {
                "status": "error",
                "source": source,
                "error": str(e)
            }
    
    def process_file(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Process a file through the ingestion pipeline
        
        Args:
            file_path: Path to the file to process
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with processing results
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Use filename as source if not provided
        source = kwargs.get("source", file_path.name)
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            self.logger.info(f"Read file: {file_path} ({len(text)} characters)")
            
            # Process the text
            return self.process_text(text, source, **kwargs)
            
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}")
            return {
                "status": "error",
                "source": source,
                "file_path": str(file_path),
                "error": str(e)
            }
    
    def process_directory(self, directory_path: str, file_pattern: str = "*.txt", **kwargs) -> Dict[str, Any]:
        """
        Process all files in a directory
        
        Args:
            directory_path: Path to directory
            file_pattern: File pattern to match (e.g., "*.txt", "*.md")
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with processing results
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        files = list(directory_path.glob(file_pattern))
        
        if not files:
            return {
                "status": "success",
                "directory": str(directory_path),
                "files_processed": 0,
                "message": f"No files found matching pattern: {file_pattern}"
            }
        
        results = []
        total_chunks = 0
        total_stored = 0
        errors = []
        
        for file_path in files:
            try:
                result = self.process_file(file_path, **kwargs)
                results.append(result)
                
                if result["status"] == "success":
                    total_chunks += result.get("chunks_created", 0)
                    total_stored += result.get("chunks_stored", 0)
                else:
                    errors.append(f"{file_path.name}: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                error_msg = f"{file_path.name}: {str(e)}"
                errors.append(error_msg)
                self.logger.error(f"Error processing file {file_path}: {e}")
        
        return {
            "status": "success" if not errors else "partial_success",
            "directory": str(directory_path),
            "file_pattern": file_pattern,
            "files_found": len(files),
            "files_processed": len(results),
            "total_chunks_created": total_chunks,
            "total_chunks_stored": total_stored,
            "errors": errors,
            "results": results
        }
    
    def search_similar_content(self, query: str, limit: int = 10, source_filter: str = None) -> List[Dict[str, Any]]:
        """
        Search for similar content in the database
        
        Args:
            query: Query text
            limit: Maximum number of results
            source_filter: Optional source filter
            
        Returns:
            List of similar chunks with similarity scores
        """
        try:
            # Generate embedding for query
            query_embedding = self.embedder.embed_single_text(query)
            
            # Search in database
            results = self.database.search_similar_chunks(
                query_embedding=query_embedding,
                limit=limit,
                source_filter=source_filter
            )
            
            self.logger.info(f"Found {len(results)} similar chunks for query: {query[:50]}...")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching similar content: {e}")
            return []
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline and database statistics"""
        try:
            db_stats = self.database.get_database_stats()
            
            return {
                "pipeline_status": "ready",
                "chunker_model": self.chunker.model,
                "embedding_model": self.embedder.model_name,
                "embedding_dimension": self.embedder.get_embedding_dimension(),
                "database_stats": db_stats
            }
            
        except Exception as e:
            self.logger.error(f"Error getting pipeline stats: {e}")
            return {
                "pipeline_status": "error",
                "error": str(e)
            }
