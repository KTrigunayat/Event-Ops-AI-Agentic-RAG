import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sqlalchemy import create_engine, Column, String, Integer, Text, DateTime, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid

from ..chunking.llm_chunker import Chunk

Base = declarative_base()

class DocumentChunk(Base):
    """SQLAlchemy model for storing document chunks and embeddings"""
    __tablename__ = "document_chunks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chunk_id = Column(String(255), unique=True, nullable=False, index=True)
    source = Column(String(255), nullable=False, index=True)
    content = Column(Text, nullable=False)
    start_index = Column(Integer, nullable=False)
    end_index = Column(Integer, nullable=False)
    embedding = Column(Float, nullable=False)  # Will be stored as array
    embedding_dimension = Column(Integer, nullable=False)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class EndeeDatabase:
    """Database manager for Endee vector storage"""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or os.getenv("DATABASE_URL", 
                                                      "postgresql://username:password@localhost:5432/endee_db")
        self.logger = logging.getLogger(__name__)
        
        try:
            self.engine = create_engine(self.database_url)
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            
            # Create tables
            Base.metadata.create_all(bind=self.engine)
            
            self.logger.info(f"Connected to database: {self.database_url}")
            
        except Exception as e:
            self.logger.error(f"Error connecting to database: {e}")
            raise
    
    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()
    
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
        session = self.get_session()
        
        try:
            for chunk, embedding in zip(chunks, embeddings):
                # Convert numpy array to list for storage
                embedding_list = embedding.tolist()
                
                # Check if chunk already exists
                existing_chunk = session.query(DocumentChunk).filter(
                    DocumentChunk.chunk_id == chunk.chunk_id
                ).first()
                
                if existing_chunk:
                    # Update existing chunk
                    existing_chunk.content = chunk.content
                    existing_chunk.start_index = chunk.start_index
                    existing_chunk.end_index = chunk.end_index
                    existing_chunk.embedding = embedding_list
                    existing_chunk.embedding_dimension = len(embedding_list)
                    existing_chunk.metadata = chunk.metadata
                    existing_chunk.updated_at = datetime.utcnow()
                    
                    stored_ids.append(str(existing_chunk.id))
                    self.logger.debug(f"Updated existing chunk: {chunk.chunk_id}")
                else:
                    # Create new chunk
                    db_chunk = DocumentChunk(
                        chunk_id=chunk.chunk_id,
                        source=chunk.source,
                        content=chunk.content,
                        start_index=chunk.start_index,
                        end_index=chunk.end_index,
                        embedding=embedding_list,
                        embedding_dimension=len(embedding_list),
                        metadata=chunk.metadata
                    )
                    
                    session.add(db_chunk)
                    session.flush()  # Get the ID without committing
                    stored_ids.append(str(db_chunk.id))
                    self.logger.debug(f"Stored new chunk: {chunk.chunk_id}")
            
            session.commit()
            self.logger.info(f"Successfully stored {len(stored_ids)} chunks")
            return stored_ids
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error storing chunks: {e}")
            raise
        finally:
            session.close()
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Retrieve a chunk by its ID"""
        session = self.get_session()
        try:
            chunk = session.query(DocumentChunk).filter(
                DocumentChunk.chunk_id == chunk_id
            ).first()
            return chunk
        except Exception as e:
            self.logger.error(f"Error retrieving chunk {chunk_id}: {e}")
            return None
        finally:
            session.close()
    
    def get_chunks_by_source(self, source: str) -> List[DocumentChunk]:
        """Retrieve all chunks from a specific source"""
        session = self.get_session()
        try:
            chunks = session.query(DocumentChunk).filter(
                DocumentChunk.source == source
            ).all()
            return chunks
        except Exception as e:
            self.logger.error(f"Error retrieving chunks from source {source}: {e}")
            return []
        finally:
            session.close()
    
    def search_similar_chunks(self, query_embedding: np.ndarray, limit: int = 10, 
                            source_filter: str = None) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using cosine similarity
        
        Args:
            query_embedding: Query embedding as numpy array
            limit: Maximum number of results
            source_filter: Optional source filter
            
        Returns:
            List of dictionaries with chunk info and similarity scores
        """
        session = self.get_session()
        
        try:
            # Get all chunks (or filtered by source)
            query_obj = session.query(DocumentChunk)
            
            if source_filter:
                query_obj = query_obj.filter(DocumentChunk.source == source_filter)
            
            chunks = query_obj.all()
            
            # Calculate similarities
            results = []
            query_embedding_norm = query_embedding / np.linalg.norm(query_embedding)
            
            for chunk in chunks:
                # Convert stored embedding back to numpy array
                chunk_embedding = np.array(chunk.embedding)
                chunk_embedding_norm = chunk_embedding / np.linalg.norm(chunk_embedding)
                
                # Calculate cosine similarity
                similarity = np.dot(query_embedding_norm, chunk_embedding_norm)
                
                results.append({
                    'chunk_id': chunk.chunk_id,
                    'source': chunk.source,
                    'content': chunk.content,
                    'similarity': float(similarity),
                    'metadata': chunk.metadata,
                    'created_at': chunk.created_at
                })
            
            # Sort by similarity and return top results
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:limit]
            
        except Exception as e:
            self.logger.error(f"Error searching similar chunks: {e}")
            return []
        finally:
            session.close()
    
    def delete_chunks_by_source(self, source: str) -> int:
        """Delete all chunks from a specific source"""
        session = self.get_session()
        try:
            deleted_count = session.query(DocumentChunk).filter(
                DocumentChunk.source == source
            ).delete()
            session.commit()
            self.logger.info(f"Deleted {deleted_count} chunks from source: {source}")
            return deleted_count
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error deleting chunks from source {source}: {e}")
            return 0
        finally:
            session.close()
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        session = self.get_session()
        try:
            total_chunks = session.query(DocumentChunk).count()
            total_sources = session.query(DocumentChunk.source).distinct().count()
            
            # Get source breakdown
            source_stats = session.query(
                DocumentChunk.source,
                session.query(DocumentChunk).filter(DocumentChunk.source == DocumentChunk.source).count()
            ).distinct().all()
            
            return {
                'total_chunks': total_chunks,
                'total_sources': total_sources,
                'source_breakdown': dict(source_stats)
            }
        except Exception as e:
            self.logger.error(f"Error getting database stats: {e}")
            return {}
        finally:
            session.close()
