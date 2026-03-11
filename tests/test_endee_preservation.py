"""
Preservation Property Tests for Endee Compliance Fix

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**

This test suite captures the baseline behavior of the Event-Ops AI functionality
using the UNFIXED JSON-based implementation. These tests establish the behavior
that MUST be preserved after replacing JSON storage with HTTP API integration.

IMPORTANT: These tests are run on UNFIXED code and should PASS, confirming
the baseline behavior to preserve. After the fix, these same tests should
continue to PASS, proving no regressions occurred.

Property 2: Preservation - Event-Ops Functional Equivalence
For any Event-Ops AI functionality that uses the EndeeVectorDatabase class
(RAG queries, document ingestion, semantic search), the fixed implementation
SHALL produce the same results and behavior as the original implementation.
"""

import pytest
import numpy as np
import tempfile
import os
import shutil
from pathlib import Path
from hypothesis import given, strategies as st, settings, assume
from hypothesis import HealthCheck

# Import the classes we need to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.endee_vector_db import EndeeVectorDatabase
from src.chunking.llm_chunker import Chunk


class TestEndeePreservation:
    """
    Preservation Tests: Verify Event-Ops AI functionality behavior patterns
    that must be preserved after the fix.
    
    These tests observe and document the current behavior on UNFIXED code.
    """
    
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path for testing"""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_endee_db")
        yield db_path
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def vector_db(self, temp_db_path):
        """Create a fresh EndeeVectorDatabase instance"""
        return EndeeVectorDatabase(database_path=temp_db_path)
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing"""
        chunks = [
            Chunk(
                content="This is the first test document about machine learning.",
                chunk_id="test_doc_1_chunk_0",
                source="test_doc_1.txt",
                start_index=0,
                end_index=56,
                metadata={"category": "ml", "importance": "high"}
            ),
            Chunk(
                content="This is the second test document about artificial intelligence.",
                chunk_id="test_doc_1_chunk_1",
                source="test_doc_1.txt",
                start_index=57,
                end_index=120,
                metadata={"category": "ai", "importance": "medium"}
            ),
            Chunk(
                content="This is a document from a different source about data science.",
                chunk_id="test_doc_2_chunk_0",
                source="test_doc_2.txt",
                start_index=0,
                end_index=63,
                metadata={"category": "ds", "importance": "high"}
            )
        ]
        return chunks
    
    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings (384-dimensional for DistilBERT)"""
        np.random.seed(42)
        return [
            np.random.randn(384).astype(np.float32),
            np.random.randn(384).astype(np.float32),
            np.random.randn(384).astype(np.float32)
        ]
    
    def test_document_ingestion_returns_chunk_ids(self, vector_db, sample_chunks, sample_embeddings):
        """
        Property 2: Preservation - Document Ingestion
        
        OBSERVED BEHAVIOR: When documents are stored with embeddings,
        the system returns a list of chunk IDs (strings) that can be
        used to retrieve the chunks later.
        
        This behavior MUST be preserved after the fix.
        """
        # Store chunks with embeddings
        chunk_ids = vector_db.store_chunks_with_embeddings(sample_chunks, sample_embeddings)
        
        # Verify return type and structure
        assert isinstance(chunk_ids, list), "store_chunks_with_embeddings should return a list"
        assert len(chunk_ids) == len(sample_chunks), "Should return one ID per chunk"
        assert all(isinstance(cid, str) for cid in chunk_ids), "All chunk IDs should be strings"
        assert len(chunk_ids) == len(set(chunk_ids)), "All chunk IDs should be unique"
    
    def test_semantic_search_returns_valid_structure(self, vector_db, sample_chunks, sample_embeddings):
        """
        Property 2: Preservation - Semantic Search Structure
        
        OBSERVED BEHAVIOR: When querying with embeddings, the system returns
        a list of dictionaries containing chunk data with similarity scores,
        sorted by relevance.
        
        This behavior MUST be preserved after the fix.
        """
        # Store chunks first
        vector_db.store_chunks_with_embeddings(sample_chunks, sample_embeddings)
        
        # Create a query embedding (similar to first embedding)
        query_embedding = sample_embeddings[0] + np.random.randn(384) * 0.1
        
        # Search for similar chunks
        results = vector_db.search_similar_chunks(query_embedding, limit=5)
        
        # Verify return structure
        assert isinstance(results, list), "search_similar_chunks should return a list"
        assert len(results) > 0, "Should return at least one result"
        
        # Verify each result has required fields
        for result in results:
            assert isinstance(result, dict), "Each result should be a dictionary"
            assert 'chunk_id' in result, "Result should contain chunk_id"
            assert 'content' in result, "Result should contain content"
            assert 'similarity' in result, "Result should contain similarity score"
            assert 'source' in result, "Result should contain source"
            assert 'metadata' in result, "Result should contain metadata"
            
            # Verify data types
            assert isinstance(result['chunk_id'], str), "chunk_id should be string"
            assert isinstance(result['content'], str), "content should be string"
            assert isinstance(result['similarity'], (float, np.floating)), "similarity should be float"
            assert isinstance(result['source'], str), "source should be string"
            assert isinstance(result['metadata'], dict), "metadata should be dict"
        
        # Verify results are sorted by similarity (descending)
        similarities = [r['similarity'] for r in results]
        assert similarities == sorted(similarities, reverse=True), "Results should be sorted by similarity"
    
    def test_chunk_retrieval_by_id_returns_complete_data(self, vector_db, sample_chunks, sample_embeddings):
        """
        Property 2: Preservation - Chunk Retrieval
        
        OBSERVED BEHAVIOR: When retrieving a chunk by ID, the system returns
        a dictionary containing the chunk content, embedding, metadata, and
        source information.
        
        This behavior MUST be preserved after the fix.
        """
        # Store chunks first
        chunk_ids = vector_db.store_chunks_with_embeddings(sample_chunks, sample_embeddings)
        
        # Retrieve first chunk
        chunk_data = vector_db.get_chunk_by_id(chunk_ids[0])
        
        # Verify return structure
        assert chunk_data is not None, "get_chunk_by_id should return data for valid ID"
        assert isinstance(chunk_data, dict), "Chunk data should be a dictionary"
        
        # Verify required fields
        assert 'chunk_id' in chunk_data, "Should contain chunk_id"
        assert 'content' in chunk_data, "Should contain content"
        assert 'source' in chunk_data, "Should contain source"
        assert 'embedding' in chunk_data, "Should contain embedding"
        assert 'metadata' in chunk_data, "Should contain metadata"
        assert 'start_index' in chunk_data, "Should contain start_index"
        assert 'end_index' in chunk_data, "Should contain end_index"
        
        # Verify data types
        assert isinstance(chunk_data['chunk_id'], str), "chunk_id should be string"
        assert isinstance(chunk_data['content'], str), "content should be string"
        assert isinstance(chunk_data['source'], str), "source should be string"
        assert isinstance(chunk_data['metadata'], dict), "metadata should be dict"
        
        # Verify embedding is present and correct dimension
        embedding = chunk_data['embedding']
        assert embedding is not None, "Embedding should not be None"
        if isinstance(embedding, np.ndarray):
            assert embedding.shape[0] == 384, "Embedding should be 384-dimensional"
        elif isinstance(embedding, list):
            assert len(embedding) == 384, "Embedding should be 384-dimensional"
    
    def test_chunk_retrieval_nonexistent_id_returns_none(self, vector_db):
        """
        Property 2: Preservation - Nonexistent Chunk Handling
        
        OBSERVED BEHAVIOR: When retrieving a chunk with a nonexistent ID,
        the system returns None.
        
        This behavior MUST be preserved after the fix.
        """
        result = vector_db.get_chunk_by_id("nonexistent_id_12345")
        assert result is None, "Should return None for nonexistent chunk ID"
    
    def test_database_stats_returns_valid_format(self, vector_db, sample_chunks, sample_embeddings):
        """
        Property 2: Preservation - Database Stats
        
        OBSERVED BEHAVIOR: When requesting database statistics, the system
        returns a dictionary with total chunks, sources, and embedding dimension.
        
        This behavior MUST be preserved after the fix.
        """
        # Store chunks first
        vector_db.store_chunks_with_embeddings(sample_chunks, sample_embeddings)
        
        # Get stats
        stats = vector_db.get_database_stats()
        
        # Verify return structure
        assert isinstance(stats, dict), "get_database_stats should return a dictionary"
        
        # Verify required fields
        assert 'total_chunks' in stats, "Stats should contain total_chunks"
        assert 'total_sources' in stats, "Stats should contain total_sources"
        assert 'source_breakdown' in stats, "Stats should contain source_breakdown"
        assert 'embedding_dimension' in stats, "Stats should contain embedding_dimension"
        
        # Verify data types and values
        assert isinstance(stats['total_chunks'], int), "total_chunks should be int"
        assert isinstance(stats['total_sources'], int), "total_sources should be int"
        assert isinstance(stats['source_breakdown'], dict), "source_breakdown should be dict"
        assert stats['total_chunks'] == 3, "Should have 3 chunks"
        assert stats['total_sources'] == 2, "Should have 2 sources"
        assert stats['embedding_dimension'] == 384, "Should have 384-dimensional embeddings"
    
    def test_search_with_source_filter(self, vector_db, sample_chunks, sample_embeddings):
        """
        Property 2: Preservation - Source Filtering
        
        OBSERVED BEHAVIOR: When searching with a source filter, the system
        only returns chunks from that specific source.
        
        This behavior MUST be preserved after the fix.
        """
        # Store chunks first
        vector_db.store_chunks_with_embeddings(sample_chunks, sample_embeddings)
        
        # Search with source filter
        query_embedding = sample_embeddings[0]
        results = vector_db.search_similar_chunks(
            query_embedding, 
            limit=10, 
            source_filter="test_doc_1.txt"
        )
        
        # Verify all results are from the filtered source
        assert len(results) > 0, "Should return results"
        assert all(r['source'] == "test_doc_1.txt" for r in results), \
            "All results should be from filtered source"
    
    def test_delete_chunks_by_source(self, vector_db, sample_chunks, sample_embeddings):
        """
        Property 2: Preservation - Chunk Deletion
        
        OBSERVED BEHAVIOR: When deleting chunks by source, the system removes
        all chunks from that source and returns the count of deleted chunks.
        
        This behavior MUST be preserved after the fix.
        """
        # Store chunks first
        vector_db.store_chunks_with_embeddings(sample_chunks, sample_embeddings)
        
        # Delete chunks from one source
        deleted_count = vector_db.delete_chunks_by_source("test_doc_1.txt")
        
        # Verify deletion
        assert isinstance(deleted_count, int), "Should return integer count"
        assert deleted_count == 2, "Should delete 2 chunks from test_doc_1.txt"
        
        # Verify chunks are actually deleted
        stats = vector_db.get_database_stats()
        assert stats['total_chunks'] == 1, "Should have 1 chunk remaining"
        assert stats['total_sources'] == 1, "Should have 1 source remaining"
        assert "test_doc_1.txt" not in stats['source_breakdown'], \
            "Deleted source should not be in breakdown"
    
    # Property-Based Tests using Hypothesis
    
    @given(
        num_chunks=st.integers(min_value=1, max_value=10),
        embedding_dim=st.just(384),  # Fixed dimension for DistilBERT
        content_length=st.integers(min_value=10, max_value=200)
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_storage_retrieval_consistency(self, num_chunks, embedding_dim, content_length):
        """
        Property 2: Preservation - Storage/Retrieval Consistency
        
        PROPERTY: For any set of chunks stored with embeddings, retrieving
        each chunk by its ID should return the exact same content and metadata
        that was stored.
        
        This property MUST hold after the fix.
        """
        # Create temporary database
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_db")
        vector_db = EndeeVectorDatabase(database_path=db_path)
        
        try:
            # Generate random chunks
            chunks = []
            embeddings = []
            
            for i in range(num_chunks):
                content = f"Test content {i} " * (content_length // 20)
                chunk = Chunk(
                    content=content[:content_length],
                    chunk_id=f"test_chunk_{i}",
                    source=f"test_source_{i % 3}.txt",
                    start_index=i * 100,
                    end_index=(i + 1) * 100,
                    metadata={"index": i, "test": True}
                )
                chunks.append(chunk)
                embeddings.append(np.random.randn(embedding_dim).astype(np.float32))
            
            # Store chunks
            chunk_ids = vector_db.store_chunks_with_embeddings(chunks, embeddings)
            
            # Verify each chunk can be retrieved with correct data
            for i, chunk_id in enumerate(chunk_ids):
                retrieved = vector_db.get_chunk_by_id(chunk_id)
                
                assert retrieved is not None, f"Chunk {chunk_id} should be retrievable"
                assert retrieved['content'] == chunks[i].content, "Content should match"
                assert retrieved['source'] == chunks[i].source, "Source should match"
                assert retrieved['metadata'] == chunks[i].metadata, "Metadata should match"
                assert retrieved['start_index'] == chunks[i].start_index, "Start index should match"
                assert retrieved['end_index'] == chunks[i].end_index, "End index should match"
        
        finally:
            # Cleanup
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    @given(
        num_chunks=st.integers(min_value=3, max_value=15),
        top_k=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=15, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_search_returns_top_k_results(self, num_chunks, top_k):
        """
        Property 2: Preservation - Search Result Limit
        
        PROPERTY: For any search query with limit=top_k, the system should
        return at most top_k results, sorted by similarity in descending order.
        
        This property MUST hold after the fix.
        """
        assume(top_k <= num_chunks)
        
        # Create temporary database
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_db")
        vector_db = EndeeVectorDatabase(database_path=db_path)
        
        try:
            # Generate random chunks and embeddings
            chunks = []
            embeddings = []
            
            for i in range(num_chunks):
                chunk = Chunk(
                    content=f"Test document {i} with some content",
                    chunk_id=f"test_chunk_{i}",
                    source=f"source_{i}.txt",
                    start_index=0,
                    end_index=50,
                    metadata={"id": i}
                )
                chunks.append(chunk)
                embeddings.append(np.random.randn(384).astype(np.float32))
            
            # Store chunks
            vector_db.store_chunks_with_embeddings(chunks, embeddings)
            
            # Search with random query
            query_embedding = np.random.randn(384).astype(np.float32)
            results = vector_db.search_similar_chunks(query_embedding, limit=top_k)
            
            # Verify result count
            assert len(results) <= top_k, f"Should return at most {top_k} results"
            
            # Verify sorting by similarity
            if len(results) > 1:
                similarities = [r['similarity'] for r in results]
                assert similarities == sorted(similarities, reverse=True), \
                    "Results should be sorted by similarity (descending)"
        
        finally:
            # Cleanup
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    @given(
        num_sources=st.integers(min_value=2, max_value=5),
        chunks_per_source=st.integers(min_value=1, max_value=4)
    )
    @settings(max_examples=15, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_source_isolation(self, num_sources, chunks_per_source):
        """
        Property 2: Preservation - Source Isolation
        
        PROPERTY: For any set of chunks from multiple sources, deleting chunks
        from one source should not affect chunks from other sources.
        
        This property MUST hold after the fix.
        """
        # Create temporary database
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_db")
        vector_db = EndeeVectorDatabase(database_path=db_path)
        
        try:
            # Generate chunks from multiple sources
            all_chunks = []
            all_embeddings = []
            
            for source_idx in range(num_sources):
                for chunk_idx in range(chunks_per_source):
                    chunk = Chunk(
                        content=f"Content from source {source_idx}, chunk {chunk_idx}",
                        chunk_id=f"source_{source_idx}_chunk_{chunk_idx}",
                        source=f"source_{source_idx}.txt",
                        start_index=chunk_idx * 100,
                        end_index=(chunk_idx + 1) * 100,
                        metadata={"source_idx": source_idx, "chunk_idx": chunk_idx}
                    )
                    all_chunks.append(chunk)
                    all_embeddings.append(np.random.randn(384).astype(np.float32))
            
            # Store all chunks
            vector_db.store_chunks_with_embeddings(all_chunks, all_embeddings)
            
            # Get initial stats
            initial_stats = vector_db.get_database_stats()
            assert initial_stats['total_chunks'] == num_sources * chunks_per_source
            assert initial_stats['total_sources'] == num_sources
            
            # Delete chunks from first source
            deleted_count = vector_db.delete_chunks_by_source("source_0.txt")
            assert deleted_count == chunks_per_source
            
            # Verify other sources are unaffected
            final_stats = vector_db.get_database_stats()
            assert final_stats['total_chunks'] == (num_sources - 1) * chunks_per_source
            assert final_stats['total_sources'] == num_sources - 1
            
            # Verify remaining sources still have correct chunk counts
            for source_idx in range(1, num_sources):
                source_name = f"source_{source_idx}.txt"
                assert source_name in final_stats['source_breakdown']
                assert final_stats['source_breakdown'][source_name] == chunks_per_source
        
        finally:
            # Cleanup
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
