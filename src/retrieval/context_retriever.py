import os
import logging
from typing import Dict, Any, List, Optional
import numpy as np
from dataclasses import dataclass

from ..database.endee_vector_db import EndeeVectorDatabase
from ..embedding.advanced_embedder import AdvancedEmbedder

@dataclass
class RetrievalResult:
    """Result from context retrieval"""
    chunks: List[Dict[str, Any]]
    query_embedding: np.ndarray
    retrieval_metadata: Dict[str, Any]
    success: bool

class ContextRetriever:
    """Advanced context retrieval system for the agentic RAG"""
    
    def __init__(self, database: EndeeVectorDatabase, embedder: AdvancedEmbedder):
        self.database = database
        self.embedder = embedder
        self.logger = logging.getLogger(__name__)
        
        # Retrieval configuration
        self.default_top_k = 10
        self.similarity_threshold = 0.3
        self.category_boost = {
            "Crisis Protocol": 1.2,
            "Vendor Information": 1.1,
            "Timeline & Schedule": 1.1,
            "Contingency Plans": 1.15
        }
    
    def retrieve_context(self, query: str, event_id: str = None, 
                       intent_type: str = None, top_k: int = None) -> RetrievalResult:
        """
        Retrieve relevant context for a query
        
        Args:
            query: User query
            event_id: Optional event ID for filtering
            intent_type: Optional intent type for specialized retrieval
            top_k: Number of chunks to retrieve
            
        Returns:
            RetrievalResult with chunks and metadata
        """
        try:
            top_k = top_k or self.default_top_k
            
            self.logger.info(f"Retrieving context for query: {query[:50]}...")
            
            # Generate query embedding
            query_embedding = self.embedder.embed_single_text(query)
            
            # Perform initial retrieval
            initial_results = self.database.search_similar_chunks(
                query_embedding=query_embedding,
                limit=top_k * 2,  # Get more for reranking
                similarity_threshold=self.similarity_threshold
            )
            
            # Filter by event if specified
            if event_id:
                initial_results = [
                    chunk for chunk in initial_results
                    if chunk.get("source", "") == event_id or event_id in chunk.get("source", "")
                ]
            
            # Rerank and filter results
            final_results = self._rerank_results(
                initial_results, query, intent_type, top_k
            )
            
            retrieval_metadata = {
                "query": query,
                "event_id": event_id,
                "intent_type": intent_type,
                "initial_results": len(initial_results),
                "final_results": len(final_results),
                "similarity_threshold": self.similarity_threshold,
                "categories_found": list(set(chunk.get("metadata", {}).get("category", "Unknown") 
                                       for chunk in final_results))
            }
            
            self.logger.info(f"Retrieved {len(final_results)} relevant chunks")
            
            return RetrievalResult(
                chunks=final_results,
                query_embedding=query_embedding,
                retrieval_metadata=retrieval_metadata,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Error in context retrieval: {e}")
            return RetrievalResult(
                chunks=[],
                query_embedding=np.array([]),
                retrieval_metadata={"error": str(e)},
                success=False
            )
    
    def _rerank_results(self, results: List[Dict[str, Any]], query: str, 
                       intent_type: str, top_k: int) -> List[Dict[str, Any]]:
        """Rerank retrieval results based on multiple factors"""
        if not results:
            return []
        
        # Calculate enhanced scores
        for result in results:
            base_score = result.get("similarity", 0.0)
            
            # Category-based boosting
            category = result.get("metadata", {}).get("category", "General Information")
            category_boost = self.category_boost.get(category, 1.0)
            
            # Intent-based boosting
            intent_boost = self._get_intent_boost(category, intent_type)
            
            # Recency boost (if timestamp available)
            recency_boost = self._get_recency_boost(result)
            
            # Content length penalty (prefer concise, relevant chunks)
            content_length = len(result.get("content", ""))
            length_penalty = 1.0 if content_length < 1000 else 0.9
            
            # Calculate final score
            final_score = base_score * category_boost * intent_boost * recency_boost * length_penalty
            result["enhanced_score"] = final_score
        
        # Sort by enhanced score and return top_k
        results.sort(key=lambda x: x.get("enhanced_score", 0), reverse=True)
        return results[:top_k]
    
    def _get_intent_boost(self, category: str, intent_type: str) -> float:
        """Get boost factor based on intent type and category"""
        if not intent_type:
            return 1.0
        
        intent_category_mapping = {
            "emergency": {
                "Crisis Protocol": 1.5,
                "Safety & Compliance": 1.3,
                "Contingency Plans": 1.4
            },
            "decision": {
                "Timeline & Schedule": 1.3,
                "Contingency Plans": 1.4,
                "Vendor Information": 1.2
            },
            "coordination": {
                "Staffing & Personnel": 1.3,
                "Communication Plan": 1.4,
                "Vendor Information": 1.2
            },
            "lookup": {
                "Vendor Information": 1.2,
                "Venue Details": 1.1,
                "Timeline & Schedule": 1.1
            }
        }
        
        return intent_category_mapping.get(intent_type, {}).get(category, 1.0)
    
    def _get_recency_boost(self, result: Dict[str, Any]) -> float:
        """Get recency boost factor"""
        # In a real implementation, this would use actual timestamps
        # For now, return neutral boost
        return 1.0
    
    def retrieve_by_category(self, category: str, event_id: str = None, 
                          limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve chunks by specific category
        
        Args:
            category: Category to filter by
            event_id: Optional event ID
            limit: Maximum number of chunks
            
        Returns:
            List of chunks in the specified category
        """
        try:
            all_chunks = self.database.get_chunks_by_source(event_id) if event_id else []
            
            # If no event_id specified, we need to get all chunks (this would need DB enhancement)
            if not all_chunks:
                # For now, return empty list - in production, implement get_all_chunks
                return []
            
            # Filter by category
            category_chunks = [
                chunk for chunk in all_chunks
                if chunk.get("metadata", {}).get("category", "") == category
            ]
            
            return category_chunks[:limit]
            
        except Exception as e:
            self.logger.error(f"Error retrieving by category {category}: {e}")
            return []
    
    def retrieve_high_priority(self, event_id: str = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve high-priority chunks
        
        Args:
            event_id: Optional event ID
            limit: Maximum number of chunks
            
        Returns:
            List of high-priority chunks
        """
        try:
            all_chunks = self.database.get_chunks_by_source(event_id) if event_id else []
            
            if not all_chunks:
                return []
            
            # Filter by high priority
            high_priority_chunks = [
                chunk for chunk in all_chunks
                if chunk.get("metadata", {}).get("priority", "low") == "high"
            ]
            
            return high_priority_chunks[:limit]
            
        except Exception as e:
            self.logger.error(f"Error retrieving high priority chunks: {e}")
            return []
    
    def semantic_search(self, query: str, filters: Dict[str, Any] = None) -> RetrievalResult:
        """
        Perform semantic search with advanced filtering
        
        Args:
            query: Search query
            filters: Dictionary of filters (category, priority, etc.)
            
        Returns:
            RetrievalResult with filtered results
        """
        try:
            # Get base retrieval
            base_result = self.retrieve_context(query)
            
            if not base_result.success:
                return base_result
            
            # Apply filters
            filtered_chunks = base_result.chunks
            
            if filters:
                if "category" in filters:
                    filtered_chunks = [
                        chunk for chunk in filtered_chunks
                        if chunk.get("metadata", {}).get("category", "") == filters["category"]
                    ]
                
                if "priority" in filters:
                    filtered_chunks = [
                        chunk for chunk in filtered_chunks
                        if chunk.get("metadata", {}).get("priority", "low") == filters["priority"]
                    ]
                
                if "min_similarity" in filters:
                    filtered_chunks = [
                        chunk for chunk in filtered_chunks
                        if chunk.get("similarity", 0) >= filters["min_similarity"]
                    ]
            
            # Update result with filtered chunks
            base_result.chunks = filtered_chunks
            base_result.retrieval_metadata["filters_applied"] = filters
            base_result.retrieval_metadata["filtered_results"] = len(filtered_chunks)
            
            return base_result
            
        except Exception as e:
            self.logger.error(f"Error in semantic search: {e}")
            return RetrievalResult(
                chunks=[],
                query_embedding=np.array([]),
                retrieval_metadata={"error": str(e)},
                success=False
            )
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about the retrieval system"""
        try:
            db_stats = self.database.get_database_stats()
            
            return {
                "database_stats": db_stats,
                "retrieval_config": {
                    "default_top_k": self.default_top_k,
                    "similarity_threshold": self.similarity_threshold,
                    "category_boosts": self.category_boost
                },
                "embedder_info": self.embedder.get_provider_info()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting retrieval stats: {e}")
            return {"error": str(e)}
