import os
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid

from ..parsers.document_parser import DocumentParser
from ..chunking.semantic_chunker import EventBriefChunker
from ..embedding.advanced_embedder import AdvancedEmbedder, EmbeddingFactory
from ..database.endee_vector_db import EndeeVectorDatabase
from ..retrieval.context_retriever import ContextRetriever
from ..agents.agentic_orchestrator import AgenticOrchestrator, UserQuery, IntentType
from ..tools.action_tools import ActionTools

@dataclass
class EventIngestionResult:
    """Result from event document ingestion"""
    success: bool
    event_id: str
    chunks_created: int
    categories_found: List[str]
    processing_time: float
    message: str

@dataclass
class QueryResult:
    """Result from processing a user query"""
    success: bool
    response: str
    intent: Dict[str, Any]
    actions: List[Dict[str, Any]]
    context_used: int
    processing_time: float
    query_id: str

class EventOpsAI:
    """Main Event-Ops AI Agentic RAG System"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Event-Ops AI system
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info("Event-Ops AI System initialized successfully")
    
    def _initialize_components(self):
        """Initialize all system components"""
        try:
            # Initialize document parser
            self.document_parser = DocumentParser()
            
            # Initialize semantic chunker
            self.semantic_chunker = EventBriefChunker(
                base_url=self.config.get("ollama_base_url"),
                model=self.config.get("ollama_model", "tinyllama")
            )
            
            # Initialize embedder
            self.embedder = EmbeddingFactory.create_from_env(
                self.config.get("embedding_provider")
            )
            
            # Initialize database
            self.database = EndeeVectorDatabase(
                database_path=self.config.get("endeedb_path")
            )
            
            # Initialize context retriever
            self.context_retriever = ContextRetriever(
                database=self.database,
                embedder=self.embedder
            )
            
            # Initialize agentic orchestrator
            self.agentic_orchestrator = AgenticOrchestrator(
                api_key=self.config.get("gemini_flash_api_key"),
                model=self.config.get("gemini_flash_model", "gemini-1.5-flash")
            )
            
            # Initialize action tools
            self.action_tools = ActionTools(self.database)
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            raise
    
    def ingest_event_documents(self, document_paths: List[str], 
                            event_id: str = None) -> EventIngestionResult:
        """
        Ingest event documents into the system
        
        Args:
            document_paths: List of paths to event documents
            event_id: Unique identifier for the event
            
        Returns:
            EventIngestionResult with processing details
        """
        start_time = datetime.now()
        
        if not event_id:
            event_id = f"event_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        try:
            self.logger.info(f"Starting ingestion for event: {event_id}")
            
            all_chunks = []
            all_categories = set()
            
            for doc_path in document_paths:
                # Parse document
                doc_content = self.document_parser.parse_document(doc_path)
                
                # Create semantic chunks
                semantic_chunks = self.semantic_chunker.chunk_event_brief(
                    text=doc_content.text,
                    source=f"{event_id}_{doc_content.file_type}"
                )
                
                # Update source for all chunks
                for chunk in semantic_chunks:
                    chunk.source = event_id
                
                all_chunks.extend(semantic_chunks)
                
                # Track categories
                categories = self.semantic_chunker.get_categories_summary(semantic_chunks)
                all_categories.update(categories.keys())
                
                self.logger.info(f"Processed {doc_content.file_name}: {len(semantic_chunks)} chunks")
            
            # Generate embeddings
            embeddings = self.embedder.embed_chunks(all_chunks)
            
            # Store in database
            stored_ids = self.database.store_chunks_with_embeddings(all_chunks, embeddings)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = EventIngestionResult(
                success=True,
                event_id=event_id,
                chunks_created=len(all_chunks),
                categories_found=list(all_categories),
                processing_time=processing_time,
                message=f"Successfully ingested {len(document_paths)} documents for event {event_id}"
            )
            
            self.logger.info(f"Ingestion completed: {result.message}")
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Error in document ingestion: {e}")
            
            return EventIngestionResult(
                success=False,
                event_id=event_id,
                chunks_created=0,
                categories_found=[],
                processing_time=processing_time,
                message=f"Ingestion failed: {str(e)}"
            )
    
    def process_user_query(self, query: str, user_id: str, event_id: str,
                         urgency: str = "medium", context: Dict[str, Any] = None) -> QueryResult:
        """
        Process a user query through the complete agentic RAG pipeline
        
        Args:
            query: User's query
            user_id: User identifier
            event_id: Event identifier
            urgency: Query urgency ("low", "medium", "high", "critical")
            context: Additional context
            
        Returns:
            QueryResult with response and metadata
        """
        start_time = datetime.now()
        query_id = f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        try:
            self.logger.info(f"Processing query {query_id}: {query[:50]}...")
            
            # Create user query object
            user_query = UserQuery(
                query=query,
                user_id=user_id,
                event_id=event_id,
                timestamp=datetime.now().isoformat(),
                urgency=urgency,
                context=context or {}
            )
            
            # Step 1: Retrieve relevant context
            retrieval_result = self.context_retriever.retrieve_context(
                query=query,
                event_id=event_id,
                top_k=10
            )
            
            if not retrieval_result.success:
                raise Exception(f"Context retrieval failed: {retrieval_result.retrieval_metadata}")
            
            # Step 2: Process through agentic orchestrator
            orchestrator_result = self.agentic_orchestrator.process_query(
                query=user_query,
                retrieved_context=retrieval_result.chunks
            )
            
            # Step 3: Execute planned actions
            executed_actions = []
            if orchestrator_result.get("processing_status") == "success":
                for action in orchestrator_result.get("actions", []):
                    tool_result = self.action_tools.execute_tool(
                        tool_name=action["type"],
                        parameters=action["parameters"]
                    )
                    executed_actions.append({
                        "action": action,
                        "execution_result": {
                            "success": tool_result.success,
                            "message": tool_result.message,
                            "data": tool_result.data,
                            "execution_time": tool_result.execution_time
                        }
                    })
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = QueryResult(
                success=orchestrator_result.get("processing_status") == "success",
                response=orchestrator_result.get("response", "I apologize, but I couldn't process your request."),
                intent=orchestrator_result.get("intent", {}),
                actions=executed_actions,
                context_used=len(retrieval_result.chunks),
                processing_time=processing_time,
                query_id=query_id
            )
            
            self.logger.info(f"Query {query_id} processed successfully in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Error processing query {query_id}: {e}")
            
            return QueryResult(
                success=False,
                response=f"I apologize, but I encountered an error: {str(e)}",
                intent={},
                actions=[],
                context_used=0,
                processing_time=processing_time,
                query_id=query_id
            )
    
    def get_event_overview(self, event_id: str) -> Dict[str, Any]:
        """
        Get overview of an event's data
        
        Args:
            event_id: Event identifier
            
        Returns:
            Event overview with statistics and categories
        """
        try:
            # Get all chunks for the event
            event_chunks = self.database.get_chunks_by_source(event_id)
            
            if not event_chunks:
                return {
                    "event_id": event_id,
                    "status": "not_found",
                    "message": f"No data found for event {event_id}"
                }
            
            # Analyze categories
            categories = {}
            priorities = {"high": 0, "medium": 0, "low": 0}
            
            for chunk in event_chunks:
                category = chunk.get("metadata", {}).get("category", "Unknown")
                categories[category] = categories.get(category, 0) + 1
                
                priority = chunk.get("metadata", {}).get("priority", "low")
                if priority in priorities:
                    priorities[priority] += 1
            
            return {
                "event_id": event_id,
                "status": "found",
                "total_chunks": len(event_chunks),
                "categories": categories,
                "priorities": priorities,
                "last_updated": max(chunk.get("created_at", "") for chunk in event_chunks),
                "sample_chunks": [
                    {
                        "content": chunk.get("content", "")[:100] + "...",
                        "category": chunk.get("metadata", {}).get("category", "Unknown"),
                        "priority": chunk.get("metadata", {}).get("priority", "low")
                    }
                    for chunk in event_chunks[:5]
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting event overview for {event_id}: {e}")
            return {
                "event_id": event_id,
                "status": "error",
                "error": str(e)
            }
    
    def search_event_documents(self, event_id: str, query: str, 
                            filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Search within event documents
        
        Args:
            event_id: Event identifier
            query: Search query
            filters: Optional filters
            
        Returns:
            List of matching chunks
        """
        try:
            retrieval_result = self.context_retriever.semantic_search(
                query=query,
                filters=filters
            )
            
            if retrieval_result.success:
                # Filter by event ID
                event_chunks = [
                    chunk for chunk in retrieval_result.chunks
                    if chunk.get("source", "") == event_id
                ]
                
                return event_chunks
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Error searching event documents: {e}")
            return []
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status and statistics"""
        try:
            db_stats = self.database.get_database_stats()
            retrieval_stats = self.context_retriever.get_retrieval_stats()
            
            return {
                "system_status": "operational",
                "components": {
                    "document_parser": "ready",
                    "semantic_chunker": "ready",
                    "embedder": self.embedder.get_provider_info(),
                    "database": db_stats,
                    "context_retriever": retrieval_stats,
                    "agentic_orchestrator": "ready",
                    "action_tools": self.action_tools.get_available_tools()
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {
                "system_status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
