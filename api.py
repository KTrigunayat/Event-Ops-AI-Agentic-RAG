from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
import sys
from pathlib import Path
import tempfile
import uuid
import logging
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.event_ops_ai import EventOpsAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Event-Ops AI API",
    description="Agentic RAG system for event operations support",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Event-Ops AI system
try:
    config = {
        "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "ollama_model": os.getenv("OLLAMA_MODEL", "tinyllama"),
        "endeedb_path": os.getenv("ENDEE_DB_PATH", "./endeedb"),
        "embedding_provider": os.getenv("EMBEDDING_PROVIDER", "openai"),
        "gemini_flash_api_key": os.getenv("GEMINI_FLASH_API_KEY"),
        "gemini_flash_model": os.getenv("GEMINI_FLASH_MODEL", "gemini-1.5-flash")
    }
    event_ops_ai = EventOpsAI(config)
    logger.info("Event-Ops AI system initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Event-Ops AI: {e}")
    event_ops_ai = None

# Pydantic models
class QueryRequest(BaseModel):
    query: str = Field(..., description="User query")
    user_id: str = Field(..., description="User identifier")
    event_id: str = Field(..., description="Event identifier")
    urgency: str = Field(default="medium", description="Query urgency (low, medium, high, critical)")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")

class SearchRequest(BaseModel):
    event_id: str = Field(..., description="Event identifier")
    query: str = Field(..., description="Search query")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Search filters")

class IngestionRequest(BaseModel):
    event_id: str = Field(..., description="Event identifier")
    document_paths: List[str] = Field(..., description="List of document paths")

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Event-Ops AI API",
        "version": "1.0.0",
        "status": "operational" if event_ops_ai else "error"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if event_ops_ai:
        try:
            status = event_ops_ai.get_system_status()
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "system_status": status
            }
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"System unhealthy: {str(e)}")
    else:
        raise HTTPException(status_code=503, detail="System not initialized")

@app.post("/query")
async def process_query(request: QueryRequest):
    """Process a user query through the agentic RAG system"""
    if not event_ops_ai:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        result = event_ops_ai.process_user_query(
            query=request.query,
            user_id=request.user_id,
            event_id=request.event_id,
            urgency=request.urgency,
            context=request.context
        )
        
        return {
            "success": result.success,
            "query_id": result.query_id,
            "response": result.response,
            "intent": result.intent,
            "actions": result.actions,
            "context_used": result.context_used,
            "processing_time": result.processing_time,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post("/ingest")
async def ingest_documents(
    background_tasks: BackgroundTasks,
    event_id: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """Ingest event documents"""
    if not event_ops_ai:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # Save uploaded files temporarily
        temp_paths = []
        for file in files:
            # Create temporary file
            suffix = Path(file.filename).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_paths.append(temp_file.name)
        
        # Process ingestion
        result = event_ops_ai.ingest_event_documents(
            document_paths=temp_paths,
            event_id=event_id
        )
        
        # Clean up temp files in background
        def cleanup_temp_files():
            for path in temp_paths:
                try:
                    os.remove(path)
                except:
                    pass
        
        background_tasks.add_task(cleanup_temp_files)
        
        return {
            "success": result.success,
            "event_id": result.event_id,
            "chunks_created": result.chunks_created,
            "categories_found": result.categories_found,
            "processing_time": result.processing_time,
            "message": result.message,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error ingesting documents: {e}")
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {str(e)}")

@app.post("/search")
async def search_documents(request: SearchRequest):
    """Search within event documents"""
    if not event_ops_ai:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        results = event_ops_ai.search_event_documents(
            event_id=request.event_id,
            query=request.query,
            filters=request.filters
        )
        
        return {
            "success": True,
            "event_id": request.event_id,
            "query": request.query,
            "results": results,
            "result_count": len(results),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(status_code=500, detail=f"Document search failed: {str(e)}")

@app.get("/events/{event_id}/overview")
async def get_event_overview(event_id: str):
    """Get overview of an event"""
    if not event_ops_ai:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        overview = event_ops_ai.get_event_overview(event_id)
        return overview
        
    except Exception as e:
        logger.error(f"Error getting event overview: {e}")
        raise HTTPException(status_code=500, detail=f"Event overview failed: {str(e)}")

@app.get("/system/status")
async def get_system_status():
    """Get detailed system status"""
    if not event_ops_ai:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        status = event_ops_ai.get_system_status()
        return status
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@app.get("/tools")
async def get_available_tools():
    """Get list of available action tools"""
    if not event_ops_ai:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        tools = event_ops_ai.action_tools.get_available_tools()
        tool_descriptions = {
            tool: event_ops_ai.action_tools.get_tool_description(tool)
            for tool in tools
        }
        
        return {
            "available_tools": tools,
            "descriptions": tool_descriptions,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting tools: {e}")
        raise HTTPException(status_code=500, detail=f"Tools retrieval failed: {str(e)}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )

# Run instructions
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
