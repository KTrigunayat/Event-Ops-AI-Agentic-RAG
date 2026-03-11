# Bugfix Requirements Document

## Introduction

The Event-Ops AI Agentic RAG project currently fails to meet mandatory submission requirements for evaluation. While the Event-Ops code has been added to the forked Endee repository (https://github.com/KTrigunayat/endee), it still uses a custom JSON-based vector database implementation instead of the actual Endee C++ vector database server. This compliance issue prevents the project from being accepted for evaluation, as the submission explicitly requires using the actual Endee vector database from the forked repository.

The fix involves replacing the custom `endee_vector_db.py` implementation with an HTTP client that communicates with the actual Endee server via its REST API endpoints, and documenting the compliance steps in the README.

## Bug Analysis

### Current Behavior (Defect)

1.1 ✅ RESOLVED: WHEN the project repository is examined THEN the system exists within the forked Endee repository (https://github.com/KTrigunayat/endee) with all Event-Ops code integrated into the `Event-Ops-AI-Agentic-RAG/` directory

1.2 ❌ CRITICAL ISSUE: WHEN the codebase is analyzed THEN the system uses a custom JSON-based implementation in `Event-Ops-AI-Agentic-RAG/src/database/endee_vector_db.py` that stores vectors in JSON files with in-memory dictionaries, instead of using the actual Endee C++ server via HTTP API calls

1.3 ❌ ISSUE: WHEN the README is reviewed THEN the main repository README does not document that the Event-Ops application is included or how to use it with the Endee server

1.4 ❌ ISSUE: WHEN the Event-Ops README is reviewed THEN it does not document the requirement to run the Endee server or how to configure the connection to it

1.5 ❌ ISSUE: WHEN the project dependencies are checked THEN the system does not include HTTP client libraries (like `requests` or `httpx`) needed to communicate with the Endee REST API

### Expected Behavior (Correct)

2.1 ✅ ALREADY SATISFIED: WHEN the project repository is examined THEN the system SHALL exist within the forked Endee repository (https://github.com/KTrigunayat/endee) with all Event-Ops code integrated into it

2.2 ❌ MUST FIX: WHEN the codebase is analyzed THEN the system SHALL use the actual Endee server from the forked repository by:
   - Refactoring `Event-Ops-AI-Agentic-RAG/src/database/endee_vector_db.py` to be an HTTP client
   - Making API calls to Endee REST endpoints (e.g., `/api/v1/index/create`, `/api/v1/index/<index>/vector/insert`, `/api/v1/index/<index>/search`)
   - Removing all custom JSON-based storage and in-memory dictionary implementations
   - Configuring the Endee server URL (default: `http://localhost:8080`)

2.3 ❌ MUST FIX: WHEN the main README is reviewed THEN it SHALL document:
   - That this fork includes the Event-Ops AI Agentic RAG application
   - How to run the Event-Ops application with the Endee server
   - Link to the Event-Ops README for detailed instructions

2.4 ❌ MUST FIX: WHEN the Event-Ops README is reviewed THEN it SHALL document:
   - That the official Endee repository was starred and forked
   - That this application uses the forked Endee server as its vector database
   - How to build and run the Endee server before starting the Event-Ops application
   - How to configure the Endee server connection (URL, authentication token)

2.5 ❌ MUST FIX: WHEN the project dependencies are checked THEN the system SHALL include:
   - HTTP client library (`requests` or `httpx`) in `requirements.txt`
   - Configuration for Endee server URL and authentication token in `.env.example`

### Unchanged Behavior (Regression Prevention)

3.1 WHEN the Event-Ops AI functionality is tested THEN the system SHALL CONTINUE TO provide the same agentic RAG capabilities for event operations

3.2 WHEN document parsing and chunking operations are performed THEN the system SHALL CONTINUE TO process documents using the existing semantic and LLM-based chunking strategies

3.3 WHEN embedding generation is requested THEN the system SHALL CONTINUE TO generate embeddings using the DistilBERT and advanced embedding models

3.4 WHEN retrieval queries are executed THEN the system SHALL CONTINUE TO retrieve relevant context from the vector database (now via Endee HTTP API instead of JSON files)

3.5 WHEN the API endpoints are accessed THEN the system SHALL CONTINUE TO respond with the same functionality through `api.py` and `app.py`

3.6 WHEN the agentic orchestrator is invoked THEN the system SHALL CONTINUE TO coordinate multi-agent workflows for event operations

## Technical Implementation Details

### Endee REST API Endpoints to Use

The refactored `endee_vector_db.py` must use these Endee HTTP API endpoints:

1. **Create Index**: `POST /api/v1/index/create`
   - Creates a new vector index with specified dimensions and distance metric
   - Request body: `{"index_name": "string", "dimension": int, "metric": "cosine|euclidean|dot"}`

2. **Insert Vectors**: `POST /api/v1/index/<index_name>/vector/insert`
   - Inserts vectors with metadata into the index
   - Request body: `{"vectors": [{"id": "string", "values": [float], "metadata": {}}]}`

3. **Search Vectors**: `POST /api/v1/index/<index_name>/search`
   - Searches for similar vectors
   - Request body: `{"vector": [float], "top_k": int, "filter": {}}`

4. **Get Vector**: `POST /api/v1/index/<index_name>/vector/get`
   - Retrieves a specific vector by ID
   - Request body: `{"id": "string"}`

5. **Delete Vector**: `DELETE /api/v1/index/<index_name>/vector/<vector_id>/delete`
   - Deletes a specific vector

6. **List Indexes**: `GET /api/v1/index/list`
   - Lists all available indexes

7. **Index Info**: `GET /api/v1/index/<index_name>/info`
   - Gets information about a specific index

8. **Health Check**: `GET /api/v1/health`
   - Checks if the Endee server is running

### Authentication

- Endee supports optional authentication via `NDD_AUTH_TOKEN` environment variable
- When enabled, all API requests must include: `Authorization: <token>` header
- Event-Ops must support both authenticated and non-authenticated modes

### Configuration Requirements

Add to `Event-Ops-AI-Agentic-RAG/.env.example`:
```env
# Endee Vector Database Configuration
ENDEE_SERVER_URL=http://localhost:8080
ENDEE_AUTH_TOKEN=  # Optional: leave empty if Endee runs without authentication
ENDEE_INDEX_NAME=event_ops_vectors
ENDEE_DIMENSION=384  # Must match embedding dimension (DistilBERT default)
ENDEE_METRIC=cosine  # Distance metric: cosine, euclidean, or dot
```

### Implementation Checklist

- [ ] Refactor `EndeeVectorDatabase` class to use HTTP client (requests/httpx)
- [ ] Remove all JSON file storage logic
- [ ] Remove in-memory dictionaries (chunks, vectors, metadata, sources)
- [ ] Implement `_create_index()` method using Endee API
- [ ] Implement `store_chunks_with_embeddings()` using `/vector/insert` endpoint
- [ ] Implement `search_similar_chunks()` using `/search` endpoint
- [ ] Implement `get_chunk_by_id()` using `/vector/get` endpoint
- [ ] Implement `delete_chunks_by_source()` using `/vector/delete` endpoint
- [ ] Implement `get_database_stats()` using `/index/info` endpoint
- [ ] Add health check method using `/health` endpoint
- [ ] Add authentication header support when `ENDEE_AUTH_TOKEN` is set
- [ ] Update `requirements.txt` to include `requests` or `httpx`
- [ ] Update Event-Ops README with Endee server setup instructions
- [ ] Update main repository README to mention Event-Ops integration
- [ ] Add compliance documentation about starring and forking Endee repository
