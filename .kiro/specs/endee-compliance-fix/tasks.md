# Implementation Plan

- [x] 1. Write bug condition exploration test
  - **Property 1: Fault Condition** - JSON File Storage Detection
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate the current implementation uses JSON files instead of HTTP API
  - **Scoped PBT Approach**: Scope the property to concrete failing cases - verify the class uses file I/O operations
  - Test that `EndeeVectorDatabase` class uses `_load_database()` and `_save_database()` methods with JSON files
  - Test that no HTTP client library (requests/httpx) is imported or used in the class
  - Test that `self.chunks`, `self.vectors`, `self.metadata` in-memory dictionaries exist
  - Test that no code makes HTTP requests to Endee endpoints (e.g., `/api/v1/index/create`, `/api/v1/index/<index>/vector/insert`)
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS (this is correct - it proves the compliance violation exists)
  - Document counterexamples found: JSON file usage, absence of HTTP client, in-memory storage
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 1.2, 1.5_

- [x] 2. Write preservation property tests (BEFORE implementing fix)
  - **Property 2: Preservation** - Event-Ops Functional Equivalence
  - **IMPORTANT**: Follow observation-first methodology
  - Observe behavior on UNFIXED code for Event-Ops AI functionality
  - Test document ingestion: store sample documents and verify chunk IDs are returned
  - Test semantic search: query with sample embeddings and verify results structure
  - Test chunk retrieval: get chunks by ID and verify data structure
  - Test stats operation: get database stats and verify format
  - Write property-based tests capturing observed behavior patterns for RAG queries and document processing
  - Property-based testing generates many test cases for stronger guarantees across different document types
  - Run tests on UNFIXED code (using JSON-based storage)
  - **EXPECTED OUTCOME**: Tests PASS (this confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [ ] 3. Fix for Endee compliance - Replace JSON storage with HTTP API integration

  - [x] 3.1 Add HTTP client dependency
    - Add `requests>=2.31.0` to `Event-Ops-AI-Agentic-RAG/requirements.txt`
    - _Bug_Condition: isBugCondition(operation) where operation.uses_http_api == false_
    - _Expected_Behavior: All vector operations use HTTP API calls to Endee server_
    - _Preservation: Event-Ops functionality remains unchanged_
    - _Requirements: 1.5, 2.5_

  - [x] 3.2 Add Endee configuration to environment template
    - Add `ENDEE_SERVER_URL`, `ENDEE_AUTH_TOKEN`, `ENDEE_INDEX_NAME`, `ENDEE_DIMENSION`, `ENDEE_METRIC` to `Event-Ops-AI-Agentic-RAG/.env.example`
    - Include default values and documentation comments
    - _Bug_Condition: Missing configuration for Endee server connection_
    - _Expected_Behavior: Configuration variables available for Endee integration_
    - _Preservation: Existing environment variables remain unchanged_
    - _Requirements: 2.5_

  - [x] 3.3 Refactor EndeeVectorDatabase class to use HTTP client
    - Remove `_load_database()` and `_save_database()` methods
    - Remove `self.database_path` and all JSON file I/O operations
    - Remove in-memory dictionaries: `self.chunks`, `self.vectors`, `self.metadata`, `self.sources`
    - Add HTTP client initialization (requests) in `__init__()`
    - Load configuration from environment: `ENDEE_SERVER_URL`, `ENDEE_AUTH_TOKEN`, `ENDEE_INDEX_NAME`, `ENDEE_DIMENSION`, `ENDEE_METRIC`
    - Add `_create_index()` method that calls `POST /api/v1/index/create`
    - Call `_create_index()` in `__init__()` to ensure index exists (idempotent)
    - Add `health_check()` method that calls `GET /api/v1/health`
    - Call `health_check()` in `__init__()` to verify Endee server is reachable
    - Add authentication header support: include `Authorization: <token>` when `ENDEE_AUTH_TOKEN` is set
    - _Bug_Condition: Class uses JSON files instead of HTTP API_
    - _Expected_Behavior: Class uses HTTP client to communicate with Endee server_
    - _Preservation: Method signatures remain unchanged_
    - _Requirements: 1.2, 2.2, 2.5_

  - [x] 3.4 Refactor store_chunks_with_embeddings() method
    - Replace JSON file writing with `POST /api/v1/index/<index>/vector/insert` API call
    - Transform chunks and embeddings to Endee format: `{"vectors": [{"id": str, "values": [float], "metadata": {}}]}`
    - Preserve return type: `List[str]` of chunk IDs
    - Handle API errors gracefully
    - _Bug_Condition: Method writes to JSON files_
    - _Expected_Behavior: Method calls Endee insert endpoint_
    - _Preservation: Return type and functionality unchanged_
    - _Requirements: 2.2_

  - [ ] 3.5 Refactor search_similar_chunks() method
    - Replace in-memory cosine similarity with `POST /api/v1/index/<index>/search` API call
    - Send query embedding and top_k parameter to Endee API
    - Transform Endee response to expected format: `List[Dict[str, Any]]`
    - Preserve metadata filtering capability
    - Preserve return type and data structure
    - _Bug_Condition: Method computes similarity in-memory from JSON data_
    - _Expected_Behavior: Method calls Endee search endpoint_
    - _Preservation: Return type and functionality unchanged_
    - _Requirements: 2.2_

  - [ ] 3.6 Refactor get_chunk_by_id() method
    - Replace in-memory dictionary lookup with `POST /api/v1/index/<index>/vector/get` API call
    - Transform Endee response to expected format
    - Preserve return type: `Optional[Dict[str, Any]]`
    - _Bug_Condition: Method reads from in-memory dictionary_
    - _Expected_Behavior: Method calls Endee get endpoint_
    - _Preservation: Return type and functionality unchanged_
    - _Requirements: 2.2_

  - [ ] 3.7 Refactor delete_chunks_by_source() method
    - Replace JSON file manipulation with `DELETE /api/v1/index/<index>/vector/<id>/delete` API calls
    - First retrieve chunk IDs by source using metadata filter
    - Delete each chunk individually via API
    - Return count of deleted chunks
    - _Bug_Condition: Method modifies JSON files_
    - _Expected_Behavior: Method calls Endee delete endpoint_
    - _Preservation: Return type and functionality unchanged_
    - _Requirements: 2.2_

  - [ ] 3.8 Refactor get_database_stats() method
    - Replace JSON file counting with `GET /api/v1/index/<index>/info` API call
    - Transform Endee index info to expected stats format
    - Include total chunks, unique sources, index size
    - Preserve return type: `Dict[str, Any]`
    - _Bug_Condition: Method counts local JSON entries_
    - _Expected_Behavior: Method calls Endee info endpoint_
    - _Preservation: Return type and functionality unchanged_
    - _Requirements: 2.2_

  - [ ] 3.9 Update Event-Ops README with Endee server setup
    - Document that official Endee repository was starred and forked
    - Document that application uses forked Endee server as vector database
    - Add section on building and running Endee server before starting Event-Ops
    - Add section on configuring Endee server connection (URL, auth token)
    - Include example commands for starting Endee server
    - _Bug_Condition: README doesn't document Endee server requirement_
    - _Expected_Behavior: README documents Endee integration and setup_
    - _Preservation: Existing README content remains unchanged_
    - _Requirements: 1.4, 2.4_

  - [ ] 3.10 Update main repository README with Event-Ops integration
    - Add section documenting that this fork includes Event-Ops AI Agentic RAG application
    - Add instructions on how to run Event-Ops with Endee server
    - Add link to Event-Ops README for detailed instructions
    - _Bug_Condition: Main README doesn't mention Event-Ops integration_
    - _Expected_Behavior: Main README documents Event-Ops inclusion_
    - _Preservation: Existing README content remains unchanged_
    - _Requirements: 1.3, 2.3_

  - [ ] 3.11 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - HTTP API Integration Verification
    - **IMPORTANT**: Re-run the SAME test from task 1 - do NOT write a new test
    - The test from task 1 encodes the expected behavior
    - When this test passes, it confirms HTTP API integration is complete
    - Run bug condition exploration test from step 1
    - Verify that `EndeeVectorDatabase` class now uses HTTP client (requests library)
    - Verify that no JSON file I/O methods exist
    - Verify that all operations make HTTP requests to Endee endpoints
    - **EXPECTED OUTCOME**: Test PASSES (confirms compliance fix is complete)
    - _Requirements: 2.2, 2.5_

  - [ ] 3.12 Verify preservation tests still pass
    - **Property 2: Preservation** - Event-Ops Functional Equivalence
    - **IMPORTANT**: Re-run the SAME tests from task 2 - do NOT write new tests
    - Run preservation property tests from step 2
    - Verify document ingestion produces same chunk IDs and structure
    - Verify semantic search returns same result format
    - Verify chunk retrieval returns same data structure
    - Verify database stats return same format
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions in Event-Ops functionality)
    - Confirm all tests still pass after fix (no regressions)

- [ ] 4. Checkpoint - Ensure all tests pass
  - Run all exploration tests - should pass (HTTP API integration verified)
  - Run all preservation tests - should pass (Event-Ops functionality preserved)
  - Run existing Event-Ops test suite if available
  - Verify Endee server can be started and Event-Ops connects successfully
  - Verify documentation is complete and accurate
  - Ask the user if questions arise
