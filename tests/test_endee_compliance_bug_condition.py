"""
Bug Condition Exploration Test for Endee Compliance Fix

**Validates: Requirements 1.2, 1.5**

This test verifies that the current implementation uses JSON file storage
instead of HTTP API calls to the Endee server. This test is EXPECTED TO FAIL
on unfixed code - failure confirms the compliance violation exists.

CRITICAL: This test encodes the EXPECTED behavior (HTTP API usage).
When run on UNFIXED code, it will FAIL, proving the bug exists.
After the fix is implemented, this test should PASS.
"""

import pytest
import inspect
import ast
import os
from pathlib import Path


class TestEndeeBugCondition:
    """
    Bug Condition Exploration: Verify current implementation uses JSON files
    instead of HTTP API calls to Endee server.
    
    This test suite checks for the compliance violation by inspecting the
    EndeeVectorDatabase class implementation.
    """
    
    @pytest.fixture
    def endee_db_source_path(self):
        """Get path to the EndeeVectorDatabase source file"""
        return Path("src/database/endee_vector_db.py")
    
    @pytest.fixture
    def endee_db_source(self, endee_db_source_path):
        """Load the source code of EndeeVectorDatabase"""
        with open(endee_db_source_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    @pytest.fixture
    def endee_db_ast(self, endee_db_source):
        """Parse the source code into an AST"""
        return ast.parse(endee_db_source)
    
    def test_http_client_library_imported(self, endee_db_source):
        """
        Property 1: Fault Condition - HTTP Client Library Usage
        
        EXPECTED BEHAVIOR: The implementation SHOULD import HTTP client libraries
        (requests or httpx) to communicate with Endee server.
        
        CURRENT BEHAVIOR (BUG): The implementation does NOT import HTTP client
        libraries, indicating it doesn't make API calls.
        
        This test will FAIL on unfixed code, confirming the bug exists.
        """
        # Check for HTTP client imports
        has_requests = "import requests" in endee_db_source or "from requests" in endee_db_source
        has_httpx = "import httpx" in endee_db_source or "from httpx" in endee_db_source
        
        assert has_requests or has_httpx, (
            "COMPLIANCE VIOLATION: EndeeVectorDatabase does not import HTTP client library "
            "(requests or httpx). The implementation should use HTTP API calls to communicate "
            "with the Endee server, not JSON file storage."
        )
    
    def test_no_json_file_operations(self, endee_db_ast):
        """
        Property 1: Fault Condition - JSON File Storage Absence
        
        EXPECTED BEHAVIOR: The implementation SHOULD NOT use JSON file operations
        for persistence. It should use HTTP API calls instead.
        
        CURRENT BEHAVIOR (BUG): The implementation uses _load_database() and
        _save_database() methods with JSON files.
        
        This test will FAIL on unfixed code, confirming the bug exists.
        """
        # Find the EndeeVectorDatabase class
        class_node = None
        for node in ast.walk(endee_db_ast):
            if isinstance(node, ast.ClassDef) and node.name == "EndeeVectorDatabase":
                class_node = node
                break
        
        assert class_node is not None, "EndeeVectorDatabase class not found"
        
        # Check for JSON file operation methods
        method_names = [method.name for method in class_node.body if isinstance(method, ast.FunctionDef)]
        
        has_load_database = "_load_database" in method_names
        has_save_database = "_save_database" in method_names
        
        assert not has_load_database and not has_save_database, (
            f"COMPLIANCE VIOLATION: EndeeVectorDatabase has JSON file operation methods "
            f"(_load_database: {has_load_database}, _save_database: {has_save_database}). "
            f"The implementation should use HTTP API calls to the Endee server instead of "
            f"JSON file storage."
        )
    
    def test_no_in_memory_storage_dictionaries(self, endee_db_source):
        """
        Property 1: Fault Condition - In-Memory Storage Absence
        
        EXPECTED BEHAVIOR: The implementation SHOULD NOT use in-memory dictionaries
        (self.chunks, self.vectors, self.metadata, self.sources) for storage.
        It should delegate storage to the Endee server via HTTP API.
        
        CURRENT BEHAVIOR (BUG): The implementation uses in-memory dictionaries
        to store all data between JSON file loads.
        
        This test will FAIL on unfixed code, confirming the bug exists.
        """
        # Check for in-memory storage dictionaries
        has_chunks_dict = "self.chunks = {}" in endee_db_source
        has_vectors_dict = "self.vectors = {}" in endee_db_source
        has_metadata_dict = "self.metadata = {}" in endee_db_source
        has_sources_dict = "self.sources = {}" in endee_db_source
        
        assert not (has_chunks_dict or has_vectors_dict or has_metadata_dict or has_sources_dict), (
            f"COMPLIANCE VIOLATION: EndeeVectorDatabase uses in-memory storage dictionaries "
            f"(chunks: {has_chunks_dict}, vectors: {has_vectors_dict}, "
            f"metadata: {has_metadata_dict}, sources: {has_sources_dict}). "
            f"The implementation should use HTTP API calls to the Endee server for storage "
            f"instead of maintaining in-memory state."
        )
    
    def test_endee_api_endpoints_referenced(self, endee_db_source):
        """
        Property 1: Fault Condition - Endee API Endpoint Usage
        
        EXPECTED BEHAVIOR: The implementation SHOULD reference Endee REST API
        endpoints (e.g., /api/v1/index/create, /api/v1/index/<index>/vector/insert).
        
        CURRENT BEHAVIOR (BUG): The implementation does not reference any Endee
        API endpoints, indicating it doesn't make HTTP requests.
        
        This test will FAIL on unfixed code, confirming the bug exists.
        """
        # Check for Endee API endpoint references
        endee_endpoints = [
            "/api/v1/index/create",
            "/api/v1/index/",
            "/vector/insert",
            "/search",
            "/vector/get",
            "/vector/delete",
            "/health"
        ]
        
        has_any_endpoint = any(endpoint in endee_db_source for endpoint in endee_endpoints)
        
        assert has_any_endpoint, (
            "COMPLIANCE VIOLATION: EndeeVectorDatabase does not reference any Endee REST API "
            "endpoints. The implementation should make HTTP requests to endpoints like "
            "/api/v1/index/create, /api/v1/index/<index>/vector/insert, etc."
        )
    
    def test_endee_server_url_configuration(self, endee_db_source):
        """
        Property 1: Fault Condition - Endee Server URL Configuration
        
        EXPECTED BEHAVIOR: The implementation SHOULD load Endee server URL
        configuration (e.g., ENDEE_SERVER_URL environment variable).
        
        CURRENT BEHAVIOR (BUG): The implementation only loads database_path
        for JSON file storage, not server URL for HTTP API.
        
        This test will FAIL on unfixed code, confirming the bug exists.
        """
        # Check for Endee server URL configuration
        has_server_url_config = (
            "ENDEE_SERVER_URL" in endee_db_source or
            "endee_server_url" in endee_db_source.lower() or
            "server_url" in endee_db_source
        )
        
        assert has_server_url_config, (
            "COMPLIANCE VIOLATION: EndeeVectorDatabase does not load Endee server URL "
            "configuration. The implementation should read ENDEE_SERVER_URL from environment "
            "variables to connect to the Endee server via HTTP API."
        )
    
    def test_http_request_methods_present(self, endee_db_ast):
        """
        Property 1: Fault Condition - HTTP Request Method Usage
        
        EXPECTED BEHAVIOR: The implementation SHOULD make HTTP requests
        (GET, POST, DELETE) to the Endee server API.
        
        CURRENT BEHAVIOR (BUG): The implementation does not make any HTTP
        requests, using JSON file I/O instead.
        
        This test will FAIL on unfixed code, confirming the bug exists.
        """
        # Find the EndeeVectorDatabase class
        class_node = None
        for node in ast.walk(endee_db_ast):
            if isinstance(node, ast.ClassDef) and node.name == "EndeeVectorDatabase":
                class_node = node
                break
        
        assert class_node is not None, "EndeeVectorDatabase class not found"
        
        # Check for HTTP request method calls in any method
        has_http_calls = False
        http_methods = ["get", "post", "put", "delete", "patch"]
        
        for method in class_node.body:
            if isinstance(method, ast.FunctionDef):
                method_source = ast.unparse(method)
                # Check for requests.get(), requests.post(), self.session.get(), self.session.post(), etc.
                for http_method in http_methods:
                    if (f"requests.{http_method}(" in method_source or 
                        f"self.client.{http_method}(" in method_source or
                        f"self.session.{http_method}(" in method_source):
                        has_http_calls = True
                        break
                if has_http_calls:
                    break
        
        assert has_http_calls, (
            "COMPLIANCE VIOLATION: EndeeVectorDatabase does not make HTTP requests "
            "(GET, POST, DELETE) to the Endee server. The implementation should use "
            "HTTP client methods to communicate with the Endee REST API."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
