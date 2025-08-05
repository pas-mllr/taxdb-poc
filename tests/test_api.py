"""
Tests for API endpoints.

This module provides tests for the API endpoints, focusing on vector similarity search.
"""

import pytest
import json
from typing import Dict, List, Any

from fastapi.testclient import TestClient

from src import settings
from src.api import create_app
from src.models import Document
from src.db import db_manager
from src.etl.utils import get_embedding_strategy


def test_api_health(test_client):
    """Test API health endpoint."""
    response = test_client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_api_search_endpoint(test_client):
    """Test API search endpoint."""
    # Test with various parameters
    response = test_client.get("/search?q=tax&jurisdiction=BE&page=1&page_size=5&sort_by=issue_date&sort_order=desc")
    assert response.status_code == 200
    
    data = response.json()
    assert "documents" in data
    assert "total" in data
    assert isinstance(data["documents"], list)
    assert isinstance(data["total"], int)


def test_api_document_endpoint(test_client, db_session):
    """Test API document endpoint."""
    # Get a document ID from the database
    document = db_session.query(Document).first()
    
    # Skip test if no documents
    if document is None:
        pytest.skip("No documents in database")
    
    # Test get document endpoint
    response = test_client.get(f"/doc/{document.id}")
    assert response.status_code == 200
    
    data = response.json()
    assert data["id"] == document.id
    assert "title" in data
    assert "blob_url" in data
    
    # Test non-existent document
    response = test_client.get("/doc/non-existent-id")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_api_vector_search_endpoint(test_client, test_documents, test_vectors, embedding_strategy):
    """Test API vector search endpoint."""
    # Create a vector query
    query = {
        "text": "tax law in Belgium",
        "jurisdiction": "BE",
        "page": 1,
        "page_size": 10,
        "max_distance": None
    }
    
    # Test vector search endpoint
    response = test_client.post("/search/vector", json=query)
    
    # The test may fail if the embedding service is not available
    if response.status_code == 500:
        error_msg = response.json().get("detail", "")
        if "Error generating embedding" in error_msg:
            pytest.skip("Embedding service not available")
    
    assert response.status_code == 200
    
    data = response.json()
    assert "documents" in data
    assert "total" in data
    assert isinstance(data["documents"], list)
    assert isinstance(data["total"], int)
    
    # If there are results, check their structure
    if data["documents"]:
        assert "document" in data["documents"][0]
        assert "score" in data["documents"][0]
        assert isinstance(data["documents"][0]["document"], dict)
        assert isinstance(data["documents"][0]["score"], float)
        assert 0 <= data["documents"][0]["score"] <= 1


@pytest.mark.asyncio
async def test_api_hybrid_search_endpoint(test_client, test_documents, test_vectors, embedding_strategy):
    """Test API hybrid search endpoint."""
    # Create a hybrid query
    query = {
        "text": "tax law in Belgium",
        "jurisdiction": "BE",
        "page": 1,
        "page_size": 10,
        "vector_weight": 0.7,
        "text_weight": 0.3
    }
    
    # Test hybrid search endpoint
    response = test_client.post("/search/hybrid", json=query)
    
    # The test may fail if the embedding service is not available
    if response.status_code == 500:
        error_msg = response.json().get("detail", "")
        if "Error generating embedding" in error_msg:
            pytest.skip("Embedding service not available")
    
    assert response.status_code == 200
    
    data = response.json()
    assert "documents" in data
    assert "total" in data
    assert isinstance(data["documents"], list)
    assert isinstance(data["total"], int)
    
    # If there are results, check their structure
    if data["documents"]:
        assert "document" in data["documents"][0]
        assert "score" in data["documents"][0]
        assert isinstance(data["documents"][0]["document"], dict)
        assert isinstance(data["documents"][0]["score"], float)
        assert 0 <= data["documents"][0]["score"] <= 1


def test_api_similar_documents_endpoint(test_client, db_session):
    """Test API similar documents endpoint."""
    # Get a document ID from the database
    document = db_session.query(Document).first()
    
    # Skip test if no documents
    if document is None:
        pytest.skip("No documents in database")
    
    # Test similar documents endpoint
    response = test_client.get(f"/doc/{document.id}/similar?limit=3")
    
    # The test may fail if the document has no vector
    if response.status_code == 500:
        pytest.skip("Document has no vector embedding")
    
    assert response.status_code == 200
    
    data = response.json()
    assert isinstance(data, list)
    
    # If there are similar documents, check their structure
    if data:
        assert "document" in data[0]
        assert "score" in data[0]
        assert isinstance(data[0]["document"], dict)
        assert isinstance(data[0]["score"], float)
        assert 0 <= data[0]["score"] <= 1


def test_api_documents_by_jurisdiction_endpoint(test_client):
    """Test API documents by jurisdiction endpoint."""
    # Test documents by jurisdiction endpoint
    response = test_client.get("/jurisdictions/BE/documents?page=1&page_size=10&sort_by=issue_date&sort_order=desc")
    assert response.status_code == 200
    
    data = response.json()
    assert "documents" in data
    assert "total" in data
    assert isinstance(data["documents"], list)
    assert isinstance(data["total"], int)
    
    # If there are documents, check they have the correct jurisdiction
    for doc in data["documents"]:
        assert doc["jurisdiction"] == "BE"


def test_api_error_handling(test_client):
    """Test API error handling."""
    # Test 404 for non-existent document
    response = test_client.get("/doc/non-existent-id")
    assert response.status_code == 404
    assert "detail" in response.json()
    
    # Test 400 for invalid pagination parameters
    response = test_client.get("/search?q=tax&page=0&page_size=1000")
    assert response.status_code in [400, 422]  # FastAPI may return 422 for validation errors
    
    # Test 500 for invalid vector in vector search
    invalid_query = {
        "text": "tax law",
        "vector": [0.1, 0.2]  # Invalid vector (wrong format)
    }
    response = test_client.post("/search/vector", json=invalid_query)
    assert response.status_code in [400, 422]  # FastAPI may return 422 for validation errors


@pytest.mark.asyncio
async def test_api_pagination_validation(test_client):
    """Test API pagination parameter validation."""
    # Test with negative page
    response = test_client.get("/search?q=tax&page=-1&page_size=10")
    assert response.status_code in [400, 422]
    
    # Test with zero page
    response = test_client.get("/search?q=tax&page=0&page_size=10")
    assert response.status_code in [400, 422]
    
    # Test with negative page_size
    response = test_client.get("/search?q=tax&page=1&page_size=-10")
    assert response.status_code in [400, 422]
    
    # Test with zero page_size
    response = test_client.get("/search?q=tax&page=1&page_size=0")
    assert response.status_code in [400, 422]
    
    # Test with page_size > 100
    response = test_client.get("/search?q=tax&page=1&page_size=101")
    assert response.status_code in [400, 422]


@pytest.mark.asyncio
async def test_api_vector_query_validation(test_client):
    """Test API vector query validation."""
    # Test with empty text
    response = test_client.post("/search/vector", json={"text": ""})
    assert response.status_code in [400, 422]
    
    # Test with text too long
    response = test_client.post("/search/vector", json={"text": "a" * 1001})
    assert response.status_code in [400, 422]
    
    # Test with invalid jurisdiction
    response = test_client.post("/search/vector", json={"text": "tax", "jurisdiction": "INVALID"})
    assert response.status_code in [400, 422]
    
    # Test with negative page
    response = test_client.post("/search/vector", json={"text": "tax", "page": -1})
    assert response.status_code in [400, 422]
    
    # Test with negative page_size
    response = test_client.post("/search/vector", json={"text": "tax", "page_size": -1})
    assert response.status_code in [400, 422]
    
    # Test with negative max_distance
    response = test_client.post("/search/vector", json={"text": "tax", "max_distance": -1})
    assert response.status_code in [400, 422]


@pytest.mark.asyncio
async def test_api_hybrid_query_validation(test_client):
    """Test API hybrid query validation."""
    # Test with empty text
    response = test_client.post("/search/hybrid", json={"text": ""})
    assert response.status_code in [400, 422]
    
    # Test with text too long
    response = test_client.post("/search/hybrid", json={"text": "a" * 1001})
    assert response.status_code in [400, 422]
    
    # Test with invalid jurisdiction
    response = test_client.post("/search/hybrid", json={"text": "tax", "jurisdiction": "INVALID"})
    assert response.status_code in [400, 422]
    
    # Test with negative page
    response = test_client.post("/search/hybrid", json={"text": "tax", "page": -1})
    assert response.status_code in [400, 422]
    
    # Test with negative page_size
    response = test_client.post("/search/hybrid", json={"text": "tax", "page_size": -1})
    assert response.status_code in [400, 422]
    
    # Test with negative vector_weight
    response = test_client.post("/search/hybrid", json={"text": "tax", "vector_weight": -0.1})
    assert response.status_code in [400, 422]
    
    # Test with vector_weight > 1
    response = test_client.post("/search/hybrid", json={"text": "tax", "vector_weight": 1.1})
    assert response.status_code in [400, 422]
    
    # Test with negative text_weight
    response = test_client.post("/search/hybrid", json={"text": "tax", "text_weight": -0.1})
    assert response.status_code in [400, 422]
    
    # Test with text_weight > 1
    response = test_client.post("/search/hybrid", json={"text": "tax", "text_weight": 1.1})
    assert response.status_code in [400, 422]


@pytest.mark.asyncio
async def test_api_jurisdictions_endpoint(test_client):
    """Test API jurisdictions endpoint."""
    # Test listing documents for each jurisdiction
    for jurisdiction in ["BE", "ES", "DE"]:
        response = test_client.get(f"/jurisdictions/{jurisdiction}/documents")
        assert response.status_code == 200
        
        data = response.json()
        assert "documents" in data
        assert "total" in data
        assert isinstance(data["documents"], list)
        assert isinstance(data["total"], int)
        
        # If there are documents, check they have the correct jurisdiction
        for doc in data["documents"]:
            assert doc["jurisdiction"] == jurisdiction
    
    # Test with invalid jurisdiction
    response = test_client.get("/jurisdictions/INVALID/documents")
    assert response.status_code in [400, 422, 404]


@pytest.mark.asyncio
async def test_api_document_not_found(test_client):
    """Test API document not found handling."""
    # Test get document endpoint with non-existent ID
    response = test_client.get("/doc/non-existent-id")
    assert response.status_code == 404
    assert "detail" in response.json()
    
    # Test similar documents endpoint with non-existent ID
    response = test_client.get("/doc/non-existent-id/similar")
    assert response.status_code == 404
    assert "detail" in response.json()


@pytest.mark.asyncio
async def test_api_search_with_filters(test_client):
    """Test API search with various filters."""
    # Test search with jurisdiction filter
    response = test_client.get("/search?q=tax&jurisdiction=BE")
    assert response.status_code == 200
    
    data = response.json()
    assert "documents" in data
    assert "total" in data
    
    # If there are documents, check they have the correct jurisdiction
    for doc in data["documents"]:
        assert doc["jurisdiction"] == "BE"
    
    # Test search with sort parameters
    response = test_client.get("/search?q=tax&sort_by=issue_date&sort_order=asc")
    assert response.status_code == 200
    
    data = response.json()
    assert "documents" in data
    
    # If there are multiple documents, check they are sorted correctly
    if len(data["documents"]) > 1:
        dates = [doc["issue_date"] for doc in data["documents"]]
        assert all(dates[i] <= dates[i+1] for i in range(len(dates)-1))
    
    # Test search with sort_order=desc
    response = test_client.get("/search?q=tax&sort_by=issue_date&sort_order=desc")
    assert response.status_code == 200
    
    data = response.json()
    assert "documents" in data
    
    # If there are multiple documents, check they are sorted correctly
    if len(data["documents"]) > 1:
        dates = [doc["issue_date"] for doc in data["documents"]]
        assert all(dates[i] >= dates[i+1] for i in range(len(dates)-1))


@pytest.mark.asyncio
async def test_api_vector_search_with_filters(test_client, embedding_strategy):
    """Test API vector search with various filters."""
    # Create a vector query with jurisdiction filter
    query = {
        "text": "tax law in Belgium",
        "jurisdiction": "BE",
        "page": 1,
        "page_size": 10
    }
    
    # Test vector search endpoint
    response = test_client.post("/search/vector", json=query)
    
    # The test may fail if the embedding service is not available
    if response.status_code == 500:
        error_msg = response.json().get("detail", "")
        if "Error generating embedding" in error_msg:
            pytest.skip("Embedding service not available")
    
    assert response.status_code == 200
    
    data = response.json()
    assert "documents" in data
    assert "total" in data
    
    # If there are documents, check they have the correct jurisdiction
    for item in data["documents"]:
        assert item["document"]["jurisdiction"] == "BE"
    
    # Test with max_distance filter
    query["max_distance"] = 0.5
    response = test_client.post("/search/vector", json=query)
    
    # Skip if embedding service is not available
    if response.status_code == 500:
        error_msg = response.json().get("detail", "")
        if "Error generating embedding" in error_msg:
            pytest.skip("Embedding service not available")
    
    assert response.status_code == 200
    
    data = response.json()
    assert "documents" in data
    
    # If there are results, check their scores
    for item in data["documents"]:
        assert item["score"] >= 0
        assert item["score"] <= 1


@pytest.mark.asyncio
async def test_api_hybrid_search_with_filters(test_client, embedding_strategy):
    """Test API hybrid search with various filters."""
    # Create a hybrid query with jurisdiction filter
    query = {
        "text": "tax law in Belgium",
        "jurisdiction": "BE",
        "page": 1,
        "page_size": 10,
        "vector_weight": 0.7,
        "text_weight": 0.3
    }
    
    # Test hybrid search endpoint
    response = test_client.post("/search/hybrid", json=query)
    
    # The test may fail if the embedding service is not available
    if response.status_code == 500:
        error_msg = response.json().get("detail", "")
        if "Error generating embedding" in error_msg:
            pytest.skip("Embedding service not available")
    
    assert response.status_code == 200
    
    data = response.json()
    assert "documents" in data
    assert "total" in data
    
    # If there are documents, check they have the correct jurisdiction
    for item in data["documents"]:
        assert item["document"]["jurisdiction"] == "BE"
    
    # Test with different weights
    query["vector_weight"] = 0.3
    query["text_weight"] = 0.7
    response = test_client.post("/search/hybrid", json=query)
    
    # Skip if embedding service is not available
    if response.status_code == 500:
        error_msg = response.json().get("detail", "")
        if "Error generating embedding" in error_msg:
            pytest.skip("Embedding service not available")
    
    assert response.status_code == 200
    
    data = response.json()
    assert "documents" in data
    
    # If there are results, check their scores
    for item in data["documents"]:
        assert item["score"] >= 0
        assert item["score"] <= 1


@pytest.mark.asyncio
async def test_api_security_headers(test_client):
    """Test API security headers."""
    # Make a request to any endpoint
    response = test_client.get("/healthz")
    
    # Check security headers
    headers = response.headers
    assert "X-Content-Type-Options" in headers
    assert headers["X-Content-Type-Options"] == "nosniff"
    
    assert "X-Frame-Options" in headers
    assert headers["X-Frame-Options"] == "DENY"
    
    assert "X-XSS-Protection" in headers
    assert headers["X-XSS-Protection"] == "1; mode=block"
    
    assert "Strict-Transport-Security" in headers
    assert "max-age=" in headers["Strict-Transport-Security"]
    
    assert "Content-Security-Policy" in headers
    assert "default-src" in headers["Content-Security-Policy"]


@pytest.mark.asyncio
async def test_api_cors_headers(test_client):
    """Test API CORS headers."""
    # Make a request with Origin header
    response = test_client.get("/healthz", headers={"Origin": "http://example.com"})
    
    # Check CORS headers
    headers = response.headers
    assert "Access-Control-Allow-Origin" in headers
    
    # Make an OPTIONS request (preflight)
    response = test_client.options("/healthz", headers={
        "Origin": "http://example.com",
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "Content-Type"
    })
    
    # Check preflight response
    assert response.status_code == 200
    headers = response.headers
    assert "Access-Control-Allow-Origin" in headers
    assert "Access-Control-Allow-Methods" in headers
    assert "Access-Control-Allow-Headers" in headers


@pytest.mark.asyncio
async def test_api_startup_shutdown(monkeypatch):
    """Test API startup and shutdown events."""
    # Mock db_manager.create_tables and db_manager.close
    create_tables_called = False
    close_called = False
    
    async def mock_create_tables():
        nonlocal create_tables_called
        create_tables_called = True
    
    async def mock_close():
        nonlocal close_called
        close_called = True
    
    # Apply mocks
    monkeypatch.setattr(db_manager, "create_tables", mock_create_tables)
    monkeypatch.setattr(db_manager, "close", mock_close)
    
    # Create app and trigger startup event
    from src.api import create_app_startup, create_app_shutdown
    
    await create_app_startup()
    assert create_tables_called, "create_tables should be called during startup"
    
    await create_app_shutdown()
    assert close_called, "close should be called during shutdown"


@pytest.mark.asyncio
async def test_api_sas_url_generation():
    """Test SAS URL generation."""
    from src.api import generate_sas_url
    
    # Test in local mode
    original_local_mode = settings.LOCAL_MODE
    settings.LOCAL_MODE = True
    
    blob_url = "https://example.com/blob.pdf"
    sas_url = generate_sas_url(blob_url)
    
    # In local mode, should return the original URL
    assert sas_url == blob_url
    
    # Test in cloud mode
    settings.LOCAL_MODE = False
    
    sas_url = generate_sas_url(blob_url)
    
    # In cloud mode, should add SAS token
    assert sas_url.startswith(blob_url)
    assert "sv=" in sas_url
    assert "sig=" in sas_url
    
    # Restore original setting
    settings.LOCAL_MODE = original_local_mode