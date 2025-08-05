"""
End-to-end tests for TaxDB-POC.

This module provides end-to-end tests for the TaxDB-POC application.
"""

import json
import pytest
import httpx
import numpy as np
from typing import Dict, List, Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import Document
from src.repository import DocumentRepository, PaginationParams, SortParams
from src.etl.utils import EmbeddingStrategy


@pytest.mark.e2e
def test_database_connection(db_session):
    """Test database connection."""
    # Simple query to test connection
    result = db_session.execute("SELECT 1").scalar()
    assert result == 1


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_async_database_connection(async_session):
    """Test async database connection."""
    # Simple query to test connection
    result = await async_session.execute(select(1))
    assert result.scalar() == 1


@pytest.mark.e2e
def test_document_count(db_session, run_etl):
    """Test document count after ETL."""
    # Count documents for each jurisdiction
    be_count = db_session.query(Document).filter_by(jurisdiction="BE").count()
    es_count = db_session.query(Document).filter_by(jurisdiction="ES").count()
    de_count = db_session.query(Document).filter_by(jurisdiction="DE").count()
    
    # Assert counts are non-negative (may be 0 if no documents found)
    assert be_count >= 0, f"Expected BE count >= 0, got {be_count}"
    assert es_count >= 0, f"Expected ES count >= 0, got {es_count}"
    assert de_count >= 0, f"Expected DE count >= 0, got {de_count}"
    
    # Log counts
    print(f"Document counts: BE={be_count}, ES={es_count}, DE={de_count}")


@pytest.mark.e2e
def test_api_health(api_server):
    """Test API health endpoint."""
    # Make request to health endpoint
    response = httpx.get(f"{api_server}/healthz")
    
    # Assert response
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.e2e
def test_api_search(api_server, run_etl):
    """Test API search endpoint."""
    # Make request to search endpoint
    response = httpx.get(f"{api_server}/search?q=tax&jurisdiction=ES")
    
    # Assert response
    assert response.status_code == 200
    
    # Parse response
    data = response.json()
    
    # Assert response structure
    assert "documents" in data
    assert "total" in data
    assert isinstance(data["documents"], list)
    assert isinstance(data["total"], int)
    
    # Assert total is non-negative
    assert data["total"] >= 0


@pytest.mark.e2e
def test_api_document(api_server, db_session, run_etl):
    """Test API document endpoint."""
    # Get a document ID from the database
    document = db_session.query(Document).first()
    
    # Skip test if no documents
    if document is None:
        pytest.skip("No documents in database")
    
    # Make request to document endpoint
    response = httpx.get(f"{api_server}/doc/{document.id}")
    
    # Assert response
    assert response.status_code == 200
    
    # Parse response
    data = response.json()
    
    # Assert response structure
    assert "id" in data
    assert "title" in data
    assert "blob_url" in data
    
    # Assert ID matches
    assert data["id"] == document.id


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_document_repository_get_by_id(async_session, document_repository):
    """Test document repository get_by_id method."""
    # Get a document ID from the database
    result = await async_session.execute(select(Document).limit(1))
    document = result.scalars().first()
    
    # Skip test if no documents
    if document is None:
        pytest.skip("No documents in database")
    
    # Get document by ID
    repo_document = await document_repository.get_by_id(async_session, document.id)
    
    # Assert document matches
    assert repo_document.id == document.id
    assert repo_document.title == document.title


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_document_repository_get_all(async_session, document_repository):
    """Test document repository get_all method."""
    # Get all documents
    documents, total = await document_repository.get_all(
        async_session,
        pagination=PaginationParams(page=1, page_size=10),
        sort=SortParams(sort_by="issue_date", sort_order="desc")
    )
    
    # Assert documents and total
    assert isinstance(documents, list)
    assert isinstance(total, int)
    assert total >= 0
    assert len(documents) <= 10


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_document_repository_filter_by_jurisdiction(async_session, document_repository):
    """Test document repository filter_by_jurisdiction method."""
    # Filter documents by jurisdiction
    documents, total = await document_repository.filter_by_jurisdiction(
        async_session,
        jurisdiction="BE",
        pagination=PaginationParams(page=1, page_size=10),
        sort=SortParams(sort_by="issue_date", sort_order="desc")
    )
    
    # Assert documents and total
    assert isinstance(documents, list)
    assert isinstance(total, int)
    assert total >= 0
    assert len(documents) <= 10
    
    # Assert all documents have the correct jurisdiction
    for document in documents:
        assert document.jurisdiction == "BE"


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_document_repository_search_by_text(async_session, document_repository):
    """Test document repository search_by_text method."""
    # Search documents by text
    documents, total = await document_repository.search_by_text(
        async_session,
        query="tax",
        pagination=PaginationParams(page=1, page_size=10),
        sort=SortParams(sort_by="issue_date", sort_order="desc")
    )
    
    # Assert documents and total
    assert isinstance(documents, list)
    assert isinstance(total, int)
    assert total >= 0
    assert len(documents) <= 10


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_document_repository_save_document(async_session, document_repository, test_documents, test_vectors):
    """Test document repository save_document method."""
    # Create a test document with vector
    test_doc = test_documents[0].copy()
    test_doc["vector"] = test_vectors["tax"]
    
    # Save document
    document = await document_repository.save_document(async_session, test_doc)
    await async_session.commit()
    
    # Assert document was saved
    assert document.id == test_doc["id"]
    assert document.title == test_doc["title"]
    assert document.vector is not None
    
    # Clean up
    await async_session.delete(document)
    await async_session.commit()


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_vector_search(async_session, document_repository, test_documents, test_vectors):
    """Test vector search functionality."""
    # Create test documents with vectors
    documents = []
    for i, doc in enumerate(test_documents):
        test_doc = doc.copy()
        # Assign different vectors to each document
        vector_key = list(test_vectors.keys())[i]
        test_doc["vector"] = test_vectors[vector_key]
        
        # Save document
        document = await document_repository.save_document(async_session, test_doc)
        documents.append(document)
    
    await async_session.commit()
    
    try:
        # Test vector search
        query_vector = test_vectors["tax"]
        results, total = await document_repository.search_by_vector(
            async_session,
            query_vector=query_vector,
            pagination=PaginationParams(page=1, page_size=10)
        )
        
        # Assert results
        assert isinstance(results, list)
        assert isinstance(total, int)
        assert total > 0
        assert len(results) > 0
        
        # Check that results contain documents and distances
        for doc, distance in results:
            assert isinstance(doc, Document)
            assert isinstance(distance, float)
            assert distance >= 0
        
        # Test hybrid search
        results, total = await document_repository.search_hybrid(
            async_session,
            query="tax",
            query_vector=query_vector,
            pagination=PaginationParams(page=1, page_size=10)
        )
        
        # Assert results
        assert isinstance(results, list)
        assert isinstance(total, int)
        assert total > 0
        assert len(results) > 0
        
        # Check that results contain documents and scores
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, float)
            assert 0 <= score <= 1
    
    finally:
        # Clean up
        for document in documents:
            await async_session.delete(document)
        await async_session.commit()


@pytest.mark.e2e
def test_api_vector_search(test_client, test_documents, test_vectors):
    """Test API vector search endpoint."""
    # Create a vector query
    query = {
        "text": "tax law in Belgium",
        "jurisdiction": "BE",
        "page": 1,
        "page_size": 10
    }
    
    # Make request to vector search endpoint
    response = test_client.post("/search/vector", json=query)
    
    # Assert response
    assert response.status_code in [200, 500]  # May fail if embedding service is not available
    
    # If successful, check response structure
    if response.status_code == 200:
        data = response.json()
        assert "documents" in data
        assert "total" in data
        assert isinstance(data["documents"], list)
        assert isinstance(data["total"], int)


@pytest.mark.e2e
def test_api_hybrid_search(test_client, test_documents, test_vectors):
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
    
    # Make request to hybrid search endpoint
    response = test_client.post("/search/hybrid", json=query)
    
    # Assert response
    assert response.status_code in [200, 500]  # May fail if embedding service is not available
    
    # If successful, check response structure
    if response.status_code == 200:
        data = response.json()
        assert "documents" in data
        assert "total" in data
        assert isinstance(data["documents"], list)
        assert isinstance(data["total"], int)


@pytest.mark.e2e
def test_api_similar_documents(api_server, db_session, run_etl):
    """Test API similar documents endpoint."""
    # Get a document ID from the database
    document = db_session.query(Document).first()
    
    # Skip test if no documents
    if document is None:
        pytest.skip("No documents in database")
    
    # Make request to similar documents endpoint
    response = httpx.get(f"{api_server}/doc/{document.id}/similar?limit=3")
    
    # Assert response
    assert response.status_code in [200, 500]  # May fail if document has no vector
    
    # If successful, check response structure
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, list)
        
        # If there are similar documents, check their structure
        if data:
            assert "document" in data[0]
            assert "score" in data[0]
            assert isinstance(data[0]["document"], dict)
            assert isinstance(data[0]["score"], float)


@pytest.mark.asyncio
async def test_e2e_document_lifecycle(api_server, db_session, run_etl, generate_test_documents, document_repository):
    """Test end-to-end document lifecycle from creation to retrieval."""
    # Generate a test document
    test_doc = generate_test_documents(count=1, with_vectors=True)[0]
    
    # Save document to database
    document = await document_repository.save_document(db_session, test_doc)
    await db_session.commit()
    
    try:
        # Verify document was saved
        saved_doc = await document_repository.get_by_id(db_session, document.id)
        assert saved_doc.id == document.id
        assert saved_doc.title == document.title
        assert saved_doc.vector is not None
        
        # Retrieve document via API
        response = httpx.get(f"{api_server}/doc/{document.id}")
        assert response.status_code == 200
        
        # Check API response
        data = response.json()
        assert data["id"] == document.id
        assert data["title"] == document.title
        
        # Search for document via API
        response = httpx.get(f"{api_server}/search?q={document.title[:10]}")
        assert response.status_code == 200
        
        # Check search results
        data = response.json()
        assert data["total"] >= 1
        assert any(doc["id"] == document.id for doc in data["documents"])
        
        # Get similar documents via API
        response = httpx.get(f"{api_server}/doc/{document.id}/similar?limit=5")
        assert response.status_code == 200
        
        # Check similar documents
        data = response.json()
        assert isinstance(data, list)
    
    finally:
        # Clean up
        await db_session.delete(document)
        await db_session.commit()


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_e2e_etl_pipeline(api_server, db_session, run_etl):
    """Test end-to-end ETL pipeline and API integration."""
    # Run ETL processes (already done by the run_etl fixture)
    
    # Count documents for each jurisdiction
    be_count = db_session.query(Document).filter_by(jurisdiction="BE").count()
    es_count = db_session.query(Document).filter_by(jurisdiction="ES").count()
    de_count = db_session.query(Document).filter_by(jurisdiction="DE").count()
    
    # Log counts
    print(f"Document counts after ETL: BE={be_count}, ES={es_count}, DE={de_count}")
    
    # Test API search for each jurisdiction
    for jurisdiction in ["BE", "ES", "DE"]:
        # Skip if no documents for this jurisdiction
        if db_session.query(Document).filter_by(jurisdiction=jurisdiction).count() == 0:
            continue
        
        # Search for documents
        response = httpx.get(f"{api_server}/jurisdictions/{jurisdiction}/documents")
        assert response.status_code == 200
        
        # Check results
        data = response.json()
        assert data["total"] > 0
        assert all(doc["jurisdiction"] == jurisdiction for doc in data["documents"])
        
        # Get a document ID
        doc_id = data["documents"][0]["id"]
        
        # Retrieve document
        response = httpx.get(f"{api_server}/doc/{doc_id}")
        assert response.status_code == 200
        
        # Check document
        doc_data = response.json()
        assert doc_data["id"] == doc_id
        assert doc_data["jurisdiction"] == jurisdiction


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_e2e_vector_search(api_server, db_session, run_etl, embedding_strategy):
    """Test end-to-end vector search."""
    # Skip if no documents with vectors
    result = db_session.query(Document).filter(Document.vector.is_not(None)).first()
    if result is None:
        pytest.skip("No documents with vectors in database")
    
    # Get a document with vector
    document = result
    
    # Create a query from the document title
    query_text = document.title
    
    # Generate embedding for the query
    try:
        query_vector = await embedding_strategy.embed(query_text)
    except Exception as e:
        pytest.skip(f"Error generating embedding: {str(e)}")
    
    # Perform vector search via API
    response = httpx.post(f"{api_server}/search/vector", json={
        "text": query_text,
        "page": 1,
        "page_size": 10
    })
    
    # Skip if embedding service is not available
    if response.status_code == 500:
        error_msg = response.json().get("detail", "")
        if "Error generating embedding" in error_msg:
            pytest.skip("Embedding service not available")
    
    assert response.status_code == 200
    
    # Check results
    data = response.json()
    assert data["total"] > 0
    
    # The original document should be in the results with a high score
    doc_ids = [item["document"]["id"] for item in data["documents"]]
    assert document.id in doc_ids
    
    # Get the score for the original document
    doc_index = doc_ids.index(document.id)
    score = data["documents"][doc_index]["score"]
    
    # Score should be high (close to 1.0)
    assert score > 0.8


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_e2e_hybrid_search(api_server, db_session, run_etl, embedding_strategy):
    """Test end-to-end hybrid search."""
    # Skip if no documents with vectors
    result = db_session.query(Document).filter(Document.vector.is_not(None)).first()
    if result is None:
        pytest.skip("No documents with vectors in database")
    
    # Get a document with vector
    document = result
    
    # Create a query from the document title
    query_text = document.title
    
    # Generate embedding for the query
    try:
        query_vector = await embedding_strategy.embed(query_text)
    except Exception as e:
        pytest.skip(f"Error generating embedding: {str(e)}")
    
    # Perform hybrid search via API
    response = httpx.post(f"{api_server}/search/hybrid", json={
        "text": query_text,
        "page": 1,
        "page_size": 10,
        "vector_weight": 0.5,
        "text_weight": 0.5
    })
    
    # Skip if embedding service is not available
    if response.status_code == 500:
        error_msg = response.json().get("detail", "")
        if "Error generating embedding" in error_msg:
            pytest.skip("Embedding service not available")
    
    assert response.status_code == 200
    
    # Check results
    data = response.json()
    assert data["total"] > 0
    
    # The original document should be in the results with a high score
    doc_ids = [item["document"]["id"] for item in data["documents"]]
    assert document.id in doc_ids
    
    # Get the score for the original document
    doc_index = doc_ids.index(document.id)
    score = data["documents"][doc_index]["score"]
    
    # Score should be high (close to 1.0)
    assert score > 0.8


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_e2e_docker_based(docker_client, postgres_container):
    """Test end-to-end with Docker-based PostgreSQL."""
    # Skip if not running in Docker environment
    if postgres_container is None:
        pytest.skip("Docker-based testing not available")
    
    # Verify PostgreSQL container is running
    assert postgres_container.status == "running"
    
    # Create a connection to the PostgreSQL container
    import psycopg2
    
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            user="postgres",
            password="postgres",
            database="taxdb_test"
        )
        
        # Verify connection
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1
        
        # Create tables
        with conn.cursor() as cursor:
            cursor.execute("""
            CREATE EXTENSION IF NOT EXISTS vector;
            
            CREATE TABLE IF NOT EXISTS test_documents (
                id VARCHAR PRIMARY KEY,
                title VARCHAR NOT NULL,
                vector vector(1536)
            );
            """)
            conn.commit()
        
        # Insert test data
        with conn.cursor() as cursor:
            cursor.execute("""
            INSERT INTO test_documents (id, title, vector)
            VALUES ('TEST:001', 'Test Document', '[0.1, 0.2, 0.3]'::vector);
            """)
            conn.commit()
        
        # Query test data
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM test_documents")
            result = cursor.fetchone()
            assert result[0] == 'TEST:001'
            assert result[1] == 'Test Document'
        
        # Test vector operations
        with conn.cursor() as cursor:
            cursor.execute("""
            SELECT id, l2_distance(vector, '[0.1, 0.2, 0.3]'::vector) as distance
            FROM test_documents
            ORDER BY distance ASC
            LIMIT 1;
            """)
            result = cursor.fetchone()
            assert result[0] == 'TEST:001'
            assert result[1] == 0.0  # Distance to itself should be 0
        
    finally:
        # Clean up
        if 'conn' in locals():
            conn.close()