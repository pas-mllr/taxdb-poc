"""
Tests for document storage with vector embeddings.

This module tests the storage and retrieval of documents with vector embeddings,
focusing on vector similarity search and indexing performance.
"""

import asyncio
import numpy as np
import os
import pytest
import pytest_asyncio
import time
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

from pgvector.sqlalchemy import Vector
from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import Document
from src.db import db_manager
from src.repository import (
    DocumentRepository,
    PaginationParams,
    SortParams,
    EntityNotFoundError
)
from src.etl.utils import get_embedding_strategy, EmbeddingStrategy


@pytest.mark.asyncio
async def test_vector_storage_basic(async_session, clean_test_database, generate_test_documents):
    """Test basic vector storage functionality."""
    # Generate test documents with vectors
    docs = generate_test_documents(count=5, with_vectors=True)
    
    # Insert documents
    for doc in docs:
        document = Document(**doc)
        async_session.add(document)
    
    await async_session.commit()
    
    try:
        # Query to verify vectors were stored
        result = await async_session.execute(
            select(Document).where(Document.vector.is_not(None))
        )
        documents = result.scalars().all()
        
        # Check results
        assert len(documents) == 5
        for doc in documents:
            assert doc.vector is not None
            assert len(doc.vector) == 1536  # Check vector dimensions
    
    finally:
        # Clean up
        await async_session.execute(text("TRUNCATE TABLE documents CASCADE"))
        await async_session.commit()


@pytest.mark.asyncio
async def test_vector_similarity_search(async_session, clean_test_database, generate_test_documents):
    """Test vector similarity search."""
    # Generate test documents with vectors
    docs = generate_test_documents(count=10, with_vectors=True)
    
    # Insert documents
    for doc in docs:
        document = Document(**doc)
        async_session.add(document)
    
    await async_session.commit()
    
    try:
        # Create a query vector
        query_vector = np.random.rand(1536).tolist()
        
        # Perform vector similarity search using L2 distance
        result = await async_session.execute(
            select(
                Document,
                func.l2_distance(Document.vector, query_vector).label("distance")
            ).order_by(
                func.l2_distance(Document.vector, query_vector)
            ).limit(5)
        )
        
        documents_with_distances = result.all()
        
        # Check results
        assert len(documents_with_distances) == 5
        
        # Verify distances are in ascending order (closest first)
        distances = [distance for _, distance in documents_with_distances]
        assert all(distances[i] <= distances[i+1] for i in range(len(distances)-1))
        
        # Perform vector similarity search using cosine distance
        result = await async_session.execute(
            select(
                Document,
                func.cosine_distance(Document.vector, query_vector).label("distance")
            ).order_by(
                func.cosine_distance(Document.vector, query_vector)
            ).limit(5)
        )
        
        documents_with_distances = result.all()
        
        # Check results
        assert len(documents_with_distances) == 5
        
        # Verify distances are in ascending order (closest first)
        distances = [distance for _, distance in documents_with_distances]
        assert all(distances[i] <= distances[i+1] for i in range(len(distances)-1))
        
        # Perform vector similarity search with jurisdiction filter
        result = await async_session.execute(
            select(
                Document,
                func.l2_distance(Document.vector, query_vector).label("distance")
            ).where(
                Document.jurisdiction == "BE"
            ).order_by(
                func.l2_distance(Document.vector, query_vector)
            ).limit(5)
        )
        
        documents_with_distances = result.all()
        
        # Check that all documents have the correct jurisdiction
        for doc, _ in documents_with_distances:
            assert doc.jurisdiction == "BE"
    
    finally:
        # Clean up
        await async_session.execute(text("TRUNCATE TABLE documents CASCADE"))
        await async_session.commit()


@pytest.mark.asyncio
async def test_vector_operations(async_session, clean_test_database):
    """Test vector operations."""
    # Create test vectors
    vector1 = np.random.rand(1536).tolist()
    vector2 = np.random.rand(1536).tolist()
    
    # Calculate expected distances
    expected_l2 = np.linalg.norm(np.array(vector1) - np.array(vector2))
    
    # Calculate cosine similarity
    cos_sim = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    expected_cosine = 1 - cos_sim
    
    # Calculate distances using PostgreSQL functions
    result = await async_session.execute(
        select(
            func.l2_distance(Vector(vector1), Vector(vector2)),
            func.cosine_distance(Vector(vector1), Vector(vector2))
        )
    )
    
    l2_distance, cosine_distance = result.first()
    
    # Check results (with some tolerance for floating-point differences)
    assert abs(l2_distance - expected_l2) < 1e-5
    assert abs(cosine_distance - expected_cosine) < 1e-5


@pytest.mark.asyncio
async def test_repository_vector_search(async_session, clean_test_database, document_repository, generate_test_documents):
    """Test repository vector search methods."""
    # Generate test documents with vectors
    docs = generate_test_documents(count=10, with_vectors=True)
    
    # Insert documents using repository
    for doc in docs:
        await document_repository.create(async_session, doc)
    
    await async_session.commit()
    
    try:
        # Create a query vector
        query_vector = np.random.rand(1536).tolist()
        
        # Search by vector
        results, total = await document_repository.search_by_vector(
            async_session,
            query_vector=query_vector,
            pagination=PaginationParams(page=1, page_size=5)
        )
        
        # Check results
        assert len(results) == 5
        assert total == 10
        
        # Verify results are tuples of (document, distance)
        for doc, distance in results:
            assert isinstance(doc, Document)
            assert isinstance(distance, float)
        
        # Verify distances are in ascending order (closest first)
        distances = [distance for _, distance in results]
        assert all(distances[i] <= distances[i+1] for i in range(len(distances)-1))
        
        # Search by vector with jurisdiction filter
        results, total = await document_repository.search_by_vector(
            async_session,
            query_vector=query_vector,
            jurisdiction="BE",
            pagination=PaginationParams(page=1, page_size=5)
        )
        
        # Check that all documents have the correct jurisdiction
        for doc, _ in results:
            assert doc.jurisdiction == "BE"
        
        # Search by vector with max_distance
        max_distance = 0.5
        results, total = await document_repository.search_by_vector(
            async_session,
            query_vector=query_vector,
            max_distance=max_distance,
            pagination=PaginationParams(page=1, page_size=5)
        )
        
        # Check that all distances are below max_distance
        for _, distance in results:
            assert distance < max_distance
    
    finally:
        # Clean up
        await async_session.execute(text("TRUNCATE TABLE documents CASCADE"))
        await async_session.commit()


@pytest.mark.asyncio
async def test_hybrid_search(async_session, clean_test_database, document_repository, generate_test_documents):
    """Test hybrid search (vector + text)."""
    # Generate test documents with vectors
    docs = generate_test_documents(count=10, with_vectors=True)
    
    # Add specific keywords to some documents
    docs[0]["title"] = "Tax law document with specific keyword"
    docs[1]["title"] = "Another document with keyword"
    docs[2]["summary"] = "This summary contains the keyword"
    
    # Insert documents using repository
    for doc in docs:
        await document_repository.create(async_session, doc)
    
    await async_session.commit()
    
    try:
        # Create a query vector
        query_vector = np.random.rand(1536).tolist()
        
        # Perform hybrid search
        results, total = await document_repository.search_hybrid(
            async_session,
            query="keyword",
            query_vector=query_vector,
            pagination=PaginationParams(page=1, page_size=5),
            vector_weight=0.5,
            text_weight=0.5
        )
        
        # Check results
        assert len(results) == 5
        
        # Verify results are tuples of (document, score)
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, float)
            assert 0 <= score <= 1  # Score should be between 0 and 1
        
        # Documents with "keyword" should be ranked higher
        keyword_docs = [doc for doc, _ in results if "keyword" in doc.title.lower() or 
                        (doc.summary and "keyword" in doc.summary.lower())]
        assert len(keyword_docs) > 0
        
        # Test with different weights
        results_vector, _ = await document_repository.search_hybrid(
            async_session,
            query="keyword",
            query_vector=query_vector,
            pagination=PaginationParams(page=1, page_size=5),
            vector_weight=0.9,
            text_weight=0.1
        )
        
        results_text, _ = await document_repository.search_hybrid(
            async_session,
            query="keyword",
            query_vector=query_vector,
            pagination=PaginationParams(page=1, page_size=5),
            vector_weight=0.1,
            text_weight=0.9
        )
        
        # The rankings should be different
        assert [doc.id for doc, _ in results_vector] != [doc.id for doc, _ in results_text]
    
    finally:
        # Clean up
        await async_session.execute(text("TRUNCATE TABLE documents CASCADE"))
        await async_session.commit()


@pytest.mark.asyncio
async def test_similar_documents(async_session, clean_test_database, document_repository, generate_test_documents):
    """Test similar documents retrieval."""
    # Generate test documents with vectors
    docs = generate_test_documents(count=10, with_vectors=True)
    
    # Insert documents using repository
    created_docs = []
    for doc in docs:
        created_doc = await document_repository.create(async_session, doc)
        created_docs.append(created_doc)
    
    await async_session.commit()
    
    try:
        # Get similar documents for the first document
        source_doc = created_docs[0]
        similar_docs = await document_repository.get_similar_documents(
            async_session,
            document_id=source_doc.id,
            limit=5
        )
        
        # Check results
        assert len(similar_docs) == 5
        
        # Verify results are tuples of (document, score)
        for doc, score in similar_docs:
            assert isinstance(doc, Document)
            assert isinstance(score, float)
        
        # Source document should not be in the results
        assert all(doc.id != source_doc.id for doc, _ in similar_docs)
        
        # Test with non-existent document
        with pytest.raises(EntityNotFoundError):
            await document_repository.get_similar_documents(
                async_session,
                document_id="NON_EXISTENT_ID",
                limit=5
            )
    
    finally:
        # Clean up
        await async_session.execute(text("TRUNCATE TABLE documents CASCADE"))
        await async_session.commit()


@pytest.mark.asyncio
async def test_vector_index_performance(async_session, clean_test_database, generate_test_documents):
    """Test vector index performance."""
    # Skip in CI environment
    if "CI" in os.environ:
        pytest.skip("Skipping performance test in CI environment")
    
    # Generate a larger set of test documents
    doc_count = 100
    docs = generate_test_documents(count=doc_count, with_vectors=True)
    
    # Insert documents
    for doc in docs:
        document = Document(**doc)
        async_session.add(document)
    
    await async_session.commit()
    
    try:
        # Create a query vector
        query_vector = np.random.rand(1536).tolist()
        
        # Measure time for vector search
        start_time = time.time()
        
        result = await async_session.execute(
            select(
                Document,
                func.l2_distance(Document.vector, query_vector).label("distance")
            ).order_by(
                func.l2_distance(Document.vector, query_vector)
            ).limit(10)
        )
        
        documents_with_distances = result.all()
        
        search_time = time.time() - start_time
        
        # Check results
        assert len(documents_with_distances) == 10
        
        # Log performance
        print(f"Vector search time for {doc_count} documents: {search_time:.4f} seconds")
        
        # Performance should be reasonable (adjust threshold based on hardware)
        assert search_time < 1.0  # Should be fast with proper indexing
    
    finally:
        # Clean up
        await async_session.execute(text("TRUNCATE TABLE documents CASCADE"))
        await async_session.commit()


@pytest.mark.asyncio
async def test_embedding_generation(async_session, clean_test_database, embedding_strategy):
    """Test embedding generation and storage."""
    # Create a test document
    doc = Document(
        id="TEST:20250801:DOC-001",
        jurisdiction="BE",
        source_system="test",
        document_type="law",
        title="Test Tax Law",
        summary="A test tax law document",
        issue_date=date(2025, 8, 1),
        language_orig="en",
        blob_url="http://example.com/test.pdf",
        checksum="test_checksum"
    )
    
    # Generate embedding
    text = f"{doc.title} {doc.summary}"
    doc.vector = await embedding_strategy.embed(text)
    
    # Save document
    async_session.add(doc)
    await async_session.commit()
    
    try:
        # Retrieve document
        result = await async_session.execute(
            select(Document).where(Document.id == doc.id)
        )
        retrieved_doc = result.scalars().first()
        
        # Check vector
        assert retrieved_doc.vector is not None
        assert len(retrieved_doc.vector) == embedding_strategy.get_dimensions()
    
    finally:
        # Clean up
        await async_session.delete(doc)
        await async_session.commit()


@pytest.mark.asyncio
async def test_vector_nulls_handling(async_session, clean_test_database, document_repository):
    """Test handling of documents with null vectors."""
    # Create documents with and without vectors
    doc_with_vector = {
        "id": "TEST:20250801:DOC-001",
        "jurisdiction": "BE",
        "source_system": "test",
        "document_type": "law",
        "title": "Test Document With Vector",
        "issue_date": date(2025, 8, 1),
        "language_orig": "en",
        "blob_url": "http://example.com/test1.pdf",
        "checksum": "test1_checksum",
        "vector": np.random.rand(1536).tolist()
    }
    
    doc_without_vector = {
        "id": "TEST:20250801:DOC-002",
        "jurisdiction": "BE",
        "source_system": "test",
        "document_type": "law",
        "title": "Test Document Without Vector",
        "issue_date": date(2025, 8, 1),
        "language_orig": "en",
        "blob_url": "http://example.com/test2.pdf",
        "checksum": "test2_checksum"
        # No vector
    }
    
    # Insert documents
    await document_repository.create(async_session, doc_with_vector)
    await document_repository.create(async_session, doc_without_vector)
    await async_session.commit()
    
    try:
        # Query for documents with vectors
        result = await async_session.execute(
            select(Document).where(Document.vector.is_not(None))
        )
        docs_with_vectors = result.scalars().all()
        
        # Check results
        assert len(docs_with_vectors) == 1
        assert docs_with_vectors[0].id == doc_with_vector["id"]
        
        # Query for documents without vectors
        result = await async_session.execute(
            select(Document).where(Document.vector.is_(None))
        )
        docs_without_vectors = result.scalars().all()
        
        # Check results
        assert len(docs_without_vectors) == 1
        assert docs_without_vectors[0].id == doc_without_vector["id"]
        
        # Vector search should only include documents with vectors
        query_vector = np.random.rand(1536).tolist()
        results, total = await document_repository.search_by_vector(
            async_session,
            query_vector=query_vector,
            pagination=PaginationParams(page=1, page_size=10)
        )
        
        # Check results
        assert len(results) == 1
        assert results[0][0].id == doc_with_vector["id"]
    
    finally:
        # Clean up
        await async_session.execute(text("TRUNCATE TABLE documents CASCADE"))
        await async_session.commit()