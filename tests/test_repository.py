"""
Tests for repository classes.

This module provides tests for the repository classes and database connection.
"""

import os
import time
import asyncio
import logging
import pytest
import numpy as np
from typing import List, Dict, Any
from unittest.mock import MagicMock, AsyncMock

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from src.models import Document
from src.db import db_manager
from src.repository import (
    DocumentRepository,
    PaginationParams,
    SortParams,
    EntityNotFoundError
)

# Configure logging
logger = logging.getLogger("taxdb.test.repository")


@pytest.mark.asyncio
async def test_db_manager_session(async_session):
    """Test database manager session context manager."""
    async with db_manager.session() as session:
        # Simple query to test connection
        result = await session.execute(select(1))
        assert result.scalar() == 1


@pytest.mark.asyncio
async def test_document_repository_crud(async_session, document_repository, test_documents, test_vectors):
    """Test document repository CRUD operations."""
    # Create a test document with vector
    test_doc = test_documents[0].copy()
    test_doc["vector"] = test_vectors["tax"]
    
    # Create document
    document = await document_repository.create(async_session, test_doc)
    await async_session.commit()
    
    # Assert document was created
    assert document.id == test_doc["id"]
    assert document.title == test_doc["title"]
    assert document.vector is not None
    
    # Get document by ID
    retrieved_document = await document_repository.get_by_id(async_session, document.id)
    assert retrieved_document.id == document.id
    assert retrieved_document.title == document.title
    
    # Update document
    updated_data = {"title": "Updated Test Title"}
    updated_document = await document_repository.update(async_session, document.id, updated_data)
    await async_session.commit()
    
    assert updated_document.id == document.id
    assert updated_document.title == "Updated Test Title"
    
    # Delete document
    await document_repository.delete(async_session, document.id)
    await async_session.commit()
    
    # Verify document was deleted
    with pytest.raises(EntityNotFoundError):
        await document_repository.get_by_id(async_session, document.id)


@pytest.mark.asyncio
async def test_pagination_and_sorting(async_session, document_repository, test_documents, test_vectors):
    """Test pagination and sorting functionality."""
    # Create multiple test documents
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
        # Test pagination - page 1
        page1_docs, total = await document_repository.get_all(
            async_session,
            pagination=PaginationParams(page=1, page_size=2),
            sort=SortParams(sort_by="issue_date", sort_order="desc")
        )
        
        assert len(page1_docs) == 2
        assert total == 3
        
        # Test pagination - page 2
        page2_docs, total = await document_repository.get_all(
            async_session,
            pagination=PaginationParams(page=2, page_size=2),
            sort=SortParams(sort_by="issue_date", sort_order="desc")
        )
        
        assert len(page2_docs) == 1
        assert total == 3
        
        # Test sorting - ascending
        asc_docs, _ = await document_repository.get_all(
            async_session,
            pagination=PaginationParams(page=1, page_size=3),
            sort=SortParams(sort_by="issue_date", sort_order="asc")
        )
        
        # First document should have the earliest date
        assert asc_docs[0].issue_date <= asc_docs[-1].issue_date
        
        # Test sorting - descending
        desc_docs, _ = await document_repository.get_all(
            async_session,
            pagination=PaginationParams(page=1, page_size=3),
            sort=SortParams(sort_by="issue_date", sort_order="desc")
        )
        
        # First document should have the latest date
        assert desc_docs[0].issue_date >= desc_docs[-1].issue_date
    
    finally:
        # Clean up
        for document in documents:
            await async_session.delete(document)
        await async_session.commit()


@pytest.mark.asyncio
async def test_vector_search_methods(async_session, document_repository, test_documents, test_vectors):
    """Test vector search methods."""
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
        assert len(results) == 3
        assert total == 3
        
        # First result should be closest to the query vector
        assert results[0][1] <= results[1][1]  # Distance should be ascending
        
        # Test vector search with jurisdiction filter
        results, total = await document_repository.search_by_vector(
            async_session,
            query_vector=query_vector,
            jurisdiction="BE",
            pagination=PaginationParams(page=1, page_size=10)
        )
        
        # Assert results
        assert len(results) == 1
        assert total == 1
        assert results[0][0].jurisdiction == "BE"
        
        # Test hybrid search
        results, total = await document_repository.search_hybrid(
            async_session,
            query="tax",
            query_vector=query_vector,
            pagination=PaginationParams(page=1, page_size=10)
        )
        
        # Assert results
        assert len(results) == 3
        assert total == 3
        
        # Test get_similar_documents
        similar_docs = await document_repository.get_similar_documents(
            async_session,
            document_id=documents[0].id,
            limit=2
        )
        
        # Assert results
        assert len(similar_docs) == 2
        assert similar_docs[0][0].id != documents[0].id  # Should not include the source document
    
    finally:
        # Clean up
        for document in documents:
            await async_session.delete(document)
        await async_session.commit()


@pytest.mark.asyncio
async def test_filter_methods(async_session, document_repository, test_documents, test_vectors):
    """Test filter methods."""
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
        # Test filter by jurisdiction
        results, total = await document_repository.filter_by_jurisdiction(
            async_session,
            jurisdiction="BE",
            pagination=PaginationParams(page=1, page_size=10)
        )
        
        # Assert results
        assert len(results) == 1
        assert total == 1
        assert results[0].jurisdiction == "BE"
        
        # Test filter by date range
        start_date = documents[0].issue_date
        end_date = documents[-1].issue_date
        
        results, total = await document_repository.filter_by_date_range(
            async_session,
            start_date=start_date,
            end_date=end_date,
            pagination=PaginationParams(page=1, page_size=10)
        )
        
        # Assert results
        assert len(results) == 3
        assert total == 3
        
        # Test filter by date range with jurisdiction
        results, total = await document_repository.filter_by_date_range(
            async_session,
            start_date=start_date,
            end_date=end_date,
            jurisdiction="ES",
            pagination=PaginationParams(page=1, page_size=10)
        )
        
        # Assert results
        assert len(results) == 1
        assert total == 1
        assert results[0].jurisdiction == "ES"
    
    finally:
        # Clean up
        for document in documents:
            await async_session.delete(document)
        await async_session.commit()


@pytest.mark.asyncio
async def test_repository_error_handling(async_session, document_repository):
    """Test repository error handling."""
    # Test get_by_id with non-existent ID
    with pytest.raises(EntityNotFoundError) as excinfo:
        await document_repository.get_by_id(async_session, "NON_EXISTENT_ID")
    
    # Check error message
    assert "Document with ID NON_EXISTENT_ID not found" in str(excinfo.value)
    
    # Test update with non-existent ID
    with pytest.raises(EntityNotFoundError) as excinfo:
        await document_repository.update(async_session, "NON_EXISTENT_ID", {"title": "New Title"})
    
    # Check error message
    assert "Document with ID NON_EXISTENT_ID not found" in str(excinfo.value)
    
    # Test delete with non-existent ID
    with pytest.raises(EntityNotFoundError) as excinfo:
        await document_repository.delete(async_session, "NON_EXISTENT_ID")
    
    # Check error message
    assert "Document with ID NON_EXISTENT_ID not found" in str(excinfo.value)


@pytest.mark.asyncio
async def test_repository_edge_cases(async_session, document_repository, test_documents, test_vectors):
    """Test repository edge cases."""
    # Create a test document
    test_doc = test_documents[0].copy()
    test_doc["vector"] = test_vectors["tax"]
    
    document = await document_repository.create(async_session, test_doc)
    await async_session.commit()
    
    try:
        # Test pagination with page=0 (should default to page=1)
        docs, total = await document_repository.get_all(
            async_session,
            pagination=PaginationParams(page=0, page_size=10)
        )
        assert len(docs) == 1
        
        # Test pagination with negative page (should default to page=1)
        docs, total = await document_repository.get_all(
            async_session,
            pagination=PaginationParams(page=-1, page_size=10)
        )
        assert len(docs) == 1
        
        # Test pagination with page_size=0 (should default to page_size=1)
        docs, total = await document_repository.get_all(
            async_session,
            pagination=PaginationParams(page=1, page_size=0)
        )
        assert len(docs) <= 1
        
        # Test pagination with negative page_size (should default to page_size=1)
        docs, total = await document_repository.get_all(
            async_session,
            pagination=PaginationParams(page=1, page_size=-10)
        )
        assert len(docs) <= 1
        
        # Test pagination with page_size > 100 (should be capped at 100)
        docs, total = await document_repository.get_all(
            async_session,
            pagination=PaginationParams(page=1, page_size=1000)
        )
        assert len(docs) <= 100
        
        # Test sorting with invalid sort_by (should not raise error)
        docs, total = await document_repository.get_all(
            async_session,
            pagination=PaginationParams(page=1, page_size=10),
            sort=SortParams(sort_by="non_existent_column", sort_order="desc")
        )
        assert len(docs) == 1
        
        # Test sorting with invalid sort_order (should default to "desc")
        docs, total = await document_repository.get_all(
            async_session,
            pagination=PaginationParams(page=1, page_size=10),
            sort=SortParams(sort_by="issue_date", sort_order="invalid")
        )
        assert len(docs) == 1
        
        # Test vector search with empty vector
        empty_vector = [0.0] * 1536
        results, total = await document_repository.search_by_vector(
            async_session,
            query_vector=empty_vector,
            pagination=PaginationParams(page=1, page_size=10)
        )
        assert len(results) == 1
        
        # Test hybrid search with empty query
        results, total = await document_repository.search_hybrid(
            async_session,
            query="",
            query_vector=test_vectors["tax"],
            pagination=PaginationParams(page=1, page_size=10)
        )
        assert len(results) == 1
    
    finally:
        # Clean up
        await async_session.delete(document)
        await async_session.commit()


@pytest.mark.asyncio
async def test_repository_with_mock_session():
    """Test repository with mock database session."""
    # Create mock session
    mock_session = AsyncMock(spec=AsyncSession)
    
    # Create mock execute result
    mock_result = MagicMock()
    mock_result.scalars().first.return_value = None
    mock_session.execute.return_value = mock_result
    
    # Create repository
    repository = DocumentRepository()
    
    # Test get_by_id with mock session
    with pytest.raises(EntityNotFoundError):
        await repository.get_by_id(mock_session, "TEST_ID")
    
    # Verify execute was called with correct query
    mock_session.execute.assert_called_once()
    
    # Reset mock
    mock_session.reset_mock()
    
    # Create mock document for get_all
    mock_docs = [Document(id=f"TEST:{i}", title=f"Test {i}") for i in range(3)]
    mock_result = MagicMock()
    mock_result.scalars().all.return_value = mock_docs
    mock_session.execute.return_value = mock_result
    
    # Test get_all with mock session
    docs, _ = await repository.get_all(
        mock_session,
        pagination=PaginationParams(page=1, page_size=10)
    )
    
    # Verify execute was called
    assert mock_session.execute.call_count == 2  # One for query, one for count
    assert len(docs) == 3


@pytest.mark.asyncio
async def test_repository_performance(async_session, document_repository, generate_test_documents):
    """Test repository performance with larger datasets."""
    # Skip in CI environment
    if "CI" in os.environ:
        pytest.skip("Skipping performance test in CI environment")
    
    # Generate a larger set of test documents
    doc_count = 100
    test_docs = generate_test_documents(count=doc_count, with_vectors=True)
    
    # Measure time to insert documents
    start_time = time.time()
    
    # Insert documents in batches
    batch_size = 10
    for i in range(0, doc_count, batch_size):
        batch = test_docs[i:i+batch_size]
        for doc in batch:
            await document_repository.create(async_session, doc)
        await async_session.commit()
    
    insert_time = time.time() - start_time
    logger.info(f"Time to insert {doc_count} documents: {insert_time:.2f} seconds")
    
    try:
        # Measure time for pagination query
        start_time = time.time()
        
        # Query with pagination
        for page in range(1, 11):
            docs, total = await document_repository.get_all(
                async_session,
                pagination=PaginationParams(page=page, page_size=10),
                sort=SortParams(sort_by="issue_date", sort_order="desc")
            )
            assert len(docs) <= 10
        
        pagination_time = time.time() - start_time
        logger.info(f"Time for 10 paginated queries: {pagination_time:.2f} seconds")
        
        # Measure time for vector search
        start_time = time.time()
        
        # Vector search
        query_vector = np.random.rand(1536).tolist()
        results, total = await document_repository.search_by_vector(
            async_session,
            query_vector=query_vector,
            pagination=PaginationParams(page=1, page_size=10)
        )
        
        vector_search_time = time.time() - start_time
        logger.info(f"Time for vector search: {vector_search_time:.2f} seconds")
        
        # Measure time for hybrid search
        start_time = time.time()
        
        # Hybrid search
        results, total = await document_repository.search_hybrid(
            async_session,
            query="tax",
            query_vector=query_vector,
            pagination=PaginationParams(page=1, page_size=10)
        )
        
        hybrid_search_time = time.time() - start_time
        logger.info(f"Time for hybrid search: {hybrid_search_time:.2f} seconds")
        
        # Performance assertions
        assert insert_time < doc_count * 0.1  # Less than 0.1s per document
        assert pagination_time < 10.0  # Less than 10s for 10 queries
        assert vector_search_time < 5.0  # Less than 5s for vector search
        assert hybrid_search_time < 5.0  # Less than 5s for hybrid search
    
    finally:
        # Clean up - delete all test documents
        for doc in test_docs:
            try:
                document = await document_repository.get_by_id(async_session, doc["id"])
                await async_session.delete(document)
            except EntityNotFoundError:
                pass
        
        await async_session.commit()


@pytest.mark.asyncio
async def test_repository_transaction_handling(async_session, document_repository, test_documents, test_vectors):
    """Test repository transaction handling."""
    # Create a test document
    test_doc = test_documents[0].copy()
    test_doc["vector"] = test_vectors["tax"]
    
    # Start a transaction
    async with async_session.begin():
        # Create document within transaction
        document = await document_repository.create(async_session, test_doc)
        
        # Verify document exists within transaction
        retrieved_doc = await document_repository.get_by_id(async_session, document.id)
        assert retrieved_doc.id == document.id
    
    # Verify document exists after transaction commit
    retrieved_doc = await document_repository.get_by_id(async_session, document.id)
    assert retrieved_doc.id == document.id
    
    # Start another transaction for rollback
    try:
        async with async_session.begin():
            # Update document within transaction
            updated_doc = await document_repository.update(
                async_session,
                document.id,
                {"title": "Updated in transaction"}
            )
            
            # Verify update within transaction
            assert updated_doc.title == "Updated in transaction"
            
            # Raise exception to trigger rollback
            raise ValueError("Test rollback")
    except ValueError:
        pass
    
    # Verify document was not updated after rollback
    retrieved_doc = await document_repository.get_by_id(async_session, document.id)
    assert retrieved_doc.title != "Updated in transaction"
    
    # Clean up
    await document_repository.delete(async_session, document.id)
    await async_session.commit()


@pytest.mark.asyncio
async def test_repository_concurrent_operations(async_session, document_repository, test_documents, test_vectors):
    """Test repository concurrent operations."""
    # Create test documents
    docs = []
    for i in range(3):
        test_doc = test_documents[i].copy()
        test_doc["vector"] = test_vectors[list(test_vectors.keys())[i]]
        docs.append(test_doc)
    
    # Define concurrent create operation
    async def create_document(doc):
        document = await document_repository.create(async_session, doc)
        return document
    
    # Run concurrent create operations
    created_docs = await asyncio.gather(*[create_document(doc) for doc in docs])
    await async_session.commit()
    
    try:
        # Verify all documents were created
        assert len(created_docs) == 3
        
        # Define concurrent get operation
        async def get_document(doc_id):
            return await document_repository.get_by_id(async_session, doc_id)
        
        # Run concurrent get operations
        retrieved_docs = await asyncio.gather(*[get_document(doc.id) for doc in created_docs])
        
        # Verify all documents were retrieved
        assert len(retrieved_docs) == 3
        assert all(doc.id == created_docs[i].id for i, doc in enumerate(retrieved_docs))
        
        # Define concurrent update operation
        async def update_document(doc_id, index):
            return await document_repository.update(
                async_session,
                doc_id,
                {"title": f"Concurrent Update {index}"}
            )
        
        # Run concurrent update operations
        updated_docs = await asyncio.gather(*[
            update_document(doc.id, i) for i, doc in enumerate(created_docs)
        ])
        await async_session.commit()
        
        # Verify all documents were updated
        assert len(updated_docs) == 3
        assert all(doc.title.startswith("Concurrent Update") for doc in updated_docs)
    
    finally:
        # Clean up
        for doc in created_docs:
            try:
                await document_repository.delete(async_session, doc.id)
            except EntityNotFoundError:
                pass
        
        await async_session.commit()


@pytest.mark.asyncio
async def test_save_document_method(async_session, document_repository, test_documents, test_vectors):
    """Test save_document method (create or update)."""
    # Create a test document
    test_doc = test_documents[0].copy()
    test_doc["vector"] = test_vectors["tax"]
    
    # Save document (create)
    document = await document_repository.save_document(async_session, test_doc)
    await async_session.commit()
    
    try:
        # Verify document was created
        assert document.id == test_doc["id"]
        assert document.title == test_doc["title"]
        
        # Modify document data
        test_doc["title"] = "Updated Title"
        
        # Save document again (update)
        updated_document = await document_repository.save_document(async_session, test_doc)
        await async_session.commit()
        
        # Verify document was updated
        assert updated_document.id == test_doc["id"]
        assert updated_document.title == "Updated Title"
        
        # Test save_document with checksum conflict
        conflict_doc = test_documents[1].copy()
        conflict_doc["checksum"] = test_doc["checksum"]  # Same checksum
        
        # Should raise IntegrityError
        with pytest.raises(IntegrityError):
            await document_repository.save_document(async_session, conflict_doc)
            await async_session.commit()
        
        # Rollback after error
        await async_session.rollback()
    
    finally:
        # Clean up
        try:
            await document_repository.delete(async_session, document.id)
            await async_session.commit()
        except:
            await async_session.rollback()