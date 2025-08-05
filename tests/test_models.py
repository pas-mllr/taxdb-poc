"""
Tests for database models.

This module tests the database models, focusing on the Document model
and its attributes, methods, and constraints.
"""

import pytest
import pytest_asyncio
from datetime import date, datetime
from typing import List, Dict, Any

from sqlalchemy import select, func, inspect
from sqlalchemy.exc import IntegrityError
from pgvector.sqlalchemy import Vector

from src.models import Document, Base
from src.db import db_manager


@pytest.mark.unit
def test_document_model_attributes():
    """Test Document model attributes and types."""
    # Get model metadata
    mapper = inspect(Document)
    
    # Check primary key
    assert mapper.primary_key[0].name == "id"
    
    # Check column names and types
    columns = {c.name: c.type for c in mapper.columns}
    assert "id" in columns
    assert "jurisdiction" in columns
    assert "source_system" in columns
    assert "document_type" in columns
    assert "title" in columns
    assert "summary" in columns
    assert "issue_date" in columns
    assert "effective_date" in columns
    assert "language_orig" in columns
    assert "blob_url" in columns
    assert "checksum" in columns
    assert "vector" in columns
    assert "created_at" in columns
    
    # Check vector type
    assert isinstance(columns["vector"], Vector)
    
    # Check vector dimensions
    assert columns["vector"].dim == 1536


@pytest.mark.unit
def test_document_model_repr():
    """Test Document model string representation."""
    # Create a document instance
    doc = Document(
        id="TEST:20250801:DOC-001",
        jurisdiction="BE",
        source_system="test",
        document_type="law",
        title="Test Tax Law with a very long title that should be truncated",
        issue_date=date(2025, 8, 1),
        language_orig="en",
        blob_url="http://example.com/test.pdf",
        checksum="test_checksum"
    )
    
    # Check string representation
    repr_str = repr(doc)
    assert "Document" in repr_str
    assert "TEST:20250801:DOC-001" in repr_str
    assert "BE" in repr_str
    assert "Test Tax Law" in repr_str
    assert "..." in repr_str  # Title should be truncated


@pytest.mark.asyncio
@pytest.mark.unit
async def test_document_model_create(async_session, clean_test_database):
    """Test creating a Document instance in the database."""
    # Create a document
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
    
    # Add to session and commit
    async_session.add(doc)
    await async_session.commit()
    
    # Query to verify
    result = await async_session.execute(select(Document).where(Document.id == "TEST:20250801:DOC-001"))
    saved_doc = result.scalars().first()
    
    # Check attributes
    assert saved_doc is not None
    assert saved_doc.id == "TEST:20250801:DOC-001"
    assert saved_doc.jurisdiction == "BE"
    assert saved_doc.source_system == "test"
    assert saved_doc.document_type == "law"
    assert saved_doc.title == "Test Tax Law"
    assert saved_doc.summary == "A test tax law document"
    assert saved_doc.issue_date == date(2025, 8, 1)
    assert saved_doc.language_orig == "en"
    assert saved_doc.blob_url == "http://example.com/test.pdf"
    assert saved_doc.checksum == "test_checksum"
    assert saved_doc.created_at is not None
    assert isinstance(saved_doc.created_at, datetime)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_document_model_unique_id(async_session, clean_test_database):
    """Test Document model ID uniqueness constraint."""
    # Create a document
    doc1 = Document(
        id="TEST:20250801:DOC-001",
        jurisdiction="BE",
        source_system="test",
        document_type="law",
        title="Test Tax Law 1",
        issue_date=date(2025, 8, 1),
        language_orig="en",
        blob_url="http://example.com/test1.pdf",
        checksum="test1_checksum"
    )
    
    # Add to session and commit
    async_session.add(doc1)
    await async_session.commit()
    
    # Create another document with the same ID
    doc2 = Document(
        id="TEST:20250801:DOC-001",  # Same ID
        jurisdiction="ES",
        source_system="test",
        document_type="regulation",
        title="Test Tax Law 2",
        issue_date=date(2025, 8, 2),
        language_orig="es",
        blob_url="http://example.com/test2.pdf",
        checksum="test2_checksum"
    )
    
    # Add to session and expect IntegrityError on commit
    async_session.add(doc2)
    with pytest.raises(IntegrityError):
        await async_session.commit()
    
    # Rollback for cleanup
    await async_session.rollback()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_document_model_unique_checksum(async_session, clean_test_database):
    """Test Document model checksum uniqueness constraint."""
    # Create a document
    doc1 = Document(
        id="TEST:20250801:DOC-001",
        jurisdiction="BE",
        source_system="test",
        document_type="law",
        title="Test Tax Law 1",
        issue_date=date(2025, 8, 1),
        language_orig="en",
        blob_url="http://example.com/test1.pdf",
        checksum="same_checksum"
    )
    
    # Add to session and commit
    async_session.add(doc1)
    await async_session.commit()
    
    # Create another document with the same checksum
    doc2 = Document(
        id="TEST:20250801:DOC-002",
        jurisdiction="ES",
        source_system="test",
        document_type="regulation",
        title="Test Tax Law 2",
        issue_date=date(2025, 8, 2),
        language_orig="es",
        blob_url="http://example.com/test2.pdf",
        checksum="same_checksum"  # Same checksum
    )
    
    # Add to session and expect IntegrityError on commit
    async_session.add(doc2)
    with pytest.raises(IntegrityError):
        await async_session.commit()
    
    # Rollback for cleanup
    await async_session.rollback()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_document_model_required_fields(async_session, clean_test_database):
    """Test Document model required fields."""
    # Test missing jurisdiction
    doc1 = Document(
        id="TEST:20250801:DOC-001",
        # jurisdiction is missing
        source_system="test",
        document_type="law",
        title="Test Tax Law",
        issue_date=date(2025, 8, 1),
        language_orig="en",
        blob_url="http://example.com/test.pdf",
        checksum="test_checksum"
    )
    
    # Add to session and expect IntegrityError on commit
    async_session.add(doc1)
    with pytest.raises(IntegrityError):
        await async_session.commit()
    
    # Rollback for cleanup
    await async_session.rollback()
    
    # Test missing title
    doc2 = Document(
        id="TEST:20250801:DOC-001",
        jurisdiction="BE",
        source_system="test",
        document_type="law",
        # title is missing
        issue_date=date(2025, 8, 1),
        language_orig="en",
        blob_url="http://example.com/test.pdf",
        checksum="test_checksum"
    )
    
    # Add to session and expect IntegrityError on commit
    async_session.add(doc2)
    with pytest.raises(IntegrityError):
        await async_session.commit()
    
    # Rollback for cleanup
    await async_session.rollback()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_document_model_vector_operations(async_session, clean_test_database, test_vectors):
    """Test Document model vector operations."""
    # Create documents with vectors
    docs = []
    for i, (key, vector) in enumerate(test_vectors.items()):
        doc = Document(
            id=f"TEST:20250801:DOC-{i+1:03d}",
            jurisdiction="BE",
            source_system="test",
            document_type="law",
            title=f"Test {key.capitalize()} Law",
            issue_date=date(2025, 8, 1),
            language_orig="en",
            blob_url=f"http://example.com/test{i+1}.pdf",
            checksum=f"test{i+1}_checksum",
            vector=vector
        )
        docs.append(doc)
    
    # Add to session and commit
    for doc in docs:
        async_session.add(doc)
    await async_session.commit()
    
    # Test vector similarity search
    tax_vector = test_vectors["tax"]
    
    # Query using L2 distance
    stmt = select(
        Document,
        func.l2_distance(Document.vector, tax_vector).label("distance")
    ).order_by(func.l2_distance(Document.vector, tax_vector))
    
    result = await async_session.execute(stmt)
    similar_docs = result.all()
    
    # Check results
    assert len(similar_docs) == len(test_vectors)
    
    # First result should be the tax document itself (closest match)
    assert "Tax" in similar_docs[0][0].title
    
    # Check that distances are in ascending order
    distances = [doc[1] for doc in similar_docs]
    assert all(distances[i] <= distances[i+1] for i in range(len(distances)-1))


@pytest.mark.asyncio
@pytest.mark.unit
async def test_document_model_indexes(async_engine):
    """Test Document model indexes."""
    # Get table inspector
    inspector = inspect(async_engine)
    
    # Get indexes for documents table
    indexes = await inspector.get_indexes("documents")
    
    # Convert to dict for easier lookup
    index_dict = {idx["name"]: idx for idx in indexes}
    
    # Check jurisdiction_date index
    assert "ix_jurisdiction_date" in index_dict
    assert "jurisdiction" in index_dict["ix_jurisdiction_date"]["column_names"]
    assert "issue_date" in index_dict["ix_jurisdiction_date"]["column_names"]
    
    # Check vector index
    assert "ix_vector" in index_dict
    assert "vector" in index_dict["ix_vector"]["column_names"]
    assert index_dict["ix_vector"]["dialect_options"]["postgresql_using"] == "ivfflat"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_document_model_bulk_operations(async_session, clean_test_database, generate_test_documents):
    """Test Document model bulk operations."""
    # Generate test documents
    doc_data = generate_test_documents(count=50, with_vectors=True)
    
    # Create Document instances
    docs = [Document(**data) for data in doc_data]
    
    # Bulk insert
    async_session.add_all(docs)
    await async_session.commit()
    
    # Query to verify
    result = await async_session.execute(select(func.count()).select_from(Document))
    count = result.scalar()
    
    # Check count
    assert count == 50
    
    # Test bulk query by jurisdiction
    for jurisdiction in ["BE", "ES", "DE"]:
        result = await async_session.execute(
            select(func.count()).select_from(Document).where(Document.jurisdiction == jurisdiction)
        )
        jurisdiction_count = result.scalar()
        
        # Should have some documents for each jurisdiction
        assert jurisdiction_count > 0
        assert jurisdiction_count <= 50