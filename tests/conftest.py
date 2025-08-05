"""
Test fixtures for TaxDB-POC.

This module provides pytest fixtures for testing the TaxDB-POC application.
"""

import asyncio
import os
import subprocess
import threading
import time
import random
import json
import numpy as np
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncGenerator, Generator, Callable, Tuple, Union
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
import pytest_asyncio
import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
import docker

from src import settings
from src.api import create_app
from src.models import Base, Document
from src.db import db_manager, get_session, get_async_connection_string
from src.repository import DocumentRepository, PaginationParams, SortParams
from src.etl.utils import (
    get_embedding_strategy,
    EmbeddingStrategy,
    DocumentProcessor,
    CacheManager,
    DocumentFormat,
    DownloadError,
    ParsingError,
    EmbeddingError
)


# Configure pytest to handle asyncio
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def engine():
    """Create a SQLAlchemy engine for testing."""
    # Ensure we're using the local database
    os.environ["LOCAL_MODE"] = "true"
    
    # Create engine
    engine = create_engine(settings.PG_CONNSTR)
    
    # Create tables
    Base.metadata.create_all(engine)
    
    yield engine
    
    # No cleanup needed as we're using the local database


@pytest.fixture(scope="session")
def db_session(engine):
    """Create a SQLAlchemy session for testing."""
    Session = sessionmaker(bind=engine)
    session = Session()
    
    yield session
    
    session.close()


@pytest_asyncio.fixture(scope="session")
async def async_engine():
    """Create an async SQLAlchemy engine for testing."""
    # Ensure we're using the local database
    os.environ["LOCAL_MODE"] = "true"
    
    # Convert connection string to async format
    async_conn_str = get_async_connection_string(settings.PG_CONNSTR)
    
    # Create async engine
    engine = create_async_engine(
        async_conn_str,
        pool_size=5,
        max_overflow=10,
        echo=False,
    )
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Close engine
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def async_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create an async SQLAlchemy session for testing."""
    # Create session
    async_session_maker = sessionmaker(
        bind=async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    async with async_session_maker() as session:
        yield session


@pytest.fixture(scope="session")
def document_repository() -> DocumentRepository:
    """Create a document repository for testing."""
    return DocumentRepository()


@pytest_asyncio.fixture(scope="session")
async def embedding_strategy() -> EmbeddingStrategy:
    """Create an embedding strategy for testing."""
    return get_embedding_strategy()


@pytest.fixture(scope="session")
def test_vectors() -> Dict[str, List[float]]:
    """Create test vectors for testing."""
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Create test vectors
    vectors = {
        "tax": np.random.rand(1536).tolist(),
        "finance": np.random.rand(1536).tolist(),
        "law": np.random.rand(1536).tolist(),
        "regulation": np.random.rand(1536).tolist(),
        "government": np.random.rand(1536).tolist(),
    }
    
    return vectors


@pytest.fixture(scope="session")
def test_documents() -> List[Dict[str, Any]]:
    """Create test documents for testing."""
    # Create test documents
    documents = [
        {
            "id": "TEST:20250801:DOC-001",
            "jurisdiction": "BE",
            "source_system": "test",
            "document_type": "law",
            "title": "Test Tax Law 1",
            "summary": "A test tax law document for Belgium",
            "issue_date": "2025-08-01",
            "language_orig": "en",
            "blob_url": "http://example.com/test1.pdf",
            "checksum": "test1_checksum",
        },
        {
            "id": "TEST:20250802:DOC-002",
            "jurisdiction": "ES",
            "source_system": "test",
            "document_type": "regulation",
            "title": "Test Finance Regulation",
            "summary": "A test finance regulation document for Spain",
            "issue_date": "2025-08-02",
            "language_orig": "en",
            "blob_url": "http://example.com/test2.pdf",
            "checksum": "test2_checksum",
        },
        {
            "id": "TEST:20250803:DOC-003",
            "jurisdiction": "DE",
            "source_system": "test",
            "document_type": "decree",
            "title": "Test Government Decree",
            "summary": "A test government decree document for Germany",
            "issue_date": "2025-08-03",
            "language_orig": "en",
            "blob_url": "http://example.com/test3.pdf",
            "checksum": "test3_checksum",
        },
    ]
    
    return documents


@pytest.fixture(scope="session")
def api_server():
    """Start the API server in a background thread."""
    # Create FastAPI app
    app = create_app()
    
    # Start server in a thread
    def run_server():
        import uvicorn
        uvicorn.run(app, host="127.0.0.1", port=9000, log_level="error")
    
    # Start server
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    
    # Wait for server to start
    time.sleep(2)
    
    yield "http://localhost:9000"
    
    # No cleanup needed as thread is daemon


@pytest.fixture(scope="session")
def test_client() -> Generator[TestClient, None, None]:
    """Create a FastAPI test client."""
    app = create_app()
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="session")
def run_etl():
    """Run ETL processes."""
    # Set environment variables
    os.environ["LOCAL_MODE"] = "true"
    os.environ["DOC_LOOKBACK_HOURS"] = "48"
    
    # Run ETL processes
    subprocess.run(["make", "etl-run-once"], check=False)
    
    yield

# Mock fixtures for external services

@pytest.fixture
def mock_http_client():
    """Create a mock HTTP client for testing."""
    with patch("httpx.AsyncClient") as mock_client:
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={"data": "test"})
        mock_response.content = b"test content"
        
        # Setup mock client
        mock_client_instance = AsyncMock()
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.__aexit__.return_value = None
        mock_client_instance.get = AsyncMock(return_value=mock_response)
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        
        # Setup mock constructor
        mock_client.return_value = mock_client_instance
        
        yield mock_client


@pytest.fixture
def mock_embedding_strategy():
    """Create a mock embedding strategy for testing."""
    mock_strategy = MagicMock(spec=EmbeddingStrategy)
    mock_strategy.embed = AsyncMock(return_value=[0.1] * 1536)
    mock_strategy.get_dimensions = MagicMock(return_value=1536)
    mock_strategy.preprocess_text = MagicMock(return_value="preprocessed text")
    
    with patch("src.etl.utils.get_embedding_strategy", return_value=mock_strategy):
        yield mock_strategy


@pytest.fixture
def mock_document_processor():
    """Create a mock document processor for testing."""
    mock_processor = MagicMock(spec=DocumentProcessor)
    mock_processor.process_document = AsyncMock()
    mock_processor.process_batch = AsyncMock()
    
    with patch("src.etl.utils.DocumentProcessor", return_value=mock_processor):
        yield mock_processor


@pytest.fixture
def mock_cache_manager():
    """Create a mock cache manager for testing."""
    mock_cache = MagicMock(spec=CacheManager)
    mock_cache.get_download_path = MagicMock(return_value=Path("/tmp/test.bin"))
    mock_cache.get_processed_path = MagicMock(return_value=Path("/tmp/test.json"))
    mock_cache.is_cached = MagicMock(return_value=False)
    mock_cache.save_metadata = MagicMock()
    mock_cache.load_metadata = MagicMock(return_value=None)
    
    with patch("src.etl.utils.CacheManager", return_value=mock_cache):
        yield mock_cache


# ETL testing fixtures

@pytest.fixture
def etl_test_data():
    """Create test data for ETL testing."""
    return {
        "BE": {
            "xml_content": b"""
            <mb:doc xmlns:mb="http://www.ejustice.just.fgov.be/moniteur">
                <mb:eli>2025/12345</mb:eli>
                <mb:title>Test Tax Law</mb:title>
                <mb:pubdate>2025-08-01</mb:pubdate>
                <mb:text>This is a test tax law document for Belgium</mb:text>
                <mb:summary>Test summary</mb:summary>
            </mb:doc>
            """,
            "expected_doc": {
                "id": "BE:2025/12345",
                "jurisdiction": "BE",
                "source_system": "moniteur",
                "document_type": "legal",
                "title": "Test Tax Law",
                "summary": "Test summary",
                "issue_date": date(2025, 8, 1),
                "language_orig": "nl",
                "text": "This is a test tax law document for Belgium"
            }
        },
        "ES": {
            "xml_content": b"""
            <documento xmlns:boe="https://www.boe.es/xsd/boe">
                <metadatos>
                    <identificador>BOE-A-2025-12345</identificador>
                    <titulo>Test Tax Regulation</titulo>
                    <fecha_publicacion>20250801</fecha_publicacion>
                    <departamento>Ministerio de Hacienda</departamento>
                    <materia>Fiscal</materia>
                    <resumen>Test summary</resumen>
                </metadatos>
                <texto>This is a test tax regulation document for Spain</texto>
            </documento>
            """,
            "expected_doc": {
                "id": "ES:BOE-A-2025-12345",
                "jurisdiction": "ES",
                "source_system": "boe",
                "document_type": "legal",
                "title": "Test Tax Regulation",
                "summary": "Test summary",
                "issue_date": date(2025, 8, 1),
                "language_orig": "es",
                "text": "This is a test tax regulation document for Spain"
            }
        },
        "DE": {
            "json_content": json.dumps({
                "items": [
                    {
                        "id": "BGBL-2025-12345",
                        "title": "Test Tax Decree",
                        "category": "Steuer",
                        "pdfUrl": "https://www.bgbl.de/test.pdf",
                        "publicationDate": "2025-08-01"
                    }
                ]
            }).encode(),
            "pdf_content": b"%PDF-1.5\nTest content for German tax decree",
            "expected_doc": {
                "id": "DE:BGBL-2025-12345",
                "jurisdiction": "DE",
                "source_system": "bgbl",
                "document_type": "legal",
                "title": "Test Tax Decree",
                "issue_date": date(2025, 8, 1),
                "language_orig": "de",
                "category": "Steuer"
            }
        }
    }


@pytest.fixture
def mock_download_file():
    """Mock the download_file function for ETL testing."""
    async def mock_download(url, **kwargs):
        if "moniteur" in url or "BE" in kwargs.get("jurisdiction", ""):
            return etl_test_data()["BE"]["xml_content"]
        elif "boe" in url or "ES" in kwargs.get("jurisdiction", ""):
            return etl_test_data()["ES"]["xml_content"]
        elif "bgbl" in url or "DE" in kwargs.get("jurisdiction", ""):
            if url.endswith(".pdf"):
                return etl_test_data()["DE"]["pdf_content"]
            else:
                return etl_test_data()["DE"]["json_content"]
        else:
            return b"default test content"
    
    with patch("src.etl.utils.download_file", side_effect=mock_download):
        yield mock_download


# Docker-based testing fixtures

@pytest.fixture(scope="session")
def docker_client():
    """Create a Docker client for testing."""
    try:
        client = docker.from_env()
        yield client
    finally:
        client.close()


@pytest.fixture(scope="session")
def postgres_container(docker_client):
    """Start a PostgreSQL container with pgvector for testing."""
    # Skip if not running in CI environment
    if not os.environ.get("CI"):
        yield None
        return
    
    try:
        # Pull the image
        docker_client.images.pull("ankane/pgvector:latest")
        
        # Create and start the container
        container = docker_client.containers.run(
            "ankane/pgvector:latest",
            name="taxdb-test-postgres",
            environment={
                "POSTGRES_USER": "postgres",
                "POSTGRES_PASSWORD": "postgres",
                "POSTGRES_DB": "taxdb_test"
            },
            ports={"5432/tcp": 5432},
            detach=True,
            remove=True
        )
        
        # Wait for PostgreSQL to be ready
        time.sleep(5)
        
        # Set environment variables for tests
        os.environ["PG_CONNSTR"] = "postgresql://postgres:postgres@localhost:5432/taxdb_test"
        
        yield container
        
    finally:
        # Stop and remove the container
        try:
            container = docker_client.containers.get("taxdb-test-postgres")
            container.stop()
        except:
            pass


@pytest.fixture
def clean_test_database(db_session):
    """Clean the test database before and after tests."""
    # Clean before test
    db_session.execute(text("TRUNCATE TABLE documents CASCADE"))
    db_session.commit()
    
    yield
    
    # Clean after test
    db_session.execute(text("TRUNCATE TABLE documents CASCADE"))
    db_session.commit()


# Test data generation fixtures

@pytest.fixture
def generate_test_documents():
    """Generate test documents with specific characteristics."""
    def _generate(count=10, jurisdiction=None, with_vectors=True):
        documents = []
        for i in range(count):
            doc = {
                "id": f"TEST:{datetime.now().strftime('%Y%m%d')}:DOC-{i+1:03d}",
                "jurisdiction": jurisdiction or random.choice(["BE", "ES", "DE"]),
                "source_system": "test",
                "document_type": random.choice(["law", "regulation", "decree", "directive"]),
                "title": f"Test Document {i+1}",
                "summary": f"This is a test document {i+1}",
                "issue_date": date.today() - timedelta(days=random.randint(0, 365)),
                "language_orig": random.choice(["en", "nl", "fr", "es", "de"]),
                "blob_url": f"http://example.com/test{i+1}.pdf",
                "checksum": f"test{i+1}_checksum"
            }
            
            if with_vectors:
                doc["vector"] = np.random.rand(1536).tolist()
                
            documents.append(doc)
        
        return documents
    
    return _generate


@pytest.fixture
async def populate_test_database(db_session, generate_test_documents, document_repository):
    """Populate the test database with test documents."""
    async def _populate(count=10, jurisdiction=None, with_vectors=True):
        documents = generate_test_documents(count, jurisdiction, with_vectors)
        saved_docs = []
        
        for doc in documents:
            saved_doc = await document_repository.save_document(db_session, doc)
            saved_docs.append(saved_doc)
        
        await db_session.commit()
        return saved_docs
    
    return _populate