"""
Tests for ETL utility functions.

This module tests the utility functions in src/etl/utils.py, including
cache management, embedding strategies, document parsing, and pipeline execution.
"""

import os
import json
import asyncio
import tempfile
import pytest
import pytest_asyncio
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import patch, MagicMock, AsyncMock

from src.etl.utils import (
    CacheManager,
    EmbeddingStrategy,
    AzureOpenAIEmbedding,
    LocalMiniLMEmbedding,
    DocumentFormat,
    XMLParser,
    PDFParser,
    HTMLParser,
    get_parser,
    download_file,
    DocumentProcessor,
    PipelineContext,
    run_pipeline,
    calculate_checksum,
    DownloadError,
    ParsingError,
    EmbeddingError,
    ProcessingError
)


def test_calculate_checksum():
    """Test checksum calculation."""
    # Test with string input
    checksum1 = calculate_checksum("test string".encode())
    assert isinstance(checksum1, str)
    assert len(checksum1) > 0
    
    # Test with binary input
    checksum2 = calculate_checksum(b"test binary")
    assert isinstance(checksum2, str)
    assert len(checksum2) > 0
    
    # Test determinism (same input should produce same output)
    checksum3 = calculate_checksum("test string".encode())
    assert checksum1 == checksum3
    
    # Test different inputs produce different outputs
    checksum4 = calculate_checksum("different string".encode())
    assert checksum1 != checksum4


class TestCacheManager:
    """Tests for CacheManager class."""
    
    def test_init(self):
        """Test CacheManager initialization."""
        # Test with default cache directory
        cache_manager = CacheManager()
        assert cache_manager.cache_dir.exists()
        assert (cache_manager.cache_dir / "downloads").exists()
        assert (cache_manager.cache_dir / "processed").exists()
        assert (cache_manager.cache_dir / "metadata").exists()
        
        # Test with custom cache directory
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_dir = Path(temp_dir) / "custom_cache"
            cache_manager = CacheManager(cache_dir=custom_dir)
            assert cache_manager.cache_dir == custom_dir
            assert cache_manager.cache_dir.exists()
            assert (cache_manager.cache_dir / "downloads").exists()
            assert (cache_manager.cache_dir / "processed").exists()
            assert (cache_manager.cache_dir / "metadata").exists()
    
    def test_get_download_path(self):
        """Test get_download_path method."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=Path(temp_dir))
            
            # Test for different jurisdictions
            be_path = cache_manager.get_download_path("http://example.com/be", "BE")
            es_path = cache_manager.get_download_path("http://example.com/es", "ES")
            de_path = cache_manager.get_download_path("http://example.com/de", "DE")
            
            # Check paths are in correct subdirectories
            assert "be" in str(be_path)
            assert "es" in str(es_path)
            assert "de" in str(de_path)
            
            # Check paths are different for different URLs
            assert be_path != es_path
            assert be_path != de_path
            assert es_path != de_path
    
    def test_get_processed_path(self):
        """Test get_processed_path method."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=Path(temp_dir))
            
            # Test for different jurisdictions
            be_path = cache_manager.get_processed_path("DOC-001", "BE")
            es_path = cache_manager.get_processed_path("DOC-001", "ES")
            de_path = cache_manager.get_processed_path("DOC-001", "DE")
            
            # Check paths are in correct subdirectories
            assert "be" in str(be_path)
            assert "es" in str(es_path)
            assert "de" in str(de_path)
            
            # Check paths have correct extension
            assert be_path.suffix == ".json"
            assert es_path.suffix == ".json"
            assert de_path.suffix == ".json"
    
    def test_get_metadata_path(self):
        """Test get_metadata_path method."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=Path(temp_dir))
            
            # Test for different keys
            path1 = cache_manager.get_metadata_path("key1")
            path2 = cache_manager.get_metadata_path("key2")
            
            # Check paths are in correct directory
            assert path1.parent == cache_manager.cache_dir / "metadata"
            assert path2.parent == cache_manager.cache_dir / "metadata"
            
            # Check paths have correct extension
            assert path1.suffix == ".json"
            assert path2.suffix == ".json"
    
    def test_is_cached(self):
        """Test is_cached method."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=Path(temp_dir))
            
            # Create a test file
            test_file = Path(temp_dir) / "test_file.txt"
            test_file.touch()
            
            # Test with existing file
            assert cache_manager.is_cached(test_file) is True
            
            # Test with non-existent file
            non_existent = Path(temp_dir) / "non_existent.txt"
            assert cache_manager.is_cached(non_existent) is False
    
    def test_save_load_metadata(self):
        """Test save_metadata and load_metadata methods."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=Path(temp_dir))
            
            # Test data
            metadata = {"key": "value", "nested": {"key": "value"}}
            
            # Save metadata
            cache_manager.save_metadata("test_key", metadata)
            
            # Check file exists
            metadata_path = cache_manager.get_metadata_path("test_key")
            assert metadata_path.exists()
            
            # Load metadata
            loaded_metadata = cache_manager.load_metadata("test_key")
            assert loaded_metadata == metadata
            
            # Test loading non-existent metadata
            assert cache_manager.load_metadata("non_existent") is None
    
    def test_save_load_processed_document(self):
        """Test save_processed_document and load_processed_document methods."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=Path(temp_dir))
            
            # Test document
            doc = {
                "id": "TEST:20250801:DOC-001",
                "jurisdiction": "BE",
                "title": "Test Document",
                "vector": [0.1, 0.2, 0.3]  # This should be removed when caching
            }
            
            # Save document
            cache_manager.save_processed_document(doc, "BE")
            
            # Check file exists
            doc_path = cache_manager.get_processed_path(doc["id"], "BE")
            assert doc_path.exists()
            
            # Load document
            loaded_doc = cache_manager.load_processed_document(doc["id"], "BE")
            assert loaded_doc is not None
            assert loaded_doc["id"] == doc["id"]
            assert loaded_doc["title"] == doc["title"]
            assert "vector" not in loaded_doc  # Vector should be removed when caching
            
            # Test loading non-existent document
            assert cache_manager.load_processed_document("non_existent", "BE") is None
            
            # Test missing ID
            with pytest.raises(ValueError):
                cache_manager.save_processed_document({"title": "No ID"}, "BE")
    
    def test_clear_expired_cache(self):
        """Test clear_expired_cache method."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=Path(temp_dir))
            
            # Create some test files with different modification times
            # Recent file
            recent_file = cache_manager.cache_dir / "downloads" / "recent.txt"
            recent_file.parent.mkdir(exist_ok=True)
            recent_file.touch()
            
            # Old file (modify access/modification time to be old)
            old_file = cache_manager.cache_dir / "downloads" / "old.txt"
            old_file.touch()
            old_time = (datetime.now() - timedelta(days=40)).timestamp()
            os.utime(old_file, (old_time, old_time))
            
            # Clear expired cache (older than 30 days)
            deleted_count = cache_manager.clear_expired_cache(max_age_days=30)
            
            # Check results
            assert deleted_count >= 1  # At least the old file should be deleted
            assert recent_file.exists()  # Recent file should still exist
            assert not old_file.exists()  # Old file should be deleted


class TestEmbeddingStrategy:
    """Tests for EmbeddingStrategy classes."""
    
    def test_abstract_class(self):
        """Test EmbeddingStrategy is an abstract class."""
        # Attempting to instantiate the abstract class should raise TypeError
        with pytest.raises(TypeError):
            EmbeddingStrategy()
    
    def test_preprocess_text(self):
        """Test preprocess_text method."""
        # Create a concrete subclass for testing
        class TestStrategy(EmbeddingStrategy):
            async def embed(self, text):
                return [0.1, 0.2, 0.3]
            
            def get_dimensions(self):
                return 3
        
        strategy = TestStrategy()
        
        # Test with normal text
        text = "This is a test."
        processed = strategy.preprocess_text(text)
        assert processed == text
        
        # Test with excessive whitespace
        text = "  This   has \n\n excessive \t whitespace.  "
        processed = strategy.preprocess_text(text)
        assert processed == "This has excessive whitespace."
        
        # Test with max_length
        text = "This is a very long text that should be truncated."
        processed = strategy.preprocess_text(text, max_length=10)
        assert processed == "This is a "
        assert len(processed) == 10
    
    @pytest.mark.asyncio
    async def test_azure_openai_embedding(self):
        """Test AzureOpenAIEmbedding class."""
        # Mock the httpx.AsyncClient
        with patch("httpx.AsyncClient") as mock_client:
            # Setup mock response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_response.json = MagicMock(return_value={
                "data": [{"embedding": [0.1, 0.2, 0.3]}]
            })
            
            # Setup mock client
            mock_client_instance = AsyncMock()
            mock_client_instance.__aenter__.return_value = mock_client_instance
            mock_client_instance.__aexit__.return_value = None
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            
            # Setup mock constructor
            mock_client.return_value = mock_client_instance
            
            # Create embedding strategy with test credentials
            strategy = AzureOpenAIEmbedding(
                endpoint="https://test-endpoint",
                api_key="test-key",
                model="test-model"
            )
            
            # Test get_dimensions
            assert strategy.get_dimensions() == 1536
            
            # Test embed method
            embedding = await strategy.embed("Test text")
            assert embedding == [0.1, 0.2, 0.3]
            
            # Verify API call
            mock_client_instance.post.assert_called_once()
            args, kwargs = mock_client_instance.post.call_args
            assert args[0] == "https://test-endpoint/openai/deployments/test-model/embeddings?api-version=2023-05-15"
            assert kwargs["headers"]["api-key"] == "test-key"
            assert kwargs["json"]["input"] == "Test text"
    
    @pytest.mark.asyncio
    async def test_local_minilm_embedding(self):
        """Test LocalMiniLMEmbedding class."""
        # Mock the SentenceTransformer
        with patch("src.etl.utils.LocalMiniLMEmbedding._load_model") as mock_load_model:
            # Create a mock model
            mock_model = MagicMock()
            mock_model.encode = MagicMock(return_value=[0.1, 0.2, 0.3])
            
            # Setup the mock
            strategy = LocalMiniLMEmbedding()
            strategy._model = mock_model
            
            # Test get_dimensions
            assert strategy.get_dimensions() == 384
            
            # Test embed method
            embedding = await strategy.embed("Test text")
            assert embedding == [0.1, 0.2, 0.3]
            
            # Verify model call
            mock_model.encode.assert_called_once_with("Test text")


class TestDocumentParsers:
    """Tests for document parser classes."""
    
    @pytest.mark.asyncio
    async def test_xml_parser(self):
        """Test XMLParser class."""
        parser = XMLParser()
        
        # Test with valid XML
        xml_content = b"""
        <root>
            <text>This is the main text.</text>
            <summary>This is the summary.</summary>
        </root>
        """
        
        text, summary = await parser.parse(
            xml_content,
            text_xpath="//text",
            summary_xpath="//summary"
        )
        
        assert text == "This is the main text."
        assert summary == "This is the summary."
        
        # Test with missing summary
        xml_content = b"""
        <root>
            <text>This is the main text.</text>
        </root>
        """
        
        text, summary = await parser.parse(
            xml_content,
            text_xpath="//text",
            summary_xpath="//summary"
        )
        
        assert text == "This is the main text."
        assert summary is None
        
        # Test with invalid XML
        with pytest.raises(ParsingError):
            await parser.parse(
                b"<invalid XML",
                text_xpath="//text"
            )
    
    @pytest.mark.asyncio
    async def test_pdf_parser(self):
        """Test PDFParser class."""
        # Mock extract_text function
        with patch("src.etl.utils.extract_text") as mock_extract_text:
            mock_extract_text.return_value = "This is the extracted text."
            
            parser = PDFParser()
            
            # Test parsing
            text, summary = await parser.parse(
                b"%PDF-1.5\nTest PDF content",
                max_pages=None
            )
            
            assert text == "This is the extracted text."
            assert summary is None
            
            # Test with summary_pages
            text, summary = await parser.parse(
                b"%PDF-1.5\nTest PDF content",
                max_pages=None,
                summary_pages=1
            )
            
            assert text == "This is the extracted text."
            assert summary == "This is the extracted text."
            
            # Verify extract_text was called
            mock_extract_text.assert_called()
    
    @pytest.mark.asyncio
    async def test_html_parser(self):
        """Test HTMLParser class."""
        # Mock BeautifulSoup
        with patch("src.etl.utils.BeautifulSoup") as mock_bs:
            # Setup mock
            mock_soup = MagicMock()
            mock_soup.get_text.return_value = "This is the extracted text."
            mock_bs.return_value = mock_soup
            
            parser = HTMLParser()
            
            # Test parsing
            text, summary = await parser.parse(
                b"<html><body>Test HTML content</body></html>"
            )
            
            assert text == "This is the extracted text."
            assert summary is None
            
            # Verify BeautifulSoup was called
            mock_bs.assert_called_once()
    
    def test_get_parser(self):
        """Test get_parser function."""
        # Test for XML format
        parser = get_parser(DocumentFormat.XML)
        assert isinstance(parser, XMLParser)
        
        # Test for PDF format
        parser = get_parser(DocumentFormat.PDF)
        assert isinstance(parser, PDFParser)
        
        # Test for HTML format
        parser = get_parser(DocumentFormat.HTML)
        assert isinstance(parser, HTMLParser)
        
        # Test for unsupported format
        with pytest.raises(ValueError):
            get_parser("unsupported_format")


@pytest.mark.asyncio
async def test_download_file(mock_http_client, mock_cache_manager):
    """Test download_file function."""
    # Test downloading uncached file
    mock_cache_manager.is_cached.return_value = False
    
    content = await download_file(
        "http://example.com/test",
        cache_manager=mock_cache_manager,
        jurisdiction="BE"
    )
    
    assert content == b"test content"
    
    # Verify cache check and save
    mock_cache_manager.is_cached.assert_called_once()
    mock_cache_manager.get_download_path.assert_called_once()
    
    # Test downloading cached file
    mock_cache_manager.reset_mock()
    mock_cache_manager.is_cached.return_value = True
    
    # Mock reading from cache
    with patch("builtins.open", MagicMock()) as mock_open:
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = b"cached content"
        mock_open.return_value = mock_file
        
        content = await download_file(
            "http://example.com/test",
            cache_manager=mock_cache_manager,
            jurisdiction="BE"
        )
        
        assert content == b"cached content"
        
        # Verify cache check
        mock_cache_manager.is_cached.assert_called_once()
        mock_cache_manager.get_download_path.assert_called_once()
        
        # Verify HTTP client was not called
        mock_http_client.assert_not_called()


class TestDocumentProcessor:
    """Tests for DocumentProcessor class."""
    
    @pytest.mark.asyncio
    async def test_init(self, mock_embedding_strategy):
        """Test DocumentProcessor initialization."""
        # Test with default parameters
        processor = DocumentProcessor()
        assert processor.embedding_strategy is not None
        
        # Test with custom parameters
        processor = DocumentProcessor(
            embedding_strategy=mock_embedding_strategy,
            cache_manager=MagicMock(spec=CacheManager)
        )
        assert processor.embedding_strategy is mock_embedding_strategy
    
    @pytest.mark.asyncio
    async def test_process_document(self, mock_embedding_strategy):
        """Test process_document method."""
        # Create processor
        processor = DocumentProcessor(embedding_strategy=mock_embedding_strategy)
        
        # Create test document
        doc = {
            "id": "TEST:20250801:DOC-001",
            "jurisdiction": "BE",
            "title": "Test Document",
            "text": "This is the document text."
        }
        
        # Create mock download function
        async def mock_download(doc):
            return b"test content", DocumentFormat.TEXT, {}
        
        # Process document
        processed_doc = await processor.process_document(doc, mock_download)
        
        # Check results
        assert processed_doc["id"] == doc["id"]
        assert processed_doc["title"] == doc["title"]
        assert "vector" in processed_doc
        assert processed_doc["vector"] == await mock_embedding_strategy.embed("This is the document text.")
        
        # Test with download error
        async def mock_download_error(doc):
            raise DownloadError("Test error")
        
        # Should return None on error
        processed_doc = await processor.process_document(doc, mock_download_error)
        assert processed_doc is None
        
        # Test with parsing error
        async def mock_download_parse_error(doc):
            return b"invalid content", DocumentFormat.XML, {}
        
        # Mock parser to raise error
        with patch("src.etl.utils.get_parser") as mock_get_parser:
            mock_parser = MagicMock()
            mock_parser.parse = AsyncMock(side_effect=ParsingError("Test error"))
            mock_get_parser.return_value = mock_parser
            
            # Should return None on error
            processed_doc = await processor.process_document(doc, mock_download_parse_error)
            assert processed_doc is None
        
        # Test with embedding error
        mock_embedding_strategy.embed = AsyncMock(side_effect=EmbeddingError("Test error"))
        
        # Should return None on error
        processed_doc = await processor.process_document(doc, mock_download)
        assert processed_doc is None
    
    @pytest.mark.asyncio
    async def test_process_batch(self, mock_embedding_strategy):
        """Test process_batch method."""
        # Create processor
        processor = DocumentProcessor(embedding_strategy=mock_embedding_strategy)
        
        # Create test documents
        docs = [
            {
                "id": f"TEST:20250801:DOC-{i:03d}",
                "jurisdiction": "BE",
                "title": f"Test Document {i}",
                "text": f"This is document {i} text."
            }
            for i in range(5)
        ]
        
        # Create mock download function
        async def mock_download(doc):
            return b"test content", DocumentFormat.TEXT, {}
        
        # Process batch
        processed_docs = await processor.process_batch(docs, mock_download, concurrency=2)
        
        # Check results
        assert len(processed_docs) == 5
        for i, doc in enumerate(processed_docs):
            assert doc["id"] == f"TEST:20250801:DOC-{i:03d}"
            assert "vector" in doc
        
        # Test with some errors
        async def mock_download_with_errors(doc):
            if "DOC-001" in doc["id"] or "DOC-003" in doc["id"]:
                raise DownloadError("Test error")
            return b"test content", DocumentFormat.TEXT, {}
        
        # Process batch with errors
        processed_docs = await processor.process_batch(docs, mock_download_with_errors, concurrency=2)
        
        # Check results (should have 3 successful documents)
        assert len(processed_docs) == 3
        assert all("DOC-001" not in doc["id"] for doc in processed_docs)
        assert all("DOC-003" not in doc["id"] for doc in processed_docs)


@pytest.mark.asyncio
async def test_pipeline_context():
    """Test PipelineContext class."""
    # Create context
    context = PipelineContext(
        jurisdiction="BE",
        start_date=date(2025, 8, 1),
        end_date=date(2025, 8, 3),
        metadata={"key": "value"}
    )
    
    # Check attributes
    assert context.jurisdiction == "BE"
    assert context.start_date == date(2025, 8, 1)
    assert context.end_date == date(2025, 8, 3)
    assert context.metadata == {"key": "value"}
    assert context.stats["documents_found"] == 0
    assert context.stats["documents_processed"] == 0
    assert context.stats["documents_saved"] == 0
    assert context.stats["errors"] == 0
    
    # Test get_elapsed_time
    elapsed = context.get_elapsed_time()
    assert elapsed >= 0
    
    # Test get_stats
    stats = context.get_stats()
    assert "jurisdiction" in stats
    assert "start_date" in stats
    assert "end_date" in stats
    assert "elapsed_time" in stats
    assert "documents_found" in stats
    assert "documents_processed" in stats
    assert "documents_saved" in stats
    assert "errors" in stats
    assert "metadata" in stats


@pytest.mark.asyncio
async def test_run_pipeline(mock_embedding_strategy, mock_document_processor, async_session):
    """Test run_pipeline function."""
    # Create mock fetch function
    async def mock_fetch(start_date, end_date):
        return [
            {
                "id": f"TEST:20250801:DOC-{i:03d}",
                "jurisdiction": "BE",
                "title": f"Test Document {i}",
                "text": f"This is document {i} text."
            }
            for i in range(3)
        ]
    
    # Mock document processor
    mock_document_processor.process_batch.return_value = [
        {
            "id": f"TEST:20250801:DOC-{i:03d}",
            "jurisdiction": "BE",
            "title": f"Test Document {i}",
            "text": f"This is document {i} text.",
            "vector": [0.1, 0.2, 0.3]
        }
        for i in range(3)
    ]
    
    # Run pipeline
    stats = await run_pipeline(
        "BE",
        mock_fetch,
        start_date=date(2025, 8, 1),
        end_date=date(2025, 8, 3),
        metadata={"key": "value"}
    )
    
    # Check stats
    assert stats["jurisdiction"] == "BE"
    assert stats["start_date"] == date(2025, 8, 1)
    assert stats["end_date"] == date(2025, 8, 3)
    assert stats["documents_found"] == 3
    assert "elapsed_time" in stats
    
    # Test with fetch error
    async def mock_fetch_error(start_date, end_date):
        raise Exception("Test error")
    
    # Should handle error and return stats
    stats = await run_pipeline(
        "BE",
        mock_fetch_error,
        start_date=date(2025, 8, 1),
        end_date=date(2025, 8, 3)
    )
    
    assert stats["jurisdiction"] == "BE"
    assert stats["errors"] == 1