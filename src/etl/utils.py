"""
Utility functions for ETL processes.

This module provides utility functions for ETL processes, including
a generic pipeline runner, document parsers, cache management, and text embedding strategies.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime, date, timedelta
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Set, Tuple, Type, TypeVar, Union, cast, Awaitable

import httpx
from pdfminer.high_level import extract_text
from pgvector.sqlalchemy import Vector
from rich.logging import RichHandler
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker
from tenacity import (
    RetryError, 
    retry, 
    retry_if_exception_type, 
    stop_after_attempt, 
    wait_exponential,
    before_sleep_log
)
from xml.etree import ElementTree as ET

from src import settings
from src.models import Document, Base

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("taxdb")

# Type variables
T = TypeVar('T')
DocumentType = Dict[str, Any]


class ETLError(Exception):
    """Base exception for ETL-related errors."""
    pass


class DownloadError(ETLError):
    """Exception raised when a download fails."""
    pass


class ParsingError(ETLError):
    """Exception raised when document parsing fails."""
    pass


class EmbeddingError(ETLError):
    """Exception raised when embedding generation fails."""
    pass


class ProcessingError(ETLError):
    """Exception raised when document processing fails."""
    pass


class DocumentFormat(Enum):
    """Enum for supported document formats."""
    XML = "xml"
    PDF = "pdf"
    JSON = "json"
    TEXT = "txt"
    HTML = "html"


class CacheManager:
    """Manager for caching downloaded files and processing results."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the cache manager.
        
        Args:
            cache_dir: Directory to store cached files. Defaults to settings.CACHE_DIR.
        """
        self.cache_dir = cache_dir or settings.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._init_subdirs()
        
    def _init_subdirs(self) -> None:
        """Initialize cache subdirectories."""
        (self.cache_dir / "downloads").mkdir(exist_ok=True)
        (self.cache_dir / "processed").mkdir(exist_ok=True)
        (self.cache_dir / "metadata").mkdir(exist_ok=True)
        
        # Create jurisdiction-specific directories
        for jurisdiction in settings.JURISDICTIONS:
            (self.cache_dir / "downloads" / jurisdiction.lower()).mkdir(exist_ok=True)
            (self.cache_dir / "processed" / jurisdiction.lower()).mkdir(exist_ok=True)
    
    def get_download_path(self, url: str, jurisdiction: str) -> Path:
        """Get the cache path for a downloaded file.
        
        Args:
            url: URL of the file
            jurisdiction: Jurisdiction code (BE, ES, DE)
            
        Returns:
            Path to the cached file
        """
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.cache_dir / "downloads" / jurisdiction.lower() / f"{url_hash}.bin"
    
    def get_processed_path(self, doc_id: str, jurisdiction: str) -> Path:
        """Get the cache path for a processed document.
        
        Args:
            doc_id: Document ID
            jurisdiction: Jurisdiction code (BE, ES, DE)
            
        Returns:
            Path to the cached processed document
        """
        return self.cache_dir / "processed" / jurisdiction.lower() / f"{doc_id}.json"
    
    def get_metadata_path(self, key: str) -> Path:
        """Get the cache path for metadata.
        
        Args:
            key: Metadata key
            
        Returns:
            Path to the cached metadata
        """
        return self.cache_dir / "metadata" / f"{key}.json"
    
    def is_cached(self, path: Path) -> bool:
        """Check if a file is cached.
        
        Args:
            path: Path to check
            
        Returns:
            True if the file is cached, False otherwise
        """
        return path.exists()
    
    def save_metadata(self, key: str, data: Dict[str, Any]) -> None:
        """Save metadata to cache.
        
        Args:
            key: Metadata key
            data: Data to save
        """
        path = self.get_metadata_path(key)
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Load metadata from cache.
        
        Args:
            key: Metadata key
            
        Returns:
            Cached metadata or None if not found
        """
        path = self.get_metadata_path(key)
        if not path.exists():
            return None
        
        with open(path, 'r') as f:
            return json.load(f)
    
    def save_processed_document(self, doc: DocumentType, jurisdiction: str) -> None:
        """Save a processed document to cache.
        
        Args:
            doc: Document data
            jurisdiction: Jurisdiction code (BE, ES, DE)
        """
        if "id" not in doc:
            raise ValueError("Document must have an 'id' field")
        
        # Don't cache the vector (too large)
        doc_to_cache = doc.copy()
        if "vector" in doc_to_cache:
            del doc_to_cache["vector"]
            
        path = self.get_processed_path(doc["id"], jurisdiction)
        with open(path, 'w') as f:
            json.dump(doc_to_cache, f)
    
    def load_processed_document(self, doc_id: str, jurisdiction: str) -> Optional[DocumentType]:
        """Load a processed document from cache.
        
        Args:
            doc_id: Document ID
            jurisdiction: Jurisdiction code (BE, ES, DE)
            
        Returns:
            Cached document or None if not found
        """
        path = self.get_processed_path(doc_id, jurisdiction)
        if not path.exists():
            return None
        
        with open(path, 'r') as f:
            return json.load(f)
    
    def clear_expired_cache(self, max_age_days: int = 30) -> int:
        """Clear expired cache files.
        
        Args:
            max_age_days: Maximum age of cache files in days
            
        Returns:
            Number of files deleted
        """
        now = datetime.now()
        count = 0
        
        for path in self.cache_dir.glob("**/*"):
            if path.is_file():
                file_age = datetime.fromtimestamp(path.stat().st_mtime)
                if (now - file_age).days > max_age_days:
                    path.unlink()
                    count += 1
        
        return count


class EmbeddingStrategy(ABC):
    """Abstract base class for text embedding strategies."""
    
    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Embed text into a vector representation.
        
        Args:
            text: Text to embed
            
        Returns:
            Vector representation of the text
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        pass
    
    @abstractmethod
    def get_dimensions(self) -> int:
        """Get the dimensions of the embedding vector.
        
        Returns:
            Number of dimensions in the embedding vector
        """
        pass
    
    def preprocess_text(self, text: str, max_length: Optional[int] = None) -> str:
        """Preprocess text before embedding.
        
        Args:
            text: Text to preprocess
            max_length: Maximum length of text in characters
            
        Returns:
            Preprocessed text
        """
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Truncate if needed
        if max_length and len(text) > max_length:
            text = text[:max_length]
            
        return text


class AzureOpenAIEmbedding(EmbeddingStrategy):
    """Azure OpenAI embedding strategy."""
    
    def __init__(
        self, 
        endpoint: Optional[str] = None, 
        api_key: Optional[str] = None,
        model: str = "text-embedding-ada-002",
        max_length: int = 16000
    ):
        """Initialize Azure OpenAI embedding strategy.
        
        Args:
            endpoint: Azure OpenAI endpoint. Defaults to settings.AZURE_OPENAI_ENDPOINT.
            api_key: Azure OpenAI API key. Defaults to settings.AZURE_OPENAI_KEY.
            model: Model name. Defaults to "text-embedding-ada-002".
            max_length: Maximum text length in characters. Defaults to 16000.
        """
        self.endpoint = endpoint or settings.AZURE_OPENAI_ENDPOINT
        self.api_key = api_key or settings.AZURE_OPENAI_KEY
        self.model = model
        self.max_length = max_length
        
        if not self.endpoint or not self.api_key:
            raise ValueError("Azure OpenAI endpoint and API key must be provided")
    
    def get_dimensions(self) -> int:
        """Get the dimensions of the embedding vector.
        
        Returns:
            Number of dimensions in the embedding vector
        """
        return 1536  # Ada-002 model has 1536 dimensions
    
    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=4, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def embed(self, text: str) -> List[float]:
        """Embed text using Azure OpenAI.
        
        Args:
            text: Text to embed
            
        Returns:
            Vector representation of the text
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        # Preprocess text
        text = self.preprocess_text(text, self.max_length)
        
        # Construct API URL
        api_url = f"{self.endpoint}/openai/deployments/{self.model}/embeddings?api-version=2023-05-15"
        
        try:
            # Call Azure OpenAI API
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    api_url,
                    headers={
                        "Content-Type": "application/json",
                        "api-key": self.api_key
                    },
                    json={
                        "input": text,
                        "dimensions": self.get_dimensions()
                    }
                )
                response.raise_for_status()
                result = response.json()
                
                # Extract embedding from response
                embedding = result["data"][0]["embedding"]
                return embedding
                
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            logger.error(f"Azure OpenAI API error: {str(e)}")
            raise EmbeddingError(f"Failed to generate embedding: {str(e)}") from e
        except (KeyError, IndexError, ValueError) as e:
            logger.error(f"Azure OpenAI API response parsing error: {str(e)}")
            raise EmbeddingError(f"Failed to parse embedding response: {str(e)}") from e


class LocalMiniLMEmbedding(EmbeddingStrategy):
    """Local MiniLM embedding strategy using HuggingFace."""
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        max_length: int = 16000
    ):
        """Initialize local MiniLM embedding strategy.
        
        Args:
            model_name: Model name. Defaults to "all-MiniLM-L6-v2".
            max_length: Maximum text length in characters. Defaults to 16000.
        """
        self.model_name = model_name
        self.max_length = max_length
        self._model = None  # Lazy loading
    
    def get_dimensions(self) -> int:
        """Get the dimensions of the embedding vector.
        
        Returns:
            Number of dimensions in the embedding vector
        """
        return 384  # all-MiniLM-L6-v2 has 384 dimensions
    
    def _load_model(self):
        """Load the model if not already loaded."""
        if self._model is None:
            # Lazy import to avoid loading the model if not needed
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
    
    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=4, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def embed(self, text: str) -> List[float]:
        """Embed text using local MiniLM model.
        
        Args:
            text: Text to embed
            
        Returns:
            Vector representation of the text
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        # Preprocess text
        text = self.preprocess_text(text, self.max_length)
        
        try:
            # Load model
            self._load_model()
            
            # Run in a thread pool to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            model = self._model  # Store in local variable to avoid None reference
            if model is None:
                raise EmbeddingError("Model not loaded")
                
            embedding = await loop.run_in_executor(
                None, lambda: model.encode(text)
            )
            
            # Convert to list of floats
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"LocalMiniLM embedding error: {str(e)}")
            raise EmbeddingError(f"Failed to generate embedding: {str(e)}") from e


def get_embedding_strategy() -> EmbeddingStrategy:
    """Get the appropriate embedding strategy based on settings.
    
    Returns:
        EmbeddingStrategy instance
    """
    if settings.LOCAL_MODE:
        return LocalMiniLMEmbedding()
    else:
        return AzureOpenAIEmbedding()


class DocumentParser(Protocol):
    """Protocol for document parsers."""
    
    async def parse(self, content: bytes, **kwargs) -> Tuple[str, Optional[str]]:
        """Parse document content.
        
        Args:
            content: Document content as bytes
            **kwargs: Additional parser-specific arguments
            
        Returns:
            Tuple of (text, summary)
            
        Raises:
            ParsingError: If parsing fails
        """
        ...


class XMLParser:
    """Parser for XML documents."""
    
    async def parse(
        self, 
        content: bytes, 
        text_xpath: str = "//text", 
        summary_xpath: Optional[str] = None,
        encoding: str = "utf-8"
    ) -> Tuple[str, Optional[str]]:
        """Parse XML document.
        
        Args:
            content: XML content as bytes
            text_xpath: XPath to extract text
            summary_xpath: XPath to extract summary
            encoding: XML encoding
            
        Returns:
            Tuple of (text, summary)
            
        Raises:
            ParsingError: If parsing fails
        """
        try:
            # Parse XML
            xml_str = content.decode(encoding)
            root = ET.fromstring(xml_str)
            
            # Extract text
            text_elements = root.findall(text_xpath)
            text = "\n".join(elem.text for elem in text_elements if elem.text)
            
            # Extract summary if xpath provided
            summary = None
            if summary_xpath:
                summary_elements = root.findall(summary_xpath)
                if summary_elements and summary_elements[0].text:
                    summary = summary_elements[0].text
            
            return text, summary
            
        except Exception as e:
            logger.error(f"XML parsing error: {str(e)}")
            raise ParsingError(f"Failed to parse XML document: {str(e)}") from e


class PDFParser:
    """Parser for PDF documents."""
    
    async def parse(
        self, 
        content: bytes, 
        max_pages: Optional[int] = None,
        summary_pages: int = 1
    ) -> Tuple[str, Optional[str]]:
        """Parse PDF document.
        
        Args:
            content: PDF content as bytes
            max_pages: Maximum number of pages to extract
            summary_pages: Number of pages to use for summary
            
        Returns:
            Tuple of (text, summary)
            
        Raises:
            ParsingError: If parsing fails
        """
        try:
            # Create file-like object
            pdf_file = BytesIO(content)
            
            # Extract text
            text = extract_text(pdf_file, page_numbers=list(range(max_pages)) if max_pages else None)
            
            # Use first page(s) as summary
            summary = None
            if text:
                # Reset file pointer
                pdf_file.seek(0)
                summary = extract_text(pdf_file, page_numbers=list(range(summary_pages)))
            
            return text, summary
            
        except Exception as e:
            logger.error(f"PDF parsing error: {str(e)}")
            raise ParsingError(f"Failed to parse PDF document: {str(e)}") from e


class HTMLParser:
    """Parser for HTML documents."""
    
    async def parse(
        self, 
        content: bytes, 
        encoding: str = "utf-8"
    ) -> Tuple[str, Optional[str]]:
        """Parse HTML document.
        
        Args:
            content: HTML content as bytes
            encoding: HTML encoding
            
        Returns:
            Tuple of (text, summary)
            
        Raises:
            ParsingError: If parsing fails
        """
        try:
            # For a real implementation, use a proper HTML parser like BeautifulSoup
            # For this skeleton, we'll just decode the bytes and return
            html_str = content.decode(encoding)
            
            # In a real implementation, extract text from HTML
            # For now, just return the raw HTML
            return html_str, None
            
        except Exception as e:
            logger.error(f"HTML parsing error: {str(e)}")
            raise ParsingError(f"Failed to parse HTML document: {str(e)}") from e


def get_parser(format: DocumentFormat) -> DocumentParser:
    """Get the appropriate parser for a document format.
    
    Args:
        format: Document format
        
    Returns:
        DocumentParser instance
        
    Raises:
        ValueError: If format is not supported
    """
    parsers = {
        DocumentFormat.XML: XMLParser(),
        DocumentFormat.PDF: PDFParser(),
        DocumentFormat.HTML: HTMLParser(),
    }
    
    if format not in parsers:
        raise ValueError(f"Unsupported document format: {format}")
    
    return parsers[format]


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    stop=stop_after_attempt(3), 
    wait=wait_exponential(multiplier=1, min=4, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
async def download_file(
    url: str, 
    cache_manager: Optional[CacheManager] = None,
    jurisdiction: str = "",
    timeout: float = 30.0,
    headers: Optional[Dict[str, str]] = None
) -> bytes:
    """Download a file from a URL with caching.
    
    Args:
        url: URL to download
        cache_manager: Cache manager instance
        jurisdiction: Jurisdiction code (BE, ES, DE)
        timeout: Request timeout in seconds
        headers: Request headers
        
    Returns:
        File content as bytes
        
    Raises:
        DownloadError: If download fails
    """
    # Initialize cache manager if not provided
    if cache_manager is None:
        cache_manager = CacheManager()
    
    # Get cache path
    cache_path = cache_manager.get_download_path(url, jurisdiction)
    
    # Check cache first
    if cache_manager.is_cached(cache_path):
        logger.info(f"Using cached file: {cache_path}")
        return cache_path.read_bytes()
    
    # Download file
    logger.info(f"Downloading: {url}")
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            content = response.content
        
        # Cache file
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(content)
        
        return content
        
    except (httpx.HTTPError, httpx.TimeoutException) as e:
        logger.error(f"Download error: {str(e)}")
        raise DownloadError(f"Failed to download {url}: {str(e)}") from e


def calculate_checksum(content: bytes) -> str:
    """Calculate SHA-256 checksum of content."""
    return hashlib.sha256(content).hexdigest()


class DocumentProcessor:
    """Processor for documents."""
    
    def __init__(
        self,
        embedding_strategy: Optional[EmbeddingStrategy] = None,
        cache_manager: Optional[CacheManager] = None
    ):
        """Initialize document processor.
        
        Args:
            embedding_strategy: Strategy for generating embeddings
            cache_manager: Cache manager instance
        """
        self.embedding_strategy = embedding_strategy or get_embedding_strategy()
        self.cache_manager = cache_manager or CacheManager()
    
    async def process_document(
        self,
        doc: DocumentType,
        content: bytes,
        format: DocumentFormat,
        parser_kwargs: Optional[Dict[str, Any]] = None
    ) -> DocumentType:
        """Process a document.
        
        Args:
            doc: Document metadata
            content: Document content as bytes
            format: Document format
            parser_kwargs: Additional parser-specific arguments
            
        Returns:
            Processed document with text and embedding
            
        Raises:
            ProcessingError: If processing fails
        """
        if not parser_kwargs:
            parser_kwargs = {}
        
        try:
            # Calculate checksum if not already present
            if "checksum" not in doc:
                doc["checksum"] = calculate_checksum(content)
            
            # Parse document
            parser = get_parser(format)
            text, summary = await parser.parse(content, **parser_kwargs)
            
            # Add text and summary to document
            doc["text"] = text
            if summary and "summary" not in doc:
                doc["summary"] = summary
            
            # Generate embedding
            if text:
                try:
                    vector = await self.embedding_strategy.embed(text)
                    doc["vector"] = vector
                except EmbeddingError as e:
                    logger.warning(f"Error generating embedding for {doc.get('id')}: {e}")
                    doc["vector"] = None
            
            # Cache processed document
            if "jurisdiction" in doc and "id" in doc:
                self.cache_manager.save_processed_document(doc, doc["jurisdiction"])
            
            return doc
            
        except Exception as e:
            logger.error(f"Document processing error: {str(e)}")
            raise ProcessingError(f"Failed to process document: {str(e)}") from e
    
    async def process_batch(
        self,
        docs: List[DocumentType],
        download_func: Callable[[DocumentType], Awaitable[Tuple[bytes, DocumentFormat, Dict[str, Any]]]],
        concurrency: int = 5
    ) -> List[DocumentType]:
        """Process a batch of documents.
        
        Args:
            docs: List of document metadata
            download_func: Async function to download document content
            concurrency: Maximum number of concurrent downloads
            
        Returns:
            List of processed documents
            
        Raises:
            ProcessingError: If processing fails
        """
        processed_docs = []
        semaphore = asyncio.Semaphore(concurrency)
        
        async def process_one(doc: DocumentType) -> Optional[DocumentType]:
            async with semaphore:
                try:
                    # Download document
                    content, format, parser_kwargs = await download_func(doc)
                    
                    # Process document
                    processed_doc = await self.process_document(doc, content, format, parser_kwargs)
                    return processed_doc
                    
                except Exception as e:
                    logger.error(f"Error processing document {doc.get('id', 'unknown')}: {str(e)}")
                    return None
        
        # Process documents concurrently
        tasks = [process_one(doc) for doc in docs]
        results = await asyncio.gather(*tasks)
        
        # Filter out None results (failed processing)
        processed_docs = [doc for doc in results if doc is not None]
        
        return processed_docs


class PipelineContext:
    """Context for pipeline execution."""
    
    def __init__(
        self,
        jurisdiction: str,
        start_date: datetime,
        end_date: datetime,
        session: Session,
        cache_manager: CacheManager,
        embedding_strategy: EmbeddingStrategy,
        document_processor: DocumentProcessor,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize pipeline context.
        
        Args:
            jurisdiction: Jurisdiction code (BE, ES, DE)
            start_date: Start date for document search
            end_date: End date for document search
            session: Database session
            cache_manager: Cache manager instance
            embedding_strategy: Embedding strategy instance
            document_processor: Document processor instance
            metadata: Additional metadata
        """
        self.jurisdiction = jurisdiction
        self.start_date = start_date
        self.end_date = end_date
        self.session = session
        self.cache_manager = cache_manager
        self.embedding_strategy = embedding_strategy
        self.document_processor = document_processor
        self.metadata = metadata or {}
        self.processed_count = 0
        self.error_count = 0
        self.skipped_count = 0
        self.start_time = time.time()
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds.
        
        Returns:
            Elapsed time in seconds
        """
        return time.time() - self.start_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "jurisdiction": self.jurisdiction,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "processed_count": self.processed_count,
            "error_count": self.error_count,
            "skipped_count": self.skipped_count,
            "elapsed_time": self.get_elapsed_time(),
            "metadata": self.metadata
        }


async def run_pipeline(
    jurisdiction: str,
    fetch_func: Callable[[date, date], Awaitable[List[DocumentType]]],
    lookback_hours: Optional[int] = None,
    cache_manager: Optional[CacheManager] = None,
    embedding_strategy: Optional[EmbeddingStrategy] = None,
    document_processor: Optional[DocumentProcessor] = None,
    session_factory: Optional[Callable[[], Session]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Run ETL pipeline for a jurisdiction.
    
    Args:
        jurisdiction: Jurisdiction code (BE, ES, DE)
        fetch_func: Async function that fetches document metadata
        lookback_hours: Hours to look back for documents. Defaults to settings.DOC_LOOKBACK_HOURS.
        cache_manager: Cache manager instance
        embedding_strategy: Embedding strategy instance
        document_processor: Document processor instance
        session_factory: Factory function for database sessions
        metadata: Additional metadata
        
    Returns:
        Dictionary of pipeline statistics
        
    Raises:
        ETLError: If pipeline execution fails
    """
    logger.info(f"Starting ETL pipeline for {jurisdiction}")
    
    # Initialize components
    if cache_manager is None:
        cache_manager = CacheManager()
    
    if embedding_strategy is None:
        embedding_strategy = get_embedding_strategy()
    
    if document_processor is None:
        document_processor = DocumentProcessor(embedding_strategy, cache_manager)
    
    if session_factory is None:
        engine = create_engine(settings.PG_CONNSTR)
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine)
    
    # Calculate date range
    lookback_hours = lookback_hours or settings.DOC_LOOKBACK_HOURS
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=lookback_hours)
    
    # Create database session
    with session_factory() as session:
        # Create pipeline context
        context = PipelineContext(
            jurisdiction=jurisdiction,
            start_date=start_date,
            end_date=end_date,
            session=session,
            cache_manager=cache_manager,
            embedding_strategy=embedding_strategy,
            document_processor=document_processor,
            metadata=metadata or {}
        )
        
        # Fetch documents
        try:
            documents = await fetch_func(start_date.date(), end_date.date())
            logger.info(f"Fetched {len(documents)} documents from {jurisdiction}")
        except Exception as e:
            logger.exception(f"Error fetching documents from {jurisdiction}: {e}")
            context.error_count += 1
            return context.get_stats()
        
        if not documents:
            logger.info(f"No documents found for {jurisdiction} in the specified date range")
            return context.get_stats()
        
        # Process documents
        for doc in documents:
            try:
                # Check if document already exists
                if "checksum" in doc:
                    existing = session.execute(
                        select(Document).filter_by(checksum=doc["checksum"])
                    ).scalar_one_or_none()
                    
                    if existing:
                        logger.info(f"Document already exists: {doc.get('id')}")
                        context.skipped_count += 1
                        continue
                
                # Create document object
                document = Document(
                    id=doc["id"],
                    jurisdiction=doc["jurisdiction"],
                    source_system=doc["source_system"],
                    document_type=doc["document_type"],
                    title=doc["title"],
                    summary=doc.get("summary"),
                    issue_date=doc["issue_date"],
                    effective_date=doc.get("effective_date"),
                    language_orig=doc["language_orig"],
                    blob_url=doc["blob_url"],
                    checksum=doc["checksum"],
                    vector=doc.get("vector")
                )
                
                # Save to database
                try:
                    session.add(document)
                    session.commit()
                    context.processed_count += 1
                    logger.info(f"Saved document: {doc['id']}")
                except IntegrityError:
                    session.rollback()
                    logger.warning(f"Document already exists (integrity error): {doc['id']}")
                    context.skipped_count += 1
                except Exception as e:
                    session.rollback()
                    logger.exception(f"Error saving document {doc['id']}: {e}")
                    context.error_count += 1
            except Exception as e:
                logger.exception(f"Error processing document: {e}")
                context.error_count += 1
        
        logger.info(f"Completed ETL pipeline for {jurisdiction}, processed {context.processed_count} documents")
        return context.get_stats()