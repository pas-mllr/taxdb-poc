"""
German BGBL ETL module.

This module fetches and processes documents from the German Federal Law Gazette
(Bundesgesetzblatt). It handles the specific PDF format of the BGBL, including
multi-column layouts, tax-related content filtering, and jurisdiction-specific
preprocessing for optimal embedding.

The module implements async functions for all I/O operations and provides
robust error handling specific to the German source and PDF processing.
"""

import asyncio
import json
import logging
import re
from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set, cast

import httpx
import pdfplumber
from pdfminer.high_level import extract_text

from src import settings
from src.etl.utils import (
    download_file, 
    run_pipeline, 
    calculate_checksum, 
    DocumentFormat,
    DownloadError,
    ParsingError,
    CacheManager,
    DocumentProcessor
)

# Configure logger
logger = logging.getLogger("taxdb.de")

# Tax-related keywords for filtering (in German)
TAX_KEYWORDS = {
    # General tax terms
    'steuer', 'steuern', 'steuerlich', 'besteuerung', 'abgabe', 'abgaben',
    'finanzamt', 'finanzverwaltung', 'steuerpflichtig', 'steuererklärung',
    # Specific taxes
    'einkommensteuer', 'körperschaftsteuer', 'umsatzsteuer', 'mehrwertsteuer',
    'gewerbesteuer', 'grundsteuer', 'erbschaftsteuer', 'schenkungsteuer',
    'kapitalertragsteuer', 'lohnsteuer', 'grunderwerbsteuer',
    # Tax-related terms
    'steuerfreibetrag', 'steuerabzug', 'steuerbefreiung', 'steuervergünstigung',
    'bemessungsgrundlage', 'steuersatz', 'steuervorauszahlung', 'steuererstattung',
    # Authorities
    'bundesfinanzministerium', 'finanzministerium', 'bundeszentralamt für steuern'
}

# Tax-related categories in BGBL
TAX_CATEGORIES = {
    'steuer', 'abgabe', 'finanz', 'haushalt', 'wirtschaft'
}


def is_tax_related(text: str, category: Optional[str] = None) -> bool:
    """Check if a document is tax-related based on keywords and category.
    
    This function implements a sophisticated approach to identify tax-related documents by:
    1. Checking for tax-related keywords in the text
    2. Considering document categories that typically contain tax information
    
    Args:
        text: Document text content
        category: Document category or section
        
    Returns:
        True if the document is tax-related, False otherwise
    """
    text_lower = text.lower()
    
    # Check keywords in text
    keyword_match = any(keyword in text_lower for keyword in TAX_KEYWORDS)
    
    # Check category
    category_match = False
    if category:
        category_lower = category.lower()
        category_match = any(tax_cat in category_lower for tax_cat in TAX_CATEGORIES)
    
    # Document is tax-related if any of the checks match
    return keyword_match or category_match


def preprocess_text_for_embedding(text: str) -> str:
    """Preprocess text for optimal embedding specific to German documents.
    
    This function performs jurisdiction-specific preprocessing to improve
    embedding quality for German documents, including:
    - Removing excessive whitespace
    - Normalizing special characters
    - Removing common boilerplate text
    - Handling German-specific abbreviations
    
    Args:
        text: Original document text
        
    Returns:
        Preprocessed text ready for embedding
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove common boilerplate text
    text = re.sub(r'Bundesgesetzblatt', '', text)
    text = re.sub(r'www\.bgbl\.de', '', text)
    text = re.sub(r'www\.bundesanzeiger\.de', '', text)
    
    # Remove page numbers and headers
    text = re.sub(r'Seite \d+ von \d+', '', text)
    text = re.sub(r'Nr\.\s*\d+', '', text)
    
    # Normalize German abbreviations
    text = re.sub(r'(?i)Abs\.\s*', 'Absatz ', text)
    text = re.sub(r'(?i)Art\.\s*', 'Artikel ', text)
    text = re.sub(r'(?i)Nr\.\s*', 'Nummer ', text)
    text = re.sub(r'(?i)S\.\s*', 'Seite ', text)
    text = re.sub(r'(?i)v\.\s*', 'vom ', text)
    text = re.sub(r'(?i)z\.\s*B\.', 'zum Beispiel', text)
    
    return text


async def fetch_bgbl_documents(
    start_date: date, 
    end_date: date,
    cache_manager: Optional[CacheManager] = None
) -> List[Dict[str, Any]]:
    """Fetch documents from the German BGBL.
    
    This function fetches documents from the German BGBL for the specified
    date range. It implements retry logic for robust error handling and
    uses caching to avoid redundant downloads.
    
    Args:
        start_date: Start date for document search
        end_date: End date for document search
        cache_manager: Cache manager instance
        
    Returns:
        List of document dictionaries
        
    Raises:
        DownloadError: If document download fails after retries
    """
    documents: List[Dict[str, Any]] = []
    
    if cache_manager is None:
        cache_manager = CacheManager()
    
    # Process each date in the range
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        
        try:
            # Construct URL for the index JSON
            url = f"https://www.bgbl.de/api/v1/publications?date={date_str}"
            
            logger.info(f"Fetching German BGBL for {current_date}")
            
            # Download index JSON
            content = await download_file(
                url, 
                cache_manager=cache_manager,
                jurisdiction="DE",
                headers={"Accept": "application/json"}
            )
            
            # Parse index JSON
            index_data = json.loads(content)
            
            # Process documents from index
            date_docs = await process_bgbl_index(index_data, current_date, cache_manager)
            
            # Filter for tax-related documents
            tax_docs = [
                doc for doc in date_docs 
                if is_tax_related(
                    doc.get("text", "") + doc.get("title", ""),
                    doc.get("category")
                )
            ]
            
            # Add to documents list
            documents.extend(tax_docs)
            
            logger.info(
                f"Processed German BGBL for {current_date}: "
                f"found {len(date_docs)} documents, {len(tax_docs)} tax-related"
            )
        except DownloadError as e:
            logger.warning(f"Download error for German BGBL {current_date}: {e}")
        except ParsingError as e:
            logger.error(f"Parsing error for German BGBL {current_date}: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error processing German BGBL for {current_date}: {e}")
        
        # Move to next date
        current_date += timedelta(days=1)
    
    return documents


async def process_bgbl_index(
    index_data: Dict[str, Any], 
    issue_date: date,
    cache_manager: Optional[CacheManager] = None
) -> List[Dict[str, Any]]:
    """Process German BGBL index data.
    
    This function processes the index data from the German BGBL,
    downloading and extracting text from PDF documents.
    
    Args:
        index_data: Index JSON data
        issue_date: Issue date
        cache_manager: Cache manager instance
        
    Returns:
        List of document dictionaries
        
    Raises:
        ParsingError: If index parsing fails
    """
    documents: List[Dict[str, Any]] = []
    
    if cache_manager is None:
        cache_manager = CacheManager()
    
    try:
        # Extract documents from index
        items = index_data.get("items", [])
        
        for item in items:
            try:
                # Extract document ID
                doc_id = item.get("id")
                if not doc_id:
                    continue
                
                # Format document ID
                doc_id = f"DE:{doc_id}"
                
                # Extract title
                title = item.get("title", "Untitled")
                
                # Extract category
                category = item.get("category")
                
                # Extract PDF URL
                pdf_url = item.get("pdfUrl")
                if not pdf_url:
                    continue
                
                # Extract publication date
                pub_date_str = item.get("publicationDate")
                pub_date = None
                if pub_date_str:
                    try:
                        pub_date = datetime.strptime(pub_date_str, "%Y-%m-%d").date()
                    except ValueError:
                        logger.warning(f"Invalid publication date format: {pub_date_str}")
                
                # Use provided issue_date if pub_date is not available
                doc_date = pub_date or issue_date
                
                # Extract effective date
                effective_date_str = item.get("effectiveDate")
                effective_date = None
                if effective_date_str:
                    try:
                        effective_date = datetime.strptime(effective_date_str, "%Y-%m-%d").date()
                    except ValueError:
                        logger.warning(f"Invalid effective date format: {effective_date_str}")
                
                # Download PDF
                pdf_content = await download_file(
                    pdf_url, 
                    cache_manager=cache_manager,
                    jurisdiction="DE"
                )
                
                # Extract text from PDF
                text = await extract_text_from_pdf(pdf_content)
                
                # Preprocess text for better embedding
                processed_text = preprocess_text_for_embedding(text)
                
                # Calculate checksum
                checksum = calculate_checksum(f"{doc_id}:{title}:{doc_date}:{processed_text}".encode())
                
                # Create document dictionary
                document = {
                    "id": doc_id,
                    "jurisdiction": "DE",
                    "source_system": "bgbl",
                    "document_type": "legal",
                    "title": title,
                    "summary": None,  # Extract summary from first page if needed
                    "issue_date": doc_date,
                    "effective_date": effective_date,
                    "language_orig": "de",
                    "blob_url": pdf_url,
                    "checksum": checksum,
                    "text": processed_text,
                    "category": category
                }
                
                documents.append(document)
            except Exception as e:
                logger.warning(f"Error processing document: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing index: {str(e)}")
        raise ParsingError(f"Failed to process German BGBL index: {str(e)}") from e
    
    return documents


async def extract_text_from_pdf(content: bytes) -> str:
    """Extract text from PDF content with multi-column support.
    
    This function extracts text from PDF documents, handling multi-column layouts
    and other complex formatting commonly found in German BGBL documents.
    
    Args:
        content: PDF content as bytes
        
    Returns:
        Extracted text
        
    Raises:
        ParsingError: If text extraction fails
    """
    try:
        # First try with pdfplumber for better handling of multi-column layouts
        text = await _extract_with_pdfplumber(content)
        
        # If pdfplumber fails or returns empty text, fall back to pdfminer
        if not text.strip():
            text = await _extract_with_pdfminer(content)
        
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise ParsingError(f"Failed to extract text from PDF: {str(e)}") from e


async def _extract_with_pdfplumber(content: bytes) -> str:
    """Extract text from PDF using pdfplumber.
    
    This function uses pdfplumber to extract text from PDF documents,
    which provides better handling of multi-column layouts.
    
    Args:
        content: PDF content as bytes
        
    Returns:
        Extracted text
    """
    # Run in a thread pool to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    
    def extract():
        text_parts = []
        
        with pdfplumber.open(BytesIO(content)) as pdf:
            for page in pdf.pages:
                # Extract text from page
                page_text = page.extract_text(x_tolerance=3, y_tolerance=3)
                if page_text:
                    text_parts.append(page_text)
                
                # If text extraction fails, try to extract tables
                if not page_text or len(page_text) < 50:  # Likely failed or header only
                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            row_text = " ".join([cell or "" for cell in row])
                            if row_text.strip():
                                text_parts.append(row_text)
        
        return "\n\n".join(text_parts)
    
    return await loop.run_in_executor(None, extract)


async def _extract_with_pdfminer(content: bytes) -> str:
    """Extract text from PDF using pdfminer.
    
    This function uses pdfminer to extract text from PDF documents
    as a fallback method.
    
    Args:
        content: PDF content as bytes
        
    Returns:
        Extracted text
    """
    # Run in a thread pool to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    
    def extract():
        return extract_text(BytesIO(content))
    
    return await loop.run_in_executor(None, extract)


async def download_bgbl_document(
    doc: Dict[str, Any]
) -> Tuple[bytes, DocumentFormat, Dict[str, Any]]:
    """Download a document from the German BGBL.
    
    This function is used by the document processor to download
    the actual document content for processing.
    
    Args:
        doc: Document metadata dictionary
        
    Returns:
        Tuple of (content, format, parser_kwargs)
        
    Raises:
        DownloadError: If document download fails
    """
    url = doc["blob_url"]
    
    try:
        # Download document
        content = await download_file(url, jurisdiction="DE")
        
        # Determine format and parser kwargs based on URL
        if url.endswith(".pdf"):
            format = DocumentFormat.PDF
            parser_kwargs = {
                "max_pages": None,  # Process all pages
                "summary_pages": 1   # Use first page as summary
            }
        else:
            format = DocumentFormat.TEXT
            parser_kwargs = {}
        
        return content, format, parser_kwargs
    except Exception as e:
        logger.error(f"Error downloading document {doc['id']}: {str(e)}")
        raise DownloadError(f"Failed to download document {doc['id']}: {str(e)}") from e


async def process_bgbl_documents(
    documents: List[Dict[str, Any]],
    document_processor: Optional[DocumentProcessor] = None
) -> List[Dict[str, Any]]:
    """Process documents from the German BGBL.
    
    This function processes the documents fetched from the German BGBL,
    generating embeddings and preparing them for storage.
    
    Args:
        documents: List of document dictionaries
        document_processor: Document processor instance
        
    Returns:
        List of processed document dictionaries
    """
    if not document_processor:
        document_processor = DocumentProcessor()
    
    # Process documents in batches
    processed_docs = await document_processor.process_batch(
        documents,
        download_bgbl_document,
        concurrency=5
    )
    
    return processed_docs


async def main():
    """Main entry point for German BGBL ETL.
    
    This function runs the ETL pipeline for the German BGBL,
    fetching, processing, and storing documents.
    """
    # Calculate date range
    today = datetime.now().date()
    lookback = today - timedelta(hours=settings.DOC_LOOKBACK_HOURS)
    
    logger.info(f"Starting German BGBL ETL pipeline")
    
    # Run pipeline with metadata
    metadata = {
        "source": "Bundesgesetzblatt",
        "languages": ["de"],
        "document_types": ["legal", "fiscal"]
    }
    
    stats = await run_pipeline(
        "DE", 
        fetch_bgbl_documents,
        metadata=metadata
    )
    
    logger.info(f"Completed German BGBL ETL pipeline: {stats}")


if __name__ == "__main__":
    asyncio.run(main())