"""
Spanish BOE ETL module.

This module fetches and processes documents from the Spanish official bulletin
(Boletín Oficial del Estado). It handles the specific XML format
of the BOE, including namespace handling, tax-related content filtering,
and jurisdiction-specific preprocessing for optimal embedding.

The module implements async functions for all I/O operations and provides
robust error handling specific to the Spanish source.
"""

import asyncio
import logging
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set, cast

from xml.etree import ElementTree as ET

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
logger = logging.getLogger("taxdb.es")

# XML namespaces used in BOE
NAMESPACES = {
    'boe': 'https://www.boe.es/xsd/boe',
    'dc': 'http://purl.org/dc/elements/1.1/',
    'eli': 'http://data.europa.eu/eli/ontology#'
}

# Tax-related keywords for filtering (in Spanish)
TAX_KEYWORDS = {
    # General tax terms
    'impuesto', 'fiscal', 'tributario', 'tributaria', 'tributo',
    'hacienda', 'recaudación', 'contribuyente', 'declaración',
    # Specific taxes
    'iva', 'irpf', 'impuesto sobre la renta', 'impuesto sobre sociedades',
    'impuesto sobre el valor añadido', 'impuesto sobre el patrimonio',
    'impuesto sobre sucesiones', 'impuesto sobre transmisiones',
    # Tax-related terms
    'deducción', 'exención', 'bonificación', 'base imponible',
    'cuota tributaria', 'liquidación', 'retención', 'gravamen',
    # Authorities
    'agencia tributaria', 'aeat', 'ministerio de hacienda'
}

# Tax-related categories in BOE
TAX_CATEGORIES = {
    'fiscal', 'tributario', 'impuestos', 'hacienda', 'finanzas'
}

# Tax-related ministries/departments
TAX_MINISTRIES = {
    'ministerio de hacienda', 'agencia tributaria', 'ministerio de economía'
}


def is_tax_related(text: str, category: Optional[str] = None, ministry: Optional[str] = None) -> bool:
    """Check if a document is tax-related based on keywords, category, and publishing ministry.
    
    This function implements a sophisticated approach to identify tax-related documents by:
    1. Checking for tax-related keywords in the text
    2. Considering document categories that typically contain tax information
    3. Filtering based on publishing ministry or authority
    
    Args:
        text: Document text content
        category: Document category or section
        ministry: Publishing ministry or authority
        
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
    
    # Check ministry
    ministry_match = False
    if ministry:
        ministry_lower = ministry.lower()
        ministry_match = any(tax_min in ministry_lower for tax_min in TAX_MINISTRIES)
    
    # Document is tax-related if any of the checks match
    return keyword_match or category_match or ministry_match


def preprocess_text_for_embedding(text: str) -> str:
    """Preprocess text for optimal embedding specific to Spanish documents.
    
    This function performs jurisdiction-specific preprocessing to improve
    embedding quality for Spanish documents, including:
    - Removing excessive whitespace
    - Normalizing special characters
    - Removing common boilerplate text
    - Handling Spanish-specific abbreviations
    
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
    text = re.sub(r'Boletín Oficial del Estado', '', text)
    text = re.sub(r'www\.boe\.es', '', text)
    
    # Remove page numbers and headers
    text = re.sub(r'Núm\.\s*\d+', '', text)
    text = re.sub(r'Pág\.\s*\d+', '', text)
    
    # Normalize Spanish abbreviations
    text = re.sub(r'(?i)art\.\s*', 'artículo ', text)
    text = re.sub(r'(?i)núm\.\s*', 'número ', text)
    text = re.sub(r'(?i)Excmo\.\s*', 'Excelentísimo ', text)
    text = re.sub(r'(?i)Ilmo\.\s*', 'Ilustrísimo ', text)
    
    return text


async def fetch_boe_documents(
    start_date: date, 
    end_date: date,
    cache_manager: Optional[CacheManager] = None
) -> List[Dict[str, Any]]:
    """Fetch documents from the Spanish BOE.
    
    This function fetches documents from the Spanish BOE for the specified
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
        date_str = current_date.strftime("%Y%m%d")
        
        try:
            # Construct URL for the XML summary using new API
            url = f"https://www.boe.es/datosabiertos/api/boe/sumario/{date_str}"
            
            logger.info(f"Fetching Spanish BOE for {current_date}")
            
            # Download XML summary
            content = await download_file(
                url,
                cache_manager=cache_manager,
                jurisdiction="ES",
                headers={"Accept": "application/xml"}
            )
            
            # Parse XML and extract document URLs
            doc_urls = await extract_document_urls(content)
            
            # Process each document URL
            for doc_url in doc_urls:
                try:
                    # Download document XML
                    doc_content = await download_file(
                        doc_url,
                        cache_manager=cache_manager,
                        jurisdiction="ES",
                        headers={"Accept": "application/xml"}
                    )
                    
                    # Parse document XML
                    doc_data = await parse_boe_xml(doc_content, current_date)
                    
                    # Filter for tax-related documents
                    tax_docs = [
                        doc for doc in doc_data
                        if is_tax_related(
                            doc.get("text", "") + doc.get("title", ""),
                            doc.get("category"),
                            doc.get("ministry")
                        )
                    ]
                    
                    # Add to documents list
                    documents.extend(tax_docs)
                    
                    logger.info(
                        f"Processed BOE document {doc_url}: "
                        f"found {len(doc_data)} documents, {len(tax_docs)} tax-related"
                    )
                except DownloadError as e:
                    logger.warning(f"Download error for BOE document {doc_url}: {e}")
                except ParsingError as e:
                    logger.error(f"Parsing error for BOE document {doc_url}: {e}")
                except Exception as e:
                    logger.exception(f"Unexpected error processing BOE document {doc_url}: {e}")
            
        except DownloadError as e:
            logger.warning(f"Download error for Spanish BOE {current_date}: {e}")
        except ParsingError as e:
            logger.error(f"Parsing error for Spanish BOE {current_date}: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error processing Spanish BOE for {current_date}: {e}")
        
        # Move to next date
        current_date += timedelta(days=1)
    
    return documents


async def extract_document_urls(content: bytes) -> List[str]:
    """Extract document URLs from BOE summary XML.
    
    Args:
        content: XML content as bytes from new API
        
    Returns:
        List of document URLs
        
    Raises:
        ParsingError: If XML parsing fails
    """
    urls: List[str] = []
    
    try:
        # Parse XML
        root = ET.fromstring(content)
        
        # Check API response status
        status_elem = root.find(".//status/code")
        if status_elem is not None and status_elem.text != "200":
            error_text = root.find(".//status/text")
            error_msg = error_text.text if error_text is not None else "Unknown error"
            raise ParsingError(f"API error: {error_msg}")
        
        # Find all document items in the new structure
        # New structure: response/data/sumario/diario/seccion/departamento/epigrafe/item
        doc_elems = root.findall(".//item")
        
        for item in doc_elems:
            # Extract URL from url_xml element
            url_elem = item.find("url_xml")
            if url_elem is not None and url_elem.text:
                urls.append(url_elem.text)
        
        return urls
    except Exception as e:
        logger.error(f"Error extracting document URLs: {str(e)}")
        raise ParsingError(f"Failed to extract document URLs: {str(e)}") from e


async def parse_boe_xml(
    content: bytes, 
    issue_date: date
) -> List[Dict[str, Any]]:
    """Parse Spanish BOE XML content with namespace handling.
    
    This function parses the XML content from the Spanish BOE,
    handling the specific namespaces and structure of the documents.
    
    Args:
        content: XML content as bytes
        issue_date: Publication date
    
    Returns:
        List of document dictionaries
        
    Raises:
        ParsingError: If XML parsing fails
    """
    documents: List[Dict[str, Any]] = []
    
    try:
        # Register namespaces for proper XML parsing
        for prefix, uri in NAMESPACES.items():
            ET.register_namespace(prefix, uri)
        
        # Parse XML
        root = ET.fromstring(content)
        
        # Extract document metadata - updated for new XML structure
        doc_id_elem = root.find(".//identificador")
        
        if doc_id_elem is None or not doc_id_elem.text:
            logger.warning(f"No document ID found in XML for {issue_date}")
            return documents
        
        doc_id = f"ES:{doc_id_elem.text}"
        
        # Extract title
        title_elem = root.find(".//titulo")
        title = title_elem.text if title_elem is not None and title_elem.text else "Untitled"
        
        # Extract publication date
        pub_date_elem = root.find(".//fecha_publicacion")
        pub_date = None
        
        if pub_date_elem is not None and pub_date_elem.text:
            try:
                pub_date = datetime.strptime(pub_date_elem.text, "%Y%m%d").date()
            except ValueError:
                logger.warning(f"Invalid publication date format: {pub_date_elem.text}")
        
        # Use provided issue_date if pub_date is not available
        doc_date = pub_date or issue_date
        
        # Extract department/ministry
        ministry_elem = root.find(".//departamento")
        ministry = ministry_elem.text if ministry_elem is not None and ministry_elem.text else None
        
        # Extract category/subject (try multiple possible fields)
        category_elem = root.find(".//materia") or root.find(".//rango")
        category = category_elem.text if category_elem is not None and category_elem.text else None
        
        # Extract text content
        text_elem = root.find(".//texto")
        if text_elem is not None:
            # Handle both text content and child elements
            text_parts = []
            if text_elem.text:
                text_parts.append(text_elem.text)
            for child in text_elem:
                if child.text:
                    text_parts.append(child.text)
                if child.tail:
                    text_parts.append(child.tail)
            text = " ".join(text_parts)
        else:
            text = ""
        
        # Extract summary (may not exist in all documents)
        summary_elem = root.find(".//resumen")
        summary = summary_elem.text if summary_elem is not None and summary_elem.text else None
        
        # Extract effective date
        effective_date_elem = root.find(".//fecha_vigencia")
        effective_date = None
        
        if effective_date_elem is not None and effective_date_elem.text:
            try:
                effective_date = datetime.strptime(effective_date_elem.text, "%Y%m%d").date()
            except ValueError:
                logger.warning(f"Invalid effective date format: {effective_date_elem.text}")
        
        # Extract document type from rango
        doc_type_elem = root.find(".//rango")
        doc_type = doc_type_elem.text if doc_type_elem is not None and doc_type_elem.text else "legal"
        
        # Preprocess text for better embedding
        processed_text = preprocess_text_for_embedding(text)
        
        # Calculate checksum
        checksum = calculate_checksum(f"{doc_id}:{title}:{doc_date}:{processed_text}".encode())
        
        # Extract PDF URL from metadata
        pdf_url_elem = root.find(".//url_pdf")
        if pdf_url_elem is not None and pdf_url_elem.text:
            if pdf_url_elem.text.startswith('/'):
                blob_url = f"https://www.boe.es{pdf_url_elem.text}"
            else:
                blob_url = pdf_url_elem.text
        else:
            # Fallback to old format if no url_pdf found
            blob_url = f"https://www.boe.es/boe/dias/{doc_date.strftime('%Y/%m/%d')}/pdfs/{doc_id_elem.text}.pdf"
        
        # Create document dictionary
        document = {
            "id": doc_id,
            "jurisdiction": "ES",
            "source_system": "boe",
            "document_type": doc_type,
            "title": title,
            "summary": summary,
            "issue_date": doc_date,
            "effective_date": effective_date,
            "language_orig": "es",
            "blob_url": blob_url,
            "checksum": checksum,
            "text": processed_text,
            "category": category,
            "ministry": ministry
        }
        
        documents.append(document)
    except Exception as e:
        logger.error(f"Error parsing XML: {str(e)}")
        raise ParsingError(f"Failed to parse Spanish BOE XML: {str(e)}") from e
    
    return documents


async def download_boe_document(
    doc: Dict[str, Any]
) -> Tuple[bytes, DocumentFormat, Dict[str, Any]]:
    """Download a document from the Spanish BOE.
    
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
        content = await download_file(url, jurisdiction="ES")
        
        # Determine format and parser kwargs based on URL
        if url.endswith(".xml"):
            format = DocumentFormat.XML
            parser_kwargs = {
                "text_xpath": ".//boe:texto",
                "summary_xpath": ".//boe:resumen"
            }
        elif url.endswith(".pdf"):
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


async def process_boe_documents(
    documents: List[Dict[str, Any]],
    document_processor: Optional[DocumentProcessor] = None
) -> List[Dict[str, Any]]:
    """Process documents from the Spanish BOE.
    
    This function processes the documents fetched from the Spanish BOE,
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
        download_boe_document,
        concurrency=5
    )
    
    return processed_docs


async def main():
    """Main entry point for Spanish BOE ETL.
    
    This function runs the ETL pipeline for the Spanish BOE,
    fetching, processing, and storing documents.
    """
    # Use historical date range for testing with real documents
    # Using dates from March 2024 when documents are definitely available
    start_date = date(2024, 3, 15)
    end_date = date(2024, 3, 18)
    
    logger.info(f"Starting Spanish BOE ETL pipeline for date range: {start_date} to {end_date}")
    
    # Run pipeline with metadata and specific date range
    metadata = {
        "source": "Boletín Oficial del Estado",
        "languages": ["es"],
        "document_types": ["legal", "fiscal"]
    }
    
    # Create a wrapper function that uses our specific date range
    async def fetch_with_date_range(_, __):
        """Wrapper to use specific historical dates instead of calculated range."""
        return await fetch_boe_documents(start_date, end_date)
    
    stats = await run_pipeline(
        "ES",
        fetch_with_date_range,
        metadata=metadata
    )
    
    logger.info(f"Completed Spanish BOE ETL pipeline: {stats}")


if __name__ == "__main__":
    asyncio.run(main())