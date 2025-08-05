"""
Belgian Moniteur ETL module.

This module fetches and processes documents from the Belgian official journal
(Moniteur Belge / Belgisch Staatsblad). It handles the specific XML format
of the Moniteur Belge, including namespace handling, tax-related content filtering,
and jurisdiction-specific preprocessing for optimal embedding.

The module implements async functions for all I/O operations and provides
robust error handling specific to the Belgian source.
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
logger = logging.getLogger("taxdb.be")

# XML namespaces used in Moniteur Belge
NAMESPACES = {
    'mb': 'http://www.ejustice.just.fgov.be/moniteur',
    'eli': 'http://data.europa.eu/eli/ontology#',
    'dc': 'http://purl.org/dc/elements/1.1/'
}

# Tax-related keywords for filtering (in Dutch and French)
TAX_KEYWORDS = {
    # Dutch keywords
    'belasting', 'fiscaal', 'fiscale', 'btw', 'inkomstenbelasting', 
    'vennootschapsbelasting', 'onroerende voorheffing', 'accijnzen',
    # French keywords
    'impôt', 'fiscal', 'fiscale', 'tva', 'impôt sur le revenu',
    'impôt des sociétés', 'précompte immobilier', 'accises'
}


def is_tax_related(text: str) -> bool:
    """Check if a document is tax-related based on keywords.
    
    Args:
        text: Document text content
        
    Returns:
        True if the document is tax-related, False otherwise
    """
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in TAX_KEYWORDS)


def preprocess_text_for_embedding(text: str) -> str:
    """Preprocess text for optimal embedding specific to Belgian documents.
    
    This function performs jurisdiction-specific preprocessing to improve
    embedding quality for Belgian documents, including:
    - Removing excessive whitespace
    - Normalizing special characters
    - Removing common boilerplate text
    
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
    text = re.sub(r'Belgisch Staatsblad - Moniteur Belge', '', text)
    text = re.sub(r'www\.ejustice\.just\.fgov\.be', '', text)
    
    # Remove page numbers and headers
    text = re.sub(r'\d+\s*/\s*\d+', '', text)
    
    return text


async def fetch_moniteur_documents(
    start_date: date, 
    end_date: date,
    languages: List[str] = ["nl", "fr"]
) -> List[Dict[str, Any]]:
    """Fetch documents from the Belgian Moniteur.
    
    This function fetches documents from the Belgian Moniteur for the specified
    date range. It supports multiple languages and implements retry logic for
    robust error handling.
    
    Args:
        start_date: Start date for document search
        end_date: End date for document search
        languages: List of language codes to fetch (default: ["nl", "fr"])
    
    Returns:
        List of document dictionaries
        
    Raises:
        DownloadError: If document download fails after retries
    """
    documents: List[Dict[str, Any]] = []
    cache_manager = CacheManager()
    
    # Process each date in the range
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y%m%d")
        
        # Process each language
        for lang in languages:
            # Construct URL
            url = f"https://www.ejustice.just.fgov.be/eli/{date_str}/MONITOR/{lang}/xml"
            
            # Create cache path
            cache_path = cache_manager.get_download_path(url, "BE")
            
            try:
                logger.info(f"Fetching Belgian Moniteur for {current_date} in {lang}")
                
                # Download XML
                content = await download_file(
                    url, 
                    cache_manager=cache_manager,
                    jurisdiction="BE",
                    headers={"Accept": "application/xml"}
                )
                
                # Parse XML
                xml_docs = await parse_moniteur_xml(content, lang, current_date)
                
                # Filter for tax-related documents
                tax_docs = [doc for doc in xml_docs if is_tax_related(doc.get("text", "") + doc.get("title", ""))]
                
                # Add to documents list
                documents.extend(tax_docs)
                
                logger.info(
                    f"Processed Belgian Moniteur for {current_date} in {lang}: "
                    f"found {len(xml_docs)} documents, {len(tax_docs)} tax-related"
                )
            except DownloadError as e:
                logger.warning(f"Download error for Belgian Moniteur {current_date} in {lang}: {e}")
            except ParsingError as e:
                logger.error(f"Parsing error for Belgian Moniteur {current_date} in {lang}: {e}")
            except Exception as e:
                logger.exception(f"Unexpected error processing Belgian Moniteur for {current_date} in {lang}: {e}")
        
        # Move to next date
        current_date += timedelta(days=1)
    
    return documents


async def parse_moniteur_xml(
    content: bytes, 
    language: str,
    issue_date: date
) -> List[Dict[str, Any]]:
    """Parse Belgian Moniteur XML content with namespace handling.
    
    This function parses the XML content from the Belgian Moniteur,
    handling the specific namespaces and structure of the documents.
    
    Args:
        content: XML content as bytes
        language: Language code of the content (nl or fr)
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
        
        # Find all document elements with namespaces
        # Try different possible paths as the structure might vary
        doc_elems = root.findall(".//mb:doc", NAMESPACES)
        if not doc_elems:
            doc_elems = root.findall(".//doc", {})  # Try without namespace
        
        if not doc_elems:
            logger.warning(f"No documents found in XML for {issue_date} in {language}")
            return documents
        
        for doc_elem in doc_elems:
            try:
                # Extract ELI identifier with namespace handling
                eli = doc_elem.find(".//mb:eli", NAMESPACES) or doc_elem.find(".//eli", {})
                if eli is None or not eli.text:
                    continue
                
                doc_id = f"BE:{eli.text.split('/')[-1]}"
                
                # Extract title with namespace handling
                title_elem = doc_elem.find(".//mb:title", NAMESPACES) or doc_elem.find(".//title", {})
                title = title_elem.text if title_elem is not None and title_elem.text else "Untitled"
                
                # Extract publication date with namespace handling
                pub_date_elem = doc_elem.find(".//mb:pubdate", NAMESPACES) or doc_elem.find(".//pubdate", {})
                pub_date = None
                
                if pub_date_elem is not None and pub_date_elem.text:
                    try:
                        pub_date = datetime.strptime(pub_date_elem.text, "%Y-%m-%d").date()
                    except ValueError:
                        logger.warning(f"Invalid publication date format: {pub_date_elem.text}")
                
                # Use provided issue_date if pub_date is not available
                doc_date = pub_date or issue_date
                
                # Extract text content with namespace handling
                text_elem = doc_elem.find(".//mb:text", NAMESPACES) or doc_elem.find(".//text", {})
                text = text_elem.text if text_elem is not None and text_elem.text else ""
                
                # Extract summary with namespace handling
                summary_elem = doc_elem.find(".//mb:summary", NAMESPACES) or doc_elem.find(".//summary", {})
                summary = summary_elem.text if summary_elem is not None and summary_elem.text else None
                
                # Extract effective date with namespace handling
                effective_date_elem = doc_elem.find(".//mb:effective_date", NAMESPACES) or doc_elem.find(".//effective_date", {})
                effective_date = None
                
                if effective_date_elem is not None and effective_date_elem.text:
                    try:
                        effective_date = datetime.strptime(effective_date_elem.text, "%Y-%m-%d").date()
                    except ValueError:
                        logger.warning(f"Invalid effective date format: {effective_date_elem.text}")
                
                # Extract document type with namespace handling
                doc_type_elem = doc_elem.find(".//mb:type", NAMESPACES) or doc_elem.find(".//type", {})
                doc_type = doc_type_elem.text if doc_type_elem is not None and doc_type_elem.text else "legal"
                
                # Preprocess text for better embedding
                processed_text = preprocess_text_for_embedding(text)
                
                # Calculate checksum
                checksum = calculate_checksum(f"{doc_id}:{title}:{doc_date}:{processed_text}".encode())
                
                # Create document dictionary
                document = {
                    "id": doc_id,
                    "jurisdiction": "BE",
                    "source_system": "moniteur",
                    "document_type": doc_type,
                    "title": title,
                    "summary": summary,
                    "issue_date": doc_date,
                    "effective_date": effective_date,
                    "language_orig": language,
                    "blob_url": f"https://www.ejustice.just.fgov.be/eli/{doc_date.strftime('%Y%m%d')}/MONITOR/{language}/pdf",
                    "checksum": checksum,
                    "text": processed_text
                }
                
                documents.append(document)
            except Exception as e:
                logger.warning(f"Error parsing document: {str(e)}")
    except Exception as e:
        logger.error(f"Error parsing XML: {str(e)}")
        raise ParsingError(f"Failed to parse Belgian Moniteur XML: {str(e)}") from e
    
    return documents


async def download_moniteur_document(
    doc: Dict[str, Any]
) -> Tuple[bytes, DocumentFormat, Dict[str, Any]]:
    """Download a document from the Belgian Moniteur.
    
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
        content = await download_file(url, jurisdiction="BE")
        
        # Determine format and parser kwargs based on URL
        if url.endswith(".xml"):
            format = DocumentFormat.XML
            parser_kwargs = {
                "text_xpath": ".//mb:text",
                "summary_xpath": ".//mb:summary"
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


async def process_moniteur_documents(
    documents: List[Dict[str, Any]],
    document_processor: Optional[DocumentProcessor] = None
) -> List[Dict[str, Any]]:
    """Process documents from the Belgian Moniteur.
    
    This function processes the documents fetched from the Belgian Moniteur,
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
        download_moniteur_document,
        concurrency=5
    )
    
    return processed_docs


async def main():
    """Main entry point for Belgian Moniteur ETL.
    
    This function runs the ETL pipeline for the Belgian Moniteur,
    fetching, processing, and storing documents.
    """
    # Calculate date range
    today = datetime.now().date()
    lookback = today - timedelta(hours=settings.DOC_LOOKBACK_HOURS)
    
    logger.info(f"Starting Belgian Moniteur ETL pipeline")
    
    # Run pipeline with metadata
    metadata = {
        "source": "Moniteur Belge",
        "languages": ["nl", "fr"],
        "document_types": ["legal", "fiscal"]
    }
    
    stats = await run_pipeline(
        "BE", 
        fetch_moniteur_documents,
        metadata=metadata
    )
    
    logger.info(f"Completed Belgian Moniteur ETL pipeline: {stats}")


if __name__ == "__main__":
    asyncio.run(main())