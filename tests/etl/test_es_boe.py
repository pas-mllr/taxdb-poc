"""
Tests for Spanish BOE ETL pipeline.

This module tests the ETL pipeline for the Spanish BOE (Boletín Oficial del Estado),
including document fetching, parsing, and processing.
"""

import asyncio
import json
import pytest
import pytest_asyncio
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import patch, MagicMock, AsyncMock

from src.etl.es_boe import (
    is_tax_related,
    preprocess_text_for_embedding,
    fetch_boe_documents,
    extract_document_urls,
    parse_boe_xml,
    download_boe_document,
    process_boe_documents,
    main
)
from src.etl.utils import (
    CacheManager,
    DocumentProcessor,
    DocumentFormat,
    DownloadError,
    ParsingError
)


def test_is_tax_related():
    """Test is_tax_related function."""
    # Test with tax-related text
    assert is_tax_related("Este documento trata sobre impuestos y asuntos fiscales")
    assert is_tax_related("Tasa del IVA para servicios")
    assert is_tax_related("Impuesto sobre sociedades en España")
    
    # Test with tax-related category
    assert is_tax_related("Texto normal", category="fiscal")
    assert is_tax_related("Texto normal", category="tributario")
    assert is_tax_related("Texto normal", category="impuestos")
    
    # Test with tax-related ministry
    assert is_tax_related("Texto normal", ministry="Ministerio de Hacienda")
    assert is_tax_related("Texto normal", ministry="Agencia Tributaria")
    
    # Test with non-tax-related text, category, and ministry
    assert not is_tax_related("Este documento trata sobre el clima", 
                             category="clima", 
                             ministry="Ministerio de Medio Ambiente")


def test_preprocess_text_for_embedding():
    """Test preprocess_text_for_embedding function."""
    # Test with normal text
    text = "Este es un texto normal."
    processed = preprocess_text_for_embedding(text)
    assert processed == text
    
    # Test with excessive whitespace
    text = "  Este   tiene \n\n demasiado \t espacio en blanco.  "
    processed = preprocess_text_for_embedding(text)
    assert processed == "Este tiene demasiado espacio en blanco."
    
    # Test with boilerplate text
    text = "Boletín Oficial del Estado\nEste es un texto con boilerplate."
    processed = preprocess_text_for_embedding(text)
    assert "Boletín Oficial del Estado" not in processed
    assert "Este es un texto con boilerplate." in processed
    
    # Test with page numbers
    text = "Núm. 123 Pág. 45"
    processed = preprocess_text_for_embedding(text)
    assert "Núm. 123" not in processed
    assert "Pág. 45" not in processed
    
    # Test with URL
    text = "Visite www.boe.es para más información."
    processed = preprocess_text_for_embedding(text)
    assert "www.boe.es" not in processed
    
    # Test with Spanish abbreviations
    text = "Art. 123 del Núm. 456"
    processed = preprocess_text_for_embedding(text)
    assert "artículo" in processed
    assert "número" in processed
    
    # Test with empty text
    assert preprocess_text_for_embedding("") == ""


@pytest.mark.asyncio
async def test_extract_document_urls(etl_test_data):
    """Test extract_document_urls function."""
    # Test XML content
    xml_content = b"""
    <seccion id="A">
        <departamento>
            <item>
                <id>BOE-A-2025-12345</id>
            </item>
            <item>
                <id>BOE-A-2025-12346</id>
            </item>
        </departamento>
    </seccion>
    <seccion id="B">
        <departamento>
            <item>
                <id>BOE-B-2025-12347</id>
            </item>
        </departamento>
    </seccion>
    """
    
    # Extract URLs for section A
    urls = await extract_document_urls(xml_content, "A")
    
    # Check results
    assert len(urls) == 2
    assert "BOE-A-2025-12345" in urls[0]
    assert "BOE-A-2025-12346" in urls[1]
    
    # Extract URLs for section B
    urls = await extract_document_urls(xml_content, "B")
    
    # Check results
    assert len(urls) == 1
    assert "BOE-B-2025-12347" in urls[0]
    
    # Test with invalid XML
    with pytest.raises(ParsingError):
        await extract_document_urls(b"<invalid XML", "A")
    
    # Test with XML without matching section
    urls = await extract_document_urls(xml_content, "C")
    assert len(urls) == 0
    
    # Test with XML with namespaces
    xml_content = b"""
    <boe:seccion xmlns:boe="https://www.boe.es/xsd/boe" id="A">
        <boe:departamento>
            <boe:item>
                <boe:id>BOE-A-2025-12345</boe:id>
            </boe:item>
        </boe:departamento>
    </boe:seccion>
    """
    
    urls = await extract_document_urls(xml_content, "A")
    assert len(urls) == 1
    assert "BOE-A-2025-12345" in urls[0]


@pytest.mark.asyncio
async def test_parse_boe_xml(etl_test_data):
    """Test parse_boe_xml function."""
    # Test with valid XML
    xml_content = etl_test_data["ES"]["xml_content"]
    issue_date = date(2025, 8, 1)
    
    documents = await parse_boe_xml(xml_content, issue_date)
    
    # Check results
    assert len(documents) == 1
    assert documents[0]["id"] == "ES:BOE-A-2025-12345"
    assert documents[0]["jurisdiction"] == "ES"
    assert documents[0]["title"] == "Test Tax Regulation"
    assert documents[0]["issue_date"] == date(2025, 8, 1)
    assert documents[0]["language_orig"] == "es"
    assert documents[0]["ministry"] == "Ministerio de Hacienda"
    assert documents[0]["category"] == "Fiscal"
    assert "checksum" in documents[0]
    
    # Test with invalid XML
    with pytest.raises(ParsingError):
        await parse_boe_xml(b"<invalid XML", issue_date)
    
    # Test with XML missing required elements
    documents = await parse_boe_xml(
        b"""
        <documento xmlns:boe="https://www.boe.es/xsd/boe">
            <metadatos>
                <titulo>Test Document</titulo>
            </metadatos>
        </documento>
        """,
        issue_date
    )
    assert len(documents) == 0


@pytest.mark.asyncio
async def test_fetch_boe_documents(mock_download_file, etl_test_data):
    """Test fetch_boe_documents function."""
    # Set up test data
    start_date = date(2025, 8, 1)
    end_date = date(2025, 8, 1)  # Just one day for testing
    
    # Mock extract_document_urls and parse_boe_xml
    with patch("src.etl.es_boe.extract_document_urls") as mock_extract:
        with patch("src.etl.es_boe.parse_boe_xml") as mock_parse:
            # Set up mocks
            mock_extract.return_value = ["https://www.boe.es/diario_boe/xml.php?id=BOE-A-2025-12345"]
            mock_parse.return_value = [etl_test_data["ES"]["expected_doc"]]
            
            # Call function
            documents = await fetch_boe_documents(start_date, end_date)
            
            # Check results
            assert len(documents) == 1
            assert documents[0]["id"] == etl_test_data["ES"]["expected_doc"]["id"]
            
            # Verify download_file was called
            mock_download_file.assert_called()
            
            # Verify extract_document_urls and parse_boe_xml were called
            mock_extract.assert_called()
            mock_parse.assert_called()
    
    # Test with download error
    mock_download_file.side_effect = DownloadError("Test error")
    
    # Should handle error and return empty list
    documents = await fetch_boe_documents(start_date, end_date)
    assert len(documents) == 0
    
    # Reset mock
    mock_download_file.side_effect = None
    
    # Test with extraction error
    with patch("src.etl.es_boe.extract_document_urls") as mock_extract:
        mock_extract.side_effect = ParsingError("Test error")
        
        # Should handle error and return empty list
        documents = await fetch_boe_documents(start_date, end_date)
        assert len(documents) == 0
    
    # Test with parsing error
    with patch("src.etl.es_boe.extract_document_urls") as mock_extract:
        with patch("src.etl.es_boe.parse_boe_xml") as mock_parse:
            mock_extract.return_value = ["https://www.boe.es/diario_boe/xml.php?id=BOE-A-2025-12345"]
            mock_parse.side_effect = ParsingError("Test error")
            
            # Should handle error and return empty list
            documents = await fetch_boe_documents(start_date, end_date)
            assert len(documents) == 0


@pytest.mark.asyncio
async def test_download_boe_document(mock_download_file):
    """Test download_boe_document function."""
    # Test document
    doc = {
        "id": "ES:BOE-A-2025-12345",
        "blob_url": "https://www.boe.es/boe/dias/2025/08/01/pdfs/BOE-A-2025-12345.pdf"
    }
    
    # Test with PDF URL
    content, format, parser_kwargs = await download_boe_document(doc)
    
    # Check results
    assert content == b"test content"
    assert format == DocumentFormat.PDF
    assert "max_pages" in parser_kwargs
    assert "summary_pages" in parser_kwargs
    
    # Test with XML URL
    doc["blob_url"] = "https://www.boe.es/diario_boe/xml.php?id=BOE-A-2025-12345"
    
    content, format, parser_kwargs = await download_boe_document(doc)
    
    # Check results
    assert content == b"test content"
    assert format == DocumentFormat.XML
    assert "text_xpath" in parser_kwargs
    assert "summary_xpath" in parser_kwargs
    
    # Test with download error
    mock_download_file.side_effect = DownloadError("Test error")
    
    with pytest.raises(DownloadError):
        await download_boe_document(doc)


@pytest.mark.asyncio
async def test_process_boe_documents(mock_document_processor, etl_test_data):
    """Test process_boe_documents function."""
    # Test documents
    documents = [etl_test_data["ES"]["expected_doc"]]
    
    # Set up mock processor
    mock_document_processor.process_batch.return_value = documents
    
    # Process documents
    processed_docs = await process_boe_documents(documents, mock_document_processor)
    
    # Check results
    assert len(processed_docs) == 1
    assert processed_docs[0]["id"] == documents[0]["id"]
    
    # Verify process_batch was called with correct parameters
    mock_document_processor.process_batch.assert_called_once()
    args, kwargs = mock_document_processor.process_batch.call_args
    assert args[0] == documents
    assert kwargs["concurrency"] == 5


@pytest.mark.asyncio
async def test_main():
    """Test main function."""
    # Mock run_pipeline
    with patch("src.etl.es_boe.run_pipeline") as mock_run_pipeline:
        # Set up mock to return test stats
        mock_run_pipeline.return_value = {
            "jurisdiction": "ES",
            "documents_found": 10,
            "documents_processed": 8,
            "documents_saved": 8,
            "errors": 0
        }
        
        # Call main
        await main()
        
        # Verify run_pipeline was called with correct parameters
        mock_run_pipeline.assert_called_once()
        args, kwargs = mock_run_pipeline.call_args
        assert args[0] == "ES"
        assert args[1] == fetch_boe_documents


@pytest.mark.asyncio
async def test_integration_fetch_and_process(mock_download_file, mock_embedding_strategy, etl_test_data):
    """Test integration of fetch and process functions."""
    # Set up test data
    start_date = date(2025, 8, 1)
    end_date = date(2025, 8, 1)
    
    # Mock extract_document_urls and parse_boe_xml
    with patch("src.etl.es_boe.extract_document_urls") as mock_extract:
        with patch("src.etl.es_boe.parse_boe_xml") as mock_parse:
            # Set up mocks
            mock_extract.return_value = ["https://www.boe.es/diario_boe/xml.php?id=BOE-A-2025-12345"]
            mock_parse.return_value = [etl_test_data["ES"]["expected_doc"]]
            
            # Fetch documents
            documents = await fetch_boe_documents(start_date, end_date)
            
            # Create processor
            processor = DocumentProcessor(embedding_strategy=mock_embedding_strategy)
            
            # Process documents
            processed_docs = await process_boe_documents(documents, processor)
            
            # Check results
            assert len(processed_docs) == 1
            assert processed_docs[0]["id"] == etl_test_data["ES"]["expected_doc"]["id"]


@pytest.mark.asyncio
async def test_error_handling(mock_download_file):
    """Test error handling in the pipeline."""
    # Set up test data
    start_date = date(2025, 8, 1)
    end_date = date(2025, 8, 1)
    
    # Test with network error
    mock_download_file.side_effect = DownloadError("Network error")
    
    # Should handle error and return empty list
    documents = await fetch_boe_documents(start_date, end_date)
    assert len(documents) == 0
    
    # Reset mock
    mock_download_file.side_effect = None
    
    # Test with extraction error
    with patch("src.etl.es_boe.extract_document_urls") as mock_extract:
        mock_extract.side_effect = ParsingError("Extraction error")
        
        # Should handle error and return empty list
        documents = await fetch_boe_documents(start_date, end_date)
        assert len(documents) == 0
    
    # Test with parsing error
    with patch("src.etl.es_boe.extract_document_urls") as mock_extract:
        with patch("src.etl.es_boe.parse_boe_xml") as mock_parse:
            mock_extract.return_value = ["https://www.boe.es/diario_boe/xml.php?id=BOE-A-2025-12345"]
            mock_parse.side_effect = ParsingError("Parsing error")
            
            # Should handle error and return empty list
            documents = await fetch_boe_documents(start_date, end_date)
            assert len(documents) == 0
    
    # Test with unexpected error
    with patch("src.etl.es_boe.extract_document_urls") as mock_extract:
        mock_extract.side_effect = Exception("Unexpected error")
        
        # Should handle error and return empty list
        documents = await fetch_boe_documents(start_date, end_date)
        assert len(documents) == 0