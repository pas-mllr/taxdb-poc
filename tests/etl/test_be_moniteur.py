"""
Tests for Belgian Moniteur ETL pipeline.

This module tests the ETL pipeline for the Belgian Moniteur,
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

from src.etl.be_moniteur import (
    is_tax_related,
    preprocess_text_for_embedding,
    fetch_moniteur_documents,
    parse_moniteur_xml,
    download_moniteur_document,
    process_moniteur_documents,
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
    # Test with tax-related text (Dutch)
    assert is_tax_related("Dit document gaat over belasting en fiscale zaken")
    assert is_tax_related("BTW-tarief voor diensten")
    assert is_tax_related("Vennootschapsbelasting in België")
    
    # Test with tax-related text (French)
    assert is_tax_related("Ce document concerne l'impôt et les affaires fiscales")
    assert is_tax_related("Taux de TVA pour les services")
    assert is_tax_related("Impôt des sociétés en Belgique")
    
    # Test with non-tax-related text
    assert not is_tax_related("Dit document gaat over het weer")
    assert not is_tax_related("Ce document concerne la météo")


def test_preprocess_text_for_embedding():
    """Test preprocess_text_for_embedding function."""
    # Test with normal text
    text = "Dit is een normale tekst."
    processed = preprocess_text_for_embedding(text)
    assert processed == text
    
    # Test with excessive whitespace
    text = "  Dit   heeft \n\n te veel \t witruimte.  "
    processed = preprocess_text_for_embedding(text)
    assert processed == "Dit heeft te veel witruimte."
    
    # Test with boilerplate text
    text = "Belgisch Staatsblad - Moniteur Belge\nDit is een tekst met boilerplate."
    processed = preprocess_text_for_embedding(text)
    assert "Belgisch Staatsblad - Moniteur Belge" not in processed
    assert "Dit is een tekst met boilerplate." in processed
    
    # Test with page numbers
    text = "Dit is pagina 10 / 20 van het document."
    processed = preprocess_text_for_embedding(text)
    assert "10 / 20" not in processed
    
    # Test with URL
    text = "Bezoek www.ejustice.just.fgov.be voor meer informatie."
    processed = preprocess_text_for_embedding(text)
    assert "www.ejustice.just.fgov.be" not in processed
    
    # Test with empty text
    assert preprocess_text_for_embedding("") == ""
    assert preprocess_text_for_embedding(None) == ""


@pytest.mark.asyncio
async def test_fetch_moniteur_documents(mock_download_file, etl_test_data):
    """Test fetch_moniteur_documents function."""
    # Set up test data
    start_date = date(2025, 8, 1)
    end_date = date(2025, 8, 1)  # Just one day for testing
    
    # Mock parse_moniteur_xml
    with patch("src.etl.be_moniteur.parse_moniteur_xml") as mock_parse:
        # Set up mock to return test documents
        mock_parse.return_value = [etl_test_data["BE"]["expected_doc"]]
        
        # Call function
        documents = await fetch_moniteur_documents(start_date, end_date, languages=["nl"])
        
        # Check results
        assert len(documents) == 1
        assert documents[0]["id"] == etl_test_data["BE"]["expected_doc"]["id"]
        
        # Verify download_file was called
        mock_download_file.assert_called()
        
        # Verify parse_moniteur_xml was called
        mock_parse.assert_called()
    
    # Test with download error
    mock_download_file.side_effect = DownloadError("Test error")
    
    # Should handle error and return empty list
    documents = await fetch_moniteur_documents(start_date, end_date, languages=["nl"])
    assert len(documents) == 0
    
    # Reset mock
    mock_download_file.side_effect = None
    
    # Test with parsing error
    with patch("src.etl.be_moniteur.parse_moniteur_xml") as mock_parse:
        mock_parse.side_effect = ParsingError("Test error")
        
        # Should handle error and return empty list
        documents = await fetch_moniteur_documents(start_date, end_date, languages=["nl"])
        assert len(documents) == 0


@pytest.mark.asyncio
async def test_parse_moniteur_xml(etl_test_data):
    """Test parse_moniteur_xml function."""
    # Test with valid XML
    xml_content = etl_test_data["BE"]["xml_content"]
    issue_date = date(2025, 8, 1)
    
    documents = await parse_moniteur_xml(xml_content, "nl", issue_date)
    
    # Check results
    assert len(documents) == 1
    assert documents[0]["id"] == "BE:2025/12345"
    assert documents[0]["jurisdiction"] == "BE"
    assert documents[0]["title"] == "Test Tax Law"
    assert documents[0]["issue_date"] == date(2025, 8, 1)
    assert documents[0]["language_orig"] == "nl"
    assert "checksum" in documents[0]
    
    # Test with invalid XML
    with pytest.raises(ParsingError):
        await parse_moniteur_xml(b"<invalid XML", "nl", issue_date)
    
    # Test with empty XML
    documents = await parse_moniteur_xml(b"<root></root>", "nl", issue_date)
    assert len(documents) == 0
    
    # Test with XML missing required elements
    documents = await parse_moniteur_xml(
        b"""
        <mb:doc xmlns:mb="http://www.ejustice.just.fgov.be/moniteur">
            <mb:title>Test Document</mb:title>
        </mb:doc>
        """,
        "nl",
        issue_date
    )
    assert len(documents) == 0


@pytest.mark.asyncio
async def test_download_moniteur_document(mock_download_file):
    """Test download_moniteur_document function."""
    # Test document
    doc = {
        "id": "BE:2025/12345",
        "blob_url": "https://www.ejustice.just.fgov.be/eli/20250801/MONITOR/nl/pdf"
    }
    
    # Test with PDF URL
    content, format, parser_kwargs = await download_moniteur_document(doc)
    
    # Check results
    assert content == b"test content"
    assert format == DocumentFormat.PDF
    assert "max_pages" in parser_kwargs
    assert "summary_pages" in parser_kwargs
    
    # Test with XML URL
    doc["blob_url"] = "https://www.ejustice.just.fgov.be/eli/20250801/MONITOR/nl/xml"
    
    content, format, parser_kwargs = await download_moniteur_document(doc)
    
    # Check results
    assert content == b"test content"
    assert format == DocumentFormat.XML
    assert "text_xpath" in parser_kwargs
    assert "summary_xpath" in parser_kwargs
    
    # Test with download error
    mock_download_file.side_effect = DownloadError("Test error")
    
    with pytest.raises(DownloadError):
        await download_moniteur_document(doc)


@pytest.mark.asyncio
async def test_process_moniteur_documents(mock_document_processor, etl_test_data):
    """Test process_moniteur_documents function."""
    # Test documents
    documents = [etl_test_data["BE"]["expected_doc"]]
    
    # Set up mock processor
    mock_document_processor.process_batch.return_value = documents
    
    # Process documents
    processed_docs = await process_moniteur_documents(documents, mock_document_processor)
    
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
    with patch("src.etl.be_moniteur.run_pipeline") as mock_run_pipeline:
        # Set up mock to return test stats
        mock_run_pipeline.return_value = {
            "jurisdiction": "BE",
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
        assert args[0] == "BE"
        assert args[1] == fetch_moniteur_documents
        assert "metadata" in kwargs
        assert kwargs["metadata"]["source"] == "Moniteur Belge"


@pytest.mark.asyncio
async def test_integration_fetch_and_process(mock_download_file, mock_embedding_strategy, etl_test_data):
    """Test integration of fetch and process functions."""
    # Set up test data
    start_date = date(2025, 8, 1)
    end_date = date(2025, 8, 1)
    
    # Mock parse_moniteur_xml to return test documents
    with patch("src.etl.be_moniteur.parse_moniteur_xml") as mock_parse:
        mock_parse.return_value = [etl_test_data["BE"]["expected_doc"]]
        
        # Fetch documents
        documents = await fetch_moniteur_documents(start_date, end_date, languages=["nl"])
        
        # Create processor
        processor = DocumentProcessor(embedding_strategy=mock_embedding_strategy)
        
        # Process documents
        processed_docs = await process_moniteur_documents(documents, processor)
        
        # Check results
        assert len(processed_docs) == 1
        assert processed_docs[0]["id"] == etl_test_data["BE"]["expected_doc"]["id"]


@pytest.mark.asyncio
async def test_error_handling(mock_download_file):
    """Test error handling in the pipeline."""
    # Set up test data
    start_date = date(2025, 8, 1)
    end_date = date(2025, 8, 1)
    
    # Test with network error
    mock_download_file.side_effect = DownloadError("Network error")
    
    # Should handle error and return empty list
    documents = await fetch_moniteur_documents(start_date, end_date, languages=["nl"])
    assert len(documents) == 0
    
    # Reset mock
    mock_download_file.side_effect = None
    
    # Test with XML parsing error
    with patch("src.etl.be_moniteur.parse_moniteur_xml") as mock_parse:
        mock_parse.side_effect = ParsingError("XML parsing error")
        
        # Should handle error and return empty list
        documents = await fetch_moniteur_documents(start_date, end_date, languages=["nl"])
        assert len(documents) == 0
    
    # Test with unexpected error
    with patch("src.etl.be_moniteur.parse_moniteur_xml") as mock_parse:
        mock_parse.side_effect = Exception("Unexpected error")
        
        # Should handle error and return empty list
        documents = await fetch_moniteur_documents(start_date, end_date, languages=["nl"])
        assert len(documents) == 0


@pytest.mark.asyncio
async def test_namespaces_handling():
    """Test handling of XML namespaces."""
    # Test XML with namespaces
    xml_content = b"""
    <mb:root xmlns:mb="http://www.ejustice.just.fgov.be/moniteur">
        <mb:doc>
            <mb:eli>2025/12345</mb:eli>
            <mb:title>Test Tax Law</mb:title>
            <mb:pubdate>2025-08-01</mb:pubdate>
            <mb:text>This is a test tax law document for Belgium</mb:text>
            <mb:summary>Test summary</mb:summary>
        </mb:doc>
    </mb:root>
    """
    
    issue_date = date(2025, 8, 1)
    
    documents = await parse_moniteur_xml(xml_content, "nl", issue_date)
    
    # Check results
    assert len(documents) == 1
    assert documents[0]["id"] == "BE:2025/12345"
    
    # Test XML without namespaces
    xml_content = b"""
    <root>
        <doc>
            <eli>2025/12345</eli>
            <title>Test Tax Law</title>
            <pubdate>2025-08-01</pubdate>
            <text>This is a test tax law document for Belgium</text>
            <summary>Test summary</summary>
        </doc>
    </root>
    """
    
    documents = await parse_moniteur_xml(xml_content, "nl", issue_date)
    
    # Check results
    assert len(documents) == 1
    assert documents[0]["id"] == "BE:2025/12345"