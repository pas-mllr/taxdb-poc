"""
Tests for German BGBL ETL pipeline.

This module tests the ETL pipeline for the German Federal Law Gazette (Bundesgesetzblatt),
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

from src.etl.de_bgbl import (
    is_tax_related,
    preprocess_text_for_embedding,
    fetch_bgbl_documents,
    process_bgbl_index,
    extract_text_from_pdf,
    _extract_with_pdfplumber,
    _extract_with_pdfminer,
    download_bgbl_document,
    process_bgbl_documents,
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
    assert is_tax_related("Dieses Dokument behandelt Steuer und steuerliche Angelegenheiten")
    assert is_tax_related("Umsatzsteuer für Dienstleistungen")
    assert is_tax_related("Körperschaftsteuer in Deutschland")
    
    # Test with tax-related category
    assert is_tax_related("Normaler Text", category="steuer")
    assert is_tax_related("Normaler Text", category="abgabe")
    assert is_tax_related("Normaler Text", category="finanz")
    
    # Test with non-tax-related text and category
    assert not is_tax_related("Dieses Dokument behandelt das Wetter", category="wetter")


def test_preprocess_text_for_embedding():
    """Test preprocess_text_for_embedding function."""
    # Test with normal text
    text = "Dies ist ein normaler Text."
    processed = preprocess_text_for_embedding(text)
    assert processed == text
    
    # Test with excessive whitespace
    text = "  Dieser   hat \n\n zu viel \t Leerzeichen.  "
    processed = preprocess_text_for_embedding(text)
    assert processed == "Dieser hat zu viel Leerzeichen."
    
    # Test with boilerplate text
    text = "Bundesgesetzblatt\nDies ist ein Text mit Boilerplate."
    processed = preprocess_text_for_embedding(text)
    assert "Bundesgesetzblatt" not in processed
    assert "Dies ist ein Text mit Boilerplate." in processed
    
    # Test with page numbers
    text = "Seite 10 von 20"
    processed = preprocess_text_for_embedding(text)
    assert "Seite 10 von 20" not in processed
    
    # Test with URLs
    text = "Besuchen Sie www.bgbl.de oder www.bundesanzeiger.de für mehr Informationen."
    processed = preprocess_text_for_embedding(text)
    assert "www.bgbl.de" not in processed
    assert "www.bundesanzeiger.de" not in processed
    
    # Test with German abbreviations
    text = "Abs. 1 des Art. 123 Nr. 4 S. 5 v. 01.01.2025 z. B. wichtig"
    processed = preprocess_text_for_embedding(text)
    assert "Absatz" in processed
    assert "Artikel" in processed
    assert "Nummer" in processed
    assert "Seite" in processed
    assert "vom" in processed
    assert "zum Beispiel" in processed
    
    # Test with empty text
    assert preprocess_text_for_embedding("") == ""


@pytest.mark.asyncio
async def test_extract_text_from_pdf():
    """Test extract_text_from_pdf function."""
    # Mock _extract_with_pdfplumber and _extract_with_pdfminer
    with patch("src.etl.de_bgbl._extract_with_pdfplumber") as mock_pdfplumber:
        with patch("src.etl.de_bgbl._extract_with_pdfminer") as mock_pdfminer:
            # Set up mocks
            mock_pdfplumber.return_value = "Text extracted with pdfplumber"
            mock_pdfminer.return_value = "Text extracted with pdfminer"
            
            # Test with successful pdfplumber extraction
            text = await extract_text_from_pdf(b"%PDF-1.5\nTest content")
            assert text == "Text extracted with pdfplumber"
            
            # Test with empty pdfplumber result, falling back to pdfminer
            mock_pdfplumber.return_value = ""
            text = await extract_text_from_pdf(b"%PDF-1.5\nTest content")
            assert text == "Text extracted with pdfminer"
            
            # Test with pdfplumber error
            mock_pdfplumber.side_effect = Exception("Test error")
            mock_pdfminer.return_value = "Fallback text"
            
            with pytest.raises(ParsingError):
                await extract_text_from_pdf(b"%PDF-1.5\nTest content")


@pytest.mark.asyncio
async def test_extract_with_pdfplumber():
    """Test _extract_with_pdfplumber function."""
    # Mock pdfplumber
    with patch("pdfplumber.open") as mock_open:
        # Set up mock
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page text"
        mock_page.extract_tables.return_value = [[["Table cell"]]]
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__.return_value = mock_pdf
        mock_open.return_value = mock_pdf
        
        # Test extraction
        text = await _extract_with_pdfplumber(b"%PDF-1.5\nTest content")
        assert "Page text" in text
        
        # Test with empty page text, falling back to tables
        mock_page.extract_text.return_value = ""
        text = await _extract_with_pdfplumber(b"%PDF-1.5\nTest content")
        assert "Table cell" in text


@pytest.mark.asyncio
async def test_extract_with_pdfminer():
    """Test _extract_with_pdfminer function."""
    # Mock extract_text
    with patch("src.etl.de_bgbl.extract_text") as mock_extract:
        mock_extract.return_value = "Text extracted with pdfminer"
        
        # Test extraction
        text = await _extract_with_pdfminer(b"%PDF-1.5\nTest content")
        assert text == "Text extracted with pdfminer"


@pytest.mark.asyncio
async def test_process_bgbl_index(mock_download_file, etl_test_data):
    """Test process_bgbl_index function."""
    # Test data
    index_data = json.loads(etl_test_data["DE"]["json_content"])
    issue_date = date(2025, 8, 1)
    
    # Mock extract_text_from_pdf
    with patch("src.etl.de_bgbl.extract_text_from_pdf") as mock_extract:
        mock_extract.return_value = "Extracted text from PDF"
        
        # Process index
        documents = await process_bgbl_index(index_data, issue_date)
        
        # Check results
        assert len(documents) == 1
        assert documents[0]["id"] == "DE:BGBL-2025-12345"
        assert documents[0]["jurisdiction"] == "DE"
        assert documents[0]["title"] == "Test Tax Decree"
        assert documents[0]["issue_date"] == date(2025, 8, 1)
        assert documents[0]["language_orig"] == "de"
        assert documents[0]["category"] == "Steuer"
        assert "checksum" in documents[0]
        
        # Verify download_file was called
        mock_download_file.assert_called_once()
        
        # Verify extract_text_from_pdf was called
        mock_extract.assert_called_once()
    
    # Test with download error
    mock_download_file.side_effect = DownloadError("Test error")
    
    # Should skip the document with error
    documents = await process_bgbl_index(index_data, issue_date)
    assert len(documents) == 0
    
    # Reset mock
    mock_download_file.side_effect = None
    
    # Test with extraction error
    with patch("src.etl.de_bgbl.extract_text_from_pdf") as mock_extract:
        mock_extract.side_effect = ParsingError("Test error")
        
        # Should skip the document with error
        documents = await process_bgbl_index(index_data, issue_date)
        assert len(documents) == 0
    
    # Test with invalid index data
    invalid_index = {"items": [{"id": "BGBL-2025-12345"}]}  # Missing required fields
    documents = await process_bgbl_index(invalid_index, issue_date)
    assert len(documents) == 0
    
    # Test with unexpected error
    with patch("src.etl.de_bgbl.extract_text_from_pdf") as mock_extract:
        mock_extract.side_effect = Exception("Unexpected error")
        
        # Should handle error and continue
        documents = await process_bgbl_index(index_data, issue_date)
        assert len(documents) == 0


@pytest.mark.asyncio
async def test_fetch_bgbl_documents(mock_download_file, etl_test_data):
    """Test fetch_bgbl_documents function."""
    # Set up test data
    start_date = date(2025, 8, 1)
    end_date = date(2025, 8, 1)  # Just one day for testing
    
    # Mock process_bgbl_index
    with patch("src.etl.de_bgbl.process_bgbl_index") as mock_process:
        # Set up mock to return test documents
        mock_process.return_value = [etl_test_data["DE"]["expected_doc"]]
        
        # Call function
        documents = await fetch_bgbl_documents(start_date, end_date)
        
        # Check results
        assert len(documents) == 1
        assert documents[0]["id"] == etl_test_data["DE"]["expected_doc"]["id"]
        
        # Verify download_file was called
        mock_download_file.assert_called()
        
        # Verify process_bgbl_index was called
        mock_process.assert_called()
    
    # Test with download error
    mock_download_file.side_effect = DownloadError("Test error")
    
    # Should handle error and return empty list
    documents = await fetch_bgbl_documents(start_date, end_date)
    assert len(documents) == 0
    
    # Reset mock
    mock_download_file.side_effect = None
    
    # Test with processing error
    with patch("src.etl.de_bgbl.process_bgbl_index") as mock_process:
        mock_process.side_effect = ParsingError("Test error")
        
        # Should handle error and return empty list
        documents = await fetch_bgbl_documents(start_date, end_date)
        assert len(documents) == 0
    
    # Test with unexpected error
    with patch("src.etl.de_bgbl.process_bgbl_index") as mock_process:
        mock_process.side_effect = Exception("Unexpected error")
        
        # Should handle error and return empty list
        documents = await fetch_bgbl_documents(start_date, end_date)
        assert len(documents) == 0


@pytest.mark.asyncio
async def test_download_bgbl_document(mock_download_file):
    """Test download_bgbl_document function."""
    # Test document
    doc = {
        "id": "DE:BGBL-2025-12345",
        "blob_url": "https://www.bgbl.de/xaver/bgbl/start.xav?startbk=Bundesanzeiger_BGBl&jumpTo=bgbl125s0001.pdf"
    }
    
    # Test with PDF URL
    content, format, parser_kwargs = await download_bgbl_document(doc)
    
    # Check results
    assert content == b"test content"
    assert format == DocumentFormat.PDF
    assert "max_pages" in parser_kwargs
    assert "summary_pages" in parser_kwargs
    
    # Test with non-PDF URL
    doc["blob_url"] = "https://www.bgbl.de/xaver/bgbl/start.xav?startbk=Bundesanzeiger_BGBl&jumpTo=bgbl125s0001.txt"
    
    content, format, parser_kwargs = await download_bgbl_document(doc)
    
    # Check results
    assert content == b"test content"
    assert format == DocumentFormat.TEXT
    assert parser_kwargs == {}
    
    # Test with download error
    mock_download_file.side_effect = DownloadError("Test error")
    
    with pytest.raises(DownloadError):
        await download_bgbl_document(doc)


@pytest.mark.asyncio
async def test_process_bgbl_documents(mock_document_processor, etl_test_data):
    """Test process_bgbl_documents function."""
    # Test documents
    documents = [etl_test_data["DE"]["expected_doc"]]
    
    # Set up mock processor
    mock_document_processor.process_batch.return_value = documents
    
    # Process documents
    processed_docs = await process_bgbl_documents(documents, mock_document_processor)
    
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
    with patch("src.etl.de_bgbl.run_pipeline") as mock_run_pipeline:
        # Set up mock to return test stats
        mock_run_pipeline.return_value = {
            "jurisdiction": "DE",
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
        assert args[0] == "DE"
        assert args[1] == fetch_bgbl_documents


@pytest.mark.asyncio
async def test_integration_fetch_and_process(mock_download_file, mock_embedding_strategy, etl_test_data):
    """Test integration of fetch and process functions."""
    # Set up test data
    start_date = date(2025, 8, 1)
    end_date = date(2025, 8, 1)
    
    # Mock json.loads to return test data
    with patch("json.loads") as mock_json_loads:
        # Set up mock to return test index data
        mock_json_loads.return_value = json.loads(etl_test_data["DE"]["json_content"])
        
        # Mock extract_text_from_pdf
        with patch("src.etl.de_bgbl.extract_text_from_pdf") as mock_extract:
            mock_extract.return_value = "Extracted text from PDF"
            
            # Fetch documents
            documents = await fetch_bgbl_documents(start_date, end_date)
            
            # Create processor
            processor = DocumentProcessor(embedding_strategy=mock_embedding_strategy)
            
            # Process documents
            processed_docs = await process_bgbl_documents(documents, processor)
            
            # Check results
            assert len(processed_docs) == 1
            assert processed_docs[0]["id"] == "DE:BGBL-2025-12345"


@pytest.mark.asyncio
async def test_error_handling(mock_download_file):
    """Test error handling in the pipeline."""
    # Set up test data
    start_date = date(2025, 8, 1)
    end_date = date(2025, 8, 1)
    
    # Test with network error
    mock_download_file.side_effect = DownloadError("Network error")
    
    # Should handle error and return empty list
    documents = await fetch_bgbl_documents(start_date, end_date)
    assert len(documents) == 0
    
    # Reset mock
    mock_download_file.side_effect = None
    
    # Test with JSON parsing error
    with patch("json.loads") as mock_json_loads:
        mock_json_loads.side_effect = json.JSONDecodeError("JSON error", "", 0)
        
        # Should handle error and return empty list
        documents = await fetch_bgbl_documents(start_date, end_date)
        assert len(documents) == 0
    
    # Test with processing error
    with patch("src.etl.de_bgbl.process_bgbl_index") as mock_process:
        mock_process.side_effect = ParsingError("Processing error")
        
        # Should handle error and return empty list
        documents = await fetch_bgbl_documents(start_date, end_date)
        assert len(documents) == 0
    
    # Test with unexpected error
    with patch("src.etl.de_bgbl.process_bgbl_index") as mock_process:
        mock_process.side_effect = Exception("Unexpected error")
        
        # Should handle error and return empty list
        documents = await fetch_bgbl_documents(start_date, end_date)
        assert len(documents) == 0