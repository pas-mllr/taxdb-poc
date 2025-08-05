"""
Configuration settings for the TaxDB-POC application.

This module loads environment variables and provides configuration settings
for the application.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Local mode flag
LOCAL_MODE = os.getenv("LOCAL_MODE", "true").lower() == "true"

# Azure / OpenAI settings
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "")
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY", "")

# Database settings
if LOCAL_MODE:
    PG_CONNSTR = "postgresql://taxdb:taxdb@localhost:5432/taxdb"
else:
    PG_CONNSTR = os.getenv("AZURE_PG_CONNSTR", "")

# Blob storage settings
if LOCAL_MODE:
    BLOB_CONNSTR = "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://localhost:10000/devstoreaccount1;"
else:
    BLOB_CONNSTR = os.getenv("AZURE_BLOB_CONNSTR", "")

# ETL settings
DOC_LOOKBACK_HOURS = int(os.getenv("DOC_LOOKBACK_HOURS", "48"))
CACHE_DIR = Path.home() / ".cache" / "taxdb"

# Ensure cache directory exists
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# API settings
CORS_ORIGINS = [
    "https://*.shell.com",
    "http://localhost:*",
]

# Security settings
ALLOWED_HOSTS = ["*"]  # In production, this should be restricted to specific domains

# Document settings
JURISDICTIONS = ["BE", "ES", "DE"]
SAS_TOKEN_EXPIRY_MINUTES = 15