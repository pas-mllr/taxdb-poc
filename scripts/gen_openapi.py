#!/usr/bin/env python3
"""
OpenAPI specification generator for TaxDB-POC.

This script generates an OpenAPI specification for the TaxDB-POC API.
"""

import json
import os
from pathlib import Path

from fastapi.openapi.utils import get_openapi

from src.api import create_app


def generate_openapi_spec():
    """Generate OpenAPI specification."""
    # Create FastAPI app
    app = create_app()
    
    # Generate OpenAPI schema
    openapi_schema = get_openapi(
        title="TaxDB API",
        version="0.1.0",
        description="Tax Document Database API",
        routes=app.routes,
    )
    
    # Add additional info
    openapi_schema["info"]["contact"] = {
        "name": "TaxDB Team",
        "email": "taxdb-team@example.com",
    }
    
    openapi_schema["info"]["license"] = {
        "name": "Internal Use Only",
    }
    
    # Create openapi directory if it doesn't exist
    openapi_dir = Path(__file__).parent.parent / "openapi"
    openapi_dir.mkdir(exist_ok=True)
    
    # Write OpenAPI spec to YAML file
    openapi_path = openapi_dir / "taxdb.yaml"
    
    # Convert to YAML
    try:
        import yaml
        with open(openapi_path, "w") as f:
            yaml.dump(openapi_schema, f, sort_keys=False)
        print(f"OpenAPI specification written to {openapi_path}")
    except ImportError:
        # Fallback to JSON if PyYAML is not installed
        openapi_path = openapi_dir / "taxdb.json"
        with open(openapi_path, "w") as f:
            json.dump(openapi_schema, f, indent=2)
        print(f"OpenAPI specification written to {openapi_path} (JSON format)")


if __name__ == "__main__":
    generate_openapi_spec()