#!/bin/bash
# Sample data loader for TaxDB-POC
# This script loads sample data into the TaxDB-POC system

set -e

# Ensure we're in the project root directory
cd "$(dirname "$0")/.."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Please run 'make init' first."
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Check if Docker services are running
if ! docker ps | grep -q "taxdb-pg"; then
    echo "PostgreSQL container not running. Starting Docker services..."
    make compose-up
    
    # Wait for PostgreSQL to be ready
    echo "Waiting for PostgreSQL to be ready..."
    sleep 5
fi

# Set environment variables
export LOCAL_MODE=true
export DOC_LOOKBACK_HOURS=720  # Look back 30 days for sample data

# Create sample data directory if it doesn't exist
mkdir -p data/samples

# Download sample documents if they don't exist
if [ ! -f "data/samples/be_sample.xml" ]; then
    echo "Downloading Belgian sample document..."
    curl -s "https://www.ejustice.just.fgov.be/eli/20250101/MONITOR/nl/xml" -o data/samples/be_sample.xml
fi

if [ ! -f "data/samples/es_sample.xml" ]; then
    echo "Downloading Spanish sample document..."
    curl -s "https://www.boe.es/diario_boe/xml.php?id=BOE-B-20250101" -o data/samples/es_sample.xml
fi

# Run ETL processes
echo "Running ETL processes..."
python -m src.etl.be_moniteur
python -m src.etl.es_boe
python -m src.etl.de_bgbl

echo "Sample data loaded successfully."