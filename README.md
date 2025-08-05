# TaxDB-POC (Belgium | Spain | Germany)

## Project Overview

TaxDB-POC is a proof of concept system for fetching, processing, and searching tax-related documents from Belgium, Spain, and Germany. The system provides a unified API for accessing tax documents across these jurisdictions, with powerful search capabilities including text-based, vector similarity, and hybrid search approaches.

### Key Features

- **Automated Document Fetching**: Retrieves tax-related documents from official sources:
  - Belgian Moniteur (Belgisch Staatsblad)
  - Spanish BOE (Boletín Oficial del Estado)
  - German BGBl (Bundesgesetzblatt)
- **Intelligent Processing**: Filters for tax-relevant content and generates vector embeddings for semantic search
- **Advanced Search Capabilities**: Text-based, vector similarity, and hybrid search options
- **Document Storage**: PostgreSQL with pgvector for vector search capabilities
- **Blob Storage**: Azure Blob Storage for document files with lifecycle management
- **API Service**: FastAPI service with comprehensive search and document retrieval endpoints
- **Dual Deployment Options**: Local development mode with Docker Compose and cloud deployment with Azure

### Technologies Used

- **Backend**: Python 3.12
- **Database**: PostgreSQL with pgvector extension
- **Storage**: Azure Blob Storage (cloud) / Azurite (local)
- **Search**: Azure AI Search (cloud) / PostgreSQL (local)
- **API Framework**: FastAPI
- **Infrastructure as Code**: Azure Bicep
- **Containerization**: Docker and Docker Compose
- **Vector Embeddings**: Sentence Transformers
- **Document Processing**: PDF processing libraries (pdfminer-six, pdfplumber, pytesseract)

### Architecture Overview

The system follows a modern cloud-native architecture with the following components:

1. **ETL Pipelines**: Jurisdiction-specific pipelines fetch, filter, and process documents
2. **Storage Layer**: PostgreSQL database for document metadata and vectors, Blob storage for document files
3. **API Layer**: FastAPI service providing search and retrieval endpoints
4. **Infrastructure**: Azure resources for cloud deployment, Docker Compose for local development

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Belgian ETL    │     │  Spanish ETL    │     │  German ETL     │
│  (Moniteur)     │     │  (BOE)          │     │  (BGBl)         │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Document Processor                        │
│                                                                 │
│  - Content extraction                                           │
│  - Tax relevance filtering                                      │
│  - Vector embedding generation                                  │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 │                               │
                 ▼                               ▼
┌────────────────────────────┐     ┌────────────────────────────┐
│                            │     │                            │
│  PostgreSQL + pgvector     │     │  Blob Storage              │
│                            │     │                            │
└────────────────┬───────────┘     └────────────┬───────────────┘
                 │                               │
                 └───────────────┬───────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Service                           │
│                                                                 │
│  - Text search                                                  │
│  - Vector similarity search                                     │
│  - Hybrid search                                                │
│  - Document retrieval                                           │
└─────────────────────────────────────────────────────────────────┘
```

## Requirements

### System Requirements

- Linux, macOS, or Windows with WSL2
- 4GB RAM minimum (8GB recommended)
- 10GB free disk space
- Internet connection for document fetching

### Software Dependencies

- Python 3.12
- Docker and Docker Compose
- Make (for running Makefile commands)
- Git

### Cloud Requirements (for Azure deployment)

- Azure subscription
- Azure CLI installed and configured
- Bicep CLI installed
- Sufficient permissions to create resource groups and deploy resources

## Local Development Setup

### Clone and Initial Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/taxdb-poc.git
cd taxdb-poc

# Initialize the project (creates virtual environment and installs dependencies)
make init
```

### Environment Variables Configuration

Create a `.env` file based on the provided `.env.example`:

```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file with your preferred editor
# For local development, ensure LOCAL_MODE=true
```

Key environment variables:

```
# Development mode (true = local, false = cloud)
LOCAL_MODE=true

# Local development settings
LOCAL_PG_CONNSTR=postgresql://taxdb:taxdb@localhost:5432/taxdb
LOCAL_BLOB_CONNSTR=DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://localhost:10000/devstoreaccount1;

# ETL settings
DOC_LOOKBACK_HOURS=48

# API settings
SAS_TOKEN_EXPIRY_MINUTES=15
```

### Docker Compose Setup

Start the local services (PostgreSQL with pgvector and Azurite for blob storage):

```bash
# Start local services
make compose-up

# Verify services are running
docker ps
```

This will start:
- PostgreSQL with pgvector extension on port 5432
- Azurite (Azure Storage emulator) on port 10000

### Database Initialization

The database is automatically initialized when you start the services. The Docker Compose setup includes a SQL initialization script that creates the necessary tables and extensions.

If you need to manually initialize or reset the database:

```bash
# Load sample data (optional)
make load-sample
```

### Running the Application Locally

```bash
# Run the ETL process to fetch documents
make etl-run-once

# Start the API server
make serve-local

# Access the API at http://localhost:8000
# Swagger UI is available at http://localhost:8000/docs
```

## Cloud Deployment

### Azure Subscription Setup

```bash
# Login to Azure
az login

# Set your subscription (if you have multiple)
az account set --subscription "Your Subscription Name or ID"

# Create a resource group
export RG_NAME=taxdb-poc-rg
export LOCATION=westeurope
az group create --name $RG_NAME --location $LOCATION
```

### Deploying with Bicep Templates

```bash
# Set administrator password for PostgreSQL (store this securely)
export POSTGRES_ADMIN_PASSWORD="YourSecurePassword123!"

# Deploy the infrastructure
az deployment group create \
  --resource-group $RG_NAME \
  --template-file infra/main.bicep \
  --parameters administratorLoginPassword=$POSTGRES_ADMIN_PASSWORD

# After deployment completes, get the outputs
az deployment group show \
  --resource-group $RG_NAME \
  --name main \
  --query properties.outputs
```

The deployment creates the following Azure resources:
- PostgreSQL Flexible Server (Burstable tier)
- Storage Account with blob containers
- Azure AI Search service (Basic tier)
- Function App (Consumption plan)
- Key Vault for secrets
- Virtual Network with subnets
- Application Insights and Log Analytics

### Configuration and Secrets Management

After deployment, update your `.env` file with the cloud connection strings:

```bash
# Get connection strings and keys from deployment outputs
export PG_CONNSTR=$(az deployment group show --resource-group $RG_NAME --name main --query properties.outputs.PG_CONNSTR.value -o tsv)
export BLOB_CONNSTR=$(az deployment group show --resource-group $RG_NAME --name main --query properties.outputs.BLOB_CONNSTR.value -o tsv)
export SEARCH_ENDPOINT=$(az deployment group show --resource-group $RG_NAME --name main --query properties.outputs.SEARCH_ENDPOINT.value -o tsv)
export SEARCH_KEY=$(az search admin-key show --resource-group $RG_NAME --service-name $(az deployment group show --resource-group $RG_NAME --name main --query properties.outputs.SEARCH_SERVICE_NAME.value -o tsv) --query primaryKey -o tsv)

# Update .env file
cat > .env << EOF
LOCAL_MODE=false
AZURE_PG_CONNSTR=$PG_CONNSTR
AZURE_BLOB_CONNSTR=$BLOB_CONNSTR
AZURE_SEARCH_ENDPOINT=$SEARCH_ENDPOINT
AZURE_SEARCH_KEY=$SEARCH_KEY
DOC_LOOKBACK_HOURS=48
SAS_TOKEN_EXPIRY_MINUTES=15
EOF
```

### Monitoring and Management

```bash
# Check deployment costs projection
make cost-check

# Monitor resource usage
az monitor metrics list \
  --resource $(az deployment group show --resource-group $RG_NAME --name main --query properties.outputs.POSTGRES_SERVER_NAME.value -o tsv) \
  --resource-group $RG_NAME \
  --resource-type "Microsoft.DBforPostgreSQL/flexibleServers" \
  --metric "cpu_percent" \
  --interval PT1H

# Run cloud smoke test
make smoke-cloud
```

## ETL Pipelines

### Overview of the ETL Process

The ETL (Extract, Transform, Load) pipelines fetch tax-related documents from official sources, process them, and store them in the database. Each jurisdiction has its own ETL pipeline with specific logic for handling the source format and filtering for tax relevance.

Common ETL pipeline steps:
1. Fetch documents from the source
2. Filter for tax-related content
3. Extract text and metadata
4. Generate vector embeddings
5. Store in database and blob storage

### Running the Belgium ETL Pipeline

```bash
# Run the Belgium ETL pipeline
.venv/bin/python -m src.etl.be_moniteur

# With custom lookback hours
DOC_LOOKBACK_HOURS=72 .venv/bin/python -m src.etl.be_moniteur
```

The Belgium pipeline fetches documents from the Belgian Moniteur (Belgisch Staatsblad), filters for tax-related content using keywords in Dutch and French, and processes the documents for storage.

### Running the Spain ETL Pipeline

```bash
# Run the Spain ETL pipeline
.venv/bin/python -m src.etl.es_boe

# With custom lookback hours
DOC_LOOKBACK_HOURS=72 .venv/bin/python -m src.etl.es_boe
```

The Spain pipeline fetches documents from the Spanish BOE (Boletín Oficial del Estado), filters for tax-related content using Spanish tax keywords, and processes the documents for storage.

### Running the Germany ETL Pipeline

```bash
# Run the Germany ETL pipeline
.venv/bin/python -m src.etl.de_bgbl

# With custom lookback hours
DOC_LOOKBACK_HOURS=72 .venv/bin/python -m src.etl.de_bgbl
```

The Germany pipeline fetches documents from the German BGBl (Bundesgesetzblatt), filters for tax-related content using German tax keywords, and processes the documents for storage.

### Running All ETL Pipelines

```bash
# Run all ETL pipelines sequentially
make etl-run-once
```

### Monitoring ETL Processes

ETL processes log detailed information about their execution. You can monitor the logs to track progress and identify any issues:

```bash
# View logs for a specific ETL run
cat logs/be_moniteur_$(date +%Y%m%d).log

# Monitor ETL execution in real-time
tail -f logs/be_moniteur_$(date +%Y%m%d).log
```

## API Usage

### Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/healthz` | GET | Health check endpoint |
| `/search` | GET | Text-based search for documents |
| `/search/vector` | POST | Vector similarity search |
| `/search/hybrid` | POST | Hybrid search (text + vector) |
| `/doc/{id}` | GET | Get document by ID |
| `/doc/{id}/similar` | GET | Find similar documents |
| `/jurisdictions/{jurisdiction}/documents` | GET | List documents by jurisdiction |

### Example Requests and Responses

#### Text Search

```bash
# Search for tax documents
curl -X GET "http://localhost:8000/search?q=tax&jurisdiction=BE&page=1&page_size=10"
```

Response:
```json
{
  "documents": [
    {
      "id": "BE:20250101:123",
      "jurisdiction": "BE",
      "source_system": "moniteur",
      "document_type": "legal",
      "title": "Tax law amendment",
      "summary": "Amendment to income tax regulations",
      "issue_date": "2025-01-01",
      "effective_date": "2025-02-01",
      "language_orig": "nl",
      "blob_url": "http://localhost:10000/devstoreaccount1/parsed/BE/20250101_123.pdf",
      "created_at": "2025-08-01T12:34:56.789Z"
    }
  ],
  "total": 1
}
```

#### Vector Search

```bash
# Vector similarity search
curl -X POST "http://localhost:8000/search/vector" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "tax implications for cross-border transactions",
    "jurisdiction": "BE",
    "page": 1,
    "page_size": 10,
    "max_distance": 0.5
  }'
```

Response:
```json
{
  "documents": [
    {
      "document": {
        "id": "BE:20250101:123",
        "jurisdiction": "BE",
        "source_system": "moniteur",
        "document_type": "legal",
        "title": "Tax law amendment",
        "summary": "Amendment to income tax regulations",
        "issue_date": "2025-01-01",
        "effective_date": "2025-02-01",
        "language_orig": "nl",
        "blob_url": "http://localhost:10000/devstoreaccount1/parsed/BE/20250101_123.pdf",
        "created_at": "2025-08-01T12:34:56.789Z"
      },
      "score": 0.85
    }
  ],
  "total": 1
}
```

#### Get Document by ID

```bash
# Get document by ID
curl -X GET "http://localhost:8000/doc/BE:20250101:123"
```

Response:
```json
{
  "id": "BE:20250101:123",
  "jurisdiction": "BE",
  "source_system": "moniteur",
  "document_type": "legal",
  "title": "Tax law amendment",
  "summary": "Amendment to income tax regulations",
  "issue_date": "2025-01-01",
  "effective_date": "2025-02-01",
  "language_orig": "nl",
  "blob_url": "http://localhost:10000/devstoreaccount1/parsed/BE/20250101_123.pdf",
  "created_at": "2025-08-01T12:34:56.789Z"
}
```

### Rate Limiting Information

The API implements rate limiting to prevent abuse:

| Endpoint | Rate Limit |
|----------|------------|
| `/healthz` | 60 requests per minute per client IP |
| `/search` | 30 requests per minute per client IP |
| `/search/vector` | 20 requests per minute per client IP |
| `/search/hybrid` | 20 requests per minute per client IP |
| `/doc/{id}` | 60 requests per minute per client IP |
| `/doc/{id}/similar` | 30 requests per minute per client IP |
| `/jurisdictions/{jurisdiction}/documents` | 30 requests per minute per client IP |

### Error Handling

The API returns standard HTTP status codes:

- 200: Success
- 400: Bad Request (invalid parameters)
- 404: Not Found (document not found)
- 429: Too Many Requests (rate limit exceeded)
- 500: Internal Server Error

Error responses include a detail message:

```json
{
  "detail": "Document not found"
}
```

## Testing

### Running Tests Locally

```bash
# Run all tests
make test

# Run tests with coverage report
make test-cov

# Run tests with HTML coverage report
make test-cov-html
```

### Running Specific Test Types

```bash
# Run unit tests only
make test-unit

# Run integration tests only
make test-integration

# Run end-to-end tests only
make test-e2e
```

### Running Tests in Docker

```bash
# Run tests in Docker environment
make test-docker
```

### CI Pipeline Testing

The CI pipeline runs tests automatically on each push and pull request. The pipeline is configured in `.github/workflows/ci.yml` and includes:

- Linting and code quality checks
- Unit tests
- Integration tests
- End-to-end tests
- Coverage reporting

To view the CI pipeline status:

```bash
# View CI pipeline status
open https://github.com/yourusername/taxdb-poc/actions
```

### Test Coverage

The project aims for high test coverage. You can check the current coverage with:

```bash
# Generate coverage report
make test-cov

# View HTML coverage report
make test-cov-html
open htmlcov/index.html
```

## Troubleshooting

### Common Issues and Solutions

#### Docker Compose Issues

**Issue**: Services fail to start or are not accessible.

**Solution**:
```bash
# Stop and remove containers
make compose-down

# Check for port conflicts
netstat -tuln | grep -E '5432|10000'

# Start services again
make compose-up
```

#### ETL Pipeline Issues

**Issue**: ETL pipeline fails to fetch documents.

**Solution**:
```bash
# Check internet connectivity
ping www.ejustice.just.fgov.be

# Check logs for specific errors
cat logs/be_moniteur_$(date +%Y%m%d).log

# Run with debug logging
DEBUG=1 .venv/bin/python -m src.etl.be_moniteur
```

#### API Issues

**Issue**: API returns 500 errors.

**Solution**:
```bash
# Check database connection
psql -h localhost -U taxdb -d taxdb -c "SELECT 1"

# Check logs
cat logs/api_$(date +%Y%m%d).log

# Restart API with debug mode
DEBUG=1 make serve-local
```

### Logging and Debugging

The system uses Python's logging module for comprehensive logging:

```bash
# View API logs
cat logs/api_$(date +%Y%m%d).log

# View ETL logs
cat logs/be_moniteur_$(date +%Y%m%d).log
cat logs/es_boe_$(date +%Y%m%d).log
cat logs/de_bgbl_$(date +%Y%m%d).log

# Enable debug logging
DEBUG=1 make serve-local
```

In cloud mode, logs are available in Azure Application Insights:

```bash
# Get Application Insights ID
APPINSIGHTS_ID=$(az resource list --resource-group $RG_NAME --resource-type "microsoft.insights/components" --query "[0].id" -o tsv)

# View logs
az monitor app-insights query --app $APPINSIGHTS_ID --analytics-query "traces | where timestamp > ago(1h) | order by timestamp desc"
```

### Support Resources

- Project documentation: `docs/` directory
- API documentation: Swagger UI at `http://localhost:8000/docs`
- Issue tracker: GitHub Issues
- Contact: taxdb-team@example.com

## License

Internal use only.

## Contributors

- TaxDB Team