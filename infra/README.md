# TaxDB-POC Infrastructure

This directory contains the infrastructure as code (IaC) for the TaxDB Proof of Concept system using Azure Bicep. The infrastructure is designed to be cost-optimized, keeping cloud spend under €5/month as specified in the requirements.

## Resources Deployed

The `main.bicep` template deploys the following Azure resources:

### Storage Account
- **Purpose**: Stores raw and parsed document files
- **SKU**: Standard_LRS (Locally Redundant Storage) - most cost-effective redundancy option
- **Containers**:
  - `raw` - For storing raw document files
  - `parsed` - For storing parsed document files
- **Cost Optimization**:
  - Enhanced lifecycle management:
    - Raw files: Move to cool tier after 30 days, delete after 90 days
    - Parsed files: Move to cool tier after 15 days, delete after 60 days
  - Disabled versioning and change feed to reduce storage costs
  - Secured with service endpoints for network isolation

### PostgreSQL Flexible Server
- **Purpose**: Database with vector search capabilities
- **SKU**: Standard_B1ms (Burstable tier) - cost-effective option with good performance
- **Configuration**:
  - PostgreSQL 15 with pgvector extension
  - 32GB storage (minimum allowed)
  - 7-day backup retention (minimum for production workloads)
  - Disabled auto-grow, high availability, and geo-redundant backups to control costs
  - Optimized maintenance window during off-hours
- **Database**: `taxdb`

### Azure AI Search
- **Purpose**: Provides full-text search capabilities
- **SKU**: Basic tier - most cost-effective option with full functionality
- **Configuration**:
  - Minimal setup: 1 replica × 1 partition
  - Disabled semantic search to reduce costs
  - Secured with service endpoints

### Function App
- **Purpose**: Hosts the API for the application
- **Plan**: Consumption (Y1) - pay only for execution time
- **Runtime**: Python 3.12
- **Configuration**:
  - Integrated with Application Insights for monitoring
  - Connected to VNet for security
  - Uses Key Vault references for secrets
  - HTTPS-only access
  - TLS 1.2 enforcement

### Key Vault
- **Purpose**: Securely stores all connection strings and secrets
- **SKU**: Standard
- **Secrets**:
  - PostgreSQL connection string
  - Blob storage connection string
  - Azure AI Search endpoint and key
- **Security**:
  - Soft-delete enabled with 7-day retention
  - Network isolation with service endpoints
  - Managed identity-based access

### Networking
- **Virtual Network**: Isolates resources and enables secure communication
- **Subnets**:
  - Default subnet with service endpoints for Storage, Key Vault, SQL, and Search
  - Function subnet with delegation for the Function App
- **Security**:
  - Network ACLs to restrict access
  - Service endpoints for private communication

### Monitoring
- **Application Insights**: Tracks function app performance and errors
- **Log Analytics Workspace**: Centralizes logs with cost-controlled retention
- **Budget Alert**: Monitors spending with alerts at 80% and 100% of budget

## Cost Optimization Strategies

The infrastructure is designed to minimize costs while maintaining functionality:

1. **Right-sized Resources**:
   - Using minimum SKUs and capacities for all services
   - Burstable tier for PostgreSQL to handle varying workloads efficiently

2. **Pay-as-you-go Options**:
   - Consumption plan for Function App (pay only for execution time)
   - PerGB2018 pricing for Log Analytics (pay only for data ingested)

3. **Storage Optimization**:
   - Tiered storage lifecycle management
   - Minimum backup retention periods
   - Disabled optional features (versioning, change feed)

4. **Monitoring and Control**:
   - Budget alerts to prevent overspending
   - Log Analytics daily cap to control ingestion costs
   - Application Insights for targeted troubleshooting

5. **Resource Tagging**:
   - All resources tagged for cost tracking and allocation

The estimated monthly cost should be ≤ €5 as per requirements.

## Deployment Instructions

### Prerequisites

- Azure CLI (version 2.45.0 or later)
- Azure subscription with Owner or Contributor role
- Bash shell environment (or Azure Cloud Shell)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### Step 2: Set Environment Variables

```bash
# Set your resource group name and location
export RG_NAME="taxdb-poc-rg"
export LOCATION="westeurope"  # Choose a region close to your users

# Set PostgreSQL admin password (must meet complexity requirements)
export PG_ADMIN_PASSWORD="$(openssl rand -base64 16)"
echo "Generated PostgreSQL password: $PG_ADMIN_PASSWORD"
echo "Please save this password securely!"
```

### Step 3: Create Resource Group

```bash
# Create resource group if it doesn't exist
az group create --name $RG_NAME --location $LOCATION
```

### Step 4: Deploy the Bicep Template

```bash
# Deploy the Bicep template
az deployment group create \
  --resource-group $RG_NAME \
  --template-file infra/main.bicep \
  --parameters rgName=$RG_NAME \
  --parameters administratorLoginPassword=$PG_ADMIN_PASSWORD
```

### Step 5: Verify Deployment

```bash
# Get deployment outputs
az deployment group show \
  --resource-group $RG_NAME \
  --name main \
  --query properties.outputs
```

### Step 6: Save Connection Information

```bash
# Extract and save connection strings
FUNCTION_APP_URL=$(az deployment group show --resource-group $RG_NAME --name main --query properties.outputs.FUNCTION_APP_URL.value -o tsv)
SEARCH_ENDPOINT=$(az deployment group show --resource-group $RG_NAME --name main --query properties.outputs.SEARCH_ENDPOINT.value -o tsv)
STORAGE_ACCOUNT=$(az deployment group show --resource-group $RG_NAME --name main --query properties.outputs.STORAGE_ACCOUNT_NAME.value -o tsv)
POSTGRES_SERVER=$(az deployment group show --resource-group $RG_NAME --name main --query properties.outputs.POSTGRES_SERVER_NAME.value -o tsv)

# Save to a local .env file
cat > .env << EOF
FUNCTION_APP_URL=$FUNCTION_APP_URL
SEARCH_ENDPOINT=$SEARCH_ENDPOINT
STORAGE_ACCOUNT=$STORAGE_ACCOUNT
POSTGRES_SERVER=$POSTGRES_SERVER
PG_ADMIN_PASSWORD=$PG_ADMIN_PASSWORD
EOF

echo "Connection information saved to .env file"
```

## Post-Deployment Configuration

### Create Azure AI Search Index

```bash
# Get the search admin key
SEARCH_KEY=$(az search admin-keys show --service-name $(az deployment group show --resource-group $RG_NAME --name main --query properties.outputs.SEARCH_SERVICE_NAME.value -o tsv) --resource-group $RG_NAME --query primaryKey -o tsv)

# Create the search index
curl -X PUT "$SEARCH_ENDPOINT/indexes/documents?api-version=2023-11-01" \
  -H "Content-Type: application/json" \
  -H "api-key: $SEARCH_KEY" \
  -d '{
    "name": "documents",
    "fields": [
      {"name": "id", "type": "Edm.String", "key": true, "searchable": false},
      {"name": "title", "type": "Edm.String", "searchable": true, "filterable": true, "sortable": true},
      {"name": "content", "type": "Edm.String", "searchable": true},
      {"name": "source", "type": "Edm.String", "searchable": true, "filterable": true, "sortable": true},
      {"name": "date", "type": "Edm.DateTimeOffset", "searchable": false, "filterable": true, "sortable": true},
      {"name": "language", "type": "Edm.String", "searchable": false, "filterable": true, "sortable": true}
    ]
  }'
```

### Deploy Function App Code

```bash
# Navigate to the function app code directory
cd src

# Deploy the function app code
func azure functionapp publish $(az deployment group show --resource-group $RG_NAME --name main --query properties.outputs.FUNCTION_APP_NAME.value -o tsv)
```

## Monitoring and Management

### Monitor Costs

```bash
# View current month's costs
az consumption usage list \
  --query "[?contains(instanceName, 'taxdb')].{Resource:instanceName, Cost:pretaxCost}" \
  --output table
```

### View Logs

```bash
# Get the Log Analytics Workspace ID
WORKSPACE_ID=$(az monitor log-analytics workspace show \
  --resource-group $RG_NAME \
  --workspace-name "$(az deployment group show --resource-group $RG_NAME --name main --query properties.outputs.FUNCTION_APP_NAME.value -o tsv)-logs" \
  --query customerId -o tsv)

# Query logs
az monitor log-analytics query \
  --workspace $WORKSPACE_ID \
  --analytics-query "AppTraces | where TimeGenerated > ago(1h) | order by TimeGenerated desc"
```

### Manage Database

```bash
# Connect to PostgreSQL database
PGPASSWORD=$PG_ADMIN_PASSWORD psql \
  -h $(az postgres flexible-server show --resource-group $RG_NAME --name $POSTGRES_SERVER --query fullyQualifiedDomainName -o tsv) \
  -U taxdbadmin \
  -d taxdb
```

## Cleanup

To delete all resources when they are no longer needed:

```bash
# Delete the resource group and all resources within it
az group delete --name $RG_NAME --yes
```

## Troubleshooting

### Common Issues

1. **Deployment Failures**:
   - Check resource name availability
   - Verify parameter values meet requirements
   - Review deployment logs: `az deployment group show --resource-group $RG_NAME --name main`

2. **Connection Issues**:
   - Verify network security rules
   - Check firewall configurations
   - Ensure service endpoints are properly configured

3. **Performance Issues**:
   - Monitor CPU and memory usage
   - Review Application Insights performance metrics
   - Consider scaling up resources if necessary (but be mindful of cost impact)

### Getting Support

For issues with the infrastructure, please:
1. Check the Azure status page: https://status.azure.com/
2. Review Azure documentation for specific services
3. Open an issue in the project repository

## Security Considerations

The infrastructure includes several security features:

- All communications encrypted with TLS 1.2+
- Secrets stored in Key Vault
- Network isolation with VNet and service endpoints
- Managed identities for authentication
- HTTPS enforcement
- Minimal public exposure

## Outputs

The deployment outputs the following connection strings:

- `PG_CONNSTR` - PostgreSQL connection string
- `BLOB_CONNSTR` - Blob storage connection string
- `SEARCH_ENDPOINT` - Azure AI Search endpoint
- `FUNCTION_APP_URL` - Function App URL
- `STORAGE_ACCOUNT_NAME` - Storage account name
- `POSTGRES_SERVER_NAME` - PostgreSQL server name
- `SEARCH_SERVICE_NAME` - Search service name
- `KEY_VAULT_NAME` - Key Vault name
- `FUNCTION_APP_NAME` - Function App name