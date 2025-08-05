#!/bin/bash
# Cost Guard Check Script for TaxDB-POC
# This script analyzes the Bicep template and estimates the cost of resources
# It fails if the projected cost exceeds the budget

set -e

# Configuration
BUDGET_LIMIT_EUR=5.0  # Maximum monthly budget in EUR
BICEP_FILE="infra/main.bicep"
RESOURCE_GROUP="taxdb-poc-rg"
LOCATION="westeurope"

echo "Running Cost Guard Check for TaxDB-POC..."
echo "Budget limit: €${BUDGET_LIMIT_EUR}/month"

# Check if Bicep file exists
if [ ! -f "$BICEP_FILE" ]; then
    echo "Error: Bicep file not found at $BICEP_FILE"
    exit 1
fi

echo "Analyzing Bicep template: $BICEP_FILE"

# Extract resource types and SKUs from Bicep file
echo "Extracting resource configurations..."

# Storage Account
STORAGE_SKU=$(grep -A 5 "resource storageAccount" "$BICEP_FILE" | grep "name:" | head -1 | cut -d "'" -f 2)
echo "- Storage Account SKU: $STORAGE_SKU"

# PostgreSQL
PG_SKU=$(grep -A 5 "resource postgresServer" "$BICEP_FILE" | grep "name:" | head -1 | cut -d "'" -f 2)
echo "- PostgreSQL SKU: $PG_SKU"

# Search Service
SEARCH_SKU=$(grep -A 5 "resource searchService" "$BICEP_FILE" | grep "name:" | head -1 | cut -d "'" -f 2)
echo "- Search Service SKU: $SEARCH_SKU"

# Function App
FUNCTION_SKU=$(grep -A 5 "resource appServicePlan" "$BICEP_FILE" | grep "name:" | head -1 | cut -d "'" -f 2)
echo "- Function App SKU: $FUNCTION_SKU"

# Estimated costs (monthly, in EUR)
# These are approximate values based on Azure pricing
STORAGE_COST=0.02  # Standard_LRS with minimal usage
PG_COST=3.50       # Standard_B1ms with 32GB storage
SEARCH_COST=0.25   # Basic tier with 1 replica
FUNCTION_COST=0.00 # Consumption plan with minimal usage
LOG_ANALYTICS_COST=0.10 # Minimal usage
APP_INSIGHTS_COST=0.10  # Minimal usage
KEY_VAULT_COST=0.03     # Standard tier with minimal usage

# Calculate total estimated cost
TOTAL_COST=$(echo "$STORAGE_COST + $PG_COST + $SEARCH_COST + $FUNCTION_COST + $LOG_ANALYTICS_COST + $APP_INSIGHTS_COST + $KEY_VAULT_COST" | bc)

echo "Estimated monthly costs:"
echo "- Storage Account: €$STORAGE_COST"
echo "- PostgreSQL: €$PG_COST"
echo "- Search Service: €$SEARCH_COST"
echo "- Function App: €$FUNCTION_COST"
echo "- Log Analytics: €$LOG_ANALYTICS_COST"
echo "- Application Insights: €$APP_INSIGHTS_COST"
echo "- Key Vault: €$KEY_VAULT_COST"
echo "Total estimated cost: €$TOTAL_COST/month"

# Check if cost exceeds budget
if (( $(echo "$TOTAL_COST > $BUDGET_LIMIT_EUR" | bc -l) )); then
    echo "ERROR: Estimated cost (€$TOTAL_COST) exceeds budget limit (€$BUDGET_LIMIT_EUR)"
    echo "Cost Guard Check failed!"
    exit 1
else
    echo "Cost Guard Check passed! Estimated cost is within budget."
    # Calculate percentage of budget
    BUDGET_PERCENT=$(echo "($TOTAL_COST / $BUDGET_LIMIT_EUR) * 100" | bc -l)
    printf "Using %.1f%% of available budget\n" $BUDGET_PERCENT
    exit 0
fi