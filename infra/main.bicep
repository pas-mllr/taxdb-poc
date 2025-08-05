@description('Name of the resource group')
param rgName string

@description('Location for all resources')
param location string = resourceGroup().location

@description('PostgreSQL server admin login name')
param administratorLogin string = 'taxdbadmin'

@description('PostgreSQL server admin password')
@secure()
param administratorLoginPassword string

@description('PostgreSQL server name')
param postgresServerName string = 'taxdb-pg-${uniqueString(resourceGroup().id)}'

@description('Storage account name')
param storageAccountName string = 'taxdbstor${uniqueString(resourceGroup().id)}'

@description('Search service name')
param searchServiceName string = 'taxdb-search-${uniqueString(resourceGroup().id)}'

@description('Function app name')
param functionAppName string = 'taxdb-func-${uniqueString(resourceGroup().id)}'

@description('Key vault name')
param keyVaultName string = 'taxdb-kv-${uniqueString(resourceGroup().id)}'

@description('Tags to apply to all resources')
param tags object = {
  project: 'TaxDB-POC'
  environment: 'dev'
  costCenter: 'research'
  createdBy: 'bicep'
}

// Define a parameter for the current date in ISO format for lifecycle management
@description('Current date in ISO format for lifecycle management')
param currentDate string = utcNow('yyyy-MM-dd')

// Virtual Network for secure communication
resource vnet 'Microsoft.Network/virtualNetworks@2023-05-01' = {
  name: 'taxdb-vnet'
  location: location
  tags: tags
  properties: {
    addressSpace: {
      addressPrefixes: [
        '10.0.0.0/16'
      ]
    }
    subnets: [
      {
        name: 'default'
        properties: {
          addressPrefix: '10.0.0.0/24'
          serviceEndpoints: [
            {
              service: 'Microsoft.Storage'
            }
            {
              service: 'Microsoft.KeyVault'
            }
            {
              service: 'Microsoft.Sql'
            }
            {
              service: 'Microsoft.Search'
            }
          ]
          delegations: []
          privateEndpointNetworkPolicies: 'Disabled'
          privateLinkServiceNetworkPolicies: 'Enabled'
        }
      }
      {
        name: 'function-subnet'
        properties: {
          addressPrefix: '10.0.1.0/24'
          delegations: [
            {
              name: 'function-delegation'
              properties: {
                serviceName: 'Microsoft.Web/serverFarms'
              }
            }
          ]
          privateEndpointNetworkPolicies: 'Disabled'
          privateLinkServiceNetworkPolicies: 'Enabled'
        }
      }
    ]
  }
}

// Storage Account - Cost optimized with lifecycle management
resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: storageAccountName
  location: location
  tags: tags
  sku: {
    name: 'Standard_LRS' // Locally redundant storage is the most cost-effective option
  }
  kind: 'StorageV2'
  properties: {
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
    supportsHttpsTrafficOnly: true
    networkAcls: {
      bypass: 'AzureServices'
      defaultAction: 'Deny'
      virtualNetworkRules: [
        {
          id: '${vnet.id}/subnets/default'
          action: 'Allow'
        }
      ]
    }
    encryption: {
      services: {
        blob: {
          enabled: true
        }
        file: {
          enabled: true
        }
      }
      keySource: 'Microsoft.Storage'
    }
    accessTier: 'Hot' // Hot tier for frequently accessed data
  }
}

// Create blob containers
resource rawContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  name: '${storageAccount.name}/default/raw'
  properties: {
    publicAccess: 'None'
    metadata: {
      description: 'Raw document files'
    }
  }
}

resource parsedContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  name: '${storageAccount.name}/default/parsed'
  properties: {
    publicAccess: 'None'
    metadata: {
      description: 'Parsed document files'
    }
  }
}

// Enhanced lifecycle management policy for cost optimization
resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-01-01' = {
  name: '${storageAccount.name}/default'
  properties: {
    deleteRetentionPolicy: {
      enabled: true
      days: 30
    }
    containerDeleteRetentionPolicy: {
      enabled: true
      days: 30
    }
    isVersioningEnabled: false // Disable versioning to save costs
    changeFeed: {
      enabled: false // Disable change feed to save costs
    }
    // Advanced lifecycle management
    cors: {
      corsRules: []
    }
    // Add management policy for cost optimization
    managementPolicy: {
      policy: {
        rules: [
          {
            name: 'raw-lifecycle'
            type: 'Lifecycle'
            definition: {
              actions: {
                baseBlob: {
                  // Move to cool tier after 30 days
                  tierToCool: {
                    daysAfterModificationGreaterThan: 30
                  }
                  // Delete after 90 days
                  delete: {
                    daysAfterModificationGreaterThan: 90
                  }
                }
              }
              filters: {
                blobTypes: [
                  'blockBlob'
                ]
                prefixMatch: [
                  'raw/'
                ]
              }
            }
          }
          {
            name: 'parsed-lifecycle'
            type: 'Lifecycle'
            definition: {
              actions: {
                baseBlob: {
                  // Move to cool tier after 15 days
                  tierToCool: {
                    daysAfterModificationGreaterThan: 15
                  }
                  // Delete after 60 days
                  delete: {
                    daysAfterModificationGreaterThan: 60
                  }
                }
              }
              filters: {
                blobTypes: [
                  'blockBlob'
                ]
                prefixMatch: [
                  'parsed/'
                ]
              }
            }
          }
        ]
      }
    }
  }
}

// PostgreSQL Flexible Server - Cost optimized
resource postgresServer 'Microsoft.DBforPostgreSQL/flexibleServers@2023-03-01-preview' = {
  name: postgresServerName
  location: location
  tags: tags
  sku: {
    name: 'Standard_B1ms' // Burstable tier for cost optimization
    tier: 'Burstable'
  }
  properties: {
    version: '15'
    administratorLogin: administratorLogin
    administratorLoginPassword: administratorLoginPassword
    storage: {
      storageSizeGB: 32 // Minimum storage size
      autoGrow: 'Disabled' // Disable auto-grow to control costs
    }
    backup: {
      backupRetentionDays: 7 // Minimum backup retention
      geoRedundantBackup: 'Disabled' // Disable geo-redundant backup to save costs
    }
    highAvailability: {
      mode: 'Disabled' // Disable high availability to save costs
    }
    maintenanceWindow: {
      customWindow: 'Enabled'
      dayOfWeek: 0 // Sunday
      startHour: 3 // 3 AM
      startMinute: 0
    }
    network: {
      delegatedSubnetResourceId: null
      privateDnsZoneArmResourceId: null
    }
  }
}

// PostgreSQL Database
resource postgresDatabase 'Microsoft.DBforPostgreSQL/flexibleServers/databases@2023-03-01-preview' = {
  name: 'taxdb'
  parent: postgresServer
  properties: {
    charset: 'UTF8'
    collation: 'en_US.utf8'
  }
}

// PostgreSQL Firewall Rule - Allow Azure services
resource postgresFirewallRule 'Microsoft.DBforPostgreSQL/flexibleServers/firewallRules@2023-03-01-preview' = {
  name: 'AllowAzureServices'
  parent: postgresServer
  properties: {
    startIpAddress: '0.0.0.0'
    endIpAddress: '0.0.0.0'
  }
}

// Enable pgvector extension
resource pgvectorExtension 'Microsoft.Resources/deploymentScripts@2020-10-01' = {
  name: 'enable-pgvector-extension'
  location: location
  tags: tags
  kind: 'AzureCLI'
  properties: {
    azCliVersion: '2.45.0'
    timeout: 'PT30M'
    retentionInterval: 'P1D'
    environmentVariables: [
      {
        name: 'PGHOST'
        value: postgresServer.properties.fullyQualifiedDomainName
      }
      {
        name: 'PGUSER'
        value: administratorLogin
      }
      {
        name: 'PGPASSWORD'
        secureValue: administratorLoginPassword
      }
      {
        name: 'PGDATABASE'
        value: 'taxdb'
      }
    ]
    scriptContent: '''
      apt-get update && apt-get install -y postgresql-client
      psql -c "CREATE EXTENSION IF NOT EXISTS vector;"
    '''
  }
  dependsOn: [
    postgresDatabase
    postgresFirewallRule // Ensure firewall rule is in place before trying to connect
  ]
}

// Azure AI Search - Cost optimized
resource searchService 'Microsoft.Search/searchServices@2023-11-01' = {
  name: searchServiceName
  location: location
  tags: tags
  sku: {
    name: 'basic' // Basic tier is the most cost-effective option with full functionality
  }
  properties: {
    replicaCount: 1 // Minimum replica count
    partitionCount: 1 // Minimum partition count
    hostingMode: 'default'
    publicNetworkAccess: 'enabled'
    networkRuleSet: {
      ipRules: []
    }
    encryptionWithCmk: {
      enforcement: 'Unspecified'
    }
    disableLocalAuth: false
    authOptions: {
      apiKeyOnly: {}
    }
    semanticSearch: 'disabled' // Disable semantic search to save costs
  }
}

// Function App - Cost optimized with Consumption plan
resource appServicePlan 'Microsoft.Web/serverfarms@2023-01-01' = {
  name: '${functionAppName}-plan'
  location: location
  tags: tags
  kind: 'linux'
  sku: {
    name: 'Y1' // Consumption plan - pay only for execution time
    tier: 'Dynamic'
  }
  properties: {
    reserved: true // Required for Linux
    maximumElasticWorkerCount: 1 // Limit worker count to control costs
  }
}

// Create a storage account for the function app
resource functionStorageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: '${take(replace(functionAppName, '-', ''), 17)}stor'
  location: location
  tags: tags
  sku: {
    name: 'Standard_LRS' // Locally redundant storage is the most cost-effective option
  }
  kind: 'StorageV2'
  properties: {
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
    supportsHttpsTrafficOnly: true
    encryption: {
      services: {
        blob: {
          enabled: true
        }
        file: {
          enabled: true
        }
      }
      keySource: 'Microsoft.Storage'
    }
    accessTier: 'Hot'
  }
}

// Function App with managed identity
resource functionApp 'Microsoft.Web/sites@2023-01-01' = {
  name: functionAppName
  location: location
  tags: tags
  kind: 'functionapp,linux'
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    serverFarmId: appServicePlan.id
    httpsOnly: true // Force HTTPS for security
    virtualNetworkSubnetId: '${vnet.id}/subnets/function-subnet' // Connect to VNet for security
    siteConfig: {
      linuxFxVersion: 'Python|3.12'
      minTlsVersion: '1.2' // Enforce minimum TLS version
      ftpsState: 'Disabled' // Disable FTPS for security
      appSettings: [
        {
          name: 'AzureWebJobsStorage'
          value: 'DefaultEndpointsProtocol=https;AccountName=${functionStorageAccount.name};EndpointSuffix=${environment().suffixes.storage};AccountKey=${functionStorageAccount.listKeys().keys[0].value}'
        }
        {
          name: 'FUNCTIONS_EXTENSION_VERSION'
          value: '~4'
        }
        {
          name: 'FUNCTIONS_WORKER_RUNTIME'
          value: 'python'
        }
        {
          name: 'AZURE_PG_CONNSTR'
          value: '@Microsoft.KeyVault(SecretUri=https://${keyVault.name}.vault.azure.net/secrets/AZURE-PG-CONNSTR/)'
        }
        {
          name: 'AZURE_BLOB_CONNSTR'
          value: '@Microsoft.KeyVault(SecretUri=https://${keyVault.name}.vault.azure.net/secrets/AZURE-BLOB-CONNSTR/)'
        }
        {
          name: 'AZURE_SEARCH_ENDPOINT'
          value: '@Microsoft.KeyVault(SecretUri=https://${keyVault.name}.vault.azure.net/secrets/AZURE-SEARCH-ENDPOINT/)'
        }
        {
          name: 'AZURE_SEARCH_KEY'
          value: '@Microsoft.KeyVault(SecretUri=https://${keyVault.name}.vault.azure.net/secrets/AZURE-SEARCH-KEY/)'
        }
        {
          name: 'LOCAL_MODE'
          value: 'false'
        }
        {
          name: 'WEBSITE_RUN_FROM_PACKAGE'
          value: '1' // Enable run from package for better cold start performance
        }
        {
          name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
          value: appInsights.properties.ConnectionString
        }
      ]
    }
  }
  dependsOn: [
    functionStorageAccount
    searchService
    keyVault
  ]
}

// Application Insights for monitoring
resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: '${functionAppName}-insights'
  location: location
  tags: tags
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalyticsWorkspace.id
    IngestionMode: 'LogAnalytics'
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
  }
}

// Log Analytics Workspace for centralized logging
resource logAnalyticsWorkspace 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: '${functionAppName}-logs'
  location: location
  tags: tags
  properties: {
    sku: {
      name: 'PerGB2018' // Pay-per-use pricing model
    }
    retentionInDays: 30 // Minimum retention period to save costs
    features: {
      enableLogAccessUsingOnlyResourcePermissions: true
    }
    workspaceCapping: {
      dailyQuotaGb: 1 // Limit daily ingestion to control costs
    }
  }
}

// Key Vault with access policies
resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: keyVaultName
  location: location
  tags: tags
  properties: {
    enabledForDeployment: true
    enabledForTemplateDeployment: true
    enabledForDiskEncryption: true
    tenantId: subscription().tenantId
    enableSoftDelete: true // Enable soft delete for data protection
    softDeleteRetentionInDays: 7 // Minimum retention period
    enableRbacAuthorization: false // Use access policies instead of RBAC
    networkAcls: {
      bypass: 'AzureServices'
      defaultAction: 'Deny'
      virtualNetworkRules: [
        {
          id: '${vnet.id}/subnets/default'
        }
      ]
    }
    sku: {
      name: 'standard'
      family: 'A'
    }
    accessPolicies: [] // Will be updated after function app is created
  }
}

// Update Key Vault access policies
resource keyVaultAccessPolicies 'Microsoft.KeyVault/vaults/accessPolicies@2023-07-01' = {
  name: 'add'
  parent: keyVault
  properties: {
    accessPolicies: [
      {
        tenantId: subscription().tenantId
        objectId: functionApp.identity.principalId
        permissions: {
          secrets: [
            'get'
            'list'
          ]
        }
      }
    ]
  }
}

// Store secrets in Key Vault
resource pgConnStrSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  name: '${keyVault.name}/AZURE-PG-CONNSTR'
  properties: {
    value: 'postgresql://${administratorLogin}:${administratorLoginPassword}@${postgresServer.properties.fullyQualifiedDomainName}:5432/taxdb'
    contentType: 'text/plain'
    attributes: {
      enabled: true
    }
  }
}

resource blobConnStrSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  name: '${keyVault.name}/AZURE-BLOB-CONNSTR'
  properties: {
    value: 'DefaultEndpointsProtocol=https;AccountName=${storageAccount.name};EndpointSuffix=${environment().suffixes.storage};AccountKey=${storageAccount.listKeys().keys[0].value}'
    contentType: 'text/plain'
    attributes: {
      enabled: true
    }
  }
}

resource searchEndpointSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  name: '${keyVault.name}/AZURE-SEARCH-ENDPOINT'
  properties: {
    value: 'https://${searchService.name}.search.windows.net'
    contentType: 'text/plain'
    attributes: {
      enabled: true
    }
  }
}

resource searchKeySecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  name: '${keyVault.name}/AZURE-SEARCH-KEY'
  properties: {
    value: searchService.listAdminKeys().primaryKey
    contentType: 'text/plain'
    attributes: {
      enabled: true
    }
  }
}

// Budget alert to monitor costs
resource budget 'Microsoft.Consumption/budgets@2023-05-01' = {
  name: 'taxdb-poc-budget'
  properties: {
    category: 'Cost'
    amount: 5 // €5 budget
    timeGrain: 'Monthly'
    timePeriod: {
      startDate: currentDate
    }
    filter: {
      tags: {
        name: 'project'
        operator: 'In'
        values: [
          'TaxDB-POC'
        ]
      }
    }
    notifications: {
      actual_80_Percent: {
        enabled: true
        operator: 'GreaterThanOrEqualTo'
        threshold: 80
        contactEmails: []
        contactRoles: [
          'Owner'
        ]
        thresholdType: 'Actual'
      }
      actual_100_Percent: {
        enabled: true
        operator: 'GreaterThanOrEqualTo'
        threshold: 100
        contactEmails: []
        contactRoles: [
          'Owner'
        ]
        thresholdType: 'Actual'
      }
    }
  }
}

// Outputs
output PG_CONNSTR string = 'postgresql://${administratorLogin}:${administratorLoginPassword}@${postgresServer.properties.fullyQualifiedDomainName}:5432/taxdb'
output BLOB_CONNSTR string = 'DefaultEndpointsProtocol=https;AccountName=${storageAccount.name};EndpointSuffix=${environment().suffixes.storage};AccountKey=${storageAccount.listKeys().keys[0].value}'
output SEARCH_ENDPOINT string = 'https://${searchService.name}.search.windows.net'
output FUNCTION_APP_URL string = 'https://${functionApp.properties.defaultHostName}'
output STORAGE_ACCOUNT_NAME string = storageAccount.name
output POSTGRES_SERVER_NAME string = postgresServer.name
output SEARCH_SERVICE_NAME string = searchService.name
output KEY_VAULT_NAME string = keyVault.name
output FUNCTION_APP_NAME string = functionApp.name
