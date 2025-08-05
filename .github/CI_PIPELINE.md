# TaxDB-POC CI Pipeline Documentation

This document provides detailed information about the Continuous Integration (CI) pipeline implemented for the TaxDB-POC project using GitHub Actions.

## Overview

The CI pipeline is designed to ensure code quality, test coverage, and infrastructure validation for the TaxDB-POC system. It runs automatically on push to main/develop branches and on pull requests, and can also be triggered manually.

The pipeline is optimized to:
- Complete in under 15 minutes
- Avoid hitting paid endpoints (Azure OpenAI, etc.)
- Use the LocalMiniLM embedding strategy for tests
- Cache dependencies to speed up execution
- Run tests in parallel where possible
- Provide comprehensive error reporting and notifications
- Generate code coverage reports

## Pipeline Stages

The CI pipeline consists of the following stages:

### 1. Linting and Code Quality

**Job name:** `lint`

This stage performs static code analysis to ensure code quality and adherence to style guidelines:

- **flake8**: Checks for syntax errors and style issues
- **black**: Verifies code formatting
- **isort**: Ensures imports are properly organized
- **mypy**: Performs type checking
- **bandit**: Scans for security vulnerabilities

### 2. Unit Tests

**Job name:** `unit-tests`

This stage runs unit tests that don't require external dependencies:

- Uses pytest with the `unit` marker
- Generates code coverage reports
- Uses LocalMiniLM embedding strategy to avoid hitting paid endpoints
- Uploads coverage reports to Codecov

### 3. Integration Tests

**Job name:** `integration-tests`

This stage tests interactions between components with mocked external services:

- Runs a PostgreSQL service with pgvector extension
- Initializes the database with required extensions
- Runs pytest with the `integration` marker
- Mocks external services to avoid hitting paid endpoints
- Generates and uploads coverage reports

### 4. End-to-End Tests

**Job name:** `e2e-tests`

This stage performs full system testing using Docker Compose:

- Starts all required services using Docker Compose
- Runs pytest with the `e2e` marker
- Uses LocalMiniLM embedding strategy
- Generates and uploads coverage reports
- Ensures proper cleanup of Docker resources

### 5. Infrastructure Validation

**Job name:** `infra-validation`

This stage validates the Bicep infrastructure templates:

- Builds and lints Bicep templates
- Runs cost guard checks to prevent exceeding budget
- Ensures infrastructure changes are valid and cost-effective

### 6. Code Coverage Report

**Job name:** `coverage`

This stage combines coverage reports from all test stages:

- Merges coverage data from unit, integration, and E2E tests
- Generates a combined coverage report
- Creates a coverage badge for the repository
- Uploads the combined report to Codecov

### 7. Notification

**Job name:** `notify`

This stage sends notifications about pipeline status:

- Checks the status of all previous jobs
- Sends success or failure notifications to Slack
- Provides quick feedback on pipeline results

## Environment Variables

The pipeline uses the following environment variables:

- `PYTHON_VERSION`: Python version to use (3.12.11)
- `POETRY_VERSION`: Poetry version for dependency management (1.8.2)
- `LOCAL_MODE`: Set to 'true' to use local resources instead of cloud services
- `EMBEDDING_STRATEGY`: Set to 'local_minilm' to avoid hitting paid endpoints
- `MAX_PARALLEL_TESTS`: Controls parallel test execution (4)
- `MOCK_EXTERNAL_SERVICES`: Set to 'true' to mock external service calls
- `PG_CONNSTR`: PostgreSQL connection string for tests

## Dependency Caching

The pipeline implements caching to speed up execution:

- Python dependencies are cached using the `actions/setup-python@v5` cache parameter
- Poetry dependencies are installed efficiently with proper caching
- Docker images are pulled only when needed

## Parallel Execution

Tests are executed in parallel to reduce pipeline duration:

- Unit tests, integration tests, and infrastructure validation run in parallel
- E2E tests run after unit and integration tests complete
- Coverage reporting runs after all tests complete
- Maximum parallel execution is controlled by `MAX_PARALLEL_TESTS`

## Error Reporting and Notifications

The pipeline provides comprehensive error reporting:

- Test failures are reported with detailed output
- Coverage reports highlight areas with insufficient testing
- Slack notifications provide immediate feedback on pipeline status
- GitHub status checks prevent merging code that fails the pipeline

## Branch Protection Rules

To fully implement the CI pipeline, set up the following branch protection rules in GitHub:

1. Navigate to your repository settings
2. Go to "Branches" > "Branch protection rules"
3. Add a rule for `main` and `develop` branches with:
   - Require status checks to pass before merging
   - Require branches to be up to date before merging
   - Require the following status checks:
     - lint
     - unit-tests
     - integration-tests
     - e2e-tests
     - infra-validation
     - coverage

## Cost Guard Check

The pipeline includes a cost guard check to prevent exceeding budget:

- The `scripts/cost_check.sh` script analyzes infrastructure costs
- It fails the pipeline if projected costs exceed the budget
- This prevents accidental deployment of expensive resources

## Best Practices

When working with this CI pipeline, follow these best practices:

1. **Run tests locally** before pushing to avoid unnecessary CI runs
2. **Add appropriate test markers** (unit, integration, e2e) to new tests
3. **Mock external services** in tests to avoid hitting paid endpoints
4. **Keep the pipeline fast** by optimizing tests and dependencies
5. **Monitor coverage reports** to maintain or improve test coverage
6. **Review CI logs** for warnings and optimization opportunities

## Troubleshooting

Common issues and solutions:

1. **Pipeline taking too long**: Check if tests can be optimized or if some can be moved to a different stage
2. **Tests failing in CI but passing locally**: Ensure environment variables are set correctly and external services are properly mocked
3. **Coverage reports not uploading**: Verify Codecov token and permissions
4. **Notifications not working**: Check Slack webhook configuration

## Future Improvements

Potential enhancements for the CI pipeline:

1. Add performance testing stage
2. Implement automated dependency updates
3. Add visual regression testing for UI components
4. Expand cost analysis to include runtime costs
5. Add automated documentation generation