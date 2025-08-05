# TaxDB-POC GitHub Actions Workflows

This directory contains GitHub Actions workflow files for the TaxDB-POC project.

## Available Workflows

### CI Pipeline (`ci.yml`)

A comprehensive Continuous Integration pipeline that runs on push to main/develop branches, pull requests, and can be triggered manually.

**Features:**
- Linting and code quality checks
- Unit tests with LocalMiniLM embedding strategy
- Integration tests with mocked external services
- End-to-end tests using Docker Compose
- Infrastructure validation for Bicep templates
- Code coverage reporting
- Error notifications

**Optimizations:**
- Completes in under 15 minutes
- Avoids hitting paid endpoints
- Uses dependency caching
- Runs tests in parallel
- Implements cost guard checks

For detailed documentation, see [CI_PIPELINE.md](../.github/CI_PIPELINE.md).