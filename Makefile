PY?=python3.12

init:            ## create venv & install deps
	$(PY) -m venv .venv && .venv/bin/pip install -U pip
	.venv/bin/pip install poetry==1.8.2
	.venv/bin/poetry install

compose-up:      ## start local services
	docker compose -f docker/docker-compose.yml up -d

compose-down:    ## stop local services
	docker compose -f docker/docker-compose.yml down

etl-run-once:    ## single threaded local ETL
	.venv/bin/python -m src.etl.be_moniteur
	.venv/bin/python -m src.etl.es_boe
	.venv/bin/python -m src.etl.de_bgbl

serve-local:     ## run API locally
	.venv/bin/uvicorn src.api:create_app --factory --reload

test:            ## run all tests
	.venv/bin/pytest -q

test-cov:        ## run tests with coverage
	.venv/bin/pytest --cov=src --cov-report=term-missing

test-cov-html:   ## run tests with coverage and generate HTML report
	.venv/bin/pytest --cov=src --cov-report=html

test-unit:       ## run unit tests only
	.venv/bin/pytest -m "unit" -q

test-integration: ## run integration tests only
	.venv/bin/pytest -m "integration" -q

test-e2e:        ## run end-to-end tests only
	.venv/bin/pytest -m "e2e" -q

test-docker:     ## run tests in Docker environment
	docker compose -f docker/docker-compose.yml run --rm app .venv/bin/pytest

deploy-infra:    ## deploy Azure infrastructure
	az deployment group create --resource-group $(RG_NAME) --template-file infra/main.bicep

smoke-cloud:     ## run cloud smoke test
	.venv/bin/python -m scripts.smoke_test

cost-check:      ## check Azure cost projection
	bash scripts/cost_check.sh

gen-openapi:     ## generate OpenAPI spec
	.venv/bin/python -m scripts.gen_openapi

load-sample:     ## load sample data
	bash scripts/load_sample.sh

.PHONY: init compose-up compose-down etl-run-once serve-local test test-cov test-cov-html test-unit test-integration test-e2e test-docker deploy-infra smoke-cloud cost-check gen-openapi load-sample