"""
\########################################################################################################

# Improved Machine‑Executable Specification – TaxDB‑POC (Belgium | Spain | Germany)

# v2 – Tweaks for reliability, local dev ergonomics, and cost‑control

\########################################################################################################

### 📜 Instruction Prompt  (place at the very top of **README.md**)

> **You are an AI coding assistant. Implement every item marked “`REQ:`” in this document.**
> **Absolutely no deviations** from directory layout, file names, or versions.
>
> 1. Scaffold the repo exactly as described.
> 2. Use **Python 3.12** and pin all third‑party packages *exactly* to the versions stated.
> 3. Provision Azure resources via **Bicep** – the template must deploy end‑to‑end with a single CLI command.
> 4. Provide a friction‑free **local mode** (Docker Compose: Postgres + Azurite) so that `make test` does **not** require an Azure subscription or paid OpenAI credits.
> 5. Keep the entire POC’s recurring cloud spend **≤ €5 / month** – this means *Basic* tier everywhere and consumption‑plan Functions.
> 6. CI must finish in **< 15 min** and never hit a paid Azure/OpenAI endpoint.

---

## 0. 🌡️ Definition of Done (`make test`)

`make test` (run on GitHub Actions and locally) **MUST**:

1. **Download ≥ 1 document** per jurisdiction published ≤ 24 h ago *or* gracefully skip if none available, recording “0 rows fetched” without raising.
2. **Insert** each document into the local Postgres (with pgvector) **and** a *local* search shim.
3. **Expose** them via the FastAPI service:

   * `GET /healthz` returns `{"status":"ok"}` (`200`).
   * `GET /search?q=tax&jurisdiction=ES` returns ≥ 1 hit (unless 0 docs ingested – then an empty list, still `200`).
   * `GET /doc/{id}` streams JSON metadata and a presigned blob URL.
4. **All tests green** (`pytest -q` exits 0).

Cloud deployment is validated separately by `make deploy-infra && make smoke-cloud`.

---

## 1. 🔒 Version Matrix (immutable)

| Layer                  | Version      |
| ---------------------- | ------------ |
| Python                 | **3.12.2**   |
| FastAPI                | **0.111.0**  |
| SQLAlchemy             | **2.0.29**   |
| pgvector‑python        | **0.2.5**    |
| azure‑identity         | **1.16.0**   |
| azure‑storage‑blob     | **12.19.1**  |
| azure‑search‑documents | **11.5.0b7** |
| azure‑ai‑generative    | **1.0.0b1**  |
| pdfminer.six           | **20221105** |
| pytesseract            | **0.3.12**   |
| pytest                 | **8.2.1**    |
| httpx                  | **0.27.0**   |
| Bicep                  | **0.23.2**   |

*`REQ:` Freeze these in `pyproject.toml` → `poetry.lock`.*

---

## 2. Directory Layout (💥 **MUST match exactly**) `REQ:`

```
.
├── infra/
│   ├── main.bicep
│   └── README.md
├── docker/
│   ├── docker-compose.yml
│   └── pgvector-init.sql
├── src/
│   ├── __init__.py
│   ├── settings.py
│   ├── models.py
│   ├── api.py
│   └── etl/
│       ├── __init__.py
│       ├── utils.py
│       ├── be_moniteur.py
│       ├── es_boe.py
│       └── de_bgbl.py
├── scripts/
│   ├── gen_openapi.py
│   └── load_sample.sh
├── openapi/
│   └── taxdb.yaml
├── tests/
│   ├── test_e2e.py
│   └── conftest.py
├── .env.example
├── Makefile
├── pyproject.toml
└── README.md   <-- this file
```

---

## 3. Environment Variables `.env` `REQ:`

```
# local mode (false = cloud)
LOCAL_MODE=true

# Azure / OpenAI – leave blank for local runs
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_KEY=
AZURE_SEARCH_ENDPOINT=
AZURE_SEARCH_KEY=
AZURE_PG_CONNSTR=
AZURE_BLOB_CONNSTR=

# CI convenience
DOC_LOOKBACK_HOURS=48
```

*If `LOCAL_MODE=true`, all Azure/OpenAI calls **MUST** be stubbed or routed to local shims so tests never hit the internet.*

---

## 4. Data Model `src/models.py` `REQ:`

```python
from sqlalchemy import (
    Column, Date, DateTime, Enum, Float, String, Text, func, Index
)
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase): pass

class Document(Base):
    __tablename__ = "documents"

    id            = Column(String, primary_key=True)  # "BE:20250804:AR-123"
    jurisdiction  = Column(Enum("BE", "ES", "DE", name="jurisd"))
    source_system = Column(String, nullable=False)
    document_type = Column(String, nullable=False)
    title         = Column(Text, nullable=False)
    summary       = Column(Text, nullable=True)
    issue_date    = Column(Date, nullable=False)
    effective_date= Column(Date, nullable=True)
    language_orig = Column(String(2), nullable=False)
    blob_url      = Column(Text, nullable=False)
    checksum      = Column(String(64), unique=True, nullable=False)
    vector        = Column(Vector(1536))
    created_at    = Column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("ix_jurisdiction_date", "jurisdiction", "issue_date"),
        Index("ix_vector", "vector", postgresql_using="ivfflat"),
    )
```

---

## 5. Infrastructure as Code `infra/main.bicep` `REQ:`

* Resource Group param `rgName`
* Deploy:

  * **Storage Account** – containers `raw` & `parsed` (lifecycle rule: delete after 30 days).
  * **PostgreSQL Flexible Server** (Basic SKU) + `pgvector` extension.
  * **Azure AI Search** (Basic) – index `documents`.
  * **Function App** (Python, Consumption ‑ Linux).
  * **Key Vault** – store all secrets.
* Outputs: `PG_CONNSTR`, `BLOB_CONNSTR`, `SEARCH_ENDPOINT`, `FUNCTION_APP_URL`.
* **Cost guard‑rail** – set throughput to the minimum allowed (AI Search = 1 replica × 1 partition).

---

## 6. ETL Framework `src/etl/utils.py` `REQ:`

* Implement **one generic** async function `run_pipeline(jurisdiction: str, fetch: Callable)` used by all three loaders.
* Provide two text‑embedding strategies:

  * `AzureOpenAIEmbedding` (used when `LOCAL_MODE=false`).
  * `LocalMiniLMEmbedding` (HuggingFace `all‑MiniLM‑L6‐v2`) for local/CI.
* **Download cache** (`~/.cache/taxdb/`) so repeated tests avoid network.
* **Idempotency**: skip insert if `checksum` already exists.

### Jurisdiction‑specific loaders `REQ:`

| File                | Fetch logic                                                                    | Parse logic                                       | Min. fields                    |
| ------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------- | ------------------------------ |
| **be\_moniteur.py** | GET XML: `https://www.ejustice.just.fgov.be/eli/{YYYY}{MM}{DD}/MONITOR/nl/xml` | XPath on `<doc>` nodes                            | `id` (eli), title, issue\_date |
| **es\_boe.py**      | GET XML: `https://www.boe.es/diario_boe/xml.php?id=BOE-B-{YYYY}{MM}{DD}`       | Only `<item>` where `<materia>` contains `FISCAL` | same                           |
| **de\_bgbl.py**     | Find today’s PDF via index JSON `https://www.bgbl.de/metadata.aktion?today`    | `pdfminer.six` → text; language = “de”            | same                           |

*All three loaders must call `run_pipeline`.*

---

## 7. API Service `src/api.py` `REQ:`

* FastAPI app factory pattern (`create_app(settings)`).
* **/search** – hybrid:

  1. IF `LOCAL_MODE=true` → cosine similarity via pgvector + ILIKE.
  2. ELSE → Azure AI Search hybrid query (vector + BM25).
* **/doc/{id}** – returns Pydantic DTO + **SAS URL** valid 15 min.
* **CORS** – allow origins `https://*.shell.com` + `http://localhost:*`.

---

## 8. Local Dev / CI Docker Compose `docker/docker-compose.yml` `REQ:`

```yaml
version: "3.9"
services:
  db:
    image: ankane/pgvector:0.5.1
    environment:
      POSTGRES_USER: taxdb
      POSTGRES_PASSWORD: taxdb
      POSTGRES_DB: taxdb
    ports: ["5432:5432"]
    volumes: ["./pgvector-init.sql:/docker-entrypoint-initdb.d/pgvector.sql:ro"]

  azurite:
    image: mcr.microsoft.com/azure-storage/azurite:3.29.0
    command: "azurite-blob --blobHost 0.0.0.0 -l /data"
    ports: ["10000:10000"]
```

*`pgvector-init.sql` simply: `CREATE EXTENSION IF NOT EXISTS pgvector;`*

---

## 9. Makefile `REQ:`

```make
PY?=python3.12

init:            ## create venv & install deps
	$(PY) -m venv .venv && .venv/bin/pip install -U pip
	.venv/bin/pip install poetry==1.8.2
	.venv/bin/poetry install

compose-up:      ## start local services
	docker compose -f docker/docker-compose.yml up -d

etl-run-once:    ## single threaded local ETL
	.venv/bin/python -m src.etl.be_moniteur
	.venv/bin/python -m src.etl.es_boe
	.venv/bin/python -m src.etl.de_bgbl

serve-local:     ## run API locally
	.venv/bin/uvicorn src.api:create_app --factory --reload

test:            ## run all tests
	.venv/bin/pytest -q
```

---

## 10. Tests `tests/test_e2e.py` `REQ:`

* Fixture spins up API in a background thread on port `9000`.
* Execute `etl-run-once`.
* Assert:

  ```python
  assert db.session.query(Document).filter_by(jurisdiction="BE").count() >= 0
  ```

  (count ≥ 0 because today may have no docs; the important part is *no crash*).
* HTTP `GET http://localhost:9000/healthz` → 200.
* `GET /search?q=tax&jurisdiction=ES` → 200 (payload list).

---

## 11. CI  `.github/workflows/ci.yml` `REQ:`

* Runs on `ubuntu‑latest`.
* Steps: checkout, `make init`, `make compose-up`, `make test`.
* Matrix: `{ python-version: [3.12] }`.
* Use Docker‑layer caching (buildx) to speed up.
* **No Azure login** in CI – local mode only.

---

## 12. Cloud Smoke Test `make smoke-cloud` `REQ:`

* Requires env vars pointing to real Azure resources.
* Runs a single Function execution via `az functionapp invoke`.
* Confirms that AI Search index has ≥ 1 document.

---

## 13. Cost Guard Check `scripts/cost_check.sh` `REQ:`

* Call `az consumption usage list --top 1` and fail pipeline if projected monthly cost > €5.

---

## 14. 🎯 Acceptance Checklist

* [ ] Repo structure matches **exactly**.
* [ ] `make init && make compose-up && make test` passes locally with no external network (except source downloads).
* [ ] `make deploy-infra && make smoke-cloud` works with personal Azure sub.
* [ ] README contains **clear, copy‑paste‑ready commands** for each step.

---

## 15. Recommended Implementation Order (non‑binding)

1. **Scaffold repo & poetry project.**
2. **Docker Compose** for Postgres+pgvector and Azurite.
3. **SQLAlchemy models & local search shim.**
4. **Spanish loader** (simplest API).  Run tests.
5. **FastAPI service** with stub search.  Ensure `/healthz` green.
6. **CI pipeline** (local mode).
7. Add **Belgian** XML loader.
8. Add **German** PDF loader (complex; parse only first page for POC).
9. **Azure Bicep** & smoke test.
10. Polish docs, cost‑check, OpenAPI generation, Power Connector guide.

---

### 🔧 Tips

* Use `rich` logging for pretty ETL output in GitHub Actions logs.
* For OCR fallback, cache Tesseract download in Docker layer.
* Limit OpenAI embedding calls to first **4 000 tokens** per doc chunk to cut cost.
* Wrap every network call in `tenacity.retry` with exponential back‑off.

Happy coding – remember: the bot in CI is ruthless. ✅
"""
