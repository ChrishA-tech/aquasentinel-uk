# AquaSentinel UK

**UK Sewage Spill Forecasting, River IoT Telemetry, Hydrological Network Graphs, and Ofwat PR24 / Environment Act Compliance Copilot**

AquaSentinel UK is an end-to-end environmental intelligence platform engineered for UK water companies, Environment Agency (EA) catchment analysts, and regulatory compliance teams. It links 15-minute Event Duration Monitoring (EDM) & river IoT telemetry with directed hydrological catchment graphs, uncertainty-calibrated ML spill forecasting, spatio-temporal deep learning, and a guardrailed Ofwat PR24 / Environment Act 2021 regulatory copilot.

## Architecture

- **Telemetry Ingestion & Live UK EA Connectors (`src/ingestion/hydrology_telemetry.py`, `src/ingestion/live_hydrology_connectors.py`)**
  - Live UK Environment Agency Flood & Hydrology API connector (`fetch_live_uk_ea_hydrology_and_edm`) with `tenacity` retry/backoff and graceful `LIVE_EA_FEED_DEGRADED` offline fallback, paired with calibrated CSO EDM telemetry across `THM`, `UUW`, `SVT`, and `YKS`.
  - Pandera physical-bound and sensor-health contracts (`HydrologyEDMSchema`) validating 15-minute river level (`0.0 - 12.0 m`), flow (`0.0 - 1500.0 m³/s`), dissolved oxygen (`0.0 - 20.0 mg/L`), turbidity (`0.0 - 2000.0 NTU`), valid UK water company codes (`THM`, `SVT`, `UUW`, `SWW`, `YKS`, `ANGL`, `NWT`, `SRN`, `WSH`, `WEX`), and detecting stuck/flatlined IoT sensors.
  - UK GDPR reporter/landowner ID HMAC-SHA256 pseudonymisation.
  - DuckDB analytical persistence (`data/processed/hydrology.duckdb`) + Parquet Dead-Letter Quarantine (`data/quarantine/hydrology_quarantine.parquet`).

- **Hydrological Catchment Graph (`src/analytics/river_network.py`)**
  - `NetworkX` directed catchment graph connecting upstream Combined Sewer Overflow (CSO) outfalls to downstream EA water quality stations and designated Bathing Water sites across **Thames**, **Mersey**, **Severn**, and **Aire** basins.
  - Computes upstream-to-downstream pollution travel time, Antecedent Precipitation Index (API), and dry-weather spill anomaly flags ("dry-day spilling").

- **Spill Forecasting & Spatio-Temporal Network (`src/models/spill_and_vision.py`)**
  - Catchment-grouped LightGBM + `MAPIE` 90% conformal prediction intervals forecasting CSO spill probability and duration across `t+6h`, `t+12h`, and `t+24h` horizons via `GroupKFold` by river basin, logged to MLflow SQLite (`sqlite:///mlruns/mlflow.db`).
  - PyTorch `CatchmentSpillNet` modeling multi-lag rainfall-runoff dynamics and outfall turbidity/algal plume risk.

- **Sliced Error Analysis (`src/evaluation/error_analysis.py`)**
  - Audits spill duration MAE/RMSE, Precision-Recall F1 on rare dry-day spills, and 90% conformal interval coverage across operational slices (`Urban Combined Sewer`, `Rural Chalk Stream`, `Heavy Storm Event`, `Dry-Weather Anomaly`), generating `docs/ERROR_ANALYSIS_REPORT.md`.

- **Ofwat PR24 & Environment Act Compliance Copilot (`src/agents/ofwat_copilot.py`)**
  - Disk-backed regulatory document loader (`load_ofwat_regulatory_corpus`) parsing `.pdf` (`pypdf`), `.md`, and `.txt` files from `data/regulatory_docs/`.
  - Guardrailed LangGraph workflow with hybrid BM25 + TF-IDF retrieval over Ofwat PR24 Final Determinations, Environment Act 2021 Section 141A storm overflow reduction duties, and EA EDM permitting rules.
  - Includes a citation-verifier judge node and a deterministic statutory breach/ODI penalty exposure fallback calculator.

- **Serving, Dashboard & Skills Matrix (`src/app/main.py`, `src/app/dashboard.py`, `src/run_pipeline.py`, `docs/SKILLS_MATRIX.md`)**
  - FastAPI service exposing `/health`, `/forecast-spills` (with spatial interpolation fallback on sensor dropout), and `/regulatory-audit`.
  - Interactive Streamlit dashboard for catchment spill risk monitoring and regulatory copilot queries.
  - 300-capability engineering & domain mapping in `docs/SKILLS_MATRIX.md`.

## Quickstart

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]" streamlit tabulate

# Run end-to-end live UK EA hydrology ingestion (with automatic offline fallback),
# river graph analysis, conformal + PyTorch training, error audit & copilot verification
python -m src.run_pipeline --live

# Or force offline deterministic synthetic mode
python -m src.run_pipeline --no-live

# Run tests and linter
pytest -v
ruff check src tests

# Launch FastAPI & Streamlit
uvicorn src.app.main:app --host 0.0.0.0 --port 8000
streamlit run src/app/dashboard.py
```

## Docker Compose

```bash
docker compose up --build
```

Services:
- `pgvector`: PostgreSQL 16 + `pgvector` on port `5432`
- `mlflow`: MLflow tracking server backed by `sqlite:///mlruns/mlflow.db` on port `5000`
- `api`: FastAPI service on port `8000`
