# MLOps Practitioner — Ride Duration Predictor

This repository contains a production-ready Machine Learning service that predicts NYC green taxi ride durations. Built as the Module 1 deliverable for the MLOps Practitioner course, it transforms an experimental Jupyter notebook into a pip-installable, tested, and containerized FastAPI backend.

## Quickstart

You can start this service and get a prediction on any machine with Docker installed in exactly three commands:

```bash
# 1. Pull the multi-stage optimized image
docker pull hamzaeldafrawy/prodml-api:0.1.0

# 2. Run the container locally on port 8000
docker run -d --rm -p 8000:8000 hamzaeldafrawy/prodml-api:0.1.0

# 3. Request a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"PU_DO": "166_143", "trip_distance": 2.58}'
```

**Example JSON Response:**

```json
{
  "prediction": 15.56,
  "model_version": "0.1.0",
  "correlation_id": "63d24b20-2177-42bf-9b4b-be4fc8ae5d9a",
  "latency_ms": 3.13
}
```

## Repository Tree

```text
.
├── src/
│   └── prodml/
│       ├── __init__.py
│       ├── config.py        # Settings via pydantic-settings
│       ├── data.py          # Data loading and train/val splitting
│       ├── features.py      # Feature engineering (e.g., PU_DO creation)
│       ├── train.py         # Model training execution
│       ├── predict.py       # DurationPredictor OOP interface
│       ├── export.py        # ONNX serialization
│       ├── logging_conf.py  # Structured JSON logging with correlation IDs
│       └── api/
│           ├── __init__.py
│           ├── schemas.py   # Pydantic request/response validation
│           └── main.py      # FastAPI application and lifespan events
│
├── tests/                   # Pytest suite with mock fixtures
├── models/                  # Serialized artifacts (model.pkl, model.onnx)
├── docker/
│   ├── Dockerfile           # Multi-stage optimized build
│   └── docker-compose.yml   # Local infrastructure as code
├── reports/
│   └── module-1.md          # Benchmark metrics and maturity assessment
├── pyproject.toml           # Project metadata, dependencies, and tools
└── README.md
```

## Model Pipeline

The system trains a regression model on the **NYC TLC Green Taxi trip data (January 2023)**.

* **Task:** Ride duration regression (minutes)
* **Features:** `PU_DO` (pickup/dropoff combination) and `trip_distance`
* **Artifacts:** Persisted as standard Pickle (`model.pkl`) and optimized ONNX (`model.onnx`) formats.
* **Evaluation:**

  * Validation MAE: 5.89 minutes
  * Validation RMSE: 68.59 minutes

## Local Development Workflow

If you are developing locally, the pipeline is managed via `uv` or `pip`.

```bash
# Install package in editable mode with development dependencies
pip install -e ".[dev]"

# Run static analysis (Linting & Formatting)
ruff check src tests && black --check src tests

# Execute the test suite with the 70% coverage gate
pytest -v --cov=src/prodml --cov-report=term-missing

# Train a new model artifact
python -m prodml.train

# Serve the API locally
uvicorn prodml.api.main:app --reload --port 8000
```
