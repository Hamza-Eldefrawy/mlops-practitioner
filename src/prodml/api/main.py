import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from prodml.predict import DurationPredictor
from prodml.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
)
from prodml.logger import get_logger, request_id_context

logger = get_logger(__name__)

# The predictor object is initialized here, but NOT loaded yet.
predictor = DurationPredictor()


# 1. LIFESPAN MANAGER
# Loads the model into memory exactly once when the server boots up.
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Server starting up. Loading ONNX model into memory...")
    predictor.load()
    yield
    logger.info("Server shutting down. Cleaning up...")


app = FastAPI(title="Ride Duration Predictor", lifespan=lifespan)


# 2. MIDDLEWARE (The Request ID Injector)
@app.middleware("http")
async def inject_correlation_id(request: Request, call_next):
    # Generate a unique ID for this HTTP request
    correlation_id = str(uuid.uuid4())
    # Save it to our global context variable so the Logger can see it
    request_id_context.set(correlation_id)

    response = await call_next(request)

    # Return it to the client in the HTTP Headers
    response.headers["X-Correlation-ID"] = correlation_id
    return response


# 3. EXCEPTION HANDLERS
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Catches bad payloads and returns a clean 422 error without leaking stack traces."""
    logger.warning("Validation Error", extra={"errors": exc.errors()})
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Invalid payload format", "errors": exc.errors()},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catches all other crashes (500) and logs them, but hides the stack trace from the user."""
    logger.error("Unexpected Error", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error. Please try again later."},
    )


# 4. ENDPOINTS
@app.get("/health")
def health_check():
    # Only returns 200 OK if the ONNX session actually exists in memory
    if predictor.session is None or predictor.dv is None:
        return JSONResponse(status_code=503, content={"status": "Model not loaded"})
    return {"status": "healthy"}


@app.get("/metadata")
def metadata():
    return {
        "model_version": "v1.0.0",
        "training_date": "2026-08-25",
        "framework": "ONNX",
        "features": list(PredictionRequest.model_fields.keys()),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    start_time = time.perf_counter()

    # .model_dump() converts Pydantic back to a standard dictionary
    duration = predictor.predict_one(payload.model_dump())

    latency_ms = (time.perf_counter() - start_time) * 1000

    return PredictionResponse(
        prediction=duration,
        model_version="v1.0.0",
        correlation_id=request_id_context.get(),
        latency_ms=latency_ms,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchPredictionRequest):
    start_time = time.perf_counter()

    # Convert list of Pydantic objects into a list of dictionaries
    rides = [r.model_dump() for r in payload.requests]
    durations = predictor.predict_batch(rides)

    latency_ms = (time.perf_counter() - start_time) * 1000

    return BatchPredictionResponse(
        predictions=durations,
        model_version="v1.0.0",
        correlation_id=request_id_context.get(),
        latency_ms=latency_ms,
    )
