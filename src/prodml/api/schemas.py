from typing import List
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    PU_DO: str = Field(description="Pickup and Dropoff location ID string")
    trip_distance: float = Field(
        gt=0, lt=200, description="Distance must be between 0 and 200 miles"
    )

    model_config = {
        "json_schema_extra": {"examples": [{"PU_DO": "138_265", "trip_distance": 15.2}]}
    }


class PredictionResponse(BaseModel):
    prediction: float
    model_version: str
    correlation_id: str
    latency_ms: float


class BatchPredictionRequest(BaseModel):
    requests: List[PredictionRequest]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "requests": [
                        {"PU_DO": "138_265", "trip_distance": 15.2},
                        {"PU_DO": "236_237", "trip_distance": 2.1},
                    ]
                }
            ]
        }
    }


class BatchPredictionResponse(BaseModel):
    predictions: List[float]
    model_version: str
    correlation_id: str
    latency_ms: float
