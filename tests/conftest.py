import pytest
import pandas as pd
from fastapi.testclient import TestClient

from prodml.predict import DurationPredictor
from prodml.api.main import app


@pytest.fixture(scope="session")
def trained_model():
    """Session-scoped fixture: loads the predictor once for all test files."""
    predictor = DurationPredictor()
    predictor.load()
    return predictor


@pytest.fixture
def sample_features():
    """Returns sample raw dataframe data for feature testing."""
    df = pd.DataFrame(
        [
            {
                "lpep_pickup_datetime": "2024-01-01 00:00:00",
                "lpep_dropoff_datetime": "2024-01-01 00:15:00",
                "PULocationID": 138,
                "DOLocationID": 265,
                "trip_distance": 5.0,
            },
            {
                "lpep_pickup_datetime": "2024-01-01 00:00:00",
                "lpep_dropoff_datetime": "2024-01-01 00:00:10",  # < 1 min (should be filtered)
                "PULocationID": 100,
                "DOLocationID": 100,
                "trip_distance": 0.1,
            },
        ]
    )
    df["lpep_pickup_datetime"] = pd.to_datetime(df["lpep_pickup_datetime"])
    df["lpep_dropoff_datetime"] = pd.to_datetime(df["lpep_dropoff_datetime"])
    return df


@pytest.fixture
def client():
    """FastAPI TestClient fixture with lifespan startup/shutdown triggered."""
    with TestClient(app) as test_client:
        yield test_client
