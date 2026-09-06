def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_happy_path_mocked(client, monkeypatch):
    """Uses monkeypatch so API test does not depend on model files."""
    # Mock predictor.predict_one to always return a static 18.5
    from prodml.api import main

    monkeypatch.setattr(main.predictor, "predict_one", lambda ride: 18.5)

    payload = {"PU_DO": "138_265", "trip_distance": 15.2}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 18.5
    assert data["model_version"] == "v1.0.0"
    assert "correlation_id" in data
    assert "latency_ms" in data


def test_predict_validation_error(client):
    """Sending a negative distance should return 422 Unprocessable Entity."""
    payload = {"PU_DO": "138_265", "trip_distance": -5.0}
    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert "errors" in response.json()


def test_metadata(client):
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["model_version"] == "v1.0.0"
    assert "PU_DO" in data["features"]


def test_predict_batch_mocked(client, monkeypatch):
    """Mocks the batch prediction to avoid requiring the real model."""
    from prodml.api import main

    # Tell the predictor to always return a list of [15.5, 20.2]
    monkeypatch.setattr(main.predictor, "predict_batch", lambda rides: [15.5, 20.2])

    payload = {
        "requests": [
            {"PU_DO": "138_265", "trip_distance": 15.2},
            {"PU_DO": "236_237", "trip_distance": 2.1},
        ]
    }
    response = client.post("/predict/batch", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["predictions"] == [15.5, 20.2]
    assert "latency_ms" in data
