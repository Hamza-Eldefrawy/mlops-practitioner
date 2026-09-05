def test_prediction_type_and_range(trained_model):
    sample_ride = {"PU_DO": "138_265", "trip_distance": 15.2}
    prediction = trained_model.predict_one(sample_ride)

    # Check type
    assert isinstance(prediction, float)
    # Check sanity range (ride duration between 1 min and 120 min)
    assert 1.0 <= prediction <= 120.0


def test_prediction_determinism(trained_model):
    sample_ride = {"PU_DO": "138_265", "trip_distance": 15.2}
    pred_1 = trained_model.predict_one(sample_ride)
    pred_2 = trained_model.predict_one(sample_ride)

    # must return same number
    assert pred_1 == pred_2
