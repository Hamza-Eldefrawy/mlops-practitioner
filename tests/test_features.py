import pytest
from sklearn.feature_extraction import DictVectorizer
from prodml import features


def test_prepare_features_filters_outliers(sample_features):
    processed_df = features.prepare_features(sample_features)
    # The 10-second ride should be dropped
    assert len(processed_df) == 1
    assert "PU_DO" in processed_df.columns
    assert processed_df.iloc[0]["PU_DO"] == "138_265"


@pytest.mark.parametrize(
    "pu,do,distance",
    [
        ("999", "999", 5.0),  # Unseen PU_DO pair
        ("138", "265", 0.0),  # Zero distance
        (None, "265", 3.2),  # Missing category
    ],
)
def test_vectorize_handles_edge_cases(pu, do, distance):
    # Train vectorizer on baseline data
    dv = DictVectorizer()
    train_data = [{"PU_DO": "138_265", "trip_distance": 10.0}]
    dv.fit(train_data)

    # Transform unseen/edge case data
    test_data = [{"PU_DO": f"{pu}_{do}", "trip_distance": distance}]
    X = dv.transform(test_data)

    # Should transform successfully
    assert X.shape[0] == 1
    assert X.shape[1] == len(dv.feature_names_)
