import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer

from prodml.config import config


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates duration, filters outliers, and creates categorical string columns."""
    df = df.copy()

    # Calculate duration
    df["duration"] = (
        df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    # Filtering
    df = df[
        (df["duration"] >= config.min_trip_duration)
        & (df["duration"] <= config.max_trip_duration)
    ].copy()

    # String conversions
    df["PULocationID"] = df["PULocationID"].astype(str)
    df["DOLocationID"] = df["DOLocationID"].astype(str)
    df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]

    return df


def vectorize_train(
    df: pd.DataFrame, dv: DictVectorizer
) -> tuple[sparse.csr_matrix, DictVectorizer]:
    """Fits and transforms the training data."""
    dicts = df[config.categorical_features + config.numerical_features].to_dict(
        orient="records"
    )
    X = dv.fit_transform(dicts)
    return X, dv


def vectorize(df: pd.DataFrame, dv: DictVectorizer) -> sparse.csr_matrix:
    """Transforms validation/test data using a pre-fitted vectorizer."""
    dicts = df[config.categorical_features + config.numerical_features].to_dict(
        orient="records"
    )
    X = dv.transform(dicts)
    return X
