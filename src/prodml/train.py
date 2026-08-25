import pickle

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from prodml import data, features
from prodml.config import config
from prodml.logger import get_logger

logger = get_logger(__name__)


def train_model(df_train: pd.DataFrame, df_val: pd.DataFrame) -> tuple:
    dv = DictVectorizer()

    X_train, dv = features.vectorize_train(df_train, dv)
    y_train = df_train["duration"].values

    X_val = features.vectorize(df_val, dv)
    y_val = df_val["duration"].values

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = root_mean_squared_error(y_val, y_pred)

    return model, dv, mae, rmse


def save_model(model, dv):
    config.models_dir.mkdir(parents=True, exist_ok=True)
    with open(config.output_model_path, "wb") as f_out:
        pickle.dump((dv, model), f_out)


def main():
    # 1. Load Data
    df = data.read_data(config.data_path)

    # 2. Extract Features
    df = features.prepare_features(df)

    # 3. Split Data
    df_train, df_val = data.split_data(df, config.train_test_split)

    # 4. Train Model
    model, dv, mae, rmse = train_model(df_train, df_val)

    logger.info(
        "Model training completed", extra={"mae": round(mae, 4), "rmse": round(rmse, 4)}
    )

    # 5. Save Artifact
    save_model(model, dv)
    logger.info(
        "Model artifact saved", extra={"model_path": str(config.output_model_path)}
    )


if __name__ == "__main__":
    main()
