import pickle
import time
import numpy as np

from prodml import data, features
from prodml.config import config
from prodml.predict import DurationPredictor


def test_onnx_sklearn_parity_and_benchmark():

    # 1. Load 500 validation rows
    df = data.read_data(config.data_path)
    df = features.prepare_features(df)
    _, df_val = data.split_data(df, config.train_test_split)

    # convert to list of dicts
    sample_rides = (
        df_val[config.categorical_features + config.numerical_features]
        .head(500)
        .to_dict(orient="records")
    )

    # 2. Setup Pickle / Scikit-Learn
    with open(config.output_model_path, "rb") as f:
        dv, sklearn_model = pickle.load(f)

    # Benchmark Scikit-Learn (Pickle)
    X_sklearn = dv.transform(sample_rides)

    start_time = time.perf_counter()
    sklearn_preds = sklearn_model.predict(X_sklearn)
    pickle_duration = time.perf_counter() - start_time

    # 3. Setup ONNX (using our Predictor class)
    predictor = DurationPredictor()
    predictor.load()

    # Benchmark ONNX
    start_time = time.perf_counter()
    onnx_preds = predictor.predict_batch(sample_rides)
    onnx_duration = time.perf_counter() - start_time

    # 4. Assert Parity to 1e-4
    assert np.allclose(sklearn_preds, onnx_preds, atol=1e-4)

    # 5. Print Benchmarks for report
    print("\n--- BENCHMARK RESULTS (500 rows) ---")
    print(f"Pickle Latency : {pickle_duration:.4f} seconds")
    print(f"ONNX Latency   : {onnx_duration:.4f} seconds")

    # Since we did batch prediction, average per row:
    print(f"Pickle Mean/row: {(pickle_duration/500)*1000:.4f} ms")
    print(f"ONNX Mean/row  : {(onnx_duration/500)*1000:.4f} ms")
